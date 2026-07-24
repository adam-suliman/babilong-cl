from __future__ import annotations

import gc
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpointing import load_checkpoint, save_checkpoint
from .config import CANONICAL_TASKS, ExperimentConfig
from .data import FixedZeroContextProvider
from .metrics import continual_metrics
from .monitoring import ExperimentMonitor
from .reporting import write_run_artifacts
from .trainer import EpochBatchCursor, UnifiedTrainer, _task_scheduler
from .util import (
    atomic_write_json,
    file_lock,
    git_manifest,
    json_sha256,
)


def prepare_data(config: ExperimentConfig, *, babi_archive: str | Path) -> dict[str, Any]:
    provider = FixedZeroContextProvider(
        data_dir=config.resolved_data_dir,
        babi_archive=babi_archive,
        tokenizer_name=config.tokenizer,
        tokenizer_revision=config.backbone_revision,
        data_seed=config.data_seed,
    )
    return provider.prepare()


def _load_tokenizer(config: ExperimentConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer,
        revision=config.backbone_revision,
    )
    tokenizer.model_max_length = 10**9
    return tokenizer


def _reference_identity(
    config: ExperimentConfig,
    *,
    task: str,
    data_manifest_sha256: str,
) -> tuple[Path, str]:
    payload = dict(config.protocol_payload)
    payload.pop("order_seed", None)
    payload.pop("task_order", None)
    payload["task"] = task
    payload["data_manifest_sha256"] = data_manifest_sha256
    if config.model == "gpt2":
        payload["cl_method"] = "none"
        payload["si_lambda"] = None
        payload["method"] = "none"
    identity = json_sha256(payload)
    architecture, method = config.architecture_and_method
    if config.model == "gpt2":
        method = "none"
    path = (
        Path(config.results_root)
        / "references"
        / architecture
        / method
        / f"replicate-{config.replicate_seed}"
        / identity[:12]
        / task
    )
    return path, identity


def ensure_references(
    config: ExperimentConfig,
    data_manifest: Mapping[str, Any],
    *,
    tensorboard: bool,
    max_steps_per_task: int | None = None,
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for task in CANONICAL_TASKS:
        path, identity = _reference_identity(
            config,
            task=task,
            data_manifest_sha256=str(data_manifest["manifest_sha256"]),
        )
        raw_path = path / "raw.json"
        with file_lock(path / ".lock"):
            if raw_path.exists():
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                if raw.get("identity") != identity or raw.get("status") != "complete":
                    raise RuntimeError(f"Incompatible reference cache: {raw_path}")
            else:
                reference_config = config
                if config.model == "gpt2" and config.cl_method == "si":
                    reference_config = replace(
                        config,
                        cl_method="none",
                        si_lambda=None,
                    )
                tokenizer = _load_tokenizer(reference_config)
                trainer = UnifiedTrainer(
                    config=reference_config,
                    data_manifest=data_manifest,
                    tokenizer=tokenizer,
                    tensorboard=False,
                )
                dataset = trainer._dataset("train", task)
                target_steps = (
                    config.slow_steps_per_task
                    if max_steps_per_task is None
                    else min(config.slow_steps_per_task, max_steps_per_task)
                )
                latest = path / "checkpoints" / "latest.pt"
                start_step = 0
                restored_sampler = None
                restored_scheduler = None
                restored_train_state = None
                if latest.exists():
                    restored = load_checkpoint(
                        latest,
                        config=reference_config,
                        data_manifest_sha256=str(data_manifest["manifest_sha256"]),
                        model=trainer.bundle.model,
                        optimizer=trainer.optimizer,
                        strategy=trainer.strategy,
                        initial_model_sha256=trainer.initial_model_sha256,
                        map_location=trainer.device,
                    )
                    progress = restored["progress"]
                    if (
                        progress.get("reference_identity") != identity
                        or progress.get("task") != task
                    ):
                        raise RuntimeError(
                            f"Incompatible reference checkpoint: {latest}"
                        )
                    start_step = int(progress["task_slow_step"])
                    restored_sampler = restored["sampler_state"]
                    restored_scheduler = restored["scheduler_state"]
                    restored_train_state = progress.get("active_train_state")
                    if trainer.bundle.fast_cell is not None:
                        counters = progress.get("fast_counters", {})
                        trainer.bundle.fast_cell.fast_update_attempts = int(
                            counters.get("attempts", 0)
                        )
                        trainer.bundle.fast_cell.fast_updates_applied = int(
                            counters.get("applied", 0)
                        )
                trainer.scheduler = _task_scheduler(
                    trainer.optimizer,
                    learning_rate=float(config.learning_rates[task]),
                    warmup_steps=min(config.warmup_steps, target_steps),
                    total_steps=target_steps,
                )
                if restored_scheduler is not None:
                    trainer.scheduler.load_state_dict(restored_scheduler)
                    for group, learning_rate in zip(
                        trainer.optimizer.param_groups,
                        restored_scheduler.get("_last_lr", []),
                    ):
                        group["lr"] = float(learning_rate)
                if trainer.strategy is not None:
                    if trainer.strategy.active_task is None:
                        trainer.strategy.begin_task(task)
                trainer.bundle.model.train()
                if start_step == 0:
                    trainer.bundle.reset_fast_memory()
                cursor = (
                    EpochBatchCursor.from_state_dict(restored_sampler)
                    if restored_sampler is not None
                    else EpochBatchCursor(
                        dataset_size=len(dataset),
                        batch_size=config.slow_batch_size,
                        task_seed=config.sampler_seeds[task],
                    )
                )
                path.mkdir(parents=True, exist_ok=True)
                atomic_write_json(
                    path / "status.json",
                    {
                        "format": "babilong-qa6-reference-status-v1",
                        "status": "running",
                        "identity": identity,
                        "task": task,
                        "resume_start_slow_step": start_step,
                    },
                )
                trainer._install_signal_handlers()
                try:
                    def checkpoint_reference(
                        task_step: int,
                        sampler: EpochBatchCursor,
                        train_state: Mapping[str, Any],
                    ) -> None:
                        save_checkpoint(
                            latest,
                            config=reference_config,
                            data_manifest_sha256=str(
                                data_manifest["manifest_sha256"]
                            ),
                            model=trainer.bundle.model,
                            optimizer=trainer.optimizer,
                            scheduler=trainer.scheduler,
                            strategy=trainer.strategy,
                            progress={
                                "reference_identity": identity,
                                "task": task,
                                "task_slow_step": task_step,
                                "active_train_state": dict(train_state),
                                "fast_counters": (
                                    {
                                        "attempts": trainer.bundle.fast_cell.fast_update_attempts,
                                        "applied": trainer.bundle.fast_cell.fast_updates_applied,
                                    }
                                    if trainer.bundle.fast_cell is not None
                                    else {"attempts": 0, "applied": 0}
                                ),
                            },
                            sampler_state=sampler.state_dict(),
                            initial_model_sha256=trainer.initial_model_sha256,
                        )

                    train = trainer.train_task(
                        task=task,
                        dataset=dataset,
                        cursor=cursor,
                        start_step=start_step,
                        target_steps=target_steps,
                        resume_train_state=restored_train_state,
                        checkpoint_callback=checkpoint_reference,
                    )
                    if trainer.strategy is not None:
                        train["si_end_task"] = trainer.strategy.end_task(task)
                    scores, predictions = trainer.evaluate_task(
                        split="eval", task=task
                    )
                    raw = {
                        "format": "babilong-qa6-single-reference-v1",
                        "identity": identity,
                        "status": "complete",
                        "task": task,
                        "config": reference_config.to_dict(),
                        "data_manifest_sha256": data_manifest["manifest_sha256"],
                        "initial_model_sha256": trainer.initial_model_sha256,
                        "train": train,
                        "scores": scores,
                        "predictions": predictions,
                    }
                    atomic_write_json(raw_path, raw)
                    verified = json.loads(raw_path.read_text(encoding="utf-8"))
                    if (
                        verified.get("identity") != identity
                        or verified.get("status") != "complete"
                    ):
                        raise RuntimeError(
                            f"Reference result verification failed: {raw_path}"
                        )
                    atomic_write_json(
                        path / "status.json",
                        {
                            "format": "babilong-qa6-reference-status-v1",
                            "status": "complete",
                            "identity": identity,
                            "task": task,
                        },
                    )
                    for checkpoint_file in (
                        latest,
                        latest.with_suffix(".json"),
                    ):
                        checkpoint_file.unlink(missing_ok=True)
                    checkpoint_dir = latest.parent
                    if checkpoint_dir.exists() and not any(checkpoint_dir.iterdir()):
                        checkpoint_dir.rmdir()
                except InterruptedError:
                    atomic_write_json(
                        path / "status.json",
                        {
                            "format": "babilong-qa6-reference-status-v1",
                            "status": "interrupted",
                            "identity": identity,
                            "task": task,
                        },
                    )
                    raise
                finally:
                    trainer._restore_signal_handlers()
                    trainer.monitor.close()
                    del trainer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        references[task] = {
            "compare_answers": float(raw["scores"]["compare_answers"]),
            "exact_match": float(raw["scores"]["exact_match"]),
            "path": str(raw_path),
            "identity": identity,
        }
    return references


def run_experiment(
    config: ExperimentConfig,
    *,
    babi_archive: str | Path,
    tensorboard: bool = True,
    resume: bool = True,
    max_tasks: int | None = None,
    max_steps_per_task: int | None = None,
    evaluation_split: str = "eval",
) -> dict[str, Any]:
    if (
        max_steps_per_task is not None
        and int(max_steps_per_task) != config.slow_steps_per_task
    ):
        raise ValueError(
            "A runtime max-step override would create a non-resumable stage boundary. "
            "Use --steps-per-task so the reduced budget is part of the protocol hash."
        )
    disk_root = Path(config.results_root).resolve().parent
    disk_root.mkdir(parents=True, exist_ok=True)
    free_disk_gb = shutil.disk_usage(disk_root).free / (1024**3)
    if free_disk_gb < config.minimum_free_disk_gb:
        raise RuntimeError(
            f"Disk preflight failed: {free_disk_gb:.1f} GB free, "
            f"{config.minimum_free_disk_gb:.1f} GB required"
        )
    repo = Path(__file__).resolve().parents[1]
    git = git_manifest(repo)
    if config.require_clean_git and git.get("available") and not git.get("clean"):
        raise RuntimeError(
            "Canonical runs require a clean Git worktree. Commit the implementation "
            "or use --allow-dirty only for smoke tests."
        )
    data_manifest = prepare_data(config, babi_archive=babi_archive)
    run_dir = config.run_dir
    with file_lock(run_dir / ".run.lock"):
        tokenizer = _load_tokenizer(config)
        trainer = UnifiedTrainer(
            config=config,
            data_manifest=data_manifest,
            tokenizer=tokenizer,
            tensorboard=tensorboard,
        )
        raw = trainer.run(
            evaluation_split=evaluation_split,
            resume=resume,
            max_tasks=max_tasks,
            max_steps_per_task=max_steps_per_task,
        )
        if (
            raw["status"] == "complete"
            and config.include_references
            and evaluation_split == "eval"
        ):
            references = ensure_references(
                config,
                data_manifest,
                tensorboard=tensorboard,
                max_steps_per_task=max_steps_per_task,
            )
            raw["references"] = references
            reference_scores = {
                task: references[task]["compare_answers"] for task in CANONICAL_TASKS
            }
            raw["metrics"]["compare_answers"] = continual_metrics(
                raw["compare_answers_matrix"],
                raw["task_order"],
                single_task_references=reference_scores,
            )
            atomic_write_json(run_dir / "raw.json", raw)
            monitor = ExperimentMonitor(config, enabled=tensorboard)
            monitor.references(
                task_order=config.resolved_order,
                metrics=raw["metrics"]["compare_answers"],
                steps_per_task=config.slow_steps_per_task,
            )
            monitor.flush()
            monitor.close()
        write_run_artifacts(raw, run_dir)
        return raw


def calibrate_si(
    base_config: ExperimentConfig,
    *,
    babi_archive: str | Path,
    lambdas: tuple[float, ...] = (0.1, 1.0, 10.0),
    tensorboard: bool = True,
    max_tasks: int | None = None,
    max_steps_per_task: int | None = None,
) -> dict[str, Any]:
    root = Path(base_config.results_root)
    calibration_root = root / "calibration" / "si"
    results = []
    for value in lambdas:
        config = replace(
            base_config,
            model="gpt2",
            cl_method="si",
            si_lambda=float(value),
            replicate_seed=48,
            order_seed=48,
            results_root=str(calibration_root),
            data_dir=str(base_config.resolved_data_dir),
            include_references=False,
        )
        raw = run_experiment(
            config,
            babi_archive=babi_archive,
            tensorboard=tensorboard,
            max_tasks=max_tasks,
            max_steps_per_task=max_steps_per_task,
            evaluation_split="validation",
        )
        score = raw["metrics"]["compare_answers"].get("final_all_task_accuracy")
        results.append(
            {
                "si_lambda": float(value),
                "final_mean_private_validation_compare_answers": score,
                "run_dir": str(config.run_dir),
                "protocol_hash": config.protocol_hash,
            }
        )
    complete = [
        item
        for item in results
        if item["final_mean_private_validation_compare_answers"] is not None
    ]
    if not complete:
        raise RuntimeError("SI calibration did not complete any full six-task run")
    selected = sorted(
        complete,
        key=lambda item: (
            -float(item["final_mean_private_validation_compare_answers"]),
            float(item["si_lambda"]),
        ),
    )[0]
    payload = {
        "format": "babilong-qa6-si-selection-v1",
        "criterion": "final mean private-validation compare_answers",
        "public_test_used": False,
        "grid": list(lambdas),
        "replicate_seed": 48,
        "order_seed": 48,
        "data_manifest_sha256": json.loads(
            (base_config.resolved_data_dir / "manifest.json").read_text()
        )["manifest_sha256"],
        "selection_context": {
            key: base_config.to_dict()[key]
            for key in (
                "backbone",
                "backbone_revision",
                "precision",
                "slow_steps_per_task",
                "warmup_steps",
                "learning_rates",
                "weight_decay",
                "clip_grad_norm",
                "label_mask_policy",
                "loss_normalization",
                "microbatch_size",
                "slow_batch_size",
                "data_seed",
                "si_epsilon",
                "si_decay",
                "si_clamp_importance",
            )
        },
        "results": results,
        "selected_si_lambda": selected["si_lambda"],
    }
    atomic_write_json(calibration_root / "selection.json", payload)
    return payload
