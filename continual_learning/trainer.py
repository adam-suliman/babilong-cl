from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .checkpointing import (
    finalize_checkpoint,
    load_checkpoint,
    model_state_sha256,
    save_checkpoint,
)
from .config import CANONICAL_TASKS, ExperimentConfig, stable_seed
from .data import OfficialCollator, TokenizedBabilongDataset
from .metrics import (
    continual_metrics,
    empty_matrix,
    score_prediction,
)
from .models import ModelBundle, autocast_context, build_model, fast_lr_for_config
from .monitoring import ExperimentMonitor
from .strategies import SynapticIntelligence, SynapticIntelligenceConfig
from .util import (
    atomic_write_json,
    environment_manifest,
    file_sha256,
    git_manifest,
    seed_everything,
)


RAW_FORMAT = "babilong-qa6-cl-raw-v1"
STATUS_FORMAT = "babilong-qa6-cl-status-v1"


@dataclass
class EpochBatchCursor:
    dataset_size: int
    batch_size: int
    task_seed: int
    epoch: int = 0
    batch_cursor: int = 0

    def __post_init__(self) -> None:
        if self.dataset_size < self.batch_size:
            raise ValueError("Dataset is smaller than one complete batch")
        self.effective_rows = (self.dataset_size // self.batch_size) * self.batch_size
        self.batches_per_epoch = self.effective_rows // self.batch_size
        if not 0 <= self.batch_cursor <= self.batches_per_epoch:
            raise ValueError("Invalid batch cursor")

    def _permutation(self) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(
            stable_seed(
                "babilong-cl-epoch-permutation-v1",
                self.task_seed,
                str(self.epoch),
            )
        )
        permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
        return permutation[: self.effective_rows]

    def next_batch(self) -> tuple[list[int], bool]:
        epoch_started = False
        if self.batch_cursor == self.batches_per_epoch:
            self.epoch += 1
            self.batch_cursor = 0
            epoch_started = True
        permutation = self._permutation()
        start = self.batch_cursor * self.batch_size
        end = start + self.batch_size
        batch = permutation[start:end]
        self.batch_cursor += 1
        return batch, epoch_started

    def state_dict(self) -> dict[str, int]:
        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "task_seed": self.task_seed,
            "epoch": self.epoch,
            "batch_cursor": self.batch_cursor,
            "effective_rows": self.effective_rows,
            "batches_per_epoch": self.batches_per_epoch,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EpochBatchCursor":
        cursor = cls(
            dataset_size=int(state["dataset_size"]),
            batch_size=int(state["batch_size"]),
            task_seed=int(state["task_seed"]),
            epoch=int(state["epoch"]),
            batch_cursor=int(state["batch_cursor"]),
        )
        if cursor.effective_rows != int(state["effective_rows"]):
            raise ValueError("Sampler effective-row count changed")
        if cursor.batches_per_epoch != int(state["batches_per_epoch"]):
            raise ValueError("Sampler batches-per-epoch changed")
        return cursor


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _task_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
):
    from transformers import get_linear_schedule_with_warmup

    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
        group["initial_lr"] = float(learning_rate)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


class UnifiedTrainer:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        data_manifest: Mapping[str, Any],
        tokenizer: Any,
        model_bundle: ModelBundle | None = None,
        tiny_model_config: Any | None = None,
        tensorboard: bool = True,
    ) -> None:
        self.config = config
        self.data_manifest = dict(data_manifest)
        self.data_manifest_sha256 = str(data_manifest["manifest_sha256"])
        self.tokenizer = tokenizer
        self.collator = OfficialCollator(tokenizer)
        self.device = self._resolve_device()
        seed_everything(config.replicate_seed, deterministic=config.deterministic)
        self.bundle = model_bundle or build_model(config, tiny_config=tiny_model_config)
        self.bundle.model.to(self.device)
        self.initial_model_sha256 = model_state_sha256(self.bundle.model)
        self.optimizer = torch.optim.AdamW(
            self.bundle.slow_parameters(),
            lr=float(config.learning_rates[config.resolved_order[0]]),
            weight_decay=config.weight_decay,
        )
        self.strategy = self._build_strategy()
        self.scheduler = None
        self.monitor = ExperimentMonitor(config, enabled=tensorboard)
        self.stop_requested = False
        self._old_signal_handlers: dict[int, Any] = {}

    def _resolve_device(self) -> torch.device:
        requested = self.config.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(requested)

    def _build_strategy(self) -> SynapticIntelligence | None:
        if self.config.cl_method == "none":
            return None
        assert self.config.si_lambda is not None
        return SynapticIntelligence(
            self.bundle.model,
            self.optimizer,
            SynapticIntelligenceConfig(
                si_lambda=self.config.si_lambda,
                epsilon=self.config.si_epsilon,
                decay=self.config.si_decay,
                clamp_importance=self.config.si_clamp_importance,
            ),
        )

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            self.stop_requested = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._old_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._old_signal_handlers.items():
            signal.signal(signum, handler)
        self._old_signal_handlers.clear()

    def _dataset(self, split: str, task: str) -> TokenizedBabilongDataset:
        info = self.data_manifest[split][task]
        path = self.config.resolved_data_dir / info["path"]
        return TokenizedBabilongDataset(
            path,
            self.tokenizer,
            require_all_fit=split != "train" or task != "qa3",
        )

    def _initial_raw(self, evaluation_split: str) -> dict[str, Any]:
        return {
            "format": RAW_FORMAT,
            "config": self.config.to_dict(),
            "data_manifest_sha256": self.data_manifest_sha256,
            "initial_model_sha256": self.initial_model_sha256,
            "task_order": list(self.config.resolved_order),
            "columns": list(CANONICAL_TASKS),
            "rows": ["pretrain"]
            + [f"after_{index:02d}_{task}" for index, task in enumerate(self.config.resolved_order, 1)],
            "evaluation_split": evaluation_split,
            "compare_answers_matrix": empty_matrix(),
            "exact_match_matrix": empty_matrix(),
            "train_tasks": [],
            "predictions": {},
            "cumulative_slow_steps": 0,
            "fast_update_attempts": 0,
            "fast_updates_applied": 0,
            "references": None,
            "metrics": {},
            "environment": environment_manifest(),
            "git": git_manifest(Path(__file__).resolve().parents[1]),
            "status": "running",
            "started_at_unix": time.time(),
        }

    def run(
        self,
        *,
        evaluation_split: str = "eval",
        resume: bool = True,
        max_tasks: int | None = None,
        max_steps_per_task: int | None = None,
    ) -> dict[str, Any]:
        run_dir = self.config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = run_dir / "config.json"
        if resolved_config_path.exists():
            existing_config = json.loads(
                resolved_config_path.read_text(encoding="utf-8")
            )
            if existing_config.get("protocol_hash") != self.config.protocol_hash:
                raise RuntimeError(
                    f"Run directory contains a different protocol: {run_dir}"
                )
        else:
            atomic_write_json(resolved_config_path, self.config.to_dict())
        latest = run_dir / "checkpoints" / "latest.pt"
        raw_path = run_dir / "raw.json"
        status_path = run_dir / "status.json"
        self._write_status(status_path, "running")
        self._install_signal_handlers()
        try:
            raw = self._initial_raw(evaluation_split)
            stage_start = 0
            active_task_step = 0
            restored_sampler_state = None
            restored_scheduler_state = None
            restored_train_state = None
            if resume and latest.exists():
                restored = load_checkpoint(
                    latest,
                    config=self.config,
                    data_manifest_sha256=self.data_manifest_sha256,
                    model=self.bundle.model,
                    optimizer=self.optimizer,
                    strategy=self.strategy,
                    initial_model_sha256=self.initial_model_sha256,
                    map_location=self.device,
                )
                progress = restored["progress"]
                raw = dict(progress["raw"])
                stage_start = int(progress["stage_index"])
                active_task_step = int(progress["task_slow_step"])
                restored_sampler_state = restored["sampler_state"]
                restored_scheduler_state = (
                    restored["scheduler_state"] if active_task_step > 0 else None
                )
                restored_train_state = (
                    progress.get("active_train_state")
                    if active_task_step > 0
                    else None
                )
                if self.bundle.fast_cell is not None:
                    fast_state = progress.get("fast_counters", {})
                    self.bundle.fast_cell.fast_update_attempts = int(
                        fast_state.get("attempts", 0)
                    )
                    self.bundle.fast_cell.fast_updates_applied = int(
                        fast_state.get("applied", 0)
                    )
                self.monitor.close()
                self.monitor = ExperimentMonitor(
                    self.config,
                    enabled=self.monitor.enabled,
                    purge_step=(
                        int(raw["cumulative_slow_steps"])
                        + active_task_step
                        + 1
                    ),
                )
            elif raw["compare_answers_matrix"][0][0] is None:
                scores, predictions = self.evaluate_all(
                    split=evaluation_split,
                    stage_index=0,
                )
                self._store_eval(raw, 0, scores, predictions)
                atomic_write_json(raw_path, raw)

            order = self.config.resolved_order
            stop_stage = len(order) if max_tasks is None else min(len(order), max_tasks)
            for stage_index in range(stage_start, stop_stage):
                task = order[stage_index]
                task_limit = (
                    self.config.slow_steps_per_task
                    if max_steps_per_task is None
                    else min(self.config.slow_steps_per_task, max_steps_per_task)
                )
                dataset = self._dataset("train", task)
                print(
                    json.dumps(
                        {
                            "event": "task_start",
                            "model": self.config.model,
                            "cl_method": self.config.cl_method,
                            "stage_index": stage_index + 1,
                            "task": task,
                            "resume_step": active_task_step,
                            "target_steps": task_limit,
                            "cumulative_completed_steps": raw[
                                "cumulative_slow_steps"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if restored_sampler_state is not None and stage_index == stage_start:
                    cursor = EpochBatchCursor.from_state_dict(restored_sampler_state)
                else:
                    cursor = EpochBatchCursor(
                        dataset_size=len(dataset),
                        batch_size=self.config.slow_batch_size,
                        task_seed=self.config.sampler_seeds[task],
                    )
                self.scheduler = _task_scheduler(
                    self.optimizer,
                    learning_rate=float(self.config.learning_rates[task]),
                    warmup_steps=min(self.config.warmup_steps, task_limit),
                    total_steps=task_limit,
                )
                if restored_scheduler_state is not None and stage_index == stage_start:
                    self.scheduler.load_state_dict(restored_scheduler_state)
                    last_lrs = restored_scheduler_state.get("_last_lr")
                    if last_lrs is not None:
                        for group, learning_rate in zip(
                            self.optimizer.param_groups, last_lrs
                        ):
                            group["lr"] = float(learning_rate)
                if self.strategy is not None and self.strategy.active_task is None:
                    self.strategy.begin_task(task)
                self.bundle.model.train()
                if active_task_step == 0:
                    self.bundle.reset_fast_memory()
                train_summary = self.train_task(
                    task=task,
                    dataset=dataset,
                    cursor=cursor,
                    start_step=active_task_step,
                    target_steps=task_limit,
                    prior_cumulative_steps=int(raw["cumulative_slow_steps"]),
                    resume_train_state=restored_train_state,
                    checkpoint_callback=lambda task_step, sampler, train_state: self._save_progress(
                        latest=latest,
                        raw=raw,
                        stage_index=stage_index,
                        task_slow_step=task_step,
                        sampler=sampler,
                        active_train_state=train_state,
                    ),
                )
                if self.strategy is not None:
                    train_summary["si_end_task"] = self.strategy.end_task(task)
                raw["train_tasks"].append(train_summary)
                raw["cumulative_slow_steps"] += int(task_limit)
                raw["fast_update_attempts"] += int(
                    train_summary["fast_update_attempts"]
                )
                raw["fast_updates_applied"] += int(
                    train_summary["fast_updates_applied"]
                )
                scores, predictions = self.evaluate_all(
                    split=evaluation_split,
                    stage_index=stage_index + 1,
                )
                self._store_eval(raw, stage_index + 1, scores, predictions)
                raw["metrics"] = {
                    "compare_answers": continual_metrics(
                        raw["compare_answers_matrix"], order
                    ),
                    "exact_match": continual_metrics(raw["exact_match_matrix"], order),
                }
                atomic_write_json(raw_path, raw)
                self._save_progress(
                    latest=latest,
                    raw=raw,
                    stage_index=stage_index + 1,
                    task_slow_step=0,
                    sampler=None,
                    active_train_state=None,
                )
                active_task_step = 0
                restored_sampler_state = None
                restored_scheduler_state = None
                restored_train_state = None

            completed = stop_stage == len(order)
            raw["status"] = "complete" if completed else "partial"
            raw["finished_at_unix"] = time.time()
            raw["metrics"] = {
                "compare_answers": continual_metrics(
                    raw["compare_answers_matrix"], order
                ),
                "exact_match": continual_metrics(raw["exact_match_matrix"], order),
            }
            atomic_write_json(raw_path, raw)
            if completed:
                finalize_checkpoint(latest, run_dir / "checkpoints" / "final.pt")
            self.monitor.matrix_text(
                compare_matrix=raw["compare_answers_matrix"],
                exact_matrix=raw["exact_match_matrix"],
                step=int(raw["cumulative_slow_steps"]),
            )
            self.monitor.flush()
            self._write_status(status_path, raw["status"])
            return raw
        except InterruptedError:
            self._write_status(status_path, "interrupted")
            raise
        except Exception as error:
            self._write_status(status_path, "failed", error=repr(error))
            raise
        finally:
            self._restore_signal_handlers()
            self.monitor.close()

    def train_task(
        self,
        *,
        task: str,
        dataset: Dataset[dict[str, Any]],
        cursor: EpochBatchCursor,
        start_step: int,
        target_steps: int,
        prior_cumulative_steps: int = 0,
        resume_train_state: Mapping[str, Any] | None = None,
        checkpoint_callback: Callable[
            [int, EpochBatchCursor, Mapping[str, Any]], None
        ]
        | None = None,
    ) -> dict[str, Any]:
        if self.scheduler is None:
            raise RuntimeError("Task scheduler has not been created")
        fast_attempts_before = (
            self.bundle.fast_cell.fast_update_attempts
            if self.bundle.fast_cell is not None
            else 0
        )
        fast_applied_before = (
            self.bundle.fast_cell.fast_updates_applied
            if self.bundle.fast_cell is not None
            else 0
        )
        if start_step > 0 and resume_train_state is None:
            raise RuntimeError("Active task checkpoint is missing training accumulators")
        state = dict(resume_train_state or {})
        loss_sum = float(state.get("loss_sum", 0.0))
        loss_count = int(state.get("loss_count", 0))
        final_loss = state.get("final_loss")
        prior_seconds = float(state.get("seconds", 0.0))
        examples_seen = int(
            state.get("examples_seen", start_step * self.config.slow_batch_size)
        )
        if loss_count != start_step:
            raise RuntimeError(
                f"Training accumulator step mismatch: {loss_count} != {start_step}"
            )
        if examples_seen != start_step * self.config.slow_batch_size:
            raise RuntimeError("Training accumulator example count mismatch")
        microbatches_per_slow = self.config.slow_batch_size // self.config.microbatch_size
        expected_fast_per_step = self.config.slow_batch_size // self.config.fast_batch_size
        self.optimizer.zero_grad(set_to_none=True)
        started = time.time()

        for task_step in range(start_step, target_steps):
            indices, epoch_started = cursor.next_batch()
            if epoch_started:
                self.bundle.reset_fast_memory()
            fast_attempts_at_step = (
                self.bundle.fast_cell.fast_update_attempts
                if self.bundle.fast_cell is not None
                else 0
            )
            step_loss = 0.0
            fast_examples = 0
            fast_diagnostics: dict[str, float | int] | None = None

            for micro_start in range(0, len(indices), self.config.microbatch_size):
                micro_indices = indices[
                    micro_start : micro_start + self.config.microbatch_size
                ]
                batch = self.collator([dataset[index] for index in micro_indices])
                batch = _to_device(batch, self.device)
                with autocast_context(self.config, self.device):
                    output = self.bundle.model(
                        input_ids=batch["input_ids"],
                        labels=batch["labels"],
                        labels_mask=batch["labels_mask"],
                        attention_mask=batch["attention_mask"],
                    )
                    task_loss = output.loss
                    penalty = (
                        self.strategy.penalty()
                        if self.strategy is not None
                        else task_loss.new_zeros(())
                    )
                    scaled_loss = (task_loss + penalty) / microbatches_per_slow
                scaled_loss.backward()
                step_loss += float(task_loss.detach()) / microbatches_per_slow

                if self.bundle.fast_cell is not None:
                    fast_examples += len(micro_indices)
                    if fast_examples == self.config.fast_batch_size:
                        fast_diagnostics = self.bundle.fast_cell.apply_fast_update(
                            fast_lr=fast_lr_for_config(self.config),
                            grad_scale=(
                                self.config.slow_batch_size
                                / self.config.fast_batch_size
                            ),
                            clip_norm=self.config.fast_clip_norm,
                        )
                        fast_examples = 0
                    elif fast_examples > self.config.fast_batch_size:
                        raise RuntimeError("Microbatch crossed a FastMem update boundary")

            if self.bundle.fast_cell is not None:
                actual = (
                    self.bundle.fast_cell.fast_update_attempts - fast_attempts_at_step
                )
                if actual != expected_fast_per_step or fast_examples != 0:
                    raise RuntimeError(
                        f"Expected {expected_fast_per_step} FastMem attempts, got {actual}"
                    )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.bundle.slow_parameters(),
                max_norm=self.config.clip_grad_norm,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Slow gradient norm is non-finite")
            if self.strategy is not None:
                self.strategy.before_optimizer_step()
            self.optimizer.step()
            if self.strategy is not None:
                self.strategy.after_optimizer_step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            loss_sum += step_loss
            loss_count += 1
            final_loss = step_loss
            examples_seen += self.config.slow_batch_size
            cumulative_step = prior_cumulative_steps + task_step + 1
            if (task_step + 1) % self.config.log_interval == 0 or task_step == start_step:
                self.monitor.training(
                    task=task,
                    loss=step_loss,
                    lr=float(self.optimizer.param_groups[0]["lr"]),
                    grad_norm=float(grad_norm),
                    examples_seen=examples_seen,
                    cumulative_step=cumulative_step,
                    fast=(
                        self.bundle.fast_cell.diagnostics()
                        if self.bundle.fast_cell is not None
                        else fast_diagnostics
                    ),
                    si=(
                        self.strategy.diagnostics()
                        if self.strategy is not None
                        else None
                    ),
                )
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "model": self.config.model,
                            "cl_method": self.config.cl_method,
                            "task": task,
                            "task_step": task_step + 1,
                            "target_steps": target_steps,
                            "cumulative_step": cumulative_step,
                            "loss": step_loss,
                            "learning_rate": float(
                                self.optimizer.param_groups[0]["lr"]
                            ),
                            "slow_gradient_norm": float(grad_norm),
                            "examples_seen": examples_seen,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if checkpoint_callback is not None and (
                (task_step + 1) % self.config.checkpoint_interval == 0
                or self.stop_requested
            ):
                checkpoint_callback(
                    task_step + 1,
                    cursor,
                    {
                        "loss_sum": loss_sum,
                        "loss_count": loss_count,
                        "final_loss": final_loss,
                        "examples_seen": examples_seen,
                        "seconds": prior_seconds + time.time() - started,
                    },
                )
            if self.stop_requested:
                raise InterruptedError("Stop requested; checkpoint saved at slow-step boundary")

        attempts = (
            self.bundle.fast_cell.fast_update_attempts - fast_attempts_before
            if self.bundle.fast_cell is not None
            else 0
        )
        applied = (
            self.bundle.fast_cell.fast_updates_applied - fast_applied_before
            if self.bundle.fast_cell is not None
            else 0
        )
        if self.bundle.fast_cell is not None:
            expected = (target_steps - start_step) * expected_fast_per_step
            if attempts != expected:
                raise RuntimeError(f"Fast update attempts mismatch: {attempts} != {expected}")
            full_attempts = target_steps * expected_fast_per_step
            full_applied = (
                full_attempts if fast_lr_for_config(self.config) != 0.0 else 0
            )
        else:
            full_attempts = 0
            full_applied = 0
        return {
            "task": task,
            "task_sampler_seed": self.config.sampler_seeds[task],
            "learning_rate": float(self.config.learning_rates[task]),
            "configured_legacy_iters": self.config.legacy_configured_iters,
            "slow_steps": target_steps,
            "executed_slow_steps_this_process": target_steps - start_step,
            "resume_start_slow_step": start_step,
            "target_slow_steps": target_steps,
            "examples_seen": target_steps * self.config.slow_batch_size,
            "dataset_rows": len(dataset),
            "effective_rows_per_epoch": cursor.effective_rows,
            "dropped_tail_rows_per_epoch": len(dataset) - cursor.effective_rows,
            "epochs_entered": cursor.epoch + 1,
            "mean_loss": loss_sum / loss_count if loss_count else None,
            "final_loss": final_loss,
            "fast_update_attempts": full_attempts,
            "fast_updates_applied": full_applied,
            "executed_fast_update_attempts_this_process": attempts,
            "executed_fast_updates_applied_this_process": applied,
            "seconds": prior_seconds + time.time() - started,
        }

    @torch.no_grad()
    def evaluate_all(
        self,
        *,
        split: str,
        stage_index: int,
    ) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]:
        was_training = self.bundle.model.training
        self.bundle.model.eval()
        scores: dict[str, dict[str, float]] = {}
        predictions_by_task: dict[str, list[dict[str, Any]]] = {}
        with self.bundle.evaluation_memory():
            for task in CANONICAL_TASKS:
                task_scores, rows = self._evaluate_one(split=split, task=task)
                scores[task] = task_scores
                predictions_by_task[task] = rows
        if was_training:
            self.bundle.model.train()
        return scores, predictions_by_task

    @torch.no_grad()
    def evaluate_task(
        self,
        *,
        split: str,
        task: str,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        was_training = self.bundle.model.training
        self.bundle.model.eval()
        with self.bundle.evaluation_memory():
            scores, rows = self._evaluate_one(split=split, task=task)
        if was_training:
            self.bundle.model.train()
        return scores, rows

    def _evaluate_one(
        self,
        *,
        split: str,
        task: str,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        dataset = self._dataset(split, task)
        source_rows = json.loads(
            (
                self.config.resolved_data_dir
                / self.data_manifest[split][task]["path"]
            ).read_text(encoding="utf-8")
        )
        rows: list[dict[str, Any]] = []
        compare_count = 0
        exact_count = 0
        for index in range(len(dataset)):
            batch = _to_device(self.collator([dataset[index]]), self.device)
            output_ids = self.bundle.model.generate(
                batch["input_ids_generate"],
                attention_mask=batch["attention_mask_generate"],
                max_new_tokens=self.config.eval_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            decoded = self.tokenizer.decode(
                output_ids[0].detach().cpu(),
                skip_special_tokens=False,
            )
            eos_text = self.tokenizer.eos_token or "<|endoftext|>"
            prediction = (
                decoded.split(eos_text, 1)[1].strip()
                if eos_text in decoded
                else decoded.strip()
            )
            target = str(batch["target_text"][0])
            source_row = source_rows[int(batch["row_index"][0])]
            result = score_prediction(
                task=task,
                target=target,
                prediction=prediction,
                question=str(source_row["question"]),
            )
            compare_count += int(result["compare_answers"])
            exact_count += int(result["exact_match"])
            rows.append(
                {
                    "row_index": int(batch["row_index"][0]),
                    "target": target,
                    "prediction": prediction,
                    **result,
                }
            )
        return (
            {
                "compare_answers": compare_count / len(dataset),
                "exact_match": exact_count / len(dataset),
                "rows": len(dataset),
            },
            rows,
        )

    def _store_eval(
        self,
        raw: dict[str, Any],
        row_index: int,
        scores: Mapping[str, Mapping[str, float]],
        predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> None:
        for column, task in enumerate(CANONICAL_TASKS):
            raw["compare_answers_matrix"][row_index][column] = float(
                scores[task]["compare_answers"]
            )
            raw["exact_match_matrix"][row_index][column] = float(
                scores[task]["exact_match"]
            )
        raw["predictions"][str(row_index)] = {
            task: list(predictions[task]) for task in CANONICAL_TASKS
        }
        compare_metrics = continual_metrics(
            raw["compare_answers_matrix"], self.config.resolved_order
        )
        stage_metrics = compare_metrics["stagewise"][row_index]
        cumulative = row_index * self.config.slow_steps_per_task
        self.monitor.stage(
            stage_index=row_index,
            cumulative_step=cumulative,
            compare_row=[float(scores[task]["compare_answers"]) for task in CANONICAL_TASKS],
            exact_row=[float(scores[task]["exact_match"]) for task in CANONICAL_TASKS],
            stage_metrics=stage_metrics,
        )
        print(
            json.dumps(
                {
                    "event": "evaluation",
                    "model": self.config.model,
                    "cl_method": self.config.cl_method,
                    "stage_index": row_index,
                    "after_task": stage_metrics.get("after_task"),
                    "cumulative_step": cumulative,
                    "compare_answers": {
                        task: float(scores[task]["compare_answers"])
                        for task in CANONICAL_TASKS
                    },
                    "exact_match": {
                        task: float(scores[task]["exact_match"])
                        for task in CANONICAL_TASKS
                    },
                    "mean_seen_accuracy": stage_metrics.get(
                        "mean_seen_accuracy"
                    ),
                    "forgetting": stage_metrics.get(
                        "forgetting_from_learning"
                    ),
                    "bwt": stage_metrics.get("bwt"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _save_progress(
        self,
        *,
        latest: Path,
        raw: Mapping[str, Any],
        stage_index: int,
        task_slow_step: int,
        sampler: EpochBatchCursor | None,
        active_train_state: Mapping[str, Any] | None,
    ) -> None:
        metadata = save_checkpoint(
            latest,
            config=self.config,
            data_manifest_sha256=self.data_manifest_sha256,
            model=self.bundle.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            strategy=self.strategy,
            progress={
                "stage_index": stage_index,
                "task_slow_step": task_slow_step,
                "raw": dict(raw),
                "active_train_state": (
                    None
                    if active_train_state is None
                    else dict(active_train_state)
                ),
                "fast_counters": (
                    {
                        "attempts": self.bundle.fast_cell.fast_update_attempts,
                        "applied": self.bundle.fast_cell.fast_updates_applied,
                    }
                    if self.bundle.fast_cell is not None
                    else {"attempts": 0, "applied": 0}
                ),
            },
            sampler_state=None if sampler is None else sampler.state_dict(),
            initial_model_sha256=self.initial_model_sha256,
        )
        print(
            json.dumps(
                {
                    "event": "checkpoint",
                    "path": str(latest),
                    "sha256": metadata["sha256"],
                    "stage_index": stage_index,
                    "task_slow_step": task_slow_step,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _write_status(
        self,
        path: Path,
        state: str,
        *,
        error: str | None = None,
    ) -> None:
        atomic_write_json(
            path,
            {
                "format": STATUS_FORMAT,
                "state": state,
                "protocol_hash": self.config.protocol_hash,
                "data_manifest_sha256": self.data_manifest_sha256,
                "updated_at_unix": time.time(),
                "error": error,
            },
        )
