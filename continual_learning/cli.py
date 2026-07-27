from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_PROTOCOL_PATH,
    ExperimentConfig,
    MODEL_KEYS,
)
from .launcher import CONDITIONS, launch_suite
from .reporting import aggregate_results
from .runner import calibrate_si, prepare_data, run_experiment
from .util import file_sha256


DEFAULT_BABI_ARCHIVE = "data/tasks_1-20_v1-2.zip"


def _add_config_arguments(parser: argparse.ArgumentParser, *, model_required: bool) -> None:
    parser.add_argument("--config", default=str(DEFAULT_PROTOCOL_PATH))
    parser.add_argument("--model", choices=MODEL_KEYS, required=model_required)
    parser.add_argument("--cl-method", choices=("none", "si"), default=None)
    parser.add_argument("--si-lambda", type=float)
    parser.add_argument("--replicate-seed", type=int)
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--task-order", nargs=6)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--results-root")
    parser.add_argument("--data-dir")
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--steps-per-task", type=int)
    parser.add_argument("--train-minibatches-per-task", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--fast-lr", type=float)
    parser.add_argument("--minimum-free-disk-gb", type=float)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-references", action="store_true")


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if (
        getattr(args, "steps_per_task", None) is not None
        and payload.get("training_budget_mode", "fixed_slow_steps")
        == "fixed_train_minibatches"
    ):
        raise ValueError(
            "--steps-per-task is incompatible with a fixed-minibatch protocol; "
            "use --train-minibatches-per-task"
        )
    if (
        getattr(args, "train_minibatches_per_task", None) is not None
        and payload.get("training_budget_mode", "fixed_slow_steps")
        != "fixed_train_minibatches"
    ):
        raise ValueError(
            "--train-minibatches-per-task requires fixed_train_minibatches"
        )
    overrides = {
        "model": getattr(args, "model", None),
        "cl_method": getattr(args, "cl_method", None),
        "si_lambda": getattr(args, "si_lambda", None),
        "replicate_seed": getattr(args, "replicate_seed", None),
        "order_seed": getattr(args, "order_seed", None),
        "task_order": (
            tuple(args.task_order) if getattr(args, "task_order", None) else None
        ),
        "data_seed": getattr(args, "data_seed", None),
        "results_root": getattr(args, "results_root", None),
        "data_dir": getattr(args, "data_dir", None),
        "device": getattr(args, "device", None),
        "precision": getattr(args, "precision", None),
        "microbatch_size": getattr(args, "microbatch_size", None),
        "slow_steps_per_task": getattr(args, "steps_per_task", None),
        "train_minibatches_per_task": getattr(
            args, "train_minibatches_per_task", None
        ),
        "warmup_steps": getattr(args, "warmup_steps", None),
        "checkpoint_interval": getattr(args, "checkpoint_interval", None),
        "log_interval": getattr(args, "log_interval", None),
        "fast_lr": getattr(args, "fast_lr", None),
        "minimum_free_disk_gb": getattr(args, "minimum_free_disk_gb", None),
    }
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value
    payload["require_clean_git"] = bool(
        payload.get("require_clean_git", True)
    ) and not bool(getattr(args, "allow_dirty", False))
    if getattr(args, "no_references", False):
        payload["include_references"] = False
    if payload.get("cl_method") != "si":
        payload["si_lambda"] = None
    return ExperimentConfig.from_mapping(payload)


def _load_selected_si(results_root: str | Path) -> dict[str, Any]:
    path = Path(results_root) / "calibration" / "si" / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"SI selection is missing: {path}. Run `python -m continual_learning calibrate-si`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_selected_si(
    selection: dict[str, Any],
    config: ExperimentConfig,
) -> None:
    expected = selection.get("selection_context")
    if not isinstance(expected, dict):
        raise ValueError("SI selection is missing its protocol context")
    fields = (
        "backbone",
        "backbone_revision",
        "precision",
        "training_budget_mode",
        "train_minibatch_size",
        "train_minibatches_per_task",
        "fastmem_slow_update_freq",
        "warmup_ratio",
        "resolved_slow_steps_per_task",
        "resolved_warmup_steps",
        "learning_rates",
        "weight_decay",
        "clip_grad_norm",
        "label_mask_policy",
        "loss_normalization",
        "microbatch_size",
        "resolved_slow_batch_size",
        "data_seed",
        "si_epsilon",
        "si_decay",
        "si_clamp_importance",
    )
    current = config.to_dict()
    mismatches = [
        field for field in fields if current.get(field) != expected.get(field)
    ]
    if mismatches:
        raise ValueError(
            "SI selection does not match the current training protocol: "
            + ", ".join(mismatches)
        )


def verify_provenance() -> dict[str, Any]:
    root = Path(__file__).parent / "third_party" / "rmt_babilong_release"
    provenance = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
    results = {}
    for relative, info in provenance["files"].items():
        actual = file_sha256(root / relative)
        expected = info["sha256"]
        if actual != expected:
            raise RuntimeError(f"Provenance mismatch for {relative}: {actual} != {expected}")
        results[relative] = actual
    strategy_root = Path(__file__).parent / "strategies"
    strategy_provenance = json.loads(
        (strategy_root / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    strategy_path = strategy_root / "synaptic_intelligence.py"
    strategy_actual = file_sha256(strategy_path)
    strategy_expected = strategy_provenance["adapted_file_sha256"]
    if strategy_actual != strategy_expected:
        raise RuntimeError(
            "SI provenance mismatch for "
            f"{strategy_path}: {strategy_actual} != {strategy_expected}"
        )
    results["strategies/synaptic_intelligence.py"] = strategy_actual
    return results


def dry_run(config: ExperimentConfig, babi_archive: str) -> dict[str, Any]:
    manifest = prepare_data(config, babi_archive=babi_archive)
    fast_attempts = (
        len(config.resolved_order)
        * config.resolved_slow_steps_per_task
        * (config.resolved_slow_batch_size // config.fast_batch_size)
        if config.model in {"fastmem0", "fastmem"}
        else 0
    )
    disk = shutil.disk_usage(Path(config.results_root).resolve().parent)
    checkpoint_state_gb = 4.1 if config.cl_method == "si" else 1.6
    task_rows = {}
    for task in config.resolved_order:
        train = manifest["train"][task]
        stats = train["token_stats"]
        fitting = int(stats["fit_1024"])
        effective = (
            fitting // config.resolved_slow_batch_size
        ) * config.resolved_slow_batch_size
        task_rows[task] = {
            "source": train["source"],
            "source_rows": int(stats["rows"]),
            "fitting_rows": fitting,
            "effective_rows_per_epoch": effective,
            "dropped_tail_rows_per_epoch": fitting - effective,
            "source_max_serialized_tokens": int(stats["max"]),
            "training_max_serialized_tokens": int(stats["fit_max"]),
            "excluded_over_1024": int(stats["excluded_over_1024"]),
            "one_segment_rows": int(stats["one_segment_512"]),
            "two_segment_rows": int(stats["two_segment_1024"]),
            "slow_update_freq": config.slow_update_freq,
            "slow_steps": config.resolved_slow_steps_per_task,
            "train_minibatches": config.resolved_train_minibatches_per_task,
            "training_examples": config.training_examples_per_task,
            "warmup_steps": config.resolved_warmup_steps,
            "sampler_seed": config.sampler_seeds[task],
            "learning_rate": float(config.learning_rates[task]),
        }
    payload = {
        "config": config.to_dict(),
        "data_manifest_sha256": manifest["manifest_sha256"],
        "resolved_task_order": list(config.resolved_order),
        "task_protocol": task_rows,
        "total_cl_slow_steps": len(config.resolved_order)
        * config.resolved_slow_steps_per_task,
        "total_cl_train_minibatches": len(config.resolved_order)
        * config.resolved_train_minibatches_per_task,
        "training_examples_per_task": config.training_examples_per_task,
        "fast_update_attempts": fast_attempts,
        "single_task_references": (
            list(config.resolved_order) if config.include_references else []
        ),
        "free_disk_gb": disk.free / (1024**3),
        "minimum_free_disk_gb": config.minimum_free_disk_gb,
        "estimated_rolling_checkpoint_gb": checkpoint_state_gb,
        "estimated_atomic_checkpoint_peak_gb": 2.0 * checkpoint_state_gb,
        "result_tree": str(config.run_dir),
        "tensorboard": str(config.tensorboard_dir),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m continual_learning",
        description="Unified six-task BABILong continual-learning experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    _add_config_arguments(prepare, model_required=False)
    prepare.add_argument("--babi-archive", default=DEFAULT_BABI_ARCHIVE)

    dry = subparsers.add_parser("dry-run")
    _add_config_arguments(dry, model_required=True)
    dry.add_argument("--babi-archive", default=DEFAULT_BABI_ARCHIVE)

    run = subparsers.add_parser("run")
    _add_config_arguments(run, model_required=True)
    run.add_argument("--babi-archive", default=DEFAULT_BABI_ARCHIVE)
    run.add_argument("--no-tensorboard", action="store_true")
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--max-tasks", type=int)
    run.add_argument("--max-steps-per-task", type=int)

    calibrate = subparsers.add_parser("calibrate-si")
    _add_config_arguments(calibrate, model_required=False)
    calibrate.add_argument("--babi-archive", default=DEFAULT_BABI_ARCHIVE)
    calibrate.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    calibrate.add_argument("--no-tensorboard", action="store_true")

    suite = subparsers.add_parser("suite")
    _add_config_arguments(suite, model_required=False)
    suite.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITIONS),
        default=list(CONDITIONS),
    )
    suite.add_argument("--replicate-seeds", nargs="+", type=int, default=[48])
    suite.add_argument("--order-seeds", nargs="+", type=int, default=[48])
    suite.add_argument("--gpus", nargs="+", default=["0"])
    suite.add_argument("--jobs-per-gpu", type=int, default=1)
    suite.add_argument("--babi-archive", default=DEFAULT_BABI_ARCHIVE)
    suite.add_argument("--no-tensorboard", action="store_true")
    suite.add_argument("--dry-run", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results-root", default="results/babilong_cl")
    aggregate.add_argument("--output-dir")

    subparsers.add_parser("verify-provenance")
    subparsers.add_parser("smoke")
    gpu_smoke = subparsers.add_parser("gpu-smoke")
    _add_config_arguments(gpu_smoke, model_required=False)
    gpu_smoke.add_argument(
        "--models",
        nargs="+",
        choices=("gpt2", "gpt2-si", "base_rmt", "fastmem0", "fastmem"),
        default=["gpt2", "gpt2-si", "base_rmt", "fastmem0", "fastmem"],
    )
    gpu_smoke.add_argument("--babi-archive", default=DEFAULT_BABI_ARCHIVE)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "verify-provenance":
        print(json.dumps(verify_provenance(), indent=2, sort_keys=True))
        return
    if args.command == "smoke":
        from .smoke_tests import main as smoke_main

        smoke_main()
        return
    if args.command == "gpu-smoke":
        from .smoke_tests import gpu_smoke

        config = _config_from_args(args)
        manifest = prepare_data(config, babi_archive=args.babi_archive)
        gpu_smoke(
            base_config=config,
            data_manifest=manifest,
            models=list(args.models),
        )
        return
    if args.command == "aggregate":
        output = args.output_dir or str(Path(args.results_root) / "aggregates" / "latest")
        payload = aggregate_results(args.results_root, output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    si_selection = None
    if (
        args.command == "run"
        and args.cl_method == "si"
        and args.si_lambda is None
    ):
        base_payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        results_root = args.results_root or base_payload["results_root"]
        si_selection = _load_selected_si(results_root)
        args.si_lambda = float(si_selection["selected_si_lambda"])
    config = _config_from_args(args)
    if si_selection is not None:
        _validate_selected_si(si_selection, config)
    if args.command == "prepare-data":
        manifest = prepare_data(config, babi_archive=args.babi_archive)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if args.command == "dry-run":
        dry_run(config, args.babi_archive)
        return
    if args.command == "run":
        try:
            raw = run_experiment(
                config,
                babi_archive=args.babi_archive,
                tensorboard=not args.no_tensorboard,
                resume=not args.no_resume,
                max_tasks=args.max_tasks,
                max_steps_per_task=args.max_steps_per_task,
            )
        except InterruptedError:
            print(
                json.dumps(
                    {
                        "run_dir": str(config.run_dir),
                        "status": "interrupted",
                        "resume": True,
                    },
                    indent=2,
                )
            )
            raise SystemExit(130)
        print(json.dumps({"run_dir": str(config.run_dir), "status": raw["status"]}, indent=2))
        return
    if args.command == "calibrate-si":
        base = replace(
            config,
            model="gpt2",
            cl_method="none",
            si_lambda=None,
            include_references=False,
        )
        try:
            payload = calibrate_si(
                base,
                babi_archive=args.babi_archive,
                lambdas=tuple(args.lambdas),
                tensorboard=not args.no_tensorboard,
            )
        except InterruptedError as error:
            print(
                json.dumps(
                    {"status": "interrupted", "detail": str(error)},
                    indent=2,
                )
            )
            raise SystemExit(130)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "suite":
        selected = args.si_lambda
        selection = None
        if "gpt2-si" in args.conditions and selected is None:
            selection = _load_selected_si(config.results_root)
            selected = float(selection["selected_si_lambda"])
        if selection is not None:
            _validate_selected_si(selection, config)
        try:
            payload = launch_suite(
                base_config=config,
                conditions=args.conditions,
                replicate_seeds=args.replicate_seeds,
                order_seeds=args.order_seeds,
                gpus=args.gpus,
                jobs_per_gpu=args.jobs_per_gpu,
                si_lambda=selected,
                babi_archive=args.babi_archive,
                tensorboard=not args.no_tensorboard,
                allow_dirty=args.allow_dirty,
                dry_run=args.dry_run,
            )
        except InterruptedError as error:
            print(json.dumps({"status": "interrupted", "detail": str(error)}, indent=2))
            raise SystemExit(130)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    raise AssertionError(f"Unhandled command: {args.command}")
