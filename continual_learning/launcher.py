from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .config import ExperimentConfig
from .util import atomic_write_json, file_sha256


CONDITIONS = {
    "gpt2": ("gpt2", "none"),
    "gpt2-si": ("gpt2", "si"),
    "base-rmt": ("base_rmt", "none"),
    "fastmem0": ("fastmem0", "none"),
    "fastmem": ("fastmem", "none"),
}


def _take_launchable_assignment(
    pending: list[dict[str, Any]],
    slots: list[tuple[str, int]],
    running: dict[
        int,
        tuple[
            subprocess.Popen[Any],
            tuple[str, int],
            dict[str, Any],
            Any,
        ],
    ],
) -> tuple[dict[str, Any], tuple[str, int]] | None:
    for job_index, job in enumerate(pending):
        for slot_index, slot in enumerate(slots):
            gpu, _ = slot
            if job["cl_method"] == "si" and any(
                running_slot[0] == gpu
                and running_job["cl_method"] == "si"
                for _, running_slot, running_job, _ in running.values()
            ):
                continue
            return pending.pop(job_index), slots.pop(slot_index)
    return None


def launch_suite(
    *,
    base_config: ExperimentConfig,
    conditions: Sequence[str],
    replicate_seeds: Sequence[int],
    order_seeds: Sequence[int],
    gpus: Sequence[str],
    jobs_per_gpu: int,
    si_lambda: float | None,
    babi_archive: str,
    tensorboard: bool,
    allow_dirty: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown suite conditions: {unknown}")
    if jobs_per_gpu <= 0:
        raise ValueError("jobs_per_gpu must be positive")
    if not gpus:
        raise ValueError("At least one GPU must be provided")
    if "gpt2-si" in conditions and si_lambda is None:
        raise ValueError("gpt2-si requires a selected SI lambda")

    suite_id = (
        f"r{'-'.join(map(str, replicate_seeds))}"
        f"_o{'-'.join(map(str, order_seeds))}_"
        f"{int(time.time())}_{os.getpid()}_{time.time_ns() % 1_000_000}"
    )
    suite_dir = Path(base_config.results_root) / "suites" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    jobs: list[dict[str, Any]] = []
    for condition in conditions:
        model, method = CONDITIONS[condition]
        for replicate_seed in replicate_seeds:
            for order_seed in order_seeds:
                config = replace(
                    base_config,
                    model=model,
                    cl_method=method,
                    si_lambda=(si_lambda if method == "si" else None),
                    replicate_seed=int(replicate_seed),
                    order_seed=int(order_seed),
                    require_clean_git=not allow_dirty,
                )
                config_path = (
                    suite_dir
                    / "configs"
                    / (
                        f"{condition}_r{replicate_seed}_o{order_seed}_"
                        f"{config.protocol_hash[:12]}.json"
                    )
                )
                atomic_write_json(config_path, config.to_dict())
                jobs.append(
                    {
                        "condition": condition,
                        "model": model,
                        "cl_method": method,
                        "replicate_seed": replicate_seed,
                        "order_seed": order_seed,
                        "resolved_order": list(config.resolved_order),
                        "run_dir": str(config.run_dir),
                        "tensorboard_dir": str(config.tensorboard_dir),
                        "protocol_hash": config.protocol_hash,
                        "config_path": str(config_path),
                    }
                )
    completed_jobs = []
    pending_jobs = []
    for job in jobs:
        raw_path = Path(job["run_dir"]) / "raw.json"
        final_path = Path(job["run_dir"]) / "checkpoints" / "final.pt"
        final_metadata_path = final_path.with_suffix(".json")
        complete = False
        if raw_path.is_file() and final_path.is_file() and final_metadata_path.is_file():
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            final_metadata = json.loads(
                final_metadata_path.read_text(encoding="utf-8")
            )
            complete = (
                raw.get("status") == "complete"
                and raw.get("config", {}).get("protocol_hash")
                == job["protocol_hash"]
                and (
                    not base_config.include_references
                    or raw.get("references") is not None
                )
                and final_metadata.get("sha256") == file_sha256(final_path)
            )
        if complete:
            completed_jobs.append({**job, "status": "skipped_complete"})
        else:
            pending_jobs.append(job)
    active_slots = min(len(pending_jobs), len(gpus) * jobs_per_gpu)
    free_disk_gb = shutil.disk_usage(
        Path(base_config.results_root).resolve().parent
    ).free / (1024**3)
    checkpoint_estimates = {
        "ordinary_rolling_gb": 1.6,
        "ordinary_atomic_peak_gb": 3.2,
        "si_rolling_gb": 4.1,
        "si_atomic_peak_gb": 8.2,
    }
    pending_atomic_peaks = sorted(
        (
            checkpoint_estimates["si_atomic_peak_gb"]
            if job["cl_method"] == "si"
            else checkpoint_estimates["ordinary_atomic_peak_gb"]
        )
        for job in pending_jobs
    )
    active_atomic_peaks = (
        pending_atomic_peaks[-active_slots:] if active_slots else []
    )
    estimated_required_gb = max(
        base_config.minimum_free_disk_gb,
        5.0 + sum(active_atomic_peaks),
    )
    disk_preflight_passed = free_disk_gb >= estimated_required_gb
    if not disk_preflight_passed and not dry_run:
        raise RuntimeError(
            f"Suite disk preflight failed: {free_disk_gb:.1f} GB free, "
            f"{estimated_required_gb:.1f} GB required"
        )
    manifest = {
        "format": "babilong-qa6-suite-v1",
        "suite_id": suite_id,
        "conditions": list(conditions),
        "replicate_seeds": list(replicate_seeds),
        "order_seeds": list(order_seeds),
        "gpus": list(gpus),
        "jobs_per_gpu": jobs_per_gpu,
        "gpu_placement_policy": "at-most-one-si-job-per-gpu",
        "si_lambda": si_lambda,
        "jobs": jobs,
        "skipped_completed_jobs": completed_jobs,
        "status": "planned" if dry_run else "running",
        "free_disk_gb": free_disk_gb,
        "estimated_required_gb": estimated_required_gb,
        "disk_preflight_passed": disk_preflight_passed,
        "checkpoint_estimates": checkpoint_estimates,
        "started_at_unix": time.time(),
    }
    atomic_write_json(suite_dir / "manifest.json", manifest)
    if dry_run:
        manifest["finished_at_unix"] = time.time()
        atomic_write_json(suite_dir / "manifest.json", manifest)
        return manifest

    slots = list(
        (gpu, slot) for gpu in gpus for slot in range(jobs_per_gpu)
    )
    pending = list(pending_jobs)
    running: dict[
        int,
        tuple[subprocess.Popen[Any], tuple[str, int], dict[str, Any], Any],
    ] = {}
    failures: list[dict[str, Any]] = []
    launched_jobs: list[dict[str, Any]] = []
    stop_requested = False

    def terminate(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        for process, _, _, _ in running.values():
            process.send_signal(signal.SIGTERM)

    previous_int = signal.signal(signal.SIGINT, terminate)
    previous_term = signal.signal(signal.SIGTERM, terminate)
    try:
        while pending or running:
            while pending and slots and not stop_requested and not failures:
                assignment = _take_launchable_assignment(
                    pending,
                    slots,
                    running,
                )
                if assignment is None:
                    break
                job, slot = assignment
                gpu, _ = slot
                log_path = suite_dir / (
                    f"{job['condition']}_r{job['replicate_seed']}"
                    f"_o{job['order_seed']}.log"
                )
                log_handle = log_path.open("a", encoding="utf-8")
                command = [
                    sys.executable,
                    "-m",
                    "continual_learning",
                    "run",
                    "--config",
                    job["config_path"],
                    "--model",
                    job["model"],
                    "--babi-archive",
                    babi_archive,
                    "--device",
                    "cuda:0",
                ]
                if not tensorboard:
                    command.append("--no-tensorboard")
                if allow_dirty:
                    command.append("--allow-dirty")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
                running[process.pid] = (process, slot, job, log_handle)
                launched_jobs.append({**job, "gpu": gpu, "log": str(log_path)})

            finished = []
            for pid, (process, slot, job, log_handle) in running.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                log_handle.close()
                slots.append(slot)
                if return_code != 0:
                    failures.append({**job, "return_code": return_code})
                finished.append(pid)
            for pid in finished:
                del running[pid]
            if running and not finished:
                time.sleep(2)
            if failures:
                if running:
                    for process, _, _, _ in running.values():
                        process.send_signal(signal.SIGTERM)
                    for process, _, _, log_handle in running.values():
                        process.wait()
                        log_handle.close()
                    running.clear()
                break
            if stop_requested and not running:
                break
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)

    manifest["status"] = (
        "interrupted"
        if stop_requested
        else ("failed" if failures else "complete")
    )
    manifest["failures"] = failures
    manifest["launched_jobs"] = launched_jobs
    manifest["unlaunched_jobs"] = list(pending)
    manifest["finished_at_unix"] = time.time()
    atomic_write_json(suite_dir / "manifest.json", manifest)
    if stop_requested:
        raise InterruptedError(
            f"Suite interrupted; child runs checkpointed at slow-step boundaries: {suite_dir}"
        )
    if failures:
        raise RuntimeError(f"{len(failures)} suite jobs failed; see {suite_dir}")
    return manifest
