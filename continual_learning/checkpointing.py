from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .config import ExperimentConfig
from .strategies import SynapticIntelligence
from .util import (
    atomic_torch_save,
    atomic_write_json,
    capture_rng_state,
    file_sha256,
    json_sha256,
    parameter_manifest,
    restore_rng_state,
)


CHECKPOINT_FORMAT = "babilong-qa6-cl-checkpoint-v1"


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_parameter_manifest(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[list[dict[str, Any]], str]:
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) in owned
    ]
    manifest = parameter_manifest(named)
    if len(manifest) != len(owned):
        raise RuntimeError("Optimizer contains an unnamed or duplicated parameter")
    return manifest, json_sha256(manifest)


def save_checkpoint(
    path: str | Path,
    *,
    config: ExperimentConfig,
    data_manifest_sha256: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    strategy: SynapticIntelligence | None,
    progress: Mapping[str, Any],
    sampler_state: Mapping[str, Any] | None,
    initial_model_sha256: str,
) -> dict[str, Any]:
    manifest, manifest_hash = optimizer_parameter_manifest(model, optimizer)
    envelope = {
        "format": CHECKPOINT_FORMAT,
        "protocol_hash": config.protocol_hash,
        "config": config.to_dict(),
        "data_manifest_sha256": data_manifest_sha256,
        "optimizer_parameter_manifest": manifest,
        "optimizer_parameter_manifest_sha256": manifest_hash,
        "initial_model_sha256": initial_model_sha256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "strategy": None if strategy is None else strategy.state_dict(),
        "progress": dict(progress),
        "sampler_state": None if sampler_state is None else dict(sampler_state),
        "rng_state": capture_rng_state(),
    }
    destination = Path(path)
    atomic_torch_save(destination, envelope)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "path": str(destination),
        "sha256": file_sha256(destination),
        "protocol_hash": config.protocol_hash,
        "data_manifest_sha256": data_manifest_sha256,
        "optimizer_parameter_manifest_sha256": manifest_hash,
        "initial_model_sha256": initial_model_sha256,
        "progress": dict(progress),
    }
    atomic_write_json(destination.with_suffix(".json"), metadata)
    return metadata


def load_checkpoint(
    path: str | Path,
    *,
    config: ExperimentConfig,
    data_manifest_sha256: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    strategy: SynapticIntelligence | None,
    initial_model_sha256: str,
    map_location: str | torch.device,
) -> dict[str, Any]:
    source = Path(path)
    state = torch.load(source, map_location=map_location)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {state.get('format')!r}")
    if state.get("protocol_hash") != config.protocol_hash:
        raise ValueError("Checkpoint protocol hash does not match")
    if state.get("data_manifest_sha256") != data_manifest_sha256:
        raise ValueError("Checkpoint data manifest hash does not match")
    if state.get("initial_model_sha256") != initial_model_sha256:
        raise ValueError("Checkpoint initial model hash does not match")
    manifest, manifest_hash = optimizer_parameter_manifest(model, optimizer)
    if state.get("optimizer_parameter_manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint optimizer parameter manifest hash changed")
    if state.get("optimizer_parameter_manifest") != manifest:
        raise ValueError("Checkpoint optimizer parameter manifest changed")
    if (state.get("strategy") is None) != (strategy is None):
        raise ValueError("Checkpoint CL strategy presence does not match")

    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if strategy is not None:
        strategy.load_state_dict(state["strategy"])
    restore_rng_state(state["rng_state"])
    return {
        "scheduler_state": state.get("scheduler"),
        "progress": dict(state["progress"]),
        "sampler_state": state.get("sampler_state"),
        "checkpoint_sha256": file_sha256(source),
    }


def finalize_checkpoint(latest: str | Path, final: str | Path) -> None:
    latest_path = Path(latest)
    final_path = Path(final)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_suffix(final_path.suffix + f".{os.getpid()}.tmp")
    try:
        os.link(latest_path, temporary)
    except OSError:
        import shutil

        shutil.copy2(latest_path, temporary)
    os.replace(temporary, final_path)
    latest_metadata = latest_path.with_suffix(".json")
    if latest_metadata.exists():
        import json

        payload = json.loads(latest_metadata.read_text(encoding="utf-8"))
        payload["path"] = str(final_path)
        payload["sha256"] = file_sha256(final_path)
        atomic_write_json(final_path.with_suffix(".json"), payload)
