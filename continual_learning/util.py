from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(destination, "")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


@contextlib.contextmanager
def file_lock(path: str | Path) -> Iterator[None]:
    import fcntl

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_rng_byte_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} RNG state must be a torch.Tensor")
    if value.dtype != torch.uint8:
        raise TypeError(f"{name} RNG state must have dtype torch.uint8")
    if value.ndim != 1:
        raise ValueError(f"{name} RNG state must be one-dimensional")
    # torch.load(map_location=<cuda device>) also remaps the RNG tensors stored
    # in a checkpoint. PyTorch's RNG restore APIs require CPU ByteTensors.
    return value.detach().cpu().contiguous()


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(
        _cpu_rng_byte_tensor(state["torch_cpu"], name="CPU")
    )
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError("CUDA RNG state must be a list or tuple")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA RNG state count does not match visible CUDA device count: "
                f"checkpoint={len(cuda_states)}, visible={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(
            [
                _cpu_rng_byte_tensor(value, name=f"CUDA device {index}")
                for index, value in enumerate(cuda_states)
            ]
        )


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def environment_manifest() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": os.uname().nodename,
    }
    if torch.cuda.is_available():
        result["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return result


def git_manifest(repo: str | Path) -> dict[str, Any]:
    root = Path(repo)

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"available": False}
    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "clean": status == "",
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def parameter_manifest(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in named_parameters
    ]


def temporary_directory(prefix: str) -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix=prefix)
