from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .util import json_sha256


PROTOCOL_VERSION = "babilong-qa6-0k-cl-v1"
CANONICAL_TASKS = ("qa1", "qa2", "qa3", "qa11", "qa12", "qa13")
MODEL_KEYS = ("gpt2", "base_rmt", "fastmem0", "fastmem")
CL_METHODS = ("none", "si")
DEFAULT_DATA_SEED = 481113
DEFAULT_REPLICATE_SEED = 48
DEFAULT_ORDER_SEED = 48
DEFAULT_RESULTS_ROOT = Path("results/babilong_cl")
DEFAULT_PROTOCOL_PATH = Path(__file__).parent / "configs" / "qa6_0k_v1.json"


def stable_seed(namespace: str, seed: int, key: str) -> int:
    digest = hashlib.sha256(
        f"{namespace}:{int(seed)}:{key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def resolve_task_order(
    order_seed: int,
    tasks: Sequence[str] = CANONICAL_TASKS,
) -> tuple[str, ...]:
    if len(tasks) != len(set(tasks)):
        raise ValueError("Task list contains duplicates")
    return tuple(
        sorted(
            tasks,
            key=lambda task: hashlib.sha256(
                f"babilong-cl-order-v1:{int(order_seed)}:{task}".encode("utf-8")
            ).digest(),
        )
    )


def task_sampler_seeds(
    replicate_seed: int,
    tasks: Sequence[str] = CANONICAL_TASKS,
) -> dict[str, int]:
    return {
        task: stable_seed("babilong-cl-task-sampler-v1", replicate_seed, task)
        for task in tasks
    }


def condition_slug(model: str, cl_method: str, si_lambda: float | None) -> tuple[str, str]:
    if model == "gpt2":
        architecture = "gpt2"
    elif model == "base_rmt":
        architecture = "base-rmt16"
    elif model in {"fastmem0", "fastmem"}:
        architecture = "fastmem-rmt16"
    else:
        raise ValueError(f"Unsupported model: {model}")
    if model == "fastmem0":
        method = "fast0"
    elif model == "fastmem":
        method = "fast5e-3"
    elif cl_method == "si":
        if si_lambda is None:
            raise ValueError("SI condition requires si_lambda")
        method = f"si-l{float_slug(si_lambda)}"
    else:
        method = "none"
    return architecture, method


def float_slug(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Non-finite values cannot be used in paths")
    return f"{value:g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    cl_method: str = "none"
    si_lambda: float | None = None
    si_epsilon: float = 0.1
    si_decay: float = 1.0
    si_clamp_importance: bool = True
    replicate_seed: int = DEFAULT_REPLICATE_SEED
    order_seed: int = DEFAULT_ORDER_SEED
    data_seed: int = DEFAULT_DATA_SEED
    task_order: tuple[str, ...] | None = None
    results_root: str = str(DEFAULT_RESULTS_ROOT)
    data_dir: str | None = None
    backbone: str = "openai-community/gpt2"
    backbone_revision: str = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    tokenizer: str = "openai-community/gpt2"
    precision: str = "fp32"
    deterministic: bool = True
    slow_steps_per_task: int = 3001
    legacy_configured_iters: int = 3000
    warmup_steps: int = 300
    learning_rates: Mapping[str, float] = field(
        default_factory=lambda: {
            task: 3e-5 if task == "qa3" else 1e-5 for task in CANONICAL_TASKS
        }
    )
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    microbatch_size: int = 1
    slow_batch_size: int = 64
    fast_batch_size: int = 32
    fast_lr: float = 0.005
    fast_clip_norm: float = 1.0
    n_mem: int = 16
    segment_size: int = 512
    max_n_segments: int = 2
    eval_batch_size: int = 1
    eval_max_new_tokens: int = 10
    log_interval: int = 30
    checkpoint_interval: int = 250
    include_references: bool = True
    require_clean_git: bool = True
    minimum_free_disk_gb: float = 15.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.model not in MODEL_KEYS:
            raise ValueError(f"model must be one of {MODEL_KEYS}")
        if self.cl_method not in CL_METHODS:
            raise ValueError(f"cl_method must be one of {CL_METHODS}")
        if self.cl_method == "si" and self.si_lambda is None:
            raise ValueError("si_lambda is required for SI")
        if self.cl_method == "none" and self.si_lambda is not None:
            raise ValueError("si_lambda is only valid with cl_method=si")
        if self.model != "gpt2" and self.cl_method == "si":
            # Supported by the strategy implementation, but not canonical.
            pass
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.slow_steps_per_task <= 0:
            raise ValueError("slow_steps_per_task must be positive")
        if self.minimum_free_disk_gb < 0:
            raise ValueError("minimum_free_disk_gb must be nonnegative")
        if self.microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if self.slow_batch_size % self.microbatch_size:
            raise ValueError("microbatch_size must divide slow_batch_size")
        if self.fast_batch_size % self.microbatch_size:
            raise ValueError("microbatch_size must divide fast_batch_size")
        if self.slow_batch_size % self.fast_batch_size:
            raise ValueError("fast_batch_size must divide slow_batch_size")
        if tuple(sorted(self.learning_rates)) != tuple(sorted(CANONICAL_TASKS)):
            raise ValueError("learning_rates must define every canonical task exactly once")
        order = self.resolved_order
        if len(order) != len(CANONICAL_TASKS) or set(order) != set(CANONICAL_TASKS):
            raise ValueError(f"task_order must be a permutation of {CANONICAL_TASKS}")

    @property
    def resolved_order(self) -> tuple[str, ...]:
        return (
            tuple(self.task_order)
            if self.task_order is not None
            else resolve_task_order(self.order_seed)
        )

    @property
    def sampler_seeds(self) -> dict[str, int]:
        return task_sampler_seeds(self.replicate_seed)

    @property
    def architecture_and_method(self) -> tuple[str, str]:
        return condition_slug(self.model, self.cl_method, self.si_lambda)

    @property
    def order_slug(self) -> str:
        return "_".join(self.resolved_order)

    @property
    def run_dir(self) -> Path:
        architecture, method = self.architecture_and_method
        return (
            Path(self.results_root)
            / "runs"
            / architecture
            / method
            / f"replicate-{self.replicate_seed}"
            / f"order-{self.order_seed}-{self.order_slug}"
        )

    @property
    def tensorboard_dir(self) -> Path:
        architecture, method = self.architecture_and_method
        return (
            Path(self.results_root)
            / "tensorboard"
            / architecture
            / method
            / f"replicate-{self.replicate_seed}"
            / f"order-{self.order_seed}-{self.order_slug}"
        )

    @property
    def resolved_data_dir(self) -> Path:
        if self.data_dir:
            return Path(self.data_dir)
        return (
            Path(self.results_root)
            / "data"
            / "qa6-0k"
            / f"data-seed-{self.data_seed}"
        )

    @property
    def protocol_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_order"] = list(self.resolved_order)
        payload["task_sampler_seeds"] = self.sampler_seeds
        payload["protocol_version"] = PROTOCOL_VERSION
        payload["architecture"] = self.architecture_and_method[0]
        payload["method"] = self.architecture_and_method[1]
        payload.pop("results_root", None)
        payload.pop("data_dir", None)
        payload.pop("device", None)
        payload.pop("require_clean_git", None)
        payload.pop("include_references", None)
        payload.pop("checkpoint_interval", None)
        payload.pop("log_interval", None)
        payload.pop("minimum_free_disk_gb", None)
        return payload

    @property
    def protocol_hash(self) -> str:
        return json_sha256(self.protocol_payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_order"] = list(self.resolved_order)
        payload["task_sampler_seeds"] = self.sampler_seeds
        payload["protocol_version"] = PROTOCOL_VERSION
        payload["protocol_hash"] = self.protocol_hash
        payload["architecture"] = self.architecture_and_method[0]
        payload["method"] = self.architecture_and_method[1]
        payload["run_dir"] = str(self.run_dir)
        payload["tensorboard_dir"] = str(self.tensorboard_dir)
        payload["data_dir"] = str(self.resolved_data_dir)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        values = dict(payload)
        values.pop("protocol_version", None)
        values.pop("protocol_hash", None)
        values.pop("run_dir", None)
        values.pop("tensorboard_dir", None)
        values.pop("task_sampler_seeds", None)
        values.pop("architecture", None)
        values.pop("method", None)
        if values.get("task_order") is not None:
            values["task_order"] = tuple(values["task_order"])
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path, **overrides: Any) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload.update({key: value for key, value in overrides.items() if value is not None})
        return cls.from_mapping(payload)
