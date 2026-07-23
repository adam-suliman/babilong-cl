"""Trainer-independent Synaptic Intelligence with MQAR-compatible equations.

This is a behavioral adaptation of the SI baseline used in the local
incremental-MQAR experiments. It is not byte-identical to that trainer mixin,
the original Path Integral implementation, or Avalanche.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn


SI_STATE_FORMAT = "babilong-si-state-v1"


@dataclass(frozen=True)
class SynapticIntelligenceConfig:
    """Numerical settings matching the local incremental-MQAR SI baseline."""

    si_lambda: float
    epsilon: float = 0.1
    decay: float = 1.0
    clamp_importance: bool = True

    def __post_init__(self) -> None:
        values = {
            "si_lambda": self.si_lambda,
            "epsilon": self.epsilon,
            "decay": self.decay,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.si_lambda < 0:
            raise ValueError("si_lambda must be nonnegative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.decay < 0:
            raise ValueError("decay must be nonnegative")

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def add_cl_strategy_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the stable CLI contract for future unified CL runners."""

    parser.add_argument("--cl_method", choices=("none", "si"), default="none")
    parser.add_argument(
        "--si_lambda",
        type=float,
        default=None,
        help="Required explicitly when --cl_method si is selected.",
    )
    parser.add_argument("--si_epsilon", type=float, default=0.1)
    parser.add_argument("--si_decay", type=float, default=1.0)
    parser.add_argument(
        "--si_clamp_importance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def si_config_from_args(args: argparse.Namespace) -> SynapticIntelligenceConfig | None:
    """Resolve the strategy CLI contract without coupling it to a runner."""

    method = str(getattr(args, "cl_method", "none"))
    si_lambda = getattr(args, "si_lambda", None)
    if method == "none":
        if si_lambda is not None:
            raise ValueError("--si_lambda is only valid with --cl_method si")
        return None
    if method != "si":
        raise ValueError(f"Unsupported continual-learning method: {method!r}")
    if si_lambda is None:
        raise ValueError("--si_lambda must be specified explicitly with --cl_method si")
    return SynapticIntelligenceConfig(
        si_lambda=float(si_lambda),
        epsilon=float(getattr(args, "si_epsilon", 0.1)),
        decay=float(getattr(args, "si_decay", 1.0)),
        clamp_importance=bool(getattr(args, "si_clamp_importance", True)),
    )


class SynapticIntelligence:
    """Track SI importance over trainable parameters owned by one optimizer.

    Call ``penalty`` while constructing every microbatch loss. After all
    accumulated backward passes and slow-gradient clipping, call
    ``before_optimizer_step``, then the optimizer step, then
    ``after_optimizer_step``. This makes one SI update correspond to one slow
    optimizer update.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: SynapticIntelligenceConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config

        self._parameters = self._optimizer_owned_parameters()
        self._manifest = self._parameter_manifest()
        self._manifest_hash = self._hash_manifest(self._manifest)

        self._importance: dict[str, torch.Tensor] | None = None
        self._reference: dict[str, torch.Tensor] | None = None
        self._task_start: dict[str, torch.Tensor] | None = None
        self._previous: dict[str, torch.Tensor] | None = None
        self._path_integral: dict[str, torch.Tensor] | None = None
        self._pending_gradients: dict[str, torch.Tensor | None] | None = None

        self._active_task: str | None = None
        self._optimizer_steps = 0
        self._task_optimizer_steps = 0
        self._completed_tasks = 0
        self._last_raw_penalty = 0.0
        self._last_weighted_penalty = 0.0
        self._last_stage_importance_mean_abs = 0.0
        self._last_stage_path_integral_mean_abs = 0.0

    @property
    def enabled(self) -> bool:
        return self.config.si_lambda != 0.0

    @property
    def active_task(self) -> str | None:
        return self._active_task

    @property
    def parameter_manifest(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._manifest]

    @property
    def parameter_manifest_hash(self) -> str:
        return self._manifest_hash

    def begin_task(self, task_id: str) -> None:
        task_id = str(task_id)
        if not task_id:
            raise ValueError("task_id must be nonempty")
        if self._active_task is not None:
            raise RuntimeError(
                f"Cannot begin task {task_id!r}; task {self._active_task!r} is active"
            )
        if self._pending_gradients is not None:
            raise RuntimeError("Cannot begin a task with an unfinished optimizer-step hook")
        self._validate_current_parameters()
        self._active_task = task_id
        self._task_optimizer_steps = 0
        if self.enabled:
            self._task_start = self._clone_parameters()
            self._previous = self._clone_parameters()
            self._path_integral = {
                name: torch.zeros_like(parameter)
                for name, parameter in self._parameters.items()
            }

    def penalty(self) -> torch.Tensor:
        """Return the weighted SI penalty for the current parameter values."""

        self._require_active_task()
        zero = self._zero_scalar()
        if not self.enabled or self._importance is None or self._reference is None:
            self._last_raw_penalty = 0.0
            self._last_weighted_penalty = 0.0
            return zero

        raw_penalty: torch.Tensor | None = None
        for name, parameter in self._parameters.items():
            term = (
                self._importance[name]
                * (parameter - self._reference[name]).pow(2)
            ).sum()
            raw_penalty = term if raw_penalty is None else raw_penalty + term
        if raw_penalty is None:
            return zero
        weighted_penalty = self.config.si_lambda * raw_penalty
        self._last_raw_penalty = float(raw_penalty.detach().cpu())
        self._last_weighted_penalty = float(weighted_penalty.detach().cpu())
        return weighted_penalty

    def before_optimizer_step(self) -> None:
        """Capture final accumulated, clipped gradients before AdamW updates."""

        self._require_active_task()
        if self._pending_gradients is not None:
            raise RuntimeError("before_optimizer_step called twice without an optimizer step")
        self._validate_current_parameters()
        if not self.enabled:
            self._pending_gradients = {}
            return
        gradients: dict[str, torch.Tensor | None] = {}
        for name, parameter in self._parameters.items():
            gradient = parameter.grad
            if gradient is not None and not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite gradient for SI parameter {name!r}")
            gradients[name] = (
                gradient.detach().clone() if gradient is not None else None
            )
        self._pending_gradients = gradients

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        """Accumulate the MQAR path integral after the slow optimizer update."""

        self._require_active_task()
        if self._pending_gradients is None:
            raise RuntimeError("after_optimizer_step called without before_optimizer_step")
        if self.enabled:
            assert self._previous is not None
            assert self._path_integral is not None
            for name, parameter in self._parameters.items():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(
                        f"Non-finite parameter after optimizer step: {name!r}"
                    )
                gradient = self._pending_gradients[name]
                if gradient is not None:
                    delta = parameter.detach() - self._previous[name]
                    self._path_integral[name].add_(-gradient * delta)
                self._previous[name] = parameter.detach().clone()
        self._pending_gradients = None
        self._optimizer_steps += 1
        self._task_optimizer_steps += 1

    @torch.no_grad()
    def end_task(self, task_id: str) -> dict[str, float | int | str | None]:
        """Consolidate importance at a task boundary and clear working state."""

        self._require_active_task()
        if str(task_id) != self._active_task:
            raise ValueError(
                f"Cannot end task {task_id!r}; active task is {self._active_task!r}"
            )
        if self._pending_gradients is not None:
            raise RuntimeError("Cannot end a task with an unfinished optimizer-step hook")
        if self._task_optimizer_steps <= 0:
            raise RuntimeError("Cannot consolidate SI for a task with no optimizer steps")

        if self.enabled:
            assert self._task_start is not None
            assert self._path_integral is not None
            stage_importance: dict[str, torch.Tensor] = {}
            for name, parameter in self._parameters.items():
                total_delta = parameter.detach() - self._task_start[name]
                importance = self._path_integral[name] / (
                    total_delta.pow(2) + self.config.epsilon
                )
                if self.config.clamp_importance:
                    importance = torch.clamp(importance, min=0.0)
                if not torch.isfinite(importance).all():
                    raise FloatingPointError(
                        f"Non-finite consolidated SI importance for {name!r}"
                    )
                stage_importance[name] = importance.detach().clone()

            if self._importance is None:
                self._importance = stage_importance
            else:
                self._importance = {
                    name: (
                        self.config.decay * self._importance[name]
                        + stage_importance[name]
                    )
                    for name in self._parameters
                }
            self._reference = self._clone_parameters()
            self._last_stage_importance_mean_abs = self._mean_abs(stage_importance)
            self._last_stage_path_integral_mean_abs = self._mean_abs(
                self._path_integral
            )

        completed_task = self._active_task
        self._completed_tasks += 1
        self._active_task = None
        self._task_optimizer_steps = 0
        self._task_start = None
        self._previous = None
        self._path_integral = None
        metrics = self.diagnostics()
        metrics["si/completed_task"] = completed_task
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Return portable SI state; saving during a pending step is forbidden."""

        if self._pending_gradients is not None:
            raise RuntimeError(
                "Cannot serialize SI between before_optimizer_step and "
                "after_optimizer_step"
            )
        self._validate_current_parameters()
        return {
            "format": SI_STATE_FORMAT,
            "config": self.config.to_dict(),
            "parameter_manifest": self.parameter_manifest,
            "parameter_manifest_hash": self._manifest_hash,
            "active_task": self._active_task,
            "optimizer_steps": self._optimizer_steps,
            "task_optimizer_steps": self._task_optimizer_steps,
            "completed_tasks": self._completed_tasks,
            "importance": self._portable_tensor_map(self._importance),
            "reference": self._portable_tensor_map(self._reference),
            "task_start": self._portable_tensor_map(self._task_start),
            "previous": self._portable_tensor_map(self._previous),
            "path_integral": self._portable_tensor_map(self._path_integral),
            "last_raw_penalty": self._last_raw_penalty,
            "last_weighted_penalty": self._last_weighted_penalty,
            "last_stage_importance_mean_abs": (
                self._last_stage_importance_mean_abs
            ),
            "last_stage_path_integral_mean_abs": (
                self._last_stage_path_integral_mean_abs
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore SI state after validating configuration and parameter geometry."""

        if self._pending_gradients is not None:
            raise RuntimeError("Cannot load SI state with an unfinished optimizer-step hook")
        if state.get("format") != SI_STATE_FORMAT:
            raise ValueError(f"Unsupported SI state format: {state.get('format')!r}")
        if state.get("config") != self.config.to_dict():
            raise ValueError("SI state configuration does not match the current configuration")
        self._validate_current_parameters()
        if state.get("parameter_manifest_hash") != self._manifest_hash:
            raise ValueError("SI optimizer parameter ordering or geometry changed")
        if state.get("parameter_manifest") != self._manifest:
            raise ValueError("SI optimizer parameter manifest changed")

        active_task = state.get("active_task")
        if active_task is not None and (not isinstance(active_task, str) or not active_task):
            raise ValueError("SI state has an invalid active task")
        optimizer_steps = self._nonnegative_int(state, "optimizer_steps")
        task_optimizer_steps = self._nonnegative_int(
            state, "task_optimizer_steps"
        )
        completed_tasks = self._nonnegative_int(state, "completed_tasks")
        if task_optimizer_steps > optimizer_steps:
            raise ValueError("SI task optimizer steps exceed total optimizer steps")
        if active_task is None and task_optimizer_steps != 0:
            raise ValueError("Inactive SI state has nonzero task optimizer steps")

        importance = self._restore_tensor_map("importance", state.get("importance"))
        reference = self._restore_tensor_map("reference", state.get("reference"))
        task_start = self._restore_tensor_map("task_start", state.get("task_start"))
        previous = self._restore_tensor_map("previous", state.get("previous"))
        path_integral = self._restore_tensor_map(
            "path_integral", state.get("path_integral")
        )
        if (importance is None) != (reference is None):
            raise ValueError("SI importance and reference must be present together")
        task_maps = (task_start, previous, path_integral)
        if active_task is None and any(value is not None for value in task_maps):
            raise ValueError("Inactive SI state contains task-working tensors")
        if self.enabled and active_task is not None and any(
            value is None for value in task_maps
        ):
            raise ValueError("Active SI state is missing task-working tensors")
        if not self.enabled and any(
            value is not None
            for value in (importance, reference, task_start, previous, path_integral)
        ):
            raise ValueError("Disabled SI state must not contain SI tensor buffers")

        self._importance = importance
        self._reference = reference
        self._task_start = task_start
        self._previous = previous
        self._path_integral = path_integral
        self._active_task = active_task
        self._optimizer_steps = optimizer_steps
        self._task_optimizer_steps = task_optimizer_steps
        self._completed_tasks = completed_tasks
        self._last_raw_penalty = self._finite_float(state, "last_raw_penalty")
        self._last_weighted_penalty = self._finite_float(
            state, "last_weighted_penalty"
        )
        self._last_stage_importance_mean_abs = self._finite_float(
            state, "last_stage_importance_mean_abs"
        )
        self._last_stage_path_integral_mean_abs = self._finite_float(
            state, "last_stage_path_integral_mean_abs"
        )

    def diagnostics(self) -> dict[str, float | int | str | None]:
        importance_mean, importance_max, nonzero_fraction = self._tensor_stats(
            self._importance
        )
        return {
            "si/active": int(self.enabled),
            "si/lambda": float(self.config.si_lambda),
            "si/epsilon": float(self.config.epsilon),
            "si/decay": float(self.config.decay),
            "si/clamp_importance": int(self.config.clamp_importance),
            "si/active_task": self._active_task,
            "si/raw_penalty": float(self._last_raw_penalty),
            "si/weighted_penalty": float(self._last_weighted_penalty),
            "si/importance_mean_abs": importance_mean,
            "si/importance_max_abs": importance_max,
            "si/importance_nonzero_fraction": nonzero_fraction,
            "si/stage_importance_mean_abs": float(
                self._last_stage_importance_mean_abs
            ),
            "si/path_integral_mean_abs": (
                self._mean_abs(self._path_integral)
                if self._path_integral is not None
                else float(self._last_stage_path_integral_mean_abs)
            ),
            "si/protected_parameter_tensors": len(self._parameters),
            "si/protected_parameters": sum(
                parameter.numel() for parameter in self._parameters.values()
            ),
            "si/buffer_bytes": self._buffer_bytes(),
            "si/optimizer_steps": self._optimizer_steps,
            "si/task_optimizer_steps": self._task_optimizer_steps,
            "si/completed_tasks": self._completed_tasks,
        }

    def _optimizer_owned_parameters(self) -> dict[str, nn.Parameter]:
        names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        result: dict[str, nn.Parameter] = {}
        seen: set[int] = set()
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for parameter_idx, parameter in enumerate(group["params"]):
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError(
                        f"Optimizer group {group_idx} item {parameter_idx} is not "
                        "an nn.Parameter"
                    )
                parameter_id = id(parameter)
                if parameter_id in seen:
                    raise ValueError("Optimizer contains the same parameter more than once")
                seen.add(parameter_id)
                name = names.get(parameter_id)
                if name is None:
                    raise ValueError(
                        "Optimizer contains a parameter that is not owned by the model"
                    )
                if parameter.requires_grad:
                    result[name] = parameter
        if not result:
            raise ValueError("Optimizer owns no trainable model parameters")
        return result

    def _parameter_manifest(self) -> list[dict[str, Any]]:
        group_position: dict[int, tuple[int, int]] = {}
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for parameter_idx, parameter in enumerate(group["params"]):
                group_position[id(parameter)] = (group_idx, parameter_idx)
        return [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
                "numel": parameter.numel(),
                "optimizer_group": group_position[id(parameter)][0],
                "optimizer_index": group_position[id(parameter)][1],
            }
            for name, parameter in self._parameters.items()
        ]

    @staticmethod
    def _hash_manifest(manifest: list[dict[str, Any]]) -> str:
        payload = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _validate_current_parameters(self) -> None:
        current = self._optimizer_owned_parameters()
        if list(current) != list(self._parameters):
            raise ValueError("SI optimizer-owned parameter names or ordering changed")
        if any(
            current[name] is not self._parameters[name] for name in self._parameters
        ):
            raise ValueError("SI optimizer-owned parameter identities changed")
        manifest = self._parameter_manifest()
        if manifest != self._manifest or self._hash_manifest(manifest) != self._manifest_hash:
            raise ValueError("SI optimizer parameter geometry changed")

    def _require_active_task(self) -> None:
        if self._active_task is None:
            raise RuntimeError("SI operation requires an active task")

    def _zero_scalar(self) -> torch.Tensor:
        parameter = next(iter(self._parameters.values()))
        return torch.zeros((), dtype=parameter.dtype, device=parameter.device)

    def _clone_parameters(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self._parameters.items()
        }

    @staticmethod
    def _portable_tensor_map(
        tensors: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor] | None:
        if tensors is None:
            return None
        return {
            name: tensor.detach().cpu().clone() for name, tensor in tensors.items()
        }

    def _restore_tensor_map(
        self,
        field: str,
        tensors: Any,
    ) -> dict[str, torch.Tensor] | None:
        if tensors is None:
            return None
        if not isinstance(tensors, Mapping) or list(tensors) != list(self._parameters):
            raise ValueError(f"SI state field {field!r} has incompatible parameter keys")
        restored: dict[str, torch.Tensor] = {}
        for name, parameter in self._parameters.items():
            tensor = tensors[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"SI state field {field!r}/{name!r} is not a tensor")
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"SI state field {field!r}/{name!r} has shape "
                    f"{tuple(tensor.shape)}, expected {tuple(parameter.shape)}"
                )
            tensor = tensor.detach().to(
                device=parameter.device, dtype=parameter.dtype
            ).clone()
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(
                    f"SI state field {field!r}/{name!r} is non-finite"
                )
            restored[name] = tensor
        return restored

    @staticmethod
    def _nonnegative_int(state: Mapping[str, Any], field: str) -> int:
        value = state.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"SI state field {field!r} must be a nonnegative integer")
        return value

    @staticmethod
    def _finite_float(state: Mapping[str, Any], field: str) -> float:
        value = float(state.get(field, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"SI state field {field!r} must be finite")
        return value

    @staticmethod
    def _mean_abs(tensors: Mapping[str, torch.Tensor]) -> float:
        means = [tensor.detach().abs().mean().item() for tensor in tensors.values()]
        return float(sum(means) / len(means)) if means else 0.0

    @staticmethod
    def _tensor_stats(
        tensors: Mapping[str, torch.Tensor] | None,
    ) -> tuple[float, float, float]:
        if not tensors:
            return 0.0, 0.0, 0.0
        total = sum(tensor.numel() for tensor in tensors.values())
        absolute_sum = sum(
            float(tensor.detach().abs().sum().cpu()) for tensor in tensors.values()
        )
        maximum = max(
            float(tensor.detach().abs().max().cpu()) for tensor in tensors.values()
        )
        nonzero = sum(
            int(torch.count_nonzero(tensor.detach()).cpu())
            for tensor in tensors.values()
        )
        return absolute_sum / total, maximum, nonzero / total

    def _buffer_bytes(self) -> int:
        maps = (
            self._importance,
            self._reference,
            self._task_start,
            self._previous,
            self._path_integral,
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor_map in maps
            if tensor_map is not None
            for tensor in tensor_map.values()
        )
