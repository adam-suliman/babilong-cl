from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn

from .config import ExperimentConfig
from .third_party.rmt_babilong_release.modeling_rmt.language_modeling import (
    MemoryCell,
    RecurrentWrapper,
)


class FastMemoryCell(MemoryCell):
    """MQAR-compatible fast-memory extension of the exact RMT memory cell."""

    is_fast_memory_cell = True

    def create_memory(self, num_mem_tokens: int) -> None:
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim = getattr(self.model.config, "n_embd", self.model.config.hidden_size)
        memory = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter(
            "initial_memory_tokens",
            nn.Parameter(memory.clone(), requires_grad=True),
        )
        self.register_parameter(
            "fast_memory_tokens",
            nn.Parameter(memory.clone(), requires_grad=True),
        )
        self.memory_policy = "train"
        self.fast_update_attempts = 0
        self.fast_updates_applied = 0
        self.last_fast_gradient_norm = 0.0
        self.last_fast_update_norm = 0.0
        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape: Any) -> torch.Tensor:
        if self.memory_policy == "initial":
            memory = self.initial_memory_tokens
        elif self.memory_policy == "fast":
            memory = self.fast_memory_tokens
        elif self.memory_policy == "train":
            memory = (
                self.fast_memory_tokens
                + self.initial_memory_tokens
                - self.initial_memory_tokens.detach()
            )
        else:
            raise ValueError(f"Unknown FastMem memory policy: {self.memory_policy}")
        return memory.repeat(input_shape[0], 1, 1)

    @torch.no_grad()
    def reset_fast_memory(self) -> None:
        self.fast_memory_tokens.copy_(self.initial_memory_tokens.detach())
        self.fast_memory_tokens.grad = None

    @torch.no_grad()
    def apply_fast_update(
        self,
        *,
        fast_lr: float,
        grad_scale: float,
        clip_norm: float,
    ) -> dict[str, float | int]:
        self.fast_update_attempts += 1
        gradient = self.fast_memory_tokens.grad
        if gradient is None:
            self.last_fast_gradient_norm = 0.0
            self.last_fast_update_norm = 0.0
            return {
                "attempted": 1,
                "applied": 0,
                "gradient_norm": 0.0,
                "update_norm": 0.0,
            }
        update_gradient = gradient.detach() * float(grad_scale)
        gradient_norm = torch.linalg.vector_norm(update_gradient.float())
        self.last_fast_gradient_norm = float(gradient_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Fast-memory gradient norm is non-finite")
        if clip_norm > 0 and float(gradient_norm) > clip_norm:
            update_gradient = update_gradient * (
                float(clip_norm) / (gradient_norm + 1e-12)
            )
        update = -float(fast_lr) * update_gradient
        if fast_lr != 0.0:
            self.fast_memory_tokens.add_(update)
            self.fast_updates_applied += 1
        self.last_fast_update_norm = float(torch.linalg.vector_norm(update.float()))
        self.fast_memory_tokens.grad = None
        return {
            "attempted": 1,
            "applied": int(fast_lr != 0.0),
            "gradient_norm": self.last_fast_gradient_norm,
            "update_norm": self.last_fast_update_norm,
        }

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "fastmem/update_attempts": int(self.fast_update_attempts),
            "fastmem/updates_applied": int(self.fast_updates_applied),
            "fastmem/gradient_norm": float(self.last_fast_gradient_norm),
            "fastmem/update_norm": float(self.last_fast_update_norm),
            "fastmem/fast_memory_norm": float(
                self.fast_memory_tokens.detach().float().norm()
            ),
            "fastmem/initializer_norm": float(
                self.initial_memory_tokens.detach().float().norm()
            ),
        }


@dataclass
class ModelBundle:
    model: nn.Module
    base_model: nn.Module
    fast_cell: FastMemoryCell | None
    variant: str

    def slow_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        fast_ids = (
            {id(self.fast_cell.fast_memory_tokens)}
            if self.fast_cell is not None
            else set()
        )
        return [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and parameter.numel() > 0
            and id(parameter) not in fast_ids
        ]

    def slow_parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.slow_named_parameters()]

    def reset_fast_memory(self) -> None:
        if self.fast_cell is not None:
            self.fast_cell.reset_fast_memory()

    @contextlib.contextmanager
    def evaluation_memory(self) -> Iterator[None]:
        if self.fast_cell is None:
            yield
            return
        previous_policy = self.fast_cell.memory_policy
        previous_fast = self.fast_cell.fast_memory_tokens.detach().clone()
        previous_grad = (
            None
            if self.fast_cell.fast_memory_tokens.grad is None
            else self.fast_cell.fast_memory_tokens.grad.detach().clone()
        )
        try:
            self.fast_cell.reset_fast_memory()
            self.fast_cell.memory_policy = "initial"
            yield
        finally:
            with torch.no_grad():
                self.fast_cell.fast_memory_tokens.copy_(previous_fast)
            self.fast_cell.fast_memory_tokens.grad = previous_grad
            self.fast_cell.memory_policy = previous_policy


def build_model(
    config: ExperimentConfig,
    *,
    tiny_config: Any | None = None,
) -> ModelBundle:
    from transformers import AutoModelForCausalLM, GPT2LMHeadModel

    if tiny_config is None:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.backbone,
            revision=config.backbone_revision,
        )
    else:
        base_model = GPT2LMHeadModel(tiny_config)

    if config.model == "gpt2":
        cell: MemoryCell = MemoryCell(base_model, num_mem_tokens=0)
        wrapper = RecurrentWrapper(
            cell,
            segment_size=1024,
            max_n_segments=1,
            k2=-1,
            segment_alignment="left",
        )
        fast_cell = None
    elif config.model == "base_rmt":
        cell = MemoryCell(base_model, num_mem_tokens=config.n_mem)
        wrapper = RecurrentWrapper(
            cell,
            segment_size=config.segment_size,
            max_n_segments=config.max_n_segments,
            k2=-1,
            segment_alignment="left",
        )
        fast_cell = None
    elif config.model in {"fastmem0", "fastmem"}:
        fast_cell = FastMemoryCell(base_model, num_mem_tokens=config.n_mem)
        wrapper = RecurrentWrapper(
            fast_cell,
            segment_size=config.segment_size,
            max_n_segments=config.max_n_segments,
            k2=-1,
            segment_alignment="left",
        )
        cell = fast_cell
    else:
        raise ValueError(f"Unsupported model variant: {config.model}")
    return ModelBundle(
        model=wrapper,
        base_model=base_model,
        fast_cell=fast_cell,
        variant=config.model,
    )


def fast_lr_for_config(config: ExperimentConfig) -> float:
    if config.model == "fastmem0":
        return 0.0
    if config.model == "fastmem":
        return float(config.fast_lr)
    return 0.0


def autocast_context(config: ExperimentConfig, device: torch.device):
    if config.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
