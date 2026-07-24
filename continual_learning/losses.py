from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MaskedCausalLoss:
    loss_sum: torch.Tensor
    supervised_tokens: int

    @property
    def mean(self) -> torch.Tensor:
        return self.loss_sum / self.supervised_tokens


def shifted_supervised_token_count(labels_mask: torch.Tensor) -> int:
    if labels_mask.ndim < 2:
        raise ValueError("labels_mask must have batch and sequence dimensions")
    if labels_mask.shape[-1] < 2:
        raise ValueError("labels_mask sequence must contain at least two tokens")
    return int(labels_mask[..., :-1].bool().sum().item())


def masked_causal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor,
) -> MaskedCausalLoss:
    """Return the exact summed loss used by the released RMT wrapper.

    The wrapper applies ``labels_mask[..., :-1]`` to next-token predictions.
    Returning a sum and an explicit denominator lets gradient accumulation
    reproduce one physical full-batch token mean.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.shape != labels_mask.shape:
        raise ValueError("labels and labels_mask shapes differ")
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("logits and labels batch/sequence shapes differ")
    if logits.shape[1] < 2:
        raise ValueError("sequence must contain at least two tokens")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = labels_mask[..., :-1].contiguous().bool()
    supervised_tokens = int(shift_mask.sum().item())
    if supervised_tokens <= 0:
        raise ValueError("batch contains no supervised next-token predictions")

    loss_sum = F.cross_entropy(
        shift_logits[shift_mask],
        shift_labels[shift_mask],
        reduction="sum",
    )
    return MaskedCausalLoss(
        loss_sum=loss_sum,
        supervised_tokens=supervised_tokens,
    )
