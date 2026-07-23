"""Unified six-task BABILong continual-learning experiment."""

from .config import (
    CANONICAL_TASKS,
    PROTOCOL_VERSION,
    ExperimentConfig,
    resolve_task_order,
)

__all__ = [
    "CANONICAL_TASKS",
    "PROTOCOL_VERSION",
    "ExperimentConfig",
    "resolve_task_order",
]
