from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import CANONICAL_TASKS, ExperimentConfig


class ExperimentMonitor:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        enabled: bool,
        purge_step: int | None = None,
    ) -> None:
        self.config = config
        self.enabled = enabled
        self.writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=str(config.tensorboard_dir),
                purge_step=purge_step,
            )
            self.writer.add_text(
                "config/resolved",
                "```json\n" + json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n```",
                0,
            )

    def scalar(self, tag: str, value: float | int | None, step: int) -> None:
        if self.writer is not None and value is not None:
            self.writer.add_scalar(tag, float(value), int(step))

    def training(
        self,
        *,
        task: str,
        loss: float,
        lr: float,
        grad_norm: float,
        examples_seen: int,
        supervised_tokens_seen: int,
        cumulative_step: int,
        fast: Mapping[str, float | int] | None,
        si: Mapping[str, Any] | None,
    ) -> None:
        self.scalar("train/loss", loss, cumulative_step)
        self.scalar("train/lr", lr, cumulative_step)
        self.scalar("train/slow_gradient_norm", grad_norm, cumulative_step)
        self.scalar("train/examples_seen", examples_seen, cumulative_step)
        self.scalar(
            "train/supervised_tokens_seen",
            supervised_tokens_seen,
            cumulative_step,
        )
        self.scalar(f"train_by_task/{task}/loss", loss, cumulative_step)
        if fast:
            for key, value in fast.items():
                self.scalar(key, value, cumulative_step)
        if si:
            for key, value in si.items():
                if isinstance(value, (int, float)):
                    self.scalar(key, value, cumulative_step)

    def stage(
        self,
        *,
        stage_index: int,
        cumulative_step: int,
        compare_row: list[float],
        exact_row: list[float],
        stage_metrics: Mapping[str, Any],
    ) -> None:
        for task, value in zip(CANONICAL_TASKS, compare_row):
            self.scalar(f"quality/compare_answers/{task}", value, cumulative_step)
        for task, value in zip(CANONICAL_TASKS, exact_row):
            self.scalar(f"quality/exact_match/{task}", value, cumulative_step)
        names = {
            "mean_all_task_accuracy": "quality/mean_all_tasks",
            "mean_seen_accuracy": "quality/mean_seen",
            "mean_old_task_accuracy": "retention/mean_old_task_accuracy",
            "current_task_accuracy": "plasticity/current_task_accuracy",
            "mean_learning_accuracy_to_date": "plasticity/mean_learning_accuracy",
            "current_task_forward_transfer": "plasticity/current_task_forward_transfer",
            "forgetting_from_learning": "retention/forgetting_from_learning",
            "best_to_current_forgetting": "retention/best_to_current_forgetting",
            "bwt": "retention/bwt",
        }
        for source, target in names.items():
            value = stage_metrics.get(source)
            self.scalar(target, value, cumulative_step)
            family, metric = target.split("/", 1)
            self.scalar(f"{family}_by_stage/{metric}", value, stage_index)
        self.scalar("stage/index", stage_index, cumulative_step)

    def references(
        self,
        *,
        task_order: tuple[str, ...],
        metrics: Mapping[str, Any],
        steps_per_task: int,
    ) -> None:
        ratios = metrics.get("plasticity_ratio_by_task", {})
        gaps = metrics.get("intransigence_by_task", {})
        references = metrics.get("single_task_reference_by_task", {})
        for stage_index, task in enumerate(task_order, start=1):
            step = stage_index * steps_per_task
            self.scalar("plasticity/single_task_reference", references.get(task), step)
            self.scalar("plasticity/intransigence", gaps.get(task), step)
            self.scalar("plasticity/reference_ratio", ratios.get(task), step)
            self.scalar(
                "plasticity_by_stage/single_task_reference",
                references.get(task),
                stage_index,
            )
            self.scalar(
                "plasticity_by_stage/intransigence",
                gaps.get(task),
                stage_index,
            )
            self.scalar(
                "plasticity_by_stage/reference_ratio",
                ratios.get(task),
                stage_index,
            )

    def matrix_text(
        self,
        *,
        compare_matrix: list[list[float | None]],
        exact_matrix: list[list[float | None]],
        step: int,
    ) -> None:
        if self.writer is None:
            return
        self.writer.add_text(
            "matrices/compare_answers",
            "```json\n" + json.dumps(compare_matrix, indent=2) + "\n```",
            step,
        )
        self.writer.add_text(
            "matrices/exact_match",
            "```json\n" + json.dumps(exact_matrix, indent=2) + "\n```",
            step,
        )

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
