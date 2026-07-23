from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from babilong.metrics import TASK_LABELS, compare_answers

from .config import CANONICAL_TASKS


def strict_exact_match(target: str, prediction: str) -> bool:
    return target.strip().lower() == prediction.strip().lower()


def score_prediction(
    *,
    task: str,
    target: str,
    prediction: str,
    question: str,
) -> dict[str, bool]:
    return {
        "compare_answers": bool(
            compare_answers(target, prediction, question, TASK_LABELS[task])
        ),
        "exact_match": strict_exact_match(target, prediction),
    }


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def empty_matrix() -> list[list[float | None]]:
    return [[None for _ in CANONICAL_TASKS] for _ in range(len(CANONICAL_TASKS) + 1)]


def validate_matrix(matrix: Sequence[Sequence[float | None]]) -> None:
    expected_rows = len(CANONICAL_TASKS) + 1
    if len(matrix) != expected_rows:
        raise ValueError(f"Expected {expected_rows} matrix rows, got {len(matrix)}")
    for row in matrix:
        if len(row) != len(CANONICAL_TASKS):
            raise ValueError("Matrix column count does not match canonical tasks")
        for value in row:
            if value is not None and (
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"Invalid accuracy value: {value}")


def complete_rows(matrix: Sequence[Sequence[float | None]]) -> int:
    validate_matrix(matrix)
    count = 0
    for row in matrix:
        if any(value is None for value in row):
            break
        count += 1
    return count


def continual_metrics(
    matrix: Sequence[Sequence[float | None]],
    task_order: Sequence[str],
    *,
    single_task_references: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    validate_matrix(matrix)
    order = tuple(task_order)
    if len(order) != len(CANONICAL_TASKS) or set(order) != set(CANONICAL_TASKS):
        raise ValueError("task_order must contain every canonical task exactly once")
    n_rows = complete_rows(matrix)
    if n_rows == 0:
        return {"complete_stage_rows": 0, "stagewise": []}
    columns = {task: index for index, task in enumerate(CANONICAL_TASKS)}
    learned_row = {task: order.index(task) + 1 for task in order}
    stagewise: list[dict[str, Any]] = []

    for row_index in range(n_rows):
        row = matrix[row_index]
        seen = list(order[:row_index])
        old = seen[:-1]
        stage: dict[str, Any] = {
            "stage_index": row_index,
            "after_task": None if row_index == 0 else order[row_index - 1],
            "mean_all_task_accuracy": mean(
                [float(row[columns[task]]) for task in CANONICAL_TASKS]
            ),
            "mean_seen_accuracy": (
                mean([float(row[columns[task]]) for task in seen]) if seen else None
            ),
            "mean_old_task_accuracy": (
                mean([float(row[columns[task]]) for task in old]) if old else None
            ),
            "current_task_accuracy": (
                float(row[columns[seen[-1]]]) if seen else None
            ),
            "mean_learning_accuracy_to_date": (
                mean(
                    [
                        float(matrix[learned_row[task]][columns[task]])
                        for task in seen
                    ]
                )
                if seen
                else None
            ),
            "current_task_forward_transfer": (
                float(matrix[row_index - 1][columns[seen[-1]]])
                - float(matrix[0][columns[seen[-1]]])
                if len(seen) > 1
                else None
            ),
        }
        forgetting_values: list[float] = []
        bwt_values: list[float] = []
        best_forgetting_values: list[float] = []
        for task in old:
            task_column = columns[task]
            learning = float(matrix[learned_row[task]][task_column])
            current = float(row[task_column])
            forgetting_values.append(max(0.0, learning - current))
            bwt_values.append(current - learning)
            best = max(
                float(matrix[prior][task_column])
                for prior in range(learned_row[task], row_index + 1)
            )
            best_forgetting_values.append(max(0.0, best - current))
        stage["forgetting_from_learning"] = (
            mean(forgetting_values) if forgetting_values else None
        )
        stage["bwt"] = mean(bwt_values) if bwt_values else None
        stage["best_to_current_forgetting"] = (
            mean(best_forgetting_values) if best_forgetting_values else None
        )
        stagewise.append(stage)

    result: dict[str, Any] = {
        "complete_stage_rows": n_rows,
        "stagewise": stagewise,
    }
    if n_rows < len(CANONICAL_TASKS) + 1:
        return result

    final = matrix[-1]
    learning_by_task = {
        task: float(matrix[learned_row[task]][columns[task]]) for task in order
    }
    old_tasks = list(order[:-1])
    final_by_task = {task: float(final[columns[task]]) for task in order}
    bwt_by_task = {
        task: final_by_task[task] - learning_by_task[task] for task in old_tasks
    }
    forgetting_by_task = {
        task: max(0.0, -bwt_by_task[task]) for task in old_tasks
    }
    best_forgetting_by_task = {}
    for task in old_tasks:
        column = columns[task]
        best = max(
            float(matrix[row][column])
            for row in range(learned_row[task], len(matrix))
        )
        best_forgetting_by_task[task] = max(0.0, best - final_by_task[task])

    fwt_by_task = {}
    for task in order[1:]:
        before_row = learned_row[task] - 1
        fwt_by_task[task] = (
            float(matrix[before_row][columns[task]])
            - float(matrix[0][columns[task]])
        )

    result.update(
        {
            "learning_accuracy_by_task": learning_by_task,
            "learning_accuracy": mean(list(learning_by_task.values())),
            "final_accuracy_by_task": final_by_task,
            "final_all_task_accuracy": mean(list(final_by_task.values())),
            "final_old_task_accuracy": mean(
                [final_by_task[task] for task in old_tasks]
            ),
            "forgetting_from_learning_by_task": forgetting_by_task,
            "forgetting_from_learning": mean(list(forgetting_by_task.values())),
            "best_to_final_forgetting_by_task": best_forgetting_by_task,
            "best_to_final_forgetting": mean(
                list(best_forgetting_by_task.values())
            ),
            "bwt_by_task": bwt_by_task,
            "bwt": mean(list(bwt_by_task.values())),
            "forward_transfer_by_task": fwt_by_task,
            "forward_transfer": mean(list(fwt_by_task.values())),
        }
    )

    if single_task_references is not None:
        if set(single_task_references) != set(CANONICAL_TASKS):
            raise ValueError("Single-task references must cover every task")
        reference = {
            task: float(single_task_references[task]) for task in CANONICAL_TASKS
        }
        intransigence = {
            task: reference[task] - learning_by_task[task] for task in CANONICAL_TASKS
        }
        plasticity_ratio = {
            task: (
                None
                if reference[task] == 0.0
                else learning_by_task[task] / reference[task]
            )
            for task in CANONICAL_TASKS
        }
        result.update(
            {
                "single_task_reference_by_task": reference,
                "intransigence_by_task": intransigence,
                "intransigence": mean(list(intransigence.values())),
                "plasticity_ratio_by_task": plasticity_ratio,
                "plasticity_ratio": mean(
                    [
                        float(value)
                        for value in plasticity_ratio.values()
                        if value is not None
                    ]
                ),
            }
        )
    return result


def matrix_long_rows(
    matrix: Sequence[Sequence[float | None]],
    task_order: Sequence[str],
    metric: str,
) -> list[dict[str, Any]]:
    validate_matrix(matrix)
    rows = []
    for stage_index, row in enumerate(matrix):
        after_task = None if stage_index == 0 else task_order[stage_index - 1]
        for task, value in zip(CANONICAL_TASKS, row):
            rows.append(
                {
                    "metric": metric,
                    "stage_index": stage_index,
                    "after_task": after_task,
                    "eval_task": task,
                    "accuracy": value,
                }
            )
    return rows
