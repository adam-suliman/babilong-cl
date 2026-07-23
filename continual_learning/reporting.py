from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import CANONICAL_TASKS
from .metrics import continual_metrics, matrix_long_rows
from .util import atomic_write_json, atomic_write_text, write_csv


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def matrix_markdown(
    matrix: Sequence[Sequence[float | None]],
    rows: Sequence[str],
) -> list[str]:
    lines = [
        "| Checkpoint | " + " | ".join(CANONICAL_TASKS) + " |",
        "|---|" + "|".join(["---:"] * len(CANONICAL_TASKS)) + "|",
    ]
    for name, values in zip(rows, matrix):
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(percent(None if value is None else float(value)) for value in values)
            + " |"
        )
    return lines


def write_run_artifacts(raw: Mapping[str, Any], run_dir: str | Path) -> None:
    destination = Path(run_dir)
    order = tuple(raw["task_order"])
    compare = raw["compare_answers_matrix"]
    exact = raw["exact_match_matrix"]
    references = raw.get("references")
    reference_compare = (
        None
        if references is None
        else {
            task: float(references[task]["compare_answers"])
            for task in CANONICAL_TASKS
        }
    )
    compare_metrics = continual_metrics(
        compare,
        order,
        single_task_references=reference_compare,
    )
    exact_metrics = continual_metrics(exact, order)
    updated = dict(raw)
    updated["metrics"] = {
        "compare_answers": compare_metrics,
        "exact_match": exact_metrics,
    }
    atomic_write_json(destination / "raw.json", updated)

    lines = [
        "# Unified BABILong Six-Task CL",
        "",
        f"- Condition: `{raw['config']['architecture']}/{raw['config']['method']}`",
        f"- Task order: `{' -> '.join(order)}`",
        f"- Replicate seed: `{raw['config']['replicate_seed']}`",
        f"- Order seed: `{raw['config']['order_seed']}`",
        f"- Data seed: `{raw['config']['data_seed']}`",
        f"- Protocol hash: `{raw['config']['protocol_hash']}`",
        f"- Data manifest: `{raw['data_manifest_sha256']}`",
        f"- Status: `{raw['status']}`",
        "",
        "## BABILong compare_answers",
        "",
        *matrix_markdown(compare, raw["rows"]),
        "",
        "## Strict exact match",
        "",
        *matrix_markdown(exact, raw["rows"]),
        "",
        "## Summary",
        "",
    ]
    summary_keys = [
        ("learning_accuracy", "Learning accuracy"),
        ("final_all_task_accuracy", "Final all-task accuracy"),
        ("final_old_task_accuracy", "Final old-task accuracy"),
        ("forgetting_from_learning", "Forgetting from learning"),
        ("best_to_final_forgetting", "Best-to-final forgetting"),
        ("bwt", "BWT"),
        ("forward_transfer", "Forward transfer"),
        ("intransigence", "Intransigence"),
        ("plasticity_ratio", "Reference-normalized plasticity"),
    ]
    for key, label in summary_keys:
        if key in compare_metrics:
            value = compare_metrics[key]
            rendered = (
                f"{float(value):.3f}"
                if key == "plasticity_ratio"
                else percent(float(value))
            )
            lines.append(f"- {label}: {rendered}")
    lines.extend(
        [
            f"- Slow optimizer steps: `{raw['cumulative_slow_steps']}`",
            f"- Fast-update attempts: `{raw['fast_update_attempts']}`",
            f"- Fast updates applied: `{raw['fast_updates_applied']}`",
            "",
            "The primary quality metric is BABILong `compare_answers` generation accuracy. "
            "JSON is the source of record; this report is a compact rendering.",
            "",
        ]
    )
    atomic_write_text(destination / "summary.md", "\n".join(lines))

    matrix_rows = matrix_long_rows(compare, order, "compare_answers")
    matrix_rows += matrix_long_rows(exact, order, "exact_match")
    write_csv(destination / "tables" / "accuracy_matrix_long.csv", matrix_rows)
    write_csv(
        destination / "tables" / "stage_metrics.csv",
        compare_metrics.get("stagewise", []),
    )
    if references is not None:
        write_csv(
            destination / "tables" / "references.csv",
            [
                {
                    "task": task,
                    **references[task],
                    "learning_accuracy": compare_metrics[
                        "learning_accuracy_by_task"
                    ][task],
                    "intransigence": compare_metrics["intransigence_by_task"][task],
                    "plasticity_ratio": compare_metrics[
                        "plasticity_ratio_by_task"
                    ][task],
                }
                for task in CANONICAL_TASKS
            ],
        )
    _plot_run(updated, destination / "plots")


def _plot_run(raw: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = raw["compare_answers_matrix"]
    complete = [row for row in matrix if all(value is not None for value in row)]
    if not complete:
        return
    x = list(range(len(complete)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for column, task in enumerate(CANONICAL_TASKS):
        axes[0].plot(x, [row[column] for row in complete], marker="o", label=task)
    axes[0].set_xlabel("Stage boundary")
    axes[0].set_ylabel("compare_answers accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.25)

    stage_metrics = raw["metrics"]["compare_answers"].get("stagewise", [])
    keys = [
        ("mean_seen_accuracy", "Mean seen"),
        ("current_task_accuracy", "Current task"),
        ("forgetting_from_learning", "Forgetting"),
        ("bwt", "BWT"),
    ]
    for key, label in keys:
        values = [
            float(item[key]) if item.get(key) is not None else float("nan")
            for item in stage_metrics
        ]
        axes[1].plot(range(len(values)), values, marker="o", label=label)
    axes[1].set_xlabel("Stage boundary")
    axes[1].set_ylabel("Metric value")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "stage_metrics.png", dpi=180)
    fig.savefig(output_dir / "stage_metrics.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(complete, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(CANONICAL_TASKS)), CANONICAL_TASKS)
    ax.set_yticks(
        range(len(complete)),
        raw["rows"][: len(complete)],
    )
    ax.set_title("BABILong compare_answers")
    fig.colorbar(image, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_heatmap.png", dpi=180)
    fig.savefig(output_dir / "accuracy_heatmap.pdf")
    plt.close(fig)

    references = raw.get("references")
    if references is not None:
        metrics = raw["metrics"]["compare_answers"]
        learning = metrics["learning_accuracy_by_task"]
        reference = metrics["single_task_reference_by_task"]
        ratios = metrics["plasticity_ratio_by_task"]
        order = list(raw["task_order"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        positions = list(range(len(order)))
        axes[0].plot(
            positions,
            [learning[task] for task in order],
            marker="o",
            label="CL learning",
        )
        axes[0].plot(
            positions,
            [reference[task] for task in order],
            marker="s",
            label="Single-task reference",
        )
        axes[0].set_xticks(positions, order, rotation=30)
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("compare_answers accuracy")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(
            positions,
            [
                float("nan") if ratios[task] is None else ratios[task]
                for task in order
            ],
            marker="o",
        )
        axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
        axes[1].set_xticks(positions, order, rotation=30)
        axes[1].set_ylabel("Reference-normalized plasticity")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "plasticity.png", dpi=180)
        fig.savefig(output_dir / "plasticity.pdf")
        plt.close(fig)


def strict_fastmem_claim(
    raws: Sequence[Mapping[str, Any]],
    *,
    metric: str = "final_all_task_accuracy",
) -> dict[str, Any]:
    by_model = {raw["config"]["model"]: raw for raw in raws}
    required = {"base_rmt", "fastmem0", "fastmem"}
    if not required.issubset(by_model):
        return {"allowed": False, "reason": "matched method-control triad is incomplete"}
    fields = [
        "backbone",
        "backbone_revision",
        "cl_method",
        "replicate_seed",
        "order_seed",
        "data_seed",
        "task_order",
        "precision",
        "deterministic",
        "slow_steps_per_task",
        "warmup_steps",
        "learning_rates",
        "weight_decay",
        "clip_grad_norm",
        "microbatch_size",
        "slow_batch_size",
        "fast_batch_size",
        "fast_clip_norm",
        "n_mem",
        "segment_size",
        "max_n_segments",
    ]
    base_config = by_model["base_rmt"]["config"]
    mismatches = []
    for model in ("fastmem0", "fastmem"):
        config = by_model[model]["config"]
        for field in fields:
            if config.get(field) != base_config.get(field):
                mismatches.append(f"{model}:{field}")
        if by_model[model]["data_manifest_sha256"] != by_model["base_rmt"][
            "data_manifest_sha256"
        ]:
            mismatches.append(f"{model}:data_manifest_sha256")
        if by_model[model].get("cumulative_slow_steps") != by_model[
            "base_rmt"
        ].get("cumulative_slow_steps"):
            mismatches.append(f"{model}:cumulative_slow_steps")
        base_train = [
            {
                key: stage.get(key)
                for key in (
                    "task",
                    "task_sampler_seed",
                    "slow_steps",
                    "examples_seen",
                    "dataset_rows",
                    "effective_rows_per_epoch",
                    "dropped_tail_rows_per_epoch",
                )
            }
            for stage in by_model["base_rmt"].get("train_tasks", [])
        ]
        model_train = [
            {
                key: stage.get(key)
                for key in (
                    "task",
                    "task_sampler_seed",
                    "slow_steps",
                    "examples_seen",
                    "dataset_rows",
                    "effective_rows_per_epoch",
                    "dropped_tail_rows_per_epoch",
                )
            }
            for stage in by_model[model].get("train_tasks", [])
        ]
        if model_train != base_train:
            mismatches.append(f"{model}:train_budget")
    for model in required:
        if by_model[model].get("status") != "complete":
            mismatches.append(f"{model}:status")
    expected_attempts = (
        int(by_model["fastmem"]["cumulative_slow_steps"])
        * int(base_config["slow_batch_size"])
        // int(base_config["fast_batch_size"])
    )
    if by_model["fastmem"].get("fast_update_attempts") != expected_attempts:
        mismatches.append("fastmem:fast_update_attempts")
    if by_model["fastmem0"].get("fast_update_attempts") != expected_attempts:
        mismatches.append("fastmem0:fast_update_attempts")
    if by_model["fastmem"].get("fast_updates_applied") != expected_attempts:
        mismatches.append("fastmem:fast_updates_applied")
    if by_model["fastmem0"].get("fast_updates_applied") != 0:
        mismatches.append("fastmem0:fast_updates_applied")
    if mismatches:
        return {
            "allowed": False,
            "reason": "conditions are not matched",
            "mismatches": mismatches,
        }
    scores = {
        model: float(raw["metrics"]["compare_answers"][metric])
        for model, raw in by_model.items()
        if model in required
    }
    return {
        "allowed": scores["fastmem"] > scores["base_rmt"]
        and scores["fastmem"] > scores["fastmem0"],
        "metric": metric,
        "scores": scores,
        "reason": (
            "nonzero FastMem beats both matched controls"
            if scores["fastmem"] > scores["base_rmt"]
            and scores["fastmem"] > scores["fastmem0"]
            else "nonzero FastMem does not beat both matched controls"
        ),
    }


def aggregate_results(results_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path(results_root)
    runs = []
    for path in root.glob("runs/*/*/replicate-*/order-*/raw.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("status") == "complete":
            runs.append({"path": str(path), "raw": raw})
    rows = []
    for item in runs:
        raw = item["raw"]
        metrics = raw["metrics"]["compare_answers"]
        rows.append(
            {
                "path": item["path"],
                "model": raw["config"]["model"],
                "architecture": raw["config"]["architecture"],
                "method": raw["config"]["method"],
                "replicate_seed": raw["config"]["replicate_seed"],
                "order_seed": raw["config"]["order_seed"],
                "task_order": "->".join(raw["task_order"]),
                **{
                    key: metrics.get(key)
                    for key in (
                        "learning_accuracy",
                        "final_all_task_accuracy",
                        "final_old_task_accuracy",
                        "forgetting_from_learning",
                        "bwt",
                        "forward_transfer",
                        "intransigence",
                        "plasticity_ratio",
                    )
                },
            }
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "run_summary.csv", rows)
    metric_keys = (
        "learning_accuracy",
        "final_all_task_accuracy",
        "final_old_task_accuracy",
        "forgetting_from_learning",
        "bwt",
        "forward_transfer",
        "intransigence",
        "plasticity_ratio",
    )
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["architecture"]),
                str(row["method"]),
                int(row["order_seed"]),
            )
        ].append(row)
    group_rows = []
    for (architecture, method, order_seed), items in sorted(grouped.items()):
        group_row: dict[str, Any] = {
            "architecture": architecture,
            "method": method,
            "order_seed": order_seed,
            "task_order": items[0]["task_order"],
            "n": len(items),
            "replicate_seeds": ",".join(
                str(item["replicate_seed"]) for item in items
            ),
        }
        for key in metric_keys:
            values = [
                float(item[key])
                for item in items
                if item.get(key) is not None
            ]
            group_row[f"{key}_mean"] = (
                statistics.fmean(values) if values else None
            )
            group_row[f"{key}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            ) if values else None
        group_rows.append(group_row)
    write_csv(destination / "group_summary.csv", group_rows)

    matched: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in runs:
        raw = item["raw"]
        if raw["config"]["model"] in {"base_rmt", "fastmem0", "fastmem"}:
            matched[
                (
                    int(raw["config"]["replicate_seed"]),
                    int(raw["config"]["order_seed"]),
                    str(raw["data_manifest_sha256"]),
                )
            ].append(raw)
    claim_guards = [
        {
            "replicate_seed": key[0],
            "order_seed": key[1],
            "data_manifest_sha256": key[2],
            **strict_fastmem_claim(value),
        }
        for key, value in sorted(matched.items())
    ]
    payload = {
        "format": "babilong-qa6-cl-aggregate-v1",
        "results_root": str(root),
        "completed_runs": len(runs),
        "runs": rows,
        "groups": group_rows,
        "fastmem_claim_guards": claim_guards,
    }
    atomic_write_json(destination / "aggregate.json", payload)
    lines = [
        "# BABILong QA6 Aggregate",
        "",
        f"Completed runs: `{len(runs)}`",
        "",
        "| Architecture | Method | Order seed | n | Learning | Final | Forgetting | BWT | Plasticity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        def render(key: str) -> str:
            value = row.get(f"{key}_mean")
            deviation = row.get(f"{key}_std")
            if value is None:
                return "n/a"
            return f"{float(value):.3f} +/- {float(deviation):.3f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["architecture"]),
                    str(row["method"]),
                    str(row["order_seed"]),
                    str(row["n"]),
                    render("learning_accuracy"),
                    render("final_all_task_accuracy"),
                    render("forgetting_from_learning"),
                    render("bwt"),
                    render("plasticity_ratio"),
                ]
            )
            + " |"
        )
    atomic_write_text(destination / "summary.md", "\n".join(lines) + "\n")
    _plot_aggregate(runs, destination / "plots")
    return payload


def _plot_aggregate(
    runs: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    if not runs:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in runs:
        raw = item["raw"]
        condition = f"{raw['config']['architecture']}/{raw['config']['method']}"
        by_condition[condition].append(raw)
    stage_keys = (
        ("mean_seen_accuracy", "Mean seen accuracy"),
        ("current_task_accuracy", "Current-task accuracy"),
        ("forgetting_from_learning", "Forgetting"),
        ("bwt", "BWT"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (metric, title) in zip(axes.flat, stage_keys):
        for condition, condition_runs in sorted(by_condition.items()):
            x = []
            means = []
            stds = []
            for stage_index in range(1, len(CANONICAL_TASKS) + 1):
                values = []
                for raw in condition_runs:
                    stagewise = raw["metrics"]["compare_answers"].get(
                        "stagewise", []
                    )
                    if stage_index >= len(stagewise):
                        continue
                    value = stagewise[stage_index].get(metric)
                    if value is not None:
                        values.append(float(value))
                if not values:
                    continue
                x.append(stage_index)
                means.append(statistics.fmean(values))
                stds.append(
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            if not x:
                continue
            axis.plot(x, means, marker="o", label=condition)
            axis.fill_between(
                x,
                [mean - std for mean, std in zip(means, stds)],
                [mean + std for mean, std in zip(means, stds)],
                alpha=0.15,
            )
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Completed tasks")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "stage_metrics.png", dpi=180)
    fig.savefig(output_dir / "stage_metrics.pdf")
    plt.close(fig)
