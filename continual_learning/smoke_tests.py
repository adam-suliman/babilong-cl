from __future__ import annotations

import copy
import gc
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from transformers import GPT2Config

from .cli import verify_provenance
from .config import (
    CANONICAL_TASKS,
    ExperimentConfig,
    resolve_task_order,
    task_sampler_seeds,
)
from .data import OfficialCollator
from .launcher import _take_launchable_assignment
from .losses import (
    masked_causal_cross_entropy,
    shifted_supervised_token_count,
)
from .metrics import continual_metrics, decode_generated_answer
from .models import build_model
from .reporting import (
    aggregate_results,
    strict_fastmem_claim,
    write_run_artifacts,
)
from .strategies import SynapticIntelligence, SynapticIntelligenceConfig
from .trainer import EpochBatchCursor, UnifiedTrainer, _task_scheduler
from .util import atomic_write_json, json_sha256, seed_everything


class TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    eos_token = "<eos>"
    model_max_length = 1024

    def __init__(self) -> None:
        self._tokens = {
            "GEN": 1,
            "mary": 2,
            "went": 3,
            "to": 4,
            "the": 5,
            "kitchen.": 6,
            "where": 7,
            "is": 8,
            "mary?": 9,
            "kitchen": 10,
        }

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        stripped = text.strip()
        if stripped == "GEN":
            return [1]
        return [self._tokens.get(token.lower(), 11) for token in stripped.split()]

    def decode(self, tokens: torch.Tensor | list[int], skip_special_tokens: bool = False) -> str:
        ids = [int(value) for value in tokens]
        inverse = {value: key for key, value in self._tokens.items()}
        return " ".join(
            self.eos_token if value == self.eos_token_id else inverse.get(value, "x")
            for value in ids
        )


class VectorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.8, -0.4]))


def _tiny_config(root: Path, model: str, *, cl_method: str = "none") -> ExperimentConfig:
    return ExperimentConfig(
        model=model,
        cl_method=cl_method,
        si_lambda=0.7 if cl_method == "si" else None,
        results_root=str(root / "results"),
        data_dir=str(root / "data"),
        device="cpu",
        require_clean_git=False,
        include_references=False,
        deterministic=True,
        slow_steps_per_task=1,
        legacy_configured_iters=0,
        warmup_steps=0,
        learning_rates={task: 1e-3 for task in CANONICAL_TASKS},
        microbatch_size=1,
        slow_batch_size=4,
        fast_batch_size=2,
        n_mem=2,
        segment_size=32,
        max_n_segments=2,
        checkpoint_interval=1,
        log_interval=1,
        eval_max_new_tokens=2,
    )


def _tiny_data(root: Path) -> dict:
    data_dir = root / "data"
    split_info = {}
    rows = [
        {
            "input": "Mary went to the kitchen. ",
            "question": "Where is Mary? ",
            "target": "kitchen",
        }
        for _ in range(4)
    ]
    for split in ("train", "validation", "eval"):
        split_info[split] = {}
        for task in CANONICAL_TASKS:
            path = data_dir / split / f"{task}.json"
            atomic_write_json(path, rows)
            split_info[split][task] = {
                "path": str(Path(split) / f"{task}.json"),
                "sha256": "tiny",
                "source": "smoke",
                "token_stats": {"fit_1024": 4},
            }
    payload = {
        "format": "babilong-qa6-0k-data-v1",
        "data_seed": 481113,
        **split_info,
    }
    payload["manifest_sha256"] = json_sha256(payload)
    atomic_write_json(data_dir / "manifest.json", payload)
    return payload


def _gpt2_tiny_model_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=32,
        n_positions=128,
        n_ctx=128,
        n_embd=24,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        use_cache=False,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )


def _variable_target_records(count: int = 64) -> list[dict]:
    records = []
    for index in range(count):
        target_tokens = [10] if index % 2 == 0 else [10, 11]
        records.append(
            {
                "input_tokens": [2, 3] + ([4] if index % 3 == 0 else []),
                "question_tokens": [7, 8],
                "target_tokens": target_tokens,
                "target_text": "kitchen" if len(target_tokens) == 1 else "kitchen x",
                "row_index": index,
            }
        )
    return records


def test_seed_contract() -> None:
    order = resolve_task_order(48)
    assert set(order) == set(CANONICAL_TASKS)
    assert order == resolve_task_order(48)
    assert order != resolve_task_order(49)
    seeds = task_sampler_seeds(48)
    assert len(set(seeds.values())) == len(CANONICAL_TASKS)
    assert seeds == task_sampler_seeds(48)


def test_config_and_collation_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="babilong-cl-config-smoke-") as temporary:
        config = _tiny_config(Path(temporary), "base_rmt")
        payload = config.to_dict()
        assert payload["architecture"] == "base-rmt16"
        assert payload["method"] == "none"
        restored = ExperimentConfig.from_mapping(payload)
        assert restored.protocol_hash == config.protocol_hash
        assert restored.run_dir == config.run_dir
        different_microbatch = replace(config, microbatch_size=2)
        assert different_microbatch.protocol_hash != config.protocol_hash
        assert different_microbatch.run_dir != config.run_dir

    tokenizer = TinyTokenizer()
    collated = OfficialCollator(tokenizer)(
        [
            {
                "input_tokens": [2, 3],
                "question_tokens": [7, 8],
                "target_tokens": [10],
                "target_text": "kitchen",
                "row_index": 0,
            }
        ]
    )
    assert collated["input_ids"].tolist() == [[2, 3, 7, 8, 1, 10, 0]]
    assert collated["labels_mask"].tolist() == [
        [False, False, False, False, True, True, False]
    ]


def test_token_normalization_and_microbatch_equivalence() -> None:
    tokenizer = TinyTokenizer()
    collator = OfficialCollator(tokenizer)
    records = _variable_target_records()
    full_batch = collator(records)
    expected_tokens = 32 * 2 + 32 * 3
    assert shifted_supervised_token_count(full_batch["labels_mask"]) == expected_tokens

    tiny = _gpt2_tiny_model_config()
    with tempfile.TemporaryDirectory(
        prefix="babilong-cl-token-normalization-"
    ) as temporary:
        root = Path(temporary)
        for model_key in ("gpt2", "base_rmt"):
            config = replace(
                _tiny_config(root / model_key, model_key),
                slow_batch_size=64,
                fast_batch_size=32,
                microbatch_size=1,
            )
            seed_everything(48)
            initial = build_model(config, tiny_config=tiny)
            initial_state = copy.deepcopy(initial.model.state_dict())

            reference = build_model(config, tiny_config=tiny)
            reference.model.load_state_dict(initial_state)
            reference.model.train()
            reference_output = reference.model(
                input_ids=full_batch["input_ids"],
                labels=full_batch["labels"],
                labels_mask=full_batch["labels_mask"],
                attention_mask=full_batch["attention_mask"],
            )
            reference_output.loss.backward()
            reference_gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in reference.slow_named_parameters()
                if parameter.grad is not None
            }

            logits_only = build_model(config, tiny_config=tiny)
            logits_only.model.load_state_dict(initial_state)
            logits_only.model.train()
            logits_output = logits_only.model(
                input_ids=full_batch["input_ids"],
                attention_mask=full_batch["attention_mask"],
            )
            exact_loss = masked_causal_cross_entropy(
                logits_output.logits,
                full_batch["labels"],
                full_batch["labels_mask"],
            )
            torch.testing.assert_close(
                exact_loss.mean,
                reference_output.loss,
                rtol=1e-6,
                atol=1e-7,
            )

            for microbatch_size in (1, 8, 16, 32):
                accumulated = build_model(config, tiny_config=tiny)
                accumulated.model.load_state_dict(initial_state)
                accumulated.model.train()
                for start in range(0, len(records), microbatch_size):
                    batch = collator(records[start : start + microbatch_size])
                    output = accumulated.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss = masked_causal_cross_entropy(
                        output.logits,
                        batch["labels"],
                        batch["labels_mask"],
                    )
                    (loss.loss_sum / expected_tokens).backward()
                accumulated_gradients = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in accumulated.slow_named_parameters()
                    if parameter.grad is not None
                }
                assert accumulated_gradients.keys() == reference_gradients.keys()
                for name in reference_gradients:
                    torch.testing.assert_close(
                        accumulated_gradients[name],
                        reference_gradients[name],
                        rtol=2e-5,
                        atol=2e-6,
                        msg=lambda message, name=name: (
                            f"{model_key} microbatch={microbatch_size} {name}: "
                            f"{message}"
                        ),
                    )

        for model_key in ("fastmem0", "fastmem"):
            final_states = {}
            summaries = {}
            for microbatch_size in (1, 8, 16, 32):
                config = replace(
                    _tiny_config(
                        root / f"{model_key}-{microbatch_size}",
                        model_key,
                    ),
                    slow_batch_size=64,
                    fast_batch_size=32,
                    microbatch_size=microbatch_size,
                    clip_grad_norm=1e6,
                    fast_clip_norm=1e6,
                )
                trainer = UnifiedTrainer(
                    config=config,
                    data_manifest={"manifest_sha256": "normalization-smoke"},
                    tokenizer=tokenizer,
                    tiny_model_config=tiny,
                    tensorboard=False,
                )
                trainer.scheduler = _task_scheduler(
                    trainer.optimizer,
                    learning_rate=float(config.learning_rates["qa1"]),
                    warmup_steps=0,
                    total_steps=1,
                )
                trainer.bundle.model.train()
                trainer.bundle.reset_fast_memory()
                summary = trainer.train_task(
                    task="qa1",
                    dataset=records,
                    cursor=EpochBatchCursor(
                        dataset_size=len(records),
                        batch_size=64,
                        task_seed=config.sampler_seeds["qa1"],
                    ),
                    start_step=0,
                    target_steps=1,
                )
                final_states[microbatch_size] = copy.deepcopy(
                    trainer.bundle.model.state_dict()
                )
                summaries[microbatch_size] = summary
                trainer.monitor.close()

            reference_state = final_states[32]
            for microbatch_size, state in final_states.items():
                assert state.keys() == reference_state.keys()
                for name in state:
                    torch.testing.assert_close(
                        state[name],
                        reference_state[name],
                        rtol=3e-5,
                        atol=3e-6,
                        msg=lambda message, name=name: (
                            f"{model_key} microbatch={microbatch_size} {name}: "
                            f"{message}"
                        ),
                    )
                assert summaries[microbatch_size]["supervised_tokens_seen"] == (
                    expected_tokens
                )
                assert summaries[microbatch_size]["fast_update_attempts"] == 2
                assert summaries[microbatch_size]["fast_updates_applied"] == (
                    2 if model_key == "fastmem" else 0
                )


def test_metrics() -> None:
    tokenizer = TinyTokenizer()
    assert (
        decode_generated_answer(tokenizer, torch.tensor([10, 0]))
        == "kitchen"
    )
    assert decode_generated_answer(tokenizer, torch.tensor([0])) == ""

    order = resolve_task_order(48)
    columns = {task: index for index, task in enumerate(CANONICAL_TASKS)}
    matrix = [[0.1] * 6 for _ in range(7)]
    for stage, task in enumerate(order, start=1):
        matrix[stage][columns[task]] = 0.9
        for old in order[: stage - 1]:
            matrix[stage][columns[old]] = max(
                0.0, matrix[stage - 1][columns[old]] - 0.1
            )
    references = {task: 1.0 for task in CANONICAL_TASKS}
    result = continual_metrics(matrix, order, single_task_references=references)
    assert result["learning_accuracy"] == 0.9
    assert result["final_all_task_accuracy"] >= 0.0
    assert result["forgetting_from_learning"] >= 0.0
    assert result["bwt"] <= 0.0
    assert result["plasticity_ratio"] == 0.9

    with tempfile.TemporaryDirectory(prefix="babilong-cl-claim-smoke-") as temporary:
        root = Path(temporary)
        train = [
            {
                "task": task,
                "task_sampler_seed": _tiny_config(
                    root, "base_rmt"
                ).sampler_seeds[task],
                "slow_steps": 1,
                "examples_seen": 4,
                "dataset_rows": 4,
                "effective_rows_per_epoch": 4,
                "dropped_tail_rows_per_epoch": 0,
            }
            for task in order
        ]
        raws = []
        for model, score in (
            ("base_rmt", 0.7),
            ("fastmem0", 0.72),
            ("fastmem", 0.75),
        ):
            config = _tiny_config(root, model)
            raws.append(
                {
                    "config": config.to_dict(),
                    "data_manifest_sha256": "matched",
                    "status": "complete",
                    "cumulative_slow_steps": 6,
                    "fast_update_attempts": (
                        12 if model in {"fastmem0", "fastmem"} else 0
                    ),
                    "fast_updates_applied": 12 if model == "fastmem" else 0,
                    "train_tasks": train,
                    "metrics": {
                        "compare_answers": {
                            "final_all_task_accuracy": score
                        }
                    },
                }
            )
        claim = strict_fastmem_claim(raws)
        assert claim["allowed"] is True


def test_protocol_v3_aggregation() -> None:
    with tempfile.TemporaryDirectory(
        prefix="babilong-cl-aggregate-smoke-"
    ) as temporary:
        root = Path(temporary)
        results_root = root / "results"
        for replicate_seed in (48, 49):
            config = replace(
                _tiny_config(root, "gpt2"),
                replicate_seed=replicate_seed,
                results_root=str(results_root),
            )
            stagewise = [
                {
                    "mean_seen_accuracy": (
                        None if stage == 0 else 0.5 + 0.01 * stage
                    ),
                    "current_task_accuracy": (
                        None if stage == 0 else 0.6 + 0.01 * stage
                    ),
                    "forgetting_from_learning": (
                        None if stage < 2 else 0.01 * stage
                    ),
                    "bwt": None if stage < 2 else -0.01 * stage,
                }
                for stage in range(len(CANONICAL_TASKS) + 1)
            ]
            atomic_write_json(
                config.run_dir / "raw.json",
                {
                    "status": "complete",
                    "config": config.to_dict(),
                    "task_order": list(config.resolved_order),
                    "data_manifest_sha256": "aggregate-smoke-data",
                    "metrics": {
                        "compare_answers": {
                            "learning_accuracy": 0.7,
                            "final_all_task_accuracy": 0.6,
                            "final_old_task_accuracy": 0.58,
                            "forgetting_from_learning": 0.1,
                            "bwt": -0.1,
                            "forward_transfer": 0.02,
                            "intransigence": 0.05,
                            "plasticity_ratio": 0.9,
                            "stagewise": stagewise,
                        }
                    },
                },
            )

        output = root / "aggregate"
        payload = aggregate_results(results_root, output)
        assert payload["completed_runs"] == 2
        assert len(payload["groups"]) == 1
        assert payload["groups"][0]["n"] == 2
        assert payload["groups"][0]["replicate_seeds"] == "48,49"
        assert (output / "aggregate.json").is_file()
        assert (output / "run_summary.csv").is_file()
        assert (output / "group_summary.csv").is_file()
        assert (output / "summary.md").is_file()
        assert (output / "plots" / "stage_metrics.png").is_file()


def test_si_gpu_placement() -> None:
    running = {
        1: (
            None,
            ("0", 0),
            {"condition": "gpt2-si", "cl_method": "si"},
            None,
        )
    }
    pending = [
        {"condition": "gpt2-si", "cl_method": "si"},
        {"condition": "base-rmt", "cl_method": "none"},
    ]
    slots = [("0", 1), ("1", 0)]
    assignment = _take_launchable_assignment(pending, slots, running)
    assert assignment is not None
    job, slot = assignment
    assert job["condition"] == "gpt2-si"
    assert slot[0] == "1"

    assert (
        _take_launchable_assignment(
            [{"condition": "gpt2-si", "cl_method": "si"}],
            [("0", 1)],
            running,
        )
        is None
    )


def test_si_equations() -> None:
    model = VectorModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.03,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.0,
    )
    config = SynapticIntelligenceConfig(
        si_lambda=0.7,
        epsilon=0.1,
        decay=1.0,
    )
    strategy = SynapticIntelligence(model, optimizer, config)
    strategy.begin_task("qa1")
    start = model.weight.detach().clone()
    path = torch.zeros_like(start)
    for target in (torch.tensor([0.0, 0.2]), torch.tensor([0.4, -0.1])):
        optimizer.zero_grad(set_to_none=True)
        loss = (model.weight - target).pow(2).sum() + strategy.penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.weight], 0.4)
        gradient = model.weight.grad.detach().clone()
        previous = model.weight.detach().clone()
        strategy.before_optimizer_step()
        optimizer.step()
        strategy.after_optimizer_step()
        path.add_(-gradient * (model.weight.detach() - previous))
    total_delta = model.weight.detach() - start
    expected = torch.clamp(path / (total_delta.pow(2) + 0.1), min=0.0)
    strategy.end_task("qa1")
    consolidated_state = strategy.state_dict()
    consolidated_model = copy.deepcopy(model.state_dict())
    torch.testing.assert_close(
        consolidated_state["importance"]["weight"],
        expected,
        rtol=0,
        atol=1e-7,
    )

    accumulated_gradients = {}
    for microbatch_count in (1, 8, 32):
        accumulated_model = VectorModel()
        accumulated_model.load_state_dict(consolidated_model)
        accumulated_optimizer = torch.optim.AdamW(
            accumulated_model.parameters(),
            lr=0.03,
            betas=(0.0, 0.0),
            eps=1e-8,
            weight_decay=0.0,
        )
        accumulated_strategy = SynapticIntelligence(
            accumulated_model,
            accumulated_optimizer,
            config,
        )
        accumulated_strategy.load_state_dict(consolidated_state)
        accumulated_strategy.begin_task("qa2")
        with torch.no_grad():
            accumulated_model.weight.add_(torch.tensor([0.15, -0.05]))
        accumulated_optimizer.zero_grad(set_to_none=True)
        task_target = torch.tensor([-0.2, 0.3])
        for _ in range(microbatch_count):
            task_loss = (
                accumulated_model.weight - task_target
            ).pow(2).sum()
            (
                task_loss / microbatch_count
                + accumulated_strategy.penalty() / microbatch_count
            ).backward()
        accumulated_gradients[microbatch_count] = (
            accumulated_model.weight.grad.detach().clone()
        )
    for microbatch_count in (8, 32):
        torch.testing.assert_close(
            accumulated_gradients[microbatch_count],
            accumulated_gradients[1],
            rtol=1e-6,
            atol=1e-7,
        )

    ordinary = VectorModel()
    controlled = copy.deepcopy(ordinary)
    optimizer_a = torch.optim.AdamW(ordinary.parameters(), lr=0.01)
    optimizer_b = torch.optim.AdamW(controlled.parameters(), lr=0.01)
    disabled = SynapticIntelligence(
        controlled,
        optimizer_b,
        SynapticIntelligenceConfig(si_lambda=0.0),
    )
    disabled.begin_task("qa1")
    target = torch.tensor([0.2, 0.1])
    optimizer_a.zero_grad()
    (ordinary.weight - target).pow(2).sum().backward()
    optimizer_a.step()
    optimizer_b.zero_grad()
    ((controlled.weight - target).pow(2).sum() + disabled.penalty()).backward()
    disabled.before_optimizer_step()
    optimizer_b.step()
    disabled.after_optimizer_step()
    disabled.end_task("qa1")
    torch.testing.assert_close(ordinary.weight, controlled.weight, rtol=0, atol=0)


def test_model_and_fastmem_contract() -> None:
    tiny = _gpt2_tiny_model_config()
    with tempfile.TemporaryDirectory(prefix="babilong-cl-model-smoke-") as temporary:
        root = Path(temporary)
        base_config = _tiny_config(root, "base_rmt")
        fast_config = _tiny_config(root, "fastmem")
        seed_everything(48)
        base = build_model(base_config, tiny_config=tiny)
        seed_everything(48)
        fast = build_model(fast_config, tiny_config=tiny)
        torch.testing.assert_close(
            base.model.memory_cell.memory,
            fast.fast_cell.initial_memory_tokens,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            fast.fast_cell.initial_memory_tokens,
            fast.fast_cell.fast_memory_tokens,
            rtol=0,
            atol=0,
        )
        fast_ids = {id(fast.fast_cell.fast_memory_tokens)}
        assert not fast_ids & {id(parameter) for parameter in fast.slow_parameters()}
        assert id(fast.fast_cell.initial_memory_tokens) in {
            id(parameter) for parameter in fast.slow_parameters()
        }
        ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        mask = torch.tensor([[False, False, True, True]])
        base.model.eval()
        fast.model.eval()
        base_output = base.model(input_ids=ids, labels=ids, labels_mask=mask)
        fast_output = fast.model(input_ids=ids, labels=ids, labels_mask=mask)
        torch.testing.assert_close(base_output.logits, fast_output.logits, rtol=0, atol=0)
        base.model.train()
        fast.model.train()
        base_output = base.model(input_ids=ids, labels=ids, labels_mask=mask)
        fast_output = fast.model(input_ids=ids, labels=ids, labels_mask=mask)
        base_output.loss.backward()
        fast_output.loss.backward()
        fast.fast_cell.fast_memory_tokens.grad.fill_(1000.0)
        before = fast.fast_cell.fast_memory_tokens.detach().clone()
        diagnostics = fast.fast_cell.apply_fast_update(
            fast_lr=0.005,
            grad_scale=1.0,
            clip_norm=1.0,
        )
        assert diagnostics["applied"] == 1
        assert not torch.equal(before, fast.fast_cell.fast_memory_tokens)


def test_tiny_end_to_end_and_resume() -> None:
    tokenizer = TinyTokenizer()
    tiny = _gpt2_tiny_model_config()
    for model, method in (
        ("gpt2", "none"),
        ("gpt2", "si"),
        ("base_rmt", "none"),
        ("fastmem0", "none"),
        ("fastmem", "none"),
    ):
        with tempfile.TemporaryDirectory(prefix=f"babilong-cl-{model}-{method}-") as temporary:
            root = Path(temporary)
            manifest = _tiny_data(root)
            config = _tiny_config(root, model, cl_method=method)
            trainer = UnifiedTrainer(
                config=config,
                data_manifest=manifest,
                tokenizer=tokenizer,
                tiny_model_config=tiny,
                tensorboard=False,
            )
            raw = trainer.run(max_tasks=1)
            assert raw["status"] == "partial"
            assert raw["cumulative_slow_steps"] == 1
            assert raw["train_tasks"][0]["slow_steps"] == 1
            if model in {"fastmem0", "fastmem"}:
                assert raw["fast_update_attempts"] == 2
                assert raw["fast_updates_applied"] == (2 if model == "fastmem" else 0)
            assert (config.run_dir / "checkpoints" / "latest.pt").is_file()
            write_run_artifacts(raw, config.run_dir)
            assert (config.run_dir / "summary.md").is_file()
            assert (
                config.run_dir / "tables" / "accuracy_matrix_long.csv"
            ).is_file()

            if model == "gpt2":
                resumed = UnifiedTrainer(
                    config=config,
                    data_manifest=manifest,
                    tokenizer=tokenizer,
                    tiny_model_config=tiny,
                    tensorboard=False,
                ).run(max_tasks=2)
                assert resumed["cumulative_slow_steps"] == 2
                assert len(resumed["train_tasks"]) == 2


def test_midtask_resume_equivalence() -> None:
    tokenizer = TinyTokenizer()
    tiny = _gpt2_tiny_model_config()
    with tempfile.TemporaryDirectory(prefix="babilong-cl-midtask-") as temporary:
        root = Path(temporary)
        manifest = _tiny_data(root)
        interrupted_config = replace(
            _tiny_config(root, "gpt2", cl_method="si"),
            slow_steps_per_task=2,
            legacy_configured_iters=1,
            checkpoint_interval=1,
        )
        interrupted = UnifiedTrainer(
            config=interrupted_config,
            data_manifest=manifest,
            tokenizer=tokenizer,
            tiny_model_config=tiny,
            tensorboard=False,
        )
        original_save = interrupted._save_progress

        def save_then_stop(**kwargs):
            original_save(**kwargs)
            if int(kwargs["task_slow_step"]) == 1:
                interrupted.stop_requested = True

        interrupted._save_progress = save_then_stop  # type: ignore[method-assign]
        try:
            interrupted.run(max_tasks=1)
        except InterruptedError:
            pass
        else:
            raise AssertionError("Injected interruption did not stop training")

        resumed = UnifiedTrainer(
            config=interrupted_config,
            data_manifest=manifest,
            tokenizer=tokenizer,
            tiny_model_config=tiny,
            tensorboard=False,
        ).run(max_tasks=1)
        assert resumed["cumulative_slow_steps"] == 2
        assert resumed["train_tasks"][0]["resume_start_slow_step"] == 1

        uninterrupted_config = replace(
            interrupted_config,
            results_root=str(root / "uninterrupted-results"),
        )
        uninterrupted = UnifiedTrainer(
            config=uninterrupted_config,
            data_manifest=manifest,
            tokenizer=tokenizer,
            tiny_model_config=tiny,
            tensorboard=False,
        ).run(max_tasks=1)
        resumed_state = torch.load(
            interrupted_config.run_dir / "checkpoints" / "latest.pt",
            map_location="cpu",
        )["model"]
        uninterrupted_state = torch.load(
            uninterrupted_config.run_dir / "checkpoints" / "latest.pt",
            map_location="cpu",
        )["model"]
        assert list(resumed_state) == list(uninterrupted_state)
        for name in resumed_state:
            torch.testing.assert_close(
                resumed_state[name], uninterrupted_state[name], rtol=0, atol=0
            )
        assert (
            resumed["compare_answers_matrix"]
            == uninterrupted["compare_answers_matrix"]
        )
        assert (
            resumed["train_tasks"][0]["mean_loss"]
            == uninterrupted["train_tasks"][0]["mean_loss"]
        )
        assert (
            resumed["train_tasks"][0]["final_loss"]
            == uninterrupted["train_tasks"][0]["final_loss"]
        )

        incompatible = replace(interrupted_config, weight_decay=0.02)
        assert incompatible.run_dir != interrupted_config.run_dir
        isolated = UnifiedTrainer(
            config=incompatible,
            data_manifest=manifest,
            tokenizer=tokenizer,
            tiny_model_config=tiny,
            tensorboard=False,
        ).run(max_tasks=1)
        assert isolated["status"] == "partial"
        assert (incompatible.run_dir / "config.json").is_file()


def gpu_smoke(
    *,
    base_config: ExperimentConfig,
    data_manifest: dict,
    models: list[str],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_config.tokenizer,
        revision=base_config.backbone_revision,
    )
    tokenizer.model_max_length = 10**9
    for model in models:
        method = "si" if model == "gpt2-si" else "none"
        model_key = "gpt2" if model == "gpt2-si" else model
        with tempfile.TemporaryDirectory(
            prefix=f"babilong-cl-gpu-{model}-"
        ) as temporary:
            config = replace(
                base_config,
                model=model_key,
                cl_method=method,
                si_lambda=1.0 if method == "si" else None,
                results_root=str(Path(temporary) / "results"),
                data_dir=str(base_config.resolved_data_dir),
                include_references=False,
                require_clean_git=False,
                slow_steps_per_task=1,
                legacy_configured_iters=0,
                warmup_steps=0,
                slow_batch_size=2,
                fast_batch_size=1,
                microbatch_size=1,
                checkpoint_interval=1,
                log_interval=1,
            )
            trainer = UnifiedTrainer(
                config=config,
                data_manifest=data_manifest,
                tokenizer=tokenizer,
                tensorboard=False,
            )
            task = "qa1"
            dataset = trainer._dataset("train", task)
            trainer.scheduler = _task_scheduler(
                trainer.optimizer,
                learning_rate=float(config.learning_rates[task]),
                warmup_steps=0,
                total_steps=1,
            )
            if trainer.strategy is not None:
                trainer.strategy.begin_task(task)
            trainer.bundle.model.train()
            trainer.bundle.reset_fast_memory()
            summary = trainer.train_task(
                task=task,
                dataset=dataset,
                cursor=EpochBatchCursor(
                    dataset_size=len(dataset),
                    batch_size=2,
                    task_seed=config.sampler_seeds[task],
                ),
                start_step=0,
                target_steps=1,
            )
            if trainer.strategy is not None:
                trainer.strategy.end_task(task)
            if model_key in {"fastmem0", "fastmem"}:
                assert summary["fast_update_attempts"] == 2
                assert summary["fast_updates_applied"] == (
                    2 if model_key == "fastmem" else 0
                )
            batch = trainer.collator([dataset[0]])
            input_ids = batch["input_ids_generate"].to(trainer.device)
            attention_mask = batch["attention_mask_generate"].to(trainer.device)
            trainer.bundle.model.eval()
            with trainer.bundle.evaluation_memory(), torch.no_grad():
                generated = trainer.bundle.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if generated.numel() == 0:
                raise RuntimeError(f"{model} generated no tokens")
            print(
                f"PASS: GPU {model} loss={summary['final_loss']:.6f} "
                f"allocated_mb={torch.cuda.memory_allocated() / 1024**2:.1f}"
            )
            trainer.monitor.close()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    tests = [
        ("provenance", verify_provenance),
        ("seed contract", test_seed_contract),
        ("config/collation contract", test_config_and_collation_contract),
        (
            "token normalization/microbatch equivalence",
            test_token_normalization_and_microbatch_equivalence,
        ),
        ("metrics", test_metrics),
        ("protocol-v3 aggregation", test_protocol_v3_aggregation),
        ("SI GPU placement", test_si_gpu_placement),
        ("SI equations", test_si_equations),
        ("model/FastMem contract", test_model_and_fastmem_contract),
        ("tiny end-to-end/resume", test_tiny_end_to_end_and_resume),
        ("mid-task resume equivalence", test_midtask_resume_equivalence),
    ]
    for name, test in tests:
        test()
        print(f"PASS: {name}")
    print("All unified continual-learning CPU smoke tests passed.")


if __name__ == "__main__":
    main()
