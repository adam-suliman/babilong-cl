from __future__ import annotations

import importlib.util
import json
import random
import shutil
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import CANONICAL_TASKS, DEFAULT_DATA_SEED
from .util import (
    atomic_write_json,
    atomic_write_text,
    bytes_sha256,
    file_lock,
    file_sha256,
    json_sha256,
)


TRAIN_REPO = "RMT-team/babilong-train-5k-samples"
TRAIN_REVISION = "b3513ef7c25c54ce706054530d47668c532019d6"
EVAL_REPO = "RMT-team/babilong"
EVAL_REVISION = "ee0d588794c7ac098062ee0d247c733d62e94fe2"
BABI_ARCHIVE_SHA256 = "50e529ad9c7144a2b176802f7496f5878d53b4e95d93cb0de957517fa84b09b2"
TOKEN_CAP = 1024
LOCATION_LABELS = ("bathroom", "bedroom", "garden", "hallway", "kitchen", "office")
OFFICIAL_TRAIN_TASKS = ("qa1", "qa2", "qa3")
GENERATED_TRAIN_TASKS = ("qa11", "qa12", "qa13")
TRAIN_STEMS = {
    "qa11": "qa11_basic-coreference",
    "qa12": "qa12_conjunction",
    "qa13": "qa13_compound-coreference",
}
TRAIN_EXPECTED_SHA256 = {
    "qa1": "acfac153e2a35afb9bcf4382426ce5efa7963cac957aa5b154d428307884cdd3",
    "qa2": "d443e95742632eed3ec549d796b889b2a3b0b20a61844ed82d4ec151d9d687ad",
    "qa3": "9180cf9abee1decf39a488bb936dc36a289f4faf066e97a7190fec0ad3bb045c",
    "qa11": "ab82922899446f4d386bd996e9e5199aca03637edcb2c7cb3947bf738f14b9f8",
    "qa12": "ad1c651403a87dcc3bbe84c3587acb084b210ce47d51ecf36cfc18d4313cd8eb",
    "qa13": "9e0430f9a0091a085e3738c528cb2ff86a5da5252cf6182b6b4cf6d8c84e6b98",
}
EVAL_EXPECTED_SHA256 = {
    "qa1": "1101607845951d2b2a0e0f01ddc0fe2a3096ea8436f6bc7d7879dc5730993cd3",
    "qa2": "7baae649af9883d84eeaec422d847064cf136b3de3eebcc08ea6d75af39c60e9",
    "qa3": "36a7bae93e535b89dd7a779621ca339674e2c2a646ceef43a505295f6d5336c2",
    "qa11": "6e59e1192ffbffc7982b634afdd821c00542b1f87ee32a867be538aabe529206",
    "qa12": "1f30c51eee4a8c4708c1b09408fec7052676f1730447c58c209bb9c229121a98",
    "qa13": "1ccbe12a20a4fbaf72947e62594bff96ea2507c5a0921f266be04036656bb288",
}
PARSER_SHA256 = "8a6e6e429d9cf41d555a2a7583cf1947640a44b53e453548dae9d745a837065c"
DATA_FORMAT = "babilong-qa6-0k-data-v1"
CANONICAL_MANIFEST_SHA256 = (
    "600b4a4791b5d213d56c3226e7553a60ef5401c2a1ed80f40ead439a51722292"
)


def _vendor_root() -> Path:
    return Path(__file__).parent / "third_party" / "rmt_babilong_release"


def _load_task_dataset() -> type[Any]:
    source = _vendor_root() / "babilong_utils.py"
    if file_sha256(source) != PARSER_SHA256:
        raise RuntimeError(f"Vendored bAbI parser hash mismatch: {source}")
    spec = importlib.util.spec_from_file_location("_babilong_cl_upstream_utils", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream parser: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TaskDataset


def normalized_signature(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        " ".join(str(row["input"]).split()),
        " ".join(str(row["question"]).split()),
        str(row["target"]).strip().lower(),
    )


def serialized_length(tokenizer: Any, row: Mapping[str, Any]) -> int:
    return (
        len(tokenizer.encode(str(row["input"]), add_special_tokens=False))
        + len(tokenizer.encode(str(row["question"]), add_special_tokens=False))
        + 1
        + len(tokenizer.encode(str(row["target"]), add_special_tokens=False))
        + 1
    )


def _token_stats(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [serialized_length(tokenizer, row) for row in rows]
    fit = [length for length in lengths if length <= TOKEN_CAP]
    return {
        "rows": len(rows),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "fit_1024": len(fit),
        "fit_min": min(fit) if fit else None,
        "fit_max": max(fit) if fit else None,
        "fit_mean": (sum(fit) / len(fit)) if fit else None,
        "excluded_over_1024": len(lengths) - len(fit),
        "effective_drop_last_64": (len(fit) // 64) * 64,
        "one_segment_512": sum(length <= 512 for length in fit),
        "two_segment_1024": sum(512 < length <= 1024 for length in fit),
    }


def _copy_hf_file(
    *,
    repo: str,
    revision: str,
    filename: str,
    destination: Path,
    expected_sha256: str,
) -> None:
    from huggingface_hub import hf_hub_download

    if destination.exists():
        actual = file_sha256(destination)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Existing file hash mismatch for {destination}: {actual} != {expected_sha256}"
            )
        return
    source = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )
    actual = file_sha256(source)
    if actual != expected_sha256:
        raise RuntimeError(f"Pinned HF file hash mismatch: {actual} != {expected_sha256}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _records_from_archive_member(
    archive: Path,
    member: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    TaskDataset = _load_task_dataset()
    with zipfile.ZipFile(archive) as handle:
        info = handle.getinfo(member)
        payload = handle.read(member)
    with tempfile.TemporaryDirectory(prefix="babilong-cl-data-") as temporary:
        source = Path(temporary) / Path(member).name
        source.write_bytes(payload)
        dataset = TaskDataset(str(source))
        sample_ids = sorted(
            int(value) for value in dataset.fact_dataset.sample_num.unique()
        )
        records = []
        for scenario_id in sample_ids:
            sample = dataset[scenario_id]
            records.append(
                {
                    "input": " ".join(str(fact) for fact in sample["facts"]),
                    "question": str(sample["question"]).rstrip() + " ",
                    "target": str(sample["answer"]).strip(),
                    "scenario_id": scenario_id,
                    "source_member": member,
                }
            )
    return records, {
        "member": member,
        "member_sha256": bytes_sha256(payload),
        "member_crc32": f"{info.CRC:08x}",
        "member_size": info.file_size,
        "parsed_scenarios": len(records),
    }


def _write_legacy_generated_json(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    # This byte format reproduces the already-used QA11-13 data hashes.
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(list(rows), indent=2, sort_keys=True)
    atomic_write_text(path, text)


class DataProvider(ABC):
    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        raise NotImplementedError


class FixedZeroContextProvider(DataProvider):
    def __init__(
        self,
        *,
        data_dir: str | Path,
        babi_archive: str | Path,
        tokenizer_name: str,
        tokenizer_revision: str,
        data_seed: int = DEFAULT_DATA_SEED,
        validation_rows: int = 400,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.babi_archive = Path(babi_archive)
        self.tokenizer_name = tokenizer_name
        self.tokenizer_revision = tokenizer_revision
        self.data_seed = int(data_seed)
        self.validation_rows = int(validation_rows)

    def prepare(self) -> dict[str, Any]:
        from transformers import AutoTokenizer

        if self.data_seed != DEFAULT_DATA_SEED:
            raise ValueError(
                f"Canonical data requires data_seed={DEFAULT_DATA_SEED}; "
                f"got {self.data_seed}. A different seed is a separate data study."
            )
        if not self.babi_archive.is_file():
            raise FileNotFoundError(f"Missing bAbI archive: {self.babi_archive}")
        archive_hash = file_sha256(self.babi_archive)
        if archive_hash != BABI_ARCHIVE_SHA256:
            raise RuntimeError(
                f"bAbI archive hash mismatch: {archive_hash} != {BABI_ARCHIVE_SHA256}"
            )
        manifest_path = self.data_dir / "manifest.json"
        with file_lock(self.data_dir / ".prepare.lock"):
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self._validate_existing(manifest)
                return manifest

            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                revision=self.tokenizer_revision,
            )
            tokenizer.model_max_length = 10**9
            train_info: dict[str, Any] = {}
            eval_info: dict[str, Any] = {}

            for task in OFFICIAL_TRAIN_TASKS:
                destination = self.data_dir / "train" / f"{task}.json"
                _copy_hf_file(
                    repo=TRAIN_REPO,
                    revision=TRAIN_REVISION,
                    filename=f"data/{task}/0k.json",
                    destination=destination,
                    expected_sha256=TRAIN_EXPECTED_SHA256[task],
                )

            for task in CANONICAL_TASKS:
                destination = self.data_dir / "eval" / f"{task}.json"
                _copy_hf_file(
                    repo=EVAL_REPO,
                    revision=EVAL_REVISION,
                    filename=f"data/{task}/0k.json",
                    destination=destination,
                    expected_sha256=EVAL_EXPECTED_SHA256[task],
                )
                rows = json.loads(destination.read_text(encoding="utf-8"))
                if len(rows) != 100:
                    raise RuntimeError(f"{task} public eval has {len(rows)} rows, expected 100")
                eval_info[task] = {
                    "path": str(Path("eval") / f"{task}.json"),
                    "sha256": file_sha256(destination),
                    "source": "official_hf",
                    "token_stats": _token_stats(tokenizer, rows),
                }

            for task_index, task in enumerate(GENERATED_TRAIN_TASKS):
                member = (
                    f"tasks_1-20_v1-2/en-10k/{TRAIN_STEMS[task]}_train.txt"
                )
                records, source_info = _records_from_archive_member(
                    self.babi_archive, member
                )
                eval_rows = json.loads(
                    (self.data_dir / "eval" / f"{task}.json").read_text(encoding="utf-8")
                )
                eval_signatures = {normalized_signature(row) for row in eval_rows}
                overlap_ids = [
                    int(row["scenario_id"])
                    for row in records
                    if normalized_signature(row) in eval_signatures
                ]
                overlap_set = set(overlap_ids)
                eligible = [
                    row for row in records if int(row["scenario_id"]) not in overlap_set
                ]
                selected_ids = sorted(
                    random.Random(self.data_seed + task_index).sample(
                        [int(row["scenario_id"]) for row in eligible],
                        5000,
                    )
                )
                selected_set = set(selected_ids)
                selected = [
                    row for row in eligible if int(row["scenario_id"]) in selected_set
                ]
                selected.sort(key=lambda row: int(row["scenario_id"]))
                invalid = sorted(
                    {str(row["target"]).lower() for row in selected}
                    - set(LOCATION_LABELS)
                )
                if invalid:
                    raise RuntimeError(f"{task} has invalid labels: {invalid}")
                destination = self.data_dir / "train" / f"{task}.json"
                _write_legacy_generated_json(destination, selected)
                actual = file_sha256(destination)
                expected = TRAIN_EXPECTED_SHA256[task]
                if actual != expected:
                    raise RuntimeError(
                        f"Generated {task} hash mismatch: {actual} != {expected}"
                    )
                train_info[task] = {
                    "path": str(Path("train") / f"{task}.json"),
                    "sha256": actual,
                    "source": "generated_babi_en_10k",
                    "selection_seed": self.data_seed + task_index,
                    "selected_scenario_ids_sha256": json_sha256(selected_ids),
                    "eligible_rows": len(eligible),
                    "excluded_public_eval_overlap_ids": overlap_ids,
                    "source_info": source_info,
                    "token_stats": _token_stats(tokenizer, selected),
                }

            for task in OFFICIAL_TRAIN_TASKS:
                path = self.data_dir / "train" / f"{task}.json"
                rows = json.loads(path.read_text(encoding="utf-8"))
                train_info[task] = {
                    "path": str(Path("train") / f"{task}.json"),
                    "sha256": file_sha256(path),
                    "source": "official_hf",
                    "token_stats": _token_stats(tokenizer, rows),
                }

            validation_info = self._prepare_validation(tokenizer)
            payload: dict[str, Any] = {
                "format": DATA_FORMAT,
                "context_length": "0k",
                "noise_source": "none",
                "data_seed": self.data_seed,
                "serialization": "input + question + GEN + target + EOS",
                "token_cap": TOKEN_CAP,
                "location_labels": list(LOCATION_LABELS),
                "tokenizer": {
                    "name": self.tokenizer_name,
                    "revision": self.tokenizer_revision,
                    "class": type(tokenizer).__name__,
                },
                "train_repo": TRAIN_REPO,
                "train_revision": TRAIN_REVISION,
                "eval_repo": EVAL_REPO,
                "eval_revision": EVAL_REVISION,
                "babi_archive": {
                    "filename": self.babi_archive.name,
                    "sha256": archive_hash,
                },
                "upstream_parser": {
                    "sha256": PARSER_SHA256,
                    "source_commit": "5a3d0722a10dc874b69075d74a8f66bffb957c2d",
                },
                "train": {task: train_info[task] for task in CANONICAL_TASKS},
                "validation": validation_info,
                "eval": {task: eval_info[task] for task in CANONICAL_TASKS},
            }
            payload["manifest_sha256"] = json_sha256(payload)
            if payload["manifest_sha256"] != CANONICAL_MANIFEST_SHA256:
                raise RuntimeError(
                    "Canonical data manifest changed: "
                    f"{payload['manifest_sha256']} != "
                    f"{CANONICAL_MANIFEST_SHA256}"
                )
            atomic_write_json(manifest_path, payload)
            return payload

    def _prepare_validation(self, tokenizer: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        excluded_by_task: dict[str, set[tuple[str, str, str]]] = {}
        for task in CANONICAL_TASKS:
            excluded: set[tuple[str, str, str]] = set()
            for split in ("train", "eval"):
                rows = json.loads(
                    (self.data_dir / split / f"{task}.json").read_text(encoding="utf-8")
                )
                excluded.update(normalized_signature(row) for row in rows)
            excluded_by_task[task] = excluded

        with zipfile.ZipFile(self.babi_archive) as archive:
            for task_index, task in enumerate(CANONICAL_TASKS):
                member = f"tasks_1-20_v1-2/en-valid-10k/{task}_valid.txt"
                payload = archive.read(member)
                with tempfile.TemporaryDirectory(prefix=f"babilong-cl-{task}-valid-") as tmp:
                    source = Path(tmp) / f"{task}_valid.txt"
                    source.write_bytes(payload)
                    TaskDataset = _load_task_dataset()
                    dataset = TaskDataset(str(source))
                    ids = sorted(
                        int(value)
                        for value in dataset.fact_dataset.sample_num.unique()
                    )
                    candidates = []
                    for scenario_id in ids:
                        sample = dataset[scenario_id]
                        candidates.append(
                            {
                                "input": " ".join(str(fact) for fact in sample["facts"]),
                                "question": str(sample["question"]).rstrip() + " ",
                                "target": str(sample["answer"]).strip(),
                                "scenario_id": scenario_id,
                                "source_member": member,
                            }
                        )
                disjoint = [
                    row
                    for row in candidates
                    if normalized_signature(row) not in excluded_by_task[task]
                ]
                fitting = [
                    row for row in disjoint if serialized_length(tokenizer, row) <= TOKEN_CAP
                ]
                if len(fitting) < self.validation_rows:
                    raise RuntimeError(
                        f"{task} has only {len(fitting)} disjoint fitting validation rows"
                    )
                selection_seed = self.data_seed + 1000 + task_index
                selected_ids = sorted(
                    random.Random(selection_seed).sample(
                        [int(row["scenario_id"]) for row in fitting],
                        self.validation_rows,
                    )
                )
                selected_set = set(selected_ids)
                selected = [
                    row for row in fitting if int(row["scenario_id"]) in selected_set
                ]
                selected.sort(key=lambda row: int(row["scenario_id"]))
                destination = self.data_dir / "validation" / f"{task}.json"
                atomic_write_json(destination, selected)
                result[task] = {
                    "path": str(Path("validation") / f"{task}.json"),
                    "sha256": file_sha256(destination),
                    "source": "babi_en_valid_10k",
                    "source_member": member,
                    "source_member_sha256": bytes_sha256(payload),
                    "source_rows": len(candidates),
                    "disjoint_rows": len(disjoint),
                    "fitting_rows": len(fitting),
                    "selected_rows": len(selected),
                    "selection_seed": selection_seed,
                    "selected_scenario_ids_sha256": json_sha256(selected_ids),
                    "token_stats": _token_stats(tokenizer, selected),
                }
        return result

    def _validate_existing(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("format") != DATA_FORMAT:
            raise ValueError("Existing data manifest has an incompatible format")
        if int(manifest.get("data_seed", -1)) != self.data_seed:
            raise ValueError("Existing data manifest has a different data seed")
        if manifest.get("context_length") != "0k" or manifest.get("noise_source") != "none":
            raise ValueError("Existing data manifest is not the fixed 0k protocol")
        tokenizer = manifest.get("tokenizer", {})
        if (
            tokenizer.get("name") != self.tokenizer_name
            or tokenizer.get("revision") != self.tokenizer_revision
        ):
            raise ValueError("Existing data manifest uses a different tokenizer")
        if (
            manifest.get("train_repo") != TRAIN_REPO
            or manifest.get("train_revision") != TRAIN_REVISION
            or manifest.get("eval_repo") != EVAL_REPO
            or manifest.get("eval_revision") != EVAL_REVISION
        ):
            raise ValueError("Existing data manifest uses different pinned HF sources")
        if (
            manifest.get("babi_archive", {}).get("sha256")
            != file_sha256(self.babi_archive)
        ):
            raise ValueError("Existing data manifest uses a different bAbI archive")
        payload = dict(manifest)
        recorded = payload.pop("manifest_sha256", None)
        if recorded != json_sha256(payload):
            raise RuntimeError("Existing data manifest hash is invalid")
        if recorded != CANONICAL_MANIFEST_SHA256:
            raise RuntimeError("Existing data manifest is not canonical protocol v1")
        for split in ("train", "validation", "eval"):
            expected_tasks = set(CANONICAL_TASKS)
            if set(manifest[split]) != expected_tasks:
                raise RuntimeError(f"Existing {split} manifest tasks do not match")
            for task, info in manifest[split].items():
                path = self.data_dir / info["path"]
                if not path.is_file() or file_sha256(path) != info["sha256"]:
                    raise RuntimeError(f"Existing data file failed validation: {path}")
                if split == "train" and info["sha256"] != TRAIN_EXPECTED_SHA256[task]:
                    raise RuntimeError(f"Unexpected canonical training hash for {task}")
                if split == "eval" and info["sha256"] != EVAL_EXPECTED_SHA256[task]:
                    raise RuntimeError(f"Unexpected canonical evaluation hash for {task}")


class UnsupportedNoiseProvider(DataProvider):
    def prepare(self) -> dict[str, Any]:
        raise NotImplementedError(
            "PG19/noisy-context generation is intentionally outside protocol v1. "
            "Implement a new DataProvider without changing the trainer."
        )


class TokenizedBabilongDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        *,
        token_cap: int = TOKEN_CAP,
        require_all_fit: bool = False,
    ) -> None:
        self.path = Path(path)
        source_rows = json.loads(self.path.read_text(encoding="utf-8"))
        self.records: list[dict[str, Any]] = []
        self.excluded: list[dict[str, int]] = []
        for row_index, row in enumerate(source_rows):
            input_tokens = tokenizer.encode(str(row["input"]), add_special_tokens=False)
            question_tokens = tokenizer.encode(
                str(row["question"]), add_special_tokens=False
            )
            target_tokens = tokenizer.encode(
                str(row["target"]), add_special_tokens=False
            )
            length = len(input_tokens) + len(question_tokens) + len(target_tokens) + 2
            if length > token_cap:
                self.excluded.append({"row_index": row_index, "tokens": length})
                continue
            self.records.append(
                {
                    "input_tokens": input_tokens,
                    "question_tokens": question_tokens,
                    "target_tokens": target_tokens,
                    "target_text": str(row["target"]),
                    "row_index": row_index,
                    "serialized_tokens": length,
                }
            )
        if require_all_fit and self.excluded:
            raise ValueError(
                f"{self.path} has {len(self.excluded)} rows over {token_cap} tokens"
            )
        if not self.records:
            raise ValueError(f"No fitting rows in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class OfficialCollator:
    def __init__(self, tokenizer: Any) -> None:
        gen_tokens = tokenizer.encode("GEN")
        if not gen_tokens:
            raise ValueError("Tokenizer produced no GEN token")
        self.gen_token = int(gen_tokens[0])
        self.eos_token = tokenizer.eos_token_id
        if self.eos_token is None:
            raise ValueError("Tokenizer has no eos_token_id")
        self.pad_token = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else self.eos_token
        )

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        targets = [torch.tensor(row["target_tokens"], dtype=torch.long) for row in batch]
        input_ids = [
            torch.tensor(
                row["input_tokens"]
                + row["question_tokens"]
                + [self.gen_token]
                + row["target_tokens"]
                + [self.eos_token],
                dtype=torch.long,
            )
            for row in batch
        ]
        generate_ids = [
            torch.tensor(
                row["input_tokens"] + row["question_tokens"] + [self.gen_token],
                dtype=torch.long,
            )
            for row in batch
        ]
        attention_mask = [torch.ones_like(ids, dtype=torch.bool) for ids in input_ids]
        labels_mask = [torch.zeros_like(ids, dtype=torch.bool) for ids in input_ids]
        for mask, target in zip(labels_mask, targets):
            mask[-len(target) - 2 :] = True
        padded_input = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token
        )
        padded_generate = pad_sequence(
            generate_ids, batch_first=True, padding_value=self.pad_token
        )
        return {
            "input_ids": padded_input,
            "labels": padded_input,
            "labels_mask": pad_sequence(labels_mask, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
            "input_ids_generate": padded_generate,
            "attention_mask_generate": padded_generate.ne(self.pad_token),
            "target_text": [str(row["target_text"]) for row in batch],
            "row_index": [int(row["row_index"]) for row in batch],
        }


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_split_paths(data_dir: str | Path, split: str) -> dict[str, Path]:
    root = Path(data_dir)
    return {task: root / split / f"{task}.json" for task in CANONICAL_TASKS}
