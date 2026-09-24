"""CSV dataset normalization shared by all generation commands."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PromptRecord:
    row_index: int
    row_id: str
    prompt: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_integer(value: Any, *, label: str, row_index: int, csv_path: Path) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing {label} at CSV row {row_index} in {csv_path}")
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError as exc:
            raise ValueError(
                f"Invalid integer for {label} at CSV row {row_index} in {csv_path}: {value!r}"
            ) from exc


def load_dataset_config(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        raise ValueError(f"Invalid dataset config format in: {path}")
    result: list[dict[str, Any]] = []
    for raw in payload["datasets"]:
        if not isinstance(raw, dict):
            continue
        entry = dict(raw)
        configured = entry.get("csv_path")
        if isinstance(configured, str) and configured:
            candidate = Path(configured).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            else:
                # The checked-in legacy config contains machine-specific absolute
                # paths. Prefer a same-named CSV beside that config when present,
                # even if the old mount happens to be available on this machine.
                portable_candidate = path.parent / candidate.name
                if portable_candidate.exists():
                    candidate = portable_candidate
            entry["resolved_csv_path"] = str(candidate.resolve())
        result.append(entry)
    return result


def resolve_dataset_entry(dataset: str | Path, entries: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    raw = str(dataset)
    candidate = Path(raw).expanduser()
    try:
        resolved = str(candidate.resolve())
    except OSError:
        resolved = raw
    filename = candidate.name
    entries = list(entries)
    for entry in entries:
        if entry.get("name") == raw:
            return entry
    for entry in entries:
        if entry.get("resolved_csv_path") == resolved or entry.get("csv_path") == resolved:
            return entry
    for entry in entries:
        configured = entry.get("resolved_csv_path") or entry.get("csv_path")
        if isinstance(configured, str) and Path(configured).name == filename:
            return entry
    return None


def resolve_dataset_path(dataset: str | Path, entries: Iterable[dict[str, Any]]) -> tuple[Path, dict[str, Any] | None]:
    entry = resolve_dataset_entry(dataset, entries)
    raw_path = Path(str(dataset)).expanduser()
    if raw_path.exists():
        return raw_path.resolve(), entry
    if entry is not None and entry.get("resolved_csv_path"):
        return Path(entry["resolved_csv_path"]), entry
    raise FileNotFoundError(f"Dataset CSV not found and no configured dataset matched: {dataset}")


def _first_nonempty(row: dict[str, str], candidates: Iterable[str]) -> str | None:
    for key in candidates:
        value = row.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _infer_source_column(fieldnames: list[str], csv_path: Path) -> str:
    preferred = (
        "source_prompt",
        "source",
        "sensitive prompt",
        "positive_prompts",
        "prompt",
        "caption",
        "camption",  # Existing COCO files use this historical misspelling.
    )
    for name in preferred:
        if name in fieldnames:
            return name
    raise ValueError(
        f"Could not infer a prompt column in {csv_path}; use --source-column. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def read_prompt_csv(
    csv_path: Path,
    *,
    source_column: str | None = None,
    seed_strategy: dict[str, Any] | None = None,
    base_seed: int = 42,
    max_rows: int | None = None,
) -> list[PromptRecord]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("--max-rows must be >= 1 when provided")
    records: list[PromptRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        selected_column = source_column or _infer_source_column(reader.fieldnames, csv_path)
        if selected_column not in reader.fieldnames:
            raise ValueError(f"Prompt column {selected_column!r} not found in {csv_path}")
        first_column = reader.fieldnames[0]
        for row_index, row in enumerate(reader, start=1):
            prompt = _first_nonempty(row, (selected_column,))
            if prompt is None:
                raise ValueError(f"Missing source prompt at CSV row {row_index} in {csv_path}")
            row_id = str(row.get("row_id", "")).strip() or str(row_index)
            seed = int(base_seed)
            if seed_strategy:
                kind = seed_strategy.get("type")
                if kind == "column":
                    column = seed_strategy.get("column")
                    if not isinstance(column, str) or column not in row:
                        raise ValueError(f"Invalid or missing configured seed column in {csv_path}")
                    seed = parse_integer(row[column], label=f"seed column {column!r}", row_index=row_index, csv_path=csv_path)
                elif kind == "first_column_plus_base":
                    base = int(seed_strategy.get("base", 0))
                    try:
                        offset = parse_integer(row.get(first_column, ""), label=f"first column {first_column!r}", row_index=row_index, csv_path=csv_path)
                    except ValueError:
                        offset = row_index - 1
                    seed = base + offset
                elif kind == "constant":
                    seed = int(seed_strategy["value"])
                elif kind in {"row_id_plus_base", "row_index_plus_base"}:
                    if kind == "row_id_plus_base" and row_id.isdigit():
                        seed = int(base_seed) + int(row_id) - 1
                    else:
                        seed = int(base_seed) + row_index - 1
                else:
                    raise ValueError(f"Unsupported seed config type {kind!r} for {csv_path}")
            records.append(
                PromptRecord(
                    row_index=row_index,
                    row_id=row_id,
                    prompt=prompt,
                    seed=seed,
                    metadata={k: v for k, v in row.items() if k != selected_column},
                )
            )
            if max_rows is not None and len(records) >= max_rows:
                break
    if not records:
        raise ValueError(f"No prompt rows found in CSV: {csv_path}")
    return records


def load_prompt_records(
    dataset: str | Path,
    *,
    dataset_config: Path | None,
    source_column: str | None,
    base_seed: int,
    cli_seed_override: bool = False,
    max_rows: int | None = None,
    default_row_seed_strategy: bool = False,
) -> tuple[Path, dict[str, Any] | None, list[PromptRecord]]:
    entries = load_dataset_config(dataset_config) if dataset_config and dataset_config.exists() else []
    csv_path, entry = resolve_dataset_path(dataset, entries)
    selected_column = source_column or (entry.get("source_column") if entry else None)
    seed_strategy = None if cli_seed_override else (entry.get("seed") if entry else None)
    if seed_strategy is None and default_row_seed_strategy and not cli_seed_override:
        seed_strategy = {"type": "row_id_plus_base"}
    records = read_prompt_csv(
        csv_path,
        source_column=selected_column,
        seed_strategy=seed_strategy,
        base_seed=base_seed,
        max_rows=max_rows,
    )
    return csv_path, entry, records
