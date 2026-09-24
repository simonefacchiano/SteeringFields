"""Output naming, image persistence, metadata and resume helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .common import json_safe


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_")
    return (slug or "run")[:max_len]


def prompt_first8words_slug(prompt: str) -> str:
    words: list[str] = []
    for token in prompt.strip().split():
        cleaned = "".join(ch.lower() for ch in token if ch.isalnum())
        if cleaned:
            words.append(cleaned)
        if len(words) == 8:
            break
    return "_".join(words) if words else "prompt"


def sanitize_filename_token(value: str) -> str:
    return slugify(value, max_len=120)


def timestamped_run_name(label: str) -> str:
    """Prefix an automatically generated run label with the local timestamp."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}"


def steering_target_label(
    target_prompt: str | None,
    safe_prompt: str | None,
    avg_target_embeddings: str | None,
) -> str:
    """Describe the effective safe/target condition used for output naming."""
    if target_prompt:
        return target_prompt
    if safe_prompt is not None:
        return safe_prompt if safe_prompt else "empty_safe_prompt"
    if avg_target_embeddings:
        return "avg_target_embeddings"
    return "target"


def create_run_directory(
    output_root: Path,
    *,
    model: str,
    workflow: str,
    mode: str,
    run_name: str | None = None,
) -> Path:
    label = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root.expanduser().resolve() / model / workflow / ("steered" if mode == "steer" else "baseline") / slugify(label, max_len=120)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_paired_image(left, right):
    from PIL import Image
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
    canvas.paste(left.convert("RGB"), (0, 0))
    canvas.paste(right.convert("RGB"), (left.width, 0))
    return canvas


def write_metadata(path: Path, metadata: Mapping[str, Any]) -> Path:
    payload = dict(metadata)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    return path


def save_images_and_metadata(
    run_dir: Path,
    images: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    metadata_name: str = "meta.json",
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        image.save(run_dir / f"{sanitize_filename_token(name)}.png")
    write_metadata(run_dir / metadata_name, metadata)
    print(f"[done] run dir: {run_dir}")
    return run_dir


def find_existing_image(name_base: str, search_dirs: list[Path], suffixes: tuple[str, ...] = (".png", ".jpg")) -> Path | None:
    for directory in search_dirs:
        for suffix in suffixes:
            exact = directory / f"{name_base}{suffix}"
            if exact.exists():
                return exact
            matches = sorted(directory.glob(f"{name_base}_*{suffix}"))
            if matches:
                return matches[0]
    return None


def save_legacy_outputs(output_root: Path, images: Mapping[str, Any], metadata: Mapping[str, Any], name: str) -> Path:
    """Preserve the legacy utils.save_outputs directory convention."""
    run_dir = output_root / slugify(name, max_len=80)
    run_dir.mkdir(parents=True, exist_ok=True)
    for key, image in images.items():
        image.save(run_dir / f"{key}.png")
    write_metadata(run_dir / "meta.json", metadata)
    print(f"[done] wrote {run_dir}")
    return run_dir
