from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any, Union


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_jsonl_binary(path: Path):
    """Yield decoded JSONL rows with stable binary byte offsets."""
    path = Path(path)
    with path.open("rb") as handle:
        line_number = 0
        while True:
            start_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            end_offset = handle.tell()
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL row: {exc}"
                ) from exc
            yield line_number, start_offset, end_offset, row


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def load_jsonl_range(path: Path, start_offset: int, end_offset: int) -> Any:
    """Decode one JSON value from an immutable binary byte range."""
    with Path(path).open("rb") as handle:
        handle.seek(int(start_offset))
        payload = handle.read(int(end_offset) - int(start_offset))
    return json.loads(payload)


def resolve_scene_path(scene_path: Path, relative: Union[str, Path]) -> Path:
    base = Path(scene_path).resolve()
    path = Path(relative)
    if path.is_absolute() or PureWindowsPath(str(relative)).is_absolute():
        raise ValueError(f"Scene-relative paths must not be absolute: {relative!r}")

    resolved = (base / path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"Scene-relative path escapes the scene directory: {relative!r}"
        ) from exc
    return resolved
