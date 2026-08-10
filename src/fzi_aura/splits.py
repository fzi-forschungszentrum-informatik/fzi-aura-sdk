from __future__ import annotations

from pathlib import Path

from .errors import SplitError

KNOWN_SPLITS = ("train", "val", "test")


def split_path(root: Path, split_version: str, split: str) -> Path:
    return root / "splits" / split_version / f"{split}.txt"


def load_split_scene_ids(root: Path, split_version: str, split: str) -> list[str]:
    if split not in KNOWN_SPLITS:
        raise SplitError(f"Unknown split {split!r}. Expected one of {KNOWN_SPLITS}.")
    path = split_path(root, split_version, split)
    if not path.exists():
        raise SplitError(f"Split file does not exist: {path}")
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ids) != len(set(ids)):
        raise SplitError(f"Duplicate scene ids found in split file: {path}")
    return ids


def validate_splits(root: Path, split_version: str, all_scene_ids: set[str]) -> dict:
    loaded: dict[str, list[str]] = {}
    errors: list[str] = []
    for split in KNOWN_SPLITS:
        try:
            ids = load_split_scene_ids(root, split_version, split)
        except SplitError as exc:
            errors.append(str(exc))
            ids = []
        loaded[split] = ids
        missing = sorted(set(ids) - all_scene_ids)
        if missing:
            errors.append(
                f"{split} contains scene ids not present in dataset.json: {missing[:5]}"
            )

    for i, left in enumerate(KNOWN_SPLITS):
        for right in KNOWN_SPLITS[i + 1 :]:
            overlap = sorted(set(loaded[left]) & set(loaded[right]))
            if overlap:
                errors.append(f"{left}/{right} overlap: {overlap[:5]}")

    covered = set().union(*(set(v) for v in loaded.values()))
    missing_from_splits = sorted(all_scene_ids - covered)
    if missing_from_splits:
        errors.append(
            f"Splits do not cover all scenes. Missing: {missing_from_splits[:5]}"
        )

    result = {
        "split_version": split_version,
        "scene_count": {name: len(ids) for name, ids in loaded.items()},
        "ok": not errors,
        "errors": errors,
    }
    if errors:
        raise SplitError("; ".join(errors))
    return result
