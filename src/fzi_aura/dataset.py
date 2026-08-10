from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .availability import DatasetAvailability
from .contract import PUBLIC_FORMAT_VERSION, require_format_version
from .errors import SceneNotFoundError, SplitError
from .io import load_json, resolve_scene_path
from .scene import FZIAURAScene
from .splits import load_split_scene_ids, validate_splits

_SCENE_DATE = r"20(?:25|26)-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
_SCENE_NAME_RE = re.compile(rf"(?:^|_)({_SCENE_DATE}_\d+)$")
_SCENE_ID_RE = re.compile(rf"(?:^|_)({_SCENE_DATE})$")


def _strip_scene_name(scene_name: str) -> str:
    match = _SCENE_NAME_RE.search(scene_name)
    return match.group(1) if match else scene_name


def _strip_scene_id(scene_id: str) -> str:
    recording_name, separator, scene_index = scene_id.partition("|")
    if not separator or not scene_index:
        return scene_id
    match = _SCENE_ID_RE.search(recording_name)
    return f"{match.group(1)}|{scene_index}" if match else scene_id


def _normalize_scene_lookup_id(scene_id: str) -> str:
    """Normalize a canonical scene ID or its path-safe scene-name spelling."""

    value = scene_id.strip()
    stripped_id = _strip_scene_id(value)
    if stripped_id != value:
        return stripped_id
    match = _SCENE_NAME_RE.search(value)
    if not match:
        return value
    recording_name, _, scene_index = match.group(1).rpartition("_")
    return f"{recording_name}|{scene_index}"


def _consumer_excluded_scene_ids(metadata: dict, source: Path) -> frozenset[str]:
    """Return manifest-declared scenes the SDK must not expose by default."""

    entries = metadata.get("consumer_excluded_scenes", [])
    if not isinstance(entries, list):
        raise SplitError(f"{source}: consumer_excluded_scenes must be a list")
    excluded: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SplitError(
                f"{source}: consumer_excluded_scenes[{index}] must be an object"
            )
        scene_id = entry.get("scene_id")
        reason = entry.get("reason")
        if not isinstance(scene_id, str) or not scene_id:
            raise SplitError(
                f"{source}: consumer_excluded_scenes[{index}] has invalid scene_id"
            )
        if not isinstance(reason, str) or not reason:
            raise SplitError(
                f"{source}: consumer_excluded_scenes[{index}] has invalid reason"
            )
        if scene_id in excluded:
            raise SplitError(
                f"{source}: duplicate consumer-excluded scene ID {scene_id!r}"
            )
        excluded.add(scene_id)
    return frozenset(excluded)


class FZIAURADataset:
    def __init__(
        self,
        root,
        split: Optional[str] = None,
        split_version: str = "v1.0",
        cache_scenes: bool = True,
    ):
        self.root = Path(root)
        self.availability = DatasetAvailability.from_root(self.root)
        self.available_layers = self.availability.layers
        self.split = split
        self.split_version = split_version
        self.cache_scenes = cache_scenes
        self.dataset_path = self.root / "dataset.json"
        self.metadata = load_json(self.dataset_path)
        require_format_version(self.metadata, str(self.dataset_path))
        self.format_version = PUBLIC_FORMAT_VERSION
        self._all_scene_rows = list(self.metadata.get("scenes", []))
        self.consumer_excluded_scene_ids = frozenset(
            _strip_scene_id(scene_id)
            for scene_id in _consumer_excluded_scene_ids(
                self.metadata, self.dataset_path
            )
        )
        for row in self._all_scene_rows:
            row["scene_id"] = _strip_scene_id(row["scene_id"])
        self._available_scene_rows = [
            row
            for row in self._all_scene_rows
            if row["scene_id"] not in self.consumer_excluded_scene_ids
        ]
        self._scene_by_id = {row["scene_id"]: row for row in self._available_scene_rows}
        self._scene_by_name = {
            _strip_scene_name(Path(row["path"]).name): row
            for row in self._available_scene_rows
        }
        if split is None:
            selected_ids = [row["scene_id"] for row in self._available_scene_rows]
        else:
            split_ids = [
                _strip_scene_id(scene_id)
                for scene_id in load_split_scene_ids(self.root, split_version, split)
            ]
            missing = [
                scene_id
                for scene_id in split_ids
                if scene_id not in {row["scene_id"] for row in self._all_scene_rows}
            ]
            if missing:
                raise SplitError(
                    f"Split {split} contains scene ids not present in dataset.json: {missing[:5]}"
                )
            selected_ids = [
                scene_id
                for scene_id in split_ids
                if scene_id not in self.consumer_excluded_scene_ids
            ]
        self._selected_rows = [self._scene_by_id[scene_id] for scene_id in selected_ids]
        self.scene_ids = tuple(row["scene_id"] for row in self._selected_rows)
        self.scene_names = tuple(Path(row["path"]).name for row in self._selected_rows)
        self._scene_cache: dict[int, FZIAURAScene] = {}

    def __len__(self) -> int:
        return len(self._selected_rows)

    def __getitem__(self, index: int) -> FZIAURAScene:
        if index < 0:
            index += len(self)
        if self.cache_scenes and index in self._scene_cache:
            return self._scene_cache[index]
        row = self._selected_rows[index]
        scene = FZIAURAScene(
            root=self.root,
            index=index,
            scene_id=row["scene_id"],
            name=_strip_scene_name(Path(row["path"]).name),
            path=resolve_scene_path(self.root, row["path"]),
            metadata=row,
            availability=self.availability,
        )
        if self.cache_scenes:
            self._scene_cache[index] = scene
        return scene

    def get_scene(self, scene_id: str) -> FZIAURAScene:
        """Return a scene by canonical ID or its path-safe folder name."""

        return self[self.index_of_scene(scene_id)]

    def index_of_scene(self, scene_id: str) -> int:
        normalized_scene_id = _normalize_scene_lookup_id(scene_id)
        try:
            return self.scene_ids.index(normalized_scene_id)
        except ValueError as exc:
            raise SceneNotFoundError(
                f"Scene id {scene_id!r} is not selected in this dataset."
            ) from exc

    def iter_scenes(self):
        for i in range(len(self)):
            yield self[i]

    def as_frames(
        self,
        sample_filter: str = "all",
        require_cameras=None,
        require_lidars=None,
        require_radars=None,
        lidar_stage: str = "motion_compensated",
        require_semantics_for=None,
        return_mode: str = "frame",
        cache_samples: bool = False,
        scene_filter=None,
    ):
        from .frame_dataset import FrameDataset

        return FrameDataset(
            self.root,
            split=self.split,
            split_version=self.split_version,
            sample_filter=sample_filter,
            require_cameras=require_cameras,
            require_lidars=require_lidars,
            require_radars=require_radars,
            lidar_stage=lidar_stage,
            require_semantics_for=require_semantics_for,
            return_mode=return_mode,
            cache_samples=cache_samples,
            scene_filter=scene_filter,
            _scene_rows=self._selected_rows,
            cache_scenes=self.cache_scenes,
            _availability=self.availability,
        )

    def stats(self) -> dict:
        counts = {
            "scenes": len(self),
            "samples": sum(
                int(row.get("counts", {}).get("samples", 0))
                for row in self._selected_rows
            ),
            "boxes_3d_rows": sum(
                int(row.get("counts", {}).get("boxes_3d_rows", 0))
                for row in self._selected_rows
            ),
            "semantic_lidar_frames": sum(
                int(row.get("counts", {}).get("semantic_lidar_frames", 0))
                for row in self._selected_rows
            ),
        }
        return {
            "split": self.split,
            "split_version": self.split_version,
            "counts": counts,
        }

    def validate_splits(self) -> dict:
        return validate_splits(self.root, self.split_version, set(self._scene_by_id))
