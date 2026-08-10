from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .availability import DatasetAvailability, is_keyframe_sample
from .contract import validate_lidar_stage
from .dataset import FZIAURADataset, _normalize_scene_lookup_id
from .frame import FZIAURAFrame
from .index import FZIAURAIndexRow
from .io import iter_jsonl_binary, load_jsonl_range, resolve_scene_path
from .scene import FZIAURAScene


class FrameDataset:
    def __init__(
        self,
        root,
        split: Optional[str] = None,
        split_version: str = "v1.0",
        sample_filter: str = "all",
        require_cameras=None,
        require_lidars=None,
        require_radars=None,
        lidar_stage: str = "motion_compensated",
        require_semantics_for=None,
        return_mode: str = "frame",
        cache_samples: bool = False,
        scene_filter: Optional[Callable[[dict], bool]] = None,
        _scene_rows=None,
        cache_scenes: bool = True,
        _availability: Optional[DatasetAvailability] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.split_version = split_version
        self.sample_filter = sample_filter
        self.require_cameras = tuple(require_cameras or ())
        self.require_lidars = tuple(require_lidars or ())
        self.require_radars = tuple(require_radars or ())
        self.lidar_stage = validate_lidar_stage(lidar_stage)
        self.require_semantics_for = tuple(require_semantics_for or ())
        self.return_mode = return_mode
        self.cache_samples = cache_samples
        self.scene_filter = scene_filter
        self.cache_scenes = cache_scenes
        self._scene_cache: dict[int, FZIAURAScene] = {}
        self.availability = _availability or DatasetAvailability.from_root(self.root)

        if sample_filter not in {
            "all",
            "boxes_3d",
            "semantic_lidar",
            "boxes_3d_and_semantic_lidar",
            "any_label",
            "sensor_available",
        }:
            raise ValueError(f"Unsupported sample_filter {sample_filter!r}")

        if return_mode not in {"frame", "dict"}:
            raise ValueError("return_mode must be 'frame' or 'dict'")

        if _scene_rows is None:
            _scene_rows = FZIAURADataset(
                root,
                split=split,
                split_version=split_version,
                cache_scenes=False,
            )._selected_rows

        if scene_filter is not None:
            _scene_rows = [row for row in _scene_rows if scene_filter(row)]

        self._scene_rows = list(_scene_rows)
        self.scene_ids = tuple(row["scene_id"] for row in self._scene_rows)
        self.scene_names = tuple(Path(row["path"]).name for row in self._scene_rows)
        self.rows: list[FZIAURAIndexRow] = []
        self._samples: Optional[list[dict]] = [] if cache_samples else None
        self.scene_ranges: dict[str, tuple[int, int]] = {}
        sample_ranges: list[tuple[int, int]] = []
        self._build_index(sample_ranges)
        self._sample_ranges: np.ndarray = np.asarray(
            sample_ranges, dtype=np.int64
        ).reshape(-1, 2)

    def _build_index(self, sample_ranges: list[tuple[int, int]]) -> None:
        for scene_index, scene_row in enumerate(self._scene_rows):
            scene_id = scene_row["scene_id"]
            scene_name = Path(scene_row["path"]).name
            scene_path = resolve_scene_path(self.root, scene_row["path"])
            start = len(self.rows)
            for _, start_offset, end_offset, sample in iter_jsonl_binary(
                scene_path / "samples.jsonl"
            ):
                frame_index = int(sample["frame_index"])
                timestamp_ns = int(sample["timestamp_ns"])
                lidar = sample["lidar"]
                lidar_raw = lidar["raw"]
                lidar_motion_compensated = lidar["motion_compensated"]
                labels = sample["labels"]
                is_keyframe = is_keyframe_sample(sample)
                semantic_sensors = (
                    tuple(labels.get("semantic_lidar", {}).keys())
                    if self.availability.allows_base()
                    else ()
                )
                camera_sensors = (
                    tuple(sample.get("cameras", {}).keys())
                    if self.availability.allows_sensor(
                        "camera", is_keyframe=is_keyframe
                    )
                    else ()
                )
                lidar_raw_sensors = (
                    tuple(lidar_raw.keys())
                    if self.availability.allows_sensor(
                        "lidar_raw", is_keyframe=is_keyframe
                    )
                    else ()
                )
                lidar_motion_compensated_sensors = (
                    tuple(lidar_motion_compensated.keys())
                    if self.availability.allows_sensor(
                        "lidar_motion_compensated", is_keyframe=is_keyframe
                    )
                    else ()
                )
                radar_sensors = (
                    tuple(sample.get("radar", {}).keys())
                    if self.availability.allows_sensor("radar", is_keyframe=is_keyframe)
                    else ()
                )
                row = FZIAURAIndexRow(
                    index=len(self.rows),
                    scene_id=scene_id,
                    scene_name=scene_name,
                    scene_path=scene_path,
                    scene_index=scene_index,
                    frame_index=frame_index,
                    timestamp_ns=timestamp_ns,
                    has_boxes_3d=bool(labels.get("boxes_3d", False)),
                    semantic_lidar_sensors=semantic_sensors,
                    camera_sensors=camera_sensors,
                    lidar_raw_sensors=lidar_raw_sensors,
                    lidar_motion_compensated_sensors=lidar_motion_compensated_sensors,
                    radar_sensors=radar_sensors,
                )
                if self._keep(row):
                    self.rows.append(row)
                    sample_ranges.append((start_offset, end_offset))
                    if self._samples is not None:
                        self._samples.append(sample)
            self.scene_ranges[scene_id] = (start, len(self.rows))

    def _sample_at(self, index: int) -> dict:
        if self._samples is not None:
            return self._samples[index]
        row = self.rows[index]
        start_offset, end_offset = self._sample_ranges[index]
        sample = load_jsonl_range(
            row.scene_path / "samples.jsonl", start_offset, end_offset
        )
        if not isinstance(sample, dict):
            raise TypeError(
                f"{row.scene_path / 'samples.jsonl'}: indexed sample is not an object"
            )
        return sample

    def _keep(self, row: FZIAURAIndexRow) -> bool:
        if not self._has_required_sensors(row):
            return False
        if self.require_semantics_for and not all(
            sensor in row.semantic_lidar_sensors
            for sensor in self.require_semantics_for
        ):
            return False
        if self.sample_filter == "all":
            return True
        if self.sample_filter == "boxes_3d":
            return row.has_boxes_3d
        if self.sample_filter == "semantic_lidar":
            return bool(row.semantic_lidar_sensors)
        if self.sample_filter == "boxes_3d_and_semantic_lidar":
            return row.has_boxes_3d and bool(row.semantic_lidar_sensors)
        if self.sample_filter == "any_label":
            return row.has_boxes_3d or bool(row.semantic_lidar_sensors)
        if self.sample_filter == "sensor_available":
            return True
        return False

    def _has_required_sensors(self, row: FZIAURAIndexRow) -> bool:
        lidar_sensors = (
            row.lidar_motion_compensated_sensors
            if self.lidar_stage == "motion_compensated"
            else row.lidar_raw_sensors
        )
        return (
            all(sensor in row.camera_sensors for sensor in self.require_cameras)
            and all(sensor in lidar_sensors for sensor in self.require_lidars)
            and all(sensor in row.radar_sensors for sensor in self.require_radars)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if self.return_mode == "dict":
            return self.to_dict(index)
        row = self.rows[index]
        scene = self._get_scene(row.scene_index)
        return FZIAURAFrame.from_sample(
            self._sample_at(index),
            index=row.index,
            scene_id=row.scene_id,
            scene_name=row.scene_name,
            scene_path=row.scene_path,
            scene_index=row.scene_index,
            frame_index=row.frame_index,
            scene=scene,
            availability=self.availability,
        )

    def _create_scene(self, scene_index: int) -> FZIAURAScene:
        row = self._scene_rows[scene_index]
        return FZIAURAScene(
            self.root,
            scene_index,
            row["scene_id"],
            Path(row["path"]).name,
            resolve_scene_path(self.root, row["path"]),
            row,
            availability=self.availability,
        )

    def _get_scene(self, scene_index: int) -> FZIAURAScene:
        if not self.cache_scenes:
            return self._create_scene(scene_index)
        if scene_index not in self._scene_cache:
            self._scene_cache[scene_index] = self._create_scene(scene_index)
        return self._scene_cache[scene_index]

    def clear_scene_cache(self) -> None:
        self._scene_cache.clear()

    def indices_for_scene(self, scene_id: str) -> list[int]:
        start, end = self.scene_ranges[_normalize_scene_lookup_id(scene_id)]
        return list(range(start, end))

    def scene_id_for_index(self, index: int) -> str:
        return self.rows[index].scene_id

    def local_index_for_global(self, index: int) -> int:
        return self.rows[index].frame_index

    def to_dict(self, index: int) -> dict:
        return self.rows[index].to_dict()

    def iter_scenes(self):
        for i in range(len(self._scene_rows)):
            yield self._get_scene(i)

    def iter_frames(self):
        for i in range(len(self)):
            yield self[i]

    def detection_keyframe_indices(self) -> list[int]:
        return [row.index for row in self.rows if row.has_boxes_3d]

    def semantic_keyframe_indices(self, sensor: Optional[str] = None) -> list[int]:
        if sensor is None:
            return [row.index for row in self.rows if row.semantic_lidar_sensors]
        return [row.index for row in self.rows if sensor in row.semantic_lidar_sensors]

    def stats(self) -> dict:
        return {
            "split": self.split,
            "split_version": self.split_version,
            "frames": len(self),
            "scenes": len(self._scene_rows),
            "box_keyframes": len(self.detection_keyframe_indices()),
            "semantic_keyframes": len(self.semantic_keyframe_indices()),
        }
