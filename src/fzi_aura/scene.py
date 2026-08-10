from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .availability import DatasetAvailability, is_keyframe_sample
from .boxes import Box3D, BoxCollection, box_from_row
from .calibration import Calibration
from .contract import require_format_version, validate_lidar_stage
from .errors import CalibrationError, FZIAURAError
from .geometry import invert_transform, transform_matrix, transform_points
from .io import (
    iter_jsonl,
    iter_jsonl_binary,
    load_json,
    load_jsonl,
    load_jsonl_range,
    resolve_scene_path,
)
from .pointcloud import AccumulatedPointCloud

VEHICLE_SIGNALS_PATH = Path("ego/vehicle_signals.parquet")
OPTIONAL_VEHICLE_SIGNAL_COLUMNS = frozenset({"visibility"})


@dataclass(frozen=True)
class _BoxRange:
    timestamp_ns: int
    start_offset: int
    end_offset: int
    box_count: int
    frame_index: int


@dataclass(frozen=True)
class _BoxFileIndex:
    expected_frame: str
    expected_sensor_id: Optional[str]
    ranges: dict[int, _BoxRange]


class FZIAURAScene:
    def __init__(
        self,
        root: Path,
        index: int,
        scene_id: str,
        name: str,
        path: Path,
        metadata: dict[str, Any],
        availability: Optional[DatasetAvailability] = None,
    ):
        self.root = Path(root)
        self.index = index
        self.scene_id = scene_id
        self.name = name
        self.path = Path(path)
        self.metadata = metadata
        self.availability = availability or DatasetAvailability.from_root(self.root)
        self.counts = metadata.get("counts", {})
        self.start_timestamp_ns = metadata.get("start_timestamp_ns")
        self.end_timestamp_ns = metadata.get("end_timestamp_ns")
        self.duration_s = metadata.get("duration_s")
        self.sample_count = int(
            self.counts.get("samples", metadata.get("sample_count", 0))
        )
        self._scene_json: Optional[dict[str, Any]] = None
        self._sample_ranges: Optional[np.ndarray] = None
        self._calibration: Optional[Calibration] = None
        self._box_offset_cache: dict[Path, _BoxFileIndex] = {}
        self._sensor_box_paths: Optional[dict[str, Path]] = None
        self._parquet_cache: dict[tuple[str, Optional[tuple[str, ...]]], Any] = {}
        self._parquet_columns_cache: dict[str, frozenset[str]] = {}
        self._vehicle_signals_bulk_cache: dict[Optional[tuple[str, ...]], Any] = {}
        self._vehicle_signal_timestamp_lookup: Optional[dict[int, int]] = None
        self._available_sensor_cache: dict[str, tuple[str, ...]] = {}

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int):
        return self.frame(index)

    def load_metadata(self) -> dict[str, Any]:
        if self._scene_json is None:
            self._scene_json = load_json(self.path / "scene.json")
            require_format_version(self._scene_json, str(self.path / "scene.json"))
        return self._scene_json

    def load_samples(self) -> list[dict[str, Any]]:
        samples = load_jsonl(self.path / "samples.jsonl")
        if not self.sample_count:
            self.sample_count = len(samples)
        return samples

    def _sample_range_index(self) -> np.ndarray:
        if self._sample_ranges is None:
            ranges = [
                (start_offset, end_offset)
                for _, start_offset, end_offset, _ in iter_jsonl_binary(
                    self.path / "samples.jsonl"
                )
            ]
            self._sample_ranges = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
            if not self.sample_count:
                self.sample_count = len(self._sample_ranges)
        return self._sample_ranges

    def _sample_at(self, local_idx: int) -> dict[str, Any]:
        ranges = self._sample_range_index()
        start_offset, end_offset = ranges[local_idx]
        sample = load_jsonl_range(
            self.path / "samples.jsonl", int(start_offset), int(end_offset)
        )
        if not isinstance(sample, dict):
            raise FZIAURAError(
                f"{self.path / 'samples.jsonl'}: sample {local_idx} is not an object"
            )
        return sample

    def frame(self, local_idx: int):
        from .frame import FZIAURAFrame

        sample_count = len(self)
        if local_idx < 0:
            local_idx += sample_count
        if local_idx < 0 or local_idx >= sample_count:
            raise IndexError("scene frame index out of range")
        sample = self._sample_at(local_idx)
        return FZIAURAFrame.from_sample(
            sample,
            index=local_idx,
            scene_id=self.scene_id,
            scene_name=self.name,
            scene_path=self.path,
            scene_index=self.index,
            frame_index=int(sample["frame_index"]),
            scene=self,
            availability=self.availability,
        )

    def frames(
        self,
        sample_filter: str = "all",
        require_cameras=None,
        require_lidars=None,
        require_radars=None,
        lidar_stage: str = "motion_compensated",
        require_semantics_for=None,
        return_mode: str = "frame",
        cache_samples: bool = False,
    ):
        from .frame_dataset import FrameDataset

        return FrameDataset(
            self.root,
            split=None,
            sample_filter=sample_filter,
            require_cameras=require_cameras,
            require_lidars=require_lidars,
            require_radars=require_radars,
            lidar_stage=lidar_stage,
            require_semantics_for=require_semantics_for,
            return_mode=return_mode,
            cache_samples=cache_samples,
            _scene_rows=[self.metadata],
            _availability=self.availability,
        )

    def available_cameras(self) -> tuple[str, ...]:
        return self._available_sample_sensors("camera")

    def available_lidars(self, stage: str = "motion_compensated") -> tuple[str, ...]:
        validate_lidar_stage(stage)
        return self._available_sample_sensors(f"lidar_{stage}")

    def available_radars(self) -> tuple[str, ...]:
        return self._available_sample_sensors("radar")

    def _available_sample_sensors(self, modality: str) -> tuple[str, ...]:
        cached = self._available_sensor_cache.get(modality)
        if cached is not None:
            return cached
        if not self.availability.allows_any_sensor(modality):
            result: tuple[str, ...] = ()
        else:
            sensors: set[str] = set()
            for sample in iter_jsonl(self.path / "samples.jsonl"):
                is_keyframe = is_keyframe_sample(sample)
                if not self.availability.allows_sensor(
                    modality, is_keyframe=is_keyframe
                ):
                    continue
                if modality == "camera":
                    paths = sample.get("cameras", {})
                elif modality == "radar":
                    paths = sample.get("radar", {})
                elif modality == "lidar_raw":
                    paths = sample.get("lidar", {}).get("raw", {})
                elif modality == "lidar_motion_compensated":
                    paths = sample.get("lidar", {}).get("motion_compensated", {})
                else:
                    raise ValueError(f"unknown sensor modality {modality!r}")
                sensors.update(str(sensor) for sensor in paths)
            result = tuple(sorted(sensors))
        self._available_sensor_cache[modality] = result
        return result

    def has_sensor(self, sensor_key: str) -> bool:
        modality, _, sensor = sensor_key.partition("/")
        if modality == "camera":
            return sensor in self.available_cameras()
        if modality == "lidar":
            return sensor in self.available_lidars() or sensor in self.available_lidars(
                "raw"
            )
        if modality == "radar":
            return sensor in self.available_radars()
        return False

    def calibration(self) -> Calibration:
        if self._calibration is None:
            self._calibration = Calibration(self.path / "calibration.json")
        return self._calibration

    def detection_keyframe_indices(self) -> list[int]:
        return [
            i
            for i, sample in enumerate(iter_jsonl(self.path / "samples.jsonl"))
            if sample["labels"].get("boxes_3d", False)
        ]

    def semantic_keyframe_indices(self, sensor: Optional[str] = None) -> list[int]:
        out: list[int] = []
        for i, sample in enumerate(iter_jsonl(self.path / "samples.jsonl")):
            sensors = sample["labels"].get("semantic_lidar", {})
            if sensor is None and sensors:
                out.append(i)
            elif sensor is not None and sensor in sensors:
                out.append(i)
        return out

    def _box_path(self, sensor_id: Optional[str] = None) -> Path:
        if sensor_id is None:
            return self.path / "labels" / "boxes_3d.jsonl"
        return (
            self.path
            / "labels"
            / "boxes_3d_sensor_frame"
            / sensor_id
            / "boxes_3d.jsonl"
        )

    def _resolve_box_request(
        self, frame: Optional[str], sensor_id: Optional[str]
    ) -> tuple[str, Optional[str]]:
        if sensor_id is None:
            canonical_frame = "base_link" if frame is None else frame
            if canonical_frame != "base_link":
                raise ValueError(
                    "Base 3D boxes require frame='base_link' when sensor_id is not set."
                )
            return "base_link", None

        calibration = self.calibration()
        try:
            sensor_key = calibration.sensor_key(sensor_id, modality="lidar")
            sensor = calibration.sensors[sensor_key]
            canonical_sensor_id = sensor["sensor_id"]
            canonical_frame = sensor["frame"]
            if not isinstance(canonical_sensor_id, str) or not canonical_sensor_id:
                raise TypeError("calibration sensor_id must be a non-empty string")
            if not isinstance(canonical_frame, str) or not canonical_frame:
                raise TypeError("calibration frame must be a non-empty string")
        except (CalibrationError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Sensor-frame 3D boxes require a calibrated lidar sensor; got "
                f"sensor_id={sensor_id!r}."
            ) from exc

        if frame is not None:
            try:
                frame_sensor_key = calibration.sensor_key(frame, modality="lidar")
            except CalibrationError as exc:
                raise ValueError(
                    f"Frame {frame!r} does not identify lidar sensor "
                    f"{canonical_sensor_id!r}; expected {canonical_frame!r}."
                ) from exc
            if frame_sensor_key != sensor_key:
                raise ValueError(
                    f"Frame {frame!r} belongs to {frame_sensor_key!r}, not "
                    f"lidar sensor {canonical_sensor_id!r}."
                )

        return canonical_frame, canonical_sensor_id

    def _sensor_box_path(self, sensor_id: str) -> Optional[Path]:
        if self._sensor_box_paths is None:
            paths: dict[str, Path] = {}
            for sample in iter_jsonl(self.path / "samples.jsonl"):
                labels = sample.get("labels", {})
                sources = labels.get("boxes_3d_sensor_frame", {})
                if not isinstance(sources, dict):
                    continue
                for declared_sensor, relative in sources.items():
                    declared_sensor = str(declared_sensor)
                    path = resolve_scene_path(self.path, relative)
                    previous = paths.get(declared_sensor)
                    if previous is not None and previous != path:
                        raise FZIAURAError(
                            f"Scene {self.scene_id} declares conflicting 3D box "
                            f"paths for sensor {declared_sensor!r}: {previous} and "
                            f"{path}."
                        )
                    paths[declared_sensor] = path
            self._sensor_box_paths = paths
        return self._sensor_box_paths.get(sensor_id)

    @staticmethod
    def _box_row_fields(row, path: Path, line_number: int) -> tuple[int, int, str]:
        try:
            timestamp_ns = int(row["timestamp_ns"])
            frame_index = int(row["frame_index"])
            box_frame = row["box"]["frame"]
            if not isinstance(box_frame, str) or not box_frame:
                raise TypeError("box.frame must be a non-empty string")
        except (KeyError, TypeError, ValueError) as exc:
            raise FZIAURAError(
                f"{path}:{line_number}: malformed 3D box row: {exc}"
            ) from exc
        return timestamp_ns, frame_index, box_frame

    def _box_file_index(
        self,
        path: Path,
        expected_frame: str,
        expected_sensor_id: Optional[str],
    ) -> _BoxFileIndex:
        path = Path(path)
        cached = self._box_offset_cache.get(path)
        if cached is not None:
            if cached.expected_frame != expected_frame:
                raise FZIAURAError(
                    f"Box file {path} was indexed for frame {cached.expected_frame!r}, "
                    f"not {expected_frame!r}."
                )
            if cached.expected_sensor_id != expected_sensor_id:
                raise FZIAURAError(
                    f"Box file {path} was indexed for sensor "
                    f"{cached.expected_sensor_id!r}, not {expected_sensor_id!r}."
                )
            return cached

        ranges: dict[int, _BoxRange] = {}
        current_timestamp: Optional[int] = None
        current_start = 0
        current_end = 0
        current_count = 0
        current_frame_index = 0

        def finish_current() -> None:
            if current_timestamp is None:
                return
            ranges[current_timestamp] = _BoxRange(
                timestamp_ns=current_timestamp,
                start_offset=current_start,
                end_offset=current_end,
                box_count=current_count,
                frame_index=current_frame_index,
            )

        try:
            for line_number, start_offset, end_offset, row in iter_jsonl_binary(path):
                timestamp_ns, frame_index, box_frame = self._box_row_fields(
                    row, path, line_number
                )
                if box_frame != expected_frame:
                    raise FZIAURAError(
                        f"{path}:{line_number}: box.frame={box_frame!r}, "
                        f"expected {expected_frame!r}."
                    )
                if (
                    expected_sensor_id is not None
                    and row.get("sensor_id") != expected_sensor_id
                ):
                    raise FZIAURAError(
                        f"{path}:{line_number}: sensor_id={row.get('sensor_id')!r}, "
                        f"expected {expected_sensor_id!r}."
                    )
                if current_timestamp is not None and timestamp_ns < current_timestamp:
                    raise FZIAURAError(
                        f"{path}:{line_number}: box timestamps are not "
                        "non-decreasing and contiguous."
                    )
                if current_timestamp != timestamp_ns:
                    finish_current()
                    current_timestamp = timestamp_ns
                    current_start = start_offset
                    current_count = 0
                    current_frame_index = frame_index
                elif frame_index != current_frame_index:
                    raise FZIAURAError(
                        f"{path}:{line_number}: timestamp_ns={timestamp_ns} has "
                        f"inconsistent frame_index values {current_frame_index} "
                        f"and {frame_index}."
                    )
                current_end = end_offset
                current_count += 1
        except (OSError, ValueError) as exc:
            raise FZIAURAError(f"Cannot index 3D box file {path}: {exc}") from exc

        finish_current()
        result = _BoxFileIndex(
            expected_frame=expected_frame,
            expected_sensor_id=expected_sensor_id,
            ranges=ranges,
        )
        self._box_offset_cache[path] = result
        return result

    def _boxes_at_path(
        self,
        path: Path,
        timestamp_ns: int,
        frame: str,
        sensor_id: Optional[str],
        expected_frame_index: Optional[int] = None,
    ) -> BoxCollection:
        path = Path(path)
        if not path.is_file():
            return BoxCollection([], frame, int(timestamp_ns))
        entry = self._box_file_index(path, frame, sensor_id).ranges.get(
            int(timestamp_ns)
        )
        if entry is None:
            return BoxCollection([], frame, int(timestamp_ns))
        if (
            expected_frame_index is not None
            and entry.frame_index != expected_frame_index
        ):
            raise FZIAURAError(
                f"{path}: frame_index={entry.frame_index} does not match requesting "
                f"sample frame_index={expected_frame_index} at "
                f"timestamp_ns={entry.timestamp_ns}."
            )

        try:
            with path.open("rb") as handle:
                handle.seek(entry.start_offset)
                payload = handle.read(entry.end_offset - entry.start_offset)
        except OSError as exc:
            raise FZIAURAError(
                f"Cannot read indexed 3D boxes from {path}: {exc}"
            ) from exc

        boxes = []
        for line in payload.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise FZIAURAError(
                    f"{path}: invalid indexed 3D box JSON: {exc}"
                ) from exc
            boxes.append(box_from_row(row, sensor_id=sensor_id))
        return BoxCollection(boxes, frame, int(timestamp_ns))

    def _iter_boxes_from_path(
        self, path: Path, frame: str, sensor_id: Optional[str] = None
    ):
        path = Path(path)
        if not path.is_file():
            return
        try:
            for line_number, _, _, row in iter_jsonl_binary(path):
                _, _, box_frame = self._box_row_fields(row, path, line_number)
                if box_frame != frame:
                    raise FZIAURAError(
                        f"{path}:{line_number}: box.frame={box_frame!r}, "
                        f"expected {frame!r}."
                    )
                yield box_from_row(row, sensor_id=sensor_id)
        except (OSError, ValueError) as exc:
            raise FZIAURAError(f"Cannot read 3D box file {path}: {exc}") from exc

    def boxes_at(
        self,
        timestamp_ns: int,
        frame: Optional[str] = None,
        sensor_id: Optional[str] = None,
    ) -> BoxCollection:
        frame, sensor_id = self._resolve_box_request(frame, sensor_id)
        if sensor_id is None:
            path = self._box_path(None)
            return self._boxes_at_path(path, timestamp_ns, frame, sensor_id)

        sensor_path = self._sensor_box_path(sensor_id)
        if sensor_path is None:
            return BoxCollection([], frame, int(timestamp_ns))
        return self._boxes_at_path(sensor_path, timestamp_ns, frame, sensor_id)

    def tracks(self, category: Optional[str] = None) -> dict[str, list]:
        tracks: dict[str, list] = {}
        for box in self._iter_boxes_from_path(
            self._box_path(None), "base_link", sensor_id=None
        ):
            if category is not None and box.category != category:
                continue
            tracks.setdefault(box.object_id, []).append(box)
        return tracks

    def get_track(self, object_id: str) -> list:
        return [
            box
            for box in self._iter_boxes_from_path(
                self._box_path(None), "base_link", sensor_id=None
            )
            if box.object_id == object_id
        ]

    def _box_frame_transform(
        self,
        source_index: int,
        source_frame: str,
        target_index: int,
        target_frame: str,
    ) -> tuple[np.ndarray, str]:
        if target_index < 0:
            target_index += len(self)
        if target_index < 0 or target_index >= len(self):
            raise IndexError("target_index out of range")

        calibration = self.calibration()
        if source_frame == "base_link":
            source_base_from_box = np.eye(4, dtype=np.float64)
        else:
            source_base_from_box = calibration.matrix("base_link", source_frame)

        odom_from_source_base = self.ego_pose_matrix(source_index)
        if target_frame == "odom":
            target_from_odom = np.eye(4, dtype=np.float64)
            canonical_target_frame = "odom"
        else:
            target_base_from_odom = invert_transform(self.ego_pose_matrix(target_index))
            if target_frame == "base_link":
                target_from_target_base = np.eye(4, dtype=np.float64)
                canonical_target_frame = "base_link"
            else:
                target_sensor_key = calibration.sensor_key(target_frame)
                canonical_target_frame = calibration.sensors[target_sensor_key]["frame"]
                target_from_target_base = calibration.matrix(
                    canonical_target_frame, "base_link"
                )
            target_from_odom = target_from_target_base @ target_base_from_odom

        return (
            target_from_odom @ odom_from_source_base @ source_base_from_box,
            canonical_target_frame,
        )

    def transform_box(
        self,
        box: Box3D,
        target_index: int,
        target_frame: str = "base_link",
    ) -> Box3D:
        """Transform a time-local box into a frame at ``target_index``.

        Canonical boxes are expressed in ``base_link`` at their own timestamp.
        Sensor-specific boxes first pass through their static sensor calibration.
        The source and target ego poses then provide the temporal transform.
        """

        transform, canonical_target_frame = self._box_frame_transform(
            box.frame_index,
            box.frame,
            target_index,
            target_frame,
        )
        return box.transformed(transform, frame=canonical_target_frame)

    def object_trajectories(
        self,
        anchor_index: int,
        offsets: list[int] | tuple[int, ...],
        target_frame: str = "base_link",
        category: Optional[str] = None,
    ) -> dict[str, list[Box3D]]:
        """Return canonical object tracks transformed into one anchor frame.

        Only observations whose frame indices match ``anchor_index + offset``
        are returned. Sparse annotation remains sparse; no interpolation or
        padding is performed.
        """

        if anchor_index < 0:
            anchor_index += len(self)
        if anchor_index < 0 or anchor_index >= len(self):
            raise IndexError("anchor_index out of range")
        selected_indices = {
            anchor_index + int(offset)
            for offset in offsets
            if 0 <= anchor_index + int(offset) < len(self)
        }
        trajectories: dict[str, list[Box3D]] = {}
        transforms: dict[tuple[int, str], tuple[np.ndarray, str]] = {}
        for object_id, boxes in self.tracks(category=category).items():
            transformed = []
            for box in boxes:
                if box.frame_index not in selected_indices:
                    continue
                key = (box.frame_index, box.frame)
                if key not in transforms:
                    transforms[key] = self._box_frame_transform(
                        box.frame_index,
                        box.frame,
                        anchor_index,
                        target_frame,
                    )
                transform, canonical_target_frame = transforms[key]
                transformed.append(
                    box.transformed(transform, frame=canonical_target_frame)
                )
            if transformed:
                trajectories[object_id] = transformed
        return trajectories

    def ego_pose_matrix(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            frame_index += len(self)
        if frame_index < 0 or frame_index >= len(self):
            raise IndexError("frame_index out of range")
        sample = self._sample_at(frame_index)
        pose = sample["ego_pose"]
        return transform_matrix(pose["position"], pose["orientation_xyzw"])

    def relative_ego_pose(self, anchor_index: int, offset: int) -> np.ndarray:
        """Return the offset ego pose expressed in the anchor ``base_link`` frame."""

        sample_count = len(self)
        if anchor_index < 0:
            anchor_index += sample_count
        if anchor_index < 0 or anchor_index >= sample_count:
            raise IndexError("anchor_index out of range")
        source_index = anchor_index + int(offset)
        if source_index < 0 or source_index >= sample_count:
            raise IndexError(f"trajectory offset {offset} is outside the scene")
        return invert_transform(
            self.ego_pose_matrix(anchor_index)
        ) @ self.ego_pose_matrix(source_index)

    def ego_velocity(
        self, frame_index: int, target_frame: str = "base_link"
    ) -> np.ndarray:
        sample_count = len(self)
        if not sample_count:
            return np.full(3, np.nan, dtype=np.float64)
        if frame_index < 0:
            frame_index += sample_count
        if frame_index < 0 or frame_index >= sample_count:
            raise IndexError("frame_index out of range")
        if sample_count < 2:
            return np.full(3, np.nan, dtype=np.float64)

        prev_idx = max(0, frame_index - 1)
        next_idx = min(sample_count - 1, frame_index + 1)
        if prev_idx == next_idx:
            return np.full(3, np.nan, dtype=np.float64)
        previous = self._sample_at(prev_idx)
        following = self._sample_at(next_idx)
        p0 = np.asarray(previous["ego_pose"]["position"], dtype=np.float64)
        p1 = np.asarray(following["ego_pose"]["position"], dtype=np.float64)
        dt_ns = int(following["timestamp_ns"]) - int(previous["timestamp_ns"])
        dt = dt_ns * 1e-9
        if dt <= 0.0:
            return np.full(3, np.nan, dtype=np.float64)
        velocity_odom = (p1 - p0) / dt
        if target_frame == "odom":
            return velocity_odom
        if target_frame == "base_link":
            base_from_odom = invert_transform(self.ego_pose_matrix(frame_index))
            return base_from_odom[:3, :3] @ velocity_odom
        raise ValueError("target_frame must be 'base_link' or 'odom'")

    def ego_trajectory(
        self,
        frame_index: int,
        offsets: list[int] | tuple[int, ...],
        target_frame: str = "base_link",
        missing: str = "nan",
    ) -> np.ndarray:
        sample_count = len(self)
        if frame_index < 0:
            frame_index += sample_count
        if frame_index < 0 or frame_index >= sample_count:
            raise IndexError("frame_index out of range")
        if target_frame not in {"base_link", "odom"}:
            raise ValueError("target_frame must be 'base_link' or 'odom'")
        if missing not in {"nan", "drop", "raise"}:
            raise ValueError("missing must be 'nan', 'drop', or 'raise'")

        positions = []
        for offset in offsets:
            idx = frame_index + int(offset)
            if 0 <= idx < sample_count:
                positions.append(self._sample_at(idx)["ego_pose"]["position"])
            elif missing == "raise":
                raise IndexError(f"trajectory offset {offset} is outside the scene")
            elif missing == "nan":
                positions.append([np.nan, np.nan, np.nan])
        points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if target_frame == "odom" or len(points) == 0:
            return points
        finite = np.isfinite(points).all(axis=1)
        out = np.full_like(points, np.nan, dtype=np.float64)
        out[finite] = transform_points(
            points[finite], invert_transform(self.ego_pose_matrix(frame_index))
        )
        return out

    def accumulate_lidar(
        self,
        sensor_id: str,
        stage: str = "motion_compensated",
        frame_indices: Optional[list[int]] = None,
        step: int = 1,
        max_frames: Optional[int] = None,
        stride: int = 1,
        target_frame: str = "odom",
        fields="all",
        missing: str = "nan",
    ) -> AccumulatedPointCloud:
        validate_lidar_stage(stage)
        if target_frame not in {"odom", "base_link"}:
            raise ValueError("target_frame must be 'odom' or 'base_link'")
        indices = (
            frame_indices
            if frame_indices is not None
            else list(range(0, len(self), step))
        )
        all_points = []
        accumulated_fields: dict[str, list[np.ndarray]] = {}
        requested_fields: Optional[tuple[str, ...]] = None
        all_frame_indices = []
        all_timestamps = []
        used = 0
        for local_idx in indices:
            if max_frames is not None and used >= max_frames:
                break
            frame = self[local_idx]
            if not frame.has_lidar(sensor_id, stage=stage):
                continue
            cloud = frame.load_lidar(sensor_id, stage=stage)
            if requested_fields is None:
                requested_fields = (
                    cloud.field_names if fields == "all" else tuple(fields)
                )
            points = cloud.xyz[::stride]
            points_base = transform_points(
                points, frame.calibration().base_from_sensor(f"lidar/{sensor_id}")
            )
            if target_frame == "odom":
                points_out = transform_points(points_base, frame.load_ego_pose())
            else:
                points_out = points_base
            for field_name in requested_fields:
                if field_name == "x":
                    values = points_out[:, 0]
                elif field_name == "y":
                    values = points_out[:, 1]
                elif field_name == "z":
                    values = points_out[:, 2]
                else:
                    values = cloud.to_numpy([field_name], missing=missing)[:, 0][
                        ::stride
                    ]
                accumulated_fields.setdefault(field_name, []).append(values)
            all_points.append(points_out)
            all_frame_indices.append(
                np.full(len(points_out), frame.frame_index, dtype=np.int32)
            )
            all_timestamps.append(
                np.full(len(points_out), frame.timestamp_ns, dtype=np.int64)
            )
            used += 1
        if not all_points:
            return AccumulatedPointCloud(
                xyz=np.empty((0, 3), dtype=np.float64),
                fields={},
                frame_index=np.empty((0,), dtype=np.int32),
                timestamp_ns=np.empty((0,), dtype=np.int64),
                sensor_id=sensor_id,
                stage=stage,
                frame=target_frame,
            )
        return AccumulatedPointCloud(
            xyz=np.vstack(all_points),
            fields={
                name: np.concatenate(values)
                for name, values in accumulated_fields.items()
            },
            frame_index=np.concatenate(all_frame_indices),
            timestamp_ns=np.concatenate(all_timestamps),
            sensor_id=sensor_id,
            stage=stage,
            frame=target_frame,
        )

    @staticmethod
    def _column_key(columns) -> Optional[tuple[str, ...]]:
        return None if columns is None else tuple(columns)

    def parquet(self, relative_path: str, columns=None):
        path = resolve_scene_path(self.path, relative_path)
        relative_path = Path(relative_path).as_posix()
        column_key = self._column_key(columns)
        cache_key = (relative_path, column_key)
        if cache_key not in self._parquet_cache:
            self._parquet_cache[cache_key] = pd.read_parquet(
                path,
                columns=column_key,
            )
        return self._parquet_cache[cache_key]

    def _parquet_columns(self, relative_path: str) -> frozenset[str]:
        relative_path = Path(relative_path).as_posix()
        if relative_path not in self._parquet_columns_cache:
            import pyarrow.parquet as pq

            path = resolve_scene_path(self.path, relative_path)
            self._parquet_columns_cache[relative_path] = frozenset(
                pq.ParquetFile(path).schema_arrow.names
            )
        return self._parquet_columns_cache[relative_path]

    def load_vehicle_signals(self, columns=None):
        """Load vehicle signals, normalizing known optional columns to nulls.

        ``timestamp_ns`` is required. Known optional columns requested by a
        consumer are returned as all-null columns when the source scene does
        not provide them, so per-scene Parquet schema differences do not break
        a uniform metadata workflow. Unknown requested column names keep the
        native Parquet error behavior.
        """

        column_key = self._column_key(columns)

        if column_key in self._vehicle_signals_bulk_cache:
            return self._vehicle_signals_bulk_cache[column_key]

        relative_path = VEHICLE_SIGNALS_PATH
        absolute_path = resolve_scene_path(self.path, relative_path)

        if not absolute_path.is_file():
            raise FZIAURAError(
                f"No vehicle signals found for scene {self.scene_id}: {absolute_path}"
            )

        if column_key is None:
            signals = self.parquet(relative_path)
            missing_optional = OPTIONAL_VEHICLE_SIGNAL_COLUMNS.difference(
                signals.columns
            )
        else:
            requested_optional = set(column_key).intersection(
                OPTIONAL_VEHICLE_SIGNAL_COLUMNS
            )
            if not requested_optional:
                signals = self.parquet(relative_path, columns=column_key)
                missing_optional = set()
            else:
                available_columns = self._parquet_columns(relative_path)
                missing_optional = requested_optional.difference(available_columns)
                missing_unknown = set(column_key).difference(
                    available_columns, OPTIONAL_VEHICLE_SIGNAL_COLUMNS
                )
                if missing_unknown:
                    # Preserve the native Parquet error for misspelled or
                    # unsupported field names instead of silently returning
                    # nulls for them.
                    signals = self.parquet(relative_path, columns=column_key)
                else:
                    selected_columns = tuple(
                        dict.fromkeys(
                            column
                            for column in column_key
                            if column in available_columns
                        )
                    )
                    if not selected_columns:
                        selected_columns = ("timestamp_ns",)
                    signals = self.parquet(relative_path, columns=selected_columns)

        if missing_optional:
            signals = signals.copy()
            for column in missing_optional:
                signals[column] = np.nan
            if column_key is not None:
                signals = signals.reindex(columns=column_key)
        self._vehicle_signals_bulk_cache[column_key] = signals
        return signals

    def _vehicle_signal_lookup(self, df) -> dict[int, int]:
        if self._vehicle_signal_timestamp_lookup is None:
            lookup: dict[int, int] = {}
            for position, value in enumerate(df["timestamp_ns"]):
                timestamp_ns = int(value)
                if timestamp_ns in lookup:
                    raise FZIAURAError(
                        f"Duplicate vehicle signal timestamp_ns={timestamp_ns} "
                        f"in scene {self.scene_id}."
                    )
                lookup[timestamp_ns] = position
            self._vehicle_signal_timestamp_lookup = lookup
        return self._vehicle_signal_timestamp_lookup

    def vehicle_signal_row(self, timestamp_ns: int):
        df = self.load_vehicle_signals()
        if "timestamp_ns" not in df.columns:
            raise FZIAURAError(
                f"Vehicle signals for scene {self.scene_id} have no timestamp_ns column."
            )
        position = self._vehicle_signal_lookup(df).get(int(timestamp_ns))
        if position is None:
            raise FZIAURAError(
                f"Vehicle signals for scene {self.scene_id} have no row "
                f"for timestamp_ns={int(timestamp_ns)}."
            )
        return df.iloc[position]

    def stats(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "name": self.name,
            "counts": self.counts,
            "sample_count": len(self),
        }

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "scene_id": self.scene_id,
            "name": self.name,
            "path": str(self.path),
            "counts": self.counts,
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "duration_s": self.duration_s,
            "sample_count": self.sample_count,
        }
