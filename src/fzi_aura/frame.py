from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .availability import DatasetAvailability, is_keyframe_sample
from .boxes import Box3D, BoxCollection
from .calibration import Calibration
from .contract import validate_lidar_stage
from .errors import AnnotationNotFoundError, FZIAURAError, SensorNotFoundError
from .geometry import project_points, transform_matrix, transform_points
from .io import resolve_scene_path
from .pointcloud import read_pcd_binary, read_pcd_header
from .semantics import FusedSemanticPointCloud, SemanticLabels, read_semantic_label


class FZIAURAFrame:
    def __init__(
        self,
        *,
        index: int,
        scene_id: str,
        scene_name: str,
        scene_path: Path,
        scene_index: int,
        frame_index: int,
        timestamp_ns: int,
        sample_dict: dict[str, Any],
        scene=None,
        availability: Optional[DatasetAvailability] = None,
    ):
        self.index = index
        self.scene_id = scene_id
        self.scene_name = scene_name
        self.scene_path = Path(scene_path)
        self.scene_index = scene_index
        self.frame_index = frame_index
        self.timestamp_ns = timestamp_ns
        self.sample_dict = sample_dict
        self._scene = scene
        self.availability = availability or DatasetAvailability.from_root(
            self.scene_path.parent.parent
        )

    @property
    def is_keyframe(self) -> bool:
        return is_keyframe_sample(self.sample_dict)

    @classmethod
    def from_sample(cls, sample: dict[str, Any], **kwargs):
        kwargs["frame_index"] = int(sample["frame_index"])
        return cls(
            timestamp_ns=int(sample["timestamp_ns"]), sample_dict=sample, **kwargs
        )

    @property
    def has_boxes_3d(self) -> bool:
        return bool(self.sample_dict["labels"].get("boxes_3d", False))

    @property
    def boxes_3d_count(self) -> Optional[int]:
        labels = self.sample_dict["labels"]
        if labels.get("boxes_3d") is not True:
            return None
        return int(labels["boxes_3d_count"])

    @property
    def semantic_lidar_sensors(self) -> tuple[str, ...]:
        if not self.availability.allows_base():
            return ()
        return tuple(
            self.sample_dict.get("labels", {}).get("semantic_lidar", {}).keys()
        )

    @property
    def annotation_types(self) -> tuple[str, ...]:
        types = []
        if self.has_boxes_3d:
            types.append("boxes_3d")
        if self.semantic_lidar_sensors:
            types.append("semantic_lidar")
        return tuple(types)

    def available_cameras(self) -> tuple[str, ...]:
        if not self.availability.allows_sensor("camera", is_keyframe=self.is_keyframe):
            return ()
        return tuple(self.sample_dict.get("cameras", {}).keys())

    def available_lidars(self, stage: str = "motion_compensated") -> tuple[str, ...]:
        validate_lidar_stage(stage)
        if not self.availability.allows_sensor(
            f"lidar_{stage}", is_keyframe=self.is_keyframe
        ):
            return ()
        return tuple(self.sample_dict["lidar"][stage].keys())

    def available_radars(self) -> tuple[str, ...]:
        if not self.availability.allows_sensor("radar", is_keyframe=self.is_keyframe):
            return ()
        return tuple(self.sample_dict.get("radar", {}).keys())

    def has_camera(self, sensor_id: str) -> bool:
        return sensor_id in self.available_cameras()

    def has_lidar(self, sensor_id: str, stage: str = "motion_compensated") -> bool:
        return sensor_id in self.available_lidars(stage)

    def has_radar(self, sensor_id: str) -> bool:
        return sensor_id in self.available_radars()

    def has_semantics(self, sensor_id: str) -> bool:
        return sensor_id in self.semantic_lidar_sensors

    def camera_path(self, sensor_id: str) -> Path:
        if sensor_id not in self.available_cameras():
            raise SensorNotFoundError(
                f"Camera {sensor_id!r} is not available in the installed layers for "
                f"scene {self.scene_id} frame {self.frame_index}."
            )
        try:
            return resolve_scene_path(
                self.scene_path, self.sample_dict["cameras"][sensor_id]
            )
        except KeyError as exc:
            raise SensorNotFoundError(
                f"Camera {sensor_id!r} not available for scene {self.scene_id} frame {self.frame_index}."
            ) from exc

    def lidar_path(self, sensor_id: str, stage: str = "motion_compensated") -> Path:
        validate_lidar_stage(stage)
        if sensor_id not in self.available_lidars(stage):
            raise SensorNotFoundError(
                f"Lidar {sensor_id!r} stage {stage!r} is not available in the installed "
                f"layers for scene {self.scene_id} frame {self.frame_index}."
            )
        try:
            return resolve_scene_path(
                self.scene_path, self.sample_dict["lidar"][stage][sensor_id]
            )
        except KeyError as exc:
            raise SensorNotFoundError(
                f"Lidar {sensor_id!r} stage {stage!r} not available for scene {self.scene_id} frame {self.frame_index}."
            ) from exc

    def radar_path(self, sensor_id: str) -> Path:
        if sensor_id not in self.available_radars():
            raise SensorNotFoundError(
                f"Radar {sensor_id!r} is not available in the installed layers for "
                f"scene {self.scene_id} frame {self.frame_index}."
            )
        try:
            return resolve_scene_path(
                self.scene_path, self.sample_dict["radar"][sensor_id]
            )
        except KeyError as exc:
            raise SensorNotFoundError(
                f"Radar {sensor_id!r} not available for scene {self.scene_id} frame {self.frame_index}."
            ) from exc

    def semantic_path(self, sensor_id: str) -> Path:
        if sensor_id not in self.semantic_lidar_sensors:
            raise AnnotationNotFoundError(
                f"Semantic labels for sensor {sensor_id!r} are not available in the "
                f"installed layers for scene {self.scene_id} frame {self.frame_index}."
            )
        try:
            return resolve_scene_path(
                self.scene_path, self.sample_dict["labels"]["semantic_lidar"][sensor_id]
            )
        except KeyError as exc:
            raise AnnotationNotFoundError(
                f"Semantic labels for sensor {sensor_id!r} not available for scene {self.scene_id} frame {self.frame_index}."
            ) from exc

    def load_camera(self, sensor_id: str, mode: str = "rgb") -> np.ndarray:
        image: Image.Image = Image.open(self.camera_path(sensor_id))
        if mode.lower() == "rgb":
            image = image.convert("RGB")
        elif mode:
            image = image.convert(mode.upper())
        return np.asarray(image)

    def load_lidar(
        self, sensor_id: str, stage: str = "motion_compensated", fields="all"
    ):
        path = self.lidar_path(sensor_id, stage)
        if stage == "motion_compensated" and "aeva" not in sensor_id.lower():
            if fields == "all":
                fields = tuple(
                    name for name in read_pcd_header(path).fields if name != "velocity"
                )
            else:
                fields = tuple(name for name in fields if name != "velocity")
        return read_pcd_binary(
            path,
            fields=fields,
            modality="lidar",
            sensor_id=sensor_id,
            stage=stage,
            timestamp_ns=self.timestamp_ns,
        )

    def load_radar(self, sensor_id: str, fields="all"):
        return read_pcd_binary(
            self.radar_path(sensor_id),
            fields=fields,
            modality="radar",
            sensor_id=sensor_id,
            stage=None,
            timestamp_ns=self.timestamp_ns,
        )

    def load_boxes(
        self,
        frame: Optional[str] = None,
        sensor_id: Optional[str] = None,
        strict: bool = False,
    ) -> BoxCollection:
        scene = self._require_scene()
        frame, sensor_id = scene._resolve_box_request(frame, sensor_id)
        if not self.has_boxes_3d:
            if strict:
                raise AnnotationNotFoundError(
                    f"No 3D boxes for scene {self.scene_id} frame {self.frame_index}."
                )
            return BoxCollection([], frame, self.timestamp_ns)

        expected_count = None
        if sensor_id is None:
            expected_count = self.boxes_3d_count
            boxes_path = scene._box_path(None)
        else:
            sensor_sources = self.sample_dict.get("labels", {}).get(
                "boxes_3d_sensor_frame", {}
            )
            if not isinstance(sensor_sources, dict):
                raise FZIAURAError(
                    f"Scene {self.scene_id} frame {self.frame_index} has invalid "
                    "labels.boxes_3d_sensor_frame metadata."
                )
            relative = sensor_sources.get(sensor_id)
            if relative is None:
                if strict:
                    raise AnnotationNotFoundError(
                        f"No sensor-frame 3D boxes for sensor {sensor_id!r} in "
                        f"scene {self.scene_id} frame {self.frame_index}."
                    )
                return BoxCollection([], frame, self.timestamp_ns)
            boxes_path = resolve_scene_path(self.scene_path, relative)

        if not boxes_path.is_file():
            if sensor_id is None:
                description = "base-link"
            else:
                description = f"sensor {sensor_id!r}"
            raise AnnotationNotFoundError(
                f"Declared {description} 3D boxes are missing for scene "
                f"{self.scene_id} frame {self.frame_index}: {boxes_path}"
            )

        boxes = scene._boxes_at_path(
            boxes_path,
            self.timestamp_ns,
            frame=frame,
            sensor_id=sensor_id,
            expected_frame_index=self.frame_index,
        )
        if expected_count is not None and len(boxes) != expected_count:
            raise FZIAURAError(
                f"Scene {self.scene_id} frame {self.frame_index} declares "
                f"boxes_3d_count={expected_count}, but {boxes_path} contains "
                f"{len(boxes)} matching boxes at timestamp_ns={self.timestamp_ns}."
            )
        return boxes

    def load_semantics(self, sensor_id: str) -> SemanticLabels:
        path = self.semantic_path(sensor_id)
        semantic, instance = read_semantic_label(path)
        classes_path = self.scene_path / "labels" / "semantic" / "classes.json"
        if not classes_path.is_file():
            raise FZIAURAError(
                f"Semantic labels exist for scene {self.scene_id}, but the mandatory "
                f"class file is missing: {classes_path}"
            )
        raw = json.loads(classes_path.read_text(encoding="utf-8"))
        semantic_classes = (
            raw.get("semantic_classes") if isinstance(raw, dict) else None
        )
        if not isinstance(semantic_classes, list):
            raise FZIAURAError(
                f"{classes_path} must contain the exporter semantic_classes structure."
            )
        try:
            class_names = {int(row["id"]): str(row["name"]) for row in semantic_classes}
        except (KeyError, TypeError, ValueError) as exc:
            raise FZIAURAError(
                f"{classes_path} has invalid semantic_classes entries."
            ) from exc
        return SemanticLabels(
            semantic, instance, path, sensor_id, self.timestamp_ns, class_names
        )

    def load_lidar_semantic_pair(
        self, sensor_id: str, fields="all", stage: str = "motion_compensated"
    ):
        cloud = self.load_lidar(sensor_id, stage=stage, fields=fields)
        labels = self.load_semantics(sensor_id)
        if len(cloud) != len(labels):
            raise AnnotationNotFoundError(
                f"Semantic label length {len(labels)} does not match {stage} lidar length {len(cloud)} "
                f"for scene {self.scene_id} frame {self.frame_index} sensor {sensor_id}."
            )
        return cloud, labels

    def lidar_points_in_frame(
        self,
        sensor_id: str,
        target_frame: str = "base_link",
        stage: str = "motion_compensated",
        stride: int = 1,
    ) -> np.ndarray:
        cloud = self.load_lidar(sensor_id, stage=stage)
        points = cloud.xyz[::stride]
        base_from_lidar = self.calibration().base_from_sensor(f"lidar/{sensor_id}")
        points_base = transform_points(points, base_from_lidar)
        if target_frame == "base_link":
            return points_base
        if target_frame == "odom":
            return transform_points(points_base, self.load_ego_pose())
        raise ValueError("target_frame must be 'base_link' or 'odom'")

    def project_lidar_to_camera(
        self,
        lidar_id: str,
        camera_id: str,
        stage: str = "motion_compensated",
        stride: int = 1,
        min_depth: float = 0.1,
    ) -> dict:
        cloud = self.load_lidar(lidar_id, stage=stage)
        points = cloud.xyz[::stride]
        calib = self.calibration()
        camera_from_lidar = calib.sensor_from_base(
            f"camera/{camera_id}"
        ) @ calib.base_from_sensor(f"lidar/{lidar_id}")
        points_cam = transform_points(points, camera_from_lidar)
        in_front = points_cam[:, 2] > min_depth
        uv, depth = project_points(
            points_cam[in_front], calib.camera_projection_matrix(camera_id)
        )
        point_indices = np.flatnonzero(in_front) * stride
        return {
            "uv": uv,
            "depth": depth,
            "point_indices": point_indices,
            "points_cam": points_cam[in_front],
            "cloud": cloud,
        }

    def project_lidar_semantics_to_camera(
        self,
        lidar_id: str,
        camera_id: str,
        stage: str = "motion_compensated",
        stride: int = 1,
        image_shape: Optional[tuple[int, ...]] = None,
        min_depth: float = 0.1,
    ) -> dict:
        from .semantics import semantic_colors

        cloud, labels = self.load_lidar_semantic_pair(lidar_id, stage=stage)
        projection = self.project_lidar_to_camera(
            lidar_id, camera_id, stage=stage, stride=stride, min_depth=min_depth
        )
        semantic_id = labels.semantic_id[projection["point_indices"]]
        instance_id = labels.instance_id[projection["point_indices"]]
        inside = np.ones(len(projection["uv"]), dtype=bool)
        if image_shape is not None:
            height, width = image_shape[:2]
            uv = projection["uv"]
            inside = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < height)
            )
        return {
            "uv": projection["uv"][inside],
            "depth": projection["depth"][inside],
            "semantic_id": semantic_id[inside],
            "instance_id": instance_id[inside],
            "colors": semantic_colors(semantic_id[inside]),
            "point_indices": projection["point_indices"][inside],
            "cloud": cloud,
            "labels": labels,
        }

    def project_boxes_to_camera(
        self,
        camera_id: str,
        boxes: Optional[BoxCollection] = None,
        frame: Optional[str] = None,
        min_depth: float = 0.1,
        sensor_id: Optional[str] = None,
    ) -> list[dict]:
        if boxes is not None and sensor_id is not None:
            raise ValueError("sensor_id cannot be set when boxes are provided")
        boxes = (
            boxes
            if boxes is not None
            else self.load_boxes(frame=frame, sensor_id=sensor_id)
        )
        calibration = self.calibration()
        projection_matrix = calibration.camera_projection_matrix(camera_id)
        camera_frame = calibration.sensor(f"camera/{camera_id}")["frame"]
        camera_from_frame: dict[str, np.ndarray] = {}
        projected = []
        for box in boxes:
            if box.frame not in camera_from_frame:
                camera_from_frame[box.frame] = calibration.matrix(
                    camera_frame, box.frame
                )
            corners_cam = transform_points(box.corners(), camera_from_frame[box.frame])
            uv, depth = project_points(corners_cam, projection_matrix)
            projected.append(
                {
                    "box": box,
                    "uv": uv,
                    "depth": depth,
                    "visible_edges": [
                        (start, end)
                        for start, end in BOX_EDGES
                        if depth[start] > min_depth and depth[end] > min_depth
                    ],
                }
            )
        return projected

    def fuse_lidar_semantics(
        self,
        sensor_ids: Optional[list[str]] = None,
        target_frame: str = "base_link",
        stage: str = "motion_compensated",
        stride: int = 1,
    ) -> FusedSemanticPointCloud:
        sensor_ids = list(sensor_ids or self.semantic_lidar_sensors)
        all_points = []
        all_semantic = []
        all_instance = []
        all_sensors = []
        calib = self.calibration()
        for sensor_id in sensor_ids:
            if not self.has_semantics(sensor_id):
                continue
            cloud, labels = self.load_lidar_semantic_pair(sensor_id, stage=stage)
            points = cloud.xyz[::stride]
            semantic_id = labels.semantic_id[::stride]
            instance_id = labels.instance_id[::stride]
            points_base = transform_points(
                points, calib.base_from_sensor(f"lidar/{sensor_id}")
            )
            if target_frame == "base_link":
                points_out = points_base
            elif target_frame == "odom":
                points_out = transform_points(points_base, self.load_ego_pose())
            else:
                raise ValueError("target_frame must be 'base_link' or 'odom'")
            all_points.append(points_out)
            all_semantic.append(semantic_id)
            all_instance.append(instance_id)
            all_sensors.extend([sensor_id] * len(points_out))
        if not all_points:
            return FusedSemanticPointCloud(
                xyz=np.empty((0, 3), dtype=np.float64),
                semantic_id=np.empty((0,), dtype=np.uint16),
                instance_id=np.empty((0,), dtype=np.uint16),
                sensor_id=np.empty((0,), dtype=object),
                frame=target_frame,
                stage=stage,
                timestamp_ns=self.timestamp_ns,
            )
        return FusedSemanticPointCloud(
            xyz=np.vstack(all_points),
            semantic_id=np.concatenate(all_semantic),
            instance_id=np.concatenate(all_instance),
            sensor_id=np.asarray(all_sensors),
            frame=target_frame,
            stage=stage,
            timestamp_ns=self.timestamp_ns,
        )

    def load_ego_pose(self) -> np.ndarray:
        pose = self.sample_dict["ego_pose"]
        return transform_matrix(pose["position"], pose["orientation_xyzw"])

    def ego_velocity(self, target_frame: str = "base_link") -> np.ndarray:
        return self._require_scene().ego_velocity(self.frame_index, target_frame)

    def relative_ego_pose(self, offset: int) -> np.ndarray:
        """Return an offset ego pose expressed in this frame's ``base_link``."""

        return self._require_scene().relative_ego_pose(self.index, offset)

    def ego_trajectory(
        self,
        offsets: list[int] | tuple[int, ...],
        target_frame: str = "base_link",
        missing: str = "nan",
    ) -> np.ndarray:
        return self._require_scene().ego_trajectory(
            self.frame_index,
            offsets,
            target_frame=target_frame,
            missing=missing,
        )

    def transform_box(self, box: Box3D, target_frame: str = "base_link") -> Box3D:
        """Transform ``box`` from its timestamp into this frame's coordinates."""

        return self._require_scene().transform_box(
            box,
            self.index,
            target_frame=target_frame,
        )

    def object_trajectories(
        self,
        offsets: list[int] | tuple[int, ...],
        target_frame: str = "base_link",
        category: Optional[str] = None,
    ) -> dict[str, list[Box3D]]:
        """Return sparse canonical object tracks expressed at this frame."""

        return self._require_scene().object_trajectories(
            self.index,
            offsets,
            target_frame=target_frame,
            category=category,
        )

    def load_vehicle_signals_row(self):
        vehicle = self.sample_dict["vehicle_signals"]
        timestamp = int(vehicle["timestamp_ns"])
        return self._require_scene().vehicle_signal_row(timestamp)

    def calibration(self) -> Calibration:
        if self._scene is not None:
            return self._scene.calibration()
        return Calibration(self.scene_path / "calibration.json")

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "scene_id": self.scene_id,
            "scene_name": self.scene_name,
            "scene_path": str(self.scene_path),
            "scene_index": self.scene_index,
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "has_boxes_3d": self.has_boxes_3d,
            "semantic_lidar_sensors": list(self.semantic_lidar_sensors),
            "camera_sensors": list(self.available_cameras()),
            "lidar_raw_sensors": list(self.available_lidars("raw")),
            "lidar_motion_compensated_sensors": list(
                self.available_lidars("motion_compensated")
            ),
            "radar_sensors": list(self.available_radars()),
        }

    def _require_scene(self):
        if self._scene is None:
            from .scene import FZIAURAScene

            metadata = {"scene_id": self.scene_id, "path": str(self.scene_path)}
            self._scene = FZIAURAScene(
                self.scene_path.parent.parent,
                self.scene_index,
                self.scene_id,
                self.scene_name,
                self.scene_path,
                metadata,
            )
        return self._scene


BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
