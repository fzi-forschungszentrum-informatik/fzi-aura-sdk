from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from .errors import CalibrationError
from .geometry import invert_transform, project_points, transform_matrix
from .io import load_json


class Calibration:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = load_json(self.path)
        self.sensors = self.data.get("sensors", {})
        self._graph: Optional[dict[str, list[tuple[str, np.ndarray]]]] = None
        self._aliases: Optional[dict[str, str]] = None

    def sensor_ids(self, modality: Optional[str] = None) -> tuple[str, ...]:
        ids: list[str] = []
        for key, sensor in self.sensors.items():
            if modality is None or sensor.get("modality") == modality:
                ids.append(sensor.get("sensor_id", key.split("/", 1)[-1]))
        return tuple(ids)

    def sensor_key(self, sensor_ref: str, modality: Optional[str] = None) -> str:
        if sensor_ref in self.sensors:
            if modality is None or self.sensors[sensor_ref].get("modality") == modality:
                return sensor_ref
            raise CalibrationError(
                f"Sensor {sensor_ref!r} is not modality {modality!r} in {self.path}"
            )
        if modality is not None and "/" not in sensor_ref:
            candidate = f"{modality}/{sensor_ref}"
            if candidate in self.sensors:
                return candidate
        matches = [
            key
            for key, sensor in self.sensors.items()
            if sensor.get("sensor_id") == sensor_ref
            and (modality is None or sensor.get("modality") == modality)
        ]
        if len(matches) == 1:
            return matches[0]
        frame_matches = [
            key
            for key, sensor in self.sensors.items()
            if sensor.get("frame") == sensor_ref
            and (modality is None or sensor.get("modality") == modality)
        ]
        if len(frame_matches) == 1:
            return frame_matches[0]
        if len(matches) + len(frame_matches) > 1:
            raise CalibrationError(
                f"Sensor reference {sensor_ref!r} is ambiguous in calibration {self.path}"
            )
        raise CalibrationError(
            f"Sensor {sensor_ref!r} not found in calibration {self.path}"
        )

    def _key(self, sensor_key: str) -> str:
        return self.sensor_key(sensor_key)

    def sensor(self, sensor_key: str) -> dict:
        return self.sensors[self._key(sensor_key)]

    def camera_projection_matrix(self, camera_id: str) -> np.ndarray:
        sensor = self.sensor(
            camera_id if camera_id.startswith("camera/") else f"camera/{camera_id}"
        )
        try:
            return np.asarray(sensor["intrinsics"]["P"], dtype=np.float64).reshape(3, 4)
        except KeyError as exc:
            raise CalibrationError(
                f"Camera {camera_id!r} has no projection matrix in {self.path}"
            ) from exc

    def base_from_sensor(self, sensor_key: str) -> np.ndarray:
        sensor = self.sensor(sensor_key)
        try:
            tf = sensor["transform_base_link_from_sensor"]
            translation = tf["translation_xyz"]
            rotation = tf["rotation_xyzw"]
        except (KeyError, TypeError) as exc:
            raise CalibrationError(
                f"Sensor {sensor_key!r} has no transform_base_link_from_sensor in {self.path}"
            ) from exc
        return transform_matrix(translation, rotation)

    def sensor_from_base(self, sensor_key: str) -> np.ndarray:
        return self.matrix(sensor_key, "base_link")

    def matrix(self, dst_frame: str, src_frame: str) -> np.ndarray:
        """Return the 4x4 transform that maps points from src_frame into dst_frame."""
        src = self._resolve_frame(src_frame)
        dst = self._resolve_frame(dst_frame)
        if src == dst:
            return np.eye(4, dtype=np.float64)
        graph = self._frame_graph()
        queue = deque([(src, np.eye(4, dtype=np.float64))])
        seen = {src}
        while queue:
            node, node_from_src = queue.popleft()
            for next_node, next_from_node in graph.get(node, ()):
                if next_node in seen:
                    continue
                next_from_src = next_from_node @ node_from_src
                if next_node == dst:
                    return next_from_src
                seen.add(next_node)
                queue.append((next_node, next_from_src))
        raise CalibrationError(
            f"No calibration transform path from {src_frame!r} to {dst_frame!r} in {self.path}"
        )

    def transform(self, points, dst_frame: str, src_frame: str) -> np.ndarray:
        return self._transform_points_with_matrix(
            points, self.matrix(dst_frame, src_frame)
        )

    def transform_points(
        self,
        points,
        *,
        dst_frame: str,
        src_frame: str,
    ) -> np.ndarray:
        return self._transform_points_with_matrix(
            points,
            self.matrix(dst_frame, src_frame),
        )

    def _frame_graph(self) -> dict[str, list[tuple[str, np.ndarray]]]:
        if self._graph is not None:
            return self._graph

        graph: dict[str, list[tuple[str, np.ndarray]]] = {}
        aliases: dict[str, str] = {"base_link": "base_link"}

        def add_alias(alias: Optional[str], node: str) -> None:
            if not alias:
                return
            existing = aliases.get(alias)
            if existing is None:
                aliases[alias] = node
            elif existing != node:
                aliases[alias] = ""

        for key, sensor in self.sensors.items():
            child = sensor["frame"]
            base_from_sensor = self._matrix_from_transform(
                sensor["transform_base_link_from_sensor"]
            )
            graph.setdefault(child, []).append(("base_link", base_from_sensor))
            graph.setdefault("base_link", []).append(
                (child, invert_transform(base_from_sensor))
            )

            add_alias(key, child)
            add_alias(child, child)
            add_alias(sensor["sensor_id"], child)
            add_alias(f"{sensor['modality']}/{sensor['sensor_id']}", child)

        self._graph = graph
        self._aliases = aliases
        return graph

    def _resolve_frame(self, frame: str) -> str:
        self._frame_graph()
        assert self._aliases is not None
        if frame in self._aliases:
            resolved = self._aliases[frame]
            if resolved:
                return resolved
            raise CalibrationError(
                f"Frame reference {frame!r} is ambiguous in calibration {self.path}"
            )
        raise CalibrationError(f"Frame {frame!r} not found in calibration {self.path}")

    @staticmethod
    def _matrix_from_transform(transform: dict) -> np.ndarray:
        return transform_matrix(
            transform["translation_xyz"], transform["rotation_xyzw"]
        )

    @staticmethod
    def _transform_points_with_matrix(points, transform) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        homog = np.hstack([points[:, :3], np.ones((len(points), 1), dtype=np.float64)])
        return (transform @ homog.T).T[:, :3]

    def project_points(
        self, points_cam, camera_id: str
    ) -> tuple[np.ndarray, np.ndarray]:
        return project_points(points_cam, self.camera_projection_matrix(camera_id))
