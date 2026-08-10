from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import (
    box_corners_lwh,
    rotation_matrix_to_quaternion_xyzw,
    transform_matrix,
    transform_points,
)

DETECTION_CLASSES: tuple[str, ...] = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "on rails",
    "motorcycle",
    "bicycle",
    "portable",
    "caravan",
    "trailer",
    "dynamic",
    "bicyclist",
    "motorcyclist",
    "portable-rider",
)

DETECTION_COLORS_UINT8: dict[str, tuple[int, int, int]] = {
    "person": (128, 64, 128),
    "rider": (244, 35, 232),
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "bus": (0, 60, 100),
    "on rails": (0, 0, 230),
    "motorcycle": (119, 11, 32),
    "bicycle": (220, 20, 60),
    "portable": (255, 0, 0),
    "caravan": (81, 0, 81),
    "trailer": (111, 74, 0),
    "dynamic": (250, 170, 30),
    "bicyclist": (107, 142, 35),
    "motorcyclist": (152, 251, 152),
    "portable-rider": (255, 255, 255),
}


@dataclass
class Box3D:
    center: np.ndarray
    size_lwh: np.ndarray
    rotation_xyzw: np.ndarray
    category: str
    object_id: str
    timestamp_ns: int
    frame_index: int
    frame: str
    sensor_id: Optional[str] = None

    @property
    def yaw(self) -> float:
        x, y, z, w = self.rotation_xyzw
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def corners(self) -> np.ndarray:
        local_corners = box_corners_lwh([0.0, 0.0, 0.0], self.size_lwh)
        transform = transform_matrix(self.center, self.rotation_xyzw)
        homog = np.hstack(
            [local_corners, np.ones((len(local_corners), 1), dtype=np.float64)]
        )
        return (transform @ homog.T).T[:, :3]

    def transformed(self, transform, frame: str) -> "Box3D":
        """Return this box transformed into ``frame`` by a rigid 4x4 matrix."""

        matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        source_rotation = transform_matrix([0.0, 0.0, 0.0], self.rotation_xyzw)[:3, :3]
        target_rotation = matrix[:3, :3] @ source_rotation
        return Box3D(
            center=transform_points(self.center.reshape(1, 3), matrix)[0],
            size_lwh=self.size_lwh.copy(),
            rotation_xyzw=rotation_matrix_to_quaternion_xyzw(target_rotation),
            category=self.category,
            object_id=self.object_id,
            timestamp_ns=self.timestamp_ns,
            frame_index=self.frame_index,
            frame=frame,
            sensor_id=self.sensor_id,
        )


@dataclass
class BoxCollection:
    boxes: list[Box3D]
    frame: str
    timestamp_ns: Optional[int]

    def __len__(self) -> int:
        return len(self.boxes)

    def __iter__(self):
        return iter(self.boxes)

    def by_category(self, category: str) -> "BoxCollection":
        return BoxCollection(
            [box for box in self.boxes if box.category == category],
            self.frame,
            self.timestamp_ns,
        )

    def as_numpy(
        self,
        include_yaw: bool = True,
        include_category: bool = True,
        include_object_id: bool = True,
    ) -> dict[str, np.ndarray]:
        data: dict[str, np.ndarray] = {
            "center": np.asarray(
                [box.center for box in self.boxes], dtype=np.float64
            ).reshape(-1, 3),
            "size_lwh": np.asarray(
                [box.size_lwh for box in self.boxes], dtype=np.float64
            ).reshape(-1, 3),
            "rotation_xyzw": np.asarray(
                [box.rotation_xyzw for box in self.boxes], dtype=np.float64
            ).reshape(-1, 4),
        }
        if include_yaw:
            data["yaw"] = np.asarray([box.yaw for box in self.boxes], dtype=np.float64)
        if include_category:
            data["category"] = np.asarray(
                [box.category for box in self.boxes], dtype=object
            )
        if include_object_id:
            data["object_id"] = np.asarray(
                [box.object_id for box in self.boxes], dtype=object
            )
        return data


def box_from_row(row: dict, sensor_id: Optional[str] = None) -> Box3D:
    box = row["box"]
    return Box3D(
        center=np.asarray(box["center"], dtype=np.float64),
        size_lwh=np.asarray(box["size_lwh"], dtype=np.float64),
        rotation_xyzw=np.asarray(box["rotation_xyzw"], dtype=np.float64),
        category=row.get("category", ""),
        object_id=row.get("object_id", ""),
        timestamp_ns=int(row["timestamp_ns"]),
        frame_index=int(row["frame_index"]),
        frame=box.get("frame", "base_link"),
        sensor_id=sensor_id,
    )
