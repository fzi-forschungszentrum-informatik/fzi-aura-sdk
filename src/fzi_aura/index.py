from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FZIAURAIndexRow:
    index: int
    scene_id: str
    scene_name: str
    scene_path: Path
    scene_index: int
    frame_index: int
    timestamp_ns: int
    has_boxes_3d: bool
    semantic_lidar_sensors: tuple[str, ...]
    camera_sensors: tuple[str, ...]
    lidar_raw_sensors: tuple[str, ...]
    lidar_motion_compensated_sensors: tuple[str, ...]
    radar_sensors: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["scene_path"] = str(self.scene_path)
        data["semantic_lidar_sensors"] = list(self.semantic_lidar_sensors)
        data["camera_sensors"] = list(self.camera_sensors)
        data["lidar_raw_sensors"] = list(self.lidar_raw_sensors)
        data["lidar_motion_compensated_sensors"] = list(
            self.lidar_motion_compensated_sensors
        )
        data["radar_sensors"] = list(self.radar_sensors)
        return data
