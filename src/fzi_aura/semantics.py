from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SEMANTIC_COLORS_UINT8: dict[int, tuple[int, int, int]] = {
    1: (128, 64, 128),
    2: (244, 35, 232),
    3: (250, 170, 160),
    4: (230, 150, 140),
    5: (220, 20, 60),
    6: (255, 0, 0),
    7: (0, 0, 142),
    8: (0, 0, 70),
    9: (0, 60, 100),
    10: (0, 80, 100),
    11: (0, 0, 230),
    12: (119, 11, 32),
    13: (0, 255, 255),
    14: (0, 0, 90),
    15: (0, 0, 110),
    16: (70, 70, 70),
    17: (102, 102, 156),
    18: (190, 153, 153),
    19: (180, 165, 180),
    20: (150, 100, 100),
    21: (150, 120, 90),
    22: (153, 153, 153),
    23: (220, 220, 0),
    24: (250, 170, 30),
    25: (107, 142, 35),
    26: (152, 251, 152),
    27: (255, 255, 255),
    28: (81, 0, 81),
    29: (111, 74, 0),
    30: (255, 20, 107),
    31: (0, 255, 0),
    32: (255, 0, 255),
    33: (255, 0, 255),
    34: (255, 0, 255),
}


@dataclass
class SemanticLabels:
    semantic_id: np.ndarray
    instance_id: np.ndarray
    path: Path
    sensor_id: str
    timestamp_ns: Optional[int]
    class_names: Optional[dict[int, str]] = None

    def __len__(self) -> int:
        return int(len(self.semantic_id))

    def has_instances(self) -> bool:
        return bool(np.any(self.instance_id != 0))


@dataclass
class FusedSemanticPointCloud:
    xyz: np.ndarray
    semantic_id: np.ndarray
    instance_id: np.ndarray
    sensor_id: np.ndarray
    frame: str
    stage: str
    timestamp_ns: Optional[int]

    def __len__(self) -> int:
        return int(len(self.xyz))

    def colors(self, normalize: bool = True) -> np.ndarray:
        return semantic_colors(self.semantic_id, normalize=normalize)


def semantic_color_map() -> dict[int, tuple[int, int, int]]:
    return dict(SEMANTIC_COLORS_UINT8)


def semantic_colors(semantic_id, normalize: bool = True) -> np.ndarray:
    semantic_id = np.asarray(semantic_id)
    colors = np.zeros((len(semantic_id), 3), dtype=np.uint8)
    for idx, value in enumerate(semantic_id):
        colors[idx] = SEMANTIC_COLORS_UINT8.get(int(value), (0, 0, 0))
    if normalize:
        return colors.astype(np.float32) / 255.0
    return colors


def read_semantic_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.fromfile(path, dtype="<u4")
    return (data & np.uint32(0xFFFF)).astype(np.uint32), (data >> np.uint32(16)).astype(
        np.uint32
    )
