from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import FZIAURAError
from .io import load_json

_SENSOR_LAYERS: dict[str, tuple[str, Optional[str]]] = {
    "camera": ("camera_keyframes", "camera_nonkeyframes"),
    "lidar_motion_compensated": (
        "lidar_motion_compensated_keyframes",
        None,  # non-keyframes are not uploaded for space reasons
    ),
    "lidar_raw": ("lidar_raw_keyframes", "lidar_raw_nonkeyframes"),
    "radar": ("radar_keyframes", "radar_nonkeyframes"),
}


def is_keyframe_sample(sample: dict) -> bool:
    """Return whether a sample belongs to a keyframe payload layer.

    A frame is a keyframe when it has 3D-box coverage (including an empty box
    set) or semantic LiDAR labels. The latter case keeps semantic-only
    annotations in the keyframe layers even when no 3D boxes are provided.
    """

    labels = sample.get("labels", {})
    return isinstance(labels, dict) and (
        labels.get("boxes_3d") is True or bool(labels.get("semantic_lidar"))
    )


@dataclass(frozen=True)
class DatasetAvailability:
    """Sensor layers materialized below a dataset root.

    ``layers=None`` represents a normal unpackaged dataset tree, where sample
    references remain the source of truth. Downloaded releases write
    ``available_data.json`` and therefore expose only explicitly selected layers.
    """

    layers: Optional[frozenset[str]] = None

    @classmethod
    def from_root(cls, root: str | Path) -> "DatasetAvailability":
        path = Path(root) / "available_data.json"
        if not path.is_file():
            return cls()
        document = load_json(path)
        layers = document.get("layers")
        if not isinstance(layers, list) or not all(
            isinstance(layer, str) for layer in layers
        ):
            raise FZIAURAError(f"{path}: layers must be a list of strings")
        return cls(frozenset(layers))

    def allows_sensor(self, modality: str, *, is_keyframe: bool) -> bool:
        if self.layers is None:
            return True
        try:
            keyframe_layer, nonkeyframe_layer = _SENSOR_LAYERS[modality]
        except KeyError as exc:
            raise ValueError(f"unknown sensor modality {modality!r}") from exc
        layer = keyframe_layer if is_keyframe else nonkeyframe_layer
        return layer is not None and layer in self.layers

    def allows_any_sensor(self, modality: str) -> bool:
        if self.layers is None:
            return True
        try:
            sensor_layers = _SENSOR_LAYERS[modality]
        except KeyError as exc:
            raise ValueError(f"unknown sensor modality {modality!r}") from exc
        return any(layer in self.layers for layer in sensor_layers)

    def allows_base(self) -> bool:
        return self.layers is None or "base_keyframes" in self.layers
