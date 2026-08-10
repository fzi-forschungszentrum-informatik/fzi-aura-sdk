from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .errors import FieldNotFoundError, UnsupportedPCDError


@dataclass(frozen=True)
class PCDSchema:
    fields: tuple[str, ...]
    sizes: tuple[int, ...]
    types: tuple[str, ...]
    counts: tuple[int, ...]
    width: int
    height: int
    points: int
    data: str
    viewpoint: Optional[tuple[float, ...]] = None


@dataclass
class PointCloud:
    xyz: np.ndarray
    fields: dict[str, np.ndarray]
    schema: PCDSchema
    path: Path
    modality: str
    sensor_id: str
    stage: Optional[str]
    timestamp_ns: Optional[int]

    def __len__(self) -> int:
        return int(self.schema.points)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.fields)

    def has_field(self, name: str) -> bool:
        return name in self.fields

    def field(self, name: str) -> np.ndarray:
        if name not in self.fields:
            available = ", ".join(self.field_names)
            raise FieldNotFoundError(
                f"Field {name} not found in PCD file {self.path}. Available fields: {available}."
            )
        return self.fields[name]

    def to_numpy(self, fields: list[str], missing: str = "raise") -> np.ndarray:
        if missing not in {"raise", "zero", "nan", "drop"}:
            raise ValueError("missing must be one of: raise, zero, nan, drop")
        arrays: list[np.ndarray] = []
        for name in fields:
            if name in self.fields:
                arrays.append(self.fields[name])
            elif missing == "raise":
                self.field(name)
            elif missing == "drop":
                continue
            elif missing == "zero":
                arrays.append(np.zeros(len(self), dtype=np.float32))
            elif missing == "nan":
                arrays.append(np.full(len(self), np.nan, dtype=np.float32))
        if not arrays:
            return np.empty((len(self), 0), dtype=np.float32)
        return np.column_stack(arrays)


@dataclass
class AccumulatedPointCloud:
    xyz: np.ndarray
    fields: dict[str, np.ndarray]
    frame_index: np.ndarray
    timestamp_ns: np.ndarray
    sensor_id: str
    stage: str
    frame: str

    def __len__(self) -> int:
        return int(len(self.xyz))

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.fields)

    def has_field(self, name: str) -> bool:
        return name in self.fields

    def field(self, name: str) -> np.ndarray:
        if name not in self.fields:
            available = ", ".join(self.field_names)
            raise FieldNotFoundError(
                f"Field {name} not found in accumulated point cloud. Available fields: {available}."
            )
        return self.fields[name]


def _parse_header(path: Path) -> tuple[PCDSchema, int]:
    values: dict[str, list[str]] = {}
    offset = 0
    with Path(path).open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise UnsupportedPCDError(f"PCD file has no DATA line: {path}")
            offset += len(line)
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            key = parts[0].upper()
            values[key] = parts[1:]
            if key == "DATA":
                break

    fields = tuple(values.get("FIELDS", []))
    sizes = tuple(int(v) for v in values.get("SIZE", []))
    types = tuple(values.get("TYPE", []))
    counts = tuple(int(v) for v in values.get("COUNT", ["1"] * len(fields)))
    width = int(values.get("WIDTH", ["0"])[0])
    height = int(values.get("HEIGHT", ["1"])[0])
    points = int(values.get("POINTS", [str(width * height)])[0])
    data = values.get("DATA", [""])[0].lower()
    viewpoint = (
        tuple(float(v) for v in values["VIEWPOINT"]) if "VIEWPOINT" in values else None
    )
    schema = PCDSchema(
        fields, sizes, types, counts, width, height, points, data, viewpoint
    )
    lengths = {
        "FIELDS": len(fields),
        "SIZE": len(sizes),
        "TYPE": len(types),
        "COUNT": len(counts),
    }
    if len(set(lengths.values())) != 1 or not fields:
        raise UnsupportedPCDError(
            f"Malformed PCD header in {path}: expected equal non-zero FIELDS/SIZE/TYPE/COUNT lengths, got {lengths}."
        )
    if width < 0 or height < 0 or points < 0:
        raise UnsupportedPCDError(
            f"Malformed PCD header in {path}: negative dimensions."
        )
    if points != width * height:
        raise UnsupportedPCDError(
            f"Malformed PCD header in {path}: POINTS {points} does not match WIDTH*HEIGHT {width * height}."
        )
    return schema, offset


def read_pcd_header(path: Path) -> PCDSchema:
    return _parse_header(path)[0]


def _dtype_for(type_code: str, size: int) -> np.dtype:
    mapping = {
        ("F", 4): np.float32,
        ("F", 8): np.float64,
        ("U", 1): np.uint8,
        ("U", 2): np.uint16,
        ("U", 4): np.uint32,
        ("I", 1): np.int8,
        ("I", 2): np.int16,
        ("I", 4): np.int32,
    }
    try:
        return np.dtype(mapping[(type_code.upper(), size)])
    except KeyError as exc:
        raise UnsupportedPCDError(
            f"Unsupported PCD field type/size: TYPE {type_code} SIZE {size}"
        ) from exc


def read_pcd_binary(
    path: Path,
    fields="all",
    *,
    modality: str = "lidar",
    sensor_id: str = "",
    stage: Optional[str] = None,
    timestamp_ns: Optional[int] = None,
) -> PointCloud:
    path = Path(path)
    schema, offset = _parse_header(path)
    if schema.height != 1:
        raise UnsupportedPCDError(
            f"Only unorganized HEIGHT 1 PCD files are supported: {path}"
        )
    if schema.data != "binary":
        raise UnsupportedPCDError(f"Only DATA binary PCD files are supported: {path}")
    if any(count != 1 for count in schema.counts):
        raise UnsupportedPCDError(f"Only COUNT 1 PCD fields are supported: {path}")
    if not {"x", "y", "z"}.issubset(schema.fields):
        raise UnsupportedPCDError(f"PCD file must contain x, y, z fields: {path}")

    dtype = np.dtype(
        [
            (name, _dtype_for(kind, size))
            for name, kind, size in zip(schema.fields, schema.types, schema.sizes)
        ]
    )
    with path.open("rb") as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype, count=schema.points)
    if len(data) != schema.points:
        raise UnsupportedPCDError(
            f"Truncated PCD payload in {path}: expected {schema.points} points, read {len(data)}."
        )
    selected = schema.fields if fields == "all" else tuple(fields)
    arrays: dict[str, np.ndarray] = {}
    for name in selected:
        if name not in schema.fields:
            available = ", ".join(schema.fields)
            raise FieldNotFoundError(
                f"Field {name} not found in PCD file {path}. Available fields: {available}."
            )
        arrays[name] = data[name]
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(
        np.float64, copy=False
    )
    return PointCloud(
        xyz, arrays, schema, path, modality, sensor_id, stage, timestamp_ns
    )
