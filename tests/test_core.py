from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fzi_aura import (
    DETECTION_CLASSES,
    Box3D,
    BoxCollection,
    Calibration,
    FrameDataset,
    FZIAURADataset,
    read_pcd_binary,
    read_pcd_header,
    read_semantic_label,
    semantic_color_map,
    semantic_colors,
)
from fzi_aura.errors import (
    AnnotationNotFoundError,
    FZIAURAError,
    SceneNotFoundError,
    SensorNotFoundError,
    UnsupportedPCDError,
)
from fzi_aura.geometry import box_corners_lwh, project_points, transform_matrix
from fzi_aura.io import resolve_scene_path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


PCD_TYPE_TO_DTYPE = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
}


def _write_schema_pcd(
    path: Path, fields: tuple[str, ...], sizes: tuple[int, ...], types: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# .PCD v0.7\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(str(v) for v in sizes)}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join('1' for _ in fields)}\n"
        "WIDTH 2\n"
        "HEIGHT 1\n"
        "POINTS 2\n"
        "DATA binary\n"
    ).encode("ascii")
    dtype = np.dtype(
        [
            (name, PCD_TYPE_TO_DTYPE[(kind, size)])
            for name, size, kind in zip(fields, sizes, types)
        ]
    )
    data = np.zeros(2, dtype=dtype)
    for i, name in enumerate(fields):
        data[name] = np.asarray([i + 1, i + 101], dtype=dtype[name])
    with path.open("wb") as f:
        f.write(header)
        data.tofile(f)


def _write_pcd(path: Path) -> None:
    _write_schema_pcd(
        path, ("x", "y", "z", "intensity"), (4, 4, 4, 4), ("F", "F", "F", "U")
    )


def make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    scene_path = root / "scenes" / "scene_0"
    _write_json(
        root / "dataset.json",
        {
            "format_version": "v1.2.1",
            "scene_count": 1,
            "scenes": [
                {
                    "scene_id": "scene|0",
                    "path": "scenes/scene_0",
                    "counts": {
                        "samples": 2,
                        "boxes_3d_rows": 1,
                        "boxes_3d_annotated_keyframes": 2,
                        "boxes_3d_empty_keyframes": 1,
                        "boxes_3d_nonempty_keyframes": 1,
                        "semantic_lidar_frames": 1,
                    },
                    "start_timestamp_ns": 100,
                    "end_timestamp_ns": 200,
                    "duration_s": 0.1,
                }
            ],
        },
    )
    split_dir = root / "splits" / "v1.0"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("scene|0\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("", encoding="utf-8")
    (split_dir / "test.txt").write_text("", encoding="utf-8")
    _write_json(
        scene_path / "scene.json",
        {
            "format_version": "v1.2.1",
            "scene_id": "scene|0",
            "name": "scene_0",
            "sample_count": 2,
            "sensors": {"camera": ["front"], "lidar": ["top"], "radar": []},
            "modalities": [
                "camera",
                "lidar_raw",
                "lidar_motion_compensated",
                "ego",
                "poses",
                "labels_3d",
                "labels_semantic_lidar",
            ],
            "paths": {
                "samples": "samples.jsonl",
                "samples_parquet": "samples.parquet",
                "calibration": "calibration.json",
                "ego_poses": "ego/poses.parquet",
                "vehicle_signals": "ego/vehicle_signals.parquet",
                "box_sources": "labels/box_sources.json",
                "boxes_3d": "labels/boxes_3d.jsonl",
                "semantic_classes": "labels/semantic/classes.json",
            },
        },
    )
    _write_json(
        scene_path / "calibration.json",
        {
            "calibration_id": "calib_scene_0",
            "format_version": "v1.2.1",
            "coordinate_system": {
                "world_frame": "odom",
                "ego_frame": "base_link",
                "box_frame": "base_link",
            },
            "sensors": {
                "lidar/top": {
                    "modality": "lidar",
                    "sensor_id": "top",
                    "frame": "lidar_top_frame",
                    "parent_frame": "base_link",
                    "transform_base_link_from_sensor": {
                        "translation_xyz": [0, 0, 0],
                        "rotation_xyzw": [0, 0, 0, 1],
                    },
                },
                "camera/front": {
                    "modality": "camera",
                    "sensor_id": "front",
                    "frame": "camera_front_frame",
                    "parent_frame": "base_link",
                    "intrinsics": {"P": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]},
                    "transform_base_link_from_sensor": {
                        "translation_xyz": [0, 0, 0],
                        "rotation_xyzw": [0, 0, 0, 1],
                    },
                },
            },
        },
    )
    _write_pcd(scene_path / "lidar" / "motion_compensated" / "top" / "100.pcd")
    _write_pcd(scene_path / "lidar" / "raw" / "top" / "100.pcd")
    _write_pcd(scene_path / "lidar" / "motion_compensated" / "top" / "200.pcd")
    _write_pcd(scene_path / "lidar" / "raw" / "top" / "200.pcd")
    camera_path = scene_path / "camera" / "front" / "100.jpg"
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color=(0, 0, 0)).save(camera_path)
    label_data = np.array([1 | (0 << 16), 7 | (4 << 16)], dtype=np.uint32)
    label_path = scene_path / "labels" / "semantic" / "top" / "100.label"
    label_path.parent.mkdir(parents=True)
    label_data.tofile(label_path)
    _write_json(
        scene_path / "labels" / "semantic" / "classes.json",
        {
            "semantic_classes": [
                {"id": 1, "name": "road"},
                {"id": 7, "name": "car"},
            ],
            "box_classes": ["car"],
            "label_array": {
                "dtype": "uint32",
                "format": "semantic_id | (instance_id << 16)",
                "attachment": "point_index",
                "matches": "raw and motion-compensated lidar sensor point order",
                "instance_assignment_coordinate_stage": "motion_compensated",
            },
        },
    )
    _write_json(
        scene_path / "labels" / "box_sources.json",
        {"default": "labels/boxes_3d.jsonl"},
    )
    _write_jsonl(
        scene_path / "labels" / "boxes_3d.jsonl",
        [
            {
                "timestamp_ns": 100,
                "frame_index": 0,
                "category": "car",
                "object_id": "car:1",
                "box": {
                    "center": [0, 0, 0],
                    "size_lwh": [2, 2, 2],
                    "rotation_xyzw": [0, 0, 0, 1],
                    "frame": "base_link",
                },
            }
        ],
    )
    _write_jsonl(
        scene_path / "samples.jsonl",
        [
            {
                "timestamp_ns": 100,
                "frame_index": 0,
                "cameras": {"front": "camera/front/100.jpg"},
                "lidar": {
                    "motion_compensated": {
                        "top": "lidar/motion_compensated/top/100.pcd"
                    },
                    "raw": {"top": "lidar/raw/top/100.pcd"},
                },
                "labels": {
                    "boxes_3d": True,
                    "boxes_3d_count": 1,
                    "semantic_segmentation": True,
                    "semantic_lidar": {"top": "labels/semantic/top/100.label"},
                },
                "vehicle_signals": {
                    "path": "ego/vehicle_signals.parquet",
                    "timestamp_ns": 100,
                },
                "ego_pose": {
                    "path": "ego/poses.parquet",
                    "position": [10, 0, 0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
            },
            {
                "timestamp_ns": 200,
                "frame_index": 1,
                "lidar": {
                    "motion_compensated": {
                        "top": "lidar/motion_compensated/top/200.pcd"
                    },
                    "raw": {"top": "lidar/raw/top/200.pcd"},
                },
                "labels": {
                    "boxes_3d": True,
                    "boxes_3d_count": 0,
                    "semantic_segmentation": False,
                },
                "ego_pose": {
                    "path": "ego/poses.parquet",
                    "position": [0, 0, 0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "vehicle_signals": {
                    "path": "ego/vehicle_signals.parquet",
                    "timestamp_ns": 200,
                },
            },
        ],
    )
    (scene_path / "ego").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp_ns": [100, 200],
            "speed_kph": [1.5, 2.5],
            "weather": [
                [{"description": "clear sky", "id": 800}],
                [{"description": "few clouds", "id": 801}],
            ],
        }
    ).to_parquet(scene_path / "ego" / "vehicle_signals.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp_ns": [100, 200],
            "frame_index": [0, 1],
            "labels": [
                {"semantic_segmentation": True},
                {"semantic_segmentation": False},
            ],
        }
    ).to_parquet(scene_path / "samples.parquet", index=False)
    pd.DataFrame(
        {
            "frame_index": [0, 1],
            "timestamp_ns": [100, 200],
            "translation_x": [10.0, 0.0],
            "translation_y": [0.0, 0.0],
            "translation_z": [0.0, 0.0],
            "rotation_x": [0.0, 0.0],
            "rotation_y": [0.0, 0.0],
            "rotation_z": [0.0, 0.0],
            "rotation_w": [1.0, 1.0],
        }
    ).to_parquet(scene_path / "ego" / "poses.parquet", index=False)
    return root


def test_dataset_rejects_non_v121_format(tmpdir):
    root = make_dataset(Path(str(tmpdir)))
    document = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    document["format_version"] = "v1.1"
    (root / "dataset.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FZIAURAError, match="expected 'v1.2.1'"):
        FZIAURADataset(root)


def test_dataset_hides_manifest_consumer_excluded_scenes(tmp_path):
    root = make_dataset(tmp_path)
    document = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    document["consumer_excluded_scenes"] = [
        {
            "scene_id": "scene|0",
            "reason": "Mandated upload; unsuitable for default consumption",
        }
    ]
    (root / "dataset.json").write_text(json.dumps(document), encoding="utf-8")

    dataset = FZIAURADataset(root)

    assert dataset.consumer_excluded_scene_ids == frozenset({"scene|0"})
    assert len(dataset) == 0
    assert dataset.scene_ids == ()
    assert dataset.stats()["counts"]["scenes"] == 0
    assert len(dataset.as_frames()) == 0
    assert len(FZIAURADataset(root, split="train")) == 0
    with pytest.raises(SceneNotFoundError, match="not selected"):
        dataset.get_scene("scene|0")


def test_scene_lookup_accepts_path_safe_scene_name(tmp_path):
    root = make_dataset(tmp_path)
    canonical_id = "2025-06-11-14-31-00|37"
    path_safe_name = "2025-06-11-14-31-00_37"
    document = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    document["scenes"][0]["scene_id"] = canonical_id
    (root / "dataset.json").write_text(json.dumps(document), encoding="utf-8")
    (root / "splits/v1.0/train.txt").write_text(f"{canonical_id}\n", encoding="utf-8")

    dataset = FZIAURADataset(root, split="train")

    assert dataset.scene_ids == (canonical_id,)
    assert dataset.get_scene(path_safe_name).scene_id == canonical_id
    assert dataset.index_of_scene(path_safe_name) == 0
    assert dataset.as_frames().indices_for_scene(path_safe_name) == [0, 1]
    with pytest.raises(SceneNotFoundError):
        dataset.get_scene("scene_0")


def test_scene_relative_paths_and_lidar_stages_are_strict(tmpdir):
    root = make_dataset(Path(str(tmpdir)))
    with pytest.raises(ValueError, match="must not be absolute"):
        resolve_scene_path(root, "/outside/file.pcd")
    with pytest.raises(ValueError, match="escapes the scene directory"):
        resolve_scene_path(root, "../outside/file.pcd")
    with pytest.raises(ValueError, match="must not be absolute"):
        resolve_scene_path(root, r"C:\data\file.pcd")
    frame = FZIAURADataset(root)[0][0]
    with pytest.raises(ValueError, match="raw.*motion_compensated"):
        frame.load_lidar("top", stage="unknown")


def test_cli_validator_accepts_minimal_v121_export(tmpdir):
    from fzi_aura.cli.validate import _validate_scene

    root = make_dataset(Path(str(tmpdir)))
    errors = []
    _validate_scene(FZIAURADataset(root)[0], errors)
    assert errors == []


def test_cli_validator_respects_selected_payload_layers(tmp_path):
    from fzi_aura.cli.validate import _validate_scene

    root = make_dataset(tmp_path)
    scene_path = root / "scenes" / "scene_0"
    samples_path = scene_path / "samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    samples[1]["labels"]["boxes_3d"] = False
    samples[1]["labels"].pop("boxes_3d_count")
    _write_jsonl(samples_path, samples)
    _write_json(
        root / "available_data.json",
        {
            "version": "v1.0",
            "layers": [
                "base_keyframes",
                "lidar_motion_compensated_keyframes",
            ],
        },
    )
    # These references remain in the canonical sample index, but their layers
    # were deliberately not installed. The remaining MC keyframe is installed.
    (scene_path / "camera" / "front" / "100.jpg").unlink()
    (scene_path / "lidar" / "raw" / "top" / "100.pcd").unlink()
    (scene_path / "lidar" / "raw" / "top" / "200.pcd").unlink()
    (scene_path / "lidar" / "motion_compensated" / "top" / "200.pcd").unlink()

    errors = []
    _validate_scene(FZIAURADataset(root)[0], errors)
    assert errors == []

    # A payload from a selected layer must still be reported as missing.
    (scene_path / "lidar" / "motion_compensated" / "top" / "100.pcd").unlink()
    errors = []
    _validate_scene(FZIAURADataset(root)[0], errors)
    assert any("lidar/motion_compensated/top" in error for error in errors)


def test_scene_and_frame_index(tmpdir):
    root = make_dataset(Path(str(tmpdir)))
    scenes = FZIAURADataset(root, split="train")
    assert len(scenes) == 1
    scene = scenes[0]
    assert len(scene) == 2
    frames = scenes.as_frames(sample_filter="boxes_3d")
    assert len(frames) == 2
    assert frames[0].scene_id == "scene|0"
    assert len(FrameDataset(root, split="train", sample_filter="boxes_3d")) == len(
        frames
    )
    camera_frames = scene.frames(
        sample_filter="sensor_available", require_cameras=["front"], return_mode="dict"
    )
    assert len(camera_frames) == 1
    assert camera_frames[0]["frame_index"] == 0


def test_scene_helpers_do_not_retain_full_sample_list(tmp_path):
    root = make_dataset(tmp_path)
    scene = FZIAURADataset(root, split="train")[0]

    assert not hasattr(scene, "_samples")
    assert scene._sample_ranges is None
    assert len(scene) == 2
    assert scene.available_lidars() == ("top",)
    assert scene.detection_keyframe_indices() == [0, 1]
    assert scene.semantic_keyframe_indices() == [0]
    assert scene._sample_ranges is None

    assert scene[0].timestamp_ns == 100
    assert scene.ego_pose_matrix(1).shape == (4, 4)
    assert scene.ego_velocity(0, target_frame="odom").shape == (3,)
    assert scene.ego_trajectory(0, (0, 1), target_frame="odom").shape == (2, 3)
    assert scene._sample_ranges.shape == (2, 2)
    assert scene._sample_ranges.dtype == np.int64
    assert scene._sample_ranges.nbytes == 32
    assert not hasattr(scene, "_samples")

    first_load = scene.load_samples()
    second_load = scene.load_samples()
    assert first_load == second_load
    assert first_load is not second_load
    assert first_load[0] is not second_load[0]
    assert not hasattr(scene, "_samples")


def test_frame_dataset_sample_retention_is_explicit(tmp_path):
    root = make_dataset(tmp_path)

    frames = FrameDataset(root, split="train")
    assert frames.cache_samples is False
    assert frames._samples is None
    assert frames._sample_ranges.shape == (2, 2)
    assert frames._sample_ranges.dtype == np.int64
    assert frames._sample_ranges.nbytes == 32
    first = frames[0]
    assert first.timestamp_ns == 100
    assert frames._samples is None

    cached = FrameDataset(root, split="train", cache_samples=True)
    assert cached.cache_samples is True
    assert len(cached._samples) == len(cached)
    assert cached[0].sample_dict is cached._samples[0]

    dataset_cached = FZIAURADataset(root, split="train").as_frames(cache_samples=True)
    scene_cached = FZIAURADataset(root, split="train")[0].frames(cache_samples=True)
    assert len(dataset_cached._samples) == len(dataset_cached)
    assert len(scene_cached._samples) == len(scene_cached)


def test_sensor_availability_respects_downloaded_layers(tmp_path):
    root = make_dataset(tmp_path)
    scene_path = root / "scenes" / "scene_0"
    samples_path = scene_path / "samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    samples[1]["labels"]["boxes_3d"] = False
    samples[1]["cameras"] = {"front": "camera/front/200.jpg"}
    _write_jsonl(samples_path, samples)
    Image.new("RGB", (1, 1), color=(0, 0, 0)).save(
        scene_path / "camera" / "front" / "200.jpg"
    )
    _write_json(
        root / "available_data.json",
        {
            "version": "v1.0",
            "layers": [
                "base_keyframes",
                "camera_keyframes",
                "lidar_raw_keyframes",
            ],
        },
    )

    dataset = FZIAURADataset(root, split="train")
    scene = dataset[0]
    keyframe = scene[0]
    nonkeyframe = scene[1]

    assert dataset.available_layers == frozenset(
        {"base_keyframes", "camera_keyframes", "lidar_raw_keyframes"}
    )
    assert scene.available_cameras() == ("front",)
    assert scene.available_lidars("raw") == ("top",)
    assert scene.available_lidars("motion_compensated") == ()
    assert keyframe.available_cameras() == ("front",)
    assert keyframe.available_lidars("raw") == ("top",)
    assert keyframe.available_lidars("motion_compensated") == ()
    assert keyframe.semantic_lidar_sensors == ("top",)
    assert nonkeyframe.available_cameras() == ()
    assert nonkeyframe.available_lidars("raw") == ()
    assert len(dataset.as_frames(require_cameras=["front"])) == 1
    assert len(dataset.as_frames(require_lidars=["top"], lidar_stage="raw")) == 1
    assert len(dataset.as_frames(require_lidars=["top"])) == 0
    with pytest.raises(SensorNotFoundError, match="installed layers"):
        keyframe.load_lidar("top", stage="motion_compensated")


def test_semantic_only_sample_uses_keyframe_payload_layers(tmp_path):
    from fzi_aura.availability import is_keyframe_sample
    from fzi_aura.cli.validate import _validate_scene

    root = make_dataset(tmp_path)
    scene_path = root / "scenes" / "scene_0"
    samples_path = scene_path / "samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    labels = samples[1]["labels"]
    labels["boxes_3d"] = False
    labels.pop("boxes_3d_count")
    labels["semantic_segmentation"] = True
    labels["semantic_lidar"] = {"top": "labels/semantic/top/100.label"}
    _write_jsonl(samples_path, samples)
    _write_json(
        root / "available_data.json",
        {
            "version": "v1.0",
            "layers": [
                "base_keyframes",
                "lidar_motion_compensated_keyframes",
            ],
        },
    )

    dataset = FZIAURADataset(root, split="train")
    semantic_only_frame = dataset[0][1]
    assert is_keyframe_sample(semantic_only_frame.sample_dict)
    assert semantic_only_frame.is_keyframe
    assert semantic_only_frame.available_lidars() == ("top",)
    assert dataset[0].available_lidars() == ("top",)
    assert len(dataset.as_frames(require_lidars=["top"])) == 2

    errors = []
    _validate_scene(dataset[0], errors)
    assert errors == []


def test_frame_dataset_reuses_and_clears_scene_cache(tmpdir):
    root = make_dataset(Path(str(tmpdir)))

    frames = FrameDataset(root, split="train")
    first_scene = frames[0]._scene
    assert frames[1]._scene is first_scene
    assert next(frames.iter_scenes()) is first_scene
    assert len(frames._scene_cache) == 1

    frames.clear_scene_cache()
    assert frames._scene_cache == {}
    assert frames[0]._scene is not first_scene

    uncached = FrameDataset(root, split="train", cache_scenes=False)
    assert uncached[0]._scene is not uncached[1]._scene
    assert next(uncached.iter_scenes()) is not uncached[0]._scene
    assert uncached._scene_cache == {}

    dataset_uncached = FZIAURADataset(root, split="train", cache_scenes=False)
    frame_dataset = dataset_uncached.as_frames()
    assert frame_dataset.cache_scenes is False
    assert frame_dataset._scene_cache == {}


def test_vehicle_signal_bulk_load_and_exact_timestamp_lookup(tmpdir, monkeypatch):
    from pyarrow.lib import ArrowInvalid

    root = make_dataset(Path(str(tmpdir)))
    read_calls = []
    read_parquet = pd.read_parquet

    def tracked_read_parquet(path, *args, **kwargs):
        read_calls.append(path)
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", tracked_read_parquet)

    frames = FrameDataset(root, split="train")
    frame_zero = frames[0]
    frame_one = frames[1]
    scene = frame_zero._scene
    assert frame_zero.load_vehicle_signals_row()["speed_kph"] == 1.5
    assert frame_one.load_vehicle_signals_row()["speed_kph"] == 2.5
    bulk = scene.load_vehicle_signals()
    assert bulk is scene.load_vehicle_signals()
    assert scene._vehicle_signals_bulk_cache[None] is bulk
    assert bulk["visibility"].dtype == np.float64
    assert bulk["visibility"].isna().all()
    assert len(read_calls) == 1

    projected = scene.load_vehicle_signals(columns=["timestamp_ns", "speed_kph"])
    assert list(projected.columns) == ["timestamp_ns", "speed_kph"]
    assert projected is scene.load_vehicle_signals(
        columns=("timestamp_ns", "speed_kph")
    )
    assert projected is not bulk
    assert scene.load_vehicle_signals() is bulk
    assert scene._vehicle_signals_bulk_cache[("timestamp_ns", "speed_kph")] is projected
    assert len(read_calls) == 2

    optional = scene.load_vehicle_signals(columns=("timestamp_ns", "visibility"))
    assert list(optional.columns) == ["timestamp_ns", "visibility"]
    assert optional["visibility"].dtype == np.float64
    assert optional["visibility"].isna().all()
    assert optional is scene.load_vehicle_signals(
        columns=("timestamp_ns", "visibility")
    )
    assert len(read_calls) == 3

    with pytest.raises(ArrowInvalid):
        scene.load_vehicle_signals(columns=("timestamp_ns", "visiblity"))


def test_vehicle_signal_projection_with_nested_and_missing_optional_columns(tmpdir):
    root = make_dataset(Path(str(tmpdir)))
    scene = FZIAURADataset(root, split="train")[0]

    projected = scene.load_vehicle_signals(
        columns=("timestamp_ns", "weather", "visibility")
    )

    assert list(projected.columns) == ["timestamp_ns", "weather", "visibility"]
    assert projected["weather"].tolist() == [
        [{"description": "clear sky", "id": 800}],
        [{"description": "few clouds", "id": 801}],
    ]
    assert projected["visibility"].dtype == np.float64
    assert projected["visibility"].isna().all()


def test_pcd_and_semantics(tmpdir):
    root = make_dataset(Path(str(tmpdir)))
    frame = FZIAURADataset(root, split="train").as_frames(
        sample_filter="semantic_lidar"
    )[0]
    cloud, labels = frame.load_lidar_semantic_pair("top")
    raw_cloud, raw_labels = frame.load_lidar_semantic_pair("top", stage="raw")
    assert len(cloud) == 2
    assert len(labels) == 2
    assert len(raw_cloud) == 2
    assert len(raw_labels) == 2
    assert cloud.field("intensity").tolist() == [4, 104]
    assert raw_cloud.field("intensity").tolist() == [4, 104]
    assert labels.semantic_id.tolist() == [1, 7]
    assert raw_labels.semantic_id.tolist() == [1, 7]
    assert labels.instance_id.tolist() == [0, 4]
    assert read_pcd_header(frame.lidar_path("top")).fields == (
        "x",
        "y",
        "z",
        "intensity",
    )
    assert read_pcd_binary(frame.lidar_path("top")).xyz.shape == (2, 3)
    selected = read_pcd_binary(frame.lidar_path("top"), fields=["intensity"])
    assert selected.field_names == ("intensity",)
    assert selected.xyz.shape == (2, 3)
    sem, inst = read_semantic_label(frame.semantic_path("top"))
    assert sem.tolist() == [1, 7]
    assert inst.tolist() == [0, 4]


def test_detection_classes_are_public():
    assert DETECTION_CLASSES == (
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


def test_load_lidar_hides_velocity_for_non_aeva_motion_compensated_sensor(tmpdir):
    root = make_dataset(Path(str(tmpdir)))
    motion_path = (
        root / "scenes" / "scene_0" / "lidar" / "motion_compensated" / "top" / "100.pcd"
    )
    _write_schema_pcd(
        motion_path,
        ("x", "y", "z", "intensity", "velocity"),
        (4, 4, 4, 4, 4),
        ("F", "F", "F", "U", "F"),
    )
    frame = FZIAURADataset(root, split="train").as_frames()[0]

    cloud = frame.load_lidar("top", stage="motion_compensated")
    assert read_pcd_header(motion_path).fields[-1] == "velocity"
    assert cloud.field_names == ("x", "y", "z", "intensity")
    assert not cloud.has_field("velocity")
    selected = frame.load_lidar(
        "top", stage="motion_compensated", fields=["x", "velocity"]
    )
    assert selected.field_names == ("x",)

    frame.sample_dict["lidar"]["motion_compensated"]["aeva_front_center"] = (
        "lidar/motion_compensated/top/100.pcd"
    )
    aeva_cloud = frame.load_lidar("aeva_front_center", stage="motion_compensated")
    assert aeva_cloud.field_names == ("x", "y", "z", "intensity", "velocity")


def test_all_known_pcd_schema_fields_are_usable(tmpdir):
    schemas = [
        (
            ("x", "y", "z", "rcs", "elevation_angle"),
            (4, 4, 4, 4, 4),
            ("F", "F", "F", "F", "F"),
        ),
        (
            ("x", "y", "z", "rcs", "elevation_angle"),
            (4, 4, 4, 1, 4),
            ("F", "F", "F", "I", "F"),
        ),
        (
            ("x", "y", "z", "t", "reflectivity", "ring", "range"),
            (4, 4, 4, 4, 2, 2, 4),
            ("F", "F", "F", "U", "U", "U", "U"),
        ),
        (
            ("x", "y", "z", "t", "reflectivity", "ring", "range", "velocity"),
            (4, 4, 4, 4, 2, 2, 4, 4),
            ("F", "F", "F", "U", "U", "U", "U", "F"),
        ),
        (
            (
                "x",
                "y",
                "z",
                "velocity",
                "intensity",
                "signal_quality",
                "reflectivity",
                "time_offset_ns",
                "point_flags_lsb",
                "point_flags_msb",
            ),
            (4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
            ("F", "F", "F", "F", "F", "F", "F", "I", "U", "U"),
        ),
        (
            (
                "x",
                "y",
                "z",
                "intensity",
                "t",
                "reflectivity",
                "ring",
                "range",
                "velocity",
            ),
            (4, 4, 4, 4, 4, 2, 2, 4, 4),
            ("F", "F", "F", "F", "U", "U", "U", "U", "F"),
        ),
    ]
    for idx, (fields, sizes, types) in enumerate(schemas):
        path = Path(str(tmpdir)) / f"schema_{idx}.pcd"
        _write_schema_pcd(path, fields, sizes, types)
        cloud = read_pcd_binary(path)
        assert cloud.schema.fields == fields
        assert cloud.schema.sizes == sizes
        assert cloud.schema.types == types
        assert len(cloud) == 2
        assert cloud.xyz.shape == (2, 3)
        assert cloud.field_names == fields
        for name, size, kind in zip(fields, sizes, types):
            assert cloud.has_field(name)
            field = cloud.field(name)
            assert len(field) == 2
            assert field.dtype == np.dtype(PCD_TYPE_TO_DTYPE[(kind, size)])
        stacked = cloud.to_numpy(list(fields))
        assert stacked.shape == (2, len(fields))
        assert cloud.to_numpy(["x", "missing"], missing="zero").shape == (2, 2)
        assert cloud.to_numpy(["x", "missing"], missing="nan").shape == (2, 2)
        assert cloud.to_numpy(["x", "missing"], missing="drop").shape == (2, 1)


def test_pcd_rejects_malformed_and_truncated_payloads(tmpdir):
    root = Path(str(tmpdir))
    bad_header = root / "bad_header.pcd"
    bad_header.write_bytes(
        b"# .PCD v0.7\n"
        b"FIELDS x y z\n"
        b"SIZE 4 4\n"
        b"TYPE F F F\n"
        b"COUNT 1 1 1\n"
        b"WIDTH 1\n"
        b"HEIGHT 1\n"
        b"POINTS 1\n"
        b"DATA binary\n"
    )
    try:
        read_pcd_binary(bad_header)
    except UnsupportedPCDError as exc:
        assert "Malformed PCD header" in str(exc)
    else:
        raise AssertionError("Malformed PCD header was accepted")

    truncated = root / "truncated.pcd"
    _write_pcd(truncated)
    data = truncated.read_bytes()
    truncated.write_bytes(data[:-4])
    try:
        read_pcd_binary(truncated)
    except UnsupportedPCDError as exc:
        assert "Truncated PCD payload" in str(exc)
    else:
        raise AssertionError("Truncated PCD payload was accepted")


def test_boxes_calibration_and_geometry(tmpdir):
    root = make_dataset(Path(str(tmpdir)))
    scene = FZIAURADataset(root, split="train")[0]
    frame = scene[0]
    assert len(frame.load_boxes()) == 1
    assert frame.boxes_3d_count == 1
    box_arrays = frame.load_boxes().as_numpy()
    assert box_arrays["center"].shape == (1, 3)
    assert box_arrays["size_lwh"].tolist() == [[2.0, 2.0, 2.0]]
    assert box_arrays["rotation_xyzw"].shape == (1, 4)
    assert box_arrays["yaw"].tolist() == [0.0]
    assert box_arrays["category"].tolist() == ["car"]
    assert box_arrays["object_id"].tolist() == ["car:1"]
    assert frame.calibration().camera_projection_matrix("front").shape == (3, 4)
    assert transform_matrix([0, 0, 0], [0, 0, 0, 1]).shape == (4, 4)
    assert np.allclose(transform_matrix([0, 0, 0], [0, 0, 0, 2])[:3, :3], np.eye(3))
    try:
        transform_matrix([0, 0, 0], [0, 0, 0, 0])
    except ValueError as exc:
        assert "non-zero quaternion" in str(exc)
    else:
        raise AssertionError("Zero quaternion was accepted")
    rotated_box = (
        frame.load_boxes()
        .boxes[0]
        .transformed(
            transform_matrix(
                [1, 2, 3],
                [0, 0, np.sqrt(0.5), np.sqrt(0.5)],
            ),
            frame="odom",
        )
    )
    assert rotated_box.center.tolist() == [1.0, 2.0, 3.0]
    assert np.isclose(rotated_box.yaw, np.pi / 2.0)
    assert rotated_box.frame == "odom"
    assert box_corners_lwh([0, 0, 0], [2, 2, 2]).shape == (8, 3)
    uv, depth = project_points(np.array([[1.0, 2.0, 1.0]]), np.eye(3, 4))
    assert uv.tolist() == [[1.0, 2.0]]
    assert depth.tolist() == [1.0]
    empty_annotation = scene[1]
    assert empty_annotation.boxes_3d_count == 0
    assert len(empty_annotation.load_boxes(strict=True)) == 0


def test_box_access_retains_only_binary_offsets(tmp_path):
    root = make_dataset(tmp_path)
    box_path = root / "scenes" / "scene_0" / "labels" / "boxes_3d.jsonl"
    row = json.loads(box_path.read_text(encoding="utf-8"))
    row["object_id"] = "cär:1"
    box_path.write_bytes(
        b"\n" + json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    scene = FZIAURADataset(root, split="train")[0]
    assert scene._box_offset_cache == {}

    first = scene[0].load_boxes(strict=True)
    assert len(first) == 1
    assert not hasattr(scene, "_box_cache")
    assert len(scene._box_offset_cache) == 1
    box_index = next(iter(scene._box_offset_cache.values()))
    entry = box_index.ranges[100]
    assert entry.timestamp_ns == 100
    assert entry.frame_index == 0
    assert entry.box_count == 1
    assert entry.start_offset < entry.end_offset
    assert first.boxes[0].object_id == "cär:1"
    assert all(not isinstance(value, Box3D) for value in box_index.ranges.values())

    second = scene[0].load_boxes(strict=True)
    assert second.boxes[0] is not first.boxes[0]
    assert len(scene[1].load_boxes(strict=True)) == 0

    uncached_scene = FZIAURADataset(root, split="train")[0]
    assert len(uncached_scene.tracks()["cär:1"]) == 1
    assert len(uncached_scene.get_track("cär:1")) == 1
    assert uncached_scene.get_track("missing") == []
    assert uncached_scene._box_offset_cache == {}


def test_sensor_frame_boxes_use_declared_path_and_offset_index(tmp_path):
    root = make_dataset(tmp_path)
    scene_path = root / "scenes" / "scene_0"
    samples_path = scene_path / "samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    relative = "labels/custom/top_boxes.jsonl"
    for sample in samples:
        sample["labels"]["boxes_3d_sensor_frame"] = {"top": relative}
    _write_jsonl(samples_path, samples)
    _write_jsonl(
        scene_path / relative,
        [
            {
                "timestamp_ns": 100,
                "frame_index": 0,
                "category": "car",
                "object_id": "car:1",
                "sensor_id": "top",
                "box": {
                    "center": [1, 2, 3],
                    "size_lwh": [2, 2, 2],
                    "rotation_xyzw": [0, 0, 0, 1],
                    "frame": "lidar_top_frame",
                },
            }
        ],
    )

    scene = FZIAURADataset(root, split="train")[0]
    sensor_boxes = scene[0].load_boxes(sensor_id="top", strict=True)
    assert len(sensor_boxes) == 1
    assert sensor_boxes.boxes[0].sensor_id == "top"
    assert sensor_boxes.boxes[0].frame == "lidar_top_frame"
    assert sensor_boxes.frame == "lidar_top_frame"
    assert len(scene[0].load_boxes(frame="lidar/top", sensor_id="top")) == 1
    assert len(scene[0].load_boxes(frame="lidar_top_frame", sensor_id="top")) == 1
    sensor_projection = scene[0].project_boxes_to_camera("front", sensor_id="top")
    assert len(sensor_projection) == 1
    assert sensor_projection[0]["box"].sensor_id == "top"
    assert sensor_projection[0]["uv"].shape == (8, 2)
    with pytest.raises(ValueError, match="does not identify lidar sensor"):
        scene[0].load_boxes(frame="base_link", sensor_id="top")
    with pytest.raises(ValueError, match="require frame='base_link'"):
        scene[0].load_boxes(frame="lidar/top")
    assert len(scene[1].load_boxes(sensor_id="top", strict=True)) == 0
    assert len(scene.boxes_at(100, sensor_id="top")) == 1
    assert scene.boxes_at(200, sensor_id="top").boxes == []
    assert scene._sensor_box_paths == {"top": scene_path / relative}
    assert not scene._box_path("top").exists()
    with pytest.raises(ValueError, match="calibrated lidar sensor"):
        scene[0].load_boxes(sensor_id="missing", strict=True)

    from fzi_aura.cli.validate import _validate_scene

    errors = []
    _validate_scene(scene, errors)
    assert errors == []

    wrong_frame = {
        **json.loads((scene_path / relative).read_text().strip()),
        "box": {
            **json.loads((scene_path / relative).read_text().strip())["box"],
            "frame": "wrong",
        },
    }
    _write_jsonl(scene_path / relative, [wrong_frame])
    scene = FZIAURADataset(root, split="train")[0]
    errors = []
    _validate_scene(scene, errors)
    assert any(
        "box.frame='wrong', expected 'lidar_top_frame'" in error for error in errors
    )


def test_box_strictness_distinguishes_coverage_empty_and_missing(tmp_path):
    from fzi_aura.cli.validate import _validate_scene

    root = make_dataset(tmp_path)
    scene_path = root / "scenes" / "scene_0"
    samples_path = scene_path / "samples.jsonl"
    base_box_path = scene_path / "labels" / "boxes_3d.jsonl"
    sensor_relative = "labels/custom/top_boxes.jsonl"
    sensor_box_path = scene_path / sensor_relative
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    base_row = json.loads(base_box_path.read_text().strip())
    sensor_row = {
        **base_row,
        "sensor_id": "top",
        "box": {**base_row["box"], "frame": "lidar_top_frame"},
    }
    for sample in samples:
        sample["labels"]["boxes_3d_sensor_frame"] = {"top": sensor_relative}
    _write_jsonl(samples_path, samples)
    _write_jsonl(sensor_box_path, [sensor_row])

    scene = FZIAURADataset(root, split="train")[0]
    assert len(scene[1].load_boxes(strict=True)) == 0
    assert len(scene[1].load_boxes(sensor_id="top", strict=True)) == 0

    samples[1]["labels"]["boxes_3d"] = False
    samples[1]["labels"].pop("boxes_3d_count")
    _write_jsonl(samples_path, samples)
    scene = FZIAURADataset(root, split="train")[0]
    assert len(scene[1].load_boxes()) == 0
    assert len(scene[1].load_boxes(sensor_id="top")) == 0
    with pytest.raises(AnnotationNotFoundError, match="No 3D boxes"):
        scene[1].load_boxes(strict=True)
    with pytest.raises(AnnotationNotFoundError, match="No 3D boxes"):
        scene[1].load_boxes(sensor_id="top", strict=True)

    uncovered_sensor_row = {
        **sensor_row,
        "timestamp_ns": 200,
        "frame_index": 1,
    }
    _write_jsonl(sensor_box_path, [sensor_row, uncovered_sensor_row])
    scene = FZIAURADataset(root, split="train")[0]
    errors = []
    _validate_scene(scene, errors)
    assert any("uncovered sample timestamp 200" in error for error in errors)

    _write_jsonl(sensor_box_path, [sensor_row])
    samples[0]["labels"]["boxes_3d_sensor_frame"] = {}
    _write_jsonl(samples_path, samples)
    scene = FZIAURADataset(root, split="train")[0]
    assert len(scene[0].load_boxes(sensor_id="top")) == 0
    with pytest.raises(AnnotationNotFoundError, match="sensor 'top'"):
        scene[0].load_boxes(sensor_id="top", strict=True)

    samples[0]["labels"]["boxes_3d_sensor_frame"] = {"top": sensor_relative}
    _write_jsonl(samples_path, samples)
    sensor_box_path.unlink()
    scene = FZIAURADataset(root, split="train")[0]
    for strict in (False, True):
        with pytest.raises(AnnotationNotFoundError, match="Declared sensor 'top'"):
            scene[0].load_boxes(sensor_id="top", strict=strict)

    base_box_path.unlink()
    scene = FZIAURADataset(root, split="train")[0]
    for strict in (False, True):
        with pytest.raises(AnnotationNotFoundError, match="Declared base-link"):
            scene[0].load_boxes(strict=strict)


def test_sensor_box_source_path_must_be_scene_consistent(tmp_path):
    from fzi_aura.cli.validate import _validate_scene

    root = make_dataset(tmp_path)
    scene_path = root / "scenes" / "scene_0"
    samples_path = scene_path / "samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    first = "labels/custom/top_boxes_a.jsonl"
    second = "labels/custom/top_boxes_b.jsonl"
    samples[0]["labels"]["boxes_3d_sensor_frame"] = {"top": first}
    samples[1]["labels"]["boxes_3d_sensor_frame"] = {"top": second}
    _write_jsonl(samples_path, samples)

    scene = FZIAURADataset(root, split="train")[0]
    with pytest.raises(FZIAURAError, match="conflicting 3D box paths"):
        scene.boxes_at(100, sensor_id="top")

    errors = []
    _validate_scene(scene, errors)
    assert any("conflicting 3D box paths for sensor 'top'" in error for error in errors)


def test_box_offset_index_rejects_corrupt_order_frame_and_count(tmp_path):
    from fzi_aura.cli.validate import _validate_scene

    root = make_dataset(tmp_path)
    scene_path = root / "scenes" / "scene_0"
    box_path = scene_path / "labels" / "boxes_3d.jsonl"
    original = json.loads(box_path.read_text().strip())
    later = dict(original)
    later["timestamp_ns"] = 200
    later["frame_index"] = 1
    _write_jsonl(box_path, [later, original])
    scene = FZIAURADataset(root, split="train")[0]
    with pytest.raises(FZIAURAError, match="non-decreasing"):
        scene[0].load_boxes(strict=True)
    errors = []
    _validate_scene(scene, errors)
    assert any("non-decreasing" in error for error in errors)

    wrong_frame = dict(original)
    wrong_frame["box"] = dict(original["box"], frame="wrong")
    _write_jsonl(box_path, [wrong_frame])
    scene = FZIAURADataset(root, split="train")[0]
    with pytest.raises(FZIAURAError, match="box.frame='wrong'"):
        scene[0].load_boxes(strict=True)
    errors = []
    _validate_scene(scene, errors)
    assert any("box.frame='wrong'" in error for error in errors)

    wrong_frame_index = dict(original)
    wrong_frame_index["frame_index"] = 1
    _write_jsonl(box_path, [wrong_frame_index])
    scene = FZIAURADataset(root, split="train")[0]
    with pytest.raises(FZIAURAError, match="does not match requesting sample"):
        scene[0].load_boxes(strict=True)
    errors = []
    _validate_scene(scene, errors)
    assert any(
        "does not match samples.jsonl frame_index=0" in error for error in errors
    )

    _write_jsonl(box_path, [original])
    samples_path = scene_path / "samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    samples[0]["labels"]["boxes_3d_count"] = 2
    _write_jsonl(samples_path, samples)
    scene = FZIAURADataset(root, split="train")[0]
    with pytest.raises(FZIAURAError, match="declares boxes_3d_count=2"):
        scene[0].load_boxes()
    errors = []
    _validate_scene(scene, errors)
    assert any("boxes_3d_count=2" in error for error in errors)


def test_calibration_sensor_composition(tmpdir):
    path = Path(str(tmpdir)) / "calibration.json"
    _write_json(
        path,
        {
            "sensors": {
                "lidar/a": {
                    "modality": "lidar",
                    "sensor_id": "a",
                    "frame": "lidar_a_frame",
                    "parent_frame": "base_link",
                    "transform_base_link_from_sensor": {
                        "translation_xyz": [1, 0, 0],
                        "rotation_xyzw": [0, 0, 0, 1],
                    },
                },
                "lidar/b": {
                    "modality": "lidar",
                    "sensor_id": "b",
                    "frame": "lidar_b_frame",
                    "parent_frame": "base_link",
                    "transform_base_link_from_sensor": {
                        "translation_xyz": [3, 0, 0],
                        "rotation_xyzw": [0, 0, 0, 1],
                    },
                },
                "camera/front": {
                    "modality": "camera",
                    "sensor_id": "front",
                    "frame": "camera_front_frame",
                    "parent_frame": "base_link",
                    "transform_base_link_from_sensor": {
                        "translation_xyz": [0, 0, 0],
                        "rotation_xyzw": [0, 0, 0, 1],
                    },
                    "intrinsics": {"P": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]},
                },
            },
        },
    )
    calib = Calibration(path)

    assert calib.base_from_sensor("lidar/a")[:3, 3].tolist() == [1.0, 0.0, 0.0]
    assert calib.sensor_from_base("lidar/b")[:3, 3].tolist() == [-3.0, 0.0, 0.0]
    lidar_a_points = np.asarray([[0.0, 0.0, 0.0]])
    lidar_b_points = calib.transform(lidar_a_points, "lidar/b", "lidar/a")
    assert lidar_b_points.tolist() == [[-2.0, 0.0, 0.0]]
    assert np.allclose(
        calib.transform_points(
            lidar_a_points, dst_frame="lidar/b", src_frame="lidar/a"
        ),
        lidar_b_points,
    )
    assert calib.camera_projection_matrix("front").shape == (3, 4)


def test_frame_projection_fusion_and_scene_accumulation(tmpdir):
    root = make_dataset(Path(str(tmpdir)))
    scene = FZIAURADataset(root, split="train")[0]
    frame = scene[0]

    assert np.allclose(
        scene.ego_velocity(0, target_frame="odom"), [-100000000.0, 0.0, 0.0]
    )
    assert np.allclose(
        frame.ego_velocity(target_frame="base_link"), [-100000000.0, 0.0, 0.0]
    )
    assert frame.ego_trajectory([0, 1], target_frame="base_link").tolist() == [
        [0.0, 0.0, 0.0],
        [-10.0, 0.0, 0.0],
    ]
    assert frame.relative_ego_pose(1)[:3, 3].tolist() == [-10.0, 0.0, 0.0]

    points_base = frame.lidar_points_in_frame("top")
    assert points_base.shape == (2, 3)

    projection = frame.project_lidar_to_camera("top", "front")
    assert projection["uv"].shape == (2, 2)
    assert projection["depth"].shape == (2,)

    semantic_projection = frame.project_lidar_semantics_to_camera("top", "front")
    assert semantic_projection["uv"].shape == (2, 2)
    assert semantic_projection["semantic_id"].tolist() == [1, 7]
    assert semantic_projection["colors"].shape == (2, 3)

    box_projection = frame.project_boxes_to_camera("front")
    assert len(box_projection) == 1
    assert box_projection[0]["uv"].shape == (8, 2)
    sensor_box = Box3D(
        center=np.zeros(3),
        size_lwh=np.ones(3),
        rotation_xyzw=np.array([0, 0, 0, 1], dtype=np.float64),
        category="car",
        object_id="sensor-frame",
        timestamp_ns=frame.timestamp_ns,
        frame_index=frame.frame_index,
        frame="lidar/top",
        sensor_id="top",
    )
    sensor_box_projection = frame.project_boxes_to_camera(
        "front", boxes=BoxCollection([sensor_box], "lidar/top", frame.timestamp_ns)
    )
    assert len(sensor_box_projection) == 1
    assert sensor_box_projection[0]["box"] is sensor_box
    assert sensor_box_projection[0]["uv"].shape == (8, 2)

    anchor = scene[1]
    transformed_sensor_box = anchor.transform_box(sensor_box)
    assert transformed_sensor_box.center.tolist() == [10.0, 0.0, 0.0]
    assert transformed_sensor_box.frame == "base_link"
    assert transformed_sensor_box.sensor_id == "top"
    transformed_box = anchor.transform_box(frame.load_boxes().boxes[0])
    assert transformed_box.center.tolist() == [10.0, 0.0, 0.0]
    assert transformed_box.frame == "base_link"
    assert transformed_box.object_id == "car:1"
    assert anchor.transform_box(
        frame.load_boxes().boxes[0], target_frame="odom"
    ).center.tolist() == [10.0, 0.0, 0.0]
    trajectories = anchor.object_trajectories([-1, 0])
    assert list(trajectories) == ["car:1"]
    assert trajectories["car:1"][0].center.tolist() == [10.0, 0.0, 0.0]
    assert anchor.object_trajectories([-1, 0], category="person") == {}

    fused = frame.fuse_lidar_semantics()
    assert len(fused) == 2
    assert fused.frame == "base_link"
    assert fused.semantic_id.tolist() == [1, 7]
    assert fused.colors().shape == (2, 3)
    empty_fused = scene[1].fuse_lidar_semantics()
    assert len(empty_fused) == 0
    assert empty_fused.xyz.shape == (0, 3)

    accumulated = scene.accumulate_lidar("top", max_frames=1)
    assert len(accumulated) == 2
    assert accumulated.frame == "odom"
    assert accumulated.frame_index.tolist() == [0, 0]
    assert accumulated.field_names == ("x", "y", "z", "intensity")
    assert accumulated.field("x").tolist() == accumulated.xyz[:, 0].tolist()
    assert accumulated.field("x").tolist() == [11.0, 111.0]
    assert accumulated.field("intensity").tolist() == [4, 104]
    assert accumulated.has_field("intensity")

    accumulated_selected = scene.accumulate_lidar(
        "top", max_frames=1, fields=["intensity"]
    )
    assert accumulated_selected.field_names == ("intensity",)
    assert accumulated_selected.field("intensity").tolist() == [4, 104]

    assert semantic_color_map()[7] == (0, 0, 142)
    assert semantic_colors(np.asarray([7]), normalize=False).tolist() == [[0, 0, 142]]
