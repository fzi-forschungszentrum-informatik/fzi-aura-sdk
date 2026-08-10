from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from fzi_aura import FZIAURADataset
from fzi_aura.availability import is_keyframe_sample
from fzi_aura.contract import PUBLIC_FORMAT_VERSION
from fzi_aura.errors import CalibrationError, SplitError
from fzi_aura.io import iter_jsonl_binary, resolve_scene_path
from fzi_aura.pointcloud import read_pcd_header

CANONICAL_EGO_POSES = "ego/poses.parquet"
CANONICAL_VEHICLE_SIGNALS = "ego/vehicle_signals.parquet"


def _payload_is_installed(scene, modality: str, sample: dict) -> bool:
    """Whether this sample's sensor payload belongs to an installed layer.

    Unpackaged exporter trees have ``layers=None`` and are therefore checked in
    full. Downloaded roots declare their installed layers in
    ``available_data.json``; references to a deliberately omitted layer must
    not be reported as missing files.
    """

    return scene.availability.allows_sensor(
        modality, is_keyframe=is_keyframe_sample(sample)
    )


def _scene_file(
    scene, relative, description: str, errors: list[str], *, required: bool = True
):
    if not isinstance(relative, str):
        errors.append(
            f"{scene.scene_id}: {description} must be a relative path string; got {relative!r}"
        )
        return None
    try:
        path = resolve_scene_path(scene.path, relative)
    except ValueError as exc:
        errors.append(f"{scene.scene_id}: invalid {description}: {exc}")
        return None
    if required and not path.is_file():
        errors.append(f"{scene.scene_id}: missing {description} file {path}")
        return None
    return path


def _check_pcd(path: Path, prefix: str, errors: list[str]) -> None:
    try:
        read_pcd_header(path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{prefix}: unreadable lidar/radar PCD header {path}: {exc}")


def _validate_sample_contract(scene, sample: dict, errors: list[str]) -> None:
    frame_index = sample.get("frame_index", "?")
    prefix = f"{scene.scene_id} frame {frame_index}"
    required = (
        "frame_index",
        "timestamp_ns",
        "lidar",
        "labels",
        "ego_pose",
        "vehicle_signals",
    )
    for key in required:
        if key not in sample:
            errors.append(f"{prefix}: missing mandatory sample field {key}")
    if not all(key in sample for key in required):
        return

    if type(sample["frame_index"]) is not int:
        errors.append(f"{prefix}: frame_index must be an integer")
    try:
        timestamp_ns = int(sample["timestamp_ns"])
    except (TypeError, ValueError):
        errors.append(f"{prefix}: timestamp_ns must be an integer")
        timestamp_ns = None

    lidar = sample["lidar"]
    if not isinstance(lidar, dict):
        errors.append(f"{prefix}: lidar must be an object")
    else:
        for stage in ("raw", "motion_compensated"):
            if stage not in lidar:
                errors.append(f"{prefix}: missing mandatory sample field lidar.{stage}")
            elif not isinstance(lidar[stage], dict):
                errors.append(f"{prefix}: lidar.{stage} must be an object")

    labels = sample["labels"]
    if not isinstance(labels, dict):
        errors.append(f"{prefix}: labels must be an object")
    else:
        if type(labels.get("semantic_segmentation")) is not bool:
            errors.append(f"{prefix}: labels.semantic_segmentation must be boolean")
        semantic_lidar = labels.get("semantic_lidar", {})
        if not isinstance(semantic_lidar, dict):
            errors.append(
                f"{prefix}: labels.semantic_lidar must be an object when present"
            )
        elif labels.get("semantic_segmentation") != bool(semantic_lidar):
            errors.append(
                f"{prefix}: labels.semantic_segmentation does not match semantic_lidar paths"
            )
        if "boxes_3d" in labels:
            covered = labels["boxes_3d"]
            count = labels.get("boxes_3d_count")
            if type(covered) is not bool:
                errors.append(f"{prefix}: labels.boxes_3d must be boolean")
            elif covered and (type(count) is not int or count < 0):
                errors.append(
                    f"{prefix}: labels.boxes_3d=true requires non-negative boxes_3d_count"
                )
            elif not covered and count is not None:
                errors.append(
                    f"{prefix}: labels.boxes_3d=false must not have boxes_3d_count"
                )
        sensor_boxes = labels.get("boxes_3d_sensor_frame", {})
        if not isinstance(sensor_boxes, dict):
            errors.append(
                f"{prefix}: labels.boxes_3d_sensor_frame must be an object when present"
            )
        else:
            for sensor_id, relative in sensor_boxes.items():
                if not isinstance(sensor_id, str) or not sensor_id:
                    errors.append(
                        f"{prefix}: boxes_3d_sensor_frame sensor IDs must be "
                        "non-empty strings"
                    )
                if not isinstance(relative, str) or not relative:
                    errors.append(
                        f"{prefix}: boxes_3d_sensor_frame/{sensor_id} must be a "
                        "relative path string"
                    )

    ego_pose = sample["ego_pose"]
    if not isinstance(ego_pose, dict):
        errors.append(f"{prefix}: ego_pose must be an object")
    else:
        if ego_pose.get("path") != CANONICAL_EGO_POSES:
            errors.append(f"{prefix}: ego_pose.path must be {CANONICAL_EGO_POSES!r}")
        for key in ("path", "position", "orientation_xyzw"):
            if key not in ego_pose:
                errors.append(
                    f"{prefix}: missing mandatory sample field ego_pose.{key}"
                )

    vehicle = sample["vehicle_signals"]
    if not isinstance(vehicle, dict):
        errors.append(f"{prefix}: vehicle_signals must be an object")
    else:
        if vehicle.get("path") != CANONICAL_VEHICLE_SIGNALS:
            errors.append(
                f"{prefix}: vehicle_signals.path must be {CANONICAL_VEHICLE_SIGNALS!r}"
            )
        if "timestamp_ns" not in vehicle:
            errors.append(
                f"{prefix}: missing mandatory sample field vehicle_signals.timestamp_ns"
            )
        elif timestamp_ns is not None:
            try:
                if int(vehicle["timestamp_ns"]) != timestamp_ns:
                    errors.append(
                        f"{prefix}: vehicle_signals.timestamp_ns must equal timestamp_ns"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"{prefix}: vehicle_signals.timestamp_ns must be an integer"
                )


def _validate_frame_paths(scene, sample: dict, errors: list[str]) -> None:
    frame_index = sample.get("frame_index", "?")
    prefix = f"{scene.scene_id} frame {frame_index}"

    if _payload_is_installed(scene, "camera", sample):
        for sensor_id, rel in sample.get("cameras", {}).items():
            path = _scene_file(scene, rel, f"camera/{sensor_id}", errors)
            if path is not None:
                try:
                    from PIL import Image

                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"{prefix}: unreadable camera/{sensor_id} file {path}: {exc}"
                    )

    lidar = sample["lidar"]
    for stage in ("raw", "motion_compensated"):
        if not _payload_is_installed(scene, f"lidar_{stage}", sample):
            continue
        for sensor_id, rel in lidar[stage].items():
            path = _scene_file(scene, rel, f"lidar/{stage}/{sensor_id}", errors)
            if path is not None:
                _check_pcd(path, prefix, errors)

    if _payload_is_installed(scene, "radar", sample):
        for sensor_id, rel in sample.get("radar", {}).items():
            path = _scene_file(scene, rel, f"radar/{sensor_id}", errors)
            if path is not None:
                _check_pcd(path, prefix, errors)

    labels = sample["labels"]
    if not scene.availability.allows_base():
        return
    for sensor_id, rel in labels.get("semantic_lidar", {}).items():
        path = _scene_file(scene, rel, f"semantic_lidar/{sensor_id}", errors)
        if path is None:
            continue
        if path.stat().st_size % 4:
            errors.append(
                f"{prefix}: semantic label file is not a uint32 array: {path}"
            )
        label_count = path.stat().st_size // 4
        for stage in ("raw", "motion_compensated"):
            if not _payload_is_installed(scene, f"lidar_{stage}", sample):
                continue
            lidar_rel = lidar[stage].get(sensor_id)
            if lidar_rel is None:
                errors.append(
                    f"{prefix}: semantic_lidar/{sensor_id} has no {stage} lidar source"
                )
                continue

            lidar_path = _scene_file(
                scene, lidar_rel, f"lidar/{stage}/{sensor_id}", errors
            )
            if lidar_path is None:
                continue
            try:
                point_count = read_pcd_header(lidar_path).points
            except Exception:
                continue
            if label_count != point_count:
                errors.append(
                    f"{prefix}: semantic_lidar/{sensor_id} has {label_count} labels "
                    f"but {stage} lidar has {point_count} points"
                )

    for sensor_id, rel in labels.get("boxes_3d_sensor_frame", {}).items():
        _scene_file(scene, rel, f"boxes_3d_sensor_frame/{sensor_id}", errors)

    _scene_file(scene, sample["ego_pose"].get("path"), "ego_pose", errors)
    _scene_file(scene, sample["vehicle_signals"].get("path"), "vehicle_signals", errors)


def _validate_semantic_classes(scene, has_semantics: bool, errors: list[str]) -> None:
    classes_path = scene.path / "labels" / "semantic" / "classes.json"
    if not has_semantics:
        return
    if not classes_path.is_file():
        errors.append(
            f"{scene.scene_id}: missing mandatory semantic class file {classes_path}"
        )
        return
    try:
        document = json.loads(classes_path.read_text(encoding="utf-8"))
        semantic_classes = document["semantic_classes"]
        if not isinstance(semantic_classes, list):
            raise TypeError
        for item in semantic_classes:
            if (
                not isinstance(item, dict)
                or type(item.get("id")) is not int
                or not isinstance(item.get("name"), str)
            ):
                raise TypeError
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        errors.append(
            f"{scene.scene_id}: invalid exporter semantic_classes file {classes_path}: {exc}"
        )


def _validate_box_file(
    path: Path,
    coverage: dict[int, tuple[int, Optional[int]]],
    errors: list[str],
    *,
    expected_frame: Optional[str],
    expected_sensor_id: Optional[str] = None,
) -> None:
    actual: dict[int, int] = {}
    previous_timestamp: Optional[int] = None
    frame_for_timestamp: Optional[int] = None
    observed_frame = expected_frame
    try:
        for line_number, _, _, row in iter_jsonl_binary(path):
            timestamp_ns = int(row["timestamp_ns"])
            frame_index = int(row["frame_index"])
            box_frame = row["box"]["frame"]
            if not isinstance(box_frame, str) or not box_frame:
                raise TypeError("box.frame must be a non-empty string")
            if previous_timestamp is not None and timestamp_ns < previous_timestamp:
                errors.append(
                    f"{path}:{line_number}: box timestamps are not "
                    "non-decreasing and contiguous"
                )
            if timestamp_ns != previous_timestamp:
                frame_for_timestamp = frame_index
            elif frame_index != frame_for_timestamp:
                errors.append(
                    f"{path}:{line_number}: timestamp_ns={timestamp_ns} has "
                    "inconsistent frame_index values"
                )
            previous_timestamp = timestamp_ns
            if observed_frame is None:
                observed_frame = box_frame
            elif box_frame != observed_frame:
                errors.append(
                    f"{path}:{line_number}: box.frame={box_frame!r}, expected "
                    f"{observed_frame!r}"
                )
            if (
                expected_sensor_id is not None
                and row.get("sensor_id") != expected_sensor_id
            ):
                errors.append(
                    f"{path}:{line_number}: sensor_id={row.get('sensor_id')!r}, "
                    f"expected {expected_sensor_id!r}"
                )
            actual[timestamp_ns] = actual.get(timestamp_ns, 0) + 1
            expected = coverage.get(timestamp_ns)
            if expected is not None and frame_index != expected[0]:
                errors.append(
                    f"{path}:{line_number}: frame_index={frame_index} does not "
                    f"match samples.jsonl frame_index={expected[0]} at "
                    f"timestamp_ns={timestamp_ns}"
                )
    except (
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"{path}: unreadable 3D box file: {exc}")
        return

    for timestamp_ns, (_, expected_count) in coverage.items():
        if expected_count is not None and actual.get(timestamp_ns, 0) != expected_count:
            errors.append(
                f"{path}: boxes_3d_count={expected_count} does not match "
                f"{actual.get(timestamp_ns, 0)} rows at {timestamp_ns}"
            )
    for timestamp_ns in sorted(set(actual) - set(coverage)):
        errors.append(
            f"{path}: box row belongs to uncovered sample timestamp {timestamp_ns}"
        )


def _validate_box_coverage(
    scene, samples: list[dict], scene_doc: dict, errors: list[str]
) -> None:
    coverage: dict[int, tuple[int, Optional[int]]] = {}
    for sample in samples:
        labels = sample.get("labels", {})
        if isinstance(labels, dict) and labels.get("boxes_3d") is True:
            coverage[int(sample["timestamp_ns"])] = (
                int(sample["frame_index"]),
                int(labels["boxes_3d_count"]),
            )

    boxes_path = scene.path / "labels" / "boxes_3d.jsonl"
    if coverage and not boxes_path.is_file():
        errors.append(
            f"{scene.scene_id}: annotated box frames require labels/boxes_3d.jsonl, including empty files"
        )
    if not boxes_path.is_file():
        return

    _validate_box_file(
        boxes_path,
        coverage,
        errors,
        expected_frame="base_link",
    )

    paths = scene_doc.get("paths", {})
    expected_path = "labels/boxes_3d.jsonl" if coverage else None
    if isinstance(paths, dict) and paths.get("boxes_3d") != expected_path:
        errors.append(
            f"{scene.path / 'scene.json'}: paths.boxes_3d does not match annotation coverage"
        )


def _validate_sensor_box_coverage(
    scene, samples: list[dict], errors: list[str]
) -> None:
    sources: dict[str, tuple[Path, dict[int, tuple[int, Optional[int]]]]] = {}
    reported_conflicts: set[tuple[str, Path]] = set()
    for sample in samples:
        labels = sample.get("labels", {})
        if not isinstance(labels, dict):
            continue
        declared = labels.get("boxes_3d_sensor_frame", {})
        if not isinstance(declared, dict):
            continue
        for sensor_id, relative in declared.items():
            if not isinstance(sensor_id, str) or not isinstance(relative, str):
                continue
            try:
                path = resolve_scene_path(scene.path, relative)
            except ValueError:
                continue
            source = sources.get(sensor_id)
            if source is None:
                coverage: dict[int, tuple[int, Optional[int]]] = {}
                sources[sensor_id] = (path, coverage)
            else:
                expected_path, coverage = source
                if path != expected_path:
                    conflict = (sensor_id, path)
                    if conflict not in reported_conflicts:
                        errors.append(
                            f"{scene.scene_id}: conflicting 3D box paths for sensor "
                            f"{sensor_id!r}: {expected_path} and {path}"
                        )
                        reported_conflicts.add(conflict)
                    continue
            if labels.get("boxes_3d") is True:
                coverage[int(sample["timestamp_ns"])] = (
                    int(sample["frame_index"]),
                    None,
                )

    for sensor_id, (path, coverage) in sources.items():
        if not path.is_file():
            continue
        try:
            sensor_key = scene.calibration().sensor_key(sensor_id, modality="lidar")
            expected_frame = scene.calibration().sensors[sensor_key]["frame"]
            if not isinstance(expected_frame, str) or not expected_frame:
                raise TypeError("calibration frame must be a non-empty string")
        except (CalibrationError, KeyError, TypeError) as exc:
            errors.append(
                f"{path}: sensor-frame boxes reference uncalibrated lidar "
                f"sensor {sensor_id!r}: {exc}"
            )
            continue
        _validate_box_file(
            path,
            coverage,
            errors,
            expected_frame=expected_frame,
            expected_sensor_id=sensor_id,
        )


def _validate_scene(scene, errors: list[str]) -> None:
    if not scene.path.is_dir():
        errors.append(f"{scene.scene_id}: scene path does not exist: {scene.path}")
        return

    required_files = (
        "scene.json",
        "samples.jsonl",
        "samples.parquet",
        "calibration.json",
        "ego/poses.parquet",
        CANONICAL_VEHICLE_SIGNALS,
        "labels/box_sources.json",
    )
    for name in required_files:
        if not (scene.path / name).is_file():
            errors.append(f"{scene.scene_id}: missing {scene.path / name}")

    try:
        scene_doc = scene.load_metadata()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{scene.scene_id}: failed to load scene.json: {exc}")
        scene_doc = {}
    if isinstance(scene_doc, dict):
        paths = scene_doc.get("paths", {})
        if not isinstance(paths, dict):
            errors.append(f"{scene.scene_id}: scene.json paths must be an object")
        else:
            for name in (
                "samples",
                "samples_parquet",
                "calibration",
                "ego_poses",
                "vehicle_signals",
                "box_sources",
            ):
                if name not in paths or paths[name] is None:
                    errors.append(
                        f"{scene.scene_id}: scene.json paths.{name} is mandatory"
                    )
            for name, relative in paths.items():
                if relative is not None:
                    _scene_file(scene, relative, f"scene path {name}", errors)
            if paths.get("samples") != "samples.jsonl":
                errors.append(
                    f"{scene.scene_id}: scene.json paths.samples must be 'samples.jsonl'"
                )
            if paths.get("samples_parquet") != "samples.parquet":
                errors.append(
                    f"{scene.scene_id}: scene.json paths.samples_parquet must be 'samples.parquet'"
                )
            if paths.get("calibration") != "calibration.json":
                errors.append(
                    f"{scene.scene_id}: scene.json paths.calibration must be 'calibration.json'"
                )
            if paths.get("ego_poses") != CANONICAL_EGO_POSES:
                errors.append(
                    f"{scene.scene_id}: scene.json paths.ego_poses must be "
                    f"{CANONICAL_EGO_POSES!r}"
                )
            if paths.get("vehicle_signals") != CANONICAL_VEHICLE_SIGNALS:
                errors.append(
                    f"{scene.scene_id}: scene.json paths.vehicle_signals must be "
                    f"{CANONICAL_VEHICLE_SIGNALS!r}"
                )
            if paths.get("box_sources") != "labels/box_sources.json":
                errors.append(
                    f"{scene.scene_id}: scene.json paths.box_sources must be "
                    "'labels/box_sources.json'"
                )

    try:
        calibration = scene.calibration()
        calibration_id = calibration.data.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id.startswith(
            "calib_"
        ):
            errors.append(
                f"{scene.scene_id}: calibration.json calibration_id must start with 'calib_'"
            )
        if calibration.data.get("format_version") != PUBLIC_FORMAT_VERSION:
            errors.append(
                f"{scene.scene_id}: calibration.json format_version must be "
                f"{PUBLIC_FORMAT_VERSION!r}"
            )
        coordinate_system = calibration.data.get("coordinate_system")
        if not isinstance(coordinate_system, dict):
            errors.append(
                f"{scene.scene_id}: calibration.json coordinate_system must be an object"
            )
        else:
            expected_coordinate_system = {
                "world_frame": "odom",
                "ego_frame": "base_link",
                "box_frame": "base_link",
            }
            if coordinate_system != expected_coordinate_system:
                errors.append(
                    f"{scene.scene_id}: calibration.json coordinate_system must be "
                    f"{expected_coordinate_system!r}"
                )
        if "frame_transforms" in calibration.data:
            errors.append(
                f"{scene.scene_id}: calibration.json contains unsupported frame_transforms"
            )
        for sensor_key, sensor in calibration.sensors.items():
            if sensor.get("parent_frame") != "base_link":
                errors.append(
                    f"{scene.scene_id}: {sensor_key} must have parent_frame='base_link'"
                )
            if not isinstance(sensor.get("frame"), str):
                errors.append(
                    f"{scene.scene_id}: {sensor_key} is missing calibration frame"
                )
            if "transform_parent_from_sensor" in sensor:
                errors.append(
                    f"{scene.scene_id}: {sensor_key} uses unsupported transform_parent_from_sensor"
                )
            calibration.base_from_sensor(sensor_key)
            if sensor.get("modality") == "camera":
                calibration.camera_projection_matrix(sensor_key)
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"{scene.scene_id}: failed to load public calibration.json: {exc}"
        )

    try:
        samples = scene.load_samples()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{scene.scene_id}: failed to load samples.jsonl: {exc}")
        return

    expected = int(scene.metadata.get("counts", {}).get("samples", len(samples)))
    if expected != len(samples):
        errors.append(
            f"{scene.scene_id}: counts.samples={expected} but samples.jsonl has {len(samples)} rows"
        )
    timestamps = []
    has_semantics = False
    for sample in samples:
        if not isinstance(sample, dict):
            errors.append(f"{scene.scene_id}: sample row is not an object")
            continue
        _validate_sample_contract(scene, sample, errors)
        if "timestamp_ns" in sample:
            try:
                timestamps.append(int(sample["timestamp_ns"]))
            except (TypeError, ValueError):
                pass
        labels = sample.get("labels", {})
        if isinstance(labels, dict):
            has_semantics = has_semantics or bool(labels.get("semantic_lidar"))
        if all(
            key in sample for key in ("lidar", "labels", "ego_pose", "vehicle_signals")
        ):
            try:
                _validate_frame_paths(scene, sample, errors)
            except (KeyError, TypeError, AttributeError) as exc:
                errors.append(f"{scene.scene_id}: malformed sample path fields: {exc}")

    if timestamps != sorted(timestamps):
        errors.append(f"{scene.scene_id}: sample timestamps are not monotonic")

    vehicle_path = scene.path / CANONICAL_VEHICLE_SIGNALS
    if vehicle_path.is_file():
        try:
            vehicle_df = pd.read_parquet(vehicle_path)
            if "timestamp_ns" not in vehicle_df.columns:
                errors.append(f"{vehicle_path}: missing timestamp_ns column")
            else:
                vehicle_timestamps = set(
                    int(value) for value in vehicle_df["timestamp_ns"]
                )
                for sample in samples:
                    vehicle = sample.get("vehicle_signals", {})
                    if isinstance(vehicle, dict) and "timestamp_ns" in vehicle:
                        timestamp_ns = int(vehicle["timestamp_ns"])
                        if timestamp_ns not in vehicle_timestamps:
                            errors.append(
                                f"{vehicle_path}: missing exact row for timestamp_ns={timestamp_ns}"
                            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{vehicle_path}: cannot read parquet: {exc}")

    samples_parquet = scene.path / "samples.parquet"
    if samples_parquet.is_file():
        try:
            parquet_rows = pd.read_parquet(samples_parquet).to_dict("records")
            if len(parquet_rows) != len(samples):
                errors.append(
                    f"{samples_parquet}: row count does not match samples.jsonl"
                )
            elif [row.get("timestamp_ns") for row in parquet_rows] != [
                row.get("timestamp_ns") for row in samples
            ]:
                errors.append(
                    f"{samples_parquet}: timestamps do not match samples.jsonl"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{samples_parquet}: cannot read parquet: {exc}")

    _validate_semantic_classes(scene, has_semantics, errors)
    _validate_box_coverage(scene, samples, scene_doc, errors)
    _validate_sensor_box_coverage(scene, samples, errors)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate the FZI-AURA public exporter v1.2.1 contract."
    )
    ap.add_argument("root")
    ap.add_argument("--split-version", default="v1.0")
    ap.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Limit scene file checks for quick local smoke tests. Default checks all scenes.",
    )
    args = ap.parse_args()
    errors: list[str] = []
    try:
        ds = FZIAURADataset(args.root, split=None, split_version=args.split_version)
    except Exception as exc:  # noqa: BLE001
        result = {
            "root": str(Path(args.root)),
            "format_version": PUBLIC_FORMAT_VERSION,
            "scene_files_checked": 0,
            "ok": False,
            "errors": [str(exc)],
        }
        print(json.dumps(result, indent=2))
        sys.exit(1)

    scene_rows = list(ds.metadata.get("scenes", []))
    ids = [row.get("scene_id") for row in scene_rows]
    duplicate_ids = sorted({scene_id for scene_id in ids if ids.count(scene_id) > 1})
    if duplicate_ids:
        errors.append(f"dataset.json contains duplicate scene IDs: {duplicate_ids[:5]}")

    try:
        split_result = ds.validate_splits()
    except SplitError as exc:
        split_result = {"ok": False, "errors": [str(exc)]}
        errors.append(str(exc))

    checked = 0
    for scene in ds.iter_scenes():
        _validate_scene(scene, errors)
        checked += 1
        if args.max_scenes is not None and checked >= args.max_scenes:
            break
    result = {
        "root": str(Path(args.root)),
        "format_version": PUBLIC_FORMAT_VERSION,
        "split_version": args.split_version,
        "split_validation": split_result,
        "scene_files_checked": checked,
        "scene_file_check_scope": "all" if args.max_scenes is None else "limited",
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
