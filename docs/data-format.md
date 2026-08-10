# FZI-AURA data format

FZI-AURA is organized as a dataset index followed by self-contained scene
directories. Paths stored in the metadata are relative to the corresponding scene directory.

The SDK is the easiest way to work with these files. This page is a reference
for users who also need to inspect the on-disk data or build their own data
loader.

## Dataset layout

```text
fzi-aura/
  dataset.json
  available_data.json
  LICENSE
  THIRD_PARTY_NOTICES.MD
  splits/
    v1.0/
      train.txt
      val.txt
      test.txt
  scenes/
    <scene_directory>/
      scene.json
      samples.jsonl
      samples.parquet
      calibration.json
      camera/<camera_id>/*.jpg
      lidar/raw/<sensor_id>/*.pcd
      lidar/motion_compensated/<sensor_id>/*.pcd
      radar/<sensor_id>/*.pcd
      labels/
        boxes_3d.jsonl
        boxes_3d_sensor_frame/<sensor_id>/boxes_3d.jsonl
        semantic/<sensor_id>/*.label
        semantic/classes.json
      ego/poses.parquet
      ego/vehicle_signals.parquet
```

Selective downloads contain the same layout with only the chosen sensor and
temporal layers.

## Dataset index and splits

`dataset.json` is the entry point for a dataset root. Its `scenes` array maps
each public `scene_id` to a directory below `scenes/` and includes scene-level
sample and annotation counts.

Split files contain one `scene_id` per line. Splits are scene-level, so all
frames from a scene remain together.

The downloader writes `available_data.json` with the layers, splits, scenes,
and published blocks selected for that dataset root. The SDK and validator use
this file to distinguish installed payloads from sensor data that was not part
of the download.

## Scene metadata

Each scene begins with `scene.json`. It records the scene ID and name, sample
count, available sensor IDs, modalities, and paths to the scene's sample,
calibration, annotation, and ego-data files.

`samples.jsonl` is the frame index. Each line describes one frame and contains:

- `timestamp_ns` and `frame_index`
- camera, LiDAR, and radar paths for that frame
- 3D-box and semantic-label availability
- the ego pose at the frame timestamp
- the matching row in `ego/vehicle_signals.parquet`

Sample timestamps are stored as integer nanoseconds and ordered through the
scene. Sensor and annotation paths are grouped by sensor ID. A shortened sample
row looks like this:

```json
{
  "timestamp_ns": 1780482739200000000,
  "frame_index": 100,
  "cameras": {
    "front_medium": "camera/front_medium/1780482739_200002668.jpg"
  },
  "lidar": {
    "raw": {
      "top_left": "lidar/raw/top_left/1780482739_100295610.pcd"
    },
    "motion_compensated": {
      "top_left": "lidar/motion_compensated/top_left/1780482739_200000000.pcd"
    }
  },
  "labels": {
    "boxes_3d": true,
    "boxes_3d_count": 16,
    "semantic_lidar": {
      "top_left": "labels/semantic/top_left/1780482739_200000000.label"
    }
  }
}
```

The exact sensor IDs and paths depend on the scene and installed layers. Use
the sample associations or the SDK rather than constructing payload paths from
timestamps.

`samples.parquet` contains the same frame table in a columnar representation.

## Keyframes

The sample stream runs at 10 Hz. Annotation keyframes occur at 2 Hz and contain
3D-box coverage, semantic LiDAR labels, or both. A frame with completed 3D-box
annotation remains a keyframe when its box count is zero.

Keyframe and non-keyframe sensor payloads are separate download layers. Their
paths remain associated with the corresponding rows in `samples.jsonl`.

## Camera images

Camera images are stored as JPEG files below `camera/<camera_id>/`. The sample
index associates synchronized camera paths with each frame.

The SDK returns camera images as RGB NumPy arrays:

```python
image = frame.load_camera("front_medium")
```

## LiDAR and radar

LiDAR and radar point clouds use binary PCD files. Point fields are defined by
the header of each file and can differ between sensors and processing stages.
Inspect the fields before selecting model inputs:

```python
cloud = frame.load_lidar("top_left", stage="motion_compensated")
print(cloud.field_names)
```

LiDAR coordinates and timestamp conventions are described in
[Coordinate systems and transforms](coordinate-systems.md).

## 3D boxes

`labels/boxes_3d.jsonl` contains the canonical 3D boxes in `base_link`. Each
line represents one box:

```json
{
  "timestamp_ns": 1780482739200000000,
  "frame_index": 100,
  "category": "car",
  "object_id": "car:42",
  "box": {
    "center": [12.4, -3.1, 0.8],
    "size_lwh": [4.5, 1.9, 1.6],
    "rotation_xyzw": [0.0, 0.0, 0.12, 0.99],
    "frame": "base_link"
  }
}
```

Box centers and dimensions are measured in meters. `size_lwh` gives the full
length, width, and height along the box-local X, Y, and Z axes.
`rotation_xyzw` rotates the box from its local axes into the declared frame.

Rows are ordered by timestamp, with all boxes for one frame stored together.
The `boxes_3d_count` value in `samples.jsonl` records annotation coverage and
also represents annotated frames with zero boxes.

Object IDs form tracks within one scene. The same ID in another scene does not
refer to the same object.

The files below `labels/boxes_3d_sensor_frame/` contain boxes transformed into
the corresponding LiDAR frame and restricted to boxes containing points from
that sensor.

## Semantic LiDAR labels

Semantic labels are little-endian `uint32` arrays with one value per LiDAR
point. Each value stores a semantic class ID and an instance ID:

```python
semantic_id = value & 0xFFFF
instance_id = value >> 16
```

Semantic labeling was performed on the temporally aligned,
motion-compensated points. Raw and motion-compensated clouds preserve the same
point order, so the labels can also be applied to the corresponding raw cloud.
The SDK checks the point and label counts when loading them together:

```python
cloud, labels = frame.load_lidar_semantic_pair(
    "top_left", stage="motion_compensated"
)
```

`labels/semantic/classes.json` provides the semantic ID-to-name mapping,
detection categories, and label-array description for the scene.

## Calibration and coordinate frames

`calibration.json` contains one entry per sensor. Every entry gives the sensor
frame and its translation and quaternion relative to `base_link`. Camera entries
also provide the projection matrix `P`.

FZI-AURA uses the standard ROS `base_link` axes: X forward, Y left, and Z up.
Ego poses transform points from `base_link` into the scene-wide `odom` frame.

See [Coordinate systems and transforms](coordinate-systems.md) for transform
directions, LiDAR timing, temporal alignment, and camera projection.

The SDK composes sensor transforms through `base_link` and provides helpers for
point transformation and camera projection. See
[`notebooks/02_annotations_calibration_projection.ipynb`](../notebooks/02_annotations_calibration_projection.ipynb)
for examples.

## Ego poses and vehicle signals

`ego/poses.parquet` contains the vehicle position and orientation for every
sample timestamp. Positions are expressed in `odom`; orientations are stored as
XYZW quaternions describing the `base_link` pose in that frame.

`ego/vehicle_signals.parquet` contains timestamped vehicle state and recording
context. Sample rows reference the matching `timestamp_ns`; joins use that exact
integer timestamp.

The context and trajectory notebook shows how to select signal columns and
construct temporal windows:
[`notebooks/03_context_vehicle_state_and_trajectories.ipynb`](../notebooks/03_context_vehicle_state_and_trajectories.ipynb).

## Validation

Use the SDK validator to check metadata, payload files, annotations,
calibration, Parquet tables, and semantic point alignment:

```bash
fzi-aura-validate "$FZI_AURA_ROOT"
```
