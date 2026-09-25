<h1 align="center">
  <img src="https://raw.githubusercontent.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/main/assets/FZI_Logo.svg" style="height: 1em; vertical-align: middle;">
  FZI-AURA SDK
</h1>

<div align="center">
  Python tools for downloading, validating, and working with the
  FZI-AURA dataset.
</div>

<div align="center">
  <a href="https://huggingface.co/datasets/fzi-forschungszentrum-informatik/FZI-AURA">Dataset</a>
  · <a href="https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/notebooks/00_quickstart.ipynb">Quickstart notebook</a>
  · <a href="https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/issues">Issues</a>
</div>

## Compatibility

FZI-AURA SDK 1.0.x supports:

- FZI-AURA dataset release v1.X
- FZI-AURA data format v1.2.1
- Python 3.8+

## Installation

Install the SDK from PyPI with:

```bash
python -m pip install fzi-aura
```

The downloader is an optional dependency:

```bash
python -m pip install "fzi-aura[download]"
```

Visualization packages used by the notebooks can be installed with:

```bash
python -m pip install "fzi-aura[vis]"
```

For development from a repository checkout, install the development,
downloader, and notebook visualization dependencies together:

```bash
python -m pip install -e ".[dev,download,vis]"
```

## Download FZI-AURA

Accept the dataset terms on
[Hugging Face](https://huggingface.co/datasets/fzi-forschungszentrum-informatik/FZI-AURA)
and sign in:

```bash
hf auth login
```

Download and extract the published keyframe data:

```bash
fzi-aura-download /data/fzi-aura
```

The downloader can select splits, scenes, and data layers. Use `--dry-run` to
inspect a selection before downloading it:

```bash
fzi-aura-download /data/fzi-aura \
  --splits train \
  --layers camera_keyframes,lidar_raw_keyframes \
  --dry-run
```

Data is published in complete scene blocks, so a scene selection may include
neighboring scenes from the same block. Re-running the downloader picks up newly
published blocks and keeps data that is already present.

Set the dataset root for the examples and notebooks:

```bash
export FZI_AURA_ROOT=/data/fzi-aura
```

See [Downloading FZI-AURA](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/docs/downloading.md) for layer selection, archive
retention, verification, parallel extraction, and archive mounting.

## Quickstart

```python
import os

from fzi_aura import FZIAURADataset

dataset = FZIAURADataset(os.environ["FZI_AURA_ROOT"], split="train")
scene = dataset[0]
frame = scene[0]

print(scene.scene_id, len(scene))
print(frame.timestamp_ns)
print(frame.available_cameras())
print(frame.available_lidars())
```

`FZIAURADataset` provides scene-level access. A scene contains lazy frame
objects, and sensor data is read when a `load_*` method is called.

```text
FZIAURADataset
└── FZIAURAScene
    └── FZIAURAFrame
```

Use `dataset.as_frames()` when you need a flat frame index across a split:

```python
frames = dataset.as_frames(sample_filter="any_label")
print(len(frames))
```

## Working with the dataset

Frame filters make it easy to select samples with the data required by a task.
For example, select a frame with a front camera and motion-compensated LiDAR:

```python
frames = scene.frames(
    sample_filter="sensor_available",
    require_cameras=["front_medium"],
    require_lidars=["top_left"],
)

frame = frames[0]
image = frame.load_camera("front_medium")
cloud = frame.load_lidar("top_left", stage="motion_compensated")
```

Load 3D boxes from an annotated frame:

```python
box_frame = scene.frames(sample_filter="boxes_3d")[0]
boxes = box_frame.load_boxes(frame="base_link")
```

Load point-aligned semantic and instance labels:

```python
semantic_frame = scene.frames(
    sample_filter="semantic_lidar",
    require_lidars=["top_left"],
    require_semantics_for=["top_left"],
)[0]

cloud, labels = semantic_frame.load_lidar_semantic_pair(
    "top_left", stage="motion_compensated"
)
```

## Data conventions

- Sensor data and 3D boxes can be transformed through the ROS `base_link`
  frame: X points forward, Y left, and Z up.
- Raw and motion-compensated LiDAR may contain different point fields. Inspect
  `cloud.field_names` before selecting fields.
- Semantic labels follow LiDAR point order. Use
  `load_lidar_semantic_pair(...)` to load and check the pair together.
- The sample stream runs at 10 Hz and annotations are provided on 2 Hz
  keyframes. An annotated frame may contain zero boxes.
- Object IDs identify tracks within one scene.

The notebooks cover calibration, coordinate transforms, camera projection, ego
poses, and temporal trajectories with complete examples.

See [FZI-AURA data format](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/docs/data-format.md) for the public dataset layout,
sample index, sensor files, annotations, calibration, and ego data.
Transform directions, temporal alignment, and projection conventions are
covered in [Coordinate systems and transforms](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/docs/coordinate-systems.md).

## Validation

Validate an extracted or selectively downloaded dataset with:

```bash
fzi-aura-validate "$FZI_AURA_ROOT"
```

The validator checks the dataset structure, metadata, installed sensor files,
annotations, calibration, and point-label alignment.

## Examples and notebooks

- [`examples/quickstart.py`](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/examples/quickstart.py): compact SDK example
- [`notebooks/00_quickstart.ipynb`](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/notebooks/00_quickstart.ipynb): dataset,
  scene, and frame access
- [`notebooks/01_sensor_data_and_pcd_fields.ipynb`](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/notebooks/01_sensor_data_and_pcd_fields.ipynb): camera, LiDAR, radar, and PCD fields
- [`notebooks/02_annotations_calibration_projection.ipynb`](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/notebooks/02_annotations_calibration_projection.ipynb):
  boxes, semantic labels, calibration, projection, and point-cloud fusion
- [`notebooks/03_context_vehicle_state_and_trajectories.ipynb`](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/notebooks/03_context_vehicle_state_and_trajectories.ipynb):
  context, vehicle state, ego motion, and object trajectories

## Contributing and support

Use [GitHub Issues](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/issues)
for bugs and usage questions. Pull requests are welcome.

The release process is documented in
[docs/releasing.md](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/docs/releasing.md).

## License

The SDK is licensed under [Apache License 2.0](https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk/blob/main/LICENSE). The FZI-AURA dataset is distributed separately and is subject to its own
license. See the [FZI-AURA dataset page](https://huggingface.co/datasets/fzi-forschungszentrum-informatik/FZI-AURA#license-and-third-party-data)
for dataset licensing information.

## Citation

Please use the citation provided in the [FZI-AURA dataset card](https://huggingface.co/datasets/fzi-forschungszentrum-informatik/FZI-AURA/blob/main/README.md#citation).
