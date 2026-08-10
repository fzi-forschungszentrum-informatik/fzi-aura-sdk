# Coordinate systems and transforms

FZI-AURA uses static sensor calibration together with a time-dependent ego pose.
Static calibration moves data between a sensor and the vehicle. Ego poses move
the vehicle between timestamps or into the scene-wide frame.

## Coordinate frames

```text
sensor frame
    │  transform_base_link_from_sensor
    ▼
base_link at timestamp t
    │  ego pose at timestamp t
    ▼
  odom
```

Sensor frames belong to individual cameras, LiDARs, and radars. Their names and
extrinsics are stored in each scene's `calibration.json`.

`base_link` is the vehicle frame and uses the standard ROS axes:

- X points forward
- Y points left
- Z points up

`odom` is the scene-wide frame used for ego motion and temporal alignment.

## Transform direction

Transform names state the destination first and the source second. For column
vectors:

```text
p_destination = T_destination_from_source @ p_source
```

The calibration entry `transform_base_link_from_sensor` therefore maps sensor
coordinates into `base_link`.

```python
calibration = frame.calibration()

base_from_lidar = calibration.base_from_sensor("lidar/top_left")
lidar_from_base = calibration.sensor_from_base("lidar/top_left")
```

`frame.load_ego_pose()` returns the transform from `base_link` at that frame's
timestamp into `odom`:

```python
odom_from_base = frame.load_ego_pose()
```

## Transforming between sensors

Sensor-to-sensor transforms are composed through `base_link`.

```python
camera_from_lidar = calibration.matrix(
    "camera/front_medium",
    "lidar/top_left",
)
```

The same operation can be applied directly to an array of XYZ points:

```python
points_camera = calibration.transform_points(
    cloud.xyz,
    dst_frame="camera/front_medium",
    src_frame="lidar/top_left",
)
```

Full sensor keys such as `lidar/front_left` and `radar/front_left` keep sensor
references unambiguous when modalities use the same ID.

## LiDAR coordinates and timing

Raw LiDAR points are stored in their sensor frame at acquisition time.
Motion-compensated clouds correct point positions for ego motion during the
sweep and align them to the end of that sweep.

Raw timestamps refer to sweep start. Motion-compensated timestamps refer to
sweep end, which is also the temporal target of the motion compensation. The
frame associations in `samples.jsonl` pair both stages with the corresponding
sample.

The SDK can load a cloud directly in `base_link` or `odom`:

```python
points_base = frame.lidar_points_in_frame(
    "top_left",
    target_frame="base_link",
    stage="motion_compensated",
)

points_odom = frame.lidar_points_in_frame(
    "top_left",
    target_frame="odom",
    stage="motion_compensated",
)
```

## Moving data between timestamps

Canonical 3D boxes start in `base_link` at their annotation timestamp.
Sensor-frame boxes start in the corresponding calibrated LiDAR frame. Their
serialized representation is described in
[FZI-AURA data format](data-format.md#3d-boxes).

`base_link` changes with the ego pose. Moving a box into another timestamp
therefore follows this path:

```text
base_link at box timestamp
    │  source ego pose
    ▼
  odom
    │  inverse anchor ego pose
    ▼
base_link at anchor timestamp
```

`transform_box` performs this temporal transform and can return the result in
the anchor `base_link`, `odom`, or an anchor-time sensor frame:

```python
box_frames = scene.frames(sample_filter="boxes_3d")
source_frame = next(frame for frame in box_frames if frame.boxes_3d_count)
box = next(iter(source_frame.load_boxes()))

anchor_frame = scene[len(scene) // 2]
box_at_anchor = anchor_frame.transform_box(box, target_frame="base_link")
```

Ego-motion helpers use frame offsets within the scene:

```python
offset_pose_in_anchor = anchor_frame.relative_ego_pose(offset=5)
positions_in_anchor = anchor_frame.ego_trajectory(
    offsets=(-10, -5, 0, 5, 10),
    target_frame="base_link",
)
```

## Camera projection

Camera projection first transforms points into the camera frame and then applies
the camera projection matrix `P`. Points with positive camera depth can be
projected into image coordinates.

```python
projection = frame.project_lidar_to_camera(
    "top_left",
    "front_medium",
    stage="motion_compensated",
)

uv = projection["uv"]
depth = projection["depth"]
```

Semantic projection can also select points inside the image:

```python
image = frame.load_camera("front_medium")
projection = frame.project_lidar_semantics_to_camera(
    "top_left",
    "front_medium",
    image_shape=image.shape,
)
```

The same calibration is used by `frame.project_boxes_to_camera(...)`.

## Practical rules

- Read transform names as `destination_from_source`.
- Use static calibration for sensor changes at one timestamp.
- Use ego poses when moving data between timestamps.
- Use complete sensor keys when a bare sensor ID is shared across modalities.
- Keep the box timestamp when building object tracks or temporal targets.

The calibration and projection notebook provides complete visual examples:
[`notebooks/02_annotations_calibration_projection.ipynb`](../notebooks/02_annotations_calibration_projection.ipynb).

Temporal box and ego-motion workflows are covered in:
[`notebooks/03_context_vehicle_state_and_trajectories.ipynb`](../notebooks/03_context_vehicle_state_and_trajectories.ipynb).
