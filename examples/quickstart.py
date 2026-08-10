from __future__ import annotations

import os

from fzi_aura import FZIAURADataset


def main() -> None:
    root = os.environ.get("FZI_AURA_ROOT")
    if root is None:
        raise SystemExit("Set FZI_AURA_ROOT to your dataset_export directory.")
    scenes = FZIAURADataset(root, split="train")
    print("scenes", len(scenes), "available layers", scenes.available_layers)
    scene = scenes[0]
    print(scene.scene_id, scene.name, len(scene))
    frame = scene[0]
    print(frame.timestamp_ns, frame.available_cameras(), frame.available_lidars())

    box_frames = scene.frames(sample_filter="boxes_3d")
    print("box frames", len(box_frames))
    if len(box_frames):
        boxes = box_frames[0].load_boxes(frame="base_link")
        print(
            "box arrays",
            {name: value.shape for name, value in boxes.as_numpy().items()},
        )

    sensor_frames = scene.frames(
        sample_filter="sensor_available",
        require_cameras=["front_medium"],
        require_lidars=["top_left"],
    )
    if len(sensor_frames):
        sensor_frame = sensor_frames[0]
        image = sensor_frame.load_camera("front_medium")
        cloud = sensor_frame.load_lidar("top_left", stage="motion_compensated")
        print("camera", image.shape, image.dtype)
        print("lidar", cloud.xyz.shape, cloud.field_names)
    else:
        print("camera/LiDAR skipped: install the corresponding keyframe layers")

    semantic_frames = scene.frames(
        sample_filter="semantic_lidar",
        require_lidars=["top_left"],
        require_semantics_for=["top_left"],
    )
    if len(semantic_frames):
        cloud, labels = semantic_frames[0].load_lidar_semantic_pair("top_left")
        print("semantic pair", cloud.xyz.shape, labels.semantic_id.shape)
    else:
        print("semantics skipped: compatible top_left labels and LiDAR are unavailable")


if __name__ == "__main__":
    main()
