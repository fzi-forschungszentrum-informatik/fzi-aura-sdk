from __future__ import annotations

import argparse
import json
from collections import Counter

from fzi_aura import FZIAURADataset, read_pcd_header


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--split", default=None)
    ap.add_argument("--split-version", default="v1.0")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    frames = FZIAURADataset(
        args.root, split=args.split, split_version=args.split_version
    ).as_frames(return_mode="frame")
    counts: Counter[str] = Counter()
    inspected = 0
    for frame in frames.iter_frames():
        for sensor in frame.available_lidars("motion_compensated"):
            schema = read_pcd_header(frame.lidar_path(sensor, "motion_compensated"))
            counts[f"lidar/motion_compensated/{sensor}:{','.join(schema.fields)}"] += 1
            inspected += 1
            if inspected >= args.limit:
                print(json.dumps(dict(counts), indent=2))
                return
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
