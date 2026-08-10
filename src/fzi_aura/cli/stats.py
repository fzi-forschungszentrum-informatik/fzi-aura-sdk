from __future__ import annotations

import argparse
import json

from fzi_aura import FZIAURADataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--split", default=None)
    ap.add_argument("--split-version", default="v1.0")
    args = ap.parse_args()
    ds = FZIAURADataset(args.root, split=args.split, split_version=args.split_version)
    frames = ds.as_frames(sample_filter="all", return_mode="dict")
    stats = ds.stats()
    stats["frames"] = frames.stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
