# Downloading FZI-AURA

The FZI-AURA downloader selects data from the release on Hugging Face, downloads
the required archives, and prepares a dataset root for the SDK.

## Setup

Install the downloader from PyPI:

```bash
python -m pip install "fzi-aura[download]"
```

Accept the dataset terms on the
[FZI-AURA dataset page](https://huggingface.co/datasets/fzi-forschungszentrum-informatik/FZI-AURA)
and sign in:

```bash
hf auth login
```

## Default download

```bash
fzi-aura-download /data/fzi-aura
```

This downloads the published keyframe layers for the train, validation, and
test splits. The archives are extracted in parallel, and each archive is removed
immediately after it has been extracted successfully.

This keeps disk use close to the archive or extracted dataset size plus the
archives currently being processed, rather than retaining a complete archive
copy alongside the extracted dataset.

The resulting directory can be used directly with the SDK:

```bash
export FZI_AURA_ROOT=/data/fzi-aura
```

```python
import os

from fzi_aura import FZIAURADataset

dataset = FZIAURADataset(os.environ["FZI_AURA_ROOT"], split="train")
```

## Data layers

| Layer | Contents |
| --- | --- |
| `base_keyframes` | Scene metadata, indexes, calibration, ego poses, vehicle signals, 3D boxes, semantic labels, and class metadata |
| `camera_keyframes` | Camera images at annotation keyframes |
| `lidar_motion_compensated_keyframes` | Motion-compensated LiDAR at annotation keyframes |
| `lidar_raw_keyframes` | Raw LiDAR at annotation keyframes |
| `radar_keyframes` | Radar point clouds at annotation keyframes |
| `camera_nonkeyframes` | Camera images between annotation keyframes |
| `lidar_raw_nonkeyframes` | Raw LiDAR between annotation keyframes |
| `radar_nonkeyframes` | Radar point clouds between annotation keyframes |

The default selection consists of `base_keyframes` and the four sensor
keyframe layers. `base_keyframes` is added to every selection because it
contains the metadata needed to use the downloaded sensor data.

LiDAR layers are downloaded as compressed `.tar.xz` archives. Normal mode
extracts them into PCD files, while mount mode stores them as uncompressed
`.tar` files. In either layout, plan 40–60% more disk space than the compressed
download size reported for the LiDAR layers.

Select layers with a comma-separated list:

```bash
fzi-aura-download /data/fzi-aura \
  --layers camera_keyframes,lidar_motion_compensated_keyframes
```

For dense camera and raw-LiDAR sequences, add their non-keyframe layers:

```bash
fzi-aura-download /data/fzi-aura \
  --layers camera_keyframes,lidar_raw_keyframes,camera_nonkeyframes,lidar_raw_nonkeyframes
```

When adding data to an existing selective download, make the new command
describe the complete dataset root you want: retain the earlier layers and
splits or scenes, then add the new selection. The selection recorded in
`available_data.json` tells the SDK and validator which data belongs to that
dataset root.

## Select splits and scenes

Download one or more splits with `--splits`:

```bash
fzi-aura-download /data/fzi-aura \
  --splits train,val \
  --layers camera_keyframes,lidar_raw_keyframes
```

Individual scenes can be given on the command line:

```bash
fzi-aura-download /data/fzi-aura \
  --splits train \
  --scenes "2025-06-11-08-18-57|1,2025-06-11-08-18-57|2" \
  --layers camera_keyframes,lidar_raw_keyframes
```

For longer selections, put one scene ID on each line of a text file:

```bash
fzi-aura-download /data/fzi-aura \
  --splits train \
  --scene-ids-file scenes.txt \
  --layers camera_keyframes,lidar_raw_keyframes
```

FZI-AURA is published in synchronized scene blocks. Selecting a scene downloads
its complete block, which can include up to 19 neighboring scenes. The command summary
reports requested scenes, block scenes, archive count, and compressed size.

## Inspect a selection

Use `--dry-run` before a large download:

```bash
fzi-aura-download /data/fzi-aura \
  --layers camera_keyframes,lidar_raw_keyframes \
  --dry-run
```

The downloader fetches the release metadata and prints the resolved selection
without downloading its data archives.

For a reproducible selection, pin a Hugging Face tag or commit:

```bash
fzi-aura-download /data/fzi-aura \
  --revision <tag-or-commit> \
  --layers camera_keyframes,lidar_raw_keyframes \
  --dry-run
```

## Extraction and verification

Extraction uses four workers by default. Set `--jobs` for the storage system
and archive sizes in use:

```bash
fzi-aura-download /data/fzi-aura \
  --layers camera_keyframes,lidar_raw_keyframes \
  --jobs 8
```

Add `--verify` to check each archive against the SHA-256 value in the release
metadata before it is extracted:

```bash
fzi-aura-download /data/fzi-aura \
  --layers camera_keyframes,lidar_raw_keyframes \
  --jobs 8 \
  --verify
```

Existing extracted files are reused when their contents match the archive. A
different file at the same path stops extraction instead of being overwritten.
An archive is retained when its extraction fails. Successfully extracted
archives are removed without waiting for the rest of the selection.

The Hugging Face download pool is controlled separately with `--max-workers`.

## Keep or reuse archives

Use `--keep-archives` to retain the downloaded archives after extraction:

```bash
fzi-aura-download /data/fzi-aura \
  --layers camera_keyframes,lidar_raw_keyframes \
  --keep-archives
```

`--no-download` runs the selection and extraction steps using release metadata
and complete archives already stored under the output directory:

```bash
fzi-aura-download /data/fzi-aura \
  --layers camera_keyframes,lidar_raw_keyframes \
  --no-download
```

This is useful after an interrupted extraction or when archives were retained
for reuse.

Re-running a normal download refreshes the release metadata and includes newly
published complete blocks. Matching extracted files remain in place.

## Mount archives

Archive mounting is intended for storage systems where extracting many small
files is undesirable. Install the mount dependencies:

```bash
python -m pip install "fzi-aura[mount]"
```

Then pass a separate mount point:

```bash
fzi-aura-download /data/fzi-aura-archives \
  --layers camera_keyframes,lidar_motion_compensated_keyframes \
  --mount /data/fzi-aura-mounted \
  --jobs 8 \
  --verify

export FZI_AURA_ROOT=/data/fzi-aura-mounted
```

Camera and metadata archives are mounted as published. Compressed LiDAR and
radar `.tar.xz` archives are converted to `.tar` before mounting. Conversion is
streamed and atomic, and archive members remain inside the tar files. The first
mount can take longer while `ratarmount` builds its indexes.

The converted tar files can be mounted again without downloading or converting
them a second time:

```bash
fzi-aura-download /data/fzi-aura-archives \
  --layers camera_keyframes,lidar_motion_compensated_keyframes \
  --no-download \
  --mount /data/fzi-aura-mounted
```

The layer and split selection must match the archives kept in the archive
directory. `--keep-archives` and `--mount` are separate workflows and cannot be
used in the same command.

## Command reference

Run the CLI help for the full option list:

```bash
fzi-aura-download --help
```
