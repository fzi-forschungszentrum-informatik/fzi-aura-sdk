from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from fzi_aura import (
    FZIAURADownloadError as PackageFZIAURADownloadError,
    FZIAURAError,
)
from fzi_aura.download import (
    REPOSITORY_DOCUMENTS,
    DownloadSelection,
    FZIAURADownloader,
    FZIAURADownloadError,
    _extract_archive_safely,
    build_parser,
    main,
)
from fzi_aura.errors import FZIAURADownloadError as ErrorsFZIAURADownloadError


def test_download_error_is_part_of_public_sdk_error_hierarchy() -> None:
    assert FZIAURADownloadError is ErrorsFZIAURADownloadError
    assert FZIAURADownloadError is PackageFZIAURADownloadError
    assert issubclass(FZIAURADownloadError, FZIAURAError)


def _write_metadata(root: Path) -> tuple[str, str, str]:
    metadata = root / "metadata" / "v1.0"
    metadata.mkdir(parents=True)
    scene_ids = (
        "2025-06-11-08-18-57|1",
        "2025-06-11-08-18-57|2",
        "2025-06-11-08-18-57|3",
    )
    scene_rows = []
    for ordinal, scene_id in enumerate(scene_ids, 1):
        scene_name = scene_id.replace("|", "_")
        entry = {
            "scene_id": scene_id,
            "path": f"scenes/{scene_name}",
            "counts": {"samples": 1},
        }
        scene_rows.append(
            {
                "split": "train",
                "scene_id": scene_id,
                "scene_path": entry["path"],
                "scene": scene_name,
                "scene_ordinal": ordinal,
                "scene_block": 0 if ordinal < 3 else 1,
                "dataset_scene_json": json.dumps(entry),
            }
        )
    pd.DataFrame(scene_rows).to_parquet(metadata / "scene_blocks.parquet", index=False)

    chunk_rows = []
    for block in (0, 1):
        for layer, suffix in (
            ("base_keyframes", ".tar.xz"),
            ("camera_keyframes", ".tar"),
        ):
            path = (
                f"chunks/v1.0/train/{layer}/block-{block:06d}/{layer}-{block}{suffix}"
            )
            chunk_rows.append(
                {
                    "chunk_path": path,
                    "split": "train",
                    "layer": layer,
                    "scene_block": block,
                    "status": "ready",
                    "size_bytes": 100 + block,
                    "sha256": "unused",
                }
            )
    pd.DataFrame(chunk_rows).to_parquet(metadata / "chunks.parquet", index=False)
    (metadata / "dataset.json").write_text(
        json.dumps(
            {
                "format_version": "v1.2.1",
                "scenes": [],
                "consumer_excluded_scenes": [],
            }
        ),
        encoding="utf-8",
    )
    return scene_ids


def test_fetch_metadata_requests_and_keeps_repository_documents(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["local_dir"])
        for name in REPOSITORY_DOCUMENTS:
            (output / name).write_text(name, encoding="utf-8")
        return str(output)

    class FakeHfApi:
        def __init__(self, **kwargs) -> None:
            pass

        def list_repo_files(self, **kwargs):
            return list(REPOSITORY_DOCUMENTS)

    fake_huggingface_hub = ModuleType("huggingface_hub")
    fake_huggingface_hub.snapshot_download = fake_snapshot_download
    fake_huggingface_hub.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)

    FZIAURADownloader(tmp_path).fetch_metadata()

    assert calls[0]["allow_patterns"] == [
        "metadata/v1.0/**",
        *REPOSITORY_DOCUMENTS,
    ]
    assert all((tmp_path / name).is_file() for name in REPOSITORY_DOCUMENTS)


def test_download_requests_repository_documents(tmp_path: Path, monkeypatch) -> None:
    calls = []
    archive_rel = "chunks/v1.0/train/base_keyframes/block-000000/base.tar.xz"
    selection = DownloadSelection(
        layers=("base_keyframes",),
        splits=("train",),
        requested_scene_ids=("scene",),
        block_scene_ids=("scene",),
        block_keys=(("train", 0),),
        chunk_paths=(archive_rel,),
        size_bytes=1,
    )

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["local_dir"])
        for name in REPOSITORY_DOCUMENTS:
            (output / name).write_text(name, encoding="utf-8")
        archive = output / archive_rel
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"archive")
        return str(output)

    fake_huggingface_hub = ModuleType("huggingface_hub")
    fake_huggingface_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)

    FZIAURADownloader(tmp_path).download(selection)

    assert calls[0]["allow_patterns"] == [
        "metadata/v1.0/**",
        *REPOSITORY_DOCUMENTS,
        archive_rel,
        f"{archive_rel}.chunk.json",
    ]


def test_selection_uses_only_complete_available_blocks(tmp_path: Path) -> None:
    scene_1, scene_2, scene_3 = _write_metadata(tmp_path)
    downloader = FZIAURADownloader(tmp_path)
    chunks, _ = downloader._release_tables()
    available = set(chunks[chunks["scene_block"] == 0]["chunk_path"])
    available.add(
        chunks[(chunks["scene_block"] == 1) & (chunks["layer"] == "base_keyframes")][
            "chunk_path"
        ].iloc[0]
    )

    selection = downloader.select(
        layers=("base_keyframes", "camera_keyframes"),
        splits=("train",),
        available_chunk_paths=available,
    )

    assert selection.requested_scene_ids == (scene_1, scene_2)
    assert selection.block_keys == (("train", 0),)
    assert scene_3 not in selection.requested_scene_ids


def test_explicit_scene_fails_if_its_block_is_incomplete(tmp_path: Path) -> None:
    _, _, scene_3 = _write_metadata(tmp_path)
    downloader = FZIAURADownloader(tmp_path)
    chunks, _ = downloader._release_tables()
    available = set(chunks[chunks["layer"] == "base_keyframes"]["chunk_path"])

    with pytest.raises(FZIAURADownloadError, match="not completely available"):
        downloader.select(
            layers=("base_keyframes", "camera_keyframes"),
            splits=("train",),
            scene_ids=(scene_3,),
            available_chunk_paths=available,
        )


def test_explicit_scene_accepts_path_safe_scene_name(tmp_path: Path) -> None:
    scene_1, _, _ = _write_metadata(tmp_path)
    downloader = FZIAURADownloader(tmp_path)
    chunks, _ = downloader._release_tables()

    selection = downloader.select(
        layers=("base_keyframes", "camera_keyframes"),
        splits=("train",),
        scene_ids=(scene_1.replace("|", "_"),),
        available_chunk_paths=set(chunks["chunk_path"]),
    )

    assert selection.requested_scene_ids == (scene_1,)


def _prepare_extractable_base_chunk(
    tmp_path: Path,
) -> tuple[FZIAURADownloader, DownloadSelection, Path, bytes, str]:
    scene_1, _, _ = _write_metadata(tmp_path)
    downloader = FZIAURADownloader(tmp_path)
    archive_rel = "chunks/v1.0/train/base_keyframes/block-000000/base-0.tar.xz"
    archive = tmp_path / archive_rel
    archive.parent.mkdir(parents=True)
    payload = b'{"format_version":"v1.2.1"}\n'
    info = tarfile.TarInfo("scenes/2025-06-11-08-18-57_1/scene.json")
    info.size = len(payload)
    with tarfile.open(archive, "w:xz") as tar:
        tar.addfile(info, io.BytesIO(payload))

    chunks_path = downloader.metadata_dir / "chunks.parquet"
    chunks = pd.read_parquet(chunks_path)
    extra = pd.DataFrame(
        [
            {
                "chunk_path": archive_rel,
                "split": "train",
                "layer": "base_keyframes",
                "scene_block": 0,
                "status": "ready",
                "size_bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        ]
    )
    pd.concat([chunks, extra], ignore_index=True).to_parquet(chunks_path, index=False)
    selection = DownloadSelection(
        layers=("base_keyframes",),
        splits=("train",),
        requested_scene_ids=(scene_1,),
        block_scene_ids=(scene_1,),
        block_keys=(("train", 0),),
        chunk_paths=(archive_rel,),
        size_bytes=archive.stat().st_size,
    )
    return downloader, selection, archive, payload, scene_1


def test_extracts_tar_xz_and_deletes_archives_by_default(tmp_path: Path) -> None:
    downloader, selection, archive, payload, scene_1 = _prepare_extractable_base_chunk(
        tmp_path
    )

    result = downloader.extract(selection, jobs=1, verify=True)

    assert (
        tmp_path / "scenes/2025-06-11-08-18-57_1/scene.json"
    ).read_bytes() == payload
    dataset = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))
    assert [scene["scene_id"] for scene in dataset["scenes"]] == [scene_1]
    assert (tmp_path / "splits/v1.0/train.txt").read_text(
        encoding="utf-8"
    ) == f"{scene_1}\n"
    assert not archive.exists()
    assert result["archives_deleted"] is True


def test_extract_rejects_symlinked_destination_parent(tmp_path: Path) -> None:
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "scenes").symlink_to(outside, target_is_directory=True)

    archive = tmp_path / "adversarial.tar"
    payload = b"must stay inside the extraction root"
    member = tarfile.TarInfo("scenes/escape.txt")
    member.size = len(payload)
    with tarfile.open(archive, "w") as tar:
        tar.addfile(member, io.BytesIO(payload))

    with pytest.raises(FZIAURADownloadError, match="through symlink"):
        _extract_archive_safely(archive, output)

    assert not (outside / "escape.txt").exists()


def test_extract_can_keep_archives(tmp_path: Path) -> None:
    downloader, selection, archive, _, _ = _prepare_extractable_base_chunk(tmp_path)

    result = downloader.extract(selection, jobs=1, verify=True, delete_archives=False)

    assert archive.is_file()
    assert result["archives_deleted"] is False


def test_successful_archive_is_deleted_when_another_extraction_fails(
    tmp_path: Path,
) -> None:
    downloader, selection, archive, _, _ = _prepare_extractable_base_chunk(tmp_path)
    missing_archive = (
        "chunks/v1.0/train/base_keyframes/block-000000/base_keyframes-0.tar.xz"
    )
    selection = replace(
        selection, chunk_paths=(selection.chunk_paths[0], missing_archive)
    )

    with pytest.raises(FZIAURADownloadError, match="selected archive is missing"):
        downloader.extract(selection, jobs=1)

    assert not archive.exists()


def test_archive_is_deleted_as_soon_as_its_worker_finishes(
    tmp_path: Path, monkeypatch
) -> None:
    downloader, selection, first_archive, _, _ = _prepare_extractable_base_chunk(
        tmp_path
    )
    second_archive = first_archive.with_name("base-1.tar.xz")
    second_archive.write_bytes(b"second archive")
    second_chunk_path = str(second_archive.relative_to(tmp_path))

    chunks_path = downloader.metadata_dir / "chunks.parquet"
    chunks = pd.read_parquet(chunks_path)
    first_row = chunks[chunks["chunk_path"] == selection.chunk_paths[0]].iloc[0]
    second_row = first_row.copy()
    second_row["chunk_path"] = second_chunk_path
    second_row["size_bytes"] = second_archive.stat().st_size
    pd.concat([chunks, second_row.to_frame().T], ignore_index=True).to_parquet(
        chunks_path, index=False
    )
    selection = replace(
        selection,
        chunk_paths=(selection.chunk_paths[0], second_chunk_path),
    )

    def fake_extract(archive: Path, output_dir: Path) -> None:
        del output_dir
        if archive == first_archive:
            return
        deadline = time.monotonic() + 2.0
        while first_archive.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not first_archive.exists()

    monkeypatch.setattr("fzi_aura.download._extract_archive_safely", fake_extract)

    downloader.extract(selection, jobs=2)

    assert not first_archive.exists()
    assert not second_archive.exists()


def test_parser_extracts_by_default_and_accepts_keep_archives() -> None:
    parser = build_parser()

    assert not parser.parse_args(["/tmp/fzi-aura"]).keep_archives
    assert parser.parse_args(["/tmp/fzi-aura", "--keep-archives"]).keep_archives
    with pytest.raises(SystemExit):
        parser.parse_args(["/tmp/fzi-aura", "--no-base"])
    with pytest.raises(SystemExit):
        parser.parse_args(["/tmp/fzi-aura", "--extract"])
    with pytest.raises(SystemExit):
        parser.parse_args(["/tmp/fzi-aura", "--delete-archives"])


def test_main_extracts_by_default(tmp_path: Path, monkeypatch) -> None:
    selection = DownloadSelection(
        layers=("base_keyframes",),
        splits=("train",),
        requested_scene_ids=("scene",),
        block_scene_ids=("scene",),
        block_keys=(("train", 0),),
        chunk_paths=("chunk.tar.xz",),
        size_bytes=1,
    )
    calls = []

    class FakeDownloader:
        metadata_dir = tmp_path / "metadata"

        def __init__(self, *args, **kwargs) -> None:
            calls.append(("init", args, kwargs))

        def fetch_metadata(self, **kwargs) -> None:
            calls.append(("fetch_metadata", kwargs))

        def select(self, **kwargs) -> DownloadSelection:
            calls.append(("select", kwargs))
            return selection

        def download(self, selected, **kwargs) -> None:
            assert selected is selection
            calls.append(("download", kwargs))

        def extract(self, selected, **kwargs) -> dict:
            assert selected is selection
            calls.append(("extract", kwargs))
            return {"archives_deleted": True}

    monkeypatch.setattr("fzi_aura.download.FZIAURADownloader", FakeDownloader)

    main(
        [
            str(tmp_path),
            "--jobs",
            "7",
            "--verify",
            "--layers",
            "camera_keyframes",
        ]
    )

    assert (
        "select",
        {
            "layers": ["base_keyframes", "camera_keyframes"],
            "splits": ["train", "val", "test"],
            "scene_ids": None,
            "available_chunk_paths": None,
        },
    ) in calls
    assert ("extract", {"jobs": 7, "verify": True, "delete_archives": True}) in calls


def test_mount_decompresses_tar_xz_without_extracting_members(
    tmp_path: Path, monkeypatch
) -> None:
    downloader, selection, archive, payload, _ = _prepare_extractable_base_chunk(
        tmp_path
    )
    commands = []
    monkeypatch.setattr(
        "fzi_aura.download.shutil.which", lambda _: "/usr/bin/ratarmount"
    )
    monkeypatch.setattr(
        "fzi_aura.download.subprocess.run",
        lambda command, check: commands.append((command, check)),
    )

    result = downloader.mount(selection, tmp_path / "mounted", jobs=1, verify=True)

    mounted_archive = archive.with_suffix("")
    assert not archive.exists()
    assert mounted_archive.is_file()
    with tarfile.open(mounted_archive) as tar:
        member = tar.extractfile("scenes/2025-06-11-08-18-57_1/scene.json")
        assert member is not None
        assert member.read() == payload
    assert not (tmp_path / "scenes").exists()
    assert result["decompressed_chunks"] == 1
    assert commands == [
        (
            [
                "/usr/bin/ratarmount",
                str(mounted_archive),
                result["metadata_overlay"],
                str((tmp_path / "mounted").resolve()),
            ],
            True,
        )
    ]
    assert selection.chunk_paths[0] in downloader.local_chunk_paths(
        allow_decompressed_archives=True
    )
    assert selection.chunk_paths[0] not in downloader.local_chunk_paths()
