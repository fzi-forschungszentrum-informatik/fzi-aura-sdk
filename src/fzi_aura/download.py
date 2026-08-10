"""Selective Hugging Face download, extraction, and archive mounting for FZI-AURA."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Optional

import pandas as pd

from .dataset import _normalize_scene_lookup_id
from .errors import FZIAURADownloadError

DEFAULT_REPO_ID = "fzi-forschungszentrum-informatik/FZI-AURA"
DEFAULT_VERSION = "v1.0"
REPOSITORY_DOCUMENTS = ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.MD")
ALL_SPLITS = ("train", "val", "test")
ALL_LAYERS = (
    "base_keyframes",
    "camera_keyframes",
    "lidar_motion_compensated_keyframes",
    "lidar_raw_keyframes",
    "radar_keyframes",
    "camera_nonkeyframes",
    # "lidar_motion_compensated_nonkeyframes", not uploaded for space reasons
    "lidar_raw_nonkeyframes",
    "radar_nonkeyframes",
)
DEFAULT_LAYERS = (
    "base_keyframes",
    "camera_keyframes",
    "lidar_motion_compensated_keyframes",
    "lidar_raw_keyframes",
    "radar_keyframes",
)


@dataclass(frozen=True)
class DownloadSelection:
    """Resolved synchronized release blocks and archive paths."""

    layers: tuple[str, ...]
    splits: tuple[str, ...]
    requested_scene_ids: tuple[str, ...]
    block_scene_ids: tuple[str, ...]
    block_keys: tuple[tuple[str, int], ...]
    chunk_paths: tuple[str, ...]
    size_bytes: int

    @property
    def boundary_neighbor_count(self) -> int:
        return len(self.block_scene_ids) - len(self.requested_scene_ids)

    def summary(self) -> dict[str, Any]:
        return {
            "layers": list(self.layers),
            "splits": list(self.splits),
            "requested_scenes": len(self.requested_scene_ids),
            "block_scenes": len(self.block_scene_ids),
            "boundary_neighbor_scenes": self.boundary_neighbor_count,
            "block_groups": len(self.block_keys),
            "chunks": len(self.chunk_paths),
            "size_bytes": self.size_bytes,
            "size_gb": self.size_bytes / 1000**3,
        }


def _read_scene_ids(path: Path) -> list[str]:
    try:
        rows = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise FZIAURADownloadError(f"cannot read scene IDs file {path}: {exc}") from exc
    if not rows:
        raise FZIAURADownloadError(f"scene IDs file is empty: {path}")
    return rows


def _safe_member_path(name: str) -> Path:
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise FZIAURADownloadError(f"unsafe archive member path: {name!r}")
    return Path(*pure.parts)


def _safe_extraction_destination(output_dir: Path, relative: Path) -> Path:
    """Resolve an archive member below ``output_dir`` without following symlinks."""

    try:
        root = output_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FZIAURADownloadError(
            f"cannot resolve archive extraction root {output_dir}: {exc}"
        ) from exc
    if not root.is_dir():
        raise FZIAURADownloadError(
            f"archive extraction root is not a directory: {output_dir}"
        )

    destination = root / relative
    current = root
    for part in relative.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise FZIAURADownloadError(
                f"cannot inspect archive extraction path {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise FZIAURADownloadError(
                f"refusing archive extraction through symlink: {current}"
            )
        if current != destination and not stat.S_ISDIR(mode):
            raise FZIAURADownloadError(
                f"archive extraction parent is not a directory: {current}"
            )

    try:
        resolved_destination = destination.resolve(strict=False)
        resolved_destination.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise FZIAURADownloadError(
            f"archive member escapes extraction root: {relative}"
        ) from exc
    return destination


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream).hex()


def _sha256_stream(stream: Any) -> bytes:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.digest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class FZIAURADownloader:
    """Select and download FZI-AURA release layers from synchronized scene blocks."""

    def __init__(
        self,
        output_dir: str | Path,
        repo_id: str = DEFAULT_REPO_ID,
        version: str = DEFAULT_VERSION,
        revision: Optional[str] = None,
        token: str | bool | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.repo_id = repo_id
        self.version = version
        self.revision = revision
        self.token = token
        self._remote_files: Optional[frozenset[str]] = None

    @property
    def metadata_dir(self) -> Path:
        return self.output_dir / "metadata" / self.version

    def fetch_metadata(self, *, max_workers: int = 8) -> Path:
        """Download release metadata needed to resolve layers, scenes, and blocks."""

        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise FZIAURADownloadError(
                'install download support with: pip install "fzi-aura[download]"'
            ) from exc
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result = Path(
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                local_dir=self.output_dir,
                allow_patterns=[
                    f"metadata/{self.version}/**",
                    *REPOSITORY_DOCUMENTS,
                ],
                token=self.token,
                max_workers=max_workers,
            )
        )
        api = HfApi(token=self.token)
        self._remote_files = frozenset(
            api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
            )
        )
        return result

    def _release_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        chunks_path = self.metadata_dir / "chunks.parquet"
        scene_blocks_path = self.metadata_dir / "scene_blocks.parquet"
        if not chunks_path.is_file() or not scene_blocks_path.is_file():
            raise FZIAURADownloadError(
                f"release metadata is incomplete under {self.metadata_dir}; fetch metadata first"
            )
        chunks = pd.read_parquet(chunks_path)
        scene_blocks = pd.read_parquet(scene_blocks_path)
        return chunks, scene_blocks

    def select(
        self,
        *,
        layers: Iterable[str] = DEFAULT_LAYERS,
        splits: Iterable[str] = ALL_SPLITS,
        scene_ids: Optional[Iterable[str]] = None,
        available_chunk_paths: Optional[Iterable[str]] = None,
    ) -> DownloadSelection:
        """Resolve complete synchronized blocks from archives that are available."""

        chunks, scene_blocks = self._release_tables()
        selected_layers = tuple(dict.fromkeys(str(layer) for layer in layers))
        selected_splits = tuple(dict.fromkeys(str(split) for split in splits))
        invalid_layers = sorted(set(selected_layers).difference(ALL_LAYERS))
        if invalid_layers:
            raise FZIAURADownloadError(f"unknown layers: {', '.join(invalid_layers)}")
        invalid_splits = sorted(set(selected_splits).difference(ALL_SPLITS))
        if invalid_splits:
            raise FZIAURADownloadError(f"unknown splits: {', '.join(invalid_splits)}")
        if not selected_layers:
            raise FZIAURADownloadError("select at least one layer")
        if not selected_splits:
            raise FZIAURADownloadError("select at least one split")

        if available_chunk_paths is None:
            if self._remote_files is None:
                raise FZIAURADownloadError(
                    "archive availability is unknown; fetch metadata or pass "
                    "available_chunk_paths explicitly"
                )
            available_paths = self._remote_files
        else:
            available_paths = frozenset(str(path) for path in available_chunk_paths)

        split_rows = scene_blocks[scene_blocks["split"].isin(selected_splits)].copy()
        explicit_scene_selection = scene_ids is not None
        if scene_ids is not None:
            normalized = [
                _normalize_scene_lookup_id(str(scene_id)) for scene_id in scene_ids
            ]
            duplicates = sorted(
                scene_id
                for scene_id in set(normalized)
                if normalized.count(scene_id) > 1
            )
            if duplicates:
                raise FZIAURADownloadError(
                    "duplicate scene IDs after normalization: "
                    + ", ".join(duplicates[:20])
                )
            rows_by_id = scene_blocks.set_index("scene_id", drop=False)
            missing = sorted(set(normalized).difference(rows_by_id.index))
            if missing:
                raise FZIAURADownloadError(
                    f"{len(missing)} requested scenes are absent from release metadata: "
                    + ", ".join(missing[:20])
                )
            requested_rows = rows_by_id.loc[normalized]
            outside_splits = requested_rows[
                ~requested_rows["split"].isin(selected_splits)
            ]
            if not outside_splits.empty:
                raise FZIAURADownloadError(
                    "requested scenes are outside the selected splits: "
                    + ", ".join(outside_splits["scene_id"].tolist()[:20])
                )
            candidate_block_keys = {
                (str(row["split"]), int(row["scene_block"]))
                for row in requested_rows.to_dict("records")
            }
        else:
            candidate_block_keys = {
                (str(row["split"]), int(row["scene_block"]))
                for row in split_rows.to_dict("records")
            }

        selected_manifest_rows = chunks[
            chunks["split"].isin(selected_splits)
            & chunks["layer"].isin(selected_layers)
        ].copy()
        rows_by_block: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in selected_manifest_rows.to_dict("records"):
            key = (str(row["split"]), int(row["scene_block"]))
            if key in candidate_block_keys:
                rows_by_block.setdefault(key, []).append(row)

        block_keys: set[tuple[str, int]] = set()
        incomplete: dict[tuple[str, int], list[str]] = {}
        for key in candidate_block_keys:
            block_chunks = rows_by_block.get(key, [])
            missing_paths = [
                str(row["chunk_path"])
                for row in block_chunks
                if str(row["status"]) != "ready"
                or str(row["chunk_path"]) not in available_paths
            ]
            if not block_chunks:
                missing_paths.append("<no matching chunks in release metadata>")
            if "base_keyframes" in selected_layers and not any(
                str(row["layer"]) == "base_keyframes" for row in block_chunks
            ):
                missing_paths.append(
                    "<base_keyframes chunk missing from release metadata>"
                )
            if missing_paths:
                incomplete[key] = missing_paths
            else:
                block_keys.add(key)

        if explicit_scene_selection and incomplete:
            details = [
                f"{key}: {path}"
                for key in sorted(incomplete)
                for path in incomplete[key]
            ]
            raise FZIAURADownloadError(
                "requested scenes need blocks that are not completely available for the "
                "selected layers:\n" + "\n".join(details[:50])
            )
        if not block_keys:
            raise FZIAURADownloadError(
                "no complete available blocks match the requested layers and splits"
            )

        block_mask = [
            (str(row["split"]), int(row["scene_block"])) in block_keys
            for row in scene_blocks.to_dict("records")
        ]
        block_rows = scene_blocks.loc[block_mask].sort_values(
            ["split", "scene_ordinal"]
        )
        if not explicit_scene_selection:
            requested_rows = block_rows
        chunk_mask = [
            (str(row["split"]), int(row["scene_block"])) in block_keys
            and str(row["layer"]) in selected_layers
            for row in chunks.to_dict("records")
        ]
        selected_chunks = chunks.loc[chunk_mask].copy()
        if selected_chunks.empty:
            raise FZIAURADownloadError(
                "no available chunks match the requested layers and splits"
            )

        return DownloadSelection(
            layers=selected_layers,
            splits=selected_splits,
            requested_scene_ids=tuple(
                str(value) for value in requested_rows["scene_id"].tolist()
            ),
            block_scene_ids=tuple(
                str(value) for value in block_rows["scene_id"].tolist()
            ),
            block_keys=tuple(sorted(block_keys)),
            chunk_paths=tuple(
                sorted(str(value) for value in selected_chunks["chunk_path"].tolist())
            ),
            size_bytes=int(selected_chunks["size_bytes"].sum()),
        )

    def local_chunk_paths(
        self, *, allow_decompressed_archives: bool = False
    ) -> frozenset[str]:
        """Return manifest chunk paths whose usable archives exist locally.

        Mounting converts ``.tar.xz`` chunks to ``.tar`` files. These are usable
        only for mounting because the release manifest identifies the compressed
        source archive.
        """

        chunks, _ = self._release_tables()
        return frozenset(
            str(path)
            for path in chunks["chunk_path"].tolist()
            if (self.output_dir / str(path)).is_file()
            or (
                allow_decompressed_archives
                and str(path).endswith(".tar.xz")
                and _uncompressed_tar_path(self.output_dir / str(path)).is_file()
            )
        )

    def download(
        self,
        selection: DownloadSelection,
        *,
        max_workers: int = 8,
        dry_run: bool = False,
        allow_decompressed_archives: bool = False,
    ) -> dict[str, Any]:
        """Download only the exact archives selected from the release manifests."""

        patterns = [f"metadata/{self.version}/**", *REPOSITORY_DOCUMENTS]
        for chunk_path in selection.chunk_paths:
            archive = self.output_dir / chunk_path
            if archive.is_file() or (
                allow_decompressed_archives
                and chunk_path.endswith(".tar.xz")
                and _uncompressed_tar_path(archive).is_file()
            ):
                continue
            patterns.extend((chunk_path, f"{chunk_path}.chunk.json"))
        result = selection.summary()
        result.update(
            {
                "repo": self.repo_id,
                "revision": self.revision,
                "output": str(self.output_dir),
                "dry_run": dry_run,
            }
        )
        if dry_run:
            return result
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise FZIAURADownloadError(
                'install download support with: pip install "fzi-aura[download]"'
            ) from exc
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.output_dir,
            allow_patterns=patterns,
            token=self.token,
            max_workers=max_workers,
        )
        missing = [
            path
            for path in selection.chunk_paths
            if not (self.output_dir / path).is_file()
            and not (
                allow_decompressed_archives
                and path.endswith(".tar.xz")
                and _uncompressed_tar_path(self.output_dir / path).is_file()
            )
        ]
        if missing:
            raise FZIAURADownloadError(
                f"download completed without {len(missing)} selected chunks: "
                + ", ".join(missing[:10])
            )
        return result

    def _selected_chunk_rows(self, selection: DownloadSelection) -> pd.DataFrame:
        chunks, _ = self._release_tables()
        return chunks[chunks["chunk_path"].isin(selection.chunk_paths)].copy()

    def materialize_metadata(
        self, selection: DownloadSelection, root: Optional[Path] = None
    ) -> Path:
        """Write a filtered SDK dataset root for only the requested public scenes."""

        target = (root or self.output_dir).resolve()
        dataset_template_path = self.metadata_dir / "dataset.json"
        if not dataset_template_path.is_file():
            raise FZIAURADownloadError(
                f"missing dataset metadata template: {dataset_template_path}"
            )
        _, scene_blocks = self._release_tables()
        rows_by_id = scene_blocks.set_index("scene_id", drop=False)
        requested_rows = rows_by_id.loc[
            list(selection.requested_scene_ids)
        ].sort_values(["split", "scene_ordinal"])
        scene_entries: list[dict[str, Any]] = []
        for row in requested_rows.to_dict("records"):
            try:
                entry = json.loads(str(row["dataset_scene_json"]))
            except (TypeError, json.JSONDecodeError) as exc:
                raise FZIAURADownloadError(
                    f"invalid dataset_scene_json for {row['scene_id']}: {exc}"
                ) from exc
            scene_entries.append(entry)
        template = json.loads(dataset_template_path.read_text(encoding="utf-8"))
        template["scenes"] = scene_entries
        excluded_ids = {
            str(entry.get("scene_id"))
            for entry in template.get("consumer_excluded_scenes", [])
        }
        template["consumer_excluded_scenes"] = [
            entry
            for entry in template.get("consumer_excluded_scenes", [])
            if str(entry.get("scene_id")) in set(selection.requested_scene_ids)
        ]
        _atomic_json(target / "dataset.json", template)
        for split in ALL_SPLITS:
            split_ids = [
                str(row["scene_id"])
                for row in requested_rows.to_dict("records")
                if row["split"] == split and str(row["scene_id"]) not in excluded_ids
            ]
            _atomic_text(
                target / "splits" / self.version / f"{split}.txt",
                "".join(f"{scene_id}\n" for scene_id in split_ids),
            )
        _atomic_json(
            target / "available_data.json",
            {
                **selection.summary(),
                "repo": self.repo_id,
                "revision": self.revision,
                "version": self.version,
            },
        )
        return target

    def extract(
        self,
        selection: DownloadSelection,
        *,
        jobs: int = 4,
        verify: bool = False,
        delete_archives: bool = True,
    ) -> dict[str, Any]:
        """Safely extract selected archives and optionally remove them on success.

        Each archive is removed by default immediately after that archive has
        extracted successfully. Keeping archives is an explicit opt-in for reuse
        or mounting uncompressed layers.
        """

        if "base_keyframes" not in selection.layers:
            raise FZIAURADownloadError(
                "extraction requires base_keyframes so the SDK root is usable"
            )
        chunk_rows = self._selected_chunk_rows(selection).set_index(
            "chunk_path", drop=False
        )
        failures: list[str] = []
        completed: list[str] = []
        lock = threading.Lock()

        def extract_one(chunk_path: str) -> None:
            archive = self.output_dir / chunk_path
            if not archive.is_file():
                raise FZIAURADownloadError(f"selected archive is missing: {archive}")
            row = chunk_rows.loc[chunk_path]
            if verify and _sha256_file(archive) != str(row["sha256"]):
                raise FZIAURADownloadError(f"archive SHA-256 mismatch: {archive}")
            _extract_archive_safely(archive, self.output_dir)
            if delete_archives:
                try:
                    archive.unlink()
                except OSError as exc:
                    raise FZIAURADownloadError(
                        "archive extraction succeeded, but archive cleanup failed: "
                        f"{archive}: {exc}"
                    ) from exc
            with lock:
                completed.append(chunk_path)

        with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
            futures = {
                executor.submit(extract_one, path): path
                for path in selection.chunk_paths
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(f"{futures[future]}: {exc}")
        if failures:
            raise FZIAURADownloadError(
                "archive extraction failed:\n" + "\n".join(failures[:50])
            )
        self.materialize_metadata(selection)
        return {
            **selection.summary(),
            "extracted_chunks": len(completed),
            "archives_deleted": delete_archives,
            "root": str(self.output_dir),
        }

    def mount(
        self,
        selection: DownloadSelection,
        mountpoint: str | Path,
        *,
        jobs: int = 4,
        verify: bool = False,
    ) -> dict[str, Any]:
        """Convert selected XZ chunks to TAR and union-mount with ratarmount.

        Archive contents stay unextracted. A compressed source is removed only
        after its replacement TAR has been written atomically.
        """

        if "base_keyframes" not in selection.layers:
            raise FZIAURADownloadError(
                "mounting requires base_keyframes so the SDK root is usable"
            )
        ratarmount = shutil.which("ratarmount")
        if ratarmount is None:
            raise FZIAURADownloadError(
                'install mount support with: pip install "fzi-aura[mount]"'
            )
        archives, decompressed_chunks = self._prepare_mount_archives(
            selection, jobs=jobs, verify=verify
        )
        mount_path = Path(mountpoint).resolve()
        mount_path.mkdir(parents=True, exist_ok=True)
        metadata_overlay = (
            self.output_dir / ".fzi_aura_mount_metadata" / _selection_digest(selection)
        )
        self.materialize_metadata(selection, root=metadata_overlay)
        command = [
            ratarmount,
            *(str(path) for path in archives),
            str(metadata_overlay),
            str(mount_path),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise FZIAURADownloadError(
                f"ratarmount failed with exit status {exc.returncode}"
            ) from exc
        return {
            **selection.summary(),
            "mountpoint": str(mount_path),
            "metadata_overlay": str(metadata_overlay),
            "archives": len(archives),
            "decompressed_chunks": decompressed_chunks,
        }

    def _prepare_mount_archives(
        self,
        selection: DownloadSelection,
        *,
        jobs: int,
        verify: bool,
    ) -> tuple[list[Path], int]:
        """Return mountable TAR paths, converting XZ inputs when needed."""

        chunk_rows = self._selected_chunk_rows(selection).set_index(
            "chunk_path", drop=False
        )
        archives: dict[str, Path] = {}
        failures: list[str] = []
        decompressed_chunks = 0
        lock = threading.Lock()

        def prepare_one(chunk_path: str) -> None:
            nonlocal decompressed_chunks
            archive = self.output_dir / chunk_path
            if archive.suffix != ".xz":
                if not archive.is_file():
                    raise FZIAURADownloadError(
                        f"selected archive is missing: {archive}"
                    )
                if verify and _sha256_file(archive) != str(
                    chunk_rows.loc[chunk_path]["sha256"]
                ):
                    raise FZIAURADownloadError(f"archive SHA-256 mismatch: {archive}")
                mounted_archive = archive
            else:
                mounted_archive = _uncompressed_tar_path(archive)
                if archive.is_file():
                    if verify and _sha256_file(archive) != str(
                        chunk_rows.loc[chunk_path]["sha256"]
                    ):
                        raise FZIAURADownloadError(
                            f"archive SHA-256 mismatch: {archive}"
                        )
                    _decompress_xz_to_tar(archive, mounted_archive)
                    try:
                        archive.unlink()
                    except OSError as exc:
                        raise FZIAURADownloadError(
                            "archive decompression succeeded, but compressed archive "
                            f"cleanup failed: {archive}: {exc}"
                        ) from exc
                    with lock:
                        decompressed_chunks += 1
                elif not mounted_archive.is_file():
                    raise FZIAURADownloadError(
                        f"selected archive is missing: {archive}"
                    )
            with lock:
                archives[chunk_path] = mounted_archive

        with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
            futures = {
                executor.submit(prepare_one, path): path
                for path in selection.chunk_paths
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(f"{futures[future]}: {exc}")
        if failures:
            raise FZIAURADownloadError(
                "archive preparation for mounting failed:\n" + "\n".join(failures[:50])
            )
        return [archives[path] for path in selection.chunk_paths], decompressed_chunks


def _selection_digest(selection: DownloadSelection) -> str:
    digest = hashlib.sha256()
    for value in (
        *selection.layers,
        *selection.requested_scene_ids,
        *selection.chunk_paths,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _uncompressed_tar_path(archive: Path) -> Path:
    if archive.suffix != ".xz" or archive.with_suffix("").suffix != ".tar":
        raise ValueError(f"expected a .tar.xz archive, got: {archive}")
    return archive.with_suffix("")


def _decompress_xz_to_tar(source: Path, destination: Path) -> None:
    """Stream-decompress an XZ archive atomically without extracting members."""

    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        with lzma.open(source, "rb") as compressed, temporary.open("xb") as output:
            shutil.copyfileobj(compressed, output, length=8 * 1024 * 1024)
        os.replace(temporary, destination)
    except (lzma.LZMAError, OSError) as exc:
        raise FZIAURADownloadError(
            f"failed to decompress {source} into {destination}: {exc}"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _extract_archive_safely(archive: Path, output_dir: Path) -> None:
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            relative = _safe_member_path(member.name)
            if not member.isfile():
                raise FZIAURADownloadError(
                    f"unsupported non-file member in {archive}: {member.name}"
                )
            destination = _safe_extraction_destination(output_dir, relative)
            source = tar.extractfile(member)
            if source is None:
                raise FZIAURADownloadError(
                    f"cannot read member in {archive}: {member.name}"
                )
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise FZIAURADownloadError(
                        f"refusing to overwrite non-regular path: {destination}"
                    )
                if destination.stat().st_size != member.size:
                    raise FZIAURADownloadError(
                        f"existing extracted file has a different size: {destination}"
                    )
                with source:
                    archived_hash = _sha256_stream(source)
                with destination.open("rb") as existing:
                    existing_hash = _sha256_stream(existing)
                if existing_hash != archived_hash:
                    raise FZIAURADownloadError(
                        f"existing extracted file differs: {destination}"
                    )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination = _safe_extraction_destination(output_dir, relative)
            temporary = destination.with_name(
                f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}"
            )
            try:
                with source, temporary.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _scene_ids_from_args(args: argparse.Namespace) -> Optional[list[str]]:
    sources = sum(value is not None for value in (args.scene_ids_file, args.scenes))
    if sources > 1:
        raise FZIAURADownloadError("use only one of --scene-ids-file or --scenes")
    if args.scene_ids_file is not None:
        return _read_scene_ids(args.scene_ids_file)
    if args.scenes is not None:
        return _parse_csv(args.scenes)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Selectively download, extract, or mount FZI-AURA Hugging Face release layers."
    )
    parser.add_argument(
        "output", type=Path, help="Download directory and extracted SDK dataset root."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO_ID)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--layers",
        default=",".join(DEFAULT_LAYERS),
        help="Comma-separated exact release layers. base_keyframes is always included.",
    )
    parser.add_argument(
        "--splits",
        default=",".join(ALL_SPLITS),
        help="Comma-separated train,val,test subset.",
    )
    parser.add_argument(
        "--scene-ids-file",
        type=Path,
        default=None,
        help="Text file containing scene IDs or path-safe scene names.",
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help="Comma-separated scene IDs or path-safe scene names.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Hugging Face download workers."
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Parallel archive extraction workers."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify archive SHA-256 before extraction.",
    )
    archive_retention = parser.add_mutually_exclusive_group()
    archive_retention.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep archives after extraction for reuse or mounting uncompressed layers.",
    )
    parser.add_argument(
        "--mount",
        type=Path,
        default=None,
        help="Union-mount downloaded uncompressed .tar archives here.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use already downloaded metadata and archives.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.keep_archives and args.mount is not None:
        parser.error("--keep-archives cannot be used with --mount")
    try:
        layers = _parse_csv(args.layers)
        if "base_keyframes" not in layers:
            layers.insert(0, "base_keyframes")
        downloader = FZIAURADownloader(
            args.output,
            repo_id=args.repo,
            version=args.version,
            revision=args.revision,
            token=args.token,
        )
        if args.no_download:
            if not downloader.metadata_dir.is_dir():
                raise FZIAURADownloadError(
                    "--no-download requires existing release metadata"
                )
        else:
            downloader.fetch_metadata(max_workers=args.max_workers)
        available_chunk_paths = (
            downloader.local_chunk_paths(
                allow_decompressed_archives=args.mount is not None
            )
            if args.no_download
            else None
        )
        selection = downloader.select(
            layers=layers,
            splits=_parse_csv(args.splits),
            scene_ids=_scene_ids_from_args(args),
            available_chunk_paths=available_chunk_paths,
        )
        print(json.dumps(selection.summary(), indent=2, sort_keys=True))
        if args.dry_run:
            return
        if not args.no_download:
            downloader.download(
                selection,
                max_workers=args.max_workers,
                allow_decompressed_archives=args.mount is not None,
            )
        if args.mount is not None:
            result = downloader.mount(
                selection, args.mount, jobs=args.jobs, verify=args.verify
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            result = downloader.extract(
                selection,
                jobs=args.jobs,
                verify=args.verify,
                delete_archives=not args.keep_archives,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
    except FZIAURADownloadError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
