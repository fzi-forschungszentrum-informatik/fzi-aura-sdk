"""Verify that notebooks are included in the sdist but excluded from wheels."""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _is_notebook_path(name: str) -> bool:
    path = PurePosixPath(name)
    return "notebooks" in path.parts or path.suffix == ".ipynb"


def _repository_notebook_path(name: str) -> str:
    path = PurePosixPath(name)
    try:
        notebooks_index = path.parts.index("notebooks")
    except ValueError:
        return path.as_posix()
    return PurePosixPath(*path.parts[notebooks_index:]).as_posix()


def _check_archive(path: Path) -> list[str]:
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return [member.name for member in archive.getmembers() if member.isfile()]
    if path.suffix == ".whl":
        with ZipFile(path) as archive:
            return [name for name in archive.namelist() if not name.endswith("/")]
    raise ValueError(f"Unsupported distribution: {path}")


def main() -> int:
    dist_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    if not dist_dir.is_dir():
        raise SystemExit(f"Distribution directory not found: {dist_dir}")

    distributions = sorted(
        path
        for path in dist_dir.iterdir()
        if path.name.endswith(".tar.gz") or path.suffix == ".whl"
    )
    if not distributions:
        raise SystemExit(f"No distributions found in {dist_dir}")

    expected_notebooks = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "notebooks").glob("*.ipynb")
    }
    if not expected_notebooks:
        raise SystemExit("No repository notebooks found")

    wheels = [path for path in distributions if path.suffix == ".whl"]
    source_distributions = [
        path for path in distributions if path.name.endswith(".tar.gz")
    ]
    if not wheels or not source_distributions:
        raise SystemExit("Expected at least one wheel and one source distribution")

    failures = []
    for distribution in distributions:
        notebook_paths = [
            name for name in _check_archive(distribution) if _is_notebook_path(name)
        ]
        if distribution.suffix == ".whl":
            if notebook_paths:
                failures.append(
                    f"{distribution}: wheel contains notebook files:\n"
                    + "\n".join(f"  {name}" for name in notebook_paths)
                )
            else:
                print(f"OK: {distribution} contains no notebooks")
            continue

        actual_notebooks = {_repository_notebook_path(name) for name in notebook_paths}
        missing = sorted(expected_notebooks - actual_notebooks)
        unexpected = sorted(actual_notebooks - expected_notebooks)
        if missing or unexpected:
            details = []
            if missing:
                details.append(
                    "missing notebooks:\n" + "\n".join(f"  {name}" for name in missing)
                )
            if unexpected:
                details.append(
                    "unexpected notebooks:\n"
                    + "\n".join(f"  {name}" for name in unexpected)
                )
            failures.append(f"{distribution}:\n" + "\n".join(details))
        else:
            print(
                f"OK: {distribution} contains all {len(expected_notebooks)} notebooks"
            )

    if failures:
        raise SystemExit("\n".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
