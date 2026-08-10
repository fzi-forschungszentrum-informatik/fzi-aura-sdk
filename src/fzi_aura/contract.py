from __future__ import annotations

from typing import Final

from .errors import FZIAURAError

PUBLIC_FORMAT_VERSION: Final = "v1.2.1"
LIDAR_STAGES: Final = frozenset({"raw", "motion_compensated"})


def validate_lidar_stage(stage: str) -> str:
    if stage not in LIDAR_STAGES:
        raise ValueError(
            f"Unsupported lidar stage {stage!r}; expected 'raw' or 'motion_compensated'."
        )
    return stage


def require_format_version(document: dict, source: str) -> None:
    version = document.get("format_version")
    if version != PUBLIC_FORMAT_VERSION:
        raise FZIAURAError(
            f"{source} has unsupported format_version {version!r}; "
            f"expected {PUBLIC_FORMAT_VERSION!r}."
        )
