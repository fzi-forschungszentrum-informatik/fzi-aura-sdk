__all__ = [
    "AnnotationNotFoundError",
    "CalibrationError",
    "FieldNotFoundError",
    "FZIAURADownloadError",
    "FZIAURAError",
    "SceneNotFoundError",
    "SensorNotFoundError",
    "SplitError",
    "UnsupportedPCDError",
]


class FZIAURAError(Exception):
    """Base exception for FZI-AURA SDK errors."""


class FZIAURADownloadError(FZIAURAError):
    """Raised for release selection, download, extraction, or mount failures."""


class SplitError(FZIAURAError):
    pass


class SceneNotFoundError(FZIAURAError):
    pass


class SensorNotFoundError(FZIAURAError):
    pass


class AnnotationNotFoundError(FZIAURAError):
    pass


class UnsupportedPCDError(FZIAURAError):
    pass


class FieldNotFoundError(FZIAURAError):
    pass


class CalibrationError(FZIAURAError):
    pass
