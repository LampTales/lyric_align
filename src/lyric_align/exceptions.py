"""Stable exception types exposed by the package."""


class LyricAlignError(Exception):
    """Base class for expected user/actionable failures."""


class InputValidationError(LyricAlignError):
    """The song directory does not satisfy the input contract."""


class StageUnavailableError(LyricAlignError):
    """An optional heavy backend or its model is not available."""
