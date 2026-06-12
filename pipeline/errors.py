"""Shared exceptions for pipeline modules."""


class PipelineError(RuntimeError):
    """Base class for user-facing pipeline failures."""


class MissingDependencyError(PipelineError):
    """Raised when an optional runtime dependency is not installed."""


class MissingCheckpointError(PipelineError):
    """Raised when a configured model checkpoint path does not exist."""


class InvalidInputError(PipelineError):
    """Raised when input data is missing required fields or is malformed."""

