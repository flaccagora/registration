"""Research prototype for 2D-to-3D surgical surface registration.

This package is intentionally modular: model-specific code lives behind thin
wrappers, while correspondence parsing, pose estimation, deformation, and
export are kept independent and testable.
"""

from pipeline.config import PipelineConfig

__all__ = ["PipelineConfig"]

