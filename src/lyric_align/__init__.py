"""Public API for the model-assisted lyric alignment library.

The package deliberately keeps model locations independent.  A caller may
provide separate paths for the Demucs repository, CTC checkpoint and G2P
dictionary; no model is downloaded or bundled by this package.
"""

from .config import AlignmentConfig, ModelPaths
from .pipeline import prepare_song, validate_song
from .schema import AlignmentArtifact, AlignmentLine, ArtifactPaths

__all__ = [
    "AlignmentConfig",
    "ModelPaths",
    "AlignmentArtifact",
    "AlignmentLine",
    "ArtifactPaths",
    "prepare_song",
    "validate_song",
]

__version__ = "0.1.0"
