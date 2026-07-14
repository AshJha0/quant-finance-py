"""Cross-session persistence (port of Java ``com.quantfinlib.persist``).

One class: :class:`~quantfinlib.persist.checkpoint.Checkpoint`, a named-
section binary file format for a model's learned (cross-day) state.
See its module docstring for the wire format and durability contract.
"""

from quantfinlib.persist.checkpoint import (
    BinReader,
    BinWriter,
    Checkpoint,
    CheckpointReader,
    CheckpointWriter,
)

__all__ = [
    "BinReader",
    "BinWriter",
    "Checkpoint",
    "CheckpointReader",
    "CheckpointWriter",
]
