"""One child slice of an execution schedule (port of Java
``execution.Slice``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Slice:
    """Attributes:
        offset_millis: when this child fires, relative to schedule start.
        quantity: child quantity.
    """

    offset_millis: int
    quantity: int
