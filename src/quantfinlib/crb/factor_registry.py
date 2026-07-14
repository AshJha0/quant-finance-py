"""Dense integer ids for risk-factor names (port of Java
``com.quantfinlib.crb.FactorRegistry``).

The ``SymbolRegistry`` pattern applied to the central risk book's
factor space, so exposure arithmetic runs over primitive arrays while
the factor names stay readable (``EQ:AAPL``, ``CCY:EUR``,
``FXVEGA:EURUSD``). Grow-only; ids are registration order.
"""

from __future__ import annotations


class FactorRegistry:

    def __init__(self):
        self._ids: dict[str, int] = {}
        self._names: list[str] = []

    def id(self, name: str) -> int:
        """Returns the id for ``name``, registering it on first
        sight."""
        existing = self._ids.get(name)
        if existing is not None:
            return existing
        i = len(self._names)
        self._ids[name] = i
        self._names.append(name)
        return i

    def id_if_present(self, name: str) -> int:
        """The id if registered, -1 otherwise (never registers)."""
        return self._ids.get(name, -1)

    def name(self, id_: int) -> str:
        return self._names[id_]

    def size(self) -> int:
        return len(self._names)
