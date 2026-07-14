"""A boolean condition over a bar index (port of Java ``dsl.Rule``).

Typically closes over precomputed indicator arrays. Rules compose with
:meth:`Rule.and_`, :meth:`Rule.or_`, :meth:`Rule.not_`.

(Java's ``Rule`` is a ``@FunctionalInterface`` with default
``and``/``or``/``not`` combinators; since ``and``/``or``/``not`` are
Python keywords, this port names them ``and_``/``or_``/``not_``.)
"""

from __future__ import annotations

from typing import Callable

_Predicate = Callable[[int], bool]


class Rule:
    def __init__(self, predicate: _Predicate) -> None:
        self._predicate = predicate

    def is_satisfied(self, index: int) -> bool:
        return bool(self._predicate(index))

    def __call__(self, index: int) -> bool:
        return self.is_satisfied(index)

    def and_(self, other: "Rule") -> "Rule":
        return Rule(lambda i: self.is_satisfied(i) and other.is_satisfied(i))

    def or_(self, other: "Rule") -> "Rule":
        return Rule(lambda i: self.is_satisfied(i) or other.is_satisfied(i))

    def not_(self) -> "Rule":
        return Rule(lambda i: not self.is_satisfied(i))
