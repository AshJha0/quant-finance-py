"""Strategy DSL (port of Java ``com.quantfinlib.dsl``).

Compose :mod:`~quantfinlib.dsl.rules` into a
:class:`~quantfinlib.dsl.strategy_builder.StrategyBuilder`-built
strategy with a fluent API. Rules are predicates over a bar INDEX into
precomputed indicator arrays -- see :mod:`rules` for why that shape
makes look-ahead bias structurally harder to introduce.
"""

from quantfinlib.dsl import rules
from quantfinlib.dsl.rule import Rule
from quantfinlib.dsl.strategy_builder import DslStrategy, StrategyBuilder

__all__ = ["Rule", "rules", "StrategyBuilder", "DslStrategy"]
