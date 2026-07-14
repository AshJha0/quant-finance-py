"""Strategy Builder DSL (port of Java ``dsl.StrategyBuilder``).

Compose entry/exit rules, stop loss and take profit into a
backtestable strategy with a fluent API::

    close = series.closes()
    fast = Indicators.ema(close, 12)
    slow = Indicators.ema(close, 26)

    result = (StrategyBuilder.named("EMA momentum")
              .enter_when(rules.cross_above(fast, slow))
              .exit_when(rules.cross_below(fast, slow))
              .with_stop_loss(0.03)
              .with_take_profit(0.08)
              .build()
              .backtest(series, 100_000))
"""

from __future__ import annotations

from quantfinlib.backtest.backtest_config import BacktestConfig
from quantfinlib.backtest.backtester import BacktestResult, Backtester
from quantfinlib.backtest.signal import Signal
from quantfinlib.backtest.trading_strategy import TradingStrategy
from quantfinlib.data.bar_series import BarSeries
from quantfinlib.dsl.rule import Rule

_NEVER = Rule(lambda i: False)


class StrategyBuilder:
    def __init__(self, name: str) -> None:
        self._name = name
        self._entry_rule: Rule | None = None
        self._exit_rule: Rule | None = None
        self._stop_loss_pct = 0.0
        self._take_profit_pct = 0.0

    @staticmethod
    def named(name: str) -> "StrategyBuilder":
        return StrategyBuilder(name)

    def enter_when(self, rule: Rule) -> "StrategyBuilder":
        self._entry_rule = rule
        return self

    def exit_when(self, rule: Rule) -> "StrategyBuilder":
        self._exit_rule = rule
        return self

    def with_stop_loss(self, pct: float) -> "StrategyBuilder":
        """Per-trade stop loss as a fraction of the entry price (0.03 = 3%)."""
        self._stop_loss_pct = pct
        return self

    def with_take_profit(self, pct: float) -> "StrategyBuilder":
        """Per-trade take profit as a fraction of the entry price (0.08 = 8%)."""
        self._take_profit_pct = pct
        return self

    def build(self) -> "DslStrategy":
        if self._entry_rule is None:
            raise ValueError("entry rule is required")
        exit_rule = self._exit_rule if self._exit_rule is not None else _NEVER
        return DslStrategy(
            self._name, self._entry_rule, exit_rule, self._stop_loss_pct, self._take_profit_pct
        )


class DslStrategy(TradingStrategy):
    """A rule-based strategy produced by :class:`StrategyBuilder`."""

    def __init__(
        self, name: str, entry: Rule, exit_: Rule, stop_loss_pct: float, take_profit_pct: float
    ) -> None:
        self._name = name
        self._entry = entry
        self._exit = exit_
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct

    def name(self) -> str:
        return self._name

    def init(self, series: BarSeries) -> None:
        # Rules close over precomputed indicator arrays; nothing to do.
        pass

    def on_bar(self, index: int) -> Signal:
        if self._entry.is_satisfied(index):
            return Signal.BUY
        if self._exit.is_satisfied(index):
            return Signal.SELL
        return Signal.HOLD

    def stop_loss_pct(self) -> float:
        return self._stop_loss_pct

    def take_profit_pct(self) -> float:
        return self._take_profit_pct

    def backtest(self, series: BarSeries, capital_or_config) -> BacktestResult:
        """Convenience: run a backtest with default costs and this
        strategy's risk settings when given a starting capital amount, or
        with the supplied :class:`BacktestConfig` directly."""
        if isinstance(capital_or_config, BacktestConfig):
            config = capital_or_config
        else:
            config = BacktestConfig.defaults().with_initial_capital(capital_or_config)
        return Backtester.run(self, series, config)
