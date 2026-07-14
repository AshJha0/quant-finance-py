"""Backtest execution parameters (port of Java ``backtest.BacktestConfig``)."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest execution parameters. Rates are fractions (0.001 = 10 bps).

    ``stop_loss_pct``/``take_profit_pct`` of 0 disable the respective
    check; a strategy's own stop/take-profit settings override zeros
    here.
    """

    initial_capital: float
    commission_rate: float
    slippage_rate: float
    stop_loss_pct: float
    take_profit_pct: float
    periods_per_year: int

    @staticmethod
    def defaults() -> "BacktestConfig":
        return BacktestConfig(100_000, 0.001, 0.0, 0.0, 0.0, 252)

    def with_initial_capital(self, capital: float) -> "BacktestConfig":
        return replace(self, initial_capital=capital)

    def with_commission(self, rate: float) -> "BacktestConfig":
        return replace(self, commission_rate=rate)

    def with_stop_loss(self, pct: float) -> "BacktestConfig":
        return replace(self, stop_loss_pct=pct)

    def with_take_profit(self, pct: float) -> "BacktestConfig":
        return replace(self, take_profit_pct=pct)
