"""Trade-level analytics (port of Java ``backtest.TradeAnalytics``).

The numbers a discretionary reviewer asks for that a Sharpe ratio hides.
Two strategies with the same Sharpe can have completely different trade
signatures: one wins small and often with a few large losses (a hidden
short-gamma profile that blows up), the other loses small and often with
rare large wins (trend following, hard to hold). These statistics expose
the difference:

* **expectancy** — average P&L per trade,
  ``win_rate * avg_win - loss_rate * avg_loss``: the single number that
  says whether the edge survives being averaged over every trade;
* **payoff ratio** — ``avg_win / avg_loss``: paired with win rate it IS
  the strategy's character. A 40% win rate needs a payoff above 1.5 to
  have positive expectancy — the arithmetic that kills most "high win
  rate" systems whose few losses are huge;
* **streaks** — the longest run of consecutive losers (and winners): the
  number that decides whether the strategy can be SAT through. A
  positive-expectancy system with an 11-trade losing streak gets turned
  off by its owner at trade 8;
* **Kelly fraction** — ``W - (1 - W) / R`` for win rate W and payoff R:
  the growth-optimal bet size the trade record implies, and a reality
  check (a Kelly above ~0.25 usually means the sample is too small or
  the wins too lucky);
* **hold times** — average bars held for winners vs losers: when losers
  are held far longer than winners, that is the disposition effect
  showing up in the tape.

All statistics are on realized :class:`~quantfinlib.backtest.trade.Trade`
P&L; a strategy with no losing trades reports an infinite payoff ratio
and Kelly clamped to 1 (bet everything — which is exactly the over-fit
warning you want). Static, deterministic, research lane. Complements
``PerformanceAnalytics`` (equity-curve metrics) and
``validation.MonteCarloTradeShuffle`` (is the SEQUENCE luck?).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from quantfinlib.backtest.trade import Trade


class TradeAnalytics:
    """Static trade-record statistics; see the module docstring."""

    @dataclass(frozen=True)
    class Result:
        """Trade-record statistics.

        Attributes:
            count: Number of trades.
            win_rate: Fraction of strictly-positive trades.
            expectancy: Average P&L per trade.
            avg_win: Mean P&L of winning trades (0 if none).
            avg_loss: Mean magnitude of losing trades (0 if none).
            payoff_ratio: ``avg_win / avg_loss`` (+inf if no losers).
            max_win_streak: Longest run of consecutive winners.
            max_loss_streak: Longest run of consecutive losers.
            kelly_fraction: ``W - (1 - W) / R``, clamped to [0, 1].
            avg_bars_held_winners: Mean holding period of winners.
            avg_bars_held_losers: Mean holding period of losers.
        """

        count: int
        win_rate: float
        expectancy: float
        avg_win: float
        avg_loss: float
        payoff_ratio: float
        max_win_streak: int
        max_loss_streak: int
        kelly_fraction: float
        avg_bars_held_winners: float
        avg_bars_held_losers: float

    @staticmethod
    def analyze(trades: Sequence[Trade]) -> "TradeAnalytics.Result":
        """Analyzes a non-empty list of completed trades.

        Raises:
            ValueError: if ``trades`` is empty or any pnl is non-finite.
        """
        n = len(trades)
        if n == 0:
            raise ValueError("need at least one trade")
        wins = 0
        win_sum = 0.0
        loss_sum = 0.0
        win_bars = 0
        loss_bars = 0
        max_win = max_loss = cur_win = cur_loss = 0
        for t in trades:
            if not math.isfinite(t.pnl):
                raise ValueError("non-finite trade pnl")
            if t.pnl > 0:
                wins += 1
                win_sum += t.pnl
                win_bars += t.bars_held
                cur_win += 1
                cur_loss = 0
                max_win = max(max_win, cur_win)
            elif t.pnl < 0:
                loss_sum += -t.pnl
                loss_bars += t.bars_held
                cur_loss += 1
                cur_win = 0
                max_loss = max(max_loss, cur_loss)
            else:
                # Scratch trade breaks both streaks; belongs to neither leg.
                cur_win = 0
                cur_loss = 0
        losers = sum(1 for t in trades if t.pnl < 0)
        win_rate = wins / n
        avg_win = win_sum / wins if wins > 0 else 0.0
        avg_loss = loss_sum / losers if losers > 0 else 0.0
        loss_rate = losers / n
        expectancy = win_rate * avg_win - loss_rate * avg_loss
        payoff = avg_win / avg_loss if avg_loss > 0 else math.inf
        # Kelly = W - (1-W)/R, clamped: no losers -> bet everything (the
        # over-fit tell); non-positive edge -> zero. A zero payoff (only
        # losers) is Java's (1-W)/0.0 = +inf, W - inf = -inf -> clamp 0;
        # spelled out because Python float division by zero raises.
        if avg_loss == 0:
            kelly = 1.0
        elif payoff == 0:
            kelly = 0.0
        else:
            kelly = win_rate - (1 - win_rate) / payoff
            kelly = max(0.0, min(1.0, kelly))
        avg_win_bars = win_bars / wins if wins > 0 else 0.0
        avg_loss_bars = loss_bars / losers if losers > 0 else 0.0
        return TradeAnalytics.Result(n, win_rate, expectancy, avg_win, avg_loss,
                                     payoff, max_win, max_loss, kelly,
                                     avg_win_bars, avg_loss_bars)
