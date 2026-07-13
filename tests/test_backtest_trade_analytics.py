"""Pins for quantfinlib.backtest.trade / trade_analytics.

Java sources: BacktestAnalyticsRoundTest.java (hand arithmetic, streaks,
no-losers degeneracy) and TradeAnalyticsEdgeTest.java (all-scratch,
all-loss, expectancy = mean-P&L identity). Same tolerances as the Java
asserts, every value derivable on paper.
"""

import math

import pytest

from quantfinlib.backtest import Trade, TradeAnalytics


def _trade(pnl: float, bars: int = 1) -> Trade:
    return Trade("X", 0, bars, 0, bars, 100.0, 100.0 + pnl, 1.0, pnl,
                 pnl / 100, Trade.REASON_SIGNAL)


# ------------------------------------------------------------------- Trade

def test_trade_record_derived_fields():
    t = _trade(25.0, 4)
    assert t.is_win
    assert t.bars_held == 4          # exit_index - entry_index = 4 - 0
    assert not _trade(-5.0).is_win
    assert not _trade(0.0).is_win    # scratch is not a win (strict >)


# --------------------------------------------------------- TradeAnalytics

def test_trade_statistics_match_hand_arithmetic():
    # 3 wins (+100,+50,+50 over 2/4/6 bars), 2 losses (-40,-60 over 8/10).
    trades = [_trade(100, 2), _trade(-40, 8), _trade(50, 4),
              _trade(-60, 10), _trade(50, 6)]
    r = TradeAnalytics.analyze(trades)

    assert r.count == 5
    assert r.win_rate == pytest.approx(0.6, abs=1e-12)
    assert r.avg_win == pytest.approx(200.0 / 3, abs=1e-12)     # 200/3
    assert r.avg_loss == pytest.approx(50.0, abs=1e-12)         # 100/2
    # Expectancy = 0.6 * (200/3) - 0.4 * 50 = 40 - 20 = 20.
    assert r.expectancy == pytest.approx(0.6 * (200.0 / 3) - 0.4 * 50, abs=1e-12)
    assert r.payoff_ratio == pytest.approx((200.0 / 3) / 50.0, abs=1e-12)
    # Kelly = W - (1-W)/R = 0.6 - 0.4/(4/3) = 0.3.
    assert r.kelly_fraction == pytest.approx(
        0.6 - 0.4 / ((200.0 / 3) / 50.0), abs=1e-12)
    # Winners held 2,4,6 -> avg 4; losers 8,10 -> avg 9.
    assert r.avg_bars_held_winners == pytest.approx(4.0, abs=1e-12)
    assert r.avg_bars_held_losers == pytest.approx(9.0, abs=1e-12)


def test_streaks_and_scratch_trade_handling():
    # W W L L L 0 W : max win streak 2, max loss streak 3; the scratch
    # trade in the middle breaks both streaks.
    trades = [_trade(10), _trade(10), _trade(-5), _trade(-5), _trade(-5),
              _trade(0), _trade(10)]
    r = TradeAnalytics.analyze(trades)
    assert r.max_win_streak == 2
    assert r.max_loss_streak == 3
    # A scratch trade is neither a win nor a loss: 3 wins of 7.
    assert r.win_rate == pytest.approx(3.0 / 7, abs=1e-12)


def test_no_losers_gives_infinite_payoff_and_full_kelly():
    r = TradeAnalytics.analyze([_trade(10), _trade(20), _trade(5)])
    assert r.payoff_ratio == math.inf
    # No losers -> bet everything (the over-fit tell).
    assert r.kelly_fraction == 1.0
    with pytest.raises(ValueError):
        TradeAnalytics.analyze([])


def test_non_finite_pnl_is_rejected():
    with pytest.raises(ValueError):
        TradeAnalytics.analyze([_trade(math.nan)])


def test_all_scratch_trades_produce_the_documented_no_loser_degeneracy():
    # Every trade breaks even: no wins, no losses. Pinning CURRENT
    # behavior: avg_loss == 0 takes the documented "no losers" branch, so
    # payoff is +inf and Kelly clamps to 1 even though there are no
    # winners either — the same over-fit tell as a loss-free record.
    r = TradeAnalytics.analyze([_trade(0, 1), _trade(0, 2), _trade(0, 3)])
    assert r.count == 3
    assert r.win_rate == 0.0
    assert r.avg_win == 0.0
    assert r.avg_loss == 0.0
    assert r.expectancy == 0.0
    assert r.payoff_ratio == math.inf
    assert r.kelly_fraction == 1.0
    # Scratches break streaks and belong to neither leg.
    assert r.max_win_streak == 0
    assert r.max_loss_streak == 0
    assert r.avg_bars_held_winners == 0.0
    assert r.avg_bars_held_losers == 0.0


def test_all_losses_clamp_kelly_to_zero():
    # W = 0 and R = avg_win/avg_loss = 0: Kelly = 0 - 1/0 = -inf, which
    # the clamp turns into 0 — never bet a strategy that only loses.
    r = TradeAnalytics.analyze([_trade(-10, 2), _trade(-20, 4), _trade(-30, 6)])
    assert r.win_rate == 0.0
    assert r.avg_loss == pytest.approx(20.0, abs=1e-12)   # (10+20+30)/3
    assert r.payoff_ratio == 0.0
    assert r.kelly_fraction == 0.0
    assert r.max_loss_streak == 3
    assert r.max_win_streak == 0
    assert r.expectancy == pytest.approx(-20.0, abs=1e-12)  # 0*0 - 1*20
    # The single-loss record is the same story.
    single = TradeAnalytics.analyze([_trade(-50)])
    assert single.kelly_fraction == 0.0
    assert single.max_loss_streak == 1


def test_expectancy_is_exactly_the_mean_pnl_including_scratches():
    # win_rate*avg_win - loss_rate*avg_loss = (winSum - lossSum)/n, i.e.
    # the plain average P&L per trade with scratches diluting it:
    # (+30 - 10 + 0)/3 = 20/3.
    r = TradeAnalytics.analyze([_trade(30), _trade(-10), _trade(0)])
    assert r.expectancy == pytest.approx(20.0 / 3, abs=1e-12)
