"""Direct-vs-synthetic cross execution math, ported from Java
SyntheticCrossTest.
"""

import math

import pytest

from quantfinlib.fx import CrossOp, SyntheticCross

EU_BID = 1.0850
EU_ASK = 1.0852
UJ_BID = 161.50
UJ_ASK = 161.52


def test_multiply_crosses_both_spreads():
    syn_ask = SyntheticCross.synthetic_ask(CrossOp.MULTIPLY, EU_BID, EU_ASK, UJ_BID, UJ_ASK)
    syn_bid = SyntheticCross.synthetic_bid(CrossOp.MULTIPLY, EU_BID, EU_ASK, UJ_BID, UJ_ASK)
    assert syn_ask == pytest.approx(EU_ASK * UJ_ASK, abs=1e-9)
    assert syn_bid == pytest.approx(EU_BID * UJ_BID, abs=1e-9)
    assert syn_ask > syn_bid, "the synthetic must have a positive spread"


def test_divide_uses_opposite_sides_of_the_shared_quote_leg():
    # EURGBP = EURUSD / GBPUSD: buy EUR (ask A), sell GBP (bid B).
    gb_bid = 1.2700
    gb_ask = 1.2702
    syn_ask = SyntheticCross.synthetic_ask(CrossOp.DIVIDE, EU_BID, EU_ASK, gb_bid, gb_ask)
    syn_bid = SyntheticCross.synthetic_bid(CrossOp.DIVIDE, EU_BID, EU_ASK, gb_bid, gb_ask)
    assert syn_ask == pytest.approx(EU_ASK / gb_bid, abs=1e-9)
    assert syn_bid == pytest.approx(EU_BID / gb_ask, abs=1e-9)
    assert syn_ask > syn_bid


def test_route_choice_follows_the_cheaper_all_in():
    syn_ask = EU_ASK * UJ_ASK
    # Direct book wider than the legs: synthetic wins the buy.
    assert SyntheticCross.buy_synthetic_wins(syn_ask + 0.02, CrossOp.MULTIPLY,
                                             EU_BID, EU_ASK, UJ_BID, UJ_ASK)
    # Direct book tighter: direct wins.
    assert not SyntheticCross.buy_synthetic_wins(syn_ask - 0.02, CrossOp.MULTIPLY,
                                                 EU_BID, EU_ASK, UJ_BID, UJ_ASK)
    savings = SyntheticCross.buy_savings(syn_ask + 0.02, CrossOp.MULTIPLY,
                                         EU_BID, EU_ASK, UJ_BID, UJ_ASK)
    assert savings == pytest.approx(0.02, abs=1e-9)
    # Sell mirror.
    syn_bid = EU_BID * UJ_BID
    assert SyntheticCross.sell_synthetic_wins(syn_bid - 0.02, CrossOp.MULTIPLY,
                                              EU_BID, EU_ASK, UJ_BID, UJ_ASK)
    assert not SyntheticCross.sell_synthetic_wins(syn_bid + 0.02, CrossOp.MULTIPLY,
                                                  EU_BID, EU_ASK, UJ_BID, UJ_ASK)


def test_unpriced_routes_never_win():
    assert not SyntheticCross.buy_synthetic_wins(math.nan, CrossOp.MULTIPLY,
                                                 EU_BID, EU_ASK, UJ_BID, UJ_ASK)
    assert not SyntheticCross.buy_synthetic_wins(175.30, CrossOp.MULTIPLY,
                                                 EU_BID, math.nan, UJ_BID, UJ_ASK)
    assert not SyntheticCross.sell_synthetic_wins(175.20, CrossOp.MULTIPLY,
                                                  math.nan, EU_ASK, UJ_BID, UJ_ASK)


def test_zero_priced_legs_never_win_either():
    # Zero is the default and what an empty FxTierBook tier reads as; a
    # zero DIVIDE denominator would produce +Infinity "savings".
    assert not SyntheticCross.sell_synthetic_wins(1.0850, CrossOp.DIVIDE,
                                                  1.10, 1.101, 1.27, 0.0)      # bidA/askB=0 -> inf
    assert not SyntheticCross.buy_synthetic_wins(175.30, CrossOp.MULTIPLY,
                                                 EU_BID, EU_ASK, UJ_BID, 0.0)  # askB=0 -> synth 0
    assert not SyntheticCross.buy_synthetic_wins(0.0, CrossOp.MULTIPLY,
                                                 EU_BID, EU_ASK, UJ_BID, UJ_ASK)  # unquoted direct
    assert math.isnan(SyntheticCross.buy_savings(175.30, CrossOp.MULTIPLY,
                                                 EU_BID, -1.0, UJ_BID, UJ_ASK)), \
        "negative legs are unpriced, not tradable"
