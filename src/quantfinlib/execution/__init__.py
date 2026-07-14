"""Execution algorithms (port of the pure-logic subset of Java
``com.quantfinlib.execution``).

Schedulers (TWAP, VWAP, POV, implementation-shortfall, WMR fixing),
smart order routing (hot-lane greedy and the full-checklist adaptive
router with its venue scorecard), live benchmark-tracking and
portfolio-level execution, and the standalone execution primitives
(dark pool, mid-peg, iceberg, spread legging, futures roll, anti-gaming
jitter, order placement policy, UCB1 arm selection).

Out of scope for this port (see each module for the reasoning where
relevant):

* ``SmartOrderRouter`` -- the research-lane all-in-price router is
  superseded here by :class:`~quantfinlib.execution.adaptive_sor.AdaptiveSor`
  (the full-checklist router) and
  :class:`~quantfinlib.execution.hft_sor.HftSor` (the hot-lane greedy
  router); porting a third, narrower router added nothing the task
  list asked for.
* ``VenueBenchmark`` -- the batch post-trade venue-quality report
  (fill rate / effective spread / markout over a sample list); the
  *streaming* counterpart
  (:class:`~quantfinlib.execution.venue_scorecard.VenueScorecard`) is
  ported and is what :class:`~quantfinlib.execution.adaptive_sor.AdaptiveSor`
  actually consumes.
"""

from quantfinlib.execution.adaptive_sor import AdaptiveSor
from quantfinlib.execution.adaptive_sor import Config as AdaptiveSorConfig
from quantfinlib.execution.adaptive_sor import RouteLeg, RoutingDecision
from quantfinlib.execution.anti_gaming_jitter import AntiGamingJitter
from quantfinlib.execution.benchmark_executor import (Benchmark,
                                                       BenchmarkExecutor,
                                                       MarketState)
from quantfinlib.execution.dark_pool_simulator import DarkPoolSimulator, Fill
from quantfinlib.execution.futures_roll_algo import FuturesRollAlgo
from quantfinlib.execution.hft_sor import HftSor
from quantfinlib.execution.iceberg_order import IcebergOrder
from quantfinlib.execution.implementation_shortfall_scheduler import (
    risk_aversion_for_front_load, schedule as implementation_shortfall_schedule)
from quantfinlib.execution.liquidity_seeking_algo import (
    Config as LiquiditySeekingConfig, LiquiditySeekingAlgo)
from quantfinlib.execution.mid_peg_tracker import MidPegTracker
from quantfinlib.execution.order_placement_policy import (Placement,
                                                           PostRegion, decide,
                                                           post_region)
from quantfinlib.execution.portfolio_executor import (
    Config as PortfolioExecutorConfig, PortfolioExecutor)
from quantfinlib.execution.pov_tracker import PovTracker
from quantfinlib.execution.slice import Slice
from quantfinlib.execution.spread_execution_algo import (Children,
                                                          SpreadExecutionAlgo)
from quantfinlib.execution.twap_scheduler import (
    schedule as twap_schedule, schedule_randomized as twap_schedule_randomized)
from quantfinlib.execution.ucb1_selector import Ucb1Selector
from quantfinlib.execution.venue_quote import VenueQuote
from quantfinlib.execution.venue_scorecard import VenueScorecard
from quantfinlib.execution.vwap_scheduler import allocate_proportionally
from quantfinlib.execution.vwap_scheduler import schedule as vwap_schedule
from quantfinlib.execution.wmr_fixing_scheduler import WINDOW_MILLIS as WMR_WINDOW_MILLIS
from quantfinlib.execution.wmr_fixing_scheduler import schedule as wmr_fixing_schedule

__all__ = [
    "AdaptiveSor",
    "AdaptiveSorConfig",
    "AntiGamingJitter",
    "Benchmark",
    "BenchmarkExecutor",
    "Children",
    "DarkPoolSimulator",
    "Fill",
    "FuturesRollAlgo",
    "HftSor",
    "IcebergOrder",
    "LiquiditySeekingAlgo",
    "LiquiditySeekingConfig",
    "MarketState",
    "MidPegTracker",
    "Placement",
    "PortfolioExecutor",
    "PortfolioExecutorConfig",
    "PostRegion",
    "PovTracker",
    "RouteLeg",
    "RoutingDecision",
    "Slice",
    "SpreadExecutionAlgo",
    "Ucb1Selector",
    "VenueQuote",
    "VenueScorecard",
    "WMR_WINDOW_MILLIS",
    "allocate_proportionally",
    "decide",
    "implementation_shortfall_schedule",
    "post_region",
    "risk_aversion_for_front_load",
    "twap_schedule",
    "twap_schedule_randomized",
    "vwap_schedule",
    "wmr_fixing_schedule",
]
