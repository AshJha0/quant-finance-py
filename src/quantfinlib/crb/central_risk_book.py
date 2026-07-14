"""The central risk book (port of Java ``com.quantfinlib.crb.CentralRiskBook``).

One netted view of the firm's market risk across desks and products.
Every instrument is decomposed into a COMMON risk-factor space at
booking time, and that is the entire point: an FX-option delta nets
against a spot position, an equity-option delta nets against cash
shares, and two desks' opposite flows cancel before anyone pays a
spread to the street.

Factor naming and units (query via :meth:`exposure`):

- ``EQ:<sym>`` -- equity delta in book-currency notional (cash shares
  contribute qty*price; options contribute delta*spot*contracts*multiplier);
- ``EQGAMMA:<sym>`` / ``EQVEGA:<sym>`` -- dollar gamma per 1% move
  (gamma*S^2/100 per contract-adjusted unit) and vega per vol POINT;
- ``CCY:<ccy>`` -- currency exposure in NATIVE units of that currency
  (an EURUSD buy of 10M books CCY:EUR +10M euros and CCY:USD -10M*rate
  dollars) -- this is what lets spot, swaps, NDFs and option deltas net
  at the CURRENCY level;
- ``FXPOINTS:<pair>`` -- forward-points risk of swaps: P&L in quote
  units per 1.0 move in the far-near differential;
- ``FXGAMMA:<pair>`` / ``FXVEGA:<pair>`` -- dollar gamma per 1% and vega
  per vol point, in quote-currency units.

Sign convention: positive quantity/notional = the BOOK is long. Book
what the book absorbs (a client sell hits the book as a buy). NDFs
carry currency delta until fixing; the pending non-deliverable notional
is tracked per pair in :meth:`pending_fixing` and released via
:meth:`settle_fixing` once fixings occur. This is a RISK ledger, not a
cash ledger: premium and settlement cash legs are deliberately
untracked, and lifecycle events (expiry, exercise, settlement) re-book
as offsetting flows.

The Java ``persist.Checkpoint`` overnight persistence (writeState/
readState) is not ported -- no ``persist`` lane in this Python port.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantfinlib.crb.factor_registry import FactorRegistry
from quantfinlib.pricing.black_scholes import BlackScholes, OptionType
from quantfinlib.risk import var_engine


@dataclass(frozen=True)
class CrbReport:
    """
    gross_exposure: sum over factors of gross booked notional
    net_exposure: sum over factors of |netted| notional
    netting_efficiency: 1 - net/gross
    var: delta-normal VaR of the netted book
    es: delta-normal ES of the netted book
    standalone_desk_var: sum of per-desk standalone VaRs
    diversification_benefit: standalone_desk_var - var
    """

    gross_exposure: float
    net_exposure: float
    netting_efficiency: float
    var: float
    es: float
    standalone_desk_var: float
    diversification_benefit: float


class CentralRiskBook:

    def __init__(self):
        self._registry = FactorRegistry()
        self._net: list[float] = [0.0] * 16
        self._gross: list[float] = [0.0] * 16
        self._desk_net: dict[str, list[float]] = {}
        self._pending_fixings: dict[str, float] = {}
        self._flows_booked = 0

    # ------------------------------------------------------------------
    # Booking -- equities
    # ------------------------------------------------------------------

    def book_cash_equity(self, desk: str, symbol: str, qty: float, price: float) -> None:
        """Cash equity: ``qty`` shares (signed) at ``price``."""
        _require_desk(desk)
        _require_finite(qty, "qty")
        _require_positive(price, "price")
        self._add(desk, f"EQ:{symbol}", qty * price)
        self._flows_booked += 1

    def book_equity_option(self, desk: str, symbol: str, option_type: OptionType,
                           contracts: float, multiplier: float, spot: float,
                           strike: float, rate: float, carry: float, vol: float,
                           time_years: float) -> None:
        """Listed equity option: ``contracts`` signed, ``multiplier``
        shares per contract (100 for US listed). Greeks come from
        ``BlackScholes`` with the house q-convention (``carry`` =
        dividend yield)."""
        _require_desk(desk)
        _require_finite(contracts, "contracts")
        _require_finite(rate, "rate")
        _require_finite(carry, "carry")
        _require_positive(multiplier, "multiplier")
        _require_positive(spot, "spot")
        _require_positive(strike, "strike")
        _require_positive(vol, "vol")
        _require_positive(time_years, "timeYears")
        units = contracts * multiplier
        delta = BlackScholes.delta(option_type, spot, strike, rate, carry, vol, time_years)
        gamma = BlackScholes.gamma(spot, strike, rate, carry, vol, time_years)
        vega = BlackScholes.vega(spot, strike, rate, carry, vol, time_years)
        # Compute-validate-COMMIT: every leg must be finite before any
        # add(), or an overflow on the second leg would leave a
        # half-booked flow in the netted arrays.
        delta_leg = units * delta * spot
        gamma_leg = units * gamma * spot * spot / 100          # $ per 1%
        vega_leg = units * vega / 100                           # per vol point
        _require_finite(delta_leg, "delta leg")
        _require_finite(gamma_leg, "gamma leg")
        _require_finite(vega_leg, "vega leg")
        self._add(desk, f"EQ:{symbol}", delta_leg)
        self._add(desk, f"EQGAMMA:{symbol}", gamma_leg)
        self._add(desk, f"EQVEGA:{symbol}", vega_leg)
        self._flows_booked += 1

    # ------------------------------------------------------------------
    # Booking -- FX
    # ------------------------------------------------------------------

    def book_fx_spot(self, desk: str, pair: str, base_notional: float, rate: float) -> None:
        """FX spot on ``pair`` ("EURUSD"): buy ``base_notional`` of the
        base currency (signed) at ``rate``. Decomposes into the two
        CURRENCY legs so it nets against every other product's FX
        delta."""
        _require_desk(desk)
        _require_pair(pair)
        _require_finite(base_notional, "baseNotional")
        _require_positive(rate, "rate")
        quote_leg = -base_notional * rate
        _require_finite(quote_leg, "quote leg")
        self._add(desk, f"CCY:{pair[0:3]}", base_notional)
        self._add(desk, f"CCY:{pair[3:6]}", quote_leg)
        self._flows_booked += 1

    def book_fx_swap(self, desk: str, pair: str, base_notional: float,
                     near_rate: float, far_rate: float) -> None:
        """FX swap (buy-sell base for positive notional): near leg at
        ``near_rate``, far leg back at ``far_rate``. The base-currency
        legs cancel EXACTLY; what remains is the quote-currency
        cash-flow imbalance and the forward-POINTS risk."""
        _require_desk(desk)
        _require_pair(pair)
        _require_finite(base_notional, "baseNotional")
        _require_positive(near_rate, "nearRate")
        _require_positive(far_rate, "farRate")
        quote_imbalance = base_notional * (far_rate - near_rate)
        _require_finite(quote_imbalance, "quote-leg imbalance")
        self._add(desk, f"CCY:{pair[3:6]}", quote_imbalance)
        self._add(desk, f"FXPOINTS:{pair}", -base_notional)
        self._flows_booked += 1

    def book_ndf(self, desk: str, pair: str, base_notional: float, fwd_rate: float) -> None:
        """NDF: buy ``base_notional`` of base forward at ``fwd_rate``.
        Economically a forward until the fixing -- full currency delta
        on both legs -- with the non-deliverable notional tracked per
        pair."""
        _require_desk(desk)
        _require_pair(pair)
        _require_finite(base_notional, "baseNotional")
        _require_positive(fwd_rate, "fwdRate")
        quote_leg = -base_notional * fwd_rate
        _require_finite(quote_leg, "quote leg")
        self._add(desk, f"CCY:{pair[0:3]}", base_notional)
        self._add(desk, f"CCY:{pair[3:6]}", quote_leg)
        self._pending_fixings[pair] = self._pending_fixings.get(pair, 0.0) + abs(base_notional)
        self._flows_booked += 1

    def book_fx_option(self, desk: str, pair: str, option_type: OptionType,
                       base_notional: float, spot_rate: float, strike: float,
                       domestic_rate: float, foreign_rate: float, vol: float,
                       time_years: float) -> None:
        """FX option via Garman-Kohlhagen (``BlackScholes`` with carry =
        foreign rate): ``base_notional`` signed (long calls on base).
        Delta decomposes into the two currency legs; gamma/vega stay
        pair-keyed in quote-currency units."""
        _require_desk(desk)
        _require_pair(pair)
        _require_finite(base_notional, "baseNotional")
        _require_finite(domestic_rate, "domesticRate")
        _require_finite(foreign_rate, "foreignRate")
        _require_positive(spot_rate, "spotRate")
        _require_positive(strike, "strike")
        _require_positive(vol, "vol")
        _require_positive(time_years, "timeYears")
        delta = BlackScholes.delta(option_type, spot_rate, strike, domestic_rate,
                                   foreign_rate, vol, time_years)
        gamma = BlackScholes.gamma(spot_rate, strike, domestic_rate, foreign_rate,
                                   vol, time_years)
        vega = BlackScholes.vega(spot_rate, strike, domestic_rate, foreign_rate,
                                 vol, time_years)
        # Compute-validate-COMMIT (see book_equity_option).
        delta_base = base_notional * delta
        quote_leg = -delta_base * spot_rate
        gamma_leg = base_notional * gamma * spot_rate * spot_rate / 100
        vega_leg = base_notional * vega / 100
        _require_finite(delta_base, "delta leg")
        _require_finite(quote_leg, "quote leg")
        _require_finite(gamma_leg, "gamma leg")
        _require_finite(vega_leg, "vega leg")
        self._add(desk, f"CCY:{pair[0:3]}", delta_base)
        self._add(desk, f"CCY:{pair[3:6]}", quote_leg)
        self._add(desk, f"FXGAMMA:{pair}", gamma_leg)
        self._add(desk, f"FXVEGA:{pair}", vega_leg)
        self._flows_booked += 1

    # ------------------------------------------------------------------
    # The netted view
    # ------------------------------------------------------------------

    def exposure(self, factor: str) -> float:
        """Net exposure on a factor (0 for a factor never booked)."""
        id_ = self._registry.id_if_present(factor)
        return 0.0 if id_ < 0 or id_ >= len(self._net) else self._net[id_]

    def gross_exposure(self, factor: str) -> float:
        """Gross (sum of |flow|) on a factor -- what the desks did
        severally."""
        id_ = self._registry.id_if_present(factor)
        return 0.0 if id_ < 0 or id_ >= len(self._gross) else self._gross[id_]

    def desk_exposure(self, desk: str, factor: str) -> float:
        """One desk's net contribution to a factor."""
        d = self._desk_net.get(desk)
        if d is None:
            return 0.0
        id_ = self._registry.id_if_present(factor)
        return 0.0 if id_ < 0 or id_ >= len(d) else d[id_]

    def net_exposures(self) -> list[float]:
        """Net exposures over all factors, indexed by registry id."""
        n = self._registry.size()
        out = [0.0] * n
        m = min(len(self._net), n)
        out[:m] = self._net[:m]
        return out

    def netting_efficiency(self) -> float:
        """How much risk the netting destroyed before anyone hedged:
        ``1 - sum|net| / sum(gross)`` -- 0 when every flow is one-way,
        -> 1 when the desks' flows offset each other entirely."""
        n = min(self._registry.size(), len(self._net))
        sum_net = sum(abs(self._net[i]) for i in range(n))
        sum_gross = sum(self._gross[i] for i in range(n))
        return 0.0 if sum_gross <= 0 else 1 - sum_net / sum_gross

    def pending_fixing(self, pair: str) -> float:
        """Non-deliverable notional still awaiting its fixing, per
        pair."""
        return self._pending_fixings.get(pair, 0.0)

    def settle_fixing(self, pair: str, notional: float) -> None:
        """Releases ``notional`` (positive, gross) of pending fixing
        after the fixing occurs. Over-settling raises: releasing more
        than is pending means the caller's fixing ledger disagrees with
        the book's."""
        _require_pair(pair)
        _require_positive(notional, "notional")
        pending = self._pending_fixings.get(pair, 0.0)
        if notional > pending + 1e-9:
            raise ValueError(
                f"settling {notional} but only {pending} is pending on {pair}")
        remaining = pending - notional
        if remaining <= 1e-9:
            self._pending_fixings.pop(pair, None)
        else:
            self._pending_fixings[pair] = remaining

    def flows_booked(self) -> int:
        return self._flows_booked

    def factors(self) -> FactorRegistry:
        return self._registry

    def desks(self) -> frozenset[str]:
        return frozenset(self._desk_net.keys())

    # ------------------------------------------------------------------
    # Risk report
    # ------------------------------------------------------------------

    def report(self, covariance, confidence: float) -> CrbReport:
        """The book-level risk report. ``covariance`` is over the
        factor space in REGISTRY ORDER (``factors().name(i)``) with
        entries in (factor-return)^2 units matching each factor's
        exposure units. The headline number is the diversification
        benefit: standalone desk VaRs minus the netted book's VaR."""
        n = self._registry.size()
        if len(covariance) != n:
            raise ValueError(
                f"covariance must be {n}x{n} over factors().name(i) in registry order")
        net_vec = self.net_exposures()
        book_var = var_engine.delta_normal_var(net_vec, covariance, confidence)
        book_es = var_engine.delta_normal_es(net_vec, covariance, confidence)
        standalone = 0.0
        for d in self._desk_net.values():
            full = [0.0] * n
            m = min(len(d), n)
            full[:m] = d[:m]
            standalone += var_engine.delta_normal_var(full, covariance, confidence)
        booked = min(n, len(self._net))
        sum_net = sum(abs(self._net[i]) for i in range(booked))
        sum_gross = sum(self._gross[i] for i in range(booked))
        return CrbReport(sum_gross, sum_net, self.netting_efficiency(),
                         book_var, book_es, standalone, standalone - book_var)

    # ------------------------------------------------------------------

    def _add(self, desk: str, factor: str, amount: float) -> None:
        if not math.isfinite(amount):
            raise ValueError(f"non-finite exposure for {factor}")
        id_ = self._registry.id(factor)
        if id_ >= len(self._net):
            new_len = max(len(self._net) * 2, id_ + 1)
            self._net = self._net + [0.0] * (new_len - len(self._net))
            self._gross = self._gross + [0.0] * (new_len - len(self._gross))
        self._net[id_] += amount
        self._gross[id_] += abs(amount)
        d = self._desk_net.get(desk)
        if d is None:
            d = [0.0] * len(self._net)
            self._desk_net[desk] = d
        if id_ >= len(d):
            d = d + [0.0] * (len(self._net) - len(d))
            self._desk_net[desk] = d
        d[id_] += amount


def _require_desk(desk: str) -> None:
    if desk is None or desk.strip() == "":
        raise ValueError("desk must be named")


def _require_pair(pair: str) -> None:
    if pair is None or len(pair) != 6:
        raise ValueError(f"pair must be 6 chars like EURUSD: {pair}")


def _require_finite(x: float, name: str) -> None:
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")


def _require_positive(x: float, name: str) -> None:
    if not (x > 0) or x == math.inf:
        raise ValueError(f"{name} must be positive and finite")
