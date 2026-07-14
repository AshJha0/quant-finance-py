"""Day-type-aware seasonality (port of Java
``microstructure.DayTypeProfiles``): not every trading day has the
same shape. Options-expiry days trade 2-3x normal volume with a
violent close; half days compress the whole U-curve into a morning; FX
fixing days (month-end, the 4pm London WM/R window) concentrate flow
around the fix. A single averaged profile is wrong on exactly the days
that matter most, so this container holds one independently-learned
curve per day type -- one
:class:`~quantfinlib.microstructure.volume_curve.VolumeCurve`,
:class:`~quantfinlib.microstructure.volatility_curve.VolatilityCurve`,
or
:class:`~quantfinlib.microstructure.spread_forecaster.SpreadForecaster`
each -- and the caller selects today's profile once at session
start::

    # 0=regular, 1=expiry, 2=half day  (the caller owns the taxonomy)
    volume = DayTypeProfiles(3, lambda day_type: VolumeCurve(78, 0.1))
    today = volume.profile(1 if calendar.is_expiry(date) else 0)
    today.on_volume(bucket, qty)   # learns ONLY the expiry-day shape

The trade-off is honest and unavoidable: a per-type profile learns
from only that type's sessions, so rare types (12 expiries a year)
converge slowly. Seed a new type from the regular-day profile via the
curve's own seeding method when one exists, or accept the slower ramp.
All profiles are constructed eagerly up front -- selection is a plain
list index.
"""

from __future__ import annotations

from typing import Callable, Generic, List, TypeVar

T = TypeVar("T")


class DayTypeProfiles(Generic[T]):
    """One independently-learned profile per day type; see the module
    docstring."""

    __slots__ = ("_profiles",)

    def __init__(self, day_types: int, factory: Callable[[int], T]) -> None:
        """
        Args:
            day_types: number of day types in the caller's taxonomy,
                e.g. 3 for regular / expiry / half-day (equities) or
                regular / month-end-fixing (FX).
            factory: builds one fresh, identically-configured profile
                per day type; called with the day-type index it is
                building for (ignore the argument for an
                index-independent factory, e.g.
                ``lambda _: VolumeCurve()``).
        """
        if day_types < 1:
            raise ValueError("need dayTypes >= 1")
        self._profiles: List[T] = [factory(i) for i in range(day_types)]

    def profile(self, day_type: int) -> T:
        """The independently-learned profile for ``day_type``."""
        return self._profiles[day_type]

    def day_types(self) -> int:
        return len(self._profiles)
