"""Bit-exact port of ``java.util.SplittableRandom``'s single-seed
constructor and ``nextDouble()`` path.

Only the piece ``MonteCarloSimulator`` actually uses is reproduced: the
public ``new SplittableRandom(long seed)`` constructor sets the
generator's internal ``seed`` field to the given value and its
``gamma`` field to the fixed golden-ratio increment (the more elaborate
gamma-selection algorithm only runs for ``split()``-created child
generators, which this library never uses). Each ``nextLong()`` call
then does the textbook SplitMix64 step -- ``seed += gamma; return
mix64(seed)`` -- and ``nextDouble()`` takes the top 53 bits of that.

All state is carried as Python ints masked to 64 bits; there is no
sign ambiguity because every value here is used exclusively through
unsigned bit operations (right shifts, XORs, multiplies mod 2**64),
exactly as the JVM's ``>>>`` and wraparound ``long`` arithmetic would
produce.
"""

from __future__ import annotations

_MASK64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_DOUBLE_UNIT = 1.0 / (1 << 53)


def mix64(z: int) -> int:
    """Java's ``SplittableRandom.mix64`` -- the Stafford "mix13" finalizer."""
    z &= _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> 31)
    return z & _MASK64


def mix_seed(seed: int, stream: int) -> int:
    """Java's ``MonteCarloSimulator.mix(seed, stream)``: derives one
    per-path seed from the simulator's base seed and a 0-based path
    index."""
    z = (seed + (stream + 1) * _GOLDEN_GAMMA) & _MASK64
    return mix64(z)


class SplittableRandom64:
    """Minimal reproduction of ``java.util.SplittableRandom`` as
    constructed from a single ``long`` seed (no ``split()`` support --
    not needed here)."""

    __slots__ = ("_seed",)

    def __init__(self, seed: int) -> None:
        self._seed = seed & _MASK64

    def next_long(self) -> int:
        self._seed = (self._seed + _GOLDEN_GAMMA) & _MASK64
        return mix64(self._seed)

    def next_double(self) -> float:
        return (self.next_long() >> 11) * _DOUBLE_UNIT
