"""Explicit, engine-independent random stream shared by the reference and batch engines.

Both engines draw every chance outcome (deal order, recycle shuffles, starting-seat tie breaks) from the same
splitmix64 sequence, so identical seeds give identical chance outcomes and the engines can be compared
directly. Agents never see this state.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


@njit(cache=True)
def next_u64(state):
    """Advance ``state`` (uint64 array of length >= 1, index 0) and return the next output."""
    z = state[0] + _GAMMA
    state[0] = z
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def bounded(state, n):
    """Integer in [0, n) via multiply-shift on the top 32 bits (bias < n / 2**32, negligible for n <= 150)."""
    hi = next_u64(state) >> np.uint64(32)
    return np.int64((hi * np.uint64(n)) >> np.uint64(32))


@njit(cache=True)
def shuffle_prefix(state, arr, n):
    """In-place Fisher-Yates shuffle of arr[:n]."""
    for i in range(n - 1, 0, -1):
        j = bounded(state, i + 1)
        t = arr[i]
        arr[i] = arr[j]
        arr[j] = t


def seed_state(seed: int) -> np.ndarray:
    """Derive an RNG state from an integer seed (mixed so nearby seeds are unrelated)."""
    s = np.array([np.uint64(seed & 0xFFFFFFFFFFFFFFFF)], dtype=np.uint64)
    next_u64(s)
    return s
