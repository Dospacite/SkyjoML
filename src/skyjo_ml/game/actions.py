"""Fixed 103-way discrete action space with disjoint, phase-specific offsets."""

from __future__ import annotations

import numpy as np

from .rules import N_ROWS, N_SLOTS

# Setup: 66 unordered pairs of distinct slots.
SETUP_PAIRS = np.array([(i, j) for i in range(N_SLOTS) for j in range(i + 1, N_SLOTS)], dtype=np.int32)
N_SETUP = len(SETUP_PAIRS)  # 66
SETUP_SAME_COLUMN = np.array([i // N_ROWS == j // N_ROWS for i, j in SETUP_PAIRS], dtype=np.bool_)

A_SETUP = 0
A_DRAW = A_SETUP + N_SETUP  # 66: draw from deck
A_TAKE = A_DRAW + 1  # 67..78: take top discard and replace slot i
A_KEEP = A_TAKE + N_SLOTS  # 79..90: keep drawn card and replace slot i
A_REVEAL = A_KEEP + N_SLOTS  # 91..102: discard drawn card and reveal hidden slot i
N_ACTIONS = A_REVEAL + N_SLOTS  # 103

PAIR_INDEX = np.full((N_SLOTS, N_SLOTS), -1, dtype=np.int32)
for _k, (_i, _j) in enumerate(SETUP_PAIRS):
    PAIR_INDEX[_i, _j] = _k
    PAIR_INDEX[_j, _i] = _k


def describe(action: int) -> str:
    """Human-readable action name (columns and rows are 1-based: c1r1 .. c4r3)."""

    def slot(i: int) -> str:
        return f"c{i // N_ROWS + 1}r{i % N_ROWS + 1}"

    a = int(action)
    if A_SETUP <= a < A_DRAW:
        i, j = SETUP_PAIRS[a - A_SETUP]
        return f"reveal {slot(i)}+{slot(j)}"
    if a == A_DRAW:
        return "draw from deck"
    if A_TAKE <= a < A_KEEP:
        return f"take discard -> {slot(a - A_TAKE)}"
    if A_KEEP <= a < A_REVEAL:
        return f"keep drawn -> {slot(a - A_KEEP)}"
    if A_REVEAL <= a < N_ACTIONS:
        return f"discard drawn, reveal {slot(a - A_REVEAL)}"
    raise ValueError(f"action {action} out of range")
