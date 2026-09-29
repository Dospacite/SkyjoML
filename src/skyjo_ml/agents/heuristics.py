"""Heuristic Skyjo agents operating only on the raw observation vector.

All non-random heuristics share one-step expected-value lookahead over the actor's own board: an action is
scored by the expected value of a board evaluation after the action, with chance (drawn card, revealed card)
integrated exactly over the publicly known card distribution. Variants differ in the evaluation terms:

* ``greedy``  - expected round score only.
* ``column``  - adds a bonus for visible pairs that can complete a column.
* ``finish``  - adds finisher-doubling risk, end-of-round timing and match-end (100 point) awareness.

Parameters are a float array (see ``PARAM_NAMES``); defaults were tuned on development seeds only.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..game import obs_layout as L
from ..game.actions import A_DRAW, A_KEEP, A_REVEAL, A_SETUP, A_TAKE, N_ACTIONS, N_SETUP, SETUP_SAME_COLUMN
from ..game.rng import bounded, next_u64
from ..game.rules import MATCH_END_SCORE, MIN_VALUE, N_COLS, N_ROWS, N_SLOTS, N_VALUES

PARAM_NAMES = (
    "w_pair",  # bonus per point of a visible, completable pair value (v > 0)
    "w_hidden",  # value per remaining hidden card (negative favours revealing / progress)
    "use_finish",  # 1 enables finishing and match-end terms
    "opp_improve",  # expected improvement of each opponent on their last turn
    "spread",  # uncertainty scale (points) for comparing final scores
    "finish_bonus",  # value per point of expected lead when ending the round
    "match_weight",  # value of winning (+) / losing (-) the match when the round end decides it
    "setup_same_col",  # 1: reveal two cards of the same column at setup
    "noise",  # tie-break noise scale
    "take_bias",  # bonus for taking the known discard over drawing
)
N_PARAMS = len(PARAM_NAMES)

KIND_RANDOM = 0
KIND_EV = 1

PRESETS = {
    "random": (KIND_RANDOM, np.zeros(N_PARAMS)),
    "greedy": (KIND_EV, np.array([0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1e-3, 0.0])),
    "column": (KIND_EV, np.array([0.5, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1e-3, 0.0])),
    "finish": (KIND_EV, np.array([0.5, -0.3, 1.0, 2.0, 6.0, 0.3, 40.0, 0.0, 1e-3, 0.0])),
}


@njit(cache=True)
def _uniform(rng):
    return (next_u64(rng) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _dist(counts, out):
    t = 0.0
    for i in range(N_VALUES):
        t += max(counts[i], 0)
    if t <= 0:
        for i in range(N_VALUES):
            out[i] = 0.0
        out[5 - MIN_VALUE] = 1.0  # degenerate fallback, never expected in legal play
        return
    for i in range(N_VALUES):
        out[i] = max(counts[i], 0) / t


@njit(cache=True)
def _mean(p):
    m = 0.0
    for i in range(N_VALUES):
        m += p[i] * (i + MIN_VALUE)
    return m


@njit(cache=True)
def _board_value(codes, eh, params, ctx, in_final):
    """Evaluate the actor's board ``codes`` (12 slot codes, may contain values/hidden/absent).

    Applies column clears for three equal visible cards first. ``ctx`` holds match context:
    [0] my total, [1] min opponent expected final round score, [2] max opponent total,
    [3] min opponent total, [4] min over opponents of (total + expected final round score).
    """
    total = 0.0
    hidden = 0
    pair_bonus = 0.0
    for c in range(N_COLS):
        a = codes[c * N_ROWS]
        b = codes[c * N_ROWS + 1]
        d = codes[c * N_ROWS + 2]
        if a == L.CODE_ABSENT:
            continue
        if a < N_VALUES and a == b and b == d:
            continue  # cleared
        for r in range(N_ROWS):
            x = codes[c * N_ROWS + r]
            if x < N_VALUES:
                total += x + MIN_VALUE
            else:
                total += eh
                hidden += 1
        if not in_final:
            # visible pair with a completable third slot
            pv = -1
            if a < N_VALUES and a == b:
                pv = a
            elif a < N_VALUES and a == d:
                pv = a
            elif b < N_VALUES and b == d:
                pv = b
            if pv >= 0:
                v = pv + MIN_VALUE
                if v > 0:
                    pair_bonus += params[0] * v
    val = -total + pair_bonus
    if in_final:
        return val
    val += params[1] * hidden
    if params[2] > 0 and hidden == 0:
        # this action ends my round: others get one more turn, then scoring
        s = total
        opp = ctx[1] - params[3]
        p_not_lowest = 0.5 + (s - opp) / (2.0 * params[4])
        if p_not_lowest < 0.0:
            p_not_lowest = 0.0
        if p_not_lowest > 1.0:
            p_not_lowest = 1.0
        if s > 0:
            val -= s * p_not_lowest
            s_eff = s * (1.0 + p_not_lowest)
        else:
            s_eff = s
        val += params[5] * (opp - s)
        my_final = ctx[0] + s_eff
        opp_final = ctx[4] - params[3]
        if my_final >= MATCH_END_SCORE or ctx[2] + ctx[1] >= MATCH_END_SCORE:
            if my_final < opp_final:
                val += params[6]
            else:
                val -= params[6]
    return val


@njit(cache=True)
def _ev_action(o, m, params, rng):
    n = o[L.O_NPLAYERS]
    ph = o[L.O_PHASE]
    if ph == L.PH_SETUP:
        best_a = -1
        best_v = -1e18
        for k in range(N_SETUP):
            want = params[7] > 0.5
            v = (1.0 if SETUP_SAME_COLUMN[k] == want else 0.0) + _uniform(rng) * 0.5
            if v > best_v:
                best_v = v
                best_a = A_SETUP + k
        return best_a

    pool_p = np.empty(N_VALUES)
    _dist(o[L.O_POOL : L.O_POOL + N_VALUES], pool_p)
    eh = _mean(pool_p)
    draw_p = np.empty(N_VALUES)
    if o[L.O_DRAW_N] == 0:
        tmp = o[L.O_DISCC : L.O_DISCC + N_VALUES].copy()
        tmp[o[L.O_TOP]] -= 1
        _dist(tmp, draw_p)
    elif o[L.O_DRAW_KNOWN] == 1:
        _dist(o[L.O_DRAWK : L.O_DRAWK + N_VALUES], draw_p)
    else:
        draw_p[:] = pool_p
    in_final = o[L.O_FINISHER] >= 0

    # match context
    ctx = np.zeros(5)
    ctx[0] = o[L.O_SCORES]
    min_opp_round = 1e9
    max_opp_total = -1e9
    min_opp_total = 1e9
    min_opp_final = 1e9
    for r in range(1, n):
        s = 0.0
        for q in range(N_SLOTS):
            x = o[L.O_GRID + r * N_SLOTS + q]
            if x < N_VALUES:
                s += x + MIN_VALUE
            elif x == L.CODE_HIDDEN:
                s += eh
        if s < min_opp_round:
            min_opp_round = s
        t = o[L.O_SCORES + r]
        if t > max_opp_total:
            max_opp_total = t
        if t < min_opp_total:
            min_opp_total = t
        if t + s < min_opp_final:
            min_opp_final = t + s
    ctx[1] = min_opp_round
    ctx[2] = max_opp_total
    ctx[3] = min_opp_total
    ctx[4] = min_opp_final

    board = o[L.O_GRID : L.O_GRID + N_SLOTS].copy()
    noise = params[8]

    if ph == L.PH_RESOLVE:
        x = o[L.O_PENDING]
        best_a = -1
        best_v = -1e18
        for a in range(A_KEEP, N_ACTIONS):
            if not m[a]:
                continue
            v = _resolve_value(board, a, x, pool_p, eh, params, ctx, in_final) + noise * _uniform(rng)
            if v > best_v:
                best_v = v
                best_a = a
        return best_a

    # select phase
    best_a = -1
    best_v = -1e18
    top = o[L.O_TOP]
    for a in range(A_TAKE, A_KEEP):
        if not m[a]:
            continue
        s = a - A_TAKE
        old = board[s]
        board[s] = top
        v = _board_value(board, eh, params, ctx, in_final) + params[9] + noise * _uniform(rng)
        board[s] = old
        if v > best_v:
            best_v = v
            best_a = a
    if m[A_DRAW]:
        # discarding the drawn card and revealing does not depend on the drawn value: evaluate once
        best_reveal = -1e18
        for s in range(N_SLOTS):
            if board[s] == L.CODE_HIDDEN:
                v = _resolve_value(board, A_REVEAL + s, 0, pool_p, eh, params, ctx, in_final)
                if v > best_reveal:
                    best_reveal = v
        ev = 0.0
        for x in range(N_VALUES):
            if draw_p[x] <= 0:
                continue
            bx = best_reveal
            for s in range(N_SLOTS):
                if board[s] == L.CODE_ABSENT:
                    continue
                v = _resolve_value(board, A_KEEP + s, x, pool_p, eh, params, ctx, in_final)
                if v > bx:
                    bx = v
            ev += draw_p[x] * bx
        ev += noise * _uniform(rng)
        if ev > best_v:
            best_v = ev
            best_a = A_DRAW
    return best_a


@njit(cache=True)
def _resolve_value(board, a, x, pool_p, eh, params, ctx, in_final):
    if a < A_REVEAL:
        s = a - A_KEEP
        old = board[s]
        board[s] = x
        v = _board_value(board, eh, params, ctx, in_final)
        board[s] = old
        return v
    s = a - A_REVEAL
    old = board[s]
    v = 0.0
    for y in range(N_VALUES):
        if pool_p[y] <= 0:
            continue
        board[s] = y
        v += pool_p[y] * _board_value(board, eh, params, ctx, in_final)
    board[s] = old
    return v


@njit(cache=True)
def heuristic_action(o, m, kind, params, rng):
    if kind == KIND_RANDOM:
        cnt = 0
        for a in range(N_ACTIONS):
            if m[a]:
                cnt += 1
        k = bounded(rng, cnt)
        for a in range(N_ACTIONS):
            if m[a]:
                if k == 0:
                    return a
                k -= 1
        return -1
    return _ev_action(o, m, params, rng)


@njit(cache=True, parallel=True)
def heuristic_batch(obs, masks, sel, kinds, params, rngs, out):
    """Compute actions for rows ``sel`` of (obs, masks); per-row kind/params; writes into out[sel]."""
    for k in prange(len(sel)):
        i = sel[k]
        out[i] = heuristic_action(obs[i], masks[i], kinds[k], params[k], rngs[i])


class HeuristicAgent:
    """Stateless heuristic agent usable one observation at a time or in batches."""

    def __init__(self, name: str = "finish", params: np.ndarray | None = None, seed: int = 0):
        kind, default = PRESETS[name]
        self.name = name
        self.kind = kind
        self.params = (default if params is None else np.asarray(params, dtype=np.float64)).copy()
        self.rng = np.array([np.uint64(seed * 2654435761 + 12345)], dtype=np.uint64)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(heuristic_action(obs, mask, self.kind, self.params, self.rng))
