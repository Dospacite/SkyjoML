"""Numba-compiled batch engine: many independent matches stored as rows of one int32 array.

Implements exactly the same rules and chance-consumption order as ``skyjo_ml.game.engine`` (verified by
differential tests). Each row is a complete, copyable match state, which makes cloning for search trivial.
Agents only ever see the output of ``observe``; the raw state is private.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..game import obs_layout as L
from ..game.actions import A_DRAW, A_KEEP, A_REVEAL, A_SETUP, A_TAKE, N_ACTIONS, N_SETUP, SETUP_PAIRS
from ..game.rng import bounded, next_u64, shuffle_prefix
from ..game.rules import (
    DECK_SIZE,
    INVENTORY_COUNTS,
    MATCH_END_SCORE,
    MAX_MATCH_DECISIONS,
    MAX_PLAYERS,
    MAX_ROUND_DECISIONS,
    MIN_VALUE,
    N_COLS,
    N_ROWS,
    N_SLOTS,
    N_VALUES,
)

HIDDEN, VISIBLE, ABSENT = 0, 1, 2
NONE = -99

# ---- state layout (int32 per match)
S_NP = 0
S_PHASE = 1
S_CUR = 2
S_PENDING = 3
S_FINISHER = 4
S_FINAL_LEFT = 5
S_ROUND = 6
S_LEADER = 7
S_SETUP_COUNT = 8
S_DRAW_N = 9
S_DISC_N = 10
S_DRAW_KNOWN = 11
S_MATCH_DEC = 12
S_ROUND_TURNS = 13
S_TRUNC = 14
S_ROUND_ENDED = 15  # set by the last step if it ended a round
S_LAST_FIN = 16  # finisher of the most recently scored round
S_LAST_DBL = 17  # 1 if that finisher's score was doubled
S_SCORES = 18
S_LAST_RS = S_SCORES + MAX_PLAYERS
S_VAL = S_LAST_RS + MAX_PLAYERS
S_VIS = S_VAL + MAX_PLAYERS * N_SLOTS
S_PICK = S_VIS + MAX_PLAYERS * N_SLOTS
S_DRAW = S_PICK + MAX_PLAYERS
S_DISC = S_DRAW + DECK_SIZE
S_DRAWK = S_DISC + DECK_SIZE
STATE_SIZE = S_DRAWK + N_VALUES

_INV = INVENTORY_COUNTS.astype(np.int32)
_PAIRS = SETUP_PAIRS.astype(np.int32)


@njit(cache=True)
def _start_round(st, rng, leader):
    n = st[S_NP]
    # canonical deck, shuffled
    k = 0
    for i in range(N_VALUES):
        for _ in range(_INV[i]):
            st[S_DRAW + k] = i + MIN_VALUE
            k += 1
    shuffle_prefix(rng, st[S_DRAW : S_DRAW + DECK_SIZE], DECK_SIZE)
    dn = DECK_SIZE
    for p in range(MAX_PLAYERS):
        for s in range(N_SLOTS):
            if p < n:
                dn -= 1
                st[S_VAL + p * N_SLOTS + s] = st[S_DRAW + dn]
                st[S_VIS + p * N_SLOTS + s] = HIDDEN
            else:
                st[S_VAL + p * N_SLOTS + s] = 0
                st[S_VIS + p * N_SLOTS + s] = ABSENT
        st[S_PICK + p] = -1
    dn -= 1
    st[S_DISC] = st[S_DRAW + dn]
    st[S_DISC_N] = 1
    st[S_DRAW_N] = dn
    st[S_PENDING] = NONE
    st[S_PHASE] = L.PH_SETUP
    st[S_LEADER] = leader
    st[S_CUR] = leader
    st[S_SETUP_COUNT] = 0
    st[S_FINISHER] = -1
    st[S_FINAL_LEFT] = 0
    st[S_DRAW_KNOWN] = 0
    for i in range(N_VALUES):
        st[S_DRAWK + i] = 0
    st[S_ROUND_TURNS] = 0


@njit(cache=True)
def reset_one(st, rng, n_players):
    st[:] = 0
    st[S_NP] = n_players
    st[S_ROUND] = 0
    _start_round(st, rng, 0)


@njit(cache=True)
def _finish_setup(st, rng):
    n = st[S_NP]
    for p in range(n):
        pk = st[S_PICK + p]
        st[S_VIS + p * N_SLOTS + _PAIRS[pk, 0]] = VISIBLE
        st[S_VIS + p * N_SLOTS + _PAIRS[pk, 1]] = VISIBLE
    if st[S_ROUND] == 0:
        best = -1000
        for p in range(n):
            pk = st[S_PICK + p]
            v = st[S_VAL + p * N_SLOTS + _PAIRS[pk, 0]] + st[S_VAL + p * N_SLOTS + _PAIRS[pk, 1]]
            if v > best:
                best = v
        ntied = 0
        tied = np.empty(MAX_PLAYERS, np.int32)
        for p in range(n):
            pk = st[S_PICK + p]
            v = st[S_VAL + p * N_SLOTS + _PAIRS[pk, 0]] + st[S_VAL + p * N_SLOTS + _PAIRS[pk, 1]]
            if v == best:
                tied[ntied] = p
                ntied += 1
        if ntied > 1:
            starter = tied[bounded(rng, ntied)]
        else:
            starter = tied[0]
    else:
        starter = st[S_LEADER]
    st[S_CUR] = starter
    st[S_PHASE] = L.PH_SELECT


@njit(cache=True)
def _push_disc(st, v):
    st[S_DISC + st[S_DISC_N]] = v
    st[S_DISC_N] += 1


@njit(cache=True)
def _check_column(st, p, c):
    base = p * N_SLOTS + c * N_ROWS
    for r in range(N_ROWS):
        if st[S_VIS + base + r] != VISIBLE:
            return False
    v0 = st[S_VAL + base]
    for r in range(1, N_ROWS):
        if st[S_VAL + base + r] != v0:
            return False
    for r in range(N_ROWS):
        _push_disc(st, st[S_VAL + base + r])
        st[S_VIS + base + r] = ABSENT
    return True


@njit(cache=True)
def _hidden_count(st, p):
    h = 0
    for s in range(N_SLOTS):
        if st[S_VIS + p * N_SLOTS + s] == HIDDEN:
            h += 1
    return h


@njit(cache=True)
def _score_round(st, rng):
    n = st[S_NP]
    f = st[S_FINISHER]
    for k in range(n):
        p = (f + k) % n
        for s in range(N_SLOTS):
            if st[S_VIS + p * N_SLOTS + s] == HIDDEN:
                st[S_VIS + p * N_SLOTS + s] = VISIBLE
        for c in range(N_COLS):
            _check_column(st, p, c)
    for p in range(MAX_PLAYERS):
        st[S_LAST_RS + p] = 0
    for p in range(n):
        t = 0
        for s in range(N_SLOTS):
            if st[S_VIS + p * N_SLOTS + s] == VISIBLE:
                t += st[S_VAL + p * N_SLOTS + s]
        st[S_LAST_RS + p] = t
    rf = st[S_LAST_RS + f]
    st[S_LAST_DBL] = 0
    if rf > 0:
        dbl = False
        for q in range(n):
            if q != f and st[S_LAST_RS + q] <= rf:
                dbl = True
        if dbl:
            st[S_LAST_RS + f] = 2 * rf
    st[S_LAST_FIN] = f
    st[S_LAST_DBL] = 1 if st[S_LAST_RS + f] != rf else 0
    mx = -100000
    for p in range(n):
        st[S_SCORES + p] += st[S_LAST_RS + p]
        if st[S_SCORES + p] > mx:
            mx = st[S_SCORES + p]
    st[S_ROUND_ENDED] = 1
    if mx >= MATCH_END_SCORE:
        st[S_PHASE] = L.PH_OVER
    else:
        st[S_ROUND] += 1
        _start_round(st, rng, f)


@njit(cache=True)
def _end_turn(st, rng):
    n = st[S_NP]
    cur = st[S_CUR]
    if st[S_FINISHER] < 0:
        if _hidden_count(st, cur) == 0:
            st[S_FINISHER] = cur
            st[S_FINAL_LEFT] = n - 1
    else:
        st[S_FINAL_LEFT] -= 1
    if st[S_FINISHER] >= 0 and st[S_FINAL_LEFT] == 0:
        _score_round(st, rng)
        return
    st[S_CUR] = (cur + 1) % n
    st[S_PHASE] = L.PH_SELECT


@njit(cache=True)
def legal_one(st, m):
    """Fill boolean mask ``m`` (length N_ACTIONS) for the current actor; returns number of legal actions."""
    for i in range(N_ACTIONS):
        m[i] = False
    ph = st[S_PHASE]
    p = st[S_CUR]
    cnt = 0
    if ph == L.PH_SETUP:
        for i in range(N_SETUP):
            m[A_SETUP + i] = True
        cnt = N_SETUP
    elif ph == L.PH_SELECT:
        if st[S_DRAW_N] > 0 or st[S_DISC_N] >= 2:
            m[A_DRAW] = True
            cnt += 1
        if st[S_DISC_N] > 0:
            for s in range(N_SLOTS):
                if st[S_VIS + p * N_SLOTS + s] != ABSENT:
                    m[A_TAKE + s] = True
                    cnt += 1
    elif ph == L.PH_RESOLVE:
        for s in range(N_SLOTS):
            vs = st[S_VIS + p * N_SLOTS + s]
            if vs != ABSENT:
                m[A_KEEP + s] = True
                cnt += 1
            if vs == HIDDEN:
                m[A_REVEAL + s] = True
                cnt += 1
    return cnt


@njit(cache=True)
def step_one(st, rng, a):
    """Apply action ``a`` for the current actor. Returns 0 on success, -1 if illegal (state unchanged)."""
    m = np.zeros(N_ACTIONS, np.bool_)
    legal_one(st, m)
    if st[S_PHASE] == L.PH_OVER or a < 0 or a >= N_ACTIONS or not m[a]:
        return -1
    st[S_ROUND_ENDED] = 0
    p = st[S_CUR]
    n = st[S_NP]
    st[S_MATCH_DEC] += 1
    st[S_ROUND_TURNS] += 1
    ph = st[S_PHASE]
    if ph == L.PH_SETUP:
        st[S_PICK + p] = a - A_SETUP
        st[S_SETUP_COUNT] += 1
        st[S_CUR] = (p + 1) % n
        if st[S_SETUP_COUNT] == n:
            _finish_setup(st, rng)
    elif ph == L.PH_SELECT:
        if a == A_DRAW:
            if st[S_DRAW_N] == 0:
                dn = st[S_DISC_N]
                top = st[S_DISC + dn - 1]
                for i in range(dn - 1):
                    st[S_DRAW + i] = st[S_DISC + i]
                shuffle_prefix(rng, st[S_DRAW : S_DRAW + DECK_SIZE], dn - 1)
                st[S_DRAW_N] = dn - 1
                st[S_DISC] = top
                st[S_DISC_N] = 1
                st[S_DRAW_KNOWN] = 1
                for i in range(N_VALUES):
                    st[S_DRAWK + i] = 0
                for i in range(dn - 1):
                    st[S_DRAWK + st[S_DRAW + i] - MIN_VALUE] += 1
            st[S_DRAW_N] -= 1
            card = st[S_DRAW + st[S_DRAW_N]]
            st[S_PENDING] = card
            if st[S_DRAW_KNOWN] == 1:
                st[S_DRAWK + card - MIN_VALUE] -= 1
            st[S_PHASE] = L.PH_RESOLVE
        else:
            s = a - A_TAKE
            st[S_DISC_N] -= 1
            card = st[S_DISC + st[S_DISC_N]]
            idx = p * N_SLOTS + s
            old = st[S_VAL + idx]
            st[S_VAL + idx] = card
            st[S_VIS + idx] = VISIBLE
            _push_disc(st, old)
            _check_column(st, p, s // N_ROWS)
            _end_turn(st, rng)
    else:
        card = st[S_PENDING]
        st[S_PENDING] = NONE
        if a < A_REVEAL:
            s = a - A_KEEP
            idx = p * N_SLOTS + s
            old = st[S_VAL + idx]
            st[S_VAL + idx] = card
            st[S_VIS + idx] = VISIBLE
            _push_disc(st, old)
        else:
            s = a - A_REVEAL
            _push_disc(st, card)
            st[S_VIS + p * N_SLOTS + s] = VISIBLE
        _check_column(st, p, s // N_ROWS)
        _end_turn(st, rng)
    if st[S_PHASE] != L.PH_OVER and (
        st[S_MATCH_DEC] >= MAX_MATCH_DECISIONS or st[S_ROUND_TURNS] >= MAX_ROUND_DECISIONS
    ):
        st[S_TRUNC] = 1
        st[S_PHASE] = L.PH_OVER
    return 0


@njit(cache=True)
def observe_one(st, o):
    """Fill raw observation ``o`` (length OBS_SIZE) for the current actor."""
    for i in range(L.OBS_SIZE):
        o[i] = 0
    n = st[S_NP]
    a = st[S_CUR]
    o[L.O_NPLAYERS] = n
    o[L.O_PHASE] = st[S_PHASE]
    o[L.O_PENDING] = -1 if st[S_PENDING] == NONE else st[S_PENDING] - MIN_VALUE
    dn = st[S_DISC_N]
    o[L.O_TOP] = st[S_DISC + dn - 1] - MIN_VALUE if dn > 0 else -1
    o[L.O_DRAW_N] = st[S_DRAW_N]
    o[L.O_DISC_N] = dn
    o[L.O_DRAW_KNOWN] = st[S_DRAW_KNOWN]
    f = st[S_FINISHER]
    o[L.O_FINISHER] = -1 if f < 0 else (f - a + n) % n
    o[L.O_FINAL_LEFT] = st[S_FINAL_LEFT]
    o[L.O_ROUND] = st[S_ROUND]
    o[L.O_ROUND_TURNS] = st[S_ROUND_TURNS]
    for i in range(N_VALUES):
        o[L.O_POOL + i] = _INV[i]
    for r in range(MAX_PLAYERS):
        base = L.O_GRID + r * N_SLOTS
        if r >= n:
            for s in range(N_SLOTS):
                o[base + s] = L.CODE_NOPLAYER
            continue
        p = (a + r) % n
        o[L.O_SCORES + r] = st[S_SCORES + p]
        for s in range(N_SLOTS):
            vs = st[S_VIS + p * N_SLOTS + s]
            if vs == VISIBLE:
                c = st[S_VAL + p * N_SLOTS + s] - MIN_VALUE
                o[base + s] = c
                o[L.O_POOL + c] -= 1
            elif vs == HIDDEN:
                o[base + s] = L.CODE_HIDDEN
            else:
                o[base + s] = L.CODE_ABSENT
    for i in range(dn):
        c = st[S_DISC + i] - MIN_VALUE
        o[L.O_DISCC + c] += 1
        o[L.O_POOL + c] -= 1
    if st[S_PENDING] != NONE:
        o[L.O_POOL + st[S_PENDING] - MIN_VALUE] -= 1
    if st[S_DRAW_KNOWN] == 1:
        for i in range(N_VALUES):
            o[L.O_DRAWK + i] = st[S_DRAWK + i]
            o[L.O_POOL + i] -= st[S_DRAWK + i]


# ---------------------------------------------------------------------------- batched wrappers
@njit(cache=True, parallel=True)
def reset_batch(states, rngs, idx, n_players):
    for k in prange(len(idx)):
        i = idx[k]
        reset_one(states[i], rngs[i], n_players[k])


@njit(cache=True, parallel=True)
def observe_batch(states, obs, masks):
    for i in prange(states.shape[0]):
        observe_one(states[i], obs[i])
        legal_one(states[i], masks[i])


@njit(cache=True, parallel=True)
def step_batch(states, rngs, actions, status):
    """Step every match whose action is >= 0. ``status``: 0 ok, -1 illegal, -2 skipped."""
    for i in prange(states.shape[0]):
        if actions[i] < 0:
            status[i] = -2
        else:
            status[i] = step_one(states[i], rngs[i], actions[i])


@njit(cache=True)
def win_credit_one(st, out):
    n = st[S_NP]
    best = 1 << 30
    for p in range(n):
        if st[S_SCORES + p] < best:
            best = st[S_SCORES + p]
    k = 0
    for p in range(n):
        if st[S_SCORES + p] == best:
            k += 1
    for p in range(MAX_PLAYERS):
        out[p] = 0.0
    for p in range(n):
        if st[S_SCORES + p] == best:
            out[p] = 1.0 / k


@njit(cache=True)
def census_one(st, c):
    for i in range(N_VALUES):
        c[i] = 0
    for i in range(st[S_DRAW_N]):
        c[st[S_DRAW + i] - MIN_VALUE] += 1
    for i in range(st[S_DISC_N]):
        c[st[S_DISC + i] - MIN_VALUE] += 1
    if st[S_PENDING] != NONE:
        c[st[S_PENDING] - MIN_VALUE] += 1
    for p in range(st[S_NP]):
        for s in range(N_SLOTS):
            if st[S_VIS + p * N_SLOTS + s] != ABSENT:
                c[st[S_VAL + p * N_SLOTS + s] - MIN_VALUE] += 1


def make_rng_states(seeds: np.ndarray) -> np.ndarray:
    """One splitmix64 state per match; identical derivation to ``rng.seed_state``."""
    out = np.empty((len(seeds), 1), dtype=np.uint64)
    for k, s in enumerate(seeds):
        out[k, 0] = np.uint64(int(s) & 0xFFFFFFFFFFFFFFFF)
        next_u64(out[k])
    return out


class BatchEnv:
    """Thin Python owner of state arrays for ``B`` concurrent matches."""

    def __init__(self, batch_size: int):
        self.B = batch_size
        self.states = np.zeros((batch_size, STATE_SIZE), dtype=np.int32)
        self.rngs = np.zeros((batch_size, 1), dtype=np.uint64)
        self.obs = np.zeros((batch_size, L.OBS_SIZE), dtype=np.int32)
        self.masks = np.zeros((batch_size, N_ACTIONS), dtype=np.bool_)
        self.status = np.zeros(batch_size, dtype=np.int32)

    def reset(self, idx: np.ndarray, seeds: np.ndarray, n_players: np.ndarray) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        n_players = np.asarray(n_players, dtype=np.int32)
        if n_players.shape != idx.shape or np.any(n_players != MAX_PLAYERS):
            raise ValueError(f"n_players must contain one value of {MAX_PLAYERS} per match")
        self.rngs[idx] = make_rng_states(np.asarray(seeds))
        reset_batch(self.states, self.rngs, idx, n_players)

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        observe_batch(self.states, self.obs, self.masks)
        return self.obs, self.masks

    def step(self, actions: np.ndarray) -> np.ndarray:
        step_batch(self.states, self.rngs, np.asarray(actions, dtype=np.int32), self.status)
        if (self.status == -1).any():
            bad = np.nonzero(self.status == -1)[0]
            raise ValueError(f"illegal actions in matches {bad[:10].tolist()}")
        return self.status

    # convenience views
    @property
    def cur(self) -> np.ndarray:
        return self.states[:, S_CUR]

    @property
    def n_players(self) -> np.ndarray:
        return self.states[:, S_NP]

    @property
    def phase(self) -> np.ndarray:
        return self.states[:, S_PHASE]

    @property
    def scores(self) -> np.ndarray:
        return self.states[:, S_SCORES : S_SCORES + MAX_PLAYERS]

    @property
    def last_round_scores(self) -> np.ndarray:
        return self.states[:, S_LAST_RS : S_LAST_RS + MAX_PLAYERS]

    @property
    def round_ended(self) -> np.ndarray:
        return self.states[:, S_ROUND_ENDED] == 1

    @property
    def over(self) -> np.ndarray:
        return self.states[:, S_PHASE] == L.PH_OVER

    @property
    def truncated(self) -> np.ndarray:
        return self.states[:, S_TRUNC] == 1

    def win_credit(self) -> np.ndarray:
        out = np.zeros((self.B, MAX_PLAYERS), dtype=np.float64)
        for i in range(self.B):
            win_credit_one(self.states[i], out[i])
        return out
