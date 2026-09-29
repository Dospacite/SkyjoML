"""Compiled featurizer: raw observation -> (global, per-player, per-actor-slot) float features.

Everything here is a deterministic function of the raw observation, so it inherits the observation's
information boundary. Card accounting features are the exact public belief marginals: before the first
recycle, every unseen card is exchangeable between hidden grid slots and the draw pile; after a recycle the
draw pile's composition is public and the hidden-slot pool is the remainder (see docs/belief.md).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..game import obs_layout as L
from ..game.rules import MAX_PLAYERS, MIN_PLAYERS, MIN_VALUE, N_COLS, N_ROWS, N_SLOTS, N_VALUES

FEATURE_VERSION = "f2"  # bump whenever feature semantics change
N_COUNTS = MAX_PLAYERS - MIN_PLAYERS + 1  # player-count one-hot width
N_GLOBAL = 94 + N_COUNTS
N_PLAYER = 25 + MAX_PLAYERS
N_SLOT = 43


@njit(cache=True, inline="always")
def _val(code):
    return code + MIN_VALUE


@njit(cache=True)
def featurize_one(o, g, pl, sl):
    for i in range(N_GLOBAL):
        g[i] = 0.0
    for r in range(MAX_PLAYERS):
        for i in range(N_PLAYER):
            pl[r, i] = 0.0
    for s in range(N_SLOTS):
        for i in range(N_SLOT):
            sl[s, i] = 0.0
    n = o[L.O_NPLAYERS]
    phase = o[L.O_PHASE]
    pend = o[L.O_PENDING]
    top = o[L.O_TOP]
    draw_n = o[L.O_DRAW_N]
    known = o[L.O_DRAW_KNOWN]
    fin = o[L.O_FINISHER]
    final_left = o[L.O_FINAL_LEFT]

    # ---- card distributions
    pool_n = 0.0
    for i in range(N_VALUES):
        pool_n += max(o[L.O_POOL + i], 0)
    ph = np.zeros(N_VALUES)
    eh = 0.0
    eh2 = 0.0
    for i in range(N_VALUES):
        ph[i] = max(o[L.O_POOL + i], 0) / max(pool_n, 1.0)
        eh += ph[i] * _val(i)
        eh2 += ph[i] * _val(i) * _val(i)
    if pool_n == 0:
        eh = 5.0
    sdh = np.sqrt(max(eh2 - eh * eh, 0.0))
    pd = np.zeros(N_VALUES)
    tot = 0.0
    if draw_n == 0:
        for i in range(N_VALUES):
            c = o[L.O_DISCC + i] - (1 if i == top else 0)
            pd[i] = max(c, 0)
            tot += pd[i]
    elif known == 1:
        for i in range(N_VALUES):
            pd[i] = max(o[L.O_DRAWK + i], 0)
            tot += pd[i]
    else:
        for i in range(N_VALUES):
            pd[i] = ph[i]
        tot = 1.0
    ed = 0.0
    for i in range(N_VALUES):
        pd[i] = pd[i] / max(tot, 1e-9)
        ed += pd[i] * _val(i)
    if tot == 0:
        ed = eh

    # ---- per-player summaries
    est_sum = np.zeros(MAX_PLAYERS)
    vis_sum = np.zeros(MAX_PLAYERS)
    n_hid = np.zeros(MAX_PLAYERS)
    n_pres = np.zeros(MAX_PLAYERS)
    max_vis = np.full(MAX_PLAYERS, -2.0)
    n_pairs = np.zeros(MAX_PLAYERS)
    pair_sum = np.zeros(MAX_PLAYERS)
    top_completes = np.zeros(MAX_PLAYERS)
    n_high = np.zeros(MAX_PLAYERS)
    n_neg = np.zeros(MAX_PLAYERS)
    for r in range(n):
        base = L.O_GRID + r * N_SLOTS
        for c in range(N_COLS):
            cb = base + c * N_ROWS
            for k in range(N_ROWS):
                x = o[cb + k]
                if x < N_VALUES:
                    v = _val(x)
                    vis_sum[r] += v
                    est_sum[r] += v
                    n_pres[r] += 1
                    if v > max_vis[r]:
                        max_vis[r] = v
                    if v >= 9:
                        n_high[r] += 1
                    if v < 0:
                        n_neg[r] += 1
                elif x == L.CODE_HIDDEN:
                    est_sum[r] += eh
                    n_hid[r] += 1
                    n_pres[r] += 1
            a, b, d = o[cb], o[cb + 1], o[cb + 2]
            pv = -1
            if a < N_VALUES and a == b:
                pv = a
            elif a < N_VALUES and a == d:
                pv = a
            elif b < N_VALUES and b == d:
                pv = b
            if pv >= 0:
                n_pairs[r] += 1
                pair_sum[r] += _val(pv)
                if top == pv:
                    top_completes[r] = 1.0
    my_total = o[L.O_SCORES]
    min_opp = 1e9
    max_all = -1e9
    min_opp_est_round = 1e9
    min_opp_est_final = 1e9
    n_below = 0
    for r in range(n):
        t = o[L.O_SCORES + r]
        if t > max_all:
            max_all = t
        if r > 0:
            if t < min_opp:
                min_opp = t
            if t < my_total:
                n_below += 1
            if est_sum[r] < min_opp_est_round:
                min_opp_est_round = est_sum[r]
            if t + est_sum[r] < min_opp_est_final:
                min_opp_est_final = t + est_sum[r]
    max_est_final = -1e9
    for r in range(n):
        if o[L.O_SCORES + r] + est_sum[r] > max_est_final:
            max_est_final = o[L.O_SCORES + r] + est_sum[r]

    for r in range(n):
        t = o[L.O_SCORES + r]
        k = 0
        pl[r, k] = 1.0; k += 1  # noqa: E702
        pl[r, k + r] = 1.0
        k += MAX_PLAYERS
        pl[r, k] = t / 100.0; k += 1  # noqa: E702
        pl[r, k] = (t - my_total) / 50.0; k += 1  # noqa: E702
        pl[r, k] = max(100.0 - t, 0.0) / 100.0; k += 1  # noqa: E702
        pl[r, k] = vis_sum[r] / 40.0; k += 1  # noqa: E702
        pl[r, k] = est_sum[r] / 40.0; k += 1  # noqa: E702
        pl[r, k] = (t + est_sum[r]) / 100.0; k += 1  # noqa: E702
        pl[r, k] = n_hid[r] / 12.0; k += 1  # noqa: E702
        pl[r, k] = n_pres[r] / 12.0; k += 1  # noqa: E702
        pl[r, k] = 1.0 if n_hid[r] == 1 else 0.0; k += 1  # noqa: E702
        pl[r, k] = 1.0 if n_hid[r] <= 2 else 0.0; k += 1  # noqa: E702
        pl[r, k] = max_vis[r] / 12.0; k += 1  # noqa: E702
        pl[r, k] = n_pairs[r] / 4.0; k += 1  # noqa: E702
        pl[r, k] = pair_sum[r] / 24.0; k += 1  # noqa: E702
        pl[r, k] = n_high[r] / 12.0; k += 1  # noqa: E702
        pl[r, k] = n_neg[r] / 12.0; k += 1  # noqa: E702
        pl[r, k] = 1.0 if fin == r else 0.0; k += 1  # noqa: E702
        acts_again = 1.0 if fin < 0 or r < final_left else 0.0
        pl[r, k] = acts_again; k += 1  # noqa: E702
        pl[r, k] = top_completes[r]; k += 1  # noqa: E702
        pl[r, k] = (max_vis[r] - (_val(top) if top >= 0 else 5)) / 12.0; k += 1  # noqa: E702
        pl[r, k] = (est_sum[r] - est_sum[0]) / 40.0; k += 1  # noqa: E702
        pl[r, k] = r / max(n - 1, 1); k += 1  # noqa: E702
        pl[r, k] = 1.0 if (n_hid[r] == 0) else 0.0; k += 1  # noqa: E702
        pl[r, k] = ((t + est_sum[r]) - (my_total + est_sum[0])) / 50.0; k += 1  # noqa: E702
        pl[r, k] = 1.0 if t + est_sum[r] >= 100 else 0.0; k += 1  # noqa: E702

    # ---- global
    k = 0
    if 0 <= phase <= 2:
        g[k + phase] = 1.0
    k += 3
    g[k + pend + 1] = 1.0
    k += N_VALUES + 1
    g[k + top + 1] = 1.0
    k += N_VALUES + 1
    g[k] = _val(pend) / 12.0 if pend >= 0 else 0.0; k += 1  # noqa: E702
    g[k] = _val(top) / 12.0 if top >= 0 else 0.0; k += 1  # noqa: E702
    g[k] = draw_n / 150.0; k += 1  # noqa: E702
    g[k] = o[L.O_DISC_N] / 150.0; k += 1  # noqa: E702
    g[k] = known; k += 1  # noqa: E702
    g[k] = 1.0 if draw_n == 0 else 0.0; k += 1  # noqa: E702
    for i in range(N_VALUES):
        g[k + i] = ph[i]
    k += N_VALUES
    g[k] = pool_n / 150.0; k += 1  # noqa: E702
    for i in range(N_VALUES):
        g[k + i] = pd[i]
    k += N_VALUES
    g[k] = eh / 12.0; k += 1  # noqa: E702
    g[k] = ed / 12.0; k += 1  # noqa: E702
    g[k] = sdh / 6.0; k += 1  # noqa: E702
    g[k + n - MIN_PLAYERS] = 1.0
    k += N_COUNTS
    g[k] = o[L.O_ROUND] / 10.0; k += 1  # noqa: E702
    g[k] = o[L.O_ROUND_TURNS] / 100.0; k += 1  # noqa: E702
    g[k] = 1.0 if fin >= 0 else 0.0; k += 1  # noqa: E702
    g[k] = final_left / 8.0; k += 1  # noqa: E702
    g[k] = my_total / 100.0; k += 1  # noqa: E702
    g[k] = min_opp / 100.0; k += 1  # noqa: E702
    g[k] = (my_total - min_opp) / 50.0; k += 1  # noqa: E702
    g[k] = (100.0 - max_all) / 100.0; k += 1  # noqa: E702
    g[k] = n_below / max(n - 1, 1); k += 1  # noqa: E702
    g[k] = (est_sum[0] - min_opp_est_round) / 40.0; k += 1  # noqa: E702
    g[k] = (my_total + est_sum[0] - min_opp_est_final) / 50.0; k += 1  # noqa: E702
    g[k] = (100.0 - max_est_final) / 100.0; k += 1  # noqa: E702
    g[k] = 1.0 if max_est_final >= 100 else 0.0; k += 1  # noqa: E702
    # probability a fresh draw beats the top discard / is low
    pl_low = 0.0
    p_le_top = 0.0
    for i in range(N_VALUES):
        if _val(i) <= 0:
            pl_low += pd[i]
        if top >= 0 and i <= top:
            p_le_top += pd[i]
    g[k] = pl_low; k += 1  # noqa: E702
    g[k] = p_le_top; k += 1  # noqa: E702
    g[k] = min(n_hid[0], 12.0) / 12.0; k += 1  # noqa: E702
    g[k] = 1.0 if n_hid[0] == 1 else 0.0; k += 1  # noqa: E702
    tot_hidden = 0.0
    for r in range(n):
        tot_hidden += n_hid[r]
    g[k] = tot_hidden / (12.0 * n); k += 1  # noqa: E702
    g[k] = 1.0 if (fin >= 0 and final_left == 1) else 0.0; k += 1  # noqa: E702

    # ---- actor's slots
    base = L.O_GRID
    for s in range(N_SLOTS):
        x = o[base + s]
        c = s // N_ROWS
        if x == L.CODE_ABSENT:
            continue
        m1 = -1
        m2 = -1
        for q in range(N_ROWS):
            t = c * N_ROWS + q
            if t != s:
                if m1 < 0:
                    m1 = t
                else:
                    m2 = t
        x1 = o[base + m1]
        x2 = o[base + m2]
        vis = x < N_VALUES
        est = _val(x) if vis else eh
        e1 = _val(x1) if x1 < N_VALUES else eh
        e2 = _val(x2) if x2 < N_VALUES else eh
        k = 0
        sl[s, k] = 1.0; k += 1  # noqa: E702
        sl[s, k] = 0.0 if vis else 1.0; k += 1  # noqa: E702
        if vis:
            sl[s, k + x] = 1.0
        k += N_VALUES
        sl[s, k] = est / 12.0; k += 1  # noqa: E702
        sl[s, k] = min(e1, e2) / 12.0; k += 1  # noqa: E702
        sl[s, k] = max(e1, e2) / 12.0; k += 1  # noqa: E702
        mh = (1 if x1 == L.CODE_HIDDEN else 0) + (1 if x2 == L.CODE_HIDDEN else 0)
        sl[s, k] = mh / 2.0; k += 1  # noqa: E702
        mates_pair = x1 < N_VALUES and x1 == x2
        sl[s, k] = 1.0 if mates_pair else 0.0; k += 1  # noqa: E702
        sl[s, k] = _val(x1) / 12.0 if mates_pair else 0.0; k += 1  # noqa: E702
        eq_self = 0
        if vis:
            if x1 == x:
                eq_self += 1
            if x2 == x:
                eq_self += 1
        sl[s, k] = eq_self / 2.0; k += 1  # noqa: E702
        sl[s, k] = (est + e1 + e2) / 20.0; k += 1  # noqa: E702
        sl[s, k] = 1.0 if (not vis and n_hid[0] == 1) else 0.0; k += 1  # noqa: E702
        # candidate cards: top discard (select phase) and pending draw (resolve phase)
        for cand in (top, pend):
            if cand < 0:
                k += 8
                continue
            cv = _val(cand)
            eq = (1 if x1 == cand else 0) + (1 if x2 == cand else 0)
            clears = eq == 2
            sl[s, k] = (est - cv) / 12.0; k += 1  # noqa: E702
            sl[s, k] = eq / 2.0; k += 1  # noqa: E702
            sl[s, k] = 1.0 if clears else 0.0; k += 1  # noqa: E702
            sl[s, k] = (est + 2 * cv) / 20.0 if clears else 0.0; k += 1  # noqa: E702
            creates_pair = eq == 1 and (x1 == L.CODE_HIDDEN or x2 == L.CODE_HIDDEN)
            sl[s, k] = 1.0 if creates_pair else 0.0; k += 1  # noqa: E702
            breaks_pair = vis and eq_self >= 1 and x != cand
            sl[s, k] = 1.0 if breaks_pair else 0.0; k += 1  # noqa: E702
            # placing the card here ends my round (last hidden card replaced)
            sl[s, k] = 1.0 if (not vis and n_hid[0] == 1) else 0.0; k += 1  # noqa: E702
            gain = est - cv + ((est + 2 * cv) if clears else 0.0)
            sl[s, k] = gain / 20.0; k += 1  # noqa: E702
        sl[s, k] = 1.0 if vis else 0.0; k += 1  # noqa: E702


@njit(cache=True, parallel=True)
def featurize_batch(obs, g, pl, sl):
    for i in prange(obs.shape[0]):
        featurize_one(obs[i], g[i], pl[i], sl[i])


def featurize(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B = obs.shape[0]
    g = np.empty((B, N_GLOBAL), dtype=np.float32)
    pl = np.empty((B, MAX_PLAYERS, N_PLAYER), dtype=np.float32)
    sl = np.empty((B, N_SLOTS, N_SLOT), dtype=np.float32)
    featurize_batch(np.ascontiguousarray(obs, dtype=np.int32), g, pl, sl)
    return g, pl, sl
