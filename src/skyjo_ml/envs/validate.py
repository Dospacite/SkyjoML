"""Randomized invariant gate over the batch engine (PLAN.md section 8).

Plays many complete two-player matches with a mix of random, stalling and heuristic policies
and checks after every step: 150-card conservation by value, non-negative public card pools, legal-action
availability, and score consistency at match end.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from ..agents.heuristics import PRESETS, heuristic_action
from ..game import obs_layout as L
from ..game.actions import A_DRAW, A_KEEP, A_TAKE, N_ACTIONS
from ..game.rng import bounded
from ..game.rules import INVENTORY_COUNTS, MAX_PLAYERS, MIN_PLAYERS, N_VALUES
from .batch import (
    S_LAST_RS,
    S_PHASE,
    S_ROUND_ENDED,
    S_SCORES,
    S_TRUNC,
    STATE_SIZE,
    census_one,
    legal_one,
    make_rng_states,
    observe_one,
    reset_one,
    step_one,
)

_INV = INVENTORY_COUNTS.astype(np.int64)


@njit(cache=True)
def _play(st, rng, prng, n, mode, params, max_steps, errors):
    """Play one match. mode: 0 random, 1 stall-biased random, 2 heuristic. Returns decisions taken."""
    reset_one(st, rng, n)
    o = np.zeros(L.OBS_SIZE, np.int32)
    m = np.zeros(N_ACTIONS, np.bool_)
    c = np.zeros(N_VALUES, np.int64)
    rs_sum = np.zeros(MAX_PLAYERS, np.int64)
    steps = 0
    while st[S_PHASE] != L.PH_OVER and steps < max_steps:
        census_one(st, c)
        for i in range(N_VALUES):
            if c[i] != _INV[i]:
                errors[0] += 1
        observe_one(st, o)
        for i in range(N_VALUES):
            if o[L.O_POOL + i] < 0 or o[L.O_DRAWK + i] < 0:
                errors[1] += 1
        cnt = legal_one(st, m)
        if cnt == 0:
            errors[2] += 1
            return steps
        if mode == 2:
            a = heuristic_action(o, m, 1, params, prng)
        else:
            a = -1
            if mode == 1 and bounded(prng, 100) < 95:
                # prefer moves that reveal nothing to create long rounds and deck recycles
                for k in range(N_ACTIONS):
                    if m[k] and (k == A_DRAW or (A_TAKE <= k < A_KEEP + 12 and o[L.O_GRID + (k - A_TAKE) % 12] < N_VALUES)):
                        if bounded(prng, 3) == 0:
                            a = k
                            break
            if a < 0:
                j = bounded(prng, cnt)
                for k in range(N_ACTIONS):
                    if m[k]:
                        if j == 0:
                            a = k
                            break
                        j -= 1
        if step_one(st, rng, a) != 0:
            errors[3] += 1
            return steps
        steps += 1
        if st[S_ROUND_ENDED] == 1:
            for p in range(n):
                rs_sum[p] += st[S_LAST_RS + p]
    if st[S_PHASE] == L.PH_OVER and st[S_TRUNC] == 0:
        mx = -100000
        for p in range(n):
            if rs_sum[p] != st[S_SCORES + p]:
                errors[4] += 1
            if st[S_SCORES + p] > mx:
                mx = st[S_SCORES + p]
        if mx < 100:
            errors[4] += 1
    return steps


@njit(cache=True, parallel=True)
def _run(states, rngs, prngs, ns, modes, params, max_steps, errors, steps, truncs):
    for i in prange(states.shape[0]):
        steps[i] = _play(states[i], rngs[i], prngs[i], ns[i], modes[i], params, max_steps, errors[i])
        truncs[i] = states[i, S_TRUNC] == 1 or states[i, S_PHASE] != L.PH_OVER


def validate(n_matches: int = 100_000, seed: int = 0, chunk: int = 5000, max_steps: int = 20_000) -> dict:
    t0 = time.time()
    tot_err = np.zeros(5, np.int64)
    tot_steps = 0
    tot_trunc = 0
    by_mode = np.zeros(3, np.int64)
    trunc_by_mode = np.zeros(3, np.int64)
    rng = np.random.default_rng(seed)
    params = PRESETS["finish"][1].astype(np.float64)
    done = 0
    while done < n_matches:
        k = min(chunk, n_matches - done)
        states = np.zeros((k, STATE_SIZE), np.int32)
        seeds = np.arange(seed * 10**9 + done, seed * 10**9 + done + k)
        rngs = make_rng_states(seeds)
        prngs = make_rng_states(seeds + 777_777_777)
        ns = rng.integers(MIN_PLAYERS, MAX_PLAYERS + 1, size=k).astype(np.int64)
        modes = rng.choice(3, size=k, p=[0.45, 0.1, 0.45]).astype(np.int64)
        errors = np.zeros((k, 5), np.int64)
        steps = np.zeros(k, np.int64)
        truncs = np.zeros(k, np.bool_)
        _run(states, rngs, prngs, ns, modes, params, max_steps, errors, steps, truncs)
        tot_err += errors.sum(0)
        tot_steps += int(steps.sum())
        tot_trunc += int(truncs.sum())
        by_mode += np.bincount(modes, minlength=3)
        trunc_by_mode += np.bincount(modes[truncs], minlength=3)
        done += k
    return {
        "matches": n_matches,
        "decisions": tot_steps,
        "errors": {
            "census": int(tot_err[0]),
            "negative_public_pool": int(tot_err[1]),
            "no_legal_action": int(tot_err[2]),
            "illegal_step": int(tot_err[3]),
            "score_inconsistency": int(tot_err[4]),
        },
        "truncated_or_capped": tot_trunc,
        "truncated_by_policy": {"random": int(trunc_by_mode[0]), "stall": int(trunc_by_mode[1]),
                                "heuristic": int(trunc_by_mode[2])},
        "policy_mix": {"random": int(by_mode[0]), "stall": int(by_mode[1]), "heuristic": int(by_mode[2])},
        "seconds": round(time.time() - t0, 1),
        "ok": int(tot_err.sum()) == 0,
    }
