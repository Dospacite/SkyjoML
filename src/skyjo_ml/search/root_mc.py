"""Belief-sampled root Monte Carlo policy improvement (PLAN.md section 7).

For a decision of the root player:

1. Sample hidden worlds consistent with the root's observation only. In Skyjo no player (including the
   owner) sees unrevealed grid cards, and every drawn card becomes public when resolved, so conditional on
   public information all unseen cards are exchangeable: hidden grid slots and the draw pile are filled by a
   uniform shuffle of the public "pool" (before any recycle), or the hidden slots get the pool and the draw
   pile gets its publicly known composition (after a recycle). This is the exact posterior; no opponent
   model is needed because opponents cannot condition on values they have not seen either.
2. Keep the policy's top-k legal root actions (prior pruning) and evaluate each on the same worlds with
   common random numbers (identical world, identical engine chance stream, deterministic rollouts).
3. Roll every simulated player forward with the neural policy acting on its *own simulated observation*.
   Stop at the root player's first decision after the horizon (or at match end) and score with the
   observation-only value head (or exact terminal win credit).
4. Sequential halving reallocates rollouts to the better half of the actions each phase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from numba import njit

from ..envs.batch import (
    ABSENT,
    HIDDEN,
    NONE,
    S_CUR,
    S_DISC,
    S_DISC_N,
    S_DRAW,
    S_DRAW_KNOWN,
    S_DRAW_N,
    S_DRAWK,
    S_FINAL_LEFT,
    S_FINISHER,
    S_LEADER,
    S_MATCH_DEC,
    S_NP,
    S_PENDING,
    S_PHASE,
    S_ROUND,
    S_ROUND_TURNS,
    S_SCORES,
    S_TRUNC,
    S_VAL,
    S_VIS,
    STATE_SIZE,
    VISIBLE,
    observe_batch,
    step_one,
    win_credit_one,
)
from ..game import obs_layout as L
from ..game.actions import N_ACTIONS
from ..game.rng import shuffle_prefix
from ..game.rules import MAX_PLAYERS, MIN_VALUE, N_SLOTS, N_VALUES


@njit(cache=True)
def build_world(o, rng, st):
    """Fill engine state ``st`` with one hidden world consistent with raw observation ``o``.

    Seats are relative to the root player (root = seat 0), which preserves clockwise order.
    """
    st[:] = 0
    n = o[L.O_NPLAYERS]
    st[S_NP] = n
    st[S_PHASE] = o[L.O_PHASE]
    st[S_CUR] = 0
    st[S_PENDING] = NONE if o[L.O_PENDING] < 0 else o[L.O_PENDING] + MIN_VALUE
    st[S_FINISHER] = o[L.O_FINISHER]
    st[S_FINAL_LEFT] = o[L.O_FINAL_LEFT]
    st[S_ROUND] = max(o[L.O_ROUND], 1)  # starter rule of round 0 never matters mid-round
    st[S_LEADER] = 0
    st[S_ROUND_TURNS] = o[L.O_ROUND_TURNS]
    st[S_MATCH_DEC] = 0
    st[S_DRAW_KNOWN] = o[L.O_DRAW_KNOWN]
    for p in range(MAX_PLAYERS):
        st[S_SCORES + p] = o[L.O_SCORES + p] if p < n else 0
    # unseen pool
    pool = np.empty(160, np.int32)
    k = 0
    for i in range(N_VALUES):
        for _ in range(max(o[L.O_POOL + i], 0)):
            pool[k] = i + MIN_VALUE
            k += 1
    shuffle_prefix(rng, pool, k)
    used = 0
    for p in range(MAX_PLAYERS):
        for s in range(N_SLOTS):
            idx = p * N_SLOTS + s
            c = o[L.O_GRID + idx]
            if p >= n or c == L.CODE_ABSENT or c == L.CODE_NOPLAYER:
                st[S_VIS + idx] = ABSENT
                st[S_VAL + idx] = 0
            elif c == L.CODE_HIDDEN:
                st[S_VIS + idx] = HIDDEN
                st[S_VAL + idx] = pool[used]
                used += 1
            else:
                st[S_VIS + idx] = VISIBLE
                st[S_VAL + idx] = c + MIN_VALUE
    dn = o[L.O_DRAW_N]
    if o[L.O_DRAW_KNOWN] == 1:
        m = 0
        for i in range(N_VALUES):
            st[S_DRAWK + i] = o[L.O_DRAWK + i]
            for _ in range(o[L.O_DRAWK + i]):
                st[S_DRAW + m] = i + MIN_VALUE
                m += 1
        shuffle_prefix(rng, st[S_DRAW : S_DRAW + 150], m)
    else:
        for j in range(dn):
            st[S_DRAW + j] = pool[used + j]
    st[S_DRAW_N] = dn
    # discard pile: public composition, top card last (order below the top is irrelevant until recycled)
    top = o[L.O_TOP]
    m = 0
    for i in range(N_VALUES):
        cnt = o[L.O_DISCC + i] - (1 if i == top else 0)
        for _ in range(cnt):
            st[S_DISC + m] = i + MIN_VALUE
            m += 1
    if top >= 0:
        st[S_DISC + m] = top + MIN_VALUE
        m += 1
    st[S_DISC_N] = m
    return used


@njit(cache=True)
def _step_rows(states, rngs, actions, active):
    for i in range(states.shape[0]):
        if active[i]:
            step_one(states[i], rngs[i], actions[i])


@dataclass
class SearchConfig:
    rollouts: int = 512  # total root-action x world evaluations per decision
    top_k: int = 6  # prior pruning: candidate actions kept from the policy
    min_worlds: int = 16
    horizon: int = -1  # -1: through the end of the current round; else decisions after the root action
    deadline_s: float | None = None
    rollout_greedy: bool = True
    skip_confident: float = 0.97  # act without search when the policy is this sure


class RootMCSearch:
    """Search-enhanced agent. ``act_batch`` searches each row independently."""

    def __init__(self, net, cfg: SearchConfig | None = None, seed: int = 0, name: str = "search",
                 opponent_policy=None):
        """``opponent_policy``: optional batch policy for every non-root seat in rollouts (e.g. a known opponent
        type for analysis); by default all seats are rolled out with ``net``."""
        self.net = net.eval()
        self.opponent_policy = opponent_policy
        self.cfg = cfg or SearchConfig()
        self.name = name
        self.rng_seed = np.array([np.uint64(seed * 7919 + 17)], dtype=np.uint64)
        self.stats = {"searched": 0, "skipped": 0, "seconds": 0.0, "changed": 0}

    @torch.inference_mode()
    def _policy(self, obs, masks):
        return self.net(torch.from_numpy(obs), torch.from_numpy(masks))

    def act_batch(self, obs, masks, rows, out):
        for i in rows:
            out[i] = self.search(obs[i], masks[i])[0]

    def act(self, observation, mask, memory=None, budget=None):
        a, info = self.search(observation, mask)
        return a, memory, info

    @torch.inference_mode()
    def search(self, o: np.ndarray, mask: np.ndarray) -> tuple[int, dict]:
        cfg = self.cfg
        t0 = time.time()
        logits, v0 = self._policy(o[None].astype(np.int32), mask[None])
        prior = torch.softmax(logits, 1)[0].numpy()
        pa = int(prior.argmax())
        legal = np.flatnonzero(mask)
        if o[L.O_PHASE] == L.PH_SETUP or len(legal) == 1 or prior[pa] >= cfg.skip_confident:
            self.stats["skipped"] += 1
            return pa, {"searched": False, "prior": float(prior[pa])}
        cands = legal[np.argsort(-prior[legal])][: cfg.top_k]
        n_worlds_total = max(cfg.min_worlds, cfg.rollouts // len(cands))
        # sequential halving schedule
        phases = max(1, int(np.ceil(np.log2(len(cands)))))
        per_phase = cfg.rollouts / phases
        world_seeds = np.random.default_rng(int(self.rng_seed[0])).integers(1 << 62, size=n_worlds_total * 4)
        self.rng_seed[0] += np.uint64(1)
        sums = np.zeros(len(cands))
        cnts = np.zeros(len(cands))
        alive = np.arange(len(cands))
        used_worlds = 0
        for ph in range(phases):
            k = max(1, int(per_phase // len(alive)))
            k = max(k, cfg.min_worlds if ph == 0 else 1)
            ws = world_seeds[used_worlds : used_worlds + k]
            if len(ws) == 0:
                break
            used_worlds += len(ws)
            vals = self._evaluate(o, cands[alive], ws)
            sums[alive] += vals.sum(1)
            cnts[alive] += vals.shape[1]
            if cfg.deadline_s is not None and time.time() - t0 > cfg.deadline_s:
                break
            if len(alive) > 1 and ph < phases - 1:
                means = sums[alive] / cnts[alive]
                alive = alive[np.argsort(-means)[: max(1, (len(alive) + 1) // 2)]]
        means = np.where(cnts > 0, sums / np.maximum(cnts, 1), -np.inf)
        best = int(cands[int(np.argmax(means))])
        dt = time.time() - t0
        self.stats["searched"] += 1
        self.stats["seconds"] += dt
        self.stats["changed"] += int(best != pa)
        return best, {"searched": True, "prior_action": pa, "values": dict(zip(cands.tolist(), means.tolist(), strict=True)),
                      "value0": float(v0[0]), "seconds": dt}

    def _evaluate(self, o: np.ndarray, actions: np.ndarray, world_seeds: np.ndarray) -> np.ndarray:
        """Value of each root action on each world: array [len(actions), len(world_seeds)]."""
        cfg = self.cfg
        A, W = len(actions), len(world_seeds)
        R = A * W
        states = np.zeros((R, STATE_SIZE), dtype=np.int32)
        rngs = np.zeros((R, 1), dtype=np.uint64)
        o32 = o.astype(np.int32)
        for w in range(W):
            base = np.zeros(STATE_SIZE, dtype=np.int32)
            wr = np.array([np.uint64(world_seeds[w])], dtype=np.uint64)
            build_world(o32, wr, base)
            for a in range(A):
                states[a * W + w] = base
                rngs[a * W + w, 0] = np.uint64(world_seeds[w]) ^ np.uint64(0x5DEECE66D)  # common random numbers
        acts = np.repeat(actions, W).astype(np.int64)
        active = np.ones(R, dtype=np.bool_)
        _step_rows(states, rngs, acts, active)
        values = np.zeros(R)
        done = np.zeros(R, dtype=np.bool_)
        start_round = states[:, S_ROUND].copy()
        steps = np.ones(R, dtype=np.int64)
        obs = np.zeros((R, L.OBS_SIZE), dtype=np.int32)
        masks = np.zeros((R, N_ACTIONS), dtype=np.bool_)
        wc = np.zeros(MAX_PLAYERS)
        for _ in range(4000):
            # resolve finished rows
            for i in np.flatnonzero(~done):
                if states[i, S_PHASE] == L.PH_OVER:
                    if states[i, S_TRUNC] == 1:
                        values[i] = np.nan
                    else:
                        win_credit_one(states[i], wc)
                        values[i] = wc[0]
                    done[i] = True
            live = np.flatnonzero(~done)
            if len(live) == 0:
                break
            observe_batch(states, obs, masks)
            cur = states[:, S_CUR]
            if cfg.horizon < 0:
                past = states[:, S_ROUND] > start_round
            else:
                past = steps >= cfg.horizon
            leaf = live[(cur[live] == 0) & past[live] & (states[live, S_PHASE] != L.PH_RESOLVE)]
            if len(leaf):
                _, v = self._policy(obs[leaf], masks[leaf])
                values[leaf] = v.numpy()
                done[leaf] = True
            live = np.flatnonzero(~done)
            if len(live) == 0:
                break
            acts[:] = -1
            if self.opponent_policy is not None:
                opp_rows = live[cur[live] != 0]
                if len(opp_rows):
                    self.opponent_policy.act_batch(obs, masks, opp_rows, acts)
                net_rows = live[cur[live] == 0]
            else:
                net_rows = live
            if len(net_rows):
                logits, _ = self._policy(obs[net_rows], masks[net_rows])
                if cfg.rollout_greedy:
                    a = logits.argmax(1).numpy()
                else:
                    a = torch.multinomial(torch.softmax(logits, 1), 1).squeeze(1).numpy()
                acts[net_rows] = a
            active[:] = False
            active[live] = True
            _step_rows(states, rngs, acts, active)
            steps[live] += 1
        values[~done] = np.nan  # hit the step cap (e.g. stalling rollouts): treated as missing
        vals = values.reshape(A, W)
        if np.isnan(vals).all():
            return np.zeros_like(vals)
        return np.where(np.isnan(vals), np.nanmean(vals), vals)
