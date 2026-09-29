"""Tournament runner over the batch engine.

A tournament is a list of matches, each given as (seed, n_players, seat -> policy index). Seats are
controlled by batch policies that only see observations. Results are per-match arrays suitable for paired,
block-bootstrapped statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..envs.batch import BatchEnv
from ..game.rules import MAX_PLAYERS


@dataclass
class MatchResults:
    seeds: np.ndarray  # [M]
    n_players: np.ndarray  # [M]
    seating: np.ndarray  # [M, MAX_PLAYERS] policy index per seat (-1 unused)
    scores: np.ndarray  # [M, MAX_PLAYERS] final totals
    win_credit: np.ndarray  # [M, MAX_PLAYERS]
    rounds: np.ndarray  # [M]
    decisions: np.ndarray  # [M]
    truncated: np.ndarray  # [M] bool
    round_scores_sum: np.ndarray  # [M, MAX_PLAYERS] sum of raw round scores
    doubled: np.ndarray  # [M, MAX_PLAYERS] count of finisher penalties
    finishes: np.ndarray  # [M, MAX_PLAYERS] count of rounds finished
    elapsed: float

    def credit_of(self, policy: int) -> np.ndarray:
        """Mean win credit of ``policy`` per match (averaged over its seats in that match)."""
        sel = self.seating == policy
        cnt = sel.sum(1)
        return np.where(cnt > 0, (self.win_credit * sel).sum(1) / np.maximum(cnt, 1), np.nan)


def seat_rotation_schedule(seeds, n_players: int, lineup: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Each seed is played ``n_players`` times with the lineup rotated one seat each time."""
    if n_players != MAX_PLAYERS:
        raise ValueError(f"n_players must be {MAX_PLAYERS}")
    assert len(lineup) == n_players
    rows_seed, rows_seat = [], []
    for s in seeds:
        for r in range(n_players):
            seat = np.full(MAX_PLAYERS, -1, dtype=np.int64)
            for k in range(n_players):
                seat[(k + r) % n_players] = lineup[k]
            rows_seed.append(s)
            rows_seat.append(seat)
    return np.array(rows_seed, dtype=np.int64), np.stack(rows_seat)


def run_matches(
    policies: list,
    seeds: np.ndarray,
    n_players: np.ndarray | int,
    seating: np.ndarray,
    batch_size: int = 512,
    max_decisions: int | None = None,
) -> MatchResults:
    t0 = time.time()
    M = len(seeds)
    n_players = np.broadcast_to(np.asarray(n_players, dtype=np.int64), (M,)).copy()
    B = min(batch_size, M)
    env = BatchEnv(B)
    slot_match = np.full(B, -1, dtype=np.int64)
    next_match = 0

    scores = np.zeros((M, MAX_PLAYERS), dtype=np.int64)
    credit = np.zeros((M, MAX_PLAYERS))
    rounds = np.zeros(M, dtype=np.int64)
    decisions = np.zeros(M, dtype=np.int64)
    trunc = np.zeros(M, dtype=bool)
    rs_sum = np.zeros((M, MAX_PLAYERS), dtype=np.int64)
    doubled = np.zeros((M, MAX_PLAYERS), dtype=np.int64)
    finishes = np.zeros((M, MAX_PLAYERS), dtype=np.int64)

    def fill(idx):
        nonlocal next_match
        take = idx[: max(0, min(len(idx), M - next_match))]
        if len(take):
            ms = np.arange(next_match, next_match + len(take))
            next_match += len(take)
            slot_match[take] = ms
            env.reset(take, seeds[ms], n_players[ms])
            for p in policies:
                if hasattr(p, "reset_rows"):
                    p.reset_rows(take)
        slot_match[idx[len(take) :]] = -1

    fill(np.arange(B))
    actions = np.full(B, -1, dtype=np.int64)
    from ..envs.batch import S_LAST_DBL, S_LAST_FIN, S_LAST_RS, S_MATCH_DEC

    while True:
        active = np.flatnonzero(slot_match >= 0)
        if len(active) == 0:
            break
        obs, masks = env.observe()
        actions[:] = -1
        cur = env.cur
        ctrl = seating[slot_match[active], cur[active]]
        for pi, pol in enumerate(policies):
            rows = active[ctrl == pi]
            if len(rows):
                pol.act_batch(obs, masks, rows, actions)
        env.step(actions)
        stepped = active
        ended = stepped[env.round_ended[stepped]]
        for i in ended:
            m = slot_match[i]
            n = n_players[m]
            rs = env.states[i, S_LAST_RS : S_LAST_RS + n]
            rounds[m] += 1
            f = env.states[i, S_LAST_FIN]
            rs_sum[m, :n] += rs
            finishes[m, f] += 1
            doubled[m, f] += env.states[i, S_LAST_DBL]
        over = stepped[env.over[stepped]]
        if max_decisions is not None:
            over = np.union1d(over, stepped[env.states[stepped, S_MATCH_DEC] >= max_decisions])
        if len(over):
            wc = env.win_credit()
            for i in over:
                m = slot_match[i]
                n = n_players[m]
                scores[m, :n] = env.scores[i, :n]
                decisions[m] = env.states[i, S_MATCH_DEC]
                trunc[m] = bool(env.truncated[i]) or not bool(env.over[i])
                if not trunc[m]:
                    credit[m] = wc[i]
            fill(over)
    return MatchResults(
        seeds=np.asarray(seeds),
        n_players=n_players,
        seating=seating,
        scores=scores,
        win_credit=credit,
        rounds=rounds,
        decisions=decisions,
        truncated=trunc,
        round_scores_sum=rs_sum,
        doubled=doubled,
        finishes=finishes,
        elapsed=time.time() - t0,
    )
