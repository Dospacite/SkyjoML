"""Reference Skyjo engine (``classic_v1_2p``).

Deliberately small and explicit: plain Python lists, one match per object, every rule in one place. It is
the specification that the compiled batch engine (``skyjo_ml.envs.batch``) is differentially tested against,
and it backs replays and the interactive CLI. Throughput is not a goal here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import obs_layout as L
from .actions import A_DRAW, A_KEEP, A_REVEAL, A_SETUP, A_TAKE, N_ACTIONS, SETUP_PAIRS
from .rng import bounded, seed_state, shuffle_prefix
from .rules import (
    INVENTORY_COUNTS,
    MATCH_END_SCORE,
    MAX_MATCH_DECISIONS,
    MAX_PLAYERS,
    MAX_ROUND_DECISIONS,
    MAX_VALUE,
    MIN_PLAYERS,
    MIN_VALUE,
    N_COLS,
    N_ROWS,
    N_SLOTS,
    N_VALUES,
)

HIDDEN, VISIBLE, ABSENT = 0, 1, 2


class IllegalAction(ValueError):
    pass


class InvariantError(RuntimeError):
    pass


@dataclass
class StepResult:
    """Public consequences of one decision."""

    actor: int
    action: int
    round_ended: bool = False
    round_scores: list[int] | None = None
    match_over: bool = False
    truncated: bool = False
    events: list[dict] = field(default_factory=list)


def canonical_deck() -> np.ndarray:
    return np.repeat(np.arange(MIN_VALUE, MAX_VALUE + 1, dtype=np.int32), INVENTORY_COUNTS)


class SkyjoGame:
    def __init__(self, n_players: int, seed: int):
        if not MIN_PLAYERS <= n_players <= MAX_PLAYERS:
            raise ValueError(f"n_players must be in [{MIN_PLAYERS}, {MAX_PLAYERS}]")
        self.n = n_players
        self.seed = seed
        self.rng = seed_state(seed)
        self.scores = [0] * n_players
        self.round = 0
        self.match_decisions = 0
        self.truncated = False
        self.round_history: list[list[int]] = []
        self.finisher_history: list[int] = []
        self._start_round(leader=0)

    # ------------------------------------------------------------------ round lifecycle
    def _start_round(self, leader: int) -> None:
        deck = canonical_deck()
        shuffle_prefix(self.rng, deck, len(deck))
        self.draw: list[int] = [int(v) for v in deck]  # top of pile = last element
        self.discard: list[int] = []
        self.val = [[0] * N_SLOTS for _ in range(self.n)]
        self.vis = [[HIDDEN] * N_SLOTS for _ in range(self.n)]
        for p in range(self.n):
            for s in range(N_SLOTS):
                self.val[p][s] = self.draw.pop()
        self.discard.append(self.draw.pop())
        self.pending: int | None = None
        self.phase = L.PH_SETUP
        self.leader = leader
        self.cur = leader
        self.setup_count = 0
        self.setup_picks: list[tuple[int, int] | None] = [None] * self.n
        self.finisher = -1
        self.final_left = 0
        self.draw_known = False
        self.drawk = [0] * N_VALUES
        self.round_turns = 0

    def _finish_setup(self) -> None:
        for p in range(self.n):
            for s in self.setup_picks[p]:
                self.vis[p][s] = VISIBLE
        if self.round == 0:
            sums = [sum(self.val[p][s] for s in self.setup_picks[p]) for p in range(self.n)]
            best = max(sums)
            tied = [p for p in range(self.n) if sums[p] == best]
            starter = tied[bounded(self.rng, len(tied))] if len(tied) > 1 else tied[0]
        else:
            starter = self.leader
        self.cur = starter
        self.phase = L.PH_SELECT

    def _check_column(self, p: int, c: int) -> bool:
        slots = range(c * N_ROWS, (c + 1) * N_ROWS)
        if all(self.vis[p][s] == VISIBLE for s in slots):
            v0 = self.val[p][c * N_ROWS]
            if all(self.val[p][s] == v0 for s in slots):
                for s in slots:
                    self.discard.append(self.val[p][s])
                    self.vis[p][s] = ABSENT
                return True
        return False

    def _hidden_count(self, p: int) -> int:
        return sum(1 for s in range(N_SLOTS) if self.vis[p][s] == HIDDEN)

    def _end_turn(self, res: StepResult) -> None:
        if self.finisher < 0:
            if self._hidden_count(self.cur) == 0:
                self.finisher = self.cur
                self.final_left = self.n - 1
                res.events.append({"type": "finish", "seat": self.cur})
        else:
            self.final_left -= 1
        if self.finisher >= 0 and self.final_left == 0:
            self._score_round(res)
            return
        self.cur = (self.cur + 1) % self.n
        self.phase = L.PH_SELECT

    def _score_round(self, res: StepResult) -> None:
        f = self.finisher
        for k in range(self.n):
            p = (f + k) % self.n
            for s in range(N_SLOTS):
                if self.vis[p][s] == HIDDEN:
                    self.vis[p][s] = VISIBLE
            for c in range(N_COLS):
                self._check_column(p, c)
        rs = [sum(self.val[p][s] for s in range(N_SLOTS) if self.vis[p][s] == VISIBLE) for p in range(self.n)]
        if rs[f] > 0 and any(rs[q] <= rs[f] for q in range(self.n) if q != f):
            rs[f] *= 2
        for p in range(self.n):
            self.scores[p] += rs[p]
        self.round_history.append(rs)
        self.finisher_history.append(f)
        res.round_ended = True
        res.round_scores = rs
        res.events.append({"type": "round_end", "round_scores": list(rs), "totals": list(self.scores)})
        if max(self.scores) >= MATCH_END_SCORE:
            self.phase = L.PH_OVER
            res.match_over = True
        else:
            self.round += 1
            self._start_round(leader=f)

    # ------------------------------------------------------------------ decisions
    def legal_mask(self) -> np.ndarray:
        m = np.zeros(N_ACTIONS, dtype=np.bool_)
        p = self.cur
        if self.phase == L.PH_SETUP:
            m[A_SETUP : A_SETUP + len(SETUP_PAIRS)] = True
        elif self.phase == L.PH_SELECT:
            if self.draw or len(self.discard) >= 2:
                m[A_DRAW] = True
            if self.discard:
                for s in range(N_SLOTS):
                    if self.vis[p][s] != ABSENT:
                        m[A_TAKE + s] = True
        elif self.phase == L.PH_RESOLVE:
            for s in range(N_SLOTS):
                if self.vis[p][s] != ABSENT:
                    m[A_KEEP + s] = True
                if self.vis[p][s] == HIDDEN:
                    m[A_REVEAL + s] = True
        if self.phase != L.PH_OVER and not m.any():
            raise InvariantError("no legal action available")
        return m

    def step(self, action: int) -> StepResult:
        action = int(action)
        if self.phase == L.PH_OVER:
            raise IllegalAction("match is over")
        if not (0 <= action < N_ACTIONS) or not self.legal_mask()[action]:
            raise IllegalAction(f"illegal action {action} in phase {self.phase}")
        p = self.cur
        res = StepResult(actor=p, action=action)
        self.match_decisions += 1
        self.round_turns += 1
        if self.phase == L.PH_SETUP:
            self.setup_picks[p] = tuple(int(x) for x in SETUP_PAIRS[action - A_SETUP])
            self.setup_count += 1
            self.cur = (self.cur + 1) % self.n
            if self.setup_count == self.n:
                self._finish_setup()
                res.events.append({"type": "setup_reveal", "picks": [list(x) for x in self.setup_picks]})
        elif self.phase == L.PH_SELECT:
            if action == A_DRAW:
                if not self.draw:
                    top = self.discard.pop()
                    rest = np.array(self.discard, dtype=np.int32)
                    shuffle_prefix(self.rng, rest, len(rest))
                    self.draw = [int(v) for v in rest]
                    self.discard = [top]
                    self.draw_known = True
                    self.drawk = [0] * N_VALUES
                    for v in self.draw:
                        self.drawk[v - MIN_VALUE] += 1
                    res.events.append({"type": "recycle", "size": len(self.draw)})
                self.pending = self.draw.pop()
                if self.draw_known:
                    self.drawk[self.pending - MIN_VALUE] -= 1
                self.phase = L.PH_RESOLVE
                res.events.append({"type": "draw", "seat": p})
            else:
                s = action - A_TAKE
                card = self.discard.pop()
                old = self.val[p][s]
                self.val[p][s] = card
                self.vis[p][s] = VISIBLE
                self.discard.append(old)
                res.events.append({"type": "take", "seat": p, "slot": s, "card": card, "out": old})
                if self._check_column(p, s // N_ROWS):
                    res.events.append({"type": "column", "seat": p, "col": s // N_ROWS, "value": card})
                self._end_turn(res)
        else:  # resolve
            card = self.pending
            self.pending = None
            if action < A_REVEAL:
                s = action - A_KEEP
                old = self.val[p][s]
                self.val[p][s] = card
                self.vis[p][s] = VISIBLE
                self.discard.append(old)
                res.events.append({"type": "keep", "seat": p, "slot": s, "card": card, "out": old})
            else:
                s = action - A_REVEAL
                self.discard.append(card)
                self.vis[p][s] = VISIBLE
                res.events.append(
                    {"type": "reveal", "seat": p, "slot": s, "card": card, "revealed": self.val[p][s]}
                )
            if self._check_column(p, s // N_ROWS):
                res.events.append(
                    {"type": "column", "seat": p, "col": s // N_ROWS, "value": self.val[p][s]}
                )
            self._end_turn(res)
        if self.phase != L.PH_OVER and (
            self.match_decisions >= MAX_MATCH_DECISIONS or self.round_turns >= MAX_ROUND_DECISIONS
        ):
            self.truncated = True
            self.phase = L.PH_OVER
            res.truncated = True
        return res

    # ------------------------------------------------------------------ views
    @property
    def over(self) -> bool:
        return self.phase == L.PH_OVER

    def win_credit(self) -> list[float]:
        if not self.over or self.truncated:
            raise InvariantError("win credit is only defined for completed matches")
        best = min(self.scores)
        k = sum(1 for s in self.scores if s == best)
        return [1.0 / k if s == best else 0.0 for s in self.scores]

    def observation(self) -> np.ndarray:
        """Raw observation for the current actor (see ``obs_layout``)."""
        o = np.zeros(L.OBS_SIZE, dtype=np.int32)
        n, a = self.n, self.cur
        o[L.O_NPLAYERS] = n
        o[L.O_PHASE] = self.phase
        o[L.O_PENDING] = -1 if self.pending is None else self.pending - MIN_VALUE
        o[L.O_TOP] = self.discard[-1] - MIN_VALUE if self.discard else -1
        o[L.O_DRAW_N] = len(self.draw)
        o[L.O_DISC_N] = len(self.discard)
        o[L.O_DRAW_KNOWN] = int(self.draw_known)
        o[L.O_FINISHER] = -1 if self.finisher < 0 else (self.finisher - a) % n
        o[L.O_FINAL_LEFT] = self.final_left
        o[L.O_ROUND] = self.round
        o[L.O_ROUND_TURNS] = self.round_turns
        pool = INVENTORY_COUNTS.astype(np.int64).copy()
        for r in range(MAX_PLAYERS):
            base = L.O_GRID + r * N_SLOTS
            if r >= n:
                o[base : base + N_SLOTS] = L.CODE_NOPLAYER
                continue
            p = (a + r) % n
            o[L.O_SCORES + r] = self.scores[p]
            for s in range(N_SLOTS):
                st = self.vis[p][s]
                if st == VISIBLE:
                    o[base + s] = self.val[p][s] - MIN_VALUE
                    pool[self.val[p][s] - MIN_VALUE] -= 1
                elif st == HIDDEN:
                    o[base + s] = L.CODE_HIDDEN
                else:
                    o[base + s] = L.CODE_ABSENT
        for v in self.discard:
            o[L.O_DISCC + v - MIN_VALUE] += 1
            pool[v - MIN_VALUE] -= 1
        if self.pending is not None:
            pool[self.pending - MIN_VALUE] -= 1
        if self.draw_known:
            for i in range(N_VALUES):
                o[L.O_DRAWK + i] = self.drawk[i]
                pool[i] -= self.drawk[i]
        o[L.O_POOL : L.O_POOL + N_VALUES] = pool
        return o

    def removed_cards(self) -> list[int]:
        """Values of cards in cleared columns (their physical location is the discard pile)."""
        return [self.val[p][s] for p in range(self.n) for s in range(N_SLOTS) if self.vis[p][s] == ABSENT]

    def card_census(self) -> np.ndarray:
        """Count every physical card by value; must always equal the inventory."""
        c = np.zeros(N_VALUES, dtype=np.int64)
        for v in self.draw + self.discard:
            c[v - MIN_VALUE] += 1
        if self.pending is not None:
            c[self.pending - MIN_VALUE] += 1
        for p in range(self.n):
            for s in range(N_SLOTS):
                if self.vis[p][s] != ABSENT:
                    c[self.val[p][s] - MIN_VALUE] += 1
        return c
