"""Public-information table tracker for advising at a physical Skyjo table.

It reconstructs the raw observation of one seat from public events only (plus that seat's own draws), so
it never needs unrevealed card values. ``tests/test_tracker.py`` drives the real engine, feeds the tracker
only what the advised player could see, and checks that the reconstructed observations are identical.

Event API (seats are absolute table seats, clockwise):

* ``setup(seat, slots, values)``             two revealed setup cards of a player (any order of players)
* ``start(seat)``                            who takes the first turn of the round (after all setups)
* ``take(slot, replaced_value)``             current player takes the top discard into ``slot``
* ``draw(value=None)``                       current player draws; ``value`` only for the advised seat
* ``keep(slot, drawn_value, replaced_value)`` current player keeps the drawn card
* ``reveal(slot, drawn_value, revealed)``    current player discards the drawn card and reveals ``slot``
* ``round_scores(scores)``                   announced final round scores (after the round ends)
* ``new_round(first_discard)``               new deal: top card of the fresh discard pile
"""

from __future__ import annotations

import numpy as np

from . import obs_layout as L
from .actions import N_ACTIONS
from .rules import DECK_SIZE, INVENTORY_COUNTS, MATCH_END_SCORE, MAX_PLAYERS, MIN_VALUE, N_ROWS, N_SLOTS, N_VALUES

HID, ABS = None, "absent"


class TableTracker:
    def __init__(self, n_players: int, my_seat: int, first_discard: int):
        if n_players != MAX_PLAYERS:
            raise ValueError(f"n_players must be {MAX_PLAYERS}")
        if not 0 <= my_seat < n_players:
            raise ValueError("my_seat is outside the table")
        self.n = n_players
        self.me = my_seat
        self.scores = [0] * n_players
        self.round = 0
        self.match_over = False
        self.new_round(first_discard, leader=0)

    # ------------------------------------------------------------------ lifecycle
    def new_round(self, first_discard: int, leader: int | None = None) -> None:
        self.grid: list[list] = [[HID] * N_SLOTS for _ in range(self.n)]
        self.discard = [first_discard]
        self.draw_n = DECK_SIZE - 12 * self.n - 1
        self.draw_known = False
        self.drawk = np.zeros(N_VALUES, dtype=np.int64)
        self.pending: int | None = None
        self.phase = L.PH_SETUP
        self.leader = self.cur = leader if leader is not None else self.finisher
        self.setup_done = [False] * self.n
        self.finisher = -1
        self.final_left = 0
        self.round_turns = 0

    def setup(self, seat: int, slots: tuple[int, int], values: tuple[int, int] | None = None) -> None:
        """Record a player's setup reveal. Values may be given later (revealed together)."""
        for s, v in zip(slots, values or (None, None), strict=True):
            self.grid[seat][s] = v if v is not None else "committed"
        self.setup_done[seat] = True
        self.round_turns += 1

    def reveal_setup_values(self, seat: int, slots: tuple[int, int], values: tuple[int, int]) -> None:
        for s, v in zip(slots, values, strict=True):
            self.grid[seat][s] = v

    def start(self, seat: int) -> None:
        for p in range(self.n):
            for s in range(N_SLOTS):
                if self.grid[p][s] == "committed":
                    raise ValueError("enter every setup value before the first turn")
        self.cur = seat
        self.phase = L.PH_SELECT

    # ------------------------------------------------------------------ turns
    def _check_column(self, p: int, c: int) -> None:
        cells = [self.grid[p][c * N_ROWS + r] for r in range(N_ROWS)]
        if all(isinstance(x, int) for x in cells) and cells[0] == cells[1] == cells[2]:
            for r in range(N_ROWS):
                self.discard.append(cells[r])
                self.grid[p][c * N_ROWS + r] = ABS

    def _end_turn(self) -> None:
        p = self.cur
        self.round_turns += 1
        if self.finisher < 0:
            if not any(x is HID for x in self.grid[p]):
                self.finisher = p
                self.final_left = self.n - 1
        else:
            self.final_left -= 1
        if self.finisher >= 0 and self.final_left == 0:
            self.phase = L.PH_OVER  # waiting for round_scores()
            return
        self.cur = (p + 1) % self.n
        self.phase = L.PH_SELECT

    def take(self, slot: int, replaced_value: int) -> None:
        assert self.phase == L.PH_SELECT
        card = self.discard.pop()
        old = self.grid[self.cur][slot]
        if isinstance(old, int) and old != replaced_value:
            raise ValueError(f"slot shows {old}, not {replaced_value}")
        self.grid[self.cur][slot] = card
        self.discard.append(replaced_value)
        self._check_column(self.cur, slot // N_ROWS)
        self._end_turn()

    def draw(self, value: int | None = None) -> None:
        assert self.phase == L.PH_SELECT
        if self.draw_n == 0:
            top = self.discard.pop()
            self.drawk = np.zeros(N_VALUES, dtype=np.int64)
            for v in self.discard:
                self.drawk[v - MIN_VALUE] += 1
            self.draw_n = len(self.discard)
            self.discard = [top]
            self.draw_known = True
        self.draw_n -= 1
        self.round_turns += 1
        self.pending = value if self.cur == self.me else None
        self.phase = L.PH_RESOLVE
        self._accounted = False
        if self.pending is not None:
            self._account_draw(self.pending)

    def _account_draw(self, drawn: int) -> None:
        if self.draw_known and not self._accounted:
            self.drawk[drawn - MIN_VALUE] -= 1
        self._accounted = True

    def keep(self, slot: int, drawn_value: int, replaced_value: int) -> None:
        assert self.phase == L.PH_RESOLVE
        self._account_draw(drawn_value)
        old = self.grid[self.cur][slot]
        if isinstance(old, int) and old != replaced_value:
            raise ValueError(f"slot shows {old}, not {replaced_value}")
        self.grid[self.cur][slot] = drawn_value
        self.discard.append(replaced_value)
        self.pending = None
        self._check_column(self.cur, slot // N_ROWS)
        self._end_turn()

    def reveal(self, slot: int, drawn_value: int, revealed: int) -> None:
        assert self.phase == L.PH_RESOLVE and self.grid[self.cur][slot] is HID
        self._account_draw(drawn_value)
        self.discard.append(drawn_value)
        self.grid[self.cur][slot] = revealed
        self.pending = None
        self._check_column(self.cur, slot // N_ROWS)
        self._end_turn()

    def round_scores(self, scores: list[int]) -> None:
        for p in range(self.n):
            self.scores[p] += scores[p]
        self.round += 1
        self.match_over = max(self.scores) >= MATCH_END_SCORE

    # ------------------------------------------------------------------ views
    def observation(self) -> np.ndarray:
        """Raw observation for the advised seat (must be the current actor)."""
        o = np.zeros(L.OBS_SIZE, dtype=np.int32)
        n, a = self.n, self.me
        o[L.O_NPLAYERS] = n
        o[L.O_PHASE] = self.phase
        o[L.O_PENDING] = -1 if self.pending is None else self.pending - MIN_VALUE
        o[L.O_TOP] = self.discard[-1] - MIN_VALUE if self.discard else -1
        o[L.O_DRAW_N] = self.draw_n
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
                x = self.grid[p][s]
                if x is ABS:
                    o[base + s] = L.CODE_ABSENT
                elif isinstance(x, int) and self.phase != L.PH_SETUP:
                    o[base + s] = x - MIN_VALUE
                    pool[x - MIN_VALUE] -= 1
                else:
                    o[base + s] = L.CODE_HIDDEN
        for v in self.discard:
            o[L.O_DISCC + v - MIN_VALUE] += 1
            pool[v - MIN_VALUE] -= 1
        if self.pending is not None:
            pool[self.pending - MIN_VALUE] -= 1
        if self.draw_known:
            o[L.O_DRAWK : L.O_DRAWK + N_VALUES] = self.drawk
            pool -= self.drawk
        o[L.O_POOL : L.O_POOL + N_VALUES] = pool
        return o

    def legal_mask(self) -> np.ndarray:
        from .actions import A_DRAW, A_KEEP, A_REVEAL, A_SETUP, A_TAKE, N_SETUP

        m = np.zeros(N_ACTIONS, dtype=np.bool_)
        g = self.grid[self.me]
        if self.phase == L.PH_SETUP:
            m[A_SETUP : A_SETUP + N_SETUP] = True
        elif self.phase == L.PH_SELECT:
            m[A_DRAW] = self.draw_n > 0 or len(self.discard) >= 2
            for s in range(N_SLOTS):
                if g[s] is not ABS and self.discard:
                    m[A_TAKE + s] = True
        elif self.phase == L.PH_RESOLVE:
            for s in range(N_SLOTS):
                if g[s] is not ABS:
                    m[A_KEEP + s] = True
                if g[s] is HID:
                    m[A_REVEAL + s] = True
        return m
