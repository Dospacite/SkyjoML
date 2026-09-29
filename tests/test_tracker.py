"""The table tracker reconstructs the advised seat's observation from public events and own draws only."""

import numpy as np
import pytest

from skyjo_ml.game import obs_layout as L
from skyjo_ml.game.actions import A_DRAW, A_KEEP, A_REVEAL, A_SETUP, A_TAKE, SETUP_PAIRS
from skyjo_ml.game.engine import SkyjoGame
from skyjo_ml.game.tracker import TableTracker


def drive(n, seed, me, stall=0.0, max_steps=3000):
    g = SkyjoGame(n, seed)
    t = TableTracker(n, me, g.discard[-1])
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(max_steps):
        if g.over:
            break
        m = g.legal_mask()
        if g.cur == me:
            assert (t.observation() == g.observation()).all(), np.flatnonzero(t.observation() != g.observation())
            assert (t.legal_mask() == m).all()
            checked += 1
        legal = np.flatnonzero(m)
        if stall and rng.random() < stall:
            q = [a for a in legal if a == A_DRAW or (A_TAKE <= a < A_REVEAL and g.vis[g.cur][(a - A_TAKE) % 12] == 1)]
            legal = np.array(q) if q else legal
        a = int(rng.choice(legal))
        p = g.cur
        phase = g.phase
        if phase == L.PH_SETUP:
            pair = tuple(int(x) for x in SETUP_PAIRS[a - A_SETUP])
            t.setup(p, pair)
            res = g.step(a)
            if g.phase == L.PH_SELECT:
                for q in range(n):
                    sl = g.setup_picks[q]
                    t.reveal_setup_values(q, sl, tuple(g.val[q][s] for s in sl))
                t.start(g.cur)
            continue
        if a == A_DRAW:
            res = g.step(a)
            t.draw(g.pending if p == me else None)
        elif A_TAKE <= a < A_KEEP:
            old = g.val[p][a - A_TAKE]
            res = g.step(a)
            t.take(a - A_TAKE, old)
        elif A_KEEP <= a < A_REVEAL:
            old, drawn = g.val[p][a - A_KEEP], g.pending
            res = g.step(a)
            t.keep(a - A_KEEP, drawn, old)
        else:
            drawn, rev = g.pending, g.val[p][a - A_REVEAL]
            res = g.step(a)
            t.reveal(a - A_REVEAL, drawn, rev)
        if res.round_ended:
            t.round_scores(res.round_scores)
            if not g.over:
                t.new_round(g.discard[-1])
    assert t.scores == g.scores
    return checked


@pytest.mark.parametrize("n,me", [(2, 0), (2, 1)])
def test_tracker_matches_engine(n, me):
    assert drive(n, 10 + n, me) > 10


def test_tracker_through_recycles():
    assert drive(2, 99, 1, stall=0.97, max_steps=2500) > 100
