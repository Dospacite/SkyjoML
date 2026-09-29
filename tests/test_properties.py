"""Property tests: conservation, score consistency, determinism, and hidden-information integrity."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from skyjo_ml.game import obs_layout as L
from skyjo_ml.game.engine import HIDDEN, SkyjoGame
from skyjo_ml.game.rules import INVENTORY_COUNTS, MIN_VALUE, N_VALUES


def play(n, seed, policy_seed, max_steps=4000, check=None, stall=0.0):
    """Random legal play; ``stall`` is the probability of preferring moves that reveal nothing, which
    produces long rounds that exhaust and recycle the draw pile."""
    g = SkyjoGame(n, seed)
    rng = np.random.default_rng(policy_seed)
    actions = []
    while not g.over and len(actions) < max_steps:
        m = g.legal_mask()
        assert m.any()
        if check:
            check(g)
        legal = np.flatnonzero(m)
        if stall and rng.random() < stall:
            quiet = [a for a in legal if a == 66 or (67 <= a < 91 and g.vis[g.cur][(a - 67) % 12] != HIDDEN)]
            if quiet:
                legal = np.array(quiet)
        a = int(rng.choice(legal))
        actions.append(a)
        g.step(a)
    return g, actions


def _check_state(g: SkyjoGame):
    assert (g.card_census() == INVENTORY_COUNTS).all()
    o = g.observation()
    # public card accounting equals the true composition of non-public locations
    true_pool = np.zeros(N_VALUES, dtype=np.int64)
    for p in range(g.n):
        for s in range(12):
            if g.vis[p][s] == HIDDEN:
                true_pool[g.val[p][s] - MIN_VALUE] += 1
    draw_comp = np.zeros(N_VALUES, dtype=np.int64)
    for v in g.draw:
        draw_comp[v - MIN_VALUE] += 1
    if g.draw_known:
        assert (o[L.O_DRAWK : L.O_DRAWK + N_VALUES] == draw_comp).all()
    else:
        true_pool += draw_comp
    assert (o[L.O_POOL : L.O_POOL + N_VALUES] == true_pool).all()
    assert (o[L.O_POOL : L.O_POOL + N_VALUES] >= 0).all()


@settings(max_examples=40, deadline=None)
@given(
    n=st.just(2),
    seed=st.integers(0, 2**40),
    pseed=st.integers(0, 2**20),
    stall=st.sampled_from([0.0, 0.9, 0.98]),
)
def test_conservation_and_accounting(n, seed, pseed, stall):
    g, _ = play(n, seed, pseed, check=_check_state, stall=stall, max_steps=3000)
    if g.over and not g.truncated:
        assert [sum(r[p] for r in g.round_history) for p in range(n)] == g.scores
        assert max(g.scores) >= 100
        assert sum(g.win_credit()) == pytest.approx(1.0)


@settings(max_examples=20, deadline=None)
@given(n=st.just(2), seed=st.integers(0, 2**40), pseed=st.integers(0, 2**20))
def test_deterministic_replay(n, seed, pseed):
    g1, actions = play(n, seed, pseed)
    g2 = SkyjoGame(n, seed)
    for a in actions:
        g2.step(a)
    assert g1.scores == g2.scores and g1.round_history == g2.round_history
    assert (g1.observation() == g2.observation()).all()


def _permute_hidden(g: SkyjoGame, rng: np.random.Generator):
    """Shuffle every card whose location the actor cannot know, keeping public information fixed."""
    locs = [(p, s) for p in range(g.n) for s in range(12) if g.vis[p][s] == HIDDEN]
    vals = [g.val[p][s] for p, s in locs]
    if not g.draw_known:
        vals += g.draw
    perm = rng.permutation(len(vals))
    vals = [vals[i] for i in perm]
    for k, (p, s) in enumerate(locs):
        g.val[p][s] = vals[k]
    if not g.draw_known:
        g.draw = vals[len(locs) :]
    else:
        g.draw = [g.draw[i] for i in rng.permutation(len(g.draw))]


@settings(max_examples=30, deadline=None)
@given(
    n=st.just(2),
    seed=st.integers(0, 2**40),
    pseed=st.integers(0, 2**20),
    k=st.integers(0, 1500),
    stall=st.sampled_from([0.0, 0.95]),
)
def test_observation_invariant_to_hidden_state(n, seed, pseed, k, stall):
    g, _ = play(n, seed, pseed, max_steps=k, stall=stall)
    if g.over:
        return
    o1, m1 = g.observation(), g.legal_mask()
    _permute_hidden(g, np.random.default_rng(pseed))
    assert (g.card_census() == INVENTORY_COUNTS).all()
    assert (g.observation() == o1).all()
    assert (g.legal_mask() == m1).all()


def test_pending_card_only_visible_to_drawer():
    g = SkyjoGame(2, 5)
    for _ in range(2):
        g.step(0)
    g.step(66)
    o = g.observation()
    assert o[L.O_PENDING] >= 0 and o[L.O_PHASE] == L.PH_RESOLVE
    # after resolution the next actor sees no pending card
    g.step(91 + [s for s in range(12) if g.vis[g.cur][s] == HIDDEN][0])
    assert g.observation()[L.O_PENDING] == -1


def test_recycle_exercised_by_stalling_traces():
    seen = 0
    for seed in range(6):
        g, _ = play(2, seed, seed, max_steps=1500, stall=0.97, check=_check_state)
        seen += int(g.draw_known)
    assert seen > 0
