"""Search world sampling: consistency with the root observation and independence from true hidden state."""

import numpy as np
import torch

from skyjo_ml.envs.batch import STATE_SIZE, census_one, observe_one
from skyjo_ml.game import obs_layout as L
from skyjo_ml.game.rules import INVENTORY_COUNTS
from skyjo_ml.models.net import SkyjoNet
from skyjo_ml.search.root_mc import RootMCSearch, SearchConfig, build_world
from tests.test_properties import _permute_hidden, play


def _world_obs(o, seed):
    st = np.zeros(STATE_SIZE, dtype=np.int32)
    build_world(o.astype(np.int32), np.array([np.uint64(seed)], dtype=np.uint64), st)
    c = np.zeros(15, dtype=np.int64)
    census_one(st, c)
    wo = np.zeros(L.OBS_SIZE, dtype=np.int32)
    observe_one(st, wo)
    return c, wo


def test_sampled_worlds_reproduce_the_observation():
    for n, seed, k, stall in [(2, 1, 40, 0.0), (2, 2, 200, 0.0), (2, 3, 900, 0.97), (2, 4, 1500, 0.95)]:
        g, _ = play(n, seed, seed, max_steps=k, stall=stall)
        if g.over:
            continue
        o = g.observation()
        for ws in range(5):
            c, wo = _world_obs(o, ws)
            assert (c == INVENTORY_COUNTS).all()
            skip = {L.O_ROUND}
            diff = [i for i in np.flatnonzero(wo != o) if i not in skip]
            assert not diff, (n, k, diff)


def test_search_ignores_true_hidden_state():
    torch.manual_seed(0)
    net = SkyjoNet()
    g, _ = play(2, 5, 5, max_steps=20)
    assert not g.over
    while g.phase != L.PH_SELECT:
        g.step(int(np.flatnonzero(g.legal_mask())[0]))
    o1, m1 = g.observation(), g.legal_mask()
    cfg = SearchConfig(rollouts=48, top_k=3, min_worlds=8, skip_confident=1.1, horizon=6)
    a1, info1 = RootMCSearch(net, cfg, seed=3).search(o1, m1)
    _permute_hidden(g, np.random.default_rng(1))
    o2, m2 = g.observation(), g.legal_mask()
    assert (o1 == o2).all()
    a2, info2 = RootMCSearch(net, cfg, seed=3).search(o2, m2)
    assert a1 == a2 and info1["values"] == info2["values"]
