"""Reference engine vs compiled batch engine on identical seeds and action sequences."""

import numpy as np
import pytest

from skyjo_ml.envs import batch as B
from skyjo_ml.evaluation.tournament import seat_rotation_schedule
from skyjo_ml.game.engine import SkyjoGame


def _run_pair(n_players: int, seed: int, policy_seed: int, max_steps: int = 5000, stall: float = 0.0):
    g = SkyjoGame(n_players, seed)
    env = B.BatchEnv(1)
    env.reset(np.array([0]), np.array([seed]), np.array([n_players]))
    rng = np.random.default_rng(policy_seed)
    steps = 0
    while not g.over and steps < max_steps:
        obs, masks = env.observe()
        ro = g.observation()
        rm = g.legal_mask()
        assert np.array_equal(obs[0], ro), (steps, np.nonzero(obs[0] != ro))
        assert np.array_equal(masks[0], rm), steps
        assert env.cur[0] == g.cur
        legal = np.flatnonzero(rm)
        if stall and rng.random() < stall:  # long rounds that exhaust and recycle the draw pile
            quiet = [a for a in legal if a == 66 or (67 <= a < 91 and g.vis[g.cur][(a - 67) % 12] != 0)]
            legal = np.array(quiet) if quiet else legal
        a = int(rng.choice(legal))
        res = g.step(a)
        env.step(np.array([a]))
        assert bool(env.round_ended[0]) == res.round_ended
        if res.round_ended:
            assert env.last_round_scores[0, :n_players].tolist() == res.round_scores
        assert env.scores[0, :n_players].tolist() == g.scores
        steps += 1
    assert bool(env.over[0]) == g.over
    if g.over and not g.truncated:
        wc = env.win_credit()[0, :n_players]
        assert np.allclose(wc, g.win_credit())
    return g


@pytest.mark.parametrize("n_players", [2])
def test_engines_agree_random_play(n_players):
    for seed in range(12):
        _run_pair(n_players, seed=1000 * n_players + seed, policy_seed=seed)


@pytest.mark.parametrize("n_players", [2])
def test_engines_agree_through_recycling(n_players):
    recycled = False
    for seed in range(8):
        g = _run_pair(n_players, seed=77 + seed, policy_seed=seed, max_steps=2500, stall=0.97)
        recycled |= g.draw_known or g.round > 0
    assert recycled


def test_batch_and_tournament_reject_other_player_counts():
    env = B.BatchEnv(1)
    with pytest.raises(ValueError, match="n_players"):
        env.reset(np.array([0]), np.array([1]), np.array([4]))
    with pytest.raises(ValueError, match="n_players"):
        seat_rotation_schedule(np.array([1]), 4, [0, 1, 1, 1])
