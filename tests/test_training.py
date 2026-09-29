"""Trajectory alignment and advantage computation."""

import numpy as np

from skyjo_ml.training.ppo import compute_gae


def _gae(traj, val, rew, done, trunc, lam=0.95):
    n = len(traj)
    adv = np.zeros(n)
    valid = np.zeros(n, dtype=np.bool_)
    compute_gae(np.array(traj, dtype=np.int64), np.array(val, float), np.array(rew, float),
                np.array(done, bool), np.array(trunc, bool), max(traj) + 1, 1.0, lam, adv, valid)
    return adv, valid


def test_terminal_trajectory_monte_carlo_with_lambda_one():
    # interleaved trajectories 0 and 1 (two seats of one match), terminal rewards 1 and 0
    traj = [0, 1, 0, 1, 0, 1]
    val = [0.5, 0.5, 0.6, 0.4, 0.7, 0.3]
    rew = [0, 0, 0.1, 0, 1.0, 0.0]
    done = [0, 0, 0, 0, 1, 1]
    adv, valid = _gae(traj, val, rew, done, [0] * 6, lam=1.0)
    ret = adv + np.array(val)
    assert valid.all()
    # gamma = lambda = 1: returns are undiscounted sums of later rewards of the same seat
    assert np.allclose(ret[[0, 2, 4]], [1.1, 1.1, 1.0])
    assert np.allclose(ret[[1, 3, 5]], [0.0, 0.0, 0.0])


def test_open_trajectory_tail_is_carried_not_trained():
    traj = [0, 0, 0]
    val = [0.2, 0.4, 0.5]
    rew = [0.0, 0.0, 0.0]
    adv, valid = _gae(traj, val, rew, [0, 0, 0], [0, 0, 0], lam=0.5)
    assert valid.tolist() == [True, True, False]
    # second: delta = 0 + 0.5 - 0.4 = 0.1 ; first: delta = 0.4 - 0.2 = 0.2, + 0.5 * 0.1
    assert np.allclose(adv[:2], [0.25, 0.1])


def test_truncated_bootstraps_on_own_value():
    adv, valid = _gae([0, 0], [0.3, 0.6], [0.0, 0.0], [0, 0], [0, 1], lam=1.0)
    assert valid.all()
    assert np.allclose(adv, [0.3, 0.0])
