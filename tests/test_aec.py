import numpy as np
from pettingzoo.test import api_test

from skyjo_ml.envs.aec import SkyjoAEC


def test_pettingzoo_api():
    api_test(SkyjoAEC(2), num_cycles=300)


def test_full_match_rewards_sum_to_one():
    env = SkyjoAEC(2)
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    total = {a: 0.0 for a in env.possible_agents}
    for agent in env.agent_iter():
        obs, rew, term, trunc, _ = env.last()
        total[agent] += rew
        if term or trunc:
            env.step(None)
            continue
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    assert abs(sum(total.values()) - 1.0) < 1e-9
