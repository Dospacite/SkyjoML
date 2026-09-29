"""PettingZoo AEC adapter over the reference engine (interoperability; training uses ``BatchEnv``).

Observations are dicts ``{"observation": raw int32 vector, "action_mask": int8[103]}`` for the acting agent;
non-acting agents receive their own seat-relative observation with an all-zero mask. Rewards are terminal
win credit (1/k for k tied winners), assigned when the match ends.
"""

from __future__ import annotations

import numpy as np

try:
    from gymnasium import spaces
    from pettingzoo import AECEnv
except ImportError as e:  # pragma: no cover
    raise ImportError("install the 'aec' extra: pettingzoo and gymnasium") from e

from ..game import obs_layout as L
from ..game.actions import N_ACTIONS
from ..game.engine import SkyjoGame
from ..game.rules import MAX_PLAYERS, RULESET_NAME


class SkyjoAEC(AECEnv):
    metadata = {"name": f"skyjo_{RULESET_NAME}", "render_modes": ["ansi"], "is_parallelizable": False}

    def __init__(self, n_players: int = 2, render_mode: str | None = None):
        super().__init__()
        if n_players != MAX_PLAYERS:
            raise ValueError(f"n_players must be {MAX_PLAYERS}")
        self.n_players = n_players
        self.possible_agents = [f"player_{i}" for i in range(n_players)]
        self.render_mode = render_mode
        obs_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-(10**6), high=10**6, shape=(L.OBS_SIZE,), dtype=np.int32),
                "action_mask": spaces.Box(low=0, high=1, shape=(N_ACTIONS,), dtype=np.int8),
            }
        )
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: spaces.Discrete(N_ACTIONS) for a in self.possible_agents}
        self.game: SkyjoGame | None = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        seed = int(np.random.SeedSequence().entropy % (2**63)) if seed is None else seed
        self.game = SkyjoGame(self.n_players, seed)
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = self.agents[self.game.cur]

    def observe(self, agent):
        g = self.game
        i = self.possible_agents.index(agent)
        if g.over or i != g.cur:
            # seat-relative public view for a waiting agent: same layout, no pending card, no legal actions
            cur = g.cur
            g.cur = i
            pending, g.pending = g.pending, None
            try:
                o = g.observation()
            finally:
                g.cur, g.pending = cur, pending
            return {"observation": o, "action_mask": np.zeros(N_ACTIONS, dtype=np.int8)}
        return {"observation": g.observation(), "action_mask": g.legal_mask().astype(np.int8)}

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        g = self.game
        self._cumulative_rewards[agent] = 0.0
        res = g.step(int(action))
        self.rewards = {a: 0.0 for a in self.agents}
        if res.match_over or res.truncated:
            if res.truncated:
                self.truncations = {a: True for a in self.agents}
            else:
                wc = g.win_credit()
                self.rewards = {a: float(wc[k]) for k, a in enumerate(self.agents)}
                self.terminations = {a: True for a in self.agents}
        else:
            self.agent_selection = self.agents[g.cur]
        self._accumulate_rewards()

    def render(self):
        from ..game.replay import render_board

        g = self.game
        lines = [f"round {g.round + 1} totals {g.scores} top discard {g.discard[-1] if g.discard else None}"]
        for p in range(g.n):
            lines.append(f"P{p}: " + " | ".join(render_board(g, p)))
        return "\n".join(lines)

    def close(self):
        self.game = None
