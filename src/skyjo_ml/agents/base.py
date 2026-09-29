"""Common agent protocol.

Batch policies receive only raw observations and legal masks (``obs_layout``); they never see engine state.
``Agent.act(observation, memory, budget) -> (action, memory, diagnostics)`` is the single-decision form
used by the CLI and the search wrapper.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .heuristics import PRESETS, heuristic_batch


class BatchPolicy(Protocol):
    name: str

    def act_batch(self, obs: np.ndarray, masks: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
        """Write actions for ``rows`` of ``obs``/``masks`` into ``out[rows]``."""


class Agent(Protocol):
    name: str

    def act(self, observation: np.ndarray, mask: np.ndarray, memory: Any = None, budget: Any = None):
        """Return ``(legal_action, new_memory, diagnostics)``."""


class HeuristicBatchPolicy:
    def __init__(self, name: str, params: np.ndarray | None = None, seed: int = 0, batch_size: int = 4096):
        kind, default = PRESETS[name]
        self.name = name if params is None else f"{name}*"
        self.kind = kind
        self.params = np.asarray(default if params is None else params, dtype=np.float64)
        self.seed = seed
        self._rngs = np.zeros((0, 1), dtype=np.uint64)
        self._ensure(batch_size)

    def _ensure(self, n: int) -> None:
        if len(self._rngs) < n:
            old = len(self._rngs)
            extra = np.arange(old, n, dtype=np.uint64)
            new = (extra * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64((self.seed * 7919 + 1) & 0xFFFFFFFF)
            self._rngs = np.concatenate([self._rngs, new[:, None]])

    def act_batch(self, obs, masks, rows, out):
        if len(rows) == 0:
            return
        self._ensure(len(obs))
        kinds = np.full(len(rows), self.kind, dtype=np.int64)
        params = np.broadcast_to(self.params, (len(rows), len(self.params))).copy()
        heuristic_batch(obs, masks, rows.astype(np.int64), kinds, params, self._rngs, out)

    def act(self, observation, mask, memory=None, budget=None):
        out = np.full(1, -1, dtype=np.int64)
        self.act_batch(observation[None], mask[None], np.zeros(1, dtype=np.int64), out)
        return int(out[0]), memory, {}
