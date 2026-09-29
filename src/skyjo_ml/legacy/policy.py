"""Batch policy wrapper that lets the legacy 2-8 player model act in the 2-player engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..game import obs_layout as NEW
from . import obs_layout_mp8 as OLD
from .net_mp8 import NetConfig, SkyjoNet

LEGACY_MODEL = Path(__file__).resolve().parents[3] / "artifacts" / "legacy_multiplayer" / "skyjo_policy.pt"


def convert(obs: np.ndarray) -> np.ndarray:
    """Map current observations (NEW.MAX_PLAYERS seats) to the legacy 8-seat layout."""
    B = obs.shape[0]
    out = np.zeros((B, OLD.OBS_SIZE), dtype=np.int32)
    out[:, : NEW.O_SCORES] = obs[:, : NEW.O_SCORES]
    out[:, OLD.O_SCORES: OLD.O_SCORES + NEW.MAX_PLAYERS] = obs[:, NEW.O_SCORES: NEW.O_SCORES + NEW.MAX_PLAYERS]
    out[:, OLD.O_GRID: OLD.O_GRID + OLD.MAX_PLAYERS * OLD.N_SLOTS] = OLD.CODE_NOPLAYER
    ng = NEW.MAX_PLAYERS * NEW.N_SLOTS
    out[:, OLD.O_GRID: OLD.O_GRID + ng] = obs[:, NEW.O_GRID: NEW.O_GRID + ng]
    for old_off, new_off in [(OLD.O_POOL, NEW.O_POOL), (OLD.O_DRAWK, NEW.O_DRAWK), (OLD.O_DISCC, NEW.O_DISCC)]:
        out[:, old_off: old_off + OLD.N_VALUES] = obs[:, new_off: new_off + NEW.N_VALUES]
    return out


def load_legacy(path: str | Path = LEGACY_MODEL) -> SkyjoNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = SkyjoNet(NetConfig(**ck["net_config"]))
    net.load_state_dict(ck["model"])
    return net.eval()


class LegacyBatchPolicy:
    def __init__(self, path: str | Path = LEGACY_MODEL, name: str = "legacy_mp"):
        self.net = load_legacy(path)
        self.name = name

    @torch.inference_mode()
    def act_batch(self, obs, masks, rows, out):
        if len(rows) == 0:
            return
        o = convert(obs[rows])
        logits, _ = self.net(torch.from_numpy(o), torch.from_numpy(np.ascontiguousarray(masks[rows])))
        out[rows] = logits.argmax(1).numpy()

    def act(self, observation, mask, memory=None, budget=None):
        out = np.zeros(1, dtype=np.int64)
        self.act_batch(observation[None], mask[None], np.zeros(1, dtype=np.int64), out)
        return int(out[0]), memory, {}
