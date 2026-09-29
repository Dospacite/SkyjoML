"""Neural agents: batch policy for rollouts/evaluation and single-decision ``Agent`` wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..models.net import NetConfig, SkyjoNet


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[SkyjoNet, dict]:
    """Load a training checkpoint or an exported package; refuses packages built for other rules/schemas."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if "package_version" in ck:
        from ..game.rules import ruleset_hash
        from ..training.manifest import schema_hash

        if ck["ruleset_hash"] != ruleset_hash() or ck["schema_hash"] != schema_hash():
            raise ValueError("model package was built for a different ruleset or observation schema")
    cfg = NetConfig(**ck["net_config"])
    net = SkyjoNet(cfg)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


class NeuralBatchPolicy:
    """Frozen network acting on observation rows. ``greedy`` picks the arg-max legal action."""

    def __init__(self, net: SkyjoNet, name: str = "net", greedy: bool = False, seed: int = 0,
                 temperature: float = 1.0):
        self.net = net.eval()
        self.name = name
        self.greedy = greedy
        self.temperature = temperature
        self.gen = torch.Generator().manual_seed(seed)

    @torch.inference_mode()
    def logits_values(self, obs: np.ndarray, masks: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(torch.from_numpy(np.ascontiguousarray(obs)), torch.from_numpy(np.ascontiguousarray(masks)))

    @torch.inference_mode()
    def act_batch(self, obs, masks, rows, out):
        if len(rows) == 0:
            return
        logits, _ = self.logits_values(obs[rows], masks[rows])
        if self.greedy:
            a = logits.argmax(1)
        else:
            a = torch.multinomial(torch.softmax(logits / self.temperature, 1), 1, generator=self.gen).squeeze(1)
        out[rows] = a.numpy()

    def act(self, observation, mask, memory=None, budget=None):
        logits, value = self.logits_values(observation[None], mask[None])
        probs = torch.softmax(logits, 1)[0]
        if self.greedy:
            a = int(probs.argmax())
        else:
            a = int(torch.multinomial(probs, 1, generator=self.gen))
        return a, memory, {"value": float(value[0]), "prob": float(probs[a])}
