"""Paired evaluation of the search wrapper against its own policy-only mode.

Both variants play the same seat rotations of the same deals against the same opponents, so the difference in
win credit is a paired estimate of the search gain at a stated per-decision compute budget. Latency is
recorded per searched decision (p50/p95).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..agents.neural import NeuralBatchPolicy, load_model
from ..search.root_mc import RootMCSearch, SearchConfig
from .benchmark import make_policy, seed_block
from .stats import paired_difference, summarize
from .tournament import run_matches, seat_rotation_schedule


class _TimedSearch(RootMCSearch):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.latencies: list[float] = []

    def search(self, o, mask):
        t = time.perf_counter()
        out = super().search(o, mask)
        if out[1].get("searched"):
            self.latencies.append(time.perf_counter() - t)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--opponent", default="finish")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--rollouts", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=-1)
    ap.add_argument("--skip-confident", type=float, default=0.95)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--offset", type=int, default=700_000)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(1)
    n = a.players
    net, _ = load_model(a.checkpoint)
    seeds, seat = seat_rotation_schedule(seed_block(a.split, n, a.seeds, offset=a.offset), n, [0] + [1] * (n - 1))
    res = {}
    base = NeuralBatchPolicy(net, "policy", greedy=True)
    r0 = run_matches([base, make_policy(a.opponent, seed=9)], seeds, n, seat, batch_size=256)
    cfg = SearchConfig(rollouts=a.rollouts, top_k=a.top_k, horizon=a.horizon, skip_confident=a.skip_confident)
    srch = _TimedSearch(net, cfg, seed=1)
    t0 = time.time()
    r1 = run_matches([srch, make_policy(a.opponent, seed=9)], seeds, n, seat, batch_size=64)
    c0, c1 = r0.credit_of(0) * n, r1.credit_of(0) * n
    lat = np.array(srch.latencies) if srch.latencies else np.zeros(1)
    res = {
        "players": n,
        "checkpoint": a.checkpoint,
        "opponent": a.opponent,
        "matches": int(len(seeds)),
        "search_config": cfg.__dict__,
        "policy_only": summarize(c0, n),
        "search": summarize(c1, n),
        "paired_gain": paired_difference(c1, c0, n),
        "searched_decisions": srch.stats["searched"],
        "skipped_decisions": srch.stats["skipped"],
        "changed_action_rate": srch.stats["changed"] / max(1, srch.stats["searched"]),
        "latency_s": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                      "mean": float(lat.mean())},
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
