"""Random-search tuning of heuristic parameters on development seeds only.

A candidate parameter vector plays one seat against the current incumbent in all other seats, with paired
seeds and full seat rotation. Candidates that beat the incumbent on the search seeds are re-checked on a
fresh confirmation slice of development seeds before being accepted.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..agents.base import HeuristicBatchPolicy
from ..agents.heuristics import PARAM_NAMES, PRESETS
from ..training.manifest import SEED_RANGES
from .tournament import run_matches, seat_rotation_schedule

LOW = np.array([0.0, -1.5, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1e-3, -1.0])
HIGH = np.array([1.5, 0.5, 1.0, 5.0, 12.0, 1.0, 80.0, 1.0, 1e-3, 1.0])


def score(params, incumbent, n_players, seeds):
    cand = HeuristicBatchPolicy("finish", params=params, seed=1)
    inc = HeuristicBatchPolicy("finish", params=incumbent, seed=2)
    s, seat = seat_rotation_schedule(seeds, n_players, [0] + [1] * (n_players - 1))
    r = run_matches([cand, inc], s, n_players, seat, batch_size=1024)
    c = r.credit_of(0) * n_players  # 1.0 = parity
    blocks = c.reshape(len(seeds), n_players).mean(1)
    return float(blocks.mean()), float(blocks.std(ddof=1) / np.sqrt(len(blocks)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--players", type=int, nargs="+", default=[2])
    ap.add_argument("--out", default="configs/heuristic_finish_tuned_2p.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    inc = PRESETS["finish"][1].copy()
    base = SEED_RANGES["dev"][0] + 50_000_000
    confirm_base = base + 10_000_000
    history = []
    for it in range(a.iters):
        # perturb the incumbent (local search) or sample globally
        if rng.random() < 0.7:
            cand = inc + rng.normal(0, 0.15, size=len(inc)) * (HIGH - LOW)
        else:
            cand = rng.uniform(LOW, HIGH)
        cand = np.clip(cand, LOW, HIGH)
        seeds = np.arange(base + it * a.seeds, base + (it + 1) * a.seeds)
        res = [score(cand, inc, n, seeds + 1000 * n) for n in a.players]
        m = float(np.mean([r[0] for r in res]))
        se = float(np.sqrt(np.sum([r[1] ** 2 for r in res])) / len(res))
        accepted = False
        if m - 2 * se > 1.0:
            cseeds = np.arange(confirm_base + it * a.seeds, confirm_base + (it + 1) * a.seeds)
            cres = [score(cand, inc, n, cseeds + 1000 * n) for n in a.players]
            cm = float(np.mean([r[0] for r in cres]))
            cse = float(np.sqrt(np.sum([r[1] ** 2 for r in cres])) / len(cres))
            if cm - cse > 1.0:
                inc = cand
                accepted = True
        history.append({"iter": it, "mean_rel_credit": m, "se": se, "accepted": accepted,
                        "params": dict(zip(PARAM_NAMES, cand.round(4).tolist(), strict=True))})
        print(json.dumps(history[-1]), flush=True)
    out = {"params": dict(zip(PARAM_NAMES, inc.tolist(), strict=True)), "vector": inc.tolist(),
           "players": a.players, "seeds_per_eval": a.seeds, "seed_range": "dev", "history": history}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
