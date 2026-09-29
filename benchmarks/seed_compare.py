"""Dev-set comparison of warm start and fine-tuned seeds (paired deals, seat rotation, block bootstrap)."""

import json
import sys

import torch

from skyjo_ml.evaluation.benchmark import cell_metrics, make_policy, seed_block
from skyjo_ml.evaluation.tournament import run_matches, seat_rotation_schedule

torch.set_num_threads(1)
cands = sys.argv[1].split(",")
players = [int(x) for x in sys.argv[2].split(",")]
opps = sys.argv[3].split(",")
n_seeds = int(sys.argv[4])
out = {}
for n in players:
    for o in opps:
        for c in cands:
            seeds, seat = seat_rotation_schedule(seed_block("dev", n, n_seeds, offset=300_000), n, [0] + [1] * (n - 1))
            r = run_matches([make_policy(c, greedy=True), make_policy(o, seed=5)], seeds, n, seat, batch_size=1024)
            m = cell_metrics(r, 0, n)
            k = f"{n}p vs {o} | {c}"
            out[k] = {"credit_x_n": m["credit_x_n"], "rank": m["mean_rank"], "round_score": m["mean_round_score"]}
            cm = m["credit_x_n"]
            print(f"{k}: {cm['mean']:.3f} [{cm['ci_lo']:.3f}, {cm['ci_hi']:.3f}]", flush=True)
json.dump(out, open(sys.argv[5], "w"), indent=1)
