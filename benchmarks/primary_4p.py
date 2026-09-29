"""Primary release cell: 4 players, candidate vs 3 x finish_tuned, 2,500 sealed deals x 4 seat rotations."""

import json
import sys

import torch

from skyjo_ml.evaluation.benchmark import cell_metrics, make_policy, seed_block
from skyjo_ml.evaluation.tournament import run_matches, seat_rotation_schedule

torch.set_num_threads(1)
ck, n_seeds, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seeds, seat = seat_rotation_schedule(seed_block("release", 4, n_seeds, offset=500_000), 4, [0, 1, 1, 1])
r = run_matches([make_policy(ck), make_policy("finish_tuned", seed=3)], seeds, 4, seat, batch_size=1024)
m = cell_metrics(r, 0, 4)
print(json.dumps(m, indent=1))
json.dump({"checkpoint": ck, "matches": int(len(seeds)), "metrics": m}, open(out, "w"), indent=1)
