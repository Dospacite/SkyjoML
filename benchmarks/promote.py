"""Paired promotion test: candidate vs incumbent on identical dev deals, seats and opponents."""

import json
import sys

import torch

from skyjo_ml.evaluation.benchmark import paired_compare

torch.set_num_threads(1)
a, b, players, n_seeds, out = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4]), sys.argv[5]
res = paired_compare(a, b, players=players, opponents=("finish_tuned_2p",), n_seeds=n_seeds)
print(json.dumps(res, indent=1))
json.dump({"candidate": a, "incumbent": b, "n_seeds": n_seeds, "results": res}, open(out, "w"), indent=1)
