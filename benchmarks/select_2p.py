"""Compare the three two-player PPO seeds on a separate rotated dev deal block."""

import argparse
import json
from pathlib import Path

import torch

from skyjo_ml.evaluation.benchmark import make_policy, seed_block
from skyjo_ml.evaluation.stats import paired_difference, summarize
from skyjo_ml.evaluation.tournament import run_matches, seat_rotation_schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=600)
    ap.add_argument("--offset", type=int, default=750_000)
    ap.add_argument("--out", default="artifacts/reports/selection_2p_dev.json")
    a = ap.parse_args()
    torch.set_num_threads(1)
    seeds, seats = seat_rotation_schedule(seed_block("dev", 2, a.seeds, offset=a.offset), 2, [0, 1])
    legacy = make_policy("legacy")
    report = {"split": "dev", "offset": a.offset, "seeds": a.seeds, "opponent": "legacy", "candidates": {}}
    credits = {}
    for s in (1, 2, 3):
        name = f"ppo2p_s{s}"
        path = f"artifacts/runs/{name}/best_dev.pt"
        result = run_matches([make_policy(path), legacy], seeds, 2, seats, batch_size=512)
        credits[name] = result.credit_of(0) * 2
        report["candidates"][name] = {
            "checkpoint": path,
            "credit_x2": summarize(credits[name], 2),
            "truncated": int(result.truncated.sum()),
        }
        print(name, json.dumps(report["candidates"][name]), flush=True)
    report["vs_s2"] = {name: paired_difference(credits[name], credits["ppo2p_s2"], 2)
                       for name in ("ppo2p_s1", "ppo2p_s3")}
    report["selected"] = max(credits, key=lambda name: credits[name].mean())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("selected", report["selected"], flush=True)


if __name__ == "__main__":
    main()
