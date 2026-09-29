"""Two-player benchmark suite with deal-block bootstrap confidence intervals.

Layout for each (player count, opponent) cell: ``n_seeds`` deal seeds, each played ``n`` times with the
candidate rotated through every seat and the opponent in all other seats. The block (seed + rotations) is
the bootstrap unit. Parity is 1/n win credit; we report ``credit x n`` so that 1.0 means parity at any
table size. Per-match rows are written as Parquet.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..agents.base import HeuristicBatchPolicy
from ..game.rules import MAX_PLAYERS, ruleset_hash
from ..training.manifest import SEED_RANGES, build_manifest, schema_hash
from .stats import paired_difference, summarize
from .tournament import MatchResults, run_matches, seat_rotation_schedule

HEURISTICS = ("random", "greedy", "column", "finish")


def make_policy(spec, greedy: bool = True, seed: int = 0, tuned: str | None = "configs/heuristic_finish_tuned.json"):
    """``spec``: heuristic, tuned finish variant, legacy release, checkpoint path, or policy."""
    if not isinstance(spec, str):
        return spec
    if spec in HEURISTICS:
        return HeuristicBatchPolicy(spec, seed=seed)
    if spec == "legacy":
        from ..legacy.policy import LegacyBatchPolicy

        return LegacyBatchPolicy()
    if spec in {"finish_tuned", "finish_tuned_2p"}:
        path = "configs/heuristic_finish_tuned_2p.json" if spec == "finish_tuned_2p" else tuned
        with open(path) as f:
            p = HeuristicBatchPolicy("finish", params=np.array(json.load(f)["vector"]), seed=seed)
        p.name = spec
        return p
    from ..agents.neural import NeuralBatchPolicy, load_model

    net, _ = load_model(spec)
    return NeuralBatchPolicy(net, Path(spec).stem, greedy=greedy, seed=seed)


def seed_block(split: str, n_players: int, n_seeds: int, offset: int = 0) -> np.ndarray:
    base = SEED_RANGES[split][0] + 20_000_000 + 1_000_000 * n_players + offset
    return np.arange(base, base + n_seeds)


def cell_metrics(r: MatchResults, cand: int, n: int) -> dict:
    sel = r.seating == cand
    credit = r.credit_of(cand)
    ok = ~r.truncated
    my_score = (r.scores * sel).sum(1) / sel.sum(1)
    opp = np.where(~sel & (r.seating >= 0), r.scores, np.nan)
    best_opp = np.nanmin(opp, axis=1)
    # rank 1 = lowest total (ties share the better rank)
    rank = 1 + np.nansum(opp < my_score[:, None], axis=1)
    strict = (credit == 1.0).astype(float)
    shared = ((credit > 0) & (credit < 1)).astype(float)
    rounds = np.maximum(r.rounds, 1)
    out = {
        "credit_x_n": summarize(credit * n, n),
        "win_credit": summarize(credit, n),
        "strict_win_rate": float(strict.mean()),
        "shared_win_rate": float(shared.mean()),
        "mean_rank": float(rank.mean()),
        "score_minus_best_opp": float((my_score - best_opp).mean()),
        "mean_round_score": float(((r.round_scores_sum * sel).sum(1) / rounds).mean()),
        "finisher_penalty_per_round": float(((r.doubled * sel).sum(1) / rounds).mean()),
        "finishes_per_round": float(((r.finishes * sel).sum(1) / rounds).mean()),
        "rounds_per_match": float(r.rounds.mean()),
        "decisions_per_match": float(r.decisions.mean()),
        "truncated": int((~ok).sum()),
    }
    return out


def run_benchmark(candidate, players=(2,), n_seeds: int = 500, split: str = "dev",
                  out: str | None = None, greedy: bool = True, opponents=HEURISTICS, mixed: bool = True,
                  mirror: bool = True) -> dict:
    t0 = time.time()
    cand = make_policy(candidate, greedy=greedy, seed=11)
    rows = []
    summary: dict = {}
    for n in players:
        seeds_all = seed_block(split, n, n_seeds)
        for oi, opp_name in enumerate(opponents):
            opp = make_policy(opp_name, seed=100 + oi)
            seeds, seat = seat_rotation_schedule(seeds_all, n, [0] + [1] * (n - 1))
            r = run_matches([cand, opp], seeds, n, seat, batch_size=1024)
            summary[f"{n}p vs {opp_name}"] = cell_metrics(r, 0, n)
            rows.append(_rows(r, 0, n, f"vs {opp_name}"))
        if mixed and n >= 3:
            opps = [make_policy(o, seed=200 + i) for i, o in enumerate(opponents) if o != "random"]
            lineup = [0] + [1 + (k % len(opps)) for k in range(n - 1)]
            seeds, seat = seat_rotation_schedule(seeds_all, n, lineup)
            r = run_matches([cand, *opps], seeds, n, seat, batch_size=1024)
            summary[f"{n}p mixed"] = cell_metrics(r, 0, n)
            rows.append(_rows(r, 0, n, "mixed"))
        if mirror:
            # all seats play the candidate: per-seat credit reveals seat bias (parity 1/n each)
            seeds = seeds_all[: max(50, n_seeds // 4)]
            seat = np.full((len(seeds), MAX_PLAYERS), -1)
            seat[:, :n] = 0
            r = run_matches([cand], seeds, n, seat, batch_size=1024)
            ok = ~r.truncated
            summary[f"{n}p mirror seat credit x n"] = (r.win_credit[ok, :n].mean(0) * n).round(3).tolist()
    res = {
        "candidate": str(candidate),
        "split": split,
        "n_seeds": n_seeds,
        "greedy": greedy,
        "summary": {k: _compact(v) for k, v in summary.items()},
        "details": summary,
        "manifest": build_manifest({"players": list(players), "opponents": list(opponents)},
                                   {"ruleset_hash": ruleset_hash(), "schema_hash": schema_hash()}),
        "seconds": round(time.time() - t0, 1),
    }
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p.with_suffix(".json"), "w") as f:
            json.dump(res, f, indent=1)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            cols = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
            pq.write_table(pa.table(cols), p.with_suffix(".parquet"))
        except ImportError:
            pass
    return res


def _compact(v):
    if isinstance(v, dict) and "credit_x_n" in v:
        c = v["credit_x_n"]
        return {"credit_x_n": round(c["mean"], 3), "ci95": [round(c["ci_lo"], 3), round(c["ci_hi"], 3)],
                "strict_win": round(v["strict_win_rate"], 3), "rank": round(v["mean_rank"], 2),
                "round_score": round(v["mean_round_score"], 2), "trunc": v["truncated"]}
    return v


def _rows(r: MatchResults, cand: int, n: int, cell: str) -> dict:
    M = len(r.seeds)
    sel = r.seating == cand
    return {
        "cell": np.array([cell] * M),
        "n_players": np.full(M, n),
        "seed": r.seeds,
        "cand_seat": sel.argmax(1),
        "cand_credit": r.credit_of(cand),
        "cand_score": (r.scores * sel).sum(1),
        "min_score": np.where(r.seating >= 0, r.scores, 10**9).min(1),
        "rounds": r.rounds,
        "decisions": r.decisions,
        "truncated": r.truncated,
    }


def paired_compare(a, b, players=(2,), opponents=("finish",), n_seeds: int = 500, split: str = "dev",
                   greedy: bool = True) -> dict:
    """Paired candidate-vs-incumbent comparison on identical seeds, seats and opponents."""
    pa_ = make_policy(a, greedy=greedy, seed=1)
    pb_ = make_policy(b, greedy=greedy, seed=1)
    out = {}
    for n in players:
        seeds_all = seed_block(split, n, n_seeds, offset=500_000)
        for opp_name in opponents:
            res = []
            for pol in (pa_, pb_):
                opp = make_policy(opp_name, seed=77)
                seeds, seat = seat_rotation_schedule(seeds_all, n, [0] + [1] * (n - 1))
                r = run_matches([pol, opp], seeds, n, seat, batch_size=1024)
                res.append(r.credit_of(0) * n)
            d = paired_difference(res[0], res[1], n)
            out[f"{n}p vs {opp_name}"] = {"a": float(res[0].mean()), "b": float(res[1].mean()),
                                          "diff": round(d["mean"], 4), "ci95": [round(d["ci_lo"], 4),
                                                                                round(d["ci_hi"], 4)]}
    return out
