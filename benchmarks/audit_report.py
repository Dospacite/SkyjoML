"""Summarise decision audits (artifacts/analysis/{n}p_audit.parquet from evaluation/decision_audit.py)."""

import sys

import numpy as np
import pandas as pd

from skyjo_ml.game import obs_layout as L_NEW
from skyjo_ml.game.actions import A_REVEAL, A_TAKE
from skyjo_ml.game.rules import MIN_VALUE, N_SLOTS
from skyjo_ml.legacy import obs_layout_mp8 as L_OLD

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)


def layout(o):
    """Audits recorded before the 2-player specialisation use the legacy 8-seat observation layout."""
    return L_OLD if len(o) == L_OLD.OBS_SIZE else L_NEW


def board(o, r=0):
    L = layout(o)
    g = np.array(o[L.O_GRID + r * N_SLOTS: L.O_GRID + (r + 1) * N_SLOTS])
    rows = []
    for row in range(3):
        cells = []
        for c in range(4):
            x = g[c * 3 + row]
            cells.append("  #" if x == L.CODE_HIDDEN else "  ." if x == L.CODE_ABSENT else f"{x + MIN_VALUE:>3d}")
        rows.append(" ".join(cells))
    return rows


def enrich(d):
    obs = np.stack(d.obs.values)
    L = layout(obs[0])
    slot = np.where(d.chosen >= A_TAKE, (d.chosen - A_TAKE) % N_SLOTS, -1)
    target_code = np.where(slot >= 0, obs[np.arange(len(d)), L.O_GRID + np.maximum(slot, 0)], -1)
    targets_hidden = (slot >= 0) & ((d.chosen >= A_REVEAL) | (target_code == L.CODE_HIDDEN))
    d = d.copy()
    d["finishing_move"] = (d.my_hidden == 1) & targets_hidden
    d["could_finish"] = d.my_hidden == 1
    d["final_round"] = d["round"] == d.match_rounds - 1
    d["leading"] = d.my_total <= d.min_opp_total
    d["near_end"] = d.max_total >= 70
    d["significant"] = (d.regret > 2 * d.regret_se) & (d.regret > 0.02)
    return d


def summarize(d, by):
    g = d.groupby(by)
    t = g.agg(n=("regret", "size"), mean_regret=("regret", "mean"), se=("regret", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
              significant_rate=("significant", "mean"), total_regret=("regret", "sum"))
    t["share_of_regret"] = t.total_regret / d.regret.sum()
    return t.round(4)


def report(n):
    d = enrich(pd.read_parquet(f"artifacts/analysis/{n}p_audit.parquet"))
    print(f"\n=================== {n} players: {len(d)} audited decisions "
          f"({d.lost.sum()} from lost matches, {(~d.lost).sum()} from won)")
    print("mean regret (win credit per decision) by match outcome:")
    print(summarize(d, "lost").to_string())
    print("\nby chosen action kind:")
    print(summarize(d, "kind").to_string())
    print("\nby situation (lost matches only):")
    dl = d[d.lost]
    for col in ["finishing_move", "could_finish", "in_final_turns", "final_round", "leading", "near_end"]:
        print(f"-- {col}")
        print(summarize(dl, col).to_string())
    sig = d[d.significant]
    print(f"\nsignificant mistakes: {len(sig)} ({len(sig) / len(d):.1%}); chosen kind x recommended kind:")
    best_kind = sig.best_all_desc.str.split().str[0].map(
        {"draw": "draw", "take": "take", "keep": "keep", "discard": "reveal"})
    print(pd.crosstab(sig.kind, best_kind).to_string())
    print("\nlargest significant mistakes:")
    for _, r in sig.sort_values("regret", ascending=False).head(8).iterrows():
        top = "none" if r.top == 99 else r.top
        pend = "" if r.pending == 99 else f", drew {r.pending}"
        print(f"\n  regret {r.regret:.3f}±{r.regret_se:.3f} | round {r['round'] + 1}/{r.match_rounds} | totals me {r.my_total} "
              f"best opp {r.min_opp_total} | top discard {top}{pend} | hidden me {r.my_hidden}, opp min {r.min_opp_hidden}"
              f"{' | FINAL TURNS' if r.in_final_turns else ''}")
        print(f"  chose: {r.chosen_desc}  (p={r.prior_chosen:.2f})   better: {r.best_all_desc}   "
              f"Q {r.q_chosen:.3f} -> {r.q_best_all:.3f}")
        for line in board(r.obs):
            print("     " + line)
    return d


if __name__ == "__main__":
    for n in [int(x) for x in sys.argv[1:]] or [2, 4, 8]:
        report(n)
