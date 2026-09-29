"""Where does the model lose? Analysis of artifacts/analysis/{n}p_*.parquet (see evaluation/loss_analysis.py)."""

import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def load(n):
    base = f"artifacts/analysis/{n}p_"
    return (pd.read_parquet(base + "matches.parquet"), pd.read_parquet(base + "rounds.parquet"),
            pd.read_parquet(base + "decisions.parquet"))


def report(n):
    m, r, d = load(n)
    m = m[~m.truncated].copy()
    m["lost"] = m.credit == 0
    m["won"] = m.credit == 1
    print(f"\n==================== {n} players: {len(m)} matches, credit {m.credit.mean():.3f} (parity {1/n:.3f}), "
          f"lost {m.lost.mean():.1%}")

    # ---- A. how losses look
    L = m[m.lost]
    print("\n[A] lost matches")
    print(f"  margin to winner: median {L.margin.median():.0f}, <=5 pts {np.mean(L.margin <= 5):.1%}, "
          f"<=10 {np.mean(L.margin <= 10):.1%}, >30 {np.mean(L.margin > 30):.1%}")
    print(f"  model itself crossed 100: {L.cand_crossed.mean():.1%} (in wins: {m[m.won].cand_crossed.mean():.1%})")
    print(f"  rank distribution: {L['rank'].value_counts(normalize=True).sort_index().round(3).to_dict()}")
    print(f"  rounds per match: lost {L.rounds.mean():.2f} vs won {m[m.won].rounds.mean():.2f}")

    # ---- B. round-level comparison: model vs heuristic seats (same matches)
    r = r[r.match.isin(m.match)].copy()
    r["n_fin"] = r.groupby(["match", "round"]).finisher.transform("sum")
    r["table_min"] = r.groupby(["match", "round"]).round_score.transform("min")
    r["gap_to_min"] = r.round_score - r.table_min
    r["someone_else_finished"] = ~r.finisher
    agg = r.groupby("is_cand").agg(
        round_score=("round_score", "mean"),
        gap_to_round_min=("gap_to_min", "mean"),
        finish_rate=("finisher", "mean"),
        doubled_rate=("doubled", "mean"),
        blowup_ge25=("round_score", lambda x: np.mean(x >= 25)),
        hidden_at_end=("hidden_at_end", "mean"),
    )
    print("\n[B] per round, model (True) vs heuristic seats (False)")
    print(agg.round(3).to_string())
    fin = r[r.finisher]
    print("  when finishing: doubled share", fin.groupby("is_cand").doubled.mean().round(3).to_dict(),
          " mean score when finishing", fin.groupby("is_cand").round_score.mean().round(2).to_dict())
    caught = r[~r.finisher]
    print("  when someone else finished: hidden cards", caught.groupby("is_cand").hidden_at_end.mean().round(2).to_dict(),
          " their true value", caught.groupby("is_cand").hidden_value_at_end.mean().round(2).to_dict(),
          " visible sum", caught.groupby("is_cand").visible_at_end.mean().round(2).to_dict())

    # ---- C. decompose the deficit of lost matches by round type
    c = r[r.is_cand].merge(m[["match", "lost", "won"]], on="match")
    # reference seat: the eventual winner when the model lost, otherwise the best-placed opponent
    last = r.sort_values("round").groupby(["match", "seat"]).tail(1)
    opp_last = last[~last.is_cand]
    wseat = opp_last.loc[opp_last.groupby("match").total_after.idxmin(), ["match", "seat"]].rename(columns={"seat": "wseat"})
    w = r.merge(wseat, on="match")
    w = w[w.seat == w.wseat][["match", "round", "round_score"]].rename(columns={"round_score": "winner_rs"})
    c = c.merge(w, on=["match", "round"])
    c["deficit"] = c.round_score - c.winner_rs

    def cat(row):
        if row.doubled:
            return "model finished & doubled"
        if row.finisher:
            return "model finished, not doubled"
        if row.hidden_at_end >= 3:
            return "caught by other finisher (>=3 hidden)"
        return "caught by other finisher (<=2 hidden)"

    c["type"] = c.apply(cat, axis=1)
    lost_c = c[c.lost]
    tab = lost_c.groupby("type").agg(rounds=("deficit", "size"), total_deficit=("deficit", "sum"),
                                     mean_deficit=("deficit", "mean"))
    tab["share_of_deficit"] = tab.total_deficit / lost_c.deficit.sum()
    print("\n[C] lost matches: points lost to the eventual winner, by round type")
    print(tab.sort_values("total_deficit", ascending=False).round(3).to_string())
    won_c = c[c.won]
    print("  same round types in WON matches (points vs best opponent; negative = model gained):")
    print(won_c.groupby("type").deficit.agg(["size", "mean"]).round(2).to_string())
    print("  share of model rounds by type, lost vs won:")
    print(pd.concat([lost_c.type.value_counts(normalize=True).rename("lost"),
                     won_c.type.value_counts(normalize=True).rename("won")], axis=1).round(3).to_string())

    # ---- D. final round of lost matches
    fr = c.sort_values("round").groupby("match").tail(1)
    tot_before = r.assign(before=r.total_after - r.round_score)
    tb = tot_before.sort_values("round").groupby(["match", "seat"]).tail(1)
    tb["min_before"] = tb.groupby("match").before.transform("min")
    lead = tb[tb.is_cand][["match", "before", "min_before"]]
    lead["was_leading"] = lead.before == lead.min_before
    fr = fr.merge(lead, on="match")
    Lr = fr[fr.lost]
    print("\n[D] final round of lost matches")
    print(f"  model was (co-)leading before the final round: {Lr.was_leading.mean():.1%} of losses "
          f"({Lr.was_leading.sum()} matches = blown leads)")
    bl = Lr[Lr.was_leading]
    if len(bl):
        print("  blown leads by final-round type:", bl.type.value_counts(normalize=True).round(3).to_dict())
        print(f"  blown leads: mean final-round score {bl.round_score.mean():.1f} vs winner {bl.winner_rs.mean():.1f}")
    print(f"  in wins, model led before final round: {fr[fr.won].was_leading.mean():.1%}")

    # ---- E. value head: calibration and where the estimate collapses
    d = d.merge(m[["match", "credit", "lost"]], on="match")
    d["vbin"] = pd.cut(d.value, [-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2])
    cal = d.groupby("vbin", observed=True).agg(n=("credit", "size"), predicted=("value", "mean"), actual=("credit", "mean"))
    print("\n[E] value-head calibration (all decisions)")
    print(cal.round(3).to_string())
    d = d.sort_values(["match", "decision_idx"])
    d["dv"] = d.groupby("match").value.diff()
    d["prev_phase"] = d.groupby("match").phase.shift()
    d["prev_round"] = d.groupby("match")["round"].shift()
    d["gap"] = np.select(
        [d.prev_round != d["round"], (d.prev_phase == 1) & (d.phase == 2)],
        ["across round end", "own draw (chance)"], "opponents' turns")
    ld = d[d.lost & d.dv.notna()]
    print("  lost matches: value lost by where it happened (sum of negative changes, share):")
    neg = ld[ld.dv < 0].groupby("gap").dv.sum()
    print((neg / neg.sum()).round(3).to_dict(), " totals:", neg.round(1).to_dict())
    return m, r, d, c


if __name__ == "__main__":
    for n in [int(x) for x in sys.argv[1:]] or [2, 4, 8]:
        report(n)
