"""Instrumented matches for diagnosing where a candidate loses win credit.

Plays the candidate (arg-max) in one seat against a fixed opponent in all other seats, with seat rotation, and
logs three Parquet tables:

* ``matches``: final totals, win credit, rank, margin to the winner, who crossed 100;
* ``rounds``: every player's round score, finisher/doubling, and each player's board at the moment the round
  ended (hidden cards and their true values — used only for post-hoc analysis, never as agent input);
* ``decisions``: every candidate decision with its value estimate, action type and board context.

Development seeds only; the release range stays sealed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..envs.batch import (
    HIDDEN,
    S_LAST_DBL,
    S_LAST_FIN,
    S_LAST_RS,
    S_NP,
    S_ROUND,
    S_ROUND_TURNS,
    S_SCORES,
    S_VAL,
    S_VIS,
    VISIBLE,
    BatchEnv,
)
from ..game import obs_layout as L
from ..game.actions import A_DRAW, A_KEEP, A_REVEAL, A_TAKE
from ..game.rules import MAX_PLAYERS, MIN_VALUE, N_SLOTS, N_VALUES
from .benchmark import make_policy, seed_block
from .tournament import seat_rotation_schedule

VALUES = np.arange(MIN_VALUE, MIN_VALUE + N_VALUES)


def _action_kind(a: np.ndarray) -> np.ndarray:
    k = np.full(a.shape, "setup", dtype=object)
    k[a == A_DRAW] = "draw"
    k[(a >= A_TAKE) & (a < A_KEEP)] = "take"
    k[(a >= A_KEEP) & (a < A_REVEAL)] = "keep"
    k[a >= A_REVEAL] = "reveal"
    return k


def collect(checkpoint: str, opponent: str, n_players: int, n_seeds: int, offset: int, batch: int = 512):
    torch.set_num_threads(1)
    from ..agents.neural import load_model

    net, _ = load_model(checkpoint)
    opp_policy = make_policy(opponent, seed=21)
    seeds, seating = seat_rotation_schedule(seed_block("dev", n_players, n_seeds, offset=offset), n_players,
                                            [0] + [1] * (n_players - 1))
    M = len(seeds)
    B = min(batch, M)
    env = BatchEnv(B)
    slot_match = np.full(B, -1, dtype=np.int64)
    nxt = 0

    dec: dict[str, list] = {k: [] for k in ["match", "round", "round_turn", "phase", "action", "kind", "slot", "value",
                                             "prob", "top", "pending", "my_hidden", "my_visible_sum", "my_est",
                                             "eh", "my_total", "min_opp_total", "min_opp_est", "min_opp_hidden",
                                             "in_final", "decision_idx"]}
    rnd: dict[str, list] = {k: [] for k in ["match", "round", "seat", "is_cand", "round_score", "finisher",
                                             "doubled", "hidden_at_end", "hidden_value_at_end", "visible_at_end",
                                             "total_after", "decisions_in_round"]}
    mat: dict[str, list] = {k: [] for k in ["match", "seed", "cand_seat", "credit", "rank", "cand_total",
                                             "winner_total", "margin", "rounds", "cand_crossed", "opp_crossed",
                                             "truncated"]}
    dec_count = np.zeros(M, dtype=np.int64)

    def fill(idx):
        nonlocal nxt
        take = idx[: max(0, min(len(idx), M - nxt))]
        if len(take):
            ms = np.arange(nxt, nxt + len(take))
            nxt += len(take)
            slot_match[take] = ms
            env.reset(take, seeds[ms], np.full(len(take), n_players))
        slot_match[idx[len(take):]] = -1

    fill(np.arange(B))
    actions = np.full(B, -1, dtype=np.int64)
    t0 = time.time()
    while True:
        active = np.flatnonzero(slot_match >= 0)
        if len(active) == 0:
            break
        obs, masks = env.observe()
        cur = env.cur
        ctrl = seating[slot_match[active], cur[active]]
        actions[:] = -1
        crow = active[ctrl == 0]
        orow = active[ctrl != 0]
        if len(crow):
            with torch.inference_mode():
                logits, v = net(torch.from_numpy(obs[crow]), torch.from_numpy(masks[crow]))
                p = torch.softmax(logits, 1)
            a = logits.argmax(1).numpy()
            actions[crow] = a
            o = obs[crow]
            grid = o[:, L.O_GRID: L.O_GRID + MAX_PLAYERS * N_SLOTS].reshape(-1, MAX_PLAYERS, N_SLOTS)
            pool = np.maximum(o[:, L.O_POOL: L.O_POOL + N_VALUES], 0).astype(float)
            eh = (pool * VALUES).sum(1) / np.maximum(pool.sum(1), 1)
            vis = grid < N_VALUES
            hid = grid == L.CODE_HIDDEN
            vsum = np.where(vis, grid + MIN_VALUE, 0).sum(2)
            est = vsum + hid.sum(2) * eh[:, None]
            valid = np.arange(MAX_PLAYERS)[None, :] < o[:, L.O_NPLAYERS][:, None]
            oppm = valid.copy()
            oppm[:, 0] = False
            scores = o[:, L.O_SCORES: L.O_SCORES + MAX_PLAYERS]
            ms = slot_match[crow]
            dec["match"] += ms.tolist()
            dec["round"] += o[:, L.O_ROUND].tolist()
            dec["round_turn"] += o[:, L.O_ROUND_TURNS].tolist()
            dec["phase"] += o[:, L.O_PHASE].tolist()
            dec["action"] += a.tolist()
            dec["kind"] += _action_kind(a).tolist()
            dec["slot"] += np.where(a >= A_TAKE, (a - A_TAKE) % N_SLOTS, -1).tolist()
            dec["value"] += v.numpy().tolist()
            dec["prob"] += p.gather(1, torch.from_numpy(a)[:, None]).squeeze(1).numpy().tolist()
            dec["top"] += np.where(o[:, L.O_TOP] >= 0, o[:, L.O_TOP] + MIN_VALUE, 99).tolist()
            dec["pending"] += np.where(o[:, L.O_PENDING] >= 0, o[:, L.O_PENDING] + MIN_VALUE, 99).tolist()
            dec["my_hidden"] += hid[:, 0].sum(1).tolist()
            dec["my_visible_sum"] += vsum[:, 0].tolist()
            dec["my_est"] += est[:, 0].tolist()
            dec["eh"] += eh.tolist()
            dec["my_total"] += scores[:, 0].tolist()
            dec["min_opp_total"] += np.where(oppm, scores, 10**6).min(1).tolist()
            dec["min_opp_est"] += np.where(oppm, est, 10**6).min(1).tolist()
            dec["min_opp_hidden"] += np.where(oppm, hid.sum(2), 99).min(1).tolist()
            dec["in_final"] += (o[:, L.O_FINISHER] >= 0).tolist()
            dec["decision_idx"] += dec_count[ms].tolist()
            dec_count[ms] += 1
        if len(orow):
            opp_policy.act_batch(obs, masks, orow, actions)
        pre = env.states[active].copy()
        env.step(actions)
        ended = active[env.round_ended[active]]
        for i in ended:
            m = slot_match[i]
            st_pre = pre[np.searchsorted(active, i)]
            n = st_pre[S_NP]
            rs = env.states[i, S_LAST_RS: S_LAST_RS + n]
            fin = env.states[i, S_LAST_FIN]
            dbl = env.states[i, S_LAST_DBL]
            for s in range(n):
                visb = st_pre[S_VIS + s * N_SLOTS: S_VIS + (s + 1) * N_SLOTS]
                val = st_pre[S_VAL + s * N_SLOTS: S_VAL + (s + 1) * N_SLOTS]
                h = visb == HIDDEN
                rnd["match"].append(int(m))
                rnd["round"].append(int(st_pre[S_ROUND]))
                rnd["seat"].append(s)
                rnd["is_cand"].append(bool(seating[m, s] == 0))
                rnd["round_score"].append(int(rs[s]))
                rnd["finisher"].append(bool(fin == s))
                rnd["doubled"].append(bool(fin == s and dbl))
                rnd["hidden_at_end"].append(int(h.sum()))
                rnd["hidden_value_at_end"].append(int(val[h].sum()))
                rnd["visible_at_end"].append(int(val[visb == VISIBLE].sum()))
                rnd["total_after"].append(int(env.states[i, S_SCORES + s]))
                rnd["decisions_in_round"].append(int(st_pre[S_ROUND_TURNS]))
        over = active[env.over[active]]
        if len(over):
            wc = env.win_credit()
            for i in over:
                m = slot_match[i]
                n = n_players
                tot = env.states[i, S_SCORES: S_SCORES + n]
                cs = int(np.flatnonzero(seating[m, :n] == 0)[0])
                trunc = bool(env.truncated[i])
                others = np.delete(tot, cs)
                mat["match"].append(int(m))
                mat["seed"].append(int(seeds[m]))
                mat["cand_seat"].append(cs)
                mat["credit"].append(0.0 if trunc else float(wc[i, cs]))
                mat["rank"].append(int(1 + (others < tot[cs]).sum()))
                mat["cand_total"].append(int(tot[cs]))
                mat["winner_total"].append(int(tot.min()))
                mat["margin"].append(int(tot[cs] - others.min()))
                mat["rounds"].append(int(env.states[i, S_ROUND]) + 1)
                mat["cand_crossed"].append(bool(tot[cs] >= 100))
                mat["opp_crossed"].append(bool((others >= 100).any()))
                mat["truncated"].append(trunc)
            fill(over)
    return pa.table(mat), pa.table(rnd), pa.table(dec), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/skyjo_policy.pt")
    ap.add_argument("--opponent", default="finish_tuned_2p")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=1500)
    ap.add_argument("--offset", type=int, default=900_000)
    ap.add_argument("--out", default="artifacts/analysis")
    a = ap.parse_args()
    m, r, d, dt = collect(a.checkpoint, a.opponent, a.players, a.seeds, a.offset)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in [("matches", m), ("rounds", r), ("decisions", d)]:
        pq.write_table(t, out / f"{a.players}p_{name}.parquet")
    print(f"{a.players}p: {m.num_rows} matches, {d.num_rows} decisions in {dt:.0f}s")


if __name__ == "__main__":
    main()
