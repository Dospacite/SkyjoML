"""Decision audit: how much win credit does each of the model's decisions give away?

1. Play matches (model arg-max in one seat, a fixed heuristic elsewhere, dev seeds) and keep the raw observation
   of a random sample of the model's decisions.
2. For sampled decisions from lost and from won matches, evaluate the policy's top-k legal actions with
   belief-sampled rollouts in which the model plays its own seat and the *actual opponent heuristic* plays the
   others (``RootMCSearch`` with ``opponent_policy``), to the end of the current round (end of match in the final
   round) and scored by win credit / value head.
3. Regret = Q(best) - Q(chosen) with the best action picked on one half of the sampled worlds and the regret
   measured on the other half, which removes the optimistic bias of taking a max over noisy estimates.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..envs.batch import BatchEnv
from ..game import obs_layout as L
from ..game.actions import A_DRAW, A_KEEP, A_REVEAL, A_TAKE, describe
from ..game.rules import MAX_PLAYERS, MIN_VALUE, N_SLOTS, N_VALUES
from ..search.root_mc import RootMCSearch, SearchConfig
from .benchmark import make_policy, seed_block
from .tournament import seat_rotation_schedule

VALUES = np.arange(MIN_VALUE, MIN_VALUE + N_VALUES)


def play_and_sample(net, opponent, n, n_seeds, offset, p_keep, rng):
    seeds, seating = seat_rotation_schedule(seed_block("dev", n, n_seeds, offset=offset), n, [0] + [1] * (n - 1))
    M = len(seeds)
    B = min(512, M)
    env = BatchEnv(B)
    slot = np.full(B, -1, dtype=np.int64)
    nxt = 0
    kept_obs, kept_mask, kept_match, kept_act = [], [], [], []
    credit = np.zeros(M)
    rounds = np.zeros(M, dtype=np.int64)
    trunc = np.zeros(M, dtype=bool)

    def fill(idx):
        nonlocal nxt
        take = idx[: max(0, min(len(idx), M - nxt))]
        if len(take):
            ms = np.arange(nxt, nxt + len(take))
            nxt += len(take)
            slot[take] = ms
            env.reset(take, seeds[ms], np.full(len(take), n))
        slot[idx[len(take):]] = -1

    fill(np.arange(B))
    actions = np.full(B, -1, dtype=np.int64)
    while True:
        active = np.flatnonzero(slot >= 0)
        if len(active) == 0:
            break
        obs, masks = env.observe()
        ctrl = seating[slot[active], env.cur[active]]
        actions[:] = -1
        crow = active[ctrl == 0]
        if len(crow):
            with torch.inference_mode():
                logits, _ = net(torch.from_numpy(obs[crow]), torch.from_numpy(masks[crow]))
            a = logits.argmax(1).numpy()
            actions[crow] = a
            keep = (rng.random(len(crow)) < p_keep) & (obs[crow, L.O_PHASE] != L.PH_SETUP)
            if keep.any():
                kept_obs.append(obs[crow[keep]].copy())
                kept_mask.append(masks[crow[keep]].copy())
                kept_match.append(slot[crow[keep]].copy())
                kept_act.append(a[keep])
        orow = active[ctrl != 0]
        opponent.act_batch(obs, masks, orow, actions)
        env.step(actions)
        over = active[env.over[active]]
        if len(over):
            wc = env.win_credit()
            for i in over:
                m = slot[i]
                cs = int(np.flatnonzero(seating[m, :n] == 0)[0])
                trunc[m] = bool(env.truncated[i])
                credit[m] = 0.0 if trunc[m] else wc[i, cs]
                rounds[m] = env.states[i, 6] + 1
            fill(over)
    return (np.concatenate(kept_obs), np.concatenate(kept_mask), np.concatenate(kept_match),
            np.concatenate(kept_act), credit, rounds, trunc)


def context(o: np.ndarray) -> dict:
    grid = o[L.O_GRID: L.O_GRID + MAX_PLAYERS * N_SLOTS].reshape(MAX_PLAYERS, N_SLOTS)
    n = o[L.O_NPLAYERS]
    pool = np.maximum(o[L.O_POOL: L.O_POOL + N_VALUES], 0).astype(float)
    eh = float((pool * VALUES).sum() / max(pool.sum(), 1))
    vis = grid < N_VALUES
    hid = grid == L.CODE_HIDDEN
    est = np.where(vis, grid + MIN_VALUE, 0).sum(1) + hid.sum(1) * eh
    scores = o[L.O_SCORES: L.O_SCORES + MAX_PLAYERS]
    return {
        "phase": int(o[L.O_PHASE]),
        "round": int(o[L.O_ROUND]),
        "my_hidden": int(hid[0].sum()),
        "my_est": float(est[0]),
        "min_opp_est": float(est[1:n].min()),
        "min_opp_hidden": int(hid[1:n].sum(1).min()),
        "my_total": int(scores[0]),
        "min_opp_total": int(scores[1:n].min()),
        "max_total": int(scores[:n].max()),
        "in_final_turns": bool(o[L.O_FINISHER] >= 0),
        "top": int(o[L.O_TOP] + MIN_VALUE) if o[L.O_TOP] >= 0 else 99,
        "pending": int(o[L.O_PENDING] + MIN_VALUE) if o[L.O_PENDING] >= 0 else 99,
    }


def kind(a: int) -> str:
    if a == A_DRAW:
        return "draw"
    if A_TAKE <= a < A_KEEP:
        return "take"
    if A_KEEP <= a < A_REVEAL:
        return "keep"
    return "reveal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/skyjo_policy.pt")
    ap.add_argument("--opponent", default="finish_tuned_2p")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--offset", type=int, default=950_000)
    ap.add_argument("--p-keep", type=float, default=0.1)
    ap.add_argument("--n-lost", type=int, default=1500)
    ap.add_argument("--n-won", type=int, default=700)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--out", default="artifacts/analysis")
    a = ap.parse_args()
    torch.set_num_threads(1)
    from ..agents.neural import load_model

    net, _ = load_model(a.checkpoint)
    rng = np.random.default_rng(a.players)
    opp = make_policy(a.opponent, seed=31)
    t0 = time.time()
    obs, masks, match, acts, credit, rounds, trunc = play_and_sample(net, opp, a.players, a.seeds, a.offset,
                                                                    a.p_keep, rng)
    lost = credit[match] == 0
    ok = ~trunc[match]
    idx_l = rng.permutation(np.flatnonzero(lost & ok))[: a.n_lost]
    idx_w = rng.permutation(np.flatnonzero(~lost & ok))[: a.n_won]
    print(f"{a.players}p: played {len(credit)} matches (credit {credit.mean():.3f}), sampled {len(obs)} decisions; "
          f"auditing {len(idx_l)} lost + {len(idx_w)} won ({time.time() - t0:.0f}s)", flush=True)

    rollout_opp = make_policy(a.opponent, seed=77)
    search = RootMCSearch(net, SearchConfig(horizon=-1, rollout_greedy=True), seed=5, opponent_policy=rollout_opp)
    rows = []
    for j, i in enumerate(np.concatenate([idx_l, idx_w])):
        o, m, chosen = obs[i], masks[i], int(acts[i])
        with torch.inference_mode():
            logits, v = net(torch.from_numpy(o[None]), torch.from_numpy(m[None]))
        prior = torch.softmax(logits, 1)[0].numpy()
        legal = np.flatnonzero(m)
        cands = legal[np.argsort(-prior[legal])][: a.top_k]
        if len(cands) < 2:
            continue
        ws = rng.integers(1 << 62, size=a.worlds)
        q = search._evaluate(o, cands, ws)  # [A, W]
        h = a.worlds // 2
        qa = q[:, :h].mean(1)
        ci = int(np.flatnonzero(cands == chosen)[0])
        best_a = int(np.argmax(qa))
        diff_b = q[best_a, h:] - q[ci, h:]
        best_all = int(np.argmax(q.mean(1)))
        rows.append({
            "players": a.players, "match": int(match[i]), "lost": bool(lost[i]), "match_rounds": int(rounds[match[i]]),
            "chosen": chosen, "chosen_desc": describe(chosen), "kind": kind(chosen), "prior_chosen": float(prior[chosen]),
            "value": float(v[0]), "n_cands": len(cands),
            "regret": float(diff_b.mean()), "regret_se": float(diff_b.std(ddof=1) / np.sqrt(len(diff_b))),
            "best_split": int(cands[best_a]), "best_all": int(cands[best_all]),
            "best_all_desc": describe(int(cands[best_all])),
            "q_chosen": float(q[ci].mean()), "q_best_all": float(q[best_all].mean()),
            "obs": o.tolist(),
            **context(o),
        })
        if (j + 1) % 100 == 0:
            print(f"  audited {j + 1} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, out / f"{a.players}p_audit.parquet")
    print(f"done: {len(rows)} decisions audited in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
