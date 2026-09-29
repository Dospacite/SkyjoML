"""Warm start: imitate heuristic play and regress match outcomes.

Rationale: on a CPU-only budget, PPO from a random initialisation spends millions of decisions relearning
card-game mechanics. Cloning the strongest heuristic (with label smoothing so the policy keeps support on
every legal action) plus fitting the value head to Monte-Carlo terminal win credit gives PPO a sensible
starting point. This is a warm start only: every subsequent update optimises terminal win credit, and all
selection uses complete-match evaluation (PLAN.md section 6).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..agents.base import HeuristicBatchPolicy
from ..envs.batch import BatchEnv
from ..game.actions import N_ACTIONS
from ..game.rules import MAX_PLAYERS
from ..models.net import NetConfig, SkyjoNet
from .manifest import SEED_RANGES, build_manifest


def generate(n_decisions: int, teachers: list[str], weights: list[float], seed: int, n_envs: int = 512,
             player_probs: dict | None = None, teacher_params: dict | None = None):
    """Play complete matches between heuristics; return decisions with each seat's terminal win credit."""
    rng = np.random.default_rng(seed)
    teacher_params = teacher_params or {}
    pols = [HeuristicBatchPolicy(t, params=teacher_params.get(t), seed=seed + i, batch_size=n_envs)
            for i, t in enumerate(teachers)]
    pc = player_probs or {2: 1.0}
    pcv = np.array(list(pc.keys()))
    pcp = np.array(list(pc.values()), dtype=float)
    pcp /= pcp.sum()
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    env = BatchEnv(n_envs)
    ctrl = np.zeros((n_envs, MAX_PLAYERS), dtype=np.int64)
    next_seed = SEED_RANGES["train"][1] - 100_000_000 + seed * 10_000_000

    obs_l, mask_l, act_l, mid_l, seat_l = [], [], [], [], []
    match_id = np.zeros(n_envs, dtype=np.int64)
    next_mid = 0
    credit: dict[int, np.ndarray] = {}

    def reset(idx):
        nonlocal next_seed, next_mid
        ns = rng.choice(pcv, size=len(idx), p=pcp)
        env.reset(idx, np.arange(next_seed, next_seed + len(idx)), ns)
        next_seed += len(idx)
        for i in idx:
            ctrl[i] = rng.choice(len(pols), size=MAX_PLAYERS, p=w)
            match_id[i] = next_mid
            next_mid += 1

    reset(np.arange(n_envs))
    actions = np.full(n_envs, -1, dtype=np.int64)
    total = 0
    while total < n_decisions:
        obs, masks = env.observe()
        cur = env.cur
        c = ctrl[np.arange(n_envs), cur]
        actions[:] = -1
        for k, p in enumerate(pols):
            rows = np.flatnonzero(c == k)
            p.act_batch(obs, masks, rows, actions)
        obs_l.append(obs.astype(np.int16))
        mask_l.append(masks.copy())
        act_l.append(actions.copy())
        mid_l.append(match_id.copy())
        seat_l.append(cur.copy())
        total += n_envs
        env.step(actions)
        over = np.flatnonzero(env.over)
        if len(over):
            wc = env.win_credit()
            for i in over:
                credit[int(match_id[i])] = -np.ones(MAX_PLAYERS) if env.truncated[i] else wc[i].copy()
            reset(over)
    obs = np.concatenate(obs_l)
    masks = np.concatenate(mask_l)
    acts = np.concatenate(act_l)
    mids = np.concatenate(mid_l)
    seats = np.concatenate(seat_l)
    keep = np.array([m in credit and credit[m][0] >= 0 for m in mids])
    val = np.array([credit[m][s] if k else 0.0 for m, s, k in zip(mids, seats, keep, strict=True)], dtype=np.float32)
    return obs[keep], masks[keep], acts[keep], val[keep]


def train_bc(net: SkyjoNet, data, epochs: int, lr: float, minibatch: int, smoothing: float, seed: int,
             log=print) -> dict:
    obs, masks, acts, val = data
    N = len(acts)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_hold = min(20_000, N // 10)
    hold, tr = perm[:n_hold], perm[n_hold:]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps_total = epochs * (len(tr) // minibatch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps_total, pct_start=0.1)
    mask_t = torch.from_numpy(masks)
    acts_t = torch.from_numpy(acts)
    val_t = torch.from_numpy(val)

    def batch(idx):
        g, p, s = SkyjoNet.features(obs[idx].astype(np.int32))
        return (g, p, s), mask_t[idx], acts_t[idx], val_t[idx]

    def evaluate():
        net.eval()
        with torch.no_grad():
            f, m, a, v = batch(hold)
            logits, vh = net(None, m, f)
            acc = (logits.argmax(1) == a).float().mean().item()
            ce = torch.nn.functional.cross_entropy(logits, a).item()
            vmse = ((vh - v) ** 2).mean().item()
        net.train()
        return {"hold_acc": acc, "hold_ce": ce, "hold_vmse": vmse, "hold_v_var": float(v.var())}

    net.train()
    for ep in range(epochs):
        t0 = time.time()
        order = tr[rng.permutation(len(tr))]
        losses = []
        for s in range(0, len(order), minibatch):
            f, m, a, v = batch(order[s : s + minibatch])
            logits, vh = net(None, m, f)
            logp = torch.log_softmax(logits, 1)
            legal = m.float()
            smooth = legal / legal.sum(1, keepdim=True)
            target = (1 - smoothing) * torch.nn.functional.one_hot(a, N_ACTIONS).float() + smoothing * smooth
            pi_loss = -(target * logp.clamp_min(-1e4)).sum(1).mean()
            v_loss = ((vh - v) ** 2).mean()
            loss = pi_loss + v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append((pi_loss.item(), v_loss.item()))
        ev = evaluate()
        rec = {"epoch": ep + 1, "pi_loss": float(np.mean([x[0] for x in losses])),
               "v_loss": float(np.mean([x[1] for x in losses])), "seconds": round(time.time() - t0, 1), **ev}
        log(json.dumps(rec))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-params", default=None, help="JSON with tuned 'vector' for the finish heuristic")
    ap.add_argument("--out", default="artifacts/bc/bc_init.pt")
    ap.add_argument("--net", default="{}", help="JSON NetConfig overrides")
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    t0 = time.time()
    teachers = ["finish", "column", "greedy"]
    tp = {}
    if a.teacher_params:
        with open(a.teacher_params) as f:
            tp["finish"] = np.array(json.load(f)["vector"], dtype=np.float64)
    data = generate(a.decisions, teachers, [0.7, 0.2, 0.1], a.seed, teacher_params=tp)
    print(f"generated {len(data[2])} decisions in {time.time() - t0:.0f}s", flush=True)
    cfg = NetConfig(**json.loads(a.net))
    net = SkyjoNet(cfg)
    ev = train_bc(net, data, a.epochs, a.lr, a.minibatch, a.smoothing, a.seed, log=lambda s: print(s, flush=True))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "net_config": cfg.to_dict(),
                "manifest": build_manifest(vars(a), {"holdout": ev})}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
