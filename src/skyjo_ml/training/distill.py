"""Warm start for the two-player model: distil the legacy multi-player release into the 2-player network.

The legacy model (trained on 2-8 player tables) is the strongest available 2-player policy, far above the
heuristics. We generate two-player matches whose seats are played by the legacy model (sampled at temperature 1
or arg-max) and by heuristics, record the legacy model's full action distribution at *every* decision (also in
states reached by heuristic seats, for coverage), and train the new network on soft cross-entropy to those
distributions plus Monte-Carlo terminal win credit for the value head. This is a warm start only; PPO follows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..envs.batch import BatchEnv
from ..evaluation.benchmark import make_policy
from ..game.actions import N_ACTIONS
from ..game.rules import MAX_PLAYERS
from ..legacy.policy import convert, load_legacy
from ..models.net import NetConfig, SkyjoNet
from .manifest import SEED_RANGES, build_manifest


def generate(n_decisions: int, seed: int, n_envs: int = 512, seat_mix=None):
    rng = np.random.default_rng(seed)
    teacher = load_legacy()
    kinds = ["legacy_sample", "legacy_greedy", "finish_tuned", "finish", "column"]
    probs = np.array(seat_mix or [0.5, 0.2, 0.15, 0.1, 0.05])
    heur = {k: make_policy(k, seed=seed + i) for i, k in enumerate(kinds[2:])}
    env = BatchEnv(n_envs)
    ctrl = np.zeros((n_envs, MAX_PLAYERS), dtype=np.int64)
    mid = np.zeros(n_envs, dtype=np.int64)
    next_mid = 0
    next_seed = SEED_RANGES["train"][1] - 200_000_000 + seed * 10_000_000
    gen = torch.Generator().manual_seed(seed)

    def reset(idx):
        nonlocal next_seed, next_mid
        env.reset(idx, np.arange(next_seed, next_seed + len(idx)), np.full(len(idx), 2))
        next_seed += len(idx)
        for i in idx:
            ctrl[i] = rng.choice(len(kinds), size=MAX_PLAYERS, p=probs / probs.sum())
            mid[i] = next_mid
            next_mid += 1

    reset(np.arange(n_envs))
    O, Mk, P, MID, SEAT = [], [], [], [], []
    credit: dict[int, np.ndarray] = {}
    acts = np.full(n_envs, -1, dtype=np.int64)
    total = 0
    t0 = time.time()
    while total < n_decisions:
        obs, masks = env.observe()
        cur = env.cur
        c = ctrl[np.arange(n_envs), cur]
        with torch.inference_mode():
            logits, _ = teacher(torch.from_numpy(convert(obs)), torch.from_numpy(masks))
            p = torch.softmax(logits, 1)
        acts[:] = -1
        samp = np.flatnonzero(c == 0)
        if len(samp):
            acts[samp] = torch.multinomial(p[samp], 1, generator=gen).squeeze(1).numpy()
        grd = np.flatnonzero(c == 1)
        acts[grd] = p[grd].argmax(1).numpy()
        for k, name in enumerate(kinds[2:], start=2):
            heur[name].act_batch(obs, masks, np.flatnonzero(c == k), acts)
        O.append(obs.astype(np.int16))
        Mk.append(masks.copy())
        P.append(p.numpy().astype(np.float16))
        MID.append(mid.copy())
        SEAT.append(cur.copy())
        total += n_envs
        env.step(acts)
        over = np.flatnonzero(env.over)
        if len(over):
            wc = env.win_credit()
            for i in over:
                credit[int(mid[i])] = None if env.truncated[i] else wc[i].copy()
            reset(over)
    mids = np.concatenate(MID)
    seats = np.concatenate(SEAT)
    keep = np.array([credit.get(int(m)) is not None for m in mids])
    val = np.array([credit[int(m)][s] if k else 0.0 for m, s, k in zip(mids, seats, keep, strict=True)], dtype=np.float32)
    print(f"generated {keep.sum()} decisions from {len(credit)} matches in {time.time() - t0:.0f}s", flush=True)
    return (np.concatenate(O)[keep], np.concatenate(Mk)[keep], np.concatenate(P)[keep], val[keep])


def train(net: SkyjoNet, data, epochs: int, lr: float, minibatch: int, seed: int) -> dict:
    obs, masks, probs, val = data
    N = len(val)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_hold = min(20_000, N // 10)
    hold, tr = perm[:n_hold], perm[n_hold:]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(tr) // minibatch + 1),
                                                pct_start=0.1)

    def batch(idx):
        feats = SkyjoNet.features(obs[idx].astype(np.int32))
        m = torch.from_numpy(masks[idx])
        tgt = torch.from_numpy(probs[idx].astype(np.float32)) * m
        tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-8)
        return feats, m, tgt, torch.from_numpy(val[idx])

    ev = {}
    for ep in range(epochs):
        t0 = time.time()
        net.train()
        order = tr[rng.permutation(len(tr))]
        for s in range(0, len(order), minibatch):
            f, m, tgt, v = batch(order[s: s + minibatch])
            logits, vh = net(None, m, f)
            logp = torch.log_softmax(logits, 1).clamp_min(-1e4)
            loss = -(tgt * logp).sum(1).mean() + ((vh - v) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            f, m, tgt, v = batch(hold)
            logits, vh = net(None, m, f)
            logp = torch.log_softmax(logits, 1).clamp_min(-1e4)
            ent = -(tgt * torch.log(tgt.clamp_min(1e-12))).sum(1).mean().item()
            ev = {"epoch": ep + 1, "hold_kl": float(-(tgt * logp).sum(1).mean().item() - ent),
                  "hold_argmax_agree": float((logits.argmax(1) == tgt.argmax(1)).float().mean()),
                  "hold_vmse": float(((vh - v) ** 2).mean()), "hold_v_var": float(v.var()),
                  "seconds": round(time.time() - t0, 1)}
        print(json.dumps(ev), flush=True)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--net", default="{}")
    ap.add_argument("--out", default="artifacts/distill/distill_2p.pt")
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    data = generate(a.decisions, a.seed)
    cfg = NetConfig(**json.loads(a.net))
    net = SkyjoNet(cfg)
    ev = train(net, data, a.epochs, a.lr, a.minibatch, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "net_config": cfg.to_dict(),
                "manifest": build_manifest(vars(a), {"holdout": ev, "teacher": "legacy multi-player release"})}, out)
    print("saved", out, N_ACTIONS)


if __name__ == "__main__":
    main()
