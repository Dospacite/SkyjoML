"""Masked PPO over complete Skyjo matches with a frozen-opponent league.

Key properties (PLAN.md section 6):

* Rounds are not episode boundaries: a trajectory is one seat's decisions over a whole match.
* Final reward is terminal win credit (1/k for k tied winners), gamma = 1. An optional relative round-score
  term is annealed to zero early in training (warm start only).
* Each transition is aligned with the same seat's next decision. A seat's last transition in a rollout is
  carried into the next rollout (instead of being bootstrapped from a guess), so rewards are never dropped.
* Only learner-controlled seats enter the loss; frozen snapshots and heuristics provide context.
"""

from __future__ import annotations

import copy
import json
import math
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from numba import njit

from ..agents.neural import NeuralBatchPolicy
from ..envs.batch import S_LAST_RS, BatchEnv
from ..evaluation.benchmark import make_policy
from ..game import obs_layout as L
from ..game.actions import A_DRAW, A_KEEP, A_REVEAL, A_SETUP, N_ACTIONS, SETUP_SAME_COLUMN
from ..game.rules import MAX_PLAYERS
from ..models.net import NetConfig, SkyjoNet
from .manifest import SEED_RANGES, build_manifest


@dataclass
class TrainConfig:
    run_name: str = "run"
    seed: int = 1
    total_decisions: int = 10_000_000
    n_envs: int = 256
    decisions_per_update: int = 32_768
    epochs: int = 4
    minibatch: int = 4096
    lr: float = 3e-4
    lr_final: float = 3e-5
    weight_decay: float = 1e-4
    clip: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 1.0
    ent_coef: float = 0.01
    ent_coef_final: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    # reward: round-score warm start annealed to pure win credit
    shaping_start: float = 1.0
    shaping_anneal_decisions: int = 3_000_000
    shaping_scale: float = 20.0
    # table composition
    player_count_weights: dict = field(
        default_factory=lambda: {"2": 1.0}
    )
    p_selfplay_table: float = 0.3
    p_two_learner_seats: float = 0.2
    opp_recent: float = 0.5
    opp_old: float = 0.25
    opp_heuristic: float = 0.25
    heuristics: list = field(default_factory=lambda: ["finish", "column", "greedy"])
    heuristic_weights: list = field(default_factory=lambda: [0.6, 0.3, 0.1])
    snapshot_every: int = 10
    n_recent: int = 3
    # evaluation / checkpoints
    eval_every: int = 10
    eval_seeds: int = 150
    eval_players: list = field(default_factory=lambda: [2])
    eval_opponent: str = "finish"
    checkpoint_every: int = 10
    out_dir: str = "artifacts/runs"
    init_from: str | None = None
    extra_snapshots: list = field(default_factory=list)  # frozen policies from other runs, added as "old"
    value_warmup_updates: int = 0  # first updates train only the value head (useful after a warm start)
    torch_threads: int = 1  # small matmuls on this class of CPU are fastest single-threaded
    net: dict = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str | Path, **overrides) -> TrainConfig:
        with open(path, "rb") as f:
            d = tomllib.load(f)
        d.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**d)


@njit(cache=True)
def compute_gae(traj, val, rew, done, trunc, n_traj, gamma, lam, adv, valid):
    next_v = np.zeros(n_traj)
    next_adv = np.zeros(n_traj)
    seen = np.zeros(n_traj, np.bool_)
    for i in range(len(traj) - 1, -1, -1):
        t = traj[i]
        if not seen[t]:
            seen[t] = True
            if done[i]:
                adv[i] = rew[i] - val[i]
                valid[i] = True
            elif trunc[i]:
                adv[i] = rew[i]  # watchdog truncation: bootstrap with own value
                valid[i] = True
            else:
                adv[i] = 0.0  # continues in the next rollout (carried)
                valid[i] = False
        else:
            delta = rew[i] + gamma * next_v[t] - val[i]
            adv[i] = delta + gamma * lam * next_adv[t]
            valid[i] = True
        next_v[t] = val[i]
        next_adv[t] = adv[i]


class Buffer:
    FIELDS = {
        "obs": (np.int32, (L.OBS_SIZE,)),
        "mask": (np.bool_, (N_ACTIONS,)),
        "act": (np.int64, ()),
        "logp": (np.float32, ()),
        "val": (np.float32, ()),
        "rew": (np.float32, ()),
        "done": (np.bool_, ()),
        "trunc": (np.bool_, ()),
        "traj": (np.int64, ()),
    }

    def __init__(self, cap: int):
        self.cap = cap
        for k, (dt, sh) in self.FIELDS.items():
            setattr(self, k, np.zeros((cap, *sh), dtype=dt))
        self.n = 0

    def ensure(self, extra: int) -> None:
        if self.n + extra > self.cap:
            new = max(self.cap * 2, self.n + extra)
            for k in self.FIELDS:
                a = getattr(self, k)
                b = np.zeros((new, *a.shape[1:]), dtype=a.dtype)
                b[: self.n] = a[: self.n]
                setattr(self, k, b)
            self.cap = new


class League:
    """Frozen snapshots of the learner plus heuristic baselines."""

    def __init__(self, run_dir: Path, cfg: TrainConfig, net_cfg: NetConfig):
        self.dir = run_dir / "league"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.net_cfg = net_cfg
        self.snapshots: list[dict] = []  # {"path", "update", "decisions"}

    def add(self, net: SkyjoNet, update: int, decisions: int) -> None:
        p = self.dir / f"snap_{update:05d}.pt"
        torch.save({"model": net.state_dict(), "net_config": self.net_cfg.to_dict()}, p)
        self.snapshots.append({"path": str(p), "update": update, "decisions": decisions})

    def load(self, path: str) -> SkyjoNet:
        net = SkyjoNet(self.net_cfg)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        return net.eval()

    def to_dict(self) -> dict:
        return {"snapshots": self.snapshots}


class PPOTrainer:
    def __init__(self, cfg: TrainConfig, resume: str | None = None):
        self.cfg = cfg
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.net_cfg = NetConfig(**cfg.net)
        self.net = SkyjoNet(self.net_cfg)
        if cfg.init_from:
            ck = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
            self.net.load_state_dict(ck["model"])
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-5)
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.league = League(self.run_dir, cfg, self.net_cfg)
        self.update = 0
        self.best_eval = -1.0
        self.decisions = 0
        self.matches = 0
        self.elapsed = 0.0
        self.next_seed = SEED_RANGES["train"][0] + cfg.seed * 50_000_000
        self.next_traj = 0
        pc = cfg.player_count_weights
        self.pc_values = np.array([int(k) for k in pc], dtype=np.int64)
        self.pc_probs = np.array(list(pc.values()), dtype=np.float64)
        self.pc_probs /= self.pc_probs.sum()
        if resume:
            self._load(resume)
        known = {x["path"] for x in self.league.snapshots}
        extra = [{"path": p, "update": -1, "decisions": -1} for p in cfg.extra_snapshots if p not in known]
        self.league.snapshots = extra + self.league.snapshots  # oldest end of the list = "old" pool
        man = build_manifest(asdict(cfg), {"net_config": self.net_cfg.to_dict(), "resumed_from": resume})
        with open(self.run_dir / f"manifest_{int(time.time())}.json", "w") as f:
            json.dump(man, f, indent=1)
        self.metrics_f = open(self.run_dir / "metrics.jsonl", "a")

        # environments and per-seat bookkeeping
        B = cfg.n_envs
        self.env = BatchEnv(B)
        self.ctrl = np.full((B, MAX_PLAYERS), -1, dtype=np.int64)
        self.traj_id = np.full((B, MAX_PLAYERS), -1, dtype=np.int64)
        self.last_tr = np.full((B, MAX_PLAYERS), -1, dtype=np.int64)
        self.buf = Buffer(cfg.decisions_per_update + 4 * B * MAX_PLAYERS)
        self.opponents: list = []  # index k -> ctrl value k + 1
        self._refresh_opponents()
        self._reset_envs(np.arange(B))

    # ------------------------------------------------------------------ league / tables
    def _refresh_opponents(self) -> None:
        cfg = self.cfg
        opps: list = []
        kinds: list[str] = []
        snaps = self.league.snapshots
        if snaps:
            recent = snaps[-cfg.n_recent :]
            pick = recent[self.rng.integers(len(recent))]
            opps.append(NeuralBatchPolicy(self.league.load(pick["path"]), f"snap{pick['update']}",
                                          seed=int(self.rng.integers(1 << 30))))
            kinds.append("recent")
            older = snaps[: -cfg.n_recent]
            if older:
                pick = older[self.rng.integers(len(older))]
                opps.append(NeuralBatchPolicy(self.league.load(pick["path"]), f"snap{pick['update']}",
                                              seed=int(self.rng.integers(1 << 30))))
                kinds.append("old")
        for h in cfg.heuristics:
            opps.append(make_policy(h, seed=int(self.rng.integers(1 << 30))))
            kinds.append("heuristic:" + h)
        self.opponents = opps
        self.opp_kinds = kinds

    def _sample_opponent(self, allow_self: bool = True) -> int:
        """Return a controller id (0 = learner) for one non-learner seat."""
        cfg = self.cfg
        kinds = self.opp_kinds
        r = self.rng.random()
        recent = [i for i, k in enumerate(kinds) if k == "recent"]
        old = [i for i, k in enumerate(kinds) if k == "old"]
        heur = [i for i, k in enumerate(kinds) if k.startswith("heuristic")]
        if r < cfg.opp_recent:
            pool = recent
        elif r < cfg.opp_recent + cfg.opp_old:
            pool = old or recent
        else:
            pool = []
        if pool:
            return pool[self.rng.integers(len(pool))] + 1
        if not recent and r < cfg.opp_recent + cfg.opp_old and allow_self:
            return 0  # no snapshots yet: current policy
        w = np.array(cfg.heuristic_weights, dtype=np.float64)[: len(heur)]
        return heur[self.rng.choice(len(heur), p=w / w.sum())] + 1

    def _reset_envs(self, idx: np.ndarray) -> None:
        if len(idx) == 0:
            return
        cfg = self.cfg
        ns = self.rng.choice(self.pc_values, size=len(idx), p=self.pc_probs)
        seeds = np.arange(self.next_seed, self.next_seed + len(idx))
        self.next_seed += len(idx)
        self.env.reset(idx, seeds, ns)
        for i, n in zip(idx, ns, strict=True):
            self.ctrl[i] = -1
            if self.rng.random() < cfg.p_selfplay_table:
                self.ctrl[i, :n] = 0
            else:
                k = 2 if (n > 2 and self.rng.random() < cfg.p_two_learner_seats) else 1
                seats = self.rng.choice(n, size=k, replace=False)
                for s in range(n):
                    self.ctrl[i, s] = 0 if s in seats else self._sample_opponent()
            for s in range(n):
                if self.ctrl[i, s] == 0:
                    self.traj_id[i, s] = self.next_traj
                    self.next_traj += 1
                else:
                    self.traj_id[i, s] = -1
            self.traj_id[i, n:] = -1
            self.last_tr[i] = -1

    # ------------------------------------------------------------------ rollout
    def shaping(self) -> float:
        c = self.cfg
        if c.shaping_anneal_decisions <= 0:
            return 0.0
        return c.shaping_start * max(0.0, 1.0 - self.decisions / c.shaping_anneal_decisions)

    @torch.inference_mode()
    def collect(self) -> dict:
        cfg = self.cfg
        env = self.env
        buf = self.buf
        B = cfg.n_envs
        # carry each open trajectory's last transition to the front of the buffer
        carry = self.last_tr[self.last_tr >= 0]
        if len(carry):
            for k in Buffer.FIELDS:
                a = getattr(buf, k)
                a[: len(carry)] = a[carry]
            remap = np.full(buf.n + 1, -1, dtype=np.int64)
            remap[carry] = np.arange(len(carry))
            self.last_tr = np.where(self.last_tr >= 0, remap[np.maximum(self.last_tr, 0)], -1)
        buf.n = len(carry)
        n_carry = buf.n
        beta = self.shaping()
        actions = np.full(B, -1, dtype=np.int64)
        stats = {"matches": 0, "truncated": 0, "win_credit": [], "round_rel": [], "steps": 0}
        t0 = time.time()
        self.net.eval()
        while buf.n - n_carry < cfg.decisions_per_update:
            obs, masks = env.observe()
            cur = env.cur
            ctrl = self.ctrl[np.arange(B), cur]
            actions[:] = -1
            rows = np.flatnonzero(ctrl == 0)
            if len(rows):
                ot = torch.from_numpy(obs[rows])
                mt = torch.from_numpy(masks[rows])
                logits, v = self.net(ot, mt)
                logp_all = torch.log_softmax(logits, 1)
                a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
                lp = logp_all.gather(1, a[:, None]).squeeze(1)
                an = a.numpy()
                actions[rows] = an
                k = len(rows)
                buf.ensure(k)
                sl = slice(buf.n, buf.n + k)
                buf.obs[sl] = obs[rows]
                buf.mask[sl] = masks[rows]
                buf.act[sl] = an
                buf.logp[sl] = lp.numpy()
                buf.val[sl] = v.numpy()
                buf.rew[sl] = 0.0
                buf.done[sl] = False
                buf.trunc[sl] = False
                seats = cur[rows]
                buf.traj[sl] = self.traj_id[rows, seats]
                self.last_tr[rows, seats] = np.arange(buf.n, buf.n + k)
                buf.n += k
            for oi, pol in enumerate(self.opponents):
                r = np.flatnonzero(ctrl == oi + 1)
                if len(r):
                    pol.act_batch(obs, masks, r, actions)
            env.step(actions)
            stats["steps"] += 1

            ended = np.flatnonzero(env.round_ended)
            if beta > 0 and len(ended):
                for i in ended:
                    n = env.n_players[i]
                    rs = env.states[i, S_LAST_RS : S_LAST_RS + n].astype(np.float64)
                    rel = (rs - (rs.sum() - rs) / (n - 1)) / cfg.shaping_scale
                    for s in range(n):
                        if self.ctrl[i, s] == 0 and self.last_tr[i, s] >= 0:
                            buf.rew[self.last_tr[i, s]] -= beta * rel[s]
                    stats["round_rel"].append(rel[self.ctrl[i, :n] == 0].mean())
            over = np.flatnonzero(env.over)
            if len(over):
                wc = env.win_credit()
                for i in over:
                    n = env.n_players[i]
                    tr = bool(env.truncated[i])
                    for s in range(n):
                        j = self.last_tr[i, s]
                        if self.ctrl[i, s] == 0 and j >= 0:
                            if tr:
                                buf.trunc[j] = True
                            else:
                                buf.rew[j] += wc[i, s]
                                buf.done[j] = True
                    stats["matches"] += 1
                    stats["truncated"] += int(tr)
                    if not tr:
                        learner = self.ctrl[i, :n] == 0
                        if not learner.all():
                            stats["win_credit"].append((wc[i, :n][learner].mean(), n))
                self._reset_envs(over)
        n_new = buf.n - n_carry
        self.decisions += n_new
        self.matches += stats["matches"]
        stats["collect_time"] = time.time() - t0
        stats["n_new"] = n_new
        stats["n_carry"] = n_carry
        return stats

    # ------------------------------------------------------------------ update
    def lr_now(self) -> float:
        c = self.cfg
        frac = min(1.0, self.decisions / max(1, c.total_decisions))
        return c.lr_final + 0.5 * (c.lr - c.lr_final) * (1 + math.cos(math.pi * frac))

    def ent_now(self) -> float:
        c = self.cfg
        frac = min(1.0, self.decisions / max(1, c.total_decisions))
        return c.ent_coef + (c.ent_coef_final - c.ent_coef) * frac

    def learn(self) -> dict:
        cfg = self.cfg
        buf = self.buf
        N = buf.n
        traj = buf.traj[:N]
        uniq, compact = np.unique(traj, return_inverse=True)
        adv = np.zeros(N)
        valid = np.zeros(N, dtype=np.bool_)
        compute_gae(compact.astype(np.int64), buf.val[:N].astype(np.float64), buf.rew[:N].astype(np.float64),
                    buf.done[:N], buf.trunc[:N], len(uniq), cfg.gamma, cfg.gae_lambda, adv, valid)
        ret = adv + buf.val[:N]
        idx = np.flatnonzero(valid)
        g_all, p_all, s_all = SkyjoNet.features(buf.obs[:N])
        mask_t = torch.from_numpy(buf.mask[:N])
        act_t = torch.from_numpy(buf.act[:N])
        logp_old = torch.from_numpy(buf.logp[:N])
        val_old = torch.from_numpy(buf.val[:N])
        adv_t = torch.from_numpy(adv.astype(np.float32))
        ret_t = torch.from_numpy(ret.astype(np.float32))
        for g in self.opt.param_groups:
            g["lr"] = self.lr_now()
        ent_coef = self.ent_now()
        self.net.train()
        logs = {"pi_loss": [], "v_loss": [], "entropy": [], "kl": [], "clipfrac": []}
        stop = False
        for _ in range(cfg.epochs):
            perm = idx[self.rng.permutation(len(idx))]
            # fold a short tail into the previous minibatch (a 1-sample batch has undefined std)
            starts = list(range(0, len(perm), cfg.minibatch))
            if len(starts) > 1 and len(perm) - starts[-1] < cfg.minibatch // 4:
                starts.pop()
            for k, s in enumerate(starts):
                end = starts[k + 1] if k + 1 < len(starts) else len(perm)
                mb = torch.from_numpy(perm[s:end])
                logits, v = self.net(None, mask_t[mb], (g_all[mb], p_all[mb], s_all[mb]))
                logp_all = torch.log_softmax(logits, 1)
                lp = logp_all.gather(1, act_t[mb][:, None]).squeeze(1)
                ratio = torch.exp(lp - logp_old[mb])
                a = adv_t[mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v_clip = val_old[mb] + (v - val_old[mb]).clamp(-cfg.clip, cfg.clip)
                v_loss = 0.5 * torch.max((v - ret_t[mb]) ** 2, (v_clip - ret_t[mb]) ** 2).mean()
                p = logp_all.exp()
                ent = -(p * logp_all.clamp_min(-1e4)).sum(1).mean()
                if self.update < cfg.value_warmup_updates:
                    loss = cfg.vf_coef * v_loss
                else:
                    loss = pi_loss + cfg.vf_coef * v_loss - ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    logs.setdefault("skipped_nonfinite", []).append(1.0)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    lr_ = lp - logp_old[mb]
                    kl = ((torch.exp(lr_) - 1) - lr_).mean().item()
                logs["pi_loss"].append(pi_loss.item())
                logs["v_loss"].append(v_loss.item())
                logs["entropy"].append(ent.item())
                logs["kl"].append(kl)
                logs["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                if kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) for k, v in logs.items()}
        yv = ret[idx]
        out["explained_var"] = float(1 - np.var(yv - buf.val[:N][idx]) / (np.var(yv) + 1e-8))
        out["early_stop"] = stop
        out["n_train"] = int(len(idx))
        # per-phase action frequencies
        acts = buf.act[:N][idx]
        ph = buf.obs[:N][idx, L.O_PHASE]
        sel = ph == L.PH_SELECT
        res = ph == L.PH_RESOLVE
        stp = ph == L.PH_SETUP
        out["freq_draw"] = float((acts[sel] == A_DRAW).mean()) if sel.any() else 0.0
        out["freq_keep"] = float(((acts[res] >= A_KEEP) & (acts[res] < A_REVEAL)).mean()) if res.any() else 0.0
        out["freq_setup_samecol"] = (
            float(SETUP_SAME_COLUMN[acts[stp] - A_SETUP].mean()) if stp.any() else 0.0
        )
        return out

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, n_seeds: int | None = None) -> dict:
        from ..evaluation.tournament import run_matches, seat_rotation_schedule

        n_seeds = n_seeds or self.cfg.eval_seeds
        pol = NeuralBatchPolicy(copy.deepcopy(self.net).eval(), "cand", greedy=True)
        res = {}
        for n in self.cfg.eval_players:
            for h in [self.cfg.eval_opponent]:
                opp = make_policy(h, seed=7)
                base = SEED_RANGES["dev"][0] + 1000 * n
                seeds, seat = seat_rotation_schedule(np.arange(base, base + n_seeds), n, [0] + [1] * (n - 1))
                r = run_matches([pol, opp], seeds, n, seat, batch_size=512)
                c = r.credit_of(0)
                blocks = c.reshape(n_seeds, n).mean(1)
                res[f"eval_{n}p_vs_{h}"] = float(c.mean())
                res[f"eval_{n}p_vs_{h}_se"] = float(blocks.std(ddof=1) / math.sqrt(n_seeds))
        return res

    # ------------------------------------------------------------------ persistence
    def save(self, name: str = "latest.pt") -> Path:
        p = self.run_dir / name
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.opt.state_dict(),
                "net_config": self.net_cfg.to_dict(),
                "train_config": asdict(self.cfg),
                "update": self.update,
                "decisions": self.decisions,
                "matches": self.matches,
                "elapsed": self.elapsed,
                "next_seed": self.next_seed,
                "next_traj": self.next_traj,
                "best_eval": self.best_eval,
                "league": self.league.to_dict(),
                "np_rng": self.rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "note": "resume restarts in-flight matches; open trajectories are not persisted",
            },
            p,
        )
        return p

    def _load(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ck["model"])
        if ck.get("optimizer") is not None:
            self.opt.load_state_dict(ck["optimizer"])
        self.update = ck["update"]
        self.decisions = ck["decisions"]
        self.matches = ck["matches"]
        self.elapsed = ck["elapsed"]
        self.next_seed = ck["next_seed"]
        self.next_traj = ck["next_traj"]
        self.best_eval = ck.get("best_eval", -1.0)
        self.league.snapshots = ck["league"]["snapshots"]
        self.rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])

    def log(self, rec: dict) -> None:
        self.metrics_f.write(json.dumps(rec) + "\n")
        self.metrics_f.flush()

    # ------------------------------------------------------------------ main loop
    def train(self) -> None:
        cfg = self.cfg
        best = self.best_eval
        while self.decisions < cfg.total_decisions:
            t0 = time.time()
            st = self.collect()
            lg = self.learn()
            self.update += 1
            dt = time.time() - t0
            self.elapsed += dt
            wc = st["win_credit"]
            rec = {
                "update": self.update,
                "decisions": self.decisions,
                "matches": self.matches,
                "elapsed": round(self.elapsed, 1),
                "dps": round(st["n_new"] / dt, 1),
                "collect_s": round(st["collect_time"], 1),
                "shaping": round(self.shaping(), 4),
                "lr": self.lr_now(),
                "truncated": st["truncated"],
                "mixed_win_credit_x_n": float(np.mean([w * n for w, n in wc])) if wc else None,
                "round_rel": float(np.mean(st["round_rel"])) if st["round_rel"] else None,
                "opponents": [o.name for o in self.opponents],
                **lg,
            }
            if self.update % cfg.snapshot_every == 0:
                self.league.add(self.net, self.update, self.decisions)
            if self.update % cfg.eval_every == 0:
                ev = self.evaluate()
                rec.update(ev)
                key = next(iter(ev))
                if ev[key] > best:
                    best = self.best_eval = ev[key]
                    self.save("best_dev.pt")
            if self.update % cfg.checkpoint_every == 0:
                self.save()
            self.log(rec)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                              if k != "opponents"}), flush=True)
            self._refresh_opponents()
            self._rebind_ctrl()
        self.save()
        self.save("final.pt")

    def _rebind_ctrl(self) -> None:
        """Opponent slots changed: re-sample controllers for opponent seats of in-flight matches.

        Seats keep their learner/opponent role so learner trajectories stay intact; only which frozen
        policy plays an opponent seat changes (opponent histories carry no memory in this model).
        """
        B = self.cfg.n_envs
        for i in range(B):
            n = self.env.n_players[i]
            for s in range(n):
                if self.ctrl[i, s] > 0:
                    self.ctrl[i, s] = self._sample_opponent(allow_self=False)
