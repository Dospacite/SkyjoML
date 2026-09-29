"""Policy/value network over raw observations (featurized by ``models.features``).

Structure (PLAN.md section 6, scaled for CPU-only training):

* per-player summary tokens (visible board statistics, completable pairs, cumulative score, relative seat,
  finishing status) -> shared player encoder -> optional self-attention across players with padding masks;
* global features include exact public card accounting (belief marginals) and match context;
* policy logits are position-conditioned: one shared head scores take/keep/reveal for each of the actor's
  slots from slot features (value, column mates, what placing the top discard / drawn card would do) plus
  context; setup pairs are scored by column relation; a scalar head predicts the actor's expected terminal
  win credit.

The network is equivariant to column and within-column permutations by construction. ``forward`` takes the
raw observation tensor, so an exported model needs nothing but observations and legal masks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ..game.actions import N_ACTIONS, SETUP_SAME_COLUMN
from .features import N_GLOBAL, N_PLAYER, N_SLOT, featurize

NEG = -1e9


@dataclass
class NetConfig:
    player_dim: int = 48
    n_heads: int = 4
    n_layers: int = 1
    ctx_dim: int = 192
    slot_hidden: int = 96
    attention: bool = False  # attention over player tokens: ~2x training cost on CPU; ablation flag

    def to_dict(self) -> dict:
        return asdict(self)


class AttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention + feed-forward over at most eight player tokens.

    Written with plain batched matmuls, which is much faster on CPU than fused kernels at this size.
    """

    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, P, D = x.shape
        h = self.h
        q, k, v = self.qkv(self.ln1(x)).view(B, P, 3, h, D // h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) * (D // h) ** -0.5  # [B,h,P,P]
        att = att.masked_fill(~valid[:, None, None, :], NEG).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, P, D)
        x = x + self.out(y)
        return x + self.ff(self.ln2(x))


class SkyjoNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or NetConfig()
        d = cfg.player_dim
        self.player_enc = nn.Sequential(nn.Linear(N_PLAYER, d), nn.ReLU(), nn.Linear(d, d))
        self.attn = nn.ModuleList(
            [AttentionBlock(d, cfg.n_heads, 2 * d) for _ in range(cfg.n_layers if cfg.attention else 0)]
        )
        self.trunk = nn.Sequential(
            nn.Linear(N_GLOBAL + 3 * d, cfg.ctx_dim), nn.ReLU(),
            nn.Linear(cfg.ctx_dim, cfg.ctx_dim), nn.ReLU(),
        )
        self.slot_in = nn.Linear(N_SLOT, cfg.slot_hidden)
        self.slot_ctx = nn.Linear(cfg.ctx_dim, cfg.slot_hidden, bias=False)
        self.slot_out = nn.Sequential(
            nn.ReLU(), nn.Linear(cfg.slot_hidden, cfg.slot_hidden), nn.ReLU(), nn.Linear(cfg.slot_hidden, 3)
        )
        self.draw_head = nn.Sequential(nn.Linear(cfg.ctx_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.setup_head = nn.Sequential(nn.Linear(cfg.ctx_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.value_head = nn.Sequential(nn.Linear(cfg.ctx_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.register_buffer("same_col", torch.tensor(SETUP_SAME_COLUMN, dtype=torch.long))

    @staticmethod
    def features(obs: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else obs
        g, pl, sl = featurize(o)
        return torch.from_numpy(g), torch.from_numpy(pl), torch.from_numpy(sl)

    def forward(self, obs, mask: torch.Tensor, feats=None) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: int [B, OBS_SIZE] (ignored if ``feats`` is given); mask: bool [B, N_ACTIONS].

        Returns (masked logits [B, N_ACTIONS], value [B])."""
        g, pl, sl = feats if feats is not None else self.features(obs)
        valid = pl[:, :, 0] > 0.5  # [B,P]
        tok = self.player_enc(pl)
        for blk in self.attn:
            tok = blk(tok, valid)
        vf = valid.unsqueeze(-1).to(tok.dtype)
        tok = tok * vf
        me = tok[:, 0]
        opp = vf.clone()
        opp[:, 0] = 0
        opp_mean = (tok * opp).sum(1) / opp.sum(1).clamp_min(1)
        opp_max = tok.masked_fill(opp < 0.5, -1e4).max(1).values
        ctx = self.trunk(torch.cat([g, me, opp_mean, opp_max], -1))
        value = self.value_head(ctx).squeeze(-1)

        slot_h = self.slot_in(sl) + self.slot_ctx(ctx)[:, None, :]
        slot_logits = self.slot_out(slot_h)  # [B,S,3]: take / keep / reveal
        setup = self.setup_head(ctx)  # (different column, same column)
        logits = torch.cat(
            [setup[:, self.same_col], self.draw_head(ctx), slot_logits[..., 0], slot_logits[..., 1],
             slot_logits[..., 2]],
            1,
        )
        assert logits.shape[1] == N_ACTIONS
        return logits.masked_fill(~mask, NEG), value
