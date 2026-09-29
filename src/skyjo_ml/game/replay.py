"""Schema-versioned replays.

A replay stores the deal seed and every action. The private execution trace (seed + actions) reproduces the
match exactly, including hidden cards; ``public_trace`` is the redacted, player-visible event stream that
contains no seed and no unrevealed values.
"""

from __future__ import annotations

import json
from pathlib import Path

from .actions import describe
from .engine import SkyjoGame
from .rules import RULESET_NAME, ruleset_hash

REPLAY_SCHEMA = 1


def _public_event(ev: dict, g: SkyjoGame) -> dict:
    e = dict(ev)
    if e["type"] == "setup_reveal":
        e["values"] = [[g.val[p][s] for s in e["picks"][p]] for p in range(g.n)]
    return e


class Recorder:
    def __init__(self, n_players: int, seed: int, agents: list[str] | None = None):
        self.game = SkyjoGame(n_players, seed)
        self.actions: list[int] = []
        self.public: list[dict] = []
        self.agents = agents or [f"seat{i}" for i in range(n_players)]

    def step(self, action: int):
        g = self.game
        actor = g.cur
        res = g.step(action)
        self.actions.append(int(action))
        # setup reveal values are read right after the step, before any later re-deal
        self.public.append({"actor": actor, "action": int(action), "desc": describe(action),
                            "events": [_public_event(e, g) for e in res.events]})
        return res

    def to_dict(self) -> dict:
        g = self.game
        out = {
            "schema": REPLAY_SCHEMA,
            "ruleset": RULESET_NAME,
            "ruleset_hash": ruleset_hash(),
            "n_players": g.n,
            "seed": g.seed,
            "agents": self.agents,
            "actions": self.actions,
            "public_trace": self.public,
            "result": {
                "over": g.over,
                "truncated": g.truncated,
                "scores": g.scores,
                "round_scores": g.round_history,
                "win_credit": g.win_credit() if g.over and not g.truncated else None,
            },
        }
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)


def load(path: str | Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    if d.get("schema") != REPLAY_SCHEMA:
        raise ValueError(f"unsupported replay schema {d.get('schema')}")
    if d.get("ruleset_hash") != ruleset_hash():
        raise ValueError(f"replay ruleset hash {d.get('ruleset_hash')} != current {ruleset_hash()}")
    return d


def verify(d: dict) -> SkyjoGame:
    """Re-simulate a replay and check that the recorded result is reproduced exactly."""
    g = SkyjoGame(d["n_players"], d["seed"])
    for a in d["actions"]:
        g.step(a)
    r = d["result"]
    if g.scores != r["scores"] or g.round_history != r["round_scores"]:
        raise AssertionError("replay does not reproduce recorded result")
    return g


def card(v: int) -> str:
    return f"{v:>3d}"


def render_board(g: SkyjoGame, p: int, reveal_all: bool = False) -> list[str]:
    from .engine import ABSENT, HIDDEN

    rows = []
    for r in range(3):
        cells = []
        for c in range(4):
            s = c * 3 + r
            st = g.vis[p][s]
            if st == ABSENT:
                cells.append("  .")
            elif st == HIDDEN and not reveal_all:
                cells.append("  #")
            else:
                cells.append(card(g.val[p][s]))
        rows.append(" ".join(cells))
    return rows
