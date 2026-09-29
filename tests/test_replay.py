"""Replay fixtures, deterministic re-simulation, and fresh-environment inference of released/legacy models."""

import json
from pathlib import Path

import numpy as np
import pytest

from skyjo_ml.game import replay as R
from skyjo_ml.game.engine import SkyjoGame

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parents[1]
MODEL = ROOT / "models" / "skyjo_policy.pt"
LEGACY = ROOT / "artifacts" / "legacy_multiplayer" / "skyjo_policy.pt"


def _heuristic_fixture():
    from skyjo_ml.agents.base import HeuristicBatchPolicy

    p = FIX / "replay_heuristics_2p.json"
    if not p.exists():
        rec = R.Recorder(2, 4242, agents=["finish", "column"])
        a0, a1 = HeuristicBatchPolicy("finish", seed=1), HeuristicBatchPolicy("column", seed=2)
        while not rec.game.over:
            g = rec.game
            a, _, _ = (a0 if g.cur == 0 else a1).act(g.observation(), g.legal_mask())
            rec.step(a)
        rec.save(p)
    return p


def test_replay_fixture_reproduces():
    d = R.load(_heuristic_fixture())
    g = R.verify(d)
    assert g.over and not g.truncated
    assert all("seed" not in e for e in d["public_trace"])


def test_public_trace_has_no_hidden_values():
    d = R.load(_heuristic_fixture())
    for step in d["public_trace"]:
        for e in step["events"]:
            assert e["type"] in {"setup_reveal", "draw", "take", "keep", "reveal", "column", "finish", "round_end",
                                 "recycle"}
            if e["type"] == "draw":
                assert "card" not in e  # the drawn value is only public once resolved


@pytest.mark.skipif(not LEGACY.exists(), reason="legacy multi-player model not available")
def test_legacy_adapter_reproduces_recorded_games():
    from skyjo_ml.legacy.policy import LegacyBatchPolicy

    pol = LegacyBatchPolicy(LEGACY)
    for game in json.load(open(FIX / "legacy_mp_2p_games.json")):
        g = SkyjoGame(2, game["seed"])
        for a in game["actions"]:
            if g.cur == 0:
                assert pol.act(g.observation(), g.legal_mask())[0] == a
            g.step(a)
        assert g.scores == game["scores"]


@pytest.mark.skipif(not (FIX / "replay_model_2p.json").exists() or not MODEL.exists(), reason="no released model")
def test_released_model_replays_its_own_moves():
    from skyjo_ml.agents.neural import NeuralBatchPolicy, load_model

    d = R.load(FIX / "replay_model_2p.json")
    net, _ = load_model(MODEL)
    pol = NeuralBatchPolicy(net, greedy=True)
    g = SkyjoGame(d["n_players"], d["seed"])
    matched = 0
    for a in d["actions"]:
        if g.cur == 0:
            assert pol.act(g.observation(), g.legal_mask())[0] == a
            matched += 1
        g.step(a)
    assert matched > 20
    assert np.allclose(g.win_credit(), d["result"]["win_credit"])
