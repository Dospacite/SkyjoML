"""Portable model package: weights + network config + action/observation schema + provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .agents.neural import load_model
from .game.actions import N_ACTIONS
from .game.rules import RULESET_NAME, ruleset_hash
from .models.features import N_GLOBAL, N_PLAYER, N_SLOT
from .training.manifest import build_manifest, schema_hash

PACKAGE_VERSION = 1


def export(checkpoint: str, out: str, benchmark: str | None = None, notes: str = "") -> dict:
    # A legacy multi-player checkpoint has a different observation schema and must not
    # be stamped with the current two-player ruleset and schema hashes.
    load_model(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pkg = {
        "package_version": PACKAGE_VERSION,
        "model": ck["model"],
        "net_config": ck["net_config"],
        "ruleset": RULESET_NAME,
        "ruleset_hash": ruleset_hash(),
        "schema_hash": schema_hash(),
        "n_actions": N_ACTIONS,
        "feature_dims": {"global": N_GLOBAL, "player": N_PLAYER, "slot": N_SLOT},
        "source_checkpoint": str(checkpoint),
        "lineage": {k: ck.get(k) for k in ("update", "decisions", "matches", "elapsed", "train_config", "league")
                    if k in ck},
        "manifest": build_manifest({"export_from": str(checkpoint)}),
        "notes": notes,
    }
    if benchmark:
        with open(benchmark) as f:
            pkg["benchmark_summary"] = json.load(f).get("summary")
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pkg, p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    meta = {k: v for k, v in pkg.items() if k != "model"}
    meta["sha256"] = digest
    with open(p.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta
