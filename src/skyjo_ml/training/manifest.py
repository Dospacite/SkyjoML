"""Run manifests: code revision, environment, ruleset/schema hashes and seed bookkeeping."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from ..game import obs_layout
from ..game.actions import N_ACTIONS
from ..game.rules import ruleset_hash

ROOT = Path(__file__).resolve().parents[3]

# Disjoint seed ranges (deal seeds). The release range must never be used for selection or tuning.
SEED_RANGES = {
    "train": (0, 1_000_000_000),
    "dev": (1_000_000_000, 1_100_000_000),
    "release": (2_000_000_000, 2_100_000_000),
}


def schema_hash() -> str:
    """Hash of the observation layout, action space and featurizer interface."""
    from ..models import features

    spec = {k: getattr(obs_layout, k) for k in dir(obs_layout) if k.isupper()}
    spec["N_ACTIONS"] = N_ACTIONS
    spec["features"] = [features.FEATURE_VERSION, features.N_GLOBAL, features.N_PLAYER, features.N_SLOT]
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def source_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "src").rglob("*.py")):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:
        return None


def environment() -> dict:
    deps = {}
    for d in ["numpy", "numba", "torch", "pyarrow"]:
        try:
            deps[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            deps[d] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "deps": deps,
    }


def build_manifest(config: dict, extra: dict | None = None) -> dict:
    rev = _git("rev-parse", "HEAD")
    diff = _git("diff", "HEAD")
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ruleset_hash": ruleset_hash(),
        "schema_hash": schema_hash(),
        "source_hash": source_hash(),
        "git_revision": rev.strip() if rev else None,
        "git_dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "environment": environment(),
        "config": config,
        **(extra or {}),
    }
