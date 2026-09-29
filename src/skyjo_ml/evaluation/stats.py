"""Block-bootstrap statistics for paired, seat-rotated tournaments.

The independent unit is a deal seed together with all of its seat rotations (a *block*). Turn-level or
match-level resampling would understate uncertainty, so every interval here resamples blocks.
"""

from __future__ import annotations

import math

import numpy as np


def block_means(values: np.ndarray, block_size: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    assert len(v) % block_size == 0, "values must be laid out block by block"
    return v.reshape(-1, block_size).mean(1)


def bootstrap_ci(blocks: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI of the mean of block values."""
    b = np.asarray(blocks, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n_boot, len(b)))
    means = b[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(b.mean()), float(lo), float(hi)


def summarize(values: np.ndarray, block_size: int, alpha: float = 0.05, seed: int = 0) -> dict:
    blocks = block_means(values, block_size)
    mean, lo, hi = bootstrap_ci(blocks, alpha=alpha, seed=seed)
    return {
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "se": float(blocks.std(ddof=1) / math.sqrt(len(blocks))) if len(blocks) > 1 else float("nan"),
        "n_blocks": int(len(blocks)),
        "n_matches": int(len(values)),
    }


def paired_difference(a: np.ndarray, b: np.ndarray, block_size: int, alpha: float = 0.05, seed: int = 0) -> dict:
    """CI for mean(a - b) where a and b were played on identical seeds and seatings (same block layout)."""
    return summarize(np.asarray(a) - np.asarray(b), block_size, alpha=alpha, seed=seed)


def bonferroni(alpha: float, k: int) -> float:
    return alpha / max(1, k)
