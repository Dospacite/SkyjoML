"""Frozen rules specification ``classic_v1_2p`` (classic Skyjo rules, two players).

Every constant that affects simulation lives here so that a single hash identifies the ruleset used by
replays, checkpoints and evaluation results. Conventions for cases the publisher rules leave open are
documented in ``docs/rules.md`` and named in ``CONVENTIONS``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

RULESET_NAME = "classic_v1_2p"  # classic rules, two-player tables only

MIN_PLAYERS = 2
MAX_PLAYERS = 2  # the project is specialised for heads-up play
N_COLS = 4
N_ROWS = 3
N_SLOTS = N_COLS * N_ROWS  # slot index = col * N_ROWS + row
MATCH_END_SCORE = 100

# Card values -2..12 are stored directly; "codes" are value + 2 in 0..14.
MIN_VALUE = -2
MAX_VALUE = 12
N_VALUES = MAX_VALUE - MIN_VALUE + 1  # 15
INVENTORY = {-2: 5, -1: 10, 0: 15, **{v: 10 for v in range(1, 13)}}
DECK_SIZE = sum(INVENTORY.values())  # 150
INVENTORY_COUNTS = np.array([INVENTORY[v] for v in range(MIN_VALUE, MAX_VALUE + 1)], dtype=np.int32)
CARD_VALUES = np.arange(MIN_VALUE, MAX_VALUE + 1, dtype=np.int32)

# Watchdog: a match exceeding this many decisions is truncated (never scored as an official result).
MAX_MATCH_DECISIONS = 20_000
MAX_ROUND_DECISIONS = 3_000

CONVENTIONS = {
    "tied_first_round_start": "seeded uniform choice among tied highest initial sums",
    "tied_match_winners": "shared win; each of k tied lowest totals earns 1/k win credit",
    "exhausted_draw_pile": "keep top discard; shuffle the rest into the draw pile at the next required draw",
    "no_drawable_cards": "mask deck draw; discard exchange remains legal; no legal action is an invariant error",
    "initial_reveals": "every player commits an unordered pair before any value is shown; all revealed together",
    "final_scoring_column_clears": "players clockwise from finisher, columns left to right; no decisions",
    "empty_grid": "no remaining grid cards counts as no hidden cards and triggers the end sequence",
    "finisher_doubling": "finisher's positive score doubles unless strictly lower than every other player",
    "discard_order_on_column_clear": "exchanged/drawn card is discarded before the three column cards",
    "round_start_player": "previous round's finisher starts; first round: highest revealed pair sum",
    "setup_commit_order": "commit decisions are requested clockwise from the round's starting seat "
    "(seat 0 in round one); commit order carries no information",
    "rng": "splitmix64 per match; Fisher-Yates with multiply-shift bounded integers",
}


def ruleset_spec() -> dict:
    return {
        "name": RULESET_NAME,
        "players": [MIN_PLAYERS, MAX_PLAYERS],
        "grid": [N_COLS, N_ROWS],
        "inventory": {str(k): v for k, v in INVENTORY.items()},
        "match_end_score": MATCH_END_SCORE,
        "watchdog": {"match_decisions": MAX_MATCH_DECISIONS, "round_decisions": MAX_ROUND_DECISIONS},
        "conventions": CONVENTIONS,
    }


def ruleset_hash() -> str:
    blob = json.dumps(ruleset_spec(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


SOURCES = {
    "rulebook": {
        "url": "https://digitalassets.sallinggroup.com/image/upload/f45176eeeb808d1afe6412fef0db9e27.pdf",
        "sha256": "fed1d1b9616a3bf3f5e2a9b42c2a83e2a9e7ccd37078885b179e57788d21f5bd",
    },
    "components": {
        "url": "https://spellenlab.be/media/files/4bcf446fcb4e204c2af1da57f4d9ef6e781f9241.pdf",
        "sha256": "f27d703b128470b2028dc88a0ce656bd8b4ad9910762d9084a986acd054b414c",
    },
}
