# Rules specification `classic_v1_2p`

The project is specialised for **two-player (heads-up) Skyjo**. The rules are the classic publisher rules;
`classic_v1_2p` differs from the earlier `classic_v1` (2–8 players) only in restricting tables to two players, which
also shrinks the observation layout (schema) and the compiled engine. The multi-player release is archived in
`artifacts/legacy_multiplayer/` and remains playable at two-player tables through `skyjo_ml.legacy`.

Authoritative sources (linked, not redistributed; checksums of the copies reviewed on 2026-09-29):

| Document | SHA-256 |
| --- | --- |
| [Magilano multilingual rulebook (Salling Group mirror)](https://digitalassets.sallinggroup.com/image/upload/f45176eeeb808d1afe6412fef0db9e27.pdf) | `fed1d1b9616a3bf3f5e2a9b42c2a83e2a9e7ccd37078885b179e57788d21f5bd` |
| [Magilano component sheet / Dutch rules (Spellenlab mirror)](https://spellenlab.be/media/files/4bcf446fcb4e204c2af1da57f4d9ef6e781f9241.pdf) | `f27d703b128470b2028dc88a0ce656bd8b4ad9910762d9084a986acd054b414c` |

The component sheet confirms the 150-card inventory (5 × −2, 15 × 0, 10 × each of −1 and 1…12) and its Dutch
rules text confirms: the discard-exchange/draw-then-keep-or-reveal turn, clearing a column of three equal
face-up cards (also during final scoring; the exchanged card goes to the discard pile before the column),
one final turn for every other player after someone reveals their last card, doubling of the finisher's
positive score unless strictly lowest, and match end at 100 points with the lowest total winning.

`skyjo_ml.game.rules.ruleset_hash()` hashes the full specification including the conventions below; every
replay, checkpoint manifest and evaluation report records it. Any change produces a new ruleset name.

## Conventions for underspecified cases

| Case | Convention |
| --- | --- |
| Tied first-round starting sums | Seeded uniform choice between the tied players. |
| Tied lowest match totals | Shared win: each of the `k` tied winners receives `1/k` win credit. |
| Exhausted draw pile | Keep the top discard; shuffle the rest into a new draw pile at the next draw. The new pile's composition is public. |
| Nothing drawable after recycling | Deck draw is masked; taking the discard stays legal. No legal action raises an invariant error. |
| Initial reveals | Every player commits an unordered pair of slots before any value is shown; all pairs are revealed together. Commit requests go clockwise from the round's starting seat and carry no information. |
| Several columns at final scoring | Players clockwise from the finisher, columns left to right; no decisions. |
| Empty grid | No remaining cards counts as no hidden cards and triggers the end sequence. |
| Finisher doubling | Positive score doubles if any other player's round score is less than or equal to it. Zero and negative scores never double. |
| Watchdog | 3,000 decisions per round or 20,000 per match truncates a match. Truncated matches are never scored; they are reported separately. |
| RNG | splitmix64 stream per match; Fisher–Yates with multiply-shift bounded integers (bias < n/2³², n ≤ 150). |

## Engine and action space

- `game/engine.py` is the reference engine (plain Python); `envs/batch.py` is the Numba batch engine. Both
  draw every chance outcome from the same stream, and differential tests compare observations, masks, round
  scores, totals and win credit step by step, including through deck recycles.
- 103 actions: 66 setup pairs, draw, take-discard-into-slot (12), keep-drawn-into-slot (12),
  discard-drawn-and-reveal-slot (12). Slots are `column * 3 + row` and never compacted when columns clear.
- Illegal actions raise (`IllegalAction` in the reference engine, `ValueError` from `BatchEnv.step`).
