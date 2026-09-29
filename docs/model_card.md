# Model card: `models/skyjo_policy.pt`

| Field | Value |
| --- | --- |
| Task | Choose Skyjo moves to maximise the probability of winning a complete match (lowest total when someone reaches 100) |
| Rules | `classic_v1`, ruleset hash `7eeb6730ea5a3820` (see `docs/rules.md`); 2–8 players, one model conditioned on player count |
| Interface | Raw seat-relative public observation (`game/obs_layout.py`, schema hash `a1d6b849ca5e34a5`) + legal-action mask → 103 masked logits + scalar win-credit estimate |
| Architecture | Numba featurizer (exact public card accounting, per-player summaries, per-slot placement features) → deep-sets player encoder + 2-layer 192-unit trunk → position-conditioned slot head (take / keep / reveal per slot), draw head, setup head (same/different column), value head. 170,263 parameters. Equivariant to column and within-column permutations. |
| Training | Behaviour cloning of heuristic play (~0.87M decisions, 4 epochs) → PPO with terminal win-credit reward, gamma = 1, full-match trajectories, league of frozen snapshots + heuristics: 6M decisions (seed 3), then continued to 14.02M decisions with the other seeds' final policies in the league. ~3.6 h of PPO wall time on one core of the reference machine. |
| Selection | Paired dev-seed promotion test vs the 6M incumbent (`docs/results.md`); release seeds were used only once, for the final table. |
| Package | `sha256 4d5a4d50613c9a50b3a198e56fcf63173bed21d683efabf3df8683199edef30b`; metadata incl. manifest, lineage and embedded benchmark summary in `models/skyjo_policy.json`. `load_model` refuses packages whose ruleset or schema hash differs from the running code. |
| Reference machine | 4 vCPU Xeon E5-2689 v4 (virtualised), 8 GB RAM, no GPU; Python 3.12.14, torch 2.14.0+cpu, numba 0.67.0, numpy 2.3.5 |

## Measured strength (sealed release seeds, policy-only arg-max)

Against three copies of the strongest baseline (`finish_tuned`) at 4 players, over 10,000 paired matches: **59.3% win
credit (95% CI 58.4–60.2%) where parity is 25%**, mean finishing rank 1.66, no truncated matches.

Against `finish_tuned` at every table size (400 deals × all seat rotations per cell):

| Players | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Parity | 50.0% | 33.3% | 25.0% | 20.0% | 16.7% | 14.3% | 12.5% |
| Model win credit | 82.1% | 68.7% | 58.6% | 51.5% | 44.9% | 40.3% | 35.6% |

The full table (random, greedy, column, finish, finish_tuned, mixed tables, mirror seat check) is in `docs/results.md`.

## Search mode

`RootMCSearch` (belief-sampled root Monte Carlo, `skyjo play --search`) wraps the same network. On the 6M checkpoint
it improved 4-player win credit against 3 × `finish` by +0.33 credit × n (≈ +8 percentage points), 95% CI +0.01 to +0.67,
in 240 paired matches at 256 rollouts/decision (p50 0.58 s, p95 1.4 s per searched decision on one busy core).
Evidence on the released model is reported in `docs/results.md`. Search distillation was not performed.

## Limitations and intended use

- Strength claims are relative to the heuristics in this repository and to earlier snapshots of this model. No
  human or third-party bot evaluation was done; there is no claim of optimal or equilibrium play.
- Opponents are modelled only through the public board: the model has no memory of an opponent's past behaviour
  (the planned event GRU was not built).
- Trained and evaluated on `classic_v1` conventions (e.g. simultaneous setup reveals, shared wins on ties). House rules
  that change scoring or turn structure invalidate the model.
- Mirror self-play shows seat-position effects of up to ±0.5 credit × n in small (100-deal) samples; they are within
  what the table rules imply (first-round starter, final-turn order) but were not separately analysed.
- The value head is an estimate of win credit under self-play-like opponents; it is not calibrated against humans.
