# Skyjo ML project plan

Historical design plan for the earlier 2–8-player project. The current code and model are specialised for two players; see `README.md` and `docs/training.md` for the implemented system.

Status: architecture selected; implementation has not started.  
Research date: 2026-09-29.  
Scope: original/classic Skyjo, two through eight players, complete matches.

## 1. Objective and definition of success

Build the strongest empirically validated Skyjo agent achievable with the available compute. Optimize the probability of winning a complete match, including decisions that depend on cumulative scores. Low average round score is a diagnostic, not the final objective.

There is no justified claim of a universally optimal policy: performance depends on opponents, player count, and inference budget. Publish those conditions with every strength claim. The final deliverables are a reproducible simulator, training pipeline, benchmark suite, trained policy, search-enhanced agent, and local command-line interface.

Train one policy conditioned on player count, then evaluate each count separately. Use four-player games as the primary development benchmark and two-player games for early debugging. Final release requires evidence for all supported player counts; success at two players does not establish multiplayer strength.

## 2. Rules researched before architectural design

The authority is the Magilano rulebook, rather than a third-party online implementation. I reviewed the English rules on printed pages 4–5 of this [publisher-authored multilingual rulebook, hosted by Salling Group](https://digitalassets.sallinggroup.com/image/upload/f45176eeeb808d1afe6412fef0db9e27.pdf). Its rules establish:

- Deal twelve hidden cards per player in four columns of three; reveal two initially.
- Highest initial revealed sum starts the first round. The previous finisher starts subsequent rounds; turns proceed clockwise.
- Taking the discard requires exchanging it with an existing card, revealing the replacement and discarded card.
- A deck draw can be exchanged similarly, or discarded while revealing a hidden card. No inspecting a hidden replacement target first.
- Three exposed equal cards in a column must be discarded, including during final scoring. Discard the exchanged card before the completed column.
- Exposing all remaining cards triggers one final turn for every other player.
- Reveal remaining cards, remove matching columns, and sum remaining values. Double the finisher's positive score unless strictly lowest; ties trigger doubling, zero and negative scores do not.
- Accumulate scores across rounds. At least one total reaching 100 ends the match; the lowest total wins.

The deck inventory is 150 cards: five −2s, fifteen 0s, and ten of each other integer from −1 through 12. This is explicitly listed in the [publisher's component sheet, hosted by Spellenlab](https://spellenlab.be/media/files/4bcf446fcb4e204c2af1da57f4d9ef6e781f9241.pdf).

### Explicit conventions for underspecified cases

The reviewed English rules do not resolve every simulation detail. Implement the following as named conventions in `classic_v1`, not claims about publisher wording:

| Case | Project decision |
| --- | --- |
| Tied first-round starting sums | Seeded uniform choice among tied players. |
| Tied lowest match totals | Shared win; each of the `k` tied winners earns `1/k` win credit. |
| Exhausted draw pile | Preserve the top discard; shuffle the rest into the draw pile before the next required draw. |
| No drawable cards even after recycling | Mask deck draw; permit a legal discard exchange. Raise an engine invariant error if no legal action exists. |
| Initial reveals | Each player commits an unordered pair of positions before any starting values are exposed; reveal all committed pairs together. This preserves choice of column arrangement without introducing a setup information advantage. |
| Several columns cleared at final scoring | Process players clockwise from the finisher, then columns left to right; no further decisions occur before redealing. |
| No remaining grid cards | Counts as having no hidden cards and triggers the end sequence. |

Version these conventions and include the ruleset hash in every replay, checkpoint, and result. Any future source clarification produces a new version and a compatibility report. Obtain and checksum the referenced rule documents during implementation; link to them rather than assuming redistribution rights.

## 3. Architectural decisions

| Area | Decision and rationale |
| --- | --- |
| Language | Python 3.12 with `uv`, `pyproject.toml`, and a committed lockfile. One language keeps the first implementation inspectable. |
| Simulation | Small deterministic reference engine using integer arrays and explicit random generators; add a Numba-compiled batch implementation after correctness is established. |
| RL interface | PettingZoo AEC adapter for interoperability; direct batched engine interface for training throughput. AEC fits sequential actions and legal-action masks ([documentation](https://pettingzoo.farama.org/api/aec/)). |
| Learning | PyTorch, masked recurrent PPO, parameter sharing across seats, historical-opponent self-play. PPO is the initial trainable foundation ([original paper](https://arxiv.org/abs/1707.06347)); its effectiveness here must be measured. |
| Network | Shared card/column encoder, attention across player boards, public-event GRU, masked policy heads, and a match-outcome value head. |
| Strongest deployment mode | Belief-sampled root search with neural rollouts and leaf evaluation, followed by search distillation. |
| Experiment infrastructure | Local structured artifacts, JSONL metrics, TensorBoard, Parquet evaluation tables, and versioned checkpoint manifests. No external service required. |
| Quality | `pytest`, Hypothesis, Ruff, type checking, replay fixtures, differential engine tests, and automated hidden-information audits. |
| Product interface | Python agent API and CLI first. A graphical interface is outside the initial scope. |

Pin compatible dependency versions when bootstrapping and record hardware/software versions in run manifests. Do not choose package versions from this document's date alone.

## 4. Engine, actions, and information boundaries

### State machine

Represent `SETUP`, `SELECT_SOURCE`, `RESOLVE_DRAW`, `FINAL_TURNS`, `SCORE_ROUND`, and `MATCH_OVER` explicitly. Final-turn status is also available during source selection and draw resolution. Resolve automatic effects atomically before advancing the active player. Store the original finisher and a queue of remaining final turns; later players finishing must not restart that queue.

Keep four fixed column slots per player and mark removed columns absent. Never compact positions: stable indices make actions, replays, and observations unambiguous. Full state includes grid values, visibility, presence, ordered draw/discard piles, any pending card, current player, cumulative scores, phase, and final-turn bookkeeping.

Use independent random streams for dealing/recycling, policy sampling, opponent selection, and search. The agent never receives environment seeds or RNG state. Replays record a private execution trace and a separately redacted player-visible trace.

### Action encoding

Use phase-specific categorical heads, exposed through one fixed discrete action space with disjoint offsets:

| Phase | Choices |
| --- | --- |
| Setup | 66 unordered pairs of distinct grid positions. |
| Select source | Draw from deck, or take discard and replace position `i`: at most 13 choices. |
| Resolve deck draw | Keep and replace position `i`, or discard and reveal hidden position `i`: at most 24 choices. |

The combined action space has 103 indices. A source decision cannot depend on the unseen draw. After drawing, only the acting player receives its value until resolution. Mask absent targets and phase-incompatible actions. Compute legality entirely from information available to the actor, and fail loudly on invalid API actions instead of silently substituting a move.

### Agent-facing contract

`Agent.act(observation, memory, budget) -> (legal_action, new_memory, diagnostics)`.

The observation contains:

- Every player's visible board, with distinct encodings for hidden and absent cards.
- Seat order relative to the actor, current actor, player count, phase, and final-turn status.
- Cumulative totals and public round statistics, without clipping strategically meaningful scores.
- Top discard, pile sizes, exact public event deltas, and publicly reconstructible card counts.
- The actor's pending drawn card only when entitled to see it.
- The legal-action mask.

Neither policies nor value networks receive hidden grid values, actual draw order, private engine handles, or future events. Use the same observation-only critic at training and inference so its values can safely support search.

## 5. Belief model

Players do not know even their own unrevealed cards. Start with an explicit finite-deck probability model rather than asking a neural network to rediscover card counting.

Maintain physical card accounting: currently visible cards, cards currently in the discard pile, a pending known draw, and remaining hidden locations. Count cards by current location; subtracting every historical sighting would double-count cards that move or recycle. Completed columns enter the discard pile and can return through recycling.

Before the first recycle, unseen assignments are exchangeable conditional on the known inventory. Sample hidden layouts and draw order without replacement. A hidden position selected without seeing its value does not by itself justify an opponent-dependent value bias.

After recycling, the known composition of the new draw pile differs from the pool of still-hidden grid cards. Track those pools separately. Use a weighted particle representation when observations couple assignments between pools; update on reveals and swaps, reject inconsistent particles, and resample/rejuvenate as needed. Start with 256 particles and measure effective sample size and calibration before increasing the count.

Expose marginal distributions and expected values as network features. Preserve joint assignments for search: independent per-slot samples can create impossible decks. Validate beliefs against exhaustive enumeration on small artificial decks, including recycle boundaries. Hidden-state labels may be used to score calibration in tests, never as actor inputs.

## 6. Policy and training

### Initial network

- Embed each card's visible value/status into 32 dimensions, including column membership.
- Apply a shared column encoder and pool into a representation for each player.
- Apply two attention blocks of width 128 with four heads over player representations; pad to eight players and mask padding. Include relative turn order, which is strategically meaningful.
- Encode public event deltas with a 128-unit GRU per agent to retain opponent behavior across turns and rounds. Feed each event exactly once; retain memory across rounds and reset at match boundaries.
- Fuse board, event, belief, score, and phase features. Produce phase-specific policy logits and a scalar expected terminal win-credit value for the observing player.
- Use position-conditioned scores for board actions and pair-conditioned scores for setup. Preserve column geometry under augmentation; rotate seats and permute columns with corresponding action remapping, but never arbitrarily shuffle cards between columns.

These dimensions are starting configurations. Compare against a small feedforward model before accepting the recurrent architecture's extra cost.

### Reward and trajectories

Final training reward is terminal win credit: `1/k` for a tied winner and zero otherwise. Use `gamma = 1.0` to avoid rewarding shorter matches. No permanent reward for revealing cards, removing columns, or reducing visible sum: such incentives can conflict with winning.

Short single-round pretraining may use negative round score to learn mechanics, but it is a warm start only. All candidate selection and final optimization use complete matches. Pretrain the encoder with score/outcome prediction if sparse rewards stall; keep auxiliary targets distinct from the reward.

Collect trajectories by player, aligning each action with that player's next decision and accumulated intervening rewards. Train the learner's actions only; frozen opponents generate experience context but do not enter its PPO loss. Include both source selection and post-draw actions. Rounds are not episode boundaries. Collector cutoffs bootstrap values; true match endings do not. Preserve recurrent state, apply burn-in, and mask padded sequence positions.

Initial PPO settings: AdamW at `3e-4`, clip ratio `0.2`, GAE lambda `0.95`, four update epochs, entropy coefficient `0.01`, gradient norm cap `0.5`, 128 concurrent matches, and 32,768 learner decisions per update. Tune from diagnostics, especially KL divergence, value error, entropy, reward variance, and per-phase action frequencies. Recompute sequence memory under updated parameters rather than treating stale recurrent states as exact.

### Opponent population and curriculum

1. Implement random, greedy expected-score, column-aware, and score/finish-aware heuristic opponents. Tune heuristic parameters only on development seeds.
2. Verify learning in two-player games, then make four-player full matches the first serious training target.
3. Expand to all counts with balanced sampling over 2–8 players.
4. Initially sample opponent policies as 50% frozen recent self-play snapshots, 25% older snapshots, and 25% heuristic/search baselines. Freeze versions for each rollout batch.
5. Randomize learner seat and independently sample opponent seats so the population includes mixed tables. Add periodic all-current-policy self-play batches.
6. Maintain matchup tables and periodically train an exploiter against the incumbent to find weaknesses. Feed successful exploiters back into the population while preserving separate held-out opponents.

Track three independent training seeds for serious candidates. Never promote a model solely because it defeats its immediate predecessor: retain older diverse opponents to detect cyclic performance and forgetting.

## 7. Search and iterative improvement

The first search implementation is root Monte Carlo policy improvement, not perfect-information minimax:

1. Construct beliefs using only the actor's observable history.
2. Sample consistent hidden worlds and evaluate every legal root action with a minimum sample allocation.
3. Roll forward using neural policies. Each simulated player receives only its own simulated observations, including private draws only during its own decision.
4. Continue to terminal outcome where practical; otherwise stop at a decision for the root player after a configurable horizon and use its observation-only value estimate. Start with two table rotations.
5. Average terminal win credit or leaf value across samples; allocate further samples by sequential halving and choose the highest estimated action value.

Initialize simulated opponents' recurrent memory from the public history available to them, not the root player's private memory. Any sampled private events must obey the simulated history. Hidden-world samples provide transition outcomes, never policy inputs.

Start with 512 rollouts per decision and measure 128/512/2,048/8,192-rollout budgets. Use coupled random samples across candidate actions where valid to reduce variance, with separate RNG streams from the real game. Deliver a fast policy-only mode and a strong search mode with a deadline, returning the best completed estimate when time expires. Initial interactive target: p95 under one second on the documented reference machine, subject to measurement.

If deeper search earns a significant strength gain per unit time, extend to information-set tree search with observation/history-based nodes; the [ISMCTS paper](https://eprints.whiterose.ac.uk/id/eprint/75048/) is the methodological reference. Do not merge nodes with different information, expose determinizations to decision policies, or negate values as if a multiplayer game were alternating two-player zero-sum play. Search remains approximate and has no equilibrium guarantee.

Distill improved search action distributions into a student, then resume PPO against the population. Evaluate the student and search wrapper independently. Search supervision uses a separate supervised objective; do not pass search-chosen actions through ordinary on-policy PPO ratios.

## 8. Correctness and evaluation gates

### Engine and information integrity

Before expensive training, require:

- Hand-authored replay fixtures for every rule and convention, including final scoring, penalties, ties, exhaustion/recycling, setup choices, empty grids, and round transitions.
- Property tests for 150-card conservation, valid multiplicities, score consistency, legal-action availability, and deterministic replay.
- At least 100,000 randomized completed matches across player counts with no invariant failures; separately exercise adversarial long-running traces.
- Differential reference-versus-batched-engine tests on identical explicit chance outcomes, including observations and action masks.
- Hidden-state permutation tests: identical actor information must produce identical observation, mask, and policy output with fixed policy randomness. Search must have the same output distribution when the real hidden engine state changes but actor history does not.
- Exact small-deck belief tests and tiny-endgame expected-utility checks to verify search estimates and decision ordering.

Skyjo permits repeated exchanges without progress. Use a watchdog to detect exceptionally long games, but never silently turn a timeout into a scored official result. Save the trace; report truncations separately. In training, bootstrap a truncated trajectory. In release evaluation, require no unresolved truncations in the scored sample or explicitly disclose incomplete results.

### Strength benchmark

Maintain separate training, development, and sealed release seed manifests and opponent sets. Keep the release set out of hyperparameter selection. Benchmark:

- Candidate versus each heuristic, a tuned search baseline, and a diverse frozen policy population.
- Mixed-opponent tables, mirror self-play for seat-bias checks, and historical round-robin matchups.
- Policy-only and search modes under matched wall-time budgets.
- Each player count, each learner seat, and cumulative-score/endgame stress cases.

Primary metric: mean terminal win credit. Report strict wins, shared wins, mean finishing rank, score differences, round scores, penalty rate, match length, and inference latency as secondary metrics. A homogeneous `n`-player mirror table should average `1/n` win credit per seat after balancing.

Use paired candidate/incumbent tournaments with the same initial shuffle seeds, opponent versions, and rotated seats. Branching actions mean later random outcomes need not stay identical. Bootstrap at the independent seed/seat-rotation block level, not at individual turns. Pre-register sample sizes and comparisons; do not repeatedly peek until significance appears.

Start development comparisons with 2,000 complete matches per player count. Use at least 10,000 for release comparisons, expanding according to a precomputed precision/power target. Report 95% confidence intervals and uncertainty across training seeds. These are minimum budgets, not a promise that small improvements will be detectable.

Promote a candidate when its held-out paired improvement over the incumbent has a positive lower confidence bound on the primary four-player suite, with no material regression elsewhere. Define material regression as more than two percentage points of win credit; require the interval to rule that out for each supported count, with multiplicity adjustment. Publish the full table even if the aggregate improves.

Run ablations for card counting, recurrence, population diversity, search, distillation, and full-match training. Retain complexity only when it produces a reproducible benefit. Approximate exploiters diagnose weaknesses; they do not certify low exploitability or optimal play.

## 9. Repository and operational design

```text
skyjo-ml/
  PLAN.md
  pyproject.toml
  uv.lock
  configs/                 # Rules, models, training, search, evaluation
  src/skyjo_ml/
    game/                  # Reference engine, rules, observations, replay
    envs/                  # AEC adapter and compiled batch engine
    belief/                # Card inventory, particles, calibration
    agents/                # Heuristics and common agent protocol
    models/                # Encoders, recurrent policy/value model
    training/              # PPO, rollout collector, opponent league
    search/                # Belief rollouts and policy improvement
    evaluation/            # Tournaments, statistics, benchmark manifests
    cli.py
  tests/                   # Rules, properties, leakage, differential tests
  benchmarks/              # Engine and inference performance harnesses
  docs/                    # Rules decisions, experiments, model cards
  artifacts/               # Ignored generated runs and checkpoints
```

Planned commands: `skyjo validate`, `skyjo benchmark`, `skyjo train --config ...`, `skyjo evaluate --checkpoint ...`, `skyjo play --checkpoint ...`, and `skyjo replay ...`. The play interface supports manual entry of public events and private draws without asking a human for unrevealed card values. Save schema-versioned game sessions so long matches can resume.

Every run records the code revision, dirty diff if any, resolved config, rules/schema hashes, RNG seeds, opponent hashes, dependency versions, hardware, learner decisions, complete matches, elapsed time, and checkpoint lineage. Checkpoints include optimizer/scheduler state, RNG states, curriculum state, and opponent registry. Save at rollout boundaries; either persist in-flight environments/memory or document that resume restarts those rollouts.

Keep full replays for fixtures, failures, and a sample of evaluation games; store aggregate metrics for routine training to control disk usage. Export the final model with its observation normalizer, action schema, configuration, benchmark report, and model card. Use synthetic self-play data; no human dataset is required.

## 10. Implementation milestones

| Milestone | Deliverable | Exit condition |
| --- | --- | --- |
| M0 — Freeze specification | `classic_v1`, source hashes, action/observation schemas, project skeleton | Every rule ambiguity has a documented convention; dependencies install reproducibly. |
| M1 — Trustworthy game | Reference engine, CLI replay, fixtures, heuristic agents | Rules/property/leakage gates pass and complete matches can be played manually. |
| M2 — Measurement and speed | Tournament runner, statistical reports, compiled batch engine | Differential tests pass; baseline strength and transitions/second measured for all counts. |
| M3 — Learning baseline | Feedforward masked PPO, full-match collector, frozen self-play population | Learning exceeds random and greedy play on held-out full matches across three seeds. |
| M4 — Strong policy | Belief features, attention/GRU, diverse league, all counts | Reproducible gain over tuned heuristics and the M3 baseline; no information leakage. |
| M5 — Search improvement | Belief rollouts, timed search, search distillation | Search improves strength at a published compute budget; student benefit assessed independently. |
| M6 — Release | Sealed evaluation, portable model package, play CLI, reproducibility guide | Promotion criteria met, limitations documented, fresh-environment replay/inference verified. |

M0–M2 are hard prerequisites for large training runs. Strength milestones are measured gates, not deadlines: a failed gate triggers diagnosis and a bounded experiment, not an unsupported claim of completion.

## 11. Compute and experiment budget

Assume local Linux development, 16 CPU cores, 64 GB RAM, and one CUDA GPU with roughly 16–24 GB VRAM as a planning reference, not a purchasing requirement. Support CPU-only correctness tests and small training runs. Start with CPU simulation workers and batched GPU inference; profile before adding machines or a different runtime.

Use measured rollout throughput to schedule progressively larger runs: approximately 100,000 learner decisions for smoke checks, 10 million for a pilot, then 100 million for a serious candidate. These are budget checkpoints, not claims of sufficient training. After the pilot, estimate wall time, storage, and evaluation cost from actual measurements before scaling.

Allocate the first research budget roughly 20% to correctness/profiling and baselines, 40% to policy learning, 25% to search/distillation, and 15% to evaluation and ablations. Track strength per GPU-hour and per inference second. Stop unproductive variants after their predeclared budget; retain the best validated incumbent. Distributed training is deferred until profiling proves a single-machine bottleneck and the learning curve justifies scaling.

## 12. Main risks and responses

| Risk | Response |
| --- | --- |
| A simulator error becomes a learned strategy | Freeze rules, require replay/property tests, and independently compare engines before scaling. |
| Hidden information leaks through features, masks, memory, or search | Enforce observation-only interfaces and test invariance under changes to actual hidden state. |
| Optimizing round score harms match wins | Train and select on complete-match outcomes; keep score as a diagnostic. |
| Self-play overfits or cycles | Maintain historical opponents, mixed tables, exploiters, and held-out evaluations. |
| Beliefs become wrong after recycling | Track location-specific inventory, test exact toy cases, and measure calibration. |
| Search learns implausibly informed continuations | Simulate every policy from its own information; validate on small cases with exact answers. |
| Sparse rewards or recurrent PPO instability | Start with the smaller baseline, inspect trajectory alignment, then use auxiliary learning and bounded curriculum stages. |
| Apparent gains come from random deals or extra compute | Pair seeds, balance seats, publish confidence intervals, and compare matched inference budgets. |
| One model sacrifices a player count | Gate per-count results; if needed, fine-tune separate per-count checkpoints behind the same API. |

The project is complete when a reproducible agent package meets the published strength and correctness gates. Continued improvement remains an empirical research loop; the release should state exactly which opponents and compute budgets it beats.
