# skyjo-ml

A two-player Skyjo simulator, PPO agent, belief-sampled search, and benchmark suite. The current ruleset is `classic_v1_2p`: the reference and compiled engines, observation layout, model, and CLI accept two players. The earlier 2–8-player release is preserved under `artifacts/legacy_multiplayer/` and remains available as a benchmark opponent through `skyjo_ml.legacy`.

## Setup

```bash
uv sync
uv run pytest
```

## Play and evaluate

```bash
uv run skyjo play --checkpoint models/skyjo_policy.pt --players 2 --save game.json
uv run skyjo advise --checkpoint models/skyjo_policy.pt --players 2 --seat 0
uv run skyjo evaluate --checkpoint models/skyjo_policy.pt --players 2 --opponents legacy finish_tuned_2p
uv run skyjo replay game.json -v
```

`skyjo play` deals a match in the terminal. You are seat 0; hidden cards show as `#`. Slots are addressed as `column row`, for example `t 2 3` takes the discard into column 2, row 3. `skyjo advise` tracks a physical two-player table using public events and your own draws.

## Train

```bash
uv run python -m skyjo_ml.training.distill --decisions 1400000 --epochs 4 --out artifacts/distill/distill_2p.pt
for s in 1 2 3; do
  uv run skyjo train --config configs/ppo_2p.toml --run-name ppo2p_s$s --seed $s &
done
wait
```

The distilled warm start uses the archived multi-player model as a teacher. The three PPO seeds train only on two-player matches. See [training](docs/training.md), [results](docs/results.md), [model card](docs/model_card.md), and [rules](docs/rules.md) for configuration, evidence, and conventions. `PLAN.md` describes the earlier multi-player design; the current implementation details are in those documents.

## Layout

```text
src/skyjo_ml/game/        rules, reference engine, actions, observations, replays
src/skyjo_ml/envs/        Numba batch engine, PettingZoo adapter, validation
src/skyjo_ml/models/      two-player featurizer and policy/value network
src/skyjo_ml/training/    distillation and PPO
src/skyjo_ml/evaluation/  head-to-head tournaments and confidence intervals
src/skyjo_ml/search/      belief-sampled root search
src/skyjo_ml/legacy/      archived multi-player policy adapter
artifacts/                generated runs, reports, and archived release
```
