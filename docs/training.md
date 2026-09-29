# Two-player training

The current network and observation schema are specialised for two-player Skyjo. The earlier 2–8-player release and its training configuration are preserved under `artifacts/legacy_multiplayer/`; its weights are used only as a teacher and benchmark opponent.

## Warm start

`src/skyjo_ml/training/distill.py` generated 1,406,438 decisions from 4,339 completed two-player matches. Seats used a mix of sampled and greedy legacy-model play plus finish, tuned finish, and column heuristics. `finish_tuned` uses the earlier multi-player tuned parameters to preserve this run's configuration; `finish_tuned_2p` uses the new two-player tuning. The teacher supplied a legal-action probability distribution at every recorded decision. The two-player network trained for four epochs on soft cross-entropy and terminal win-credit value targets. The held-out argmax agreement after epoch four was 98.28%; held-out KL was 0.0063. The resulting checkpoint is `artifacts/distill/distill_2p.pt`.

## PPO

Three independent seeds used `configs/ppo_2p.toml`, each for about 8.0 million learner decisions across 244 updates. The reward was terminal match win credit, with no round-score shaping. Each match contained learner self-play or opponents drawn from frozen learner snapshots, the legacy release, and finish heuristics (including the earlier multi-player tuned parameters). The legacy policy was also the fixed opponent for periodic dev evaluation. Each seed kept its best dev checkpoint separately from its final checkpoint.

Training used 256 concurrent matches, 32,768 learner decisions per update, four PPO epochs, minibatches of 2,048, and one PyTorch thread per process. All three runs completed without nonfinite metrics. The source run logs, manifests, and checkpoints are under `artifacts/runs/ppo2p_s{1,2,3}/`.

## Selection and reporting

Checkpoint selection uses a separate dev seed block with both seat rotations against the greedy legacy model. Release seeds are reserved for the selected model's final evaluation and are not used for tuning or choosing a checkpoint. The ruleset hash and observation schema hash are embedded in exported packages; the loader checks both.
