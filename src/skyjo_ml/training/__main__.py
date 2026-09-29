import argparse

from .ppo import PPOTrainer, TrainConfig

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--run-name")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    cfg = TrainConfig.from_toml(a.config, run_name=a.run_name, seed=a.seed)
    PPOTrainer(cfg, resume=a.resume).train()
