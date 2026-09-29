import numpy as np
import torch

from skyjo_ml.agents.neural import NeuralBatchPolicy, load_model
from skyjo_ml.envs.batch import BatchEnv
from skyjo_ml.export import export
from skyjo_ml.models.net import SkyjoNet


def test_export_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = SkyjoNet()
    ck = tmp_path / "ck.pt"
    torch.save({"model": net.state_dict(), "net_config": net.cfg.to_dict()}, ck)
    meta = export(str(ck), str(tmp_path / "pkg.pt"))
    assert len(meta["sha256"]) == 64
    net2, _ = load_model(tmp_path / "pkg.pt")
    env = BatchEnv(16)
    env.reset(np.arange(16), np.arange(16), np.full(16, 2))
    obs, masks = env.observe()
    a1 = np.zeros(16, dtype=np.int64)
    a2 = np.zeros(16, dtype=np.int64)
    rows = np.arange(16)
    NeuralBatchPolicy(net, greedy=True).act_batch(obs, masks, rows, a1)
    NeuralBatchPolicy(net2, greedy=True).act_batch(obs, masks, rows, a2)
    assert (a1 == a2).all() and masks[rows, a1].all()
