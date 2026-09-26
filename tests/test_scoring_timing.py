import pytest
import torch
import torch.nn as nn
from baselines.gated_replay import GatedReplay
from methods.trainer import ContinualTrainer

class DummyTask:
    def __init__(self):
        self.task_id = 0
        self.global_classes = list(range(2))
        x = torch.randn(2, 10)
        y = torch.tensor([0, 1])
        dataset = torch.utils.data.TensorDataset(x, y)
        self.train = dataset
        self.val = dataset
        self.test = dataset
        self.train_y_clean = y.numpy()
        self.name = "dummy"

class DummyBench:
    def __init__(self):
        self.tasks = [DummyTask()]
        self.n_classes_per_task = [2]
        self.name = "dummy_bench"
        self.multihead = False

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x, task_id):
        return self.fc(x)

    def features(self, x):
        return x

    def head(self, task_id):
        return self.fc

class SpyGate:
    def __init__(self):
        self.calls = 0
        self.observed_weights = []
        self.signal = "error"

    def score(self, model, x, y, task_id, uids=None):
        self.calls += 1
        self.observed_weights.append(model.fc.weight.clone().detach())
        return torch.ones(x.shape[0]), {"signal": "error", "g_mean": 1.0, "raw_mean": 1.0}

    def reset_task(self, task_id):
        pass

def test_scoring_timing():
    torch.manual_seed(42)
    model_pre = DummyModel()
    bench = DummyBench()

    # Pre-update
    cfg_pre = {
        "name": "gated_replay",
        "buffer_size": 10,
        "batch_size": 2,
        "error_gated": True,
        "gate_level": "sample",
        "scoring_mode": "pre_update",
        "gate": {"signal": "error", "error_threshold": 1.0, "gamma": 1.0}
    }

    method_pre = GatedReplay(cfg_pre)
    method_pre.gate = SpyGate()

    initial_weight = model_pre.fc.weight.clone().detach()
    initial_state = {k: v.clone() for k, v in model_pre.state_dict().items()}

    rng_state = torch.get_rng_state()
    trainer_pre = ContinualTrainer(
        {"train": {"epochs": 1, "batch_size": 2, "eval_batch_size": 2, "probe_size": 2}, "optimizer": {"name": "sgd", "lr": 1.0}},
        bench, model_pre, method_pre, torch.device("cpu")
    )
    trainer_pre.train()

    assert method_pre.gate.calls == 1, "Score called exactly once"
    assert torch.allclose(method_pre.gate.observed_weights[0], initial_weight), "Pre-update observes BEFORE-update state"

    updated_weight = model_pre.fc.weight.clone().detach()
    assert not torch.allclose(initial_weight, updated_weight), "Optimizer step must change parameters"

    assert method_pre._cached_pre_update_g is None, "Cache cleared afterward"
    assert method_pre._cached_pre_update_correct is None, "Cache cleared afterward"

    # Post-update
    cfg_post = dict(cfg_pre)
    cfg_post["scoring_mode"] = "post_update"
    model_post = DummyModel()
    model_post.load_state_dict(initial_state)

    method_post = GatedReplay(cfg_post)
    method_post.gate = SpyGate()

    torch.set_rng_state(rng_state)

    trainer_post = ContinualTrainer(
        {"train": {"epochs": 1, "batch_size": 2, "eval_batch_size": 2, "probe_size": 2}, "optimizer": {"name": "sgd", "lr": 1.0}},
        bench, model_post, method_post, torch.device("cpu")
    )
    trainer_post.train()

    assert method_post.gate.calls == 1, "Score called exactly once"

    post_updated_weight = model_post.fc.weight.clone().detach()
    assert torch.allclose(updated_weight, post_updated_weight), "Deterministic update"

    assert torch.allclose(method_post.gate.observed_weights[0], post_updated_weight), "Post-update observes AFTER-update state"
