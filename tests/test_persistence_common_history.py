from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from baselines.finetune import Finetune
from methods.persistence_common_history import (
    P3H_SCORING_MODE,
    PersistenceCommonHistory,
    build_common_history_artifact,
)
from methods.trainer import ContinualTrainer


class _Task:
    def __init__(self):
        generator = torch.Generator().manual_seed(91)
        x = torch.randn(12, 4, generator=generator)
        y = torch.tensor([0, 1] * 6, dtype=torch.long)
        self.task_id = 0
        self.name = "task0"
        self.global_classes = [0, 1]
        self.n_classes = 2
        self.train = TensorDataset(x, y)
        self.val = TensorDataset(x[:6], y[:6])
        self.test = TensorDataset(x[6:], y[6:])
        self.train_y_clean = y.numpy().copy()
        self.train_is_noisy = np.zeros(len(y), dtype=bool)
        self.noise_rate = 0.0


class _Bench:
    def __init__(self):
        self.tasks = [_Task()]
        self.n_classes_per_task = 2
        self.name = "tiny"
        self.multihead = False


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x, task_id):
        return self.fc(x)

    def features(self, x):
        return x

    def head(self, task_id):
        return self.fc


def _method(seed=7, capacity=20):
    return PersistenceCommonHistory(
        {
            "name": "persistence_common_history",
            "seed": seed,
            "buffer_size": capacity,
            "batch_size": 4,
            "error_gated": True,
            "gate_level": "sample",
            "gate": {"signal": "error", "error_threshold": 1.0, "gamma": 3.0},
            "scoring_mode": P3H_SCORING_MODE,
        }
    )


def _observe_one_batch(method, model, task):
    x, y = task.train.tensors
    ids = torch.arange(len(y), dtype=torch.long)
    method.on_task_start(0, model)
    method.observe_batch(0, ids, y)
    method.before_update(model, x, y, 0)
    method.on_batch_end(model, x, y, 0)
    return ids


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_shadows_share_minibatch_and_match_exact_count():
    task = _Task()
    model = _Model()
    method = _method()
    ids = _observe_one_batch(method, model, task)

    assert len(method.batch_audit) == 1
    audit = method.batch_audit[0]
    assert audit["incoming_stable_ids"] == ids.tolist()
    assert set(audit["small_loss_admitted_stable_ids"]).issubset(set(ids.tolist()))
    assert set(audit["random_admitted_stable_ids"]).issubset(set(ids.tolist()))
    assert audit["small_loss_count"] == audit["random_count"]
    assert method.per_minibatch_counts_match


def test_random_shadow_selection_is_deterministic_for_fixed_seed():
    task = _Task()
    initial = _Model().state_dict()
    selections = []
    for _ in range(2):
        model = _Model()
        model.load_state_dict(initial)
        method = _method(seed=23)
        _observe_one_batch(method, model, task)
        selections.append(method.admission_events["random_count_matched"])
    assert selections[0] == selections[1]


def test_shadow_construction_preserves_model_and_optimizer_trajectory():
    bench = _Bench()
    initial_model = _Model()
    initial_state = copy.deepcopy(initial_model.state_dict())
    cfg = {
        "train": {"epochs": 2, "batch_size": 4, "eval_batch_size": 4, "probe_size": 4},
        "optimizer": {"name": "adam", "lr": 1e-3},
    }

    plain_model = _Model()
    plain_model.load_state_dict(initial_state)
    torch.manual_seed(501)
    plain_trainer = ContinualTrainer(
        cfg, bench, plain_model, Finetune({}), torch.device("cpu")
    )
    plain_trainer.train()

    shadow_model = _Model()
    shadow_model.load_state_dict(initial_state)
    torch.manual_seed(501)
    shadow_trainer = ContinualTrainer(
        cfg, bench, shadow_model, _method(seed=9), torch.device("cpu")
    )
    shadow_trainer.train()

    _assert_nested_equal(plain_model.state_dict(), shadow_model.state_dict())
    _assert_nested_equal(
        plain_trainer.optimizer.state_dict(), shadow_trainer.optimizer.state_dict()
    )


def test_shadow_buffers_are_independent():
    task = _Task()
    model = _Model()
    method = _method(capacity=50)
    _observe_one_batch(method, model, task)

    assert method.buf is not method.random_buf
    assert method.buf.x is not method.random_buf.x
    random_size = len(method.random_buf)
    x, y = task.train.tensors
    method.buf.add(x[:1], y[:1], 0, y_clean=y[:1], stable_ids=torch.tensor([999]))
    assert len(method.random_buf) == random_size
    assert 999 not in method.random_buf.stable_ids


def test_saved_artifact_has_all_amendment_002_fields(tmp_path):
    bench = _Bench()
    model = _Model()
    method = _method(seed=4)
    cfg = {
        "train": {"epochs": 1, "batch_size": 4, "eval_batch_size": 4,
                  "probe_size": 4, "max_tasks": 1},
        "optimizer": {"name": "adam", "lr": 1e-3},
    }
    torch.manual_seed(77)
    trainer = ContinualTrainer(cfg, bench, model, method, torch.device("cpu"))
    trainer.train()
    artifact = build_common_history_artifact(
        cell_id="P3H__tiny__task1_sym40__s4",
        benchmark_name="tiny",
        condition="task1_sym40",
        seed=4,
        max_train_per_task=12,
        model=model,
        optimizer=trainer.optimizer,
        task=bench.tasks[0],
        method=method,
    )
    path = tmp_path / "common_history.pt"
    torch.save(artifact, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)

    required = {
        "model_state",
        "optimizer_state",
        "shadow_buffers",
        "rng_states",
        "stable_ids",
        "admission_multiplicity",
        "corruption_information",
        "fingerprints",
    }
    assert required.issubset(saved)
    assert set(saved["shadow_buffers"]) == {"small_loss", "random_count_matched"}
    assert saved["trained_task_ids"] == [0]
    assert saved["per_minibatch_counts_match"] is True
    assert saved["continuation_executed"] is False
