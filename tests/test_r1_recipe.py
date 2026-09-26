"""Focused Amendment-004 R1 coverage."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from baselines.gated_replay import GatedReplay
from baselines.replay import Replay
from methods.r1_recipe import R1Cifar10Views, R1_OPTIMIZER, R1_RECIPE_ID, R1_TRAIN
from methods.trainer import ContinualTrainer
from methods.utils import evaluate
from scripts.run_nn_submission import _r1_provenance, _run_id, resolve_r1_frozen_cell


class _Task:
    def __init__(self, task_id: int, x: torch.Tensor, y: torch.Tensor):
        self.task_id = task_id
        self.name = f"tiny_r1_task_{task_id}"
        self.global_classes = [0, 1]
        self.n_classes = 2
        self.train = torch.utils.data.TensorDataset(x.clone(), y.clone())
        self.val = torch.utils.data.TensorDataset(x.clone(), y.clone())
        self.test = torch.utils.data.TensorDataset(x.clone(), y.clone())
        self.train_y_clean = y.numpy().copy()
        self.train_is_noisy = np.zeros(len(y), dtype=bool)
        self.noise_rate = 0.0


class _Bench:
    def __init__(self, n_tasks: int = 2):
        base = torch.linspace(0.0, 1.0, 3 * 32 * 32, dtype=torch.float32).view(1, 3, 32, 32)
        self.tasks = [_Task(task_id, base + task_id * 0.001, torch.tensor([task_id % 2]))
                      for task_id in range(n_tasks)]
        self.n_classes_per_task = 2
        self.name = "tiny_r1"
        self.multihead = True


class _ImageModel(nn.Module):
    multihead = True

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 2)
        self.eval_inputs = []

    def forward(self, x, task_id):
        if not self.training:
            self.eval_inputs.append(x.detach().clone())
        return self.fc(x.flatten(1))

    def features(self, x):
        return x.flatten(1)

    def head(self, task_id):
        return self.fc


class _StartTrackingReplay(Replay):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.task_start_weights = []

    def on_task_start(self, task_id, model):
        super().on_task_start(task_id, model)
        self.task_start_weights.append(model.fc.weight.detach().clone())


class _RecordingGate:
    signal = "error"

    def __init__(self):
        self.seen = []

    def reset_task(self, task_id):
        pass

    def score(self, model, x, y, task_id, uids=None):
        self.seen.append(x.detach().clone())
        return torch.ones(x.shape[0], device=x.device), {"signal": "error"}


def _r1_cfg(seed: int = 0, max_tasks: int | None = None):
    train = {
        "epochs": 50,
        "batch_size": 128,
        "eval_batch_size": 8,
        "probe_size": 1,
        "seed": seed,
    }
    if max_tasks is not None:
        train["max_tasks"] = max_tasks
    return {"recipe": R1_RECIPE_ID, "seed": seed, "train": train,
            "optimizer": copy.deepcopy(R1_OPTIMIZER)}


def test_r0_path_and_identity_remain_unchanged():
    bench = _Bench(n_tasks=1)
    model = _ImageModel()
    method = Replay({"name": "replay", "buffer_size": 10, "batch_size": 1})
    result = ContinualTrainer(
        {"train": {"epochs": 1, "batch_size": 1, "eval_batch_size": 1, "probe_size": 1},
         "optimizer": {"name": "sgd", "lr": 0.1}},
        bench, model, method, torch.device("cpu"),
    ).train()
    assert "r1_training" not in result
    assert "optimization_steps" not in result
    assert _run_id("split_cifar10", "sym20", "er", 0, "post_update") == "split_cifar10__sym20__er__seed0"


def test_r1_optimizer_and_scheduler_match_amendment_004():
    trainer = ContinualTrainer(
        _r1_cfg(max_tasks=1), _Bench(n_tasks=1), _ImageModel(),
        Replay({"name": "replay", "buffer_size": 500, "batch_size": 32,
                "replay_loss_coefficient": 1.0}),
        torch.device("cpu"),
    )
    optimizer, scheduler = trainer._build_optimizer_and_scheduler()
    assert isinstance(optimizer, torch.optim.SGD)
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.1
    assert group["momentum"] == 0.9
    assert group["weight_decay"] == 0.0005
    assert group["nesterov"] is False
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 50
    assert scheduler.eta_min == 0.0
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.0, abs_tol=1e-12)


def test_r1_resets_optimizer_scheduler_but_not_model_weights():
    bench = _Bench(n_tasks=2)
    model = _ImageModel()
    method = _StartTrackingReplay(
        {"name": "replay", "buffer_size": 500, "batch_size": 32,
         "replay_loss_coefficient": 1.0}
    )
    result = ContinualTrainer(_r1_cfg(), bench, model, method, torch.device("cpu")).train()
    assert len(method.task_start_weights) == 2
    assert not torch.equal(method.task_start_weights[0], method.task_start_weights[1])
    metadata = result["r1_training"]["optimizer_task_metadata"]
    assert [entry["initial_state_entries"] for entry in metadata] == [0, 0]
    assert [entry["scheduler_t_max"] for entry in metadata] == [50, 50]
    assert result["r1_training"]["scheduler_steps_per_task"] == [50, 50]
    assert result["optimization_steps"]["per_task"] == [50, 50]


def test_r1_training_augmentation_does_not_change_gate_or_eval_view():
    bench = _Bench(n_tasks=1)
    model = _ImageModel()
    method = GatedReplay({
        "name": "gated_replay", "buffer_size": 500, "batch_size": 32,
        "replay_loss_coefficient": 1.0, "scoring_mode": "pre_update",
        "gate": {"signal": "error", "error_threshold": 1.0, "gamma": 1.0},
    })
    gate = _RecordingGate()
    method.gate = gate
    trainer = ContinualTrainer(_r1_cfg(max_tasks=1), bench, model, method, torch.device("cpu"))
    canonical = bench.tasks[0].train.tensors[0]
    expected = trainer.r1_views.score_eval_view(canonical)
    augmented = trainer.r1_views.training_view(canonical, task_id=0)
    assert augmented.shape == expected.shape
    trainer.train()
    assert torch.equal(gate.seen[0], expected)
    model.eval_inputs.clear()
    evaluate(model, bench.tasks[0].test, 0, torch.device("cpu"), input_transform=trainer._score_eval_view)
    assert torch.equal(model.eval_inputs[0], expected)


def test_r1_stores_canonical_unaugmented_images_in_replay_buffer():
    bench = _Bench(n_tasks=1)
    source = bench.tasks[0].train.tensors[0]
    method = Replay({"name": "replay", "buffer_size": 500, "batch_size": 32,
                     "replay_loss_coefficient": 1.0})
    ContinualTrainer(
        _r1_cfg(max_tasks=1), bench, _ImageModel(), method, torch.device("cpu")
    ).train()
    assert method.buf.x
    assert all(any(torch.equal(stored, original) for original in source) for stored in method.buf.x)


def test_r1_augmentation_rng_does_not_perturb_data_or_admission_rngs():
    canonical = _Bench(n_tasks=1).tasks[0].train.tensors[0]
    torch.manual_seed(818)
    data_order_before = torch.get_rng_state().clone()
    admission_rng = torch.Generator().manual_seed(919)
    admission_before = admission_rng.get_state().clone()
    numpy_before = np.random.get_state()
    views = R1Cifar10Views(run_seed=3)
    views.training_view(canonical, task_id=0)
    assert torch.equal(torch.get_rng_state(), data_order_before)
    assert torch.equal(admission_rng.get_state(), admission_before)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_r1_frozen_identity_is_distinct_from_r0_identity():
    row = resolve_r1_frozen_cell("split_cifar10", "sym20", "small_loss_pre", 0)
    assert row["cell_id"] == "P4__split_cifar10__sym20__R1__small_loss_pre__s0"
    assert row["cell_id"] != _run_id("split_cifar10", "sym20", "small_loss_pre", 0, "pre_update")
    assert _run_id("split_cifar10", "sym20", "small_loss_pre", 0, "post_update", recipe="R1") == row["cell_id"]


def test_r1_required_provenance_fields_are_written():
    bench = _Bench(n_tasks=1)
    row = resolve_r1_frozen_cell("split_cifar10", "sym20", "small_loss_pre", 0)
    provenance = _r1_provenance(
        data_cfg={"name": "split_cifar10", "normalize": False, "max_train_per_task": 2500},
        train_cfg={"epochs": 50, "batch_size": 128, "seed": 0},
        method_cfg={"name": "gated_replay", "buffer_size": 500, "batch_size": 32,
                    "replay_loss_coefficient": 1.0},
        bench=bench,
        seed=0,
        cell_id=row["cell_id"],
        plan_row=row,
        runtime_s=12.5,
    )
    required = {
        "recipe_id", "recipe_sha256", "frozen_cell_id", "frozen_plan_row", "seed",
        "source_version", "data_fingerprint", "corruption_fingerprint", "runtime_s",
        "full_resolved_configuration", "canonical_buffer_storage", "gate_scoring_view",
        "evaluation_view",
    }
    assert required <= set(provenance)
    assert provenance["recipe_id"] == "R1"
    assert provenance["full_resolved_configuration"]["optimizer"] == R1_OPTIMIZER
