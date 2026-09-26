from __future__ import annotations

import copy
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.frozen_buffer_continuation import (
    FrozenBufferReplay,
    restore_parent_model_and_optimizer,
)
from methods.persistence_common_history import _fingerprint
from methods.replay_buffer import replay_ce_loss


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x, task_id):
        return self.linear(x)


def _nested_equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def _parent_artifact():
    torch.manual_seed(14)
    model = _Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.tensor([[0.2, -0.1], [0.3, 0.7]], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    F.cross_entropy(model(x, 0), y).backward()
    optimizer.step()
    buffer = {
        "capacity": 4,
        "seen_admissions": 9,
        "x": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.2], [0.3, 0.8]]),
        "y": torch.tensor([0, 1, 0, 1]),
        "y_clean": torch.tensor([1, 1, 0, 0]),
        "task_ids": torch.tensor([0, 0, 0, 0]),
        "stable_ids": torch.tensor([10, 11, 12, 13]),
        "rng_state": torch.Generator().manual_seed(5).get_state(),
    }
    return {
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "shadow_buffers": {
            "small_loss": copy.deepcopy(buffer),
            "random_count_matched": copy.deepcopy(buffer),
        },
    }


def _method(parent, *, labels="observed", coefficient=1):
    return FrozenBufferReplay(
        parent_artifact=parent,
        buffer_policy="small_loss_pre",
        replay_label_policy=labels,
        replay_coefficient=coefficient,
        frozen_seed=7,
        replay_batch_size=3,
    )


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4096), b""):
            digest.update(block)
    return digest.hexdigest()


def test_parent_model_and_optimizer_are_restored():
    parent = _parent_artifact()
    restored_model = _Model()
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    for parameter in restored_model.parameters():
        parameter.data.zero_()

    restore_parent_model_and_optimizer(restored_model, restored_optimizer, parent, torch.device("cpu"))

    assert _nested_equal(restored_model.state_dict(), parent["model_state"])
    assert _nested_equal(restored_optimizer.state_dict(), parent["optimizer_state"])


def test_frozen_buffer_is_unchanged_through_continuation_steps():
    parent = _parent_artifact()
    method = _method(parent)
    model = _Model()
    method.on_task_start(1, model)
    before = method.frozen_buffer_fingerprint
    for _ in range(3):
        loss = method.extra_loss(model, torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), 1)
        loss.backward()
        model.zero_grad(set_to_none=True)
        method.on_batch_end(model, torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), 1)

    assert method.membership_unchanged
    assert method.current_buffer_fingerprint == before
    assert _fingerprint(parent["shadow_buffers"]["small_loss"]) == before


def test_observed_and_oracle_change_only_the_replay_targets_for_same_indices():
    parent = _parent_artifact()
    observed = _method(parent, labels="observed")
    oracle = _method(parent, labels="clean_reference_oracle")
    indices = torch.tensor([0, 1, 3])
    observed_samples = observed.replay_samples_for_indices(indices)
    oracle_samples = oracle.replay_samples_for_indices(indices)

    for (obs_x, obs_y, obs_task), (oracle_x, oracle_y, oracle_task), index in zip(
        observed_samples, oracle_samples, indices.tolist()
    ):
        assert torch.equal(obs_x, oracle_x)
        assert obs_task == oracle_task == 0
        assert obs_y == int(parent["shadow_buffers"]["small_loss"]["y"][index])
        assert oracle_y == int(parent["shadow_buffers"]["small_loss"]["y_clean"][index])
    assert observed.frozen_buffer_fingerprint == oracle.frozen_buffer_fingerprint


def test_lambda_scales_only_the_replay_loss_term():
    parent = _parent_artifact()
    model = _Model()
    indices = torch.tensor([0, 2, 3])
    lambda1 = _method(parent, coefficient=1)
    lambda2 = _method(parent, coefficient=2)
    replay = replay_ce_loss(model, lambda1.replay_samples_for_indices(indices), torch.device("cpu"))
    current = F.cross_entropy(model(torch.tensor([[0.4, 0.6]]), 1), torch.tensor([1]))
    total1 = current + lambda1.replay_coefficient * replay
    total2 = current + lambda2.replay_coefficient * replay

    assert torch.allclose(total1 - current, replay)
    assert torch.allclose(total2 - current, 2 * replay)
    assert torch.allclose((total2 - total1), replay)
    assert torch.allclose(current, F.cross_entropy(model(torch.tensor([[0.4, 0.6]]), 1), torch.tensor([1])))


def test_replay_rng_does_not_perturb_global_data_order_rng():
    parent = _parent_artifact()
    method = _method(parent)
    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    method.sampled_indices()
    after = torch.get_rng_state()

    assert torch.equal(before, after)


def test_parent_p3h_artifact_file_remains_unchanged(tmp_path):
    parent = _parent_artifact()
    artifact_path = tmp_path / "parent.pt"
    torch.save(parent, artifact_path)
    before = _file_hash(artifact_path)
    loaded = torch.load(artifact_path, map_location="cpu", weights_only=False)
    method = _method(loaded)
    model = _Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    restore_parent_model_and_optimizer(model, optimizer, loaded, torch.device("cpu"))
    method.on_task_start(1, model)
    method.extra_loss(model, torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), 1)
    method.on_batch_end(model, torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), 1)

    assert _file_hash(artifact_path) == before
