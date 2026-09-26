"""Amendment-003 frozen task-1 replay support.

The module is deliberately narrow: it restores and rehearses a *saved* P3H
shadow buffer, never scores or admits a task-2+ example, and exposes provenance
needed by the frozen Phase-3 continuation cells.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import torch

from methods.base import ContinualMethod
from methods.persistence_common_history import _fingerprint
from methods.replay_buffer import replay_ce_loss


BUFFER_POLICIES = {
    "small_loss_pre": "small_loss",
    "random_count_matched": "random_count_matched",
}
REPLAY_LABEL_POLICIES = {"observed", "clean_reference_oracle"}


def derive_replay_rng_seed(
    frozen_seed: int,
    buffer_policy: str,
    replay_label_policy: str,
    replay_coefficient: int,
) -> int:
    """Stable branch-local RNG seed required by Amendment 003."""
    if buffer_policy not in BUFFER_POLICIES:
        raise ValueError(f"unknown frozen-buffer policy: {buffer_policy}")
    if replay_label_policy not in REPLAY_LABEL_POLICIES:
        raise ValueError(f"unknown replay-label policy: {replay_label_policy}")
    if replay_coefficient not in {1, 2}:
        raise ValueError("frozen-buffer replay coefficient must be 1 or 2")
    material = (
        f"p3f-replay-v1|seed={int(frozen_seed)}|buffer={buffer_policy}|"
        f"labels={replay_label_policy}|lambda={int(replay_coefficient)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move an optimizer loaded from a CPU P3H artifact onto ``device``."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_parent_model_and_optimizer(
    model,
    optimizer: torch.optim.Optimizer,
    parent_artifact: dict[str, Any],
    device: torch.device,
) -> None:
    """Restore the exact task-1 model and optimizer state from P3H."""
    model.load_state_dict(parent_artifact["model_state"])
    optimizer.load_state_dict(parent_artifact["optimizer_state"])
    move_optimizer_state(optimizer, device)


class FrozenBufferReplay(ContinualMethod):
    """Replay a selected P3H buffer without changing its membership."""

    name = "frozen_buffer_replay"

    def __init__(
        self,
        *,
        parent_artifact: dict[str, Any],
        buffer_policy: str,
        replay_label_policy: str,
        replay_coefficient: int,
        frozen_seed: int,
        replay_batch_size: int = 32,
    ):
        super().__init__({})
        if buffer_policy not in BUFFER_POLICIES:
            raise ValueError(f"unknown frozen-buffer policy: {buffer_policy}")
        if replay_label_policy not in REPLAY_LABEL_POLICIES:
            raise ValueError(f"unknown replay-label policy: {replay_label_policy}")
        if replay_coefficient not in {1, 2}:
            raise ValueError("frozen-buffer replay coefficient must be 1 or 2")
        if int(replay_batch_size) <= 0:
            raise ValueError("replay batch size must be positive")

        source_name = BUFFER_POLICIES[buffer_policy]
        shadow_buffers = parent_artifact.get("shadow_buffers", {})
        if source_name not in shadow_buffers:
            raise ValueError(f"parent artifact lacks {source_name} shadow buffer")
        # The copy disconnects continuation execution from the object loaded from
        # disk and prevents any mutation of P3H-owned tensors or membership.
        self._frozen_buffer = copy.deepcopy(shadow_buffers[source_name])
        self._membership_fingerprint = _fingerprint(self._frozen_buffer)
        self.buffer_policy = buffer_policy
        self.replay_label_policy = replay_label_policy
        self.replay_coefficient = int(replay_coefficient)
        self.replay_batch_size = int(replay_batch_size)
        self.replay_rng_seed = derive_replay_rng_seed(
            frozen_seed, buffer_policy, replay_label_policy, replay_coefficient
        )
        self._replay_rng = torch.Generator().manual_seed(self.replay_rng_seed)
        self._device: torch.device | None = None
        self.replay_presentations = 0
        self.last_unscaled_replay_loss: torch.Tensor | None = None

        required = {"x", "y", "y_clean", "task_ids", "stable_ids"}
        if not required.issubset(self._frozen_buffer):
            raise ValueError("frozen buffer is missing required saved fields")
        lengths = {len(self._frozen_buffer[key]) for key in required}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("frozen buffer must contain aligned, non-empty fields")

    @property
    def frozen_buffer_fingerprint(self) -> str:
        return self._membership_fingerprint

    @property
    def current_buffer_fingerprint(self) -> str:
        return _fingerprint(self._frozen_buffer)

    @property
    def membership_unchanged(self) -> bool:
        return self.current_buffer_fingerprint == self._membership_fingerprint

    def replay_rng_state(self) -> torch.Tensor:
        return self._replay_rng.get_state().clone()

    def on_task_start(self, task_id, model) -> None:
        self._device = next(model.parameters()).device

    def sampled_indices(self, n: int | None = None) -> torch.Tensor:
        size = len(self._frozen_buffer["x"])
        count = self.replay_batch_size if n is None else int(n)
        return torch.randint(0, size, (count,), generator=self._replay_rng)

    def replay_samples_for_indices(self, indices: torch.Tensor):
        """Return standard R0 replay tuples using only the specified labels."""
        target_key = "y" if self.replay_label_policy == "observed" else "y_clean"
        return [
            (
                self._frozen_buffer["x"][int(index)],
                int(self._frozen_buffer[target_key][int(index)]),
                int(self._frozen_buffer["task_ids"][int(index)]),
            )
            for index in indices.tolist()
        ]

    def extra_loss(self, model, x, y, task_id):
        if self._device is None:
            raise RuntimeError("frozen replay has not been attached to a model")
        indices = self.sampled_indices()
        samples = self.replay_samples_for_indices(indices)
        raw_loss = replay_ce_loss(model, samples, self._device)
        self.last_unscaled_replay_loss = raw_loss.detach()
        self.replay_presentations += len(samples)
        return raw_loss * self.replay_coefficient

    def on_batch_end(self, model, x, y, task_id) -> None:
        # Amendment 003: no rescoring, no task-2+ admissions, and no reservoir
        # replacement.  The method intentionally does not expose an add path.
        if not self.membership_unchanged:
            raise RuntimeError("frozen-buffer membership changed during continuation")

    def provenance(self) -> dict[str, Any]:
        return {
            "buffer_policy": self.buffer_policy,
            "replay_label_policy": self.replay_label_policy,
            "replay_coefficient": self.replay_coefficient,
            "replay_rng_seed": self.replay_rng_seed,
            "replay_rng_state": self.replay_rng_state().tolist(),
            "frozen_buffer_fingerprint": self.frozen_buffer_fingerprint,
            "frozen_buffer_fingerprint_after": self.current_buffer_fingerprint,
            "buffer_membership_unchanged": self.membership_unchanged,
            "new_admissions": 0,
            "replacements": 0,
        }
