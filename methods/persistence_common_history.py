"""Phase-3 task-1 common history with observational shadow buffers.

This implements only Amendment-002 common-history construction.  It deliberately
does not implement any continuation or ``pre_update_at_task1_admission`` logic.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any, Dict

import numpy as np
import torch

from baselines.gated_replay import GatedReplay
from methods.replay_buffer import ReplayBuffer


P3H_SCORING_MODE = "pre_update_shadow_small_loss_and_matched_random"
_RANDOM_SELECTION_SEED_OFFSET = 1_000_003
_RANDOM_BUFFER_SEED_OFFSET = 2_000_003


class PersistenceCommonHistory(GatedReplay):
    """Ordinary task-1 training observed by two non-replayed shadow buffers."""

    name = "persistence_common_history"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        cfg = dict(cfg or {})
        requested_mode = str(cfg.get("scoring_mode", P3H_SCORING_MODE))
        if requested_mode != P3H_SCORING_MODE:
            raise ValueError(
                f"PersistenceCommonHistory requires scoring_mode={P3H_SCORING_MODE}"
            )
        cfg["scoring_mode"] = "pre_update"
        super().__init__(cfg)
        if not self.gated or self.gate is None or self.gate.signal != "error":
            raise ValueError("P3H small-loss shadow requires the existing error gate")
        if self.level != "sample":
            raise ValueError("P3H small-loss shadow requires per-sample admission")
        if self.fixed_admission_prob is not None or self.oracle:
            raise ValueError("P3H small-loss shadow cannot use fixed-rate or oracle admission")

        seed = int(cfg.get("seed", 0))
        capacity = int(cfg.get("buffer_size", 500))
        self.protocol_scoring_mode = requested_mode
        self.random_buf = ReplayBuffer(
            capacity, seed=seed + _RANDOM_BUFFER_SEED_OFFSET
        )
        self._random_selection_rng = torch.Generator().manual_seed(
            seed + _RANDOM_SELECTION_SEED_OFFSET
        )
        self.batch_audit: list[Dict[str, Any]] = []
        self.incoming_stable_ids: list[int] = []
        self.admission_events = {
            "small_loss": [],
            "random_count_matched": [],
        }
        self.admission_multiplicity = {
            "small_loss": Counter(),
            "random_count_matched": Counter(),
        }

    def extra_loss(self, model, x, y, task_id):
        """Never replay either shadow buffer into the common training history."""
        return x.new_zeros(())

    def on_batch_end(self, model, x, y, task_id):
        if self._cached_pre_update_g is None or self._cached_pre_update_correct is None:
            raise RuntimeError("P3H pre-update scoring cache is missing")
        if self._batch_uids is None:
            raise RuntimeError("P3H requires stable IDs for every incoming example")

        g = self._cached_pre_update_g
        correct = self._cached_pre_update_correct
        self._cached_pre_update_g = None
        self._cached_pre_update_correct = None

        y_clean = self._batch_y_clean if self._batch_y_clean is not None else y
        uids = self._batch_uids.detach().cpu().long()
        self._task_candidates += int(x.shape[0])
        self._task_g.extend(g.detach().cpu().tolist())
        self._task_correct.extend(correct.tolist())
        self.incoming_stable_ids.extend(int(uid) for uid in uids.tolist())

        # This is the existing Phase-2 sample-level small-loss admission rule.
        small_mask = torch.rand(x.shape[0], generator=self._rng) < g.detach().cpu()
        small_idx = small_mask.nonzero(as_tuple=False).flatten()
        k = int(small_idx.numel())

        # Amendment 002: choose exactly k distinct positions from this same batch,
        # using a stream independent of training and small-loss admission RNGs.
        random_idx = torch.randperm(
            x.shape[0], generator=self._random_selection_rng
        )[:k]

        small_ids = uids[small_idx]
        random_ids = uids[random_idx]
        if k:
            small_device_idx = small_idx.to(x.device)
            random_device_idx = random_idx.to(x.device)
            self.buf.add(
                x[small_device_idx],
                y[small_device_idx],
                task_id,
                y_clean=y_clean[small_device_idx],
                stable_ids=small_ids,
            )
            self.random_buf.add(
                x[random_device_idx],
                y[random_device_idx],
                task_id,
                y_clean=y_clean[random_device_idx],
                stable_ids=random_ids,
            )

        small_event_ids = [int(uid) for uid in small_ids.tolist()]
        random_event_ids = [int(uid) for uid in random_ids.tolist()]
        self.admission_events["small_loss"].extend(small_event_ids)
        self.admission_events["random_count_matched"].extend(random_event_ids)
        self.admission_multiplicity["small_loss"].update(small_event_ids)
        self.admission_multiplicity["random_count_matched"].update(random_event_ids)
        self._task_admitted += k
        self.batch_audit.append(
            {
                "task_id": int(task_id),
                "incoming_stable_ids": [int(uid) for uid in uids.tolist()],
                "small_loss_admitted_stable_ids": small_event_ids,
                "random_admitted_stable_ids": random_event_ids,
                "small_loss_count": k,
                "random_count": len(random_event_ids),
            }
        )

    @property
    def per_minibatch_counts_match(self) -> bool:
        return all(
            row["small_loss_count"] == row["random_count"]
            for row in self.batch_audit
        )

    def consolidation_state(self, model) -> Dict[str, Any]:
        state = super().consolidation_state(model)
        state.update(
            {
                "scoring_mode": self.protocol_scoring_mode,
                "small_loss_buffer_size": len(self.buf),
                "random_count_matched_buffer_size": len(self.random_buf),
                "minibatches_observed": len(self.batch_audit),
                "per_minibatch_counts_match": self.per_minibatch_counts_match,
            }
        )
        return state

    def rng_states(self) -> Dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "small_loss_admission": self._rng.get_state(),
            "small_loss_buffer": self.buf._rng.get_state(),
            "random_selection": self._random_selection_rng.get_state(),
            "random_buffer": self.random_buf._rng.get_state(),
        }


def _buffer_state(buffer: ReplayBuffer) -> Dict[str, Any]:
    return {
        "capacity": int(buffer.capacity),
        "seen_admissions": int(buffer._seen),
        "x": torch.stack(buffer.x) if buffer.x else torch.empty(0),
        "y": torch.tensor(buffer.y, dtype=torch.long),
        "y_clean": torch.tensor(buffer.yc, dtype=torch.long),
        "task_ids": torch.tensor(buffer.t, dtype=torch.long),
        "stable_ids": torch.tensor(buffer.stable_ids, dtype=torch.long),
        "rng_state": buffer._rng.get_state(),
    }


def _fingerprint(value: Any) -> str:
    """Stable structural SHA-256 for tensors and nested state dictionaries."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda k: repr(k)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"list\0" if isinstance(item, list) else b"tuple\0")
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(b"\0")
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def build_common_history_artifact(
    *,
    cell_id: str,
    benchmark_name: str,
    condition: str,
    seed: int,
    max_train_per_task: int,
    model,
    optimizer,
    task,
    method: PersistenceCommonHistory,
) -> Dict[str, Any]:
    """Capture the complete Amendment-002 state after task-1 training."""
    model_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    optimizer_state = optimizer.state_dict()
    small_buffer = _buffer_state(method.buf)
    random_buffer = _buffer_state(method.random_buf)
    observed_labels = task.train.tensors[1].detach().cpu().clone()
    clean_labels = torch.as_tensor(task.train_y_clean, dtype=torch.long)
    corruption_mask = torch.as_tensor(task.train_is_noisy, dtype=torch.bool)
    task_stable_ids = torch.arange(len(task.train), dtype=torch.long)
    corruption = {
        "task_id": int(task.task_id),
        "stable_ids": task_stable_ids,
        "observed_labels": observed_labels,
        "clean_labels": clean_labels,
        "is_corrupted": corruption_mask,
        "noise_rate": float(task.noise_rate),
    }
    stable_ids = {
        "incoming_order": list(method.incoming_stable_ids),
        "unique_task1_ids": task_stable_ids,
        "small_loss_admission_events": list(method.admission_events["small_loss"]),
        "random_admission_events": list(
            method.admission_events["random_count_matched"]
        ),
        "small_loss_buffer": small_buffer["stable_ids"],
        "random_count_matched_buffer": random_buffer["stable_ids"],
    }
    multiplicity = {
        policy: {int(uid): int(count) for uid, count in sorted(counts.items())}
        for policy, counts in method.admission_multiplicity.items()
    }
    common_identity = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "corruption_information": corruption,
        "incoming_stable_ids": method.incoming_stable_ids,
    }
    fingerprints = {
        "model_state_sha256": _fingerprint(model_state),
        "optimizer_state_sha256": _fingerprint(optimizer_state),
        "small_loss_buffer_sha256": _fingerprint(small_buffer),
        "random_count_matched_buffer_sha256": _fingerprint(random_buffer),
        "corruption_information_sha256": _fingerprint(corruption),
        "common_history_identity_sha256": _fingerprint(common_identity),
    }
    return {
        "format_version": 1,
        "cell_id": cell_id,
        "phase": "persistence",
        "kind": "common_history",
        "benchmark": benchmark_name,
        "condition": condition,
        "recipe": "R0",
        "method": "ordinary_current_batch",
        "scoring_mode": P3H_SCORING_MODE,
        "seed": int(seed),
        "max_train_per_task": int(max_train_per_task),
        "trained_task_ids": [0],
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "shadow_buffers": {
            "small_loss": small_buffer,
            "random_count_matched": random_buffer,
        },
        "rng_states": method.rng_states(),
        "stable_ids": stable_ids,
        "admission_multiplicity": multiplicity,
        "batch_admission_audit": list(method.batch_audit),
        "corruption_information": corruption,
        "fingerprints": fingerprints,
        "per_minibatch_counts_match": method.per_minibatch_counts_match,
        "continuation_executed": False,
    }
