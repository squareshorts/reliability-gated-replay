"""The frozen Phase-4 R1 training recipe.

This module deliberately keeps the R1-only optimization and image-view policy
outside the legacy R0 data/training path.  R1 stores canonical CIFAR images as
unaugmented ``[0, 1]`` tensors, applies its augmentation only at training use,
and normalizes the score/evaluation view with the same statistics used by R0.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict

import torch
import torch.nn.functional as F


R1_RECIPE_ID = "R1"

# These values are the full resolved recipe settings specified by Amendment 004.
R1_OPTIMIZER: Dict[str, Any] = {
    "name": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "nesterov": False,
}
R1_TRAIN: Dict[str, Any] = {
    "epochs": 50,
    "batch_size": 128,
    "replay_batch_size": 32,
    "buffer_capacity": 500,
    "replay_loss_coefficient": 1.0,
    "max_train_per_task": 2500,
}
R1_SCHEDULER: Dict[str, Any] = {
    "name": "cosine_annealing",
    "t_max": 50,
    "eta_min": 0.0,
    "step_timing": "after_each_epoch",
    "reset_at_task_boundary": True,
}
R1_VIEWS: Dict[str, Any] = {
    "canonical_storage": "unaugmented_float_range_0_to_1",
    "training": {
        "random_crop_size": 32,
        "padding": 4,
        "horizontal_flip_probability": 0.5,
        "normalization": "r0_cifar10",
    },
    "gate_scoring": "r0_cifar10_normalization_only",
    "validation_testing": "r0_cifar10_normalization_only",
    "augmentation_rng": "independent_deterministic_per_task",
}

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def resolved_r1_recipe() -> Dict[str, Any]:
    """Return a JSON-serializable, canonical description of the R1 recipe."""
    return {
        "recipe_id": R1_RECIPE_ID,
        "architecture": "scratch_resnet18",
        "optimizer": copy.deepcopy(R1_OPTIMIZER),
        "scheduler": copy.deepcopy(R1_SCHEDULER),
        "training": copy.deepcopy(R1_TRAIN),
        "views": copy.deepcopy(R1_VIEWS),
    }


def r1_recipe_hash() -> str:
    """Stable SHA-256 of the resolved (not run-specific) R1 recipe."""
    encoded = json.dumps(resolved_r1_recipe(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def augmentation_seed(run_seed: int, task_id: int) -> int:
    """Derive a task-local stream without touching any global RNG state."""
    material = f"R1/augmentation/v1/{int(run_seed)}/{int(task_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _state_hash(state: torch.Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class R1Cifar10Views:
    """Canonical, training, and score/evaluation views for Amendment-004 R1.

    The input is always a canonical unaugmented CIFAR tensor in ``[0, 1]``.
    Separate explicit generators (one per task) keep augmentation independent
    from data-order, corruption, admission, and replay-sampling RNG streams.
    """

    def __init__(self, run_seed: int):
        self.run_seed = int(run_seed)
        self._generators: Dict[int, torch.Generator] = {}

    @staticmethod
    def _normalization_tensors(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("R1 CIFAR-10 views require [batch, 3, height, width] tensors")
        mean = torch.tensor(_CIFAR10_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_CIFAR10_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return mean, std

    def _generator(self, task_id: int) -> torch.Generator:
        task_id = int(task_id)
        generator = self._generators.get(task_id)
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(augmentation_seed(self.run_seed, task_id))
            self._generators[task_id] = generator
        return generator

    def score_eval_view(self, canonical_x: torch.Tensor) -> torch.Tensor:
        """Normalization-only view used for admission scores and evaluation."""
        mean, std = self._normalization_tensors(canonical_x)
        return (canonical_x - mean) / std

    def training_view(self, canonical_x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Crop/pad/flip then normalize, drawing only from the task-local stream."""
        if canonical_x.ndim != 4 or tuple(canonical_x.shape[-2:]) != (32, 32):
            raise ValueError("R1 random crop requires canonical CIFAR-10 [*, 3, 32, 32] inputs")
        original_dtype = canonical_x.dtype
        # Random draws stay on an explicit CPU generator.  The image operations
        # remain on the input device, so CUDA training does not round-trip every
        # batch through CPU and the global CUDA RNG remains untouched.
        x = canonical_x.detach().to(dtype=torch.float32)
        padded = F.pad(x, (4, 4, 4, 4), mode="constant", value=0.0)
        generator = self._generator(task_id)
        n = int(x.shape[0])
        rows = torch.randint(0, 9, (n,), generator=generator)
        cols = torch.randint(0, 9, (n,), generator=generator)
        crops = torch.stack(
            [padded[i, :, int(rows[i]) : int(rows[i]) + 32, int(cols[i]) : int(cols[i]) + 32]
             for i in range(n)],
            dim=0,
        )
        flip = (torch.rand(n, generator=generator) < 0.5).to(crops.device)
        if bool(flip.any()):
            crops[flip] = torch.flip(crops[flip], dims=(-1,))
        normalized = self.score_eval_view(crops)
        return normalized.to(dtype=original_dtype)

    def provenance(self) -> Dict[str, Any]:
        """RNG metadata safe to persist in a JSON result artifact."""
        task_seeds = {str(task_id): augmentation_seed(self.run_seed, task_id)
                      for task_id in sorted(self._generators)}
        task_state_hashes = {str(task_id): _state_hash(generator.get_state())
                             for task_id, generator in sorted(self._generators.items())}
        return {
            "run_seed": self.run_seed,
            "per_task_seeds": task_seeds,
            "per_task_final_state_sha256": task_state_hashes,
            "stream_isolation": "explicit_cpu_generators_only",
        }
