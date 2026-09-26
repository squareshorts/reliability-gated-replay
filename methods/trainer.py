"""The unified continual-learning training loop.

Drives any :class:`ContinualMethod` over a :class:`ContinualBenchmark`, filling
the T x T accuracy matrix ``R[i][j]`` (test acc on task j after training task i)
and recording everything the analysis layer needs: per-epoch closure/plasticity
signals, weight drift, probe representations (for CKA), consolidation state,
timing and memory.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.base import ContinualBenchmark
from utils.logging import get_logger

from .base import ContinualMethod
from .r1_recipe import R1Cifar10Views, R1_OPTIMIZER, R1_RECIPE_ID, R1_SCHEDULER, R1_TRAIN
from .utils import build_optimizer, evaluate, protected_named_parameters

# stable per-example uid = task offset + within-task index (keeps the
# reliability-gate EMA slots unique across tasks).
_UID_TASK_STRIDE = 10_000_000


class _IndexedDataset(torch.utils.data.Dataset):
    """Wrap a (x, y) dataset to also return the example's index."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, i


class ContinualTrainer:
    def __init__(
        self,
        cfg: Dict[str, Any],
        benchmark: ContinualBenchmark,
        model: nn.Module,
        method: ContinualMethod,
        device: torch.device,
        logger=None,
    ):
        self.cfg = cfg
        self.bench = benchmark
        self.model = model.to(device)
        self.method = method
        self.device = device
        self.log = logger or get_logger("trainer")

        tcfg = cfg.get("train", {})
        self.max_epochs = int(tcfg.get("epochs", 5))
        self.batch_size = int(tcfg.get("batch_size", 128))
        self.eval_batch_size = int(tcfg.get("eval_batch_size", 256))
        self.probe_size = int(tcfg.get("probe_size", 200))
        self.grad_clip = tcfg.get("grad_clip")
        self.max_tasks = min(len(self.bench.tasks), int(tcfg.get("max_tasks", len(self.bench.tasks))))
        if self.max_tasks < 1:
            raise ValueError("train.max_tasks must be at least 1")
        self.optimizer = None
        self.scheduler = None
        self.recipe = str(cfg.get("recipe", "R0"))
        if self.recipe not in {"R0", R1_RECIPE_ID}:
            raise ValueError(f"Unknown training recipe '{self.recipe}'")
        self.r1_views = None
        if self.recipe == R1_RECIPE_ID:
            self._validate_r1_configuration()
            self.r1_views = R1Cifar10Views(int(tcfg.get("seed", cfg.get("seed", 0))))
            # The planned R1 methods are replay/gated replay.  These optional
            # setters keep the legacy method interface and R0 path unchanged.
            if hasattr(self.method, "set_replay_input_transform"):
                self.method.set_replay_input_transform(self.r1_views.training_view)
            if hasattr(self.method, "set_gate_input_transform"):
                self.method.set_gate_input_transform(self.r1_views.score_eval_view)

        # snapshot init weights by name (params may later be frozen, e.g. prog_freeze)
        self._init_named = {
            n: p.detach().clone() for n, p in protected_named_parameters(self.model)
        }
        self._probe_x = self._build_probe()

    def _build_probe(self) -> torch.Tensor:
        x, _ = self.bench.tasks[0].test.tensors
        x = x[: self.probe_size]
        if self.r1_views is not None:
            x = self.r1_views.score_eval_view(x)
        return x.to(self.device)

    def _validate_r1_configuration(self) -> None:
        """Reject accidental R1 overrides instead of silently changing the recipe."""
        optimizer_cfg = self.cfg.get("optimizer", {})
        expected_optimizer = R1_OPTIMIZER
        for key, expected in expected_optimizer.items():
            actual = optimizer_cfg.get(key)
            if actual != expected:
                raise ValueError(
                    f"R1 requires optimizer.{key}={expected!r}, got {actual!r}"
                )
        if self.max_epochs != int(R1_TRAIN["epochs"]):
            raise ValueError(f"R1 requires {R1_TRAIN['epochs']} epochs per task")
        if self.batch_size != int(R1_TRAIN["batch_size"]):
            raise ValueError(f"R1 requires current batch={R1_TRAIN['batch_size']}")

    def _build_optimizer_and_scheduler(self):
        optimizer = build_optimizer(self.model, self.cfg.get("optimizer", {}))
        if self.recipe != R1_RECIPE_ID:
            return optimizer, None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(R1_SCHEDULER["t_max"]),
            eta_min=float(R1_SCHEDULER["eta_min"]),
        )
        return optimizer, scheduler

    def _score_eval_view(self, x: torch.Tensor) -> torch.Tensor:
        return self.r1_views.score_eval_view(x) if self.r1_views is not None else x

    def _training_view(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.r1_views.training_view(x, task_id) if self.r1_views is not None else x

    @torch.no_grad()
    def _probe_features(self) -> List[List[float]]:
        self.model.eval()
        feats = self.model.features(self._probe_x)
        return feats.detach().cpu().numpy().tolist()

    def _weight_drift(self) -> float:
        cur = dict(self.model.named_parameters())
        diffs = [
            (cur[n].detach() - init).reshape(-1) for n, init in self._init_named.items()
        ]
        if not diffs:  # e.g. LinearProbe: only head weights, no protected backbone params
            return 0.0
        return float(torch.norm(torch.cat(diffs)).item())

    def _eval_all(self) -> Dict[str, List[float]]:
        res = [
            evaluate(
                self.model,
                t.test,
                t.task_id,
                self.device,
                self.eval_batch_size,
                t.global_classes,
                input_transform=self._score_eval_view if self.r1_views is not None else None,
            )
            for t in self.bench.tasks
        ]
        out = {"acc": [r["acc"] for r in res]}
        if "masked_acc" in res[0]:
            out["masked_acc"] = [r["masked_acc"] for r in res]
        return out

    def train(self) -> Dict[str, Any]:
        T = len(self.bench.tasks)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        init_eval = self._eval_all()
        init_acc = init_eval["acc"]
        R: List[List[float]] = []
        R_masked: List[List[float]] = []
        task_history: List[List[Dict[str, Any]]] = []
        weight_drift: List[float] = []
        grad_norms: List[float] = []
        consolidation: List[Dict[str, Any]] = []
        representations: List[List[List[float]]] = []
        timing: List[float] = []
        optimizer_steps: List[int] = []
        scheduler_steps: List[int] = []
        optimizer_task_metadata: List[Dict[str, Any]] = []

        t_start = time.time()
        for tid, task in enumerate(self.bench.tasks[: self.max_tasks]):
            self.method.on_task_start(tid, self.model)
            optimizer, scheduler = self._build_optimizer_and_scheduler()
            self.optimizer = optimizer
            self.scheduler = scheduler
            if self.recipe == R1_RECIPE_ID:
                optimizer_task_metadata.append(
                    {
                        "task_id": tid,
                        "optimizer": type(optimizer).__name__,
                        "initial_lr": float(optimizer.param_groups[0]["lr"]),
                        "initial_state_entries": len(optimizer.state),
                        "scheduler": type(scheduler).__name__ if scheduler is not None else None,
                        "scheduler_t_max": int(R1_SCHEDULER["t_max"]),
                        "scheduler_eta_min": float(R1_SCHEDULER["eta_min"]),
                    }
                )
            train_loader = DataLoader(
                _IndexedDataset(task.train), batch_size=self.batch_size, shuffle=True
            )
            y_clean_np = task.train_y_clean  # None unless label noise is configured

            hist: List[Dict[str, Any]] = []
            task_t0 = time.time()
            gnorm_acc, gnorm_n = 0.0, 0
            task_optimizer_steps = 0
            task_scheduler_steps = 0
            for epoch in range(self.max_epochs):
                self.model.train()
                ep_loss, ep_n = 0.0, 0
                for x, y, idx in train_loader:
                    canonical_x, y = x.to(self.device), y.to(self.device)
                    train_x = self._training_view(canonical_x, tid)
                    uids = (tid * _UID_TASK_STRIDE + idx).to(self.device)
                    y_clean = (
                        torch.as_tensor(y_clean_np[idx.numpy()], dtype=torch.long).to(self.device)
                        if y_clean_np is not None
                        else y
                    )
                    self.method.observe_batch(tid, uids, y_clean)
                    optimizer.zero_grad()
                    out = self.model(train_x, tid)
                    loss = self.method.main_loss(self.model, out, train_x, y, tid)
                    if loss is None:  # default: standard current-batch cross-entropy
                        loss = F.cross_entropy(out, y)
                    total = loss + self.method.extra_loss(self.model, train_x, y, tid)
                    total.backward()
                    gnorm_acc += self._grad_norm()
                    gnorm_n += 1
                    self.method.modify_gradients(self.model)
                    if self.grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    # R1 gates score the canonical normalization-only view via
                    # their R1 transform hook; the same canonical data is stored.
                    hook_x = canonical_x if self.r1_views is not None else train_x
                    self.method.before_update(self.model, hook_x, y, tid)
                    optimizer.step()
                    task_optimizer_steps += 1
                    self.method.on_batch_end(self.model, hook_x, y, tid)
                    ep_loss += loss.item() * y.numel()
                    ep_n += y.numel()

                val = evaluate(
                    self.model,
                    task.val,
                    tid,
                    self.device,
                    self.eval_batch_size,
                    task.global_classes,
                    input_transform=self._score_eval_view if self.r1_views is not None else None,
                )
                rec = {
                    "epoch": epoch,
                    "train_loss": ep_loss / max(ep_n, 1),
                    "val_loss": val["loss"],
                    "val_acc": val["acc"],
                }
                if "masked_acc" in val:
                    rec["val_masked_acc"] = val["masked_acc"]
                stop = self.method.on_epoch_end(self.model, val, epoch)
                if scheduler is not None:
                    scheduler.step()
                    task_scheduler_steps += 1
                if self.method.history:
                    rec.update(self.method.history[-1])  # latest method signals
                hist.append(rec)
                if stop:
                    self.log.info(f"task {tid}: early stop at epoch {epoch}")
                    break

            # on_task_end consumers (EWC/MAS Fisher, sleep) iterate (x, y); give
            # them a plain loader, not the index-augmented training loader.
            self.method.on_task_end(
                tid, self.model, DataLoader(task.train, batch_size=self.batch_size)
            )
            self.method.offline_phase(self.model, tid)

            timing.append(time.time() - task_t0)
            optimizer_steps.append(task_optimizer_steps)
            scheduler_steps.append(task_scheduler_steps)
            grad_norms.append(gnorm_acc / max(gnorm_n, 1))
            weight_drift.append(self._weight_drift())
            consolidation.append(self.method.consolidation_state(self.model))
            representations.append(self._probe_features())
            eval_res = self._eval_all()
            R.append(eval_res["acc"])
            if "masked_acc" in eval_res:
                R_masked.append(eval_res["masked_acc"])
            task_history.append(hist)
            self.log.info(
                f"task {tid} done | acc-so-far={[round(a,3) for a in R[-1][:tid+1]]}"
            )

        total_time = time.time() - t_start
        peak_mem = (
            torch.cuda.max_memory_allocated(self.device) / 1e6
            if self.device.type == "cuda"
            else 0.0
        )
        n_params = sum(p.numel() for p in self.model.parameters())

        result = {
            "n_tasks": T,
            "trained_n_tasks": self.max_tasks,
            "task_names": [t.name for t in self.bench.tasks],
            "n_classes_per_task": self.bench.n_classes_per_task,
            "init_acc": init_acc,
            "acc_matrix": R,
            "masked_acc_matrix": R_masked if R_masked else None,
            "task_history": task_history,
            "weight_drift": weight_drift,
            "grad_norm": grad_norms,
            "consolidation": consolidation,
            "representations": representations,
            "timing_per_task_s": timing,
            "total_time_s": total_time,
            "peak_memory_mb": peak_mem,
            "n_params": n_params,
            "inference_cost_us_per_sample": self._inference_cost(),
        }
        if self.recipe == R1_RECIPE_ID:
            acquisition = [float(R[i][i]) for i in range(len(R))]
            final_retention = list(R[-1]) if R else []
            result.update(
                {
                    "recipe": R1_RECIPE_ID,
                    "optimization_steps": {
                        "per_task": optimizer_steps,
                        "total": int(sum(optimizer_steps)),
                    },
                    "r1_training": {
                        "optimizer_reset_at_task_boundary": True,
                        "scheduler_reset_at_task_boundary": True,
                        "scheduler_steps_per_task": scheduler_steps,
                        "scheduler_step_timing": R1_SCHEDULER["step_timing"],
                        "optimizer_task_metadata": optimizer_task_metadata,
                        "augmentation_rng": self.r1_views.provenance(),
                        "canonical_buffer_storage": True,
                        "gate_eval_view": "normalization_only",
                    },
                    "clean_data_acquisition_retention": {
                        "accuracy_matrix": R,
                        "acquisition_diagonal": acquisition,
                        "final_retention_row": final_retention,
                    },
                }
            )
        return result

    def _grad_norm(self) -> float:
        sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
        return sq**0.5

    @torch.no_grad()
    def _inference_cost(self, reps: int = 3) -> float:
        self.model.eval()
        x = self._probe_x[: min(128, self._probe_x.shape[0])]
        if x.shape[0] == 0:
            return 0.0
        # warmup
        self.model(x, 0)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            self.model(x, 0)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / (reps * x.shape[0]) * 1e6
