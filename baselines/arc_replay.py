import copy
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from methods.base import ContinualMethod
from methods.gates import ReliabilityGate
from methods.replay_buffer import ReplayBuffer, WeightedReplayBuffer, replay_ce_loss, weighted_replay_ce_loss


class ARCLite(ContinualMethod):
    name = "arc_lite"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        seed = int(self.cfg.get("seed", 0))
        self.buf = ReplayBuffer(int(self.cfg.get("buffer_size", 500)), seed=seed)
        self.replay_n = int(self.cfg.get("batch_size", 32))
        
        # small-loss high-power gate
        loss_cfg = {"signal": "error", "error_threshold": 1.0, "gamma": 3.0}
        self.gate_loss = ReliabilityGate(loss_cfg)
        
        # safe fallback gate
        safe_cfg = {"signal": "confidence", "tau": 0.5, "gamma": 6.0}
        self.gate_safe = ReliabilityGate(safe_cfg)

        self.teacher_m = float(self.cfg.get("teacher_momentum", 0.999))
        self.teacher = None

        self.adaptive_ema = float(self.cfg.get("adaptive_ema", 0.95))
        self._disagree = None

        self._device = None
        self._rng = torch.Generator().manual_seed(seed + 7)
        self.gate_history: List[Dict[str, float]] = []

        self._task_alpha: List[float] = []
        self._task_risk: List[float] = []
        self._task_g: List[float] = []
        self._task_correct: List[int] = []
        self._task_candidates = 0
        self._task_admitted = 0
        self._cur_task = 0

    def on_task_start(self, task_id, model):
        self._device = next(model.parameters()).device
        self._cur_task = task_id
        self.gate_loss.reset_task(task_id)
        self.gate_safe.reset_task(task_id)

        if self.teacher is None:
            self.teacher = copy.deepcopy(model)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def _update_teacher(self, model):
        m = self.teacher_m
        for tp, sp in zip(self.teacher.parameters(), model.parameters()):
            tp.mul_(m).add_(sp.detach(), alpha=1.0 - m)
        for tb, sb in zip(self.teacher.buffers(), model.buffers()):
            tb.copy_(sb)

    @torch.no_grad()
    def _argmax_disagreement(self, model, x, task_id) -> float:
        was = model.training
        model.eval()
        ps = model(x, task_id).argmax(1)
        pt = self.teacher(x, task_id).argmax(1)
        if was:
            model.train()
        return float((ps != pt).float().mean())

    def extra_loss(self, model, x, y, task_id):
        if len(self.buf) == 0:
            return x.new_zeros(())
        return replay_ce_loss(model, self.buf.sample(self.replay_n), self._device)

    def on_batch_end(self, model, x, y, task_id):
        y_clean = self._batch_y_clean if self._batch_y_clean is not None else y
        correct = (y.detach().cpu() == y_clean.detach().cpu()).long()
        self._task_candidates += int(x.shape[0])

        g_loss, _ = self.gate_loss.score(model, x, y, task_id, uids=self._batch_uids)
        g_safe, _ = self.gate_safe.score(model, x, y, task_id, uids=self._batch_uids)

        d_batch = self._argmax_disagreement(model, x, task_id)
        self._disagree = (
            d_batch if self._disagree is None
            else self.adaptive_ema * self._disagree + (1 - self.adaptive_ema) * d_batch
        )
        d = self._disagree
        alpha = max(0.0, min(1.0, 1.0 - d))
        
        g = alpha * g_loss + (1.0 - alpha) * g_safe
        
        self._task_alpha.append(alpha)
        self._task_risk.append(d)
        self._task_g.extend(g.cpu().tolist())
        self._task_correct.extend(correct.tolist())

        mask = torch.rand(x.shape[0], generator=self._rng) < g.cpu()
        if bool(mask.any()):
            self._admit(x, y, task_id, y_clean, mask, g)

        self._update_teacher(model)

    def _admit(self, x, y, task_id, y_clean, mask, g):
        md = mask.to(x.device)
        self.buf.add(x[md], y[md], task_id, y_clean=y_clean[md])
        self._task_admitted += int(mask.sum())

    def on_task_end(self, task_id, model, train_loader):
        g_arr = np.asarray(self._task_g, dtype=float)
        c_arr = np.asarray(self._task_correct, dtype=float)
        corr = (
            float(np.corrcoef(g_arr, c_arr)[0, 1])
            if g_arr.size > 1 and g_arr.std() > 1e-8 and c_arr.std() > 1e-8
            else 0.0
        )
        self.gate_history.append(
            {
                "task": float(task_id),
                "alpha_mean": float(np.mean(self._task_alpha)) if self._task_alpha else 1.0,
                "risk_mean": float(np.mean(self._task_risk)) if self._task_risk else 0.0,
                "gate_mean": float(g_arr.mean()) if g_arr.size else 1.0,
                "gate_on_correct": float(g_arr[c_arr == 1].mean()) if (c_arr == 1).any() else float("nan"),
                "gate_on_wrong": float(g_arr[c_arr == 0].mean()) if (c_arr == 0).any() else float("nan"),
                "corr_gate_correct": corr,
                "buffer_purity": self.buf.purity(),
                "buffer_size": len(self.buf),
                "admission_rate": (
                    self._task_admitted / self._task_candidates
                    if self._task_candidates else float("nan")
                ),
            }
        )
        self._task_g.clear()
        self._task_correct.clear()
        self._task_alpha.clear()
        self._task_risk.clear()
        self._task_candidates = 0
        self._task_admitted = 0

    def consolidation_state(self, model) -> Dict[str, Any]:
        last = self.gate_history[-1] if self.gate_history else {}
        return {
            "buffer_size": len(self.buf),
            "buffer_purity": self.buf.purity(),
            "disagree_ema": self._disagree,
            "gate_mean": last.get("gate_mean"),
            "gate_on_correct": last.get("gate_on_correct"),
            "gate_on_wrong": last.get("gate_on_wrong"),
            "corr_gate_correct": last.get("corr_gate_correct"),
            "admission_rate": last.get("admission_rate"),
            "alpha_mean": last.get("alpha_mean"),
            "risk_mean": last.get("risk_mean"),
            "gate_history": list(self.gate_history),
            "buffer_task_counts": {t: self.buf.t.count(t) for t in sorted(set(self.buf.t))},
        }


class ARCFull(ARCLite):
    name = "arc_full"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        seed = int(self.cfg.get("seed", 0))
        self.buf = WeightedReplayBuffer(int(self.cfg.get("buffer_size", 500)), seed=seed)

    def extra_loss(self, model, x, y, task_id):
        if len(self.buf) == 0:
            return x.new_zeros(())
        return weighted_replay_ce_loss(model, self.buf.sample(self.replay_n), self._device)

    def _admit(self, x, y, task_id, y_clean, mask, g):
        md = mask.to(x.device)
        self.buf.add(x[md], y[md], task_id, weight=g[md], y_clean=y_clean[md])
        self._task_admitted += int(mask.sum())
