"""Shared helpers for methods: parameter selection, evaluation, optimizers."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def head_param_ids(model: nn.Module) -> set:
    if hasattr(model, "heads"):
        return {id(p) for p in model.heads.parameters()}
    return set()


def protected_named_parameters(
    model: nn.Module, include_bias: bool = False, include_heads: bool = False
) -> Iterator[Tuple[str, torch.nn.Parameter]]:
    """Yield the parameters consolidation/importance methods act on.

    Defaults to weight matrices/kernels of the shared backbone (excludes biases,
    norm params, and the task-specific heads) -- the standard choice for
    importance-weighted regularizers (EWC/SI) and for PNN consolidation.
    """
    hids = set() if include_heads else head_param_ids(model)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in hids:
            continue
        if p.ndim < 2 and not include_bias:
            continue
        yield name, p


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg.get("name", "adam").lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(cfg.get("momentum", 0.0)),
            weight_decay=wd,
            nesterov=bool(cfg.get("nesterov", False)),
        )
    raise KeyError(f"Unknown optimizer '{name}'. Known: adam, sgd")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: TensorDataset,
    task_id: int,
    device: torch.device,
    batch_size: int = 256,
    valid_classes: List[int] | None = None,
    input_transform=None,
) -> Dict[str, float]:
    """Return {'acc','loss','masked_acc'} for ``dataset`` using head ``task_id``."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total, correct, loss_sum = 0, 0, 0.0
    correct_masked = 0
    for x, y in loader:
        if input_transform is not None:
            x = input_transform(x)
        x, y = x.to(device), y.to(device)
        out = model(x, task_id)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        
        if valid_classes is not None and not getattr(model, "multihead", True):
            mask = torch.full_like(out, float("-inf"))
            mask[:, valid_classes] = 0.0
            out_masked = out + mask
            correct_masked += (out_masked.argmax(1) == y).sum().item()
        total += y.numel()
    
    ret = {"acc": correct / max(total, 1), "loss": loss_sum / max(total, 1)}
    if valid_classes is not None and not getattr(model, "multihead", True):
        ret["masked_acc"] = correct_masked / max(total, 1)
    return ret


def flat_params(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])
