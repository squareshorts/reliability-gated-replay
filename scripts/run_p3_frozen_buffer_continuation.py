#!/usr/bin/env python
"""Run one Amendment-003 Phase-3 frozen-buffer continuation cell.

The CLI deliberately accepts exactly one frozen cell per invocation. It has no
path for P3A no-replay cells or for an unplanned replay configuration.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from datasets.registry import build_benchmark  # noqa: E402
from methods.frozen_buffer_continuation import (  # noqa: E402
    BUFFER_POLICIES,
    FrozenBufferReplay,
    restore_parent_model_and_optimizer,
)
from methods.persistence_common_history import P3H_SCORING_MODE, _fingerprint  # noqa: E402
from methods.trainer import _IndexedDataset, _UID_TASK_STRIDE  # noqa: E402
from methods.utils import build_optimizer, evaluate  # noqa: E402
from models.registry import build_model  # noqa: E402
from scripts.run_nn_submission import append_manifest, benchmarks  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.seeding import set_seed  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "revision_round1" / "p3f_validation"
PLAN_PATH = ROOT / "revision_round1" / "protocol" / "planned_cells.csv"
P3H_ARTIFACT_DIRS = (
    ROOT / "results" / "revision_round1" / "p3h_validation" / "artifacts",
    ROOT / "results" / "revision_round1" / "tranche7_p3h" / "artifacts",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_row(cell_id: str) -> dict[str, str]:
    with PLAN_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["cell_id"] == cell_id]
    if len(rows) != 1:
        raise ValueError(f"cell is not uniquely present in frozen plan: {cell_id}")
    row = rows[0]
    required = {
        "phase": "persistence",
        "kind": "continuation",
        "recipe": "R0",
        "method": "frozen_buffer_replay",
        "scoring_mode": "pre_update_at_task1_admission",
    }
    for key, value in required.items():
        if row[key] != value:
            raise ValueError(f"{cell_id}: frozen {key} is not {value!r}")
    if row["buffer_policy"] not in BUFFER_POLICIES:
        raise ValueError(f"{cell_id}: invalid buffer policy")
    if row["replay_labels"] not in {"observed", "clean_reference_oracle"}:
        raise ValueError(f"{cell_id}: invalid replay-label policy")
    if row["replay_coefficient"] not in {"1", "2"}:
        raise ValueError(f"{cell_id}: invalid replay coefficient")
    if not row["dependency"].startswith("P3H__"):
        raise ValueError(f"{cell_id}: missing P3H dependency")
    return row


def _parent_artifact_path(parent_id: str) -> Path:
    candidates = [directory / f"{parent_id}.pt" for directory in P3H_ARTIFACT_DIRS]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one P3H artifact for {parent_id}")
    return matches[0]


def _load_parent(row: dict[str, str]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    parent_id = row["dependency"]
    path = _parent_artifact_path(parent_id)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "cell_id": parent_id,
        "phase": "persistence",
        "kind": "common_history",
        "benchmark": row["benchmark"],
        "condition": row["noise"],
        "recipe": "R0",
        "method": "ordinary_current_batch",
        "scoring_mode": P3H_SCORING_MODE,
        "seed": int(row["seed"]),
        "trained_task_ids": [0],
        "continuation_executed": False,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"{parent_id}: invalid saved {key}")
    fingerprints = artifact.get("fingerprints", {})
    if fingerprints.get("model_state_sha256") != _fingerprint(artifact.get("model_state")):
        raise ValueError(f"{parent_id}: model-state fingerprint mismatch")
    if fingerprints.get("optimizer_state_sha256") != _fingerprint(artifact.get("optimizer_state")):
        raise ValueError(f"{parent_id}: optimizer-state fingerprint mismatch")
    source_config_path = path.parent.parent / "configs" / f"{parent_id}.json"
    with source_config_path.open(encoding="utf-8") as handle:
        source_config = json.load(handle)
    if (
        source_config.get("data", {}).get("max_train_per_task") != 2500
        or source_config.get("method", {}).get("buffer_size") != 500
        or source_config.get("train", {}).get("max_tasks") != 1
    ):
        raise ValueError(f"{parent_id}: invalid saved P3H config")
    return path, artifact, source_config


def _restore_parent_rng(states: dict[str, Any]) -> None:
    needed = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not needed.issubset(states):
        raise ValueError("parent P3H artifact lacks training/data RNG states")
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and states["torch_cuda"]:
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def _verify_task1_setup(bench, parent: dict[str, Any], parent_id: str) -> None:
    task = bench.tasks[0]
    info = parent["corruption_information"]
    checks = {
        "task_id": int(task.task_id) == info["task_id"] == 0,
        "stable_ids": torch.equal(torch.arange(len(task.train)), info["stable_ids"]),
        "observed_labels": torch.equal(task.train.tensors[1].cpu(), info["observed_labels"]),
        "clean_labels": torch.equal(torch.as_tensor(task.train_y_clean), info["clean_labels"]),
        "corruption_mask": torch.equal(torch.as_tensor(task.train_is_noisy), info["is_corrupted"]),
    }
    failed = [key for key, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"{parent_id}: rebuilt task-1 setup differs: {', '.join(failed)}")


@torch.no_grad()
def _evaluate_all(model, bench, device: torch.device, batch_size: int) -> dict[str, list[float]]:
    metrics = [
        evaluate(model, task.test, task.task_id, device, batch_size, task.global_classes)
        for task in bench.tasks
    ]
    result = {"acc": [float(metric["acc"]) for metric in metrics]}
    if "masked_acc" in metrics[0]:
        result["masked_acc"] = [float(metric["masked_acc"]) for metric in metrics]
    return result


def _assert_finite(value: Any, location: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite required output: {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{location}[{index}]")


def _run_one(
    row: dict[str, str],
    parent_path: Path,
    parent: dict[str, Any],
    source_config: dict[str, Any],
    parent_hash_before: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell_id = row["cell_id"]
    seed = int(row["seed"])
    data_cfg = copy.deepcopy(source_config["data"])
    train_cfg = copy.deepcopy(source_config["train"])
    train_cfg["max_tasks"] = 5
    bench = build_benchmark(data_cfg, data_root=str(ROOT / "data"))
    _verify_task1_setup(bench, parent, row["dependency"])
    if len(bench.tasks) != 5:
        raise ValueError(f"{cell_id}: expected five total tasks")

    set_seed(seed, deterministic=True)
    model = build_model(benchmarks()[row["benchmark"]]["model"], bench).to(device)
    optimizer = build_optimizer(model, {"name": "adam", "lr": 1e-3})
    restore_parent_model_and_optimizer(model, optimizer, parent, device)
    method = FrozenBufferReplay(
        parent_artifact=parent,
        buffer_policy=row["buffer_policy"],
        replay_label_policy=row["replay_labels"],
        replay_coefficient=int(row["replay_coefficient"]),
        frozen_seed=seed,
        replay_batch_size=32,
    )

    # The diagnostic evaluation must not shift task-2 shuffle order. Restore the
    # saved task-1 RNG state immediately afterward, before the first loader.
    _restore_parent_rng(parent["rng_states"])
    eval_batch_size = int(train_cfg.get("eval_batch_size", 256))
    initial_eval = _evaluate_all(model, bench, device, eval_batch_size)
    _restore_parent_rng(parent["rng_states"])

    accuracy_rows: list[list[float]] = []
    masked_rows: list[list[float]] = []
    history: list[list[dict[str, float | int]]] = []
    task_timing: list[float] = []
    direct_exposure: list[int] = []
    completed: list[int] = []
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    started = time.time()

    for task in bench.tasks[1:]:
        task_id = int(task.task_id)
        method.on_task_start(task_id, model)
        loader = DataLoader(_IndexedDataset(task.train), batch_size=batch_size, shuffle=True)
        clean_labels = task.train_y_clean
        task_history: list[dict[str, float | int]] = []
        task_start = time.time()
        exposure = 0
        for epoch in range(epochs):
            model.train()
            current_sum = replay_sum = total_sum = 0.0
            seen = 0
            for x, y, index in loader:
                x, y = x.to(device), y.to(device)
                clean = torch.as_tensor(clean_labels[index.numpy()], dtype=torch.long).to(device)
                method.observe_batch(task_id, (task_id * _UID_TASK_STRIDE + index).to(device), clean)
                optimizer.zero_grad()
                current_loss = F.cross_entropy(model(x, task_id), y)
                scaled_replay_loss = method.extra_loss(model, x, y, task_id)
                total_loss = current_loss + scaled_replay_loss
                total_loss.backward()
                optimizer.step()
                method.on_batch_end(model, x, y, task_id)
                n = y.numel()
                current_sum += float(current_loss.item()) * n
                replay_sum += float(scaled_replay_loss.item()) * n
                total_sum += float(total_loss.item()) * n
                seen += n
                exposure += n
            validation = evaluate(model, task.val, task_id, device, eval_batch_size, task.global_classes)
            record: dict[str, float | int] = {
                "epoch": epoch,
                "current_stream_loss": current_sum / max(seen, 1),
                "scaled_replay_loss": replay_sum / max(seen, 1),
                "total_loss": total_sum / max(seen, 1),
                "val_loss": float(validation["loss"]),
                "val_acc": float(validation["acc"]),
            }
            if "masked_acc" in validation:
                record["val_masked_acc"] = float(validation["masked_acc"])
            task_history.append(record)
        method.on_task_end(task_id, model, DataLoader(task.train, batch_size=batch_size))
        method.offline_phase(model, task_id)
        evaluations = _evaluate_all(model, bench, device, eval_batch_size)
        accuracy_rows.append(evaluations["acc"])
        if "masked_acc" in evaluations:
            masked_rows.append(evaluations["masked_acc"])
        history.append(task_history)
        task_timing.append(time.time() - task_start)
        direct_exposure.append(exposure)
        completed.append(task_id)

    parent_hash_after = _sha256_file(parent_path)
    result = {
        "cell_id": cell_id,
        "phase": "persistence",
        "kind": "continuation",
        "recipe": "R0",
        "method": "frozen_buffer_replay",
        "scoring_mode": "pre_update_at_task1_admission",
        "benchmark": row["benchmark"],
        "condition": row["noise"],
        "seed": seed,
        "parent_p3h": {
            "cell_id": row["dependency"],
            "artifact_path": str(parent_path),
            "artifact_sha256": parent_hash_after,
            "model_state_sha256": parent["fingerprints"]["model_state_sha256"],
            "optimizer_state_sha256": parent["fingerprints"]["optimizer_state_sha256"],
        },
        "buffer_policy": row["buffer_policy"],
        "replay_label_policy": row["replay_labels"],
        "lambda": int(row["replay_coefficient"]),
        "initial_acc_after_task1": initial_eval["acc"],
        "initial_masked_acc_after_task1": initial_eval.get("masked_acc"),
        "continuation_task_range": [1, 4],
        "continuation_tasks_completed": completed,
        "acc_matrix": accuracy_rows,
        "masked_acc_matrix": masked_rows if masked_rows else None,
        "task_history": history,
        "direct_exposure_presentations": direct_exposure,
        "replay_presentations": method.replay_presentations,
        "provenance": method.provenance(),
        "parent_artifact_unchanged": parent_hash_after == parent_hash_before,
        "final_model_state_sha256": _fingerprint({k: v.detach().cpu() for k, v in model.state_dict().items()}),
        "final_optimizer_state_sha256": _fingerprint(optimizer.state_dict()),
        "timing_per_task_s": task_timing,
        "wall_time_s": time.time() - started,
    }
    if not result["provenance"]["buffer_membership_unchanged"]:
        raise RuntimeError(f"{cell_id}: frozen buffer membership changed")
    _assert_finite(result)
    saved_config = {
        "data": data_cfg,
        "train": train_cfg,
        "method": {
            "name": "frozen_buffer_replay",
            "buffer_policy": row["buffer_policy"],
            "replay_label_policy": row["replay_labels"],
            "replay_coefficient": int(row["replay_coefficient"]),
            "replay_batch_size": 32,
            "scoring_mode": "pre_update_at_task1_admission",
        },
    }
    return result, saved_config


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, help="one frozen P3 continuation cell ID")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    row = _load_frozen_row(args.cell)
    parent_path, parent, source_config = _load_parent(row)
    parent_hash_before = _sha256_file(parent_path)
    output = Path(args.output).resolve()
    if args.dry_run:
        print(
            f"DRY-RUN PASS: {row['cell_id']} -> {row['dependency']} "
            f"buffer={row['buffer_policy']} labels={row['replay_labels']} "
            f"lambda={row['replay_coefficient']} artifact={parent_path}"
        )
        return

    raw_path = output / "raw_logs" / f"{row['cell_id']}.json"
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {raw_path}")
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    result, saved_config = _run_one(
        row, parent_path, parent, source_config, parent_hash_before,
        get_device(args.device or "auto"),
    )
    if _sha256_file(parent_path) != parent_hash_before:
        raise RuntimeError("parent P3H artifact changed during continuation")
    _write_json(raw_path, result)
    config_path = output / "configs" / f"{row['cell_id']}.json"
    _write_json(config_path, saved_config)
    append_manifest(
        output / "experiment_manifest.csv",
        {
            "run_id": row["cell_id"],
            "benchmark": row["benchmark"],
            "dataset": "split_cifar10",
            "method": "frozen_buffer_replay",
            "backbone": "resnet18",
            "noise_type": "symmetric",
            "noise_rate": "0.4" if row["noise"] == "task1_sym40" else "0.6",
            "seed": row["seed"],
            "command": "python scripts/run_p3_frozen_buffer_continuation.py",
            "config_path": str(config_path),
            "status": "done",
            "start_time": start,
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "one frozen-buffer continuation; task range=[1,2,3,4]",
        },
    )
    print(f"done: {row['cell_id']} -> {raw_path}")


if __name__ == "__main__":
    main()
