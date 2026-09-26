#!/usr/bin/env python
"""Execute only the frozen Phase-3 P3A no-replay continuations.

This runner is intentionally limited to the 20 ``P3A__...__no_replay`` cells.
It restores the exact task-1 model, optimizer, and RNG state from a validated
P3H artifact, then trains tasks 2--5 with ordinary current-batch updates.  It
contains no frozen-buffer replay path and no implementation of
``pre_update_at_task1_admission``.
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
from methods.persistence_common_history import P3H_SCORING_MODE, _fingerprint  # noqa: E402
from methods.registry import build_method  # noqa: E402
from methods.trainer import _IndexedDataset, _UID_TASK_STRIDE  # noqa: E402
from methods.utils import build_optimizer, evaluate  # noqa: E402
from models.registry import build_model  # noqa: E402
from scripts.run_nn_submission import append_manifest, benchmarks  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.seeding import set_seed  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results" / "revision_round1" / "tranche8_p3a_no_replay"
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


def _expected_cells() -> dict[str, str]:
    expected: dict[str, str] = {}
    for benchmark in ("split_cifar10", "seq_cifar10"):
        for condition in ("task1_sym40", "task1_sym60"):
            for seed in range(5):
                cell_id = f"P3A__{benchmark}__{condition}__no_replay__s{seed}"
                expected[cell_id] = f"P3H__{benchmark}__{condition}__s{seed}"
    return expected


def _load_plan_rows() -> list[dict[str, str]]:
    with PLAN_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row["phase"] == "persistence"
        and row["kind"] == "continuation"
        and row["method"] == "no_replay"
    ]
    expected = _expected_cells()
    got = {row["cell_id"] for row in selected}
    if got != set(expected) or len(selected) != 20:
        raise ValueError("planned P3A no-replay cells are not the exact frozen 20-cell set")
    for row in selected:
        cell_id = row["cell_id"]
        if (
            row["recipe"] != "R0"
            or row["scoring_mode"]
            or row["buffer_policy"]
            or row["replay_labels"]
            or row["replay_coefficient"] != "0"
            or row["dependency"] != expected[cell_id]
        ):
            raise ValueError(f"invalid frozen P3A semantics for {cell_id}")
    return sorted(selected, key=lambda row: row["cell_id"])


def _artifact_path(dependency_id: str) -> Path:
    matches = [directory / f"{dependency_id}.pt" for directory in P3H_ARTIFACT_DIRS]
    present = [path for path in matches if path.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(
            f"expected exactly one P3H artifact for {dependency_id}; found {len(present)}"
        )
    return present[0]


def _assert_dependency(artifact: dict[str, Any], dependency_id: str, row: dict[str, str]) -> None:
    benchmark = row["benchmark"]
    condition = row["noise"]
    seed = int(row["seed"])
    required = {
        "cell_id": dependency_id,
        "phase": "persistence",
        "kind": "common_history",
        "benchmark": benchmark,
        "condition": condition,
        "recipe": "R0",
        "method": "ordinary_current_batch",
        "scoring_mode": P3H_SCORING_MODE,
        "seed": seed,
        "max_train_per_task": 2500,
        "trained_task_ids": [0],
        "continuation_executed": False,
    }
    for key, expected in required.items():
        if artifact.get(key) != expected:
            raise ValueError(f"{dependency_id}: invalid {key}={artifact.get(key)!r}")
    if not isinstance(artifact.get("model_state"), dict) or not artifact["model_state"]:
        raise ValueError(f"{dependency_id}: missing model state")
    if not isinstance(artifact.get("optimizer_state"), dict) or not artifact["optimizer_state"]:
        raise ValueError(f"{dependency_id}: missing optimizer state")
    fingerprints = artifact.get("fingerprints", {})
    if fingerprints.get("model_state_sha256") != _fingerprint(artifact["model_state"]):
        raise ValueError(f"{dependency_id}: model-state fingerprint mismatch")
    if fingerprints.get("optimizer_state_sha256") != _fingerprint(artifact["optimizer_state"]):
        raise ValueError(f"{dependency_id}: optimizer-state fingerprint mismatch")


def _dependency_config(artifact_path: Path, dependency_id: str) -> dict[str, Any]:
    config_path = artifact_path.parent.parent / "configs" / f"{dependency_id}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing P3H config: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if (
        config.get("data", {}).get("max_train_per_task") != 2500
        or config.get("method", {}).get("buffer_size") != 500
        or config.get("method", {}).get("scoring_mode") != P3H_SCORING_MODE
        or config.get("train", {}).get("max_tasks") != 1
    ):
        raise ValueError(f"{dependency_id}: invalid P3H config")
    return config


def _preflight(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        dependency_id = row["dependency"]
        artifact_path = _artifact_path(dependency_id)
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        _assert_dependency(artifact, dependency_id, row)
        config = _dependency_config(artifact_path, dependency_id)
        prepared.append(
            {
                "row": row,
                "artifact_path": artifact_path,
                "artifact_sha256": _sha256_file(artifact_path),
                "artifact": artifact,
                "config": config,
            }
        )
    if len({entry["row"]["cell_id"] for entry in prepared}) != 20:
        raise ValueError("P3A dry-run did not produce 20 unique frozen cells")
    if len({entry["row"]["dependency"] for entry in prepared}) != 20:
        raise ValueError("P3A cells do not have one-to-one P3H dependencies")
    return prepared


def _restore_rng(states: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not required.issubset(states):
        raise ValueError("P3H artifact lacks continuation RNG states")
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and states["torch_cuda"]:
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _verify_task1_setup(bench, artifact: dict[str, Any], dependency_id: str) -> None:
    task = bench.tasks[0]
    corruption = artifact["corruption_information"]
    observed = task.train.tensors[1].detach().cpu()
    clean = torch.as_tensor(task.train_y_clean, dtype=torch.long)
    noisy = torch.as_tensor(task.train_is_noisy, dtype=torch.bool)
    stable_ids = torch.arange(len(task.train), dtype=torch.long)
    checks = {
        "task_id": int(task.task_id) == corruption["task_id"] == 0,
        "stable_ids": torch.equal(stable_ids, corruption["stable_ids"]),
        "observed_labels": torch.equal(observed, corruption["observed_labels"]),
        "clean_labels": torch.equal(clean, corruption["clean_labels"]),
        "is_corrupted": torch.equal(noisy, corruption["is_corrupted"]),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"{dependency_id}: task-1 setup mismatch: {', '.join(failed)}")


@torch.no_grad()
def _eval_all(model, bench, device: torch.device, batch_size: int) -> dict[str, list[float]]:
    evaluated = [
        evaluate(model, task.test, task.task_id, device, batch_size, task.global_classes)
        for task in bench.tasks
    ]
    result = {"acc": [float(item["acc"]) for item in evaluated]}
    if "masked_acc" in evaluated[0]:
        result["masked_acc"] = [float(item["masked_acc"]) for item in evaluated]
    return result


def _task_data_fingerprint(task) -> str:
    return _fingerprint(
        {
            "task_id": int(task.task_id),
            "observed_labels": task.train.tensors[1].detach().cpu(),
            "clean_labels": torch.as_tensor(task.train_y_clean, dtype=torch.long),
            "is_corrupted": torch.as_tensor(task.train_is_noisy, dtype=torch.bool),
        }
    )


def _assert_finite(value: Any, path: str = "result") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _run_cell(entry: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    row = entry["row"]
    artifact = entry["artifact"]
    dependency_id = row["dependency"]
    seed = int(row["seed"])
    benchmark_name = row["benchmark"]
    config = copy.deepcopy(entry["config"])
    data_cfg = config["data"]
    train_cfg = copy.deepcopy(config["train"])
    train_cfg.pop("max_tasks", None)
    train_cfg["max_tasks"] = 5
    spec = benchmarks()[benchmark_name]

    # Rebuild the frozen dataset and task-1 corruption before restoring the
    # post-task-1 RNG state.  The data configuration comes from its P3H source.
    bench = build_benchmark(data_cfg, data_root=str(ROOT / "data"))
    _verify_task1_setup(bench, artifact, dependency_id)
    if len(bench.tasks) != 5:
        raise ValueError(f"{row['cell_id']}: expected five continuation tasks")

    set_seed(seed, deterministic=True)
    model = build_model(spec["model"], bench).to(device)
    model.load_state_dict(artifact["model_state"])
    method = build_method({"name": "finetune", "seed": seed})
    optimizer = build_optimizer(model, {"name": "adam", "lr": 1e-3})
    optimizer.load_state_dict(artifact["optimizer_state"])
    _move_optimizer_state(optimizer, device)

    # An initial diagnostic evaluation is useful, but its iterator may consume
    # an RNG value.  Restore the saved state immediately afterward so task 2's
    # shuffle is exactly the frozen continuation order.
    _restore_rng(artifact["rng_states"])
    eval_batch_size = int(train_cfg.get("eval_batch_size", 256))
    initial_eval = _eval_all(model, bench, device, eval_batch_size)
    _restore_rng(artifact["rng_states"])

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    accuracy_rows: list[list[float]] = []
    masked_accuracy_rows: list[list[float]] = []
    histories: list[list[dict[str, float | int]]] = []
    timing: list[float] = []
    direct_exposure: list[int] = []
    completed_tasks: list[int] = []
    t0 = time.time()

    for task in bench.tasks[1:]:
        task_id = int(task.task_id)
        method.on_task_start(task_id, model)
        loader = DataLoader(_IndexedDataset(task.train), batch_size=batch_size, shuffle=True)
        clean_np = task.train_y_clean
        task_history: list[dict[str, float | int]] = []
        exposure = 0
        task_t0 = time.time()
        for epoch in range(epochs):
            model.train()
            loss_total = 0.0
            seen = 0
            for x, y, index in loader:
                x, y = x.to(device), y.to(device)
                clean = (
                    torch.as_tensor(clean_np[index.numpy()], dtype=torch.long).to(device)
                    if clean_np is not None
                    else y
                )
                uids = (task_id * _UID_TASK_STRIDE + index).to(device)
                method.observe_batch(task_id, uids, clean)
                optimizer.zero_grad()
                logits = model(x, task_id)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                method.on_batch_end(model, x, y, task_id)
                loss_total += float(loss.item()) * y.numel()
                seen += y.numel()
                exposure += y.numel()
            validation = evaluate(model, task.val, task_id, device, eval_batch_size, task.global_classes)
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": loss_total / max(seen, 1),
                "val_loss": float(validation["loss"]),
                "val_acc": float(validation["acc"]),
            }
            if "masked_acc" in validation:
                record["val_masked_acc"] = float(validation["masked_acc"])
            task_history.append(record)
        method.on_task_end(task_id, model, DataLoader(task.train, batch_size=batch_size))
        method.offline_phase(model, task_id)
        metrics = _eval_all(model, bench, device, eval_batch_size)
        accuracy_rows.append(metrics["acc"])
        if "masked_acc" in metrics:
            masked_accuracy_rows.append(metrics["masked_acc"])
        histories.append(task_history)
        timing.append(time.time() - task_t0)
        direct_exposure.append(exposure)
        completed_tasks.append(task_id)

    result: dict[str, Any] = {
        "cell_id": row["cell_id"],
        "phase": "persistence",
        "kind": "continuation",
        "recipe": "R0",
        "benchmark": benchmark_name,
        "condition": row["noise"],
        "seed": seed,
        "method": "no_replay",
        "replay_enabled": False,
        "replay_coefficient": 0,
        "replay_presentations": 0,
        "dependency": {
            "cell_id": dependency_id,
            "artifact_path": str(entry["artifact_path"]),
            "artifact_sha256": entry["artifact_sha256"],
            "model_state_sha256": artifact["fingerprints"]["model_state_sha256"],
            "optimizer_state_sha256": artifact["fingerprints"]["optimizer_state_sha256"],
        },
        "initial_acc_after_task1": initial_eval["acc"],
        "initial_masked_acc_after_task1": initial_eval.get("masked_acc"),
        "continuation_tasks_requested": [1, 2, 3, 4],
        "continuation_tasks_completed": completed_tasks,
        "acc_matrix": accuracy_rows,
        "masked_acc_matrix": masked_accuracy_rows if masked_accuracy_rows else None,
        "task_history": histories,
        "timing_per_task_s": timing,
        "direct_exposure_presentations": direct_exposure,
        "continuation_data_fingerprints": {
            str(task.task_id): _task_data_fingerprint(task) for task in bench.tasks
        },
        "final_model_state_sha256": _fingerprint(
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
        ),
        "final_optimizer_state_sha256": _fingerprint(optimizer.state_dict()),
        "wall_time_s": time.time() - t0,
    }
    _assert_finite(result)
    return result, {"data": data_cfg, "train": train_cfg, "method": {"name": "no_replay", "replay_enabled": False, "replay_coefficient": 0}}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _dry_run(prepared: list[dict[str, Any]], output: Path) -> None:
    print(f"DRY-RUN: 20 frozen P3A no-replay continuations -> {output}")
    for entry in prepared:
        row = entry["row"]
        print(
            f"{row['cell_id']} dependency={row['dependency']} "
            f"artifact={entry['artifact_path']} replay=disabled coefficient=0"
        )
    print("DRY-RUN PASS: exact 20/20 one-to-one P3A-to-P3H mappings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    prepared = _preflight(_load_plan_rows())
    if args.dry_run:
        _dry_run(prepared, output)
        return

    raw_dir = output / "raw_logs"
    config_dir = output / "configs"
    manifest_path = output / "experiment_manifest.csv"
    device = get_device(args.device or "auto")
    done = skipped = failed = 0
    for index, entry in enumerate(prepared, start=1):
        row = entry["row"]
        cell_id = row["cell_id"]
        raw_path = raw_dir / f"{cell_id}.json"
        if args.resume and raw_path.exists():
            skipped += 1
            continue
        start = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{index}/20] {cell_id} <- {row['dependency']}")
        try:
            result, saved_config = _run_cell(entry, device)
            # P3H dependency files are read-only inputs.  Detect any unexpected
            # mutation before recording the continuation as complete.
            if _sha256_file(entry["artifact_path"]) != entry["artifact_sha256"]:
                raise RuntimeError(f"{cell_id}: P3H dependency artifact changed")
            _write_json(raw_path, result)
            _write_json(config_dir / f"{cell_id}.json", saved_config)
            status, note = "done", "replay_disabled=true; replay_coefficient=0; tasks=[1,2,3,4]"
            done += 1
        except Exception as exc:  # keep every failed frozen cell explicit
            status, note = "failed", f"{type(exc).__name__}: {exc}"
            failed += 1
            print(f"  FAILED: {note}")
        append_manifest(
            manifest_path,
            {
                "run_id": cell_id,
                "benchmark": row["benchmark"],
                "dataset": "split_cifar10",
                "method": "no_replay",
                "backbone": "resnet18",
                "noise_type": "symmetric",
                "noise_rate": "0.4" if row["noise"] == "task1_sym40" else "0.6",
                "seed": row["seed"],
                "command": "python scripts/run_p3a_no_replay.py",
                "config_path": str(config_dir / f"{cell_id}.json"),
                "status": status,
                "start_time": start,
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "notes": note,
            },
        )
    print(f"done={done} skipped={skipped} failed={failed} -> {output}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
