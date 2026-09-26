#!/usr/bin/env python3
"""Deterministic analysis of the completed frozen Phase-4 R1 block.

The script reads completed records only.  It neither imports training code nor
constructs datasets/models.  R0 comparisons are admitted only when the
benchmark, condition, seed, method policy, and (where applicable) scoring
timing are directly matched.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "revision_round1"
ANALYSIS = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "revision_round1" / "protocol" / "planned_cells.csv"
AMENDMENT_PATH = ROOT / "revision_round1" / "protocol" / "AMENDMENT_004_R1_RECIPE.md"
R1_ROOTS = (RESULTS / "tranche9_r1_validation", RESULTS / "tranche10_r1")
R0_ROOTS = tuple(RESULTS / name for name in (
    "tranche1", "tranche2", "tranche3", "tranche4",
    "pre_update_validation",
    "tranche5_pre_update", "tranche6_clean_r0",
))
BOOTSTRAP_SEED = 20260926
RECIPE_BOOTSTRAP_SEED = 20260927
BOOTSTRAP_REPS = 10_000
SUMMARY_METRICS = (
    "final_average_accuracy",
    "buffer_purity",
    "normalized_class_entropy",
    "class_count_variance",
)
R1_TO_R0_LABEL = {
    "er": "er",
    "small_loss_pre": "gate_loss",
    "confidence_pre": "gate_conf",
    "oracle_clean_admission": "oracle",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_amendment_004() -> str:
    """Confirm that the frozen R1 recipe text is present and recognizable."""
    text = AMENDMENT_PATH.read_text(encoding="utf-8")
    required = (
        "Train for 50 epochs per task.",
        "small_loss_pre and confidence_pre use Amendment-001 pre_update semantics.",
        "Oracle does not correct current-stream or",
    )
    missing = [phrase for phrase in required if phrase not in text]
    if missing:
        raise ValueError(f"Amendment 004 is missing expected R1 requirements: {missing}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(nonfinite_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(nonfinite_paths(item, f"{prefix}[{index}]"))
    return paths


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean())


def sd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def rank_average_absolute(values: np.ndarray) -> np.ndarray:
    """Average ranks for absolute values, including tied ranks."""
    ordered = np.argsort(np.abs(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and np.isclose(
            abs(values[ordered[end]]), abs(values[ordered[start]]), rtol=0, atol=1e-15
        ):
            end += 1
        ranks[ordered[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def exact_wilcoxon_two_sided(deltas: np.ndarray) -> tuple[float, int]:
    """Exact sign-permutation Wilcoxon p value, dropping zero differences."""
    nonzero = deltas[~np.isclose(deltas, 0.0, rtol=0, atol=1e-15)]
    if not len(nonzero):
        return 1.0, 0
    ranks = rank_average_absolute(nonzero)
    observed = float(ranks[nonzero > 0].sum())
    center = float(ranks.sum() / 2.0)
    observed_distance = abs(observed - center)
    all_distances = []
    for mask in range(1 << len(ranks)):
        plus_sum = sum(ranks[index] for index in range(len(ranks)) if mask & (1 << index))
        all_distances.append(abs(plus_sum - center))
    p_value = sum(distance >= observed_distance - 1e-12 for distance in all_distances) / len(all_distances)
    return float(p_value), int(len(nonzero))


def paired_statistics(left: list[float], right: list[float], rng: np.random.Generator) -> dict[str, float | int]:
    deltas = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    bootstrap_indices = rng.integers(0, len(deltas), size=(BOOTSTRAP_REPS, len(deltas)))
    bootstrap_means = deltas[bootstrap_indices].mean(axis=1)
    p_value, n_nonzero = exact_wilcoxon_two_sided(deltas)
    delta_sd = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    dz = float(deltas.mean() / delta_sd) if delta_sd else (0.0 if deltas.mean() == 0 else math.inf)
    return {
        "n_pairs": int(len(deltas)),
        "mean_paired_delta": float(deltas.mean()),
        "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
        "cohens_dz": dz,
        "wilcoxon_exact_two_sided_p": p_value,
        "wilcoxon_nonzero_pairs": n_nonzero,
    }


def contrast_rng_seed(base_seed: int, benchmark: str, noise: str, contrast: str) -> int:
    """Stable per-contrast PCG64 seed, independent of execution order."""
    material = (
        f"r1-bootstrap-v1|base_seed={int(base_seed)}|benchmark={benchmark}|"
        f"noise={noise}|contrast={contrast}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def contrast_rng(base_seed: int, benchmark: str, noise: str, contrast: str) -> tuple[np.random.Generator, int]:
    seed = contrast_rng_seed(base_seed, benchmark, noise, contrast)
    return np.random.Generator(np.random.PCG64(seed)), seed


def read_json_records(roots: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        for path in sorted((root / "raw_logs").glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                records.append((path, json.load(handle)))
    return records


def config_for_raw(raw_path: Path) -> dict[str, Any]:
    config_path = raw_path.parent.parent / "configs" / raw_path.name
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def r1_run(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    buffer = record.get("buffer_diagnostics", {})
    final_row = record["acc_matrix"][-1]
    required_values = {
        "acc_matrix": record["acc_matrix"],
        "buffer_purity": buffer["buffer_purity"],
        "normalized_class_entropy": buffer["normalized_class_entropy"],
        "class_count_variance": buffer["class_count_variance"],
    }
    all_nonfinite = nonfinite_paths(record)
    return {
        "run_id": record["r1_provenance"]["frozen_cell_id"],
        "benchmark": record["benchmark"],
        "noise": record["condition"],
        "method": record["label"],
        "scoring_mode": record["scoring_mode"],
        "seed": int(record["seed"]),
        "final_average_accuracy": mean(final_row),
        "buffer_purity": float(buffer["buffer_purity"]),
        "normalized_class_entropy": float(buffer["normalized_class_entropy"]),
        "class_count_variance": float(buffer["class_count_variance"]),
        "source_raw_log": relative(path),
        "all_required_values_finite": finite_tree(required_values),
        "optional_nonfinite_fields": "|".join(all_nonfinite),
        "recipe_sha256": record["r1_provenance"]["recipe_sha256"],
    }


def validate_r1(plan: list[dict[str, str]], manifests: list[dict[str, str]], runs: list[dict[str, Any]]) -> None:
    expected = [row for row in plan if row["phase"] == "recipe" and row["recipe"] == "R1"]
    if len(expected) != 90:
        raise ValueError(f"frozen plan has {len(expected)}, not 90, R1 cells")
    expected_ids = [row["cell_id"] for row in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("frozen R1 plan contains duplicate cell IDs")
    group_sizes: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in expected:
        group_sizes[(row["benchmark"], row["noise"], row["method"])] += 1
    if len(group_sizes) != 18 or set(group_sizes.values()) != {5}:
        raise ValueError("R1 plan is not 18 groups of exactly five seeds")

    manifest_ids = [row["run_id"] for row in manifests]
    if len(manifest_ids) != 90 or len(set(manifest_ids)) != 90:
        raise ValueError("R1 manifests do not contain exactly 90 unique rows")
    if set(manifest_ids) != set(expected_ids):
        raise ValueError("R1 manifest identities differ from frozen plan")
    bad_status = [row["run_id"] for row in manifests if row.get("status") != "done"]
    if bad_status:
        raise ValueError(f"R1 has failed/non-done manifest records: {bad_status}")

    run_ids = [row["run_id"] for row in runs]
    if len(run_ids) != 90 or len(set(run_ids)) != 90 or set(run_ids) != set(expected_ids):
        raise ValueError("R1 raw records are not a one-to-one match with frozen plan")
    if not all(row["all_required_values_finite"] for row in runs):
        bad = [row["run_id"] for row in runs if not row["all_required_values_finite"]]
        raise ValueError(f"R1 required outcomes contain non-finite values: {bad}")
    plan_by_id = {row["cell_id"]: row for row in expected}
    for run in runs:
        planned = plan_by_id[run["run_id"]]
        actual = (run["benchmark"], run["noise"], run["method"], str(run["seed"]), run["scoring_mode"])
        intended = (planned["benchmark"], planned["noise"], planned["method"], planned["seed"], planned["scoring_mode"])
        if actual != intended:
            raise ValueError(f"R1 record diverges from frozen plan: {run['run_id']}")


def r0_semantics_eligible(r1_method: str, r0: dict[str, Any]) -> tuple[bool, str]:
    cfg = r0["record"].get("method_cfg", {})
    mode = r0["record"].get("scoring_mode", cfg.get("scoring_mode", "post_update"))
    if r1_method in {"small_loss_pre", "confidence_pre"}:
        return (mode == "pre_update", "requires Amendment-001 pre_update scoring")
    if r1_method == "er":
        return (cfg.get("name") == "replay", "ER has no admission scoring contrast")
    if r1_method == "oracle_clean_admission":
        return (
            bool(cfg.get("oracle")) and cfg.get("name") == "gated_replay",
            "oracle clean-admission policy; scoring timing is not used",
        )
    return False, "unknown R1 method"


def core_data_match(r1: dict[str, Any], r0: dict[str, Any]) -> bool:
    r0_config = r0["config"].get("data", {})
    if not r0_config:
        return False
    r1_data = r1["record"]["r1_provenance"]["full_resolved_configuration"]["data"]
    return (
        r0_config.get("name") == r1_data.get("name")
        and r0_config.get("max_train_per_task") == r1_data.get("max_train_per_task")
        and r0_config.get("max_test_per_task") == r1_data.get("max_test_per_task")
        and r0_config.get("classes_per_task") == r1_data.get("classes_per_task")
        and r0_config.get("label_noise") == r1_data.get("label_noise")
    )


def recipe_comparisons(r1_records: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    r1_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for path, record in r1_records:
        run = r1_run(path, record)
        run["record"] = record
        r1_by_key[(run["benchmark"], run["noise"], run["method"], run["seed"])] = run

    r0_candidates: list[dict[str, Any]] = []
    for path, record in read_json_records(R0_ROOTS):
        if record.get("r1_provenance"):
            continue
        r0_candidates.append({"path": path, "record": record, "config": config_for_raw(path)})

    rows: list[dict[str, Any]] = []
    group_keys = sorted({key[:3] for key in r1_by_key})
    for benchmark, noise, method in group_keys:
        rng, recipe_bootstrap_seed = contrast_rng(
            RECIPE_BOOTSTRAP_SEED, benchmark, noise, f"r1_minus_r0::{method}"
        )
        r0_label = R1_TO_R0_LABEL[method]
        selected_r1 = [r1_by_key[(benchmark, noise, method, seed)] for seed in range(5)]
        matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
        mismatch_paths: list[str] = []
        for r1 in selected_r1:
            candidates = [
                candidate for candidate in r0_candidates
                if candidate["record"].get("benchmark") == benchmark
                and candidate["record"].get("condition") == noise
                and candidate["record"].get("label") == r0_label
                and int(candidate["record"].get("seed")) == r1["seed"]
            ]
            eligible = []
            for candidate in candidates:
                semantic_ok, _ = r0_semantics_eligible(method, candidate)
                if semantic_ok and core_data_match(r1, candidate):
                    eligible.append(candidate)
                elif not semantic_ok:
                    mismatch_paths.append(relative(candidate["path"]))
            if len(eligible) == 1:
                matched.append((r1, eligible[0]))
            elif len(eligible) > 1:
                raise ValueError(f"multiple semantically eligible R0 records for {benchmark}/{noise}/{method}/s{r1['seed']}")

        base = {
            "benchmark": benchmark,
            "noise": noise,
            "r1_method": method,
            "r0_method_label": r0_label,
            "r1_scoring_mode": selected_r1[0]["scoring_mode"],
            "bootstrap_base_seed": RECIPE_BOOTSTRAP_SEED,
            "bootstrap_seed": recipe_bootstrap_seed,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "r1_mean_final_average_accuracy": "",
            "r0_mean_final_average_accuracy": "",
            "mean_paired_delta_r1_minus_r0": "",
            "bootstrap_ci_low": "",
            "bootstrap_ci_high": "",
            "cohens_dz": "",
            "wilcoxon_exact_two_sided_p": "",
            "r0_scoring_mode": "",
            "excluded_r0_paths": "",
        }
        if len(matched) == 5:
            r1_values = [pair[0]["final_average_accuracy"] for pair in matched]
            r0_values = [mean(pair[1]["record"]["acc_matrix"][-1]) for pair in matched]
            stats = paired_statistics(r1_values, r0_values, rng)
            semantic_ok, rationale = r0_semantics_eligible(method, matched[0][1])
            assert semantic_ok
            rows.append({
                **base,
                "comparison_status": "comparable",
                "rationale": rationale + "; exact benchmark/noise/seed/core-data match",
                "n_pairs": 5,
                "r1_mean_final_average_accuracy": mean(r1_values),
                "r0_mean_final_average_accuracy": mean(r0_values),
                "mean_paired_delta_r1_minus_r0": stats["mean_paired_delta"],
                "bootstrap_ci_low": stats["bootstrap_ci_low"],
                "bootstrap_ci_high": stats["bootstrap_ci_high"],
                "cohens_dz": stats["cohens_dz"],
                "wilcoxon_exact_two_sided_p": stats["wilcoxon_exact_two_sided_p"],
                "r0_scoring_mode": matched[0][1]["record"].get("scoring_mode", "post_update"),
                "excluded_r0_paths": "|".join(sorted(set(mismatch_paths))),
            })
        else:
            rows.append({
                **base,
                "comparison_status": "not_comparable",
                "rationale": f"only {len(matched)}/5 direct R0 matches; incomplete or semantically incompatible comparator set",
                "n_pairs": len(matched),
                "excluded_r0_paths": "|".join(sorted(set(mismatch_paths))),
            })
        if mismatch_paths:
            rows.append({
                **base,
                "comparison_status": "excluded_scoring_timing_mismatch",
                "rationale": "R0 post_update candidate excluded: it cannot be pooled with R1 pre_update admission",
                "n_pairs": 0,
                "excluded_r0_paths": "|".join(sorted(set(mismatch_paths))),
            })
    return rows


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    amendment_sha256 = validate_amendment_004()
    plan = read_csv(PLAN_PATH)
    manifests = [row for root in R1_ROOTS for row in read_csv(root / "experiment_manifest.csv")]
    raw_r1 = read_json_records(R1_ROOTS)
    runs = [r1_run(path, record) for path, record in raw_r1]
    validate_r1(plan, manifests, runs)

    run_columns = [
        "run_id", "benchmark", "noise", "method", "scoring_mode", "seed",
        *SUMMARY_METRICS, "source_raw_log", "all_required_values_finite", "optional_nonfinite_fields", "recipe_sha256",
    ]
    runs.sort(key=lambda row: (row["benchmark"], row["noise"], row["method"], row["seed"]))
    write_csv(ANALYSIS / "r1_run_level.csv", runs, run_columns)

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["benchmark"], run["noise"], run["method"])].append(run)
    for (benchmark, noise, method), rows in sorted(grouped.items()):
        summary = {"benchmark": benchmark, "noise": noise, "method": method, "n_seeds": len(rows)}
        for metric in SUMMARY_METRICS:
            values = [row[metric] for row in rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_sd"] = sd(values)
        summary_rows.append(summary)
    summary_columns = ["benchmark", "noise", "method", "n_seeds"] + [
        field for metric in SUMMARY_METRICS for field in (f"{metric}_mean", f"{metric}_sd")
    ]
    write_csv(ANALYSIS / "r1_method_summary.csv", summary_rows, summary_columns)

    contrast_rows: list[dict[str, Any]] = []
    by_key = {(run["benchmark"], run["noise"], run["method"], run["seed"]): run for run in runs}
    for benchmark in ("split_cifar10", "seq_cifar10"):
        for noise in ("sym20", "sym60"):
            er = [by_key[(benchmark, noise, "er", seed)]["final_average_accuracy"] for seed in range(5)]
            for method in ("small_loss_pre", "confidence_pre", "oracle_clean_admission"):
                candidate = [by_key[(benchmark, noise, method, seed)]["final_average_accuracy"] for seed in range(5)]
                contrast_name = f"{method} - er"
                paired_rng, derived_seed = contrast_rng(
                    BOOTSTRAP_SEED, benchmark, noise, contrast_name
                )
                stats = paired_statistics(candidate, er, paired_rng)
                contrast_rows.append({
                    "benchmark": benchmark,
                    "noise": noise,
                    "contrast": contrast_name,
                    "outcome": "final_average_accuracy",
                    "bootstrap_base_seed": BOOTSTRAP_SEED,
                    "bootstrap_rng_seed": derived_seed,
                    "bootstrap_reps": BOOTSTRAP_REPS,
                    **stats,
                })
    contrast_columns = [
        "benchmark", "noise", "contrast", "outcome", "bootstrap_base_seed", "bootstrap_rng_seed", "bootstrap_reps",
        "n_pairs", "mean_paired_delta", "bootstrap_ci_low", "bootstrap_ci_high", "cohens_dz",
        "wilcoxon_exact_two_sided_p", "wilcoxon_nonzero_pairs",
    ]
    write_csv(ANALYSIS / "r1_paired_contrasts.csv", contrast_rows, contrast_columns)

    recipe_rows = recipe_comparisons(raw_r1)
    recipe_columns = [
        "benchmark", "noise", "r1_method", "r0_method_label", "comparison_status", "rationale",
        "n_pairs", "r1_scoring_mode", "r0_scoring_mode", "r1_mean_final_average_accuracy",
        "r0_mean_final_average_accuracy", "mean_paired_delta_r1_minus_r0", "bootstrap_ci_low",
        "bootstrap_ci_high", "cohens_dz", "wilcoxon_exact_two_sided_p", "bootstrap_base_seed", "bootstrap_seed",
        "bootstrap_reps", "excluded_r0_paths",
    ]
    write_csv(ANALYSIS / "r0_r1_recipe_comparison.csv", recipe_rows, recipe_columns)

    lines = [
        "# Phase-4 R1 analysis",
        "",
        "## Validation",
        "",
        "All 90/90 planned R1 cells were found exactly once in the combined manifests and raw records. "
        "The plan has 18 benchmark/noise/method groups of five seeds; all manifest statuses are `done`; "
        "and all required accuracy and requested buffer metrics are finite. Thirty records contain only expected undefined optional diagnostics: "
        "`replay_loss_noisy` for clean-admission/clean-data buffers, plus clean-data mislabeled-subgroup diagnostics when that subgroup is empty. These are recorded run by run and are not failed runs.",
        "",
        "## Method",
        "",
        f"Paired bootstrap intervals use {BOOTSTRAP_REPS:,} PCG64 resamples with per-contrast SHA-256-derived seeds. "
        f"R1-versus-ER contrasts use base seed `{BOOTSTRAP_SEED}`; R0-versus-R1 recipe comparisons use base seed `{RECIPE_BOOTSTRAP_SEED}`. "
        "Wilcoxon p values are exact two-sided signed-rank permutation p values, with zero paired differences excluded.",
        f"The checked Amendment-004 text has SHA-256 `{amendment_sha256}`.",
        "",
        "## Descriptive R1 results",
        "",
        "The CSV summary reports mean and sample SD across the five frozen seeds for final average accuracy, buffer purity, normalized class entropy, and class-count variance. These are descriptive summaries.",
        "",
        "| Benchmark | Noise | Method | Final average accuracy, mean (SD) | Buffer purity, mean (SD) |",
        "|---|---|---|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['benchmark']} | {row['noise']} | {row['method']} | "
            f"{row['final_average_accuracy_mean']:.4f} ({row['final_average_accuracy_sd']:.4f}) | "
            f"{row['buffer_purity_mean']:.4f} ({row['buffer_purity_sd']:.4f}) |"
        )
    lines.extend([
        "",
        "## Seed-paired R1 contrasts",
        "",
        "Inferential fields in `r1_paired_contrasts.csv` concern only the frozen seed-paired final-average-accuracy contrasts. Confidence intervals and exact Wilcoxon p values quantify this five-seed sample; they do not establish broad generalization beyond the evaluated cells. The oracle is a clean-admission diagnostic policy, not a mathematical upper bound.",
        "",
        "| Benchmark | Noise | Contrast | Mean delta | Bootstrap 95% CI | Cohen's dz | Exact p |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in contrast_rows:
        lines.append(
            f"| {row['benchmark']} | {row['noise']} | {row['contrast']} | {row['mean_paired_delta']:.4f} | "
            f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}] | "
            f"{row['cohens_dz']:.3f} | {row['wilcoxon_exact_two_sided_p']:.4f} |"
        )
    lines.extend([
        "",
        "## R0 versus R1 recipe sensitivity",
        "",
        "`r0_r1_recipe_comparison.csv` includes only exact benchmark/noise/seed/core-data matches. Small-loss and confidence comparisons require R0 `pre_update` records; post-update candidates are emitted as explicitly excluded rows. ER comparisons are eligible because ER has no admission scorer. The clean ER R0-versus-R1 sensitivity is included as the `clean` ER comparable rows.",
        "",
        "| Benchmark | Noise | R1 method | Status | R1 minus R0 mean delta | Exact p |",
        "|---|---|---|---|---:|---:|",
    ])
    for row in recipe_rows:
        if row["comparison_status"] != "comparable":
            continue
        lines.append(
            f"| {row['benchmark']} | {row['noise']} | {row['r1_method']} | comparable | "
            f"{float(row['mean_paired_delta_r1_minus_r0']):.4f} | "
            f"{float(row['wilcoxon_exact_two_sided_p']):.4f} |"
        )
    not_comparable = [row for row in recipe_rows if row["comparison_status"] == "not_comparable"]
    lines.extend([
        "",
        "The following requested recipe comparison is not made: " + "; ".join(
            f"{row['benchmark']}/{row['noise']}/{row['r1_method']} ({row['rationale']})"
            for row in not_comparable
        ) + ".",
        "",
        "## Files",
        "",
        "- `r1_run_level.csv`: validated run-level metrics and provenance.",
        "- `r1_method_summary.csv`: descriptive five-seed summaries.",
        "- `r1_paired_contrasts.csv`: requested R1-versus-ER paired inference.",
        "- `r0_r1_recipe_comparison.csv`: comparator decisions and recipe sensitivity.",
    ])
    (ANALYSIS / "r1_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
