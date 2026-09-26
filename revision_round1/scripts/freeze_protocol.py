"""Freeze the round-1 execution matrix before comparative result inspection.

This script is deliberately independent of experiment outputs.  It expands the
prespecified Cartesian products from the reviewer brief into a human-readable
execution plan and a machine-readable cell ledger.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision_round1"
PROTOCOL = REVISION / "protocol"
REPORTS = REVISION / "reports"

SEEDS = range(5)
BENCHMARKS = ("split_cifar10", "seq_cifar10")
NOISES = ("sym20", "sym60")


def add_cell(rows: list[dict[str, str]], **values: object) -> None:
    row = {
        "cell_id": "",
        "phase": "",
        "kind": "training",
        "benchmark": "",
        "noise": "",
        "recipe": "",
        "method": "",
        "scoring_mode": "",
        "buffer_policy": "",
        "replay_labels": "",
        "replay_coefficient": "",
        "seed": "",
        "reuse_decision": "new_required",
        "planned_outcomes": "",
        "dependency": "",
    }
    row.update({key: str(value) for key, value in values.items()})
    rows.append(row)


def build_cells() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    core_outcomes = ";".join(
        (
            "accuracy_matrix",
            "final_accuracy",
            "ece15",
            "brier_multiclass",
            "nll",
            "admission_counts",
            "retained_counts",
            "buffer_purity_coverage",
            "gate_separation_trajectory",
            "inversion_trajectory",
            "replay_exposure",
            "runtime_memory",
            "fingerprints",
        )
    )

    # Phase 2: exactly 120 independently trained cells.
    scoring_methods = (
        ("er", "none"),
        ("small_loss", "post_update"),
        ("small_loss", "pre_update"),
        ("confidence", "post_update"),
        ("confidence", "pre_update"),
        ("oracle_clean_admission", "post_update"),
    )
    for benchmark, noise, (method, scoring_mode), seed in itertools.product(
        BENCHMARKS, NOISES, scoring_methods, SEEDS
    ):
        add_cell(
            rows,
            cell_id=f"P2__{benchmark}__{noise}__{method}__{scoring_mode}__s{seed}",
            phase="scoring",
            benchmark=benchmark,
            noise=noise,
            recipe="R0",
            method=method,
            scoring_mode=scoring_mode,
            seed=seed,
            planned_outcomes=core_outcomes + ";same_example_pre_post_gate_delta",
            reuse_decision="new_required_historical_runs_lack_required_matched_diagnostics",
        )

    # Phase 3: 20 common histories, 20 no-replay branches, and 160 replay branches.
    for benchmark, noise, seed in itertools.product(BENCHMARKS, ("task1_sym40", "task1_sym60"), SEEDS):
        history_id = f"P3H__{benchmark}__{noise}__s{seed}"
        add_cell(
            rows,
            cell_id=history_id,
            phase="persistence",
            kind="common_history",
            benchmark=benchmark,
            noise=noise,
            recipe="R0",
            method="ordinary_current_batch",
            scoring_mode="pre_update_shadow_small_loss_and_matched_random",
            seed=seed,
            planned_outcomes="task1_checkpoint;optimizer_state;two_shadow_buffers;rng_states;stable_ids;admission_multiplicity;fingerprints",
        )
        add_cell(
            rows,
            cell_id=f"P3A__{benchmark}__{noise}__no_replay__s{seed}",
            phase="persistence",
            kind="continuation",
            benchmark=benchmark,
            noise=noise,
            recipe="R0",
            method="no_replay",
            replay_coefficient=0,
            seed=seed,
            planned_outcomes="task1_retention_trajectory;subsequent_task_accuracy;native_and_masked_accuracy;direct_exposure;runtime_memory",
            dependency=history_id,
        )
        for policy, labels, coefficient in itertools.product(
            ("small_loss_pre", "random_count_matched"),
            ("observed", "clean_reference_oracle"),
            (1, 2),
        ):
            add_cell(
                rows,
                cell_id=f"P3__{benchmark}__{noise}__{policy}__{labels}__lambda{coefficient}__s{seed}",
                phase="persistence",
                kind="continuation",
                benchmark=benchmark,
                noise=noise,
                recipe="R0",
                method="frozen_buffer_replay",
                scoring_mode="pre_update_at_task1_admission",
                buffer_policy=policy,
                replay_labels=labels,
                replay_coefficient=coefficient,
                seed=seed,
                planned_outcomes="task1_retention_trajectory;subsequent_task_accuracy;native_and_masked_accuracy;direct_replay_presentations;post_task1_replay_presentations;loss_normalized_exposure;residence;multiplicity;runtime_memory",
                dependency=history_id,
            )

    # Phase 4: 80 R1 noisy cells plus 20 clean-label ER references across R0/R1.
    recipe_methods = ("er", "small_loss_pre", "confidence_pre", "oracle_clean_admission")
    for benchmark, noise, method, seed in itertools.product(BENCHMARKS, NOISES, recipe_methods, SEEDS):
        add_cell(
            rows,
            cell_id=f"P4__{benchmark}__{noise}__R1__{method}__s{seed}",
            phase="recipe",
            benchmark=benchmark,
            noise=noise,
            recipe="R1",
            method=method,
            scoring_mode="pre_update" if method in {"small_loss_pre", "confidence_pre"} else "none_or_oracle",
            seed=seed,
            planned_outcomes=core_outcomes + ";optimization_steps;clean_data_acquisition_retention",
        )
    for benchmark, recipe, seed in itertools.product(BENCHMARKS, ("R0", "R1"), SEEDS):
        add_cell(
            rows,
            cell_id=f"P4C__{benchmark}__clean__{recipe}__er__s{seed}",
            phase="recipe",
            benchmark=benchmark,
            noise="clean",
            recipe=recipe,
            method="er",
            seed=seed,
            planned_outcomes="accuracy_matrix;final_accuracy;ece15;brier_multiclass;nll;acquisition_retention;runtime_memory;fingerprints",
            reuse_decision="new_required_for_matched_clean_reference_and_required_outputs",
        )

    # Phase 8: only the seven explicitly missing external ER cells; conditional.
    for noise, seeds in (("sym20", range(3)), ("asym40", range(3)), ("sym60", (2,))):
        for seed in seeds:
            add_cell(
                rows,
                cell_id=f"P8__mammoth_seq_cifar10__{noise}__er__s{seed}",
                phase="aer",
                benchmark="mammoth_seq_cifar10",
                noise=noise,
                recipe="pinned_mammoth_e75a491c",
                method="er",
                seed=seed,
                planned_outcomes="accuracy;runtime;source_and_data_provenance;sample_sd_inventory",
                reuse_decision="conditional_only_if_existing_AER_provenance_can_be_matched",
            )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    path = PROTOCOL / "planned_cells.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contrasts() -> list[str]:
    items: list[str] = []
    for benchmark, noise, gate, seed in itertools.product(
        BENCHMARKS, NOISES, ("small_loss", "confidence"), SEEDS
    ):
        items.append(f"P2 | {benchmark} | {noise} | {gate} pre_update - post_update | seed {seed}")
        items.append(f"P2 | {benchmark} | {noise} | gain-over-ER difference: {gate} pre vs post | seed {seed}")
    for benchmark, noise, seed, policy, coefficient in itertools.product(
        BENCHMARKS,
        ("task1_sym40", "task1_sym60"),
        SEEDS,
        ("small_loss_pre", "random_count_matched"),
        (1, 2),
    ):
        items.append(
            f"P3 | {benchmark} | {noise} | {policy} | observed - clean-reference replay labels | lambda {coefficient} | seed {seed}"
        )
    for benchmark, noise, seed, labels in itertools.product(
        BENCHMARKS,
        ("task1_sym40", "task1_sym60"),
        SEEDS,
        ("observed", "clean_reference_oracle"),
    ):
        items.append(
            f"P3 | {benchmark} | {noise} | small-loss - count-matched-random admission | {labels} labels | seed {seed}"
        )
        items.append(
            f"P3 | {benchmark} | {noise} | lambda 2 - lambda 1 | {labels} labels, policy-paired | seed {seed}"
        )
    for benchmark, noise, method, seed in itertools.product(
        BENCHMARKS, NOISES, ("small_loss_pre", "confidence_pre", "oracle_clean_admission"), SEEDS
    ):
        items.append(f"P4 | {benchmark} | {noise} | R1 {method} - R1 ER | seed {seed}")
        items.append(f"P4 | {benchmark} | {noise} | (R1 {method}-ER) - (R0 {method}-ER) | seed {seed}")
    for benchmark, seed in itertools.product(BENCHMARKS, SEEDS):
        items.append(f"P4 | {benchmark} | clean ER | R1 - R0 | seed {seed}")
    return items


def plan_markdown(rows: list[dict[str, str]], contrast_rows: list[str]) -> str:
    lines = [
        "# Frozen round-1 execution plan",
        "",
        "Frozen before systematic comparative-outcome inspection. This plan is generated only from the reviewer brief and repository-structure audit.",
        "",
        "## Provenance and reuse rule",
        "",
        "The requested working path is used. The local origin/commit must be recorded by the audit stage. No historical training cell is presumed reusable: reuse requires an exact match of data IDs, initialization, corruption mask, recipe, scoring convention, and required outputs. The required matched diagnostics are absent from the legacy manifest schema, so all core cells are initially `new_required`. External AER completion remains conditional on provenance reconstruction.",
        "",
        "## Prespecified execution order",
        "",
        "1. Audit, protected-file hash baseline, environment check, tests, and measured smoke test.",
        "2. Phase 2 scoring cells.",
        "3. Phase 3 common histories and frozen-buffer continuations.",
        "4. Phase 4 R1 sensitivity and clean ER references.",
        "5. Phase 5 representation diagnostics on the same saved Class-IL checkpoints.",
        "6. Phase 6 calibration recovery/audit, Phase 7 dependence analyses, reports.",
        "7. Phase 8 external AER cells only if the pinned provenance is verified and measured cost is feasible.",
        "8. Phase 9 factual novelty matrix from primary sources.",
        "",
        "## Global evaluation conventions",
        "",
        "- New native training seeds: 0, 1, 2, 3, 4.",
        "- Calibration: clean test targets; top-label ECE with 15 equal-width bins; multiclass Brier as class-sum then example-mean; NLL of the clean target; task-macro and pooled estimates stored separately.",
        "- Class-IL outputs: native shared-head and true-task-masked diagnostics kept separate.",
        "- Checkpoints: after every task for Class-IL cells; optimizer state separately where continuation requires it.",
        "- Representation: fixed-label-space task probes and balanced all-seen-class probes, standardized on fit only, regularization chosen on validation only.",
        "- Statistics: seed-paired rows; every seed-level difference retained; 10,000 cluster-aware resamples; exact paired tests where applicable; sample SD (`ddof=1`).",
        "- Failures remain explicit. No seed/epoch/sample reduction and no silent synthetic fallback.",
        "",
        "## Planned contrasts",
        "",
    ]
    lines.extend(f"- {item}" for item in contrast_rows)
    lines.extend(
        [
            "",
            "Derived contrasts additionally include native-minus-task-masked head performance, head-minus-probe trajectories at every saved Class-IL task boundary, initial acquisition versus later retention, full and prespecified benchmark-exclusion associations, and threshold-validation macro-fold versus pooled metrics. These reuse the listed trained cells and do not create independent training replicates.",
            "",
            "## Complete planned cell ledger",
            "",
            f"Total listed execution cells: **{len(rows)}** (Phase 2: 120; Phase 3: 200; Phase 4: 100; conditional Phase 8: 7).",
            "",
            "| cell_id | phase | kind | benchmark | noise | recipe | method | scoring | buffer | labels | lambda | seed | reuse | outcomes | dependency |",
            "|---|---|---|---|---|---|---|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        fields = (
            row["cell_id"],
            row["phase"],
            row["kind"],
            row["benchmark"],
            row["noise"],
            row["recipe"],
            row["method"],
            row["scoring_mode"],
            row["buffer_policy"],
            row["replay_labels"],
            row["replay_coefficient"],
            row["seed"],
            row["reuse_decision"],
            row["planned_outcomes"],
            row["dependency"],
        )
        lines.append("| " + " | ".join(value.replace("|", "/") for value in fields) + " |")
    lines.extend(
        [
            "",
            "The same ledger is stored as `planned_cells.csv` for the driver. Phase-5/6 derived evaluations are attached to their source cell IDs rather than counted as new random training cells.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = build_cells()
    contrast_rows = contrasts()
    write_csv(rows)
    (PROTOCOL / "execution_plan.md").write_text(
        plan_markdown(rows, contrast_rows), encoding="utf-8", newline="\n"
    )
    print(f"froze {len(rows)} cells and {len(contrast_rows)} explicit seed-level contrasts")


if __name__ == "__main__":
    main()
