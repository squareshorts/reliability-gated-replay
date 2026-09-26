# Phase-4 R1 analysis

## Validation

All 90/90 planned R1 cells were found exactly once in the combined manifests and raw records. The plan has 18 benchmark/noise/method groups of five seeds; all manifest statuses are `done`; and all required accuracy and requested buffer metrics are finite. Thirty records contain only expected undefined optional diagnostics: `replay_loss_noisy` for clean-admission/clean-data buffers, plus clean-data mislabeled-subgroup diagnostics when that subgroup is empty. These are recorded run by run and are not failed runs.

## Method

Paired bootstrap intervals use 10,000 PCG64 resamples with per-contrast SHA-256-derived seeds. R1-versus-ER contrasts use base seed `20260926`; R0-versus-R1 recipe comparisons use base seed `20260927`. Wilcoxon p values are exact two-sided signed-rank permutation p values, with zero paired differences excluded.
The checked Amendment-004 text has SHA-256 `f4f365ed89657bded12311a01c90a040cb643ba46d4964bcf03b81c2f9b16a17`.

## Descriptive R1 results

The CSV summary reports mean and sample SD across the five frozen seeds for final average accuracy, buffer purity, normalized class entropy, and class-count variance. These are descriptive summaries.

| Benchmark | Noise | Method | Final average accuracy, mean (SD) | Buffer purity, mean (SD) |
|---|---|---|---:|---:|
| seq_cifar10 | clean | er | 0.1962 (0.0260) | 1.0000 (0.0000) |
| seq_cifar10 | sym20 | confidence_pre | 0.1934 (0.0212) | 0.8200 (0.0233) |
| seq_cifar10 | sym20 | er | 0.1764 (0.0114) | 0.8048 (0.0150) |
| seq_cifar10 | sym20 | oracle_clean_admission | 0.2061 (0.0224) | 1.0000 (0.0000) |
| seq_cifar10 | sym20 | small_loss_pre | 0.1792 (0.0119) | 0.9424 (0.0220) |
| seq_cifar10 | sym60 | confidence_pre | 0.1456 (0.0140) | 0.3980 (0.0229) |
| seq_cifar10 | sym60 | er | 0.1602 (0.0081) | 0.4020 (0.0222) |
| seq_cifar10 | sym60 | oracle_clean_admission | 0.2026 (0.0174) | 1.0000 (0.0000) |
| seq_cifar10 | sym60 | small_loss_pre | 0.1179 (0.0150) | 0.4324 (0.0410) |
| split_cifar10 | clean | er | 0.8217 (0.0093) | 1.0000 (0.0000) |
| split_cifar10 | sym20 | confidence_pre | 0.6939 (0.0365) | 0.8124 (0.0227) |
| split_cifar10 | sym20 | er | 0.7086 (0.0248) | 0.8048 (0.0150) |
| split_cifar10 | sym20 | oracle_clean_admission | 0.7855 (0.0191) | 1.0000 (0.0000) |
| split_cifar10 | sym20 | small_loss_pre | 0.7381 (0.0200) | 0.8552 (0.0282) |
| split_cifar10 | sym60 | confidence_pre | 0.3927 (0.0309) | 0.3976 (0.0204) |
| split_cifar10 | sym60 | er | 0.4457 (0.0190) | 0.4020 (0.0222) |
| split_cifar10 | sym60 | oracle_clean_admission | 0.6919 (0.0235) | 1.0000 (0.0000) |
| split_cifar10 | sym60 | small_loss_pre | 0.4068 (0.0133) | 0.3980 (0.0262) |

## Seed-paired R1 contrasts

Inferential fields in `r1_paired_contrasts.csv` concern only the frozen seed-paired final-average-accuracy contrasts. Confidence intervals and exact Wilcoxon p values quantify this five-seed sample; they do not establish broad generalization beyond the evaluated cells. The oracle is a clean-admission diagnostic policy, not a mathematical upper bound.

| Benchmark | Noise | Contrast | Mean delta | Bootstrap 95% CI | Cohen's dz | Exact p |
|---|---|---|---:|---:|---:|---:|
| split_cifar10 | sym20 | small_loss_pre - er | 0.0294 | [0.0043, 0.0576] | 0.873 | 0.1250 |
| split_cifar10 | sym20 | confidence_pre - er | -0.0147 | [-0.0283, 0.0007] | -0.808 | 0.1875 |
| split_cifar10 | sym20 | oracle_clean_admission - er | 0.0769 | [0.0550, 0.0954] | 2.999 | 0.0625 |
| split_cifar10 | sym60 | small_loss_pre - er | -0.0389 | [-0.0566, -0.0239] | -1.906 | 0.0625 |
| split_cifar10 | sym60 | confidence_pre - er | -0.0531 | [-0.0739, -0.0322] | -2.011 | 0.0625 |
| split_cifar10 | sym60 | oracle_clean_admission - er | 0.2462 | [0.2251, 0.2650] | 9.154 | 0.0625 |
| seq_cifar10 | sym20 | small_loss_pre - er | 0.0028 | [-0.0154, 0.0175] | 0.136 | 0.8125 |
| seq_cifar10 | sym20 | confidence_pre - er | 0.0170 | [-0.0002, 0.0311] | 0.874 | 0.1875 |
| seq_cifar10 | sym20 | oracle_clean_admission - er | 0.0297 | [0.0078, 0.0513] | 1.049 | 0.1250 |
| seq_cifar10 | sym60 | small_loss_pre - er | -0.0422 | [-0.0522, -0.0308] | -2.969 | 0.0625 |
| seq_cifar10 | sym60 | confidence_pre - er | -0.0146 | [-0.0309, -0.0003] | -0.753 | 0.1875 |
| seq_cifar10 | sym60 | oracle_clean_admission - er | 0.0424 | [0.0250, 0.0548] | 2.145 | 0.0625 |

## R0 versus R1 recipe sensitivity

`r0_r1_recipe_comparison.csv` includes only exact benchmark/noise/seed/core-data matches. Small-loss and confidence comparisons require R0 `pre_update` records; post-update candidates are emitted as explicitly excluded rows. ER comparisons are eligible because ER has no admission scorer. The clean ER R0-versus-R1 sensitivity is included as the `clean` ER comparable rows.

| Benchmark | Noise | R1 method | Status | R1 minus R0 mean delta | Exact p |
|---|---|---|---|---:|---:|
| seq_cifar10 | clean | er | comparable | 0.0111 | 0.8125 |
| seq_cifar10 | sym20 | confidence_pre | comparable | -0.0124 | 0.1250 |
| seq_cifar10 | sym20 | er | comparable | -0.0234 | 0.1250 |
| seq_cifar10 | sym20 | oracle_clean_admission | comparable | -0.0151 | 0.4375 |
| seq_cifar10 | sym20 | small_loss_pre | comparable | -0.0566 | 0.0625 |
| seq_cifar10 | sym60 | confidence_pre | comparable | 0.0038 | 0.4375 |
| seq_cifar10 | sym60 | er | comparable | 0.0203 | 0.0625 |
| seq_cifar10 | sym60 | oracle_clean_admission | comparable | 0.0014 | 1.0000 |
| seq_cifar10 | sym60 | small_loss_pre | comparable | -0.0517 | 0.0625 |
| split_cifar10 | clean | er | comparable | 0.0047 | 0.6250 |
| split_cifar10 | sym20 | confidence_pre | comparable | 0.0198 | 0.4375 |
| split_cifar10 | sym20 | er | comparable | 0.0456 | 0.0625 |
| split_cifar10 | sym20 | oracle_clean_admission | comparable | 0.0263 | 0.1250 |
| split_cifar10 | sym20 | small_loss_pre | comparable | 0.0278 | 0.1250 |
| split_cifar10 | sym60 | confidence_pre | comparable | -0.0582 | 0.0625 |
| split_cifar10 | sym60 | er | comparable | -0.0012 | 1.0000 |
| split_cifar10 | sym60 | oracle_clean_admission | comparable | 0.0494 | 0.0625 |
| split_cifar10 | sym60 | small_loss_pre | comparable | -0.0296 | 0.1250 |

The following requested recipe comparison is not made: .

## Files

- `r1_run_level.csv`: validated run-level metrics and provenance.
- `r1_method_summary.csv`: descriptive five-seed summaries.
- `r1_paired_contrasts.csv`: requested R1-versus-ER paired inference.
- `r0_r1_recipe_comparison.csv`: comparator decisions and recipe sensitivity.
