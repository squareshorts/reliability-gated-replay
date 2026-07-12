# Independent reviewer-risk analysis

## 1. Executive verdict

**Risk reduced but still open.**

The grouped, out-of-sample analysis supports gate separation as a predictor, but less strongly than the pooled manuscript analysis: leave-one-condition-out mean ROC-AUC was 0.774 and leave-one-benchmark-out mean ROC-AUC was 0.706, versus the manuscript's pooled 0.85. Bootstrap intervals across held-out folds were wide. A cluster-robust interaction model found a positive aligned-regime separation slope and a large negative representation-limited interaction.

The stronger-training-recipe risk is unresolved. No established stronger recipe is released in this repository, the native runner hard-codes Adam at 0.001, and the native data/training path exposes neither augmentation nor a scheduler. Selecting or implementing a new recipe would violate the predeclared selection requirement. No strong-recipe runs were fabricated or launched.

## 2. Protocol-comparison table

| Field | Native Seq-CIFAR-10 | Native Split-CIFAR-10 | Mammoth ER bridge | Mammoth AER |
|---|---|---|---|---|
| Dataset/stream | Split CIFAR-10, 5 disjoint 2-class tasks | Split CIFAR-10, 5 disjoint 2-class tasks | Mammoth Seq-CIFAR-10 | Mammoth Seq-CIFAR-10 |
| Architecture | ResNet-18, scratch | ResNet-18, scratch | ResNet-18 | ResNet-18 |
| Classifier/evaluation | Shared 10-class head; Class-IL | Separately parameterized task heads; Task-IL | Shared Class-IL head; also task-masked accuracy | Shared Class-IL head; also task-masked accuracy |
| Task identity | Not supplied at evaluation | Supplied/selects head | Not used for Class-IL; mask used for reported Task-IL | Same |
| Task order/seeds | Fixed by native loader; seeds 0–4 | Fixed by native loader; seeds 0–4 | Mammoth order; seeds 0–1 only | Mammoth order; seeds 0–2 |
| Train examples/task | Capped at 1,500 | Capped at 2,500 | No cap appears in released command; Mammoth dataset default | No cap appears in released command; Mammoth dataset default |
| Memory/replay batch | 500 / 32 | 500 / 32 | 500 / 32 | 500 / 32 |
| Optimizer | Adam, lr 0.001, wd 0 | Adam, lr 0.001, wd 0 | Adam, lr 0.001, wd 0 | Adam, lr 0.001, wd 0 |
| Schedule/budget | No scheduler; 8 epochs/task | No scheduler; 8 epochs/task | 8 epochs/task; scheduler not stated in released evidence | 8 epochs/task; scheduler not stated in released evidence |
| Batch size | 128 | 128 | 128 | 128 |
| Augmentation | Normalization; no augmentation | Normalization; no augmentation | Mammoth dataset transforms; exact pinned source absent | Mammoth dataset transforms; exact pinned source absent |
| Noise | Native corruption masks; symmetric/asymmetric | Same | Synthetic symmetric 60% only | Symmetric 20%, symmetric 60%, asymmetric 40% |
| Accuracy metric | Final average Class-IL accuracy | Final average native multihead Task-IL accuracy | Class-IL and task-masked shared-head Task-IL | Class-IL and task-masked shared-head Task-IL |

Differences preventing direct numerical comparison: native sample caps versus uncapped Mammoth command; native multihead Task-IL versus Mammoth masked shared-head Task-IL; potentially different transforms/augmentation; independent task orders and corruption realizations; Mammoth source checkout and raw logs absent; ER bridge has two seeds and one noise condition. Matching optimizer, epoch count, memory, and batch sizes does not remove these differences.

## 3. Strong-recipe sensitivity

| Item | Result |
|---|---|
| Selection rule | Select one established existing project configuration without test-set search |
| Eligible released stronger recipe | **None found** |
| Proposed exact configuration | **Unavailable; not invented** |
| Planned design if a qualifying recipe existed | 2 benchmarks × 2 noise levels × 4 methods × 5 seeds = 80 runs (60 excluding optional DER++) |
| Empirical cost reference | Existing 80 matching native-recipe runs total 3.25 GPU-hours; mean 128–164 seconds/run on recorded hardware |
| Strong-recipe runs completed | 0/80 |
| Method-specific tuning | None performed |

Every planned ER, small-loss, oracle, and DER++ seed cell is **not run: no eligible predeclared stronger recipe and missing native augmentation/scheduler support**. Consequently, no strong-recipe seed values, bootstrap intervals, Cohen's dz, Wilcoxon tests, or method × recipe interactions exist.

## 4. Method × recipe interaction results

**Not estimable.** Only the manuscript recipe is present. Tests of recipe reversal, separation-sign reversal, and Seq-CIFAR-10 representation limitation cannot be performed without a second recipe.

## 5. Grouped out-of-sample diagnostic results

Target: realizable gated replay beats seed-matched ER. Dataset: 530 matched run pairs, 26 benchmark-condition clusters, six benchmark families; 54 oracle pairs excluded exactly reproducing the manuscript's 530-pair scope. Thresholds were fitted on training folds only.

| Validation | Folds | ROC-AUC | Balanced acc. | Sensitivity | Specificity | PPV | Mean threshold (SD) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Leave one condition out | 26 | 0.774 | 0.636 | 0.635 | 0.636 | 0.746 | 0.01620 (0.00010) |
| Leave one benchmark out | 6 | 0.706 | 0.642 | 0.736 | 0.548 | 0.694 | 0.01898 (0.00663) |

Bootstrap intervals across the 26 held-out condition folds (10,000 resamples): ROC-AUC [0.689, 0.848], balanced accuracy [0.567, 0.702], sensitivity [0.487, 0.771], specificity [0.491, 0.765], and PPV [0.656, 0.807]. Bootstrap intervals across the six held-out benchmark-family folds (10,000 resamples): ROC-AUC [0.528, 0.868], balanced accuracy [0.518, 0.763], sensitivity [0.569, 0.887], specificity [0.264, 0.824], and PPV [0.531, 0.863]. Cross-validation was not refitted within bootstrap replicates; the bootstrap sampled already-valid held-out fold-result rows with replacement.

Ordinary run-level regression `gain ~ separation + purity`: separation coefficient 0.2184 (ordinary 95% CI [0.1914, 0.2454]), purity coefficient -0.0039, R² 0.550. Cluster-robust model with regime interaction and method fixed effects: aligned-regime separation coefficient 0.1550 (cluster-robust 95% CI [0.0701, 0.2399], p=0.00035); inverted interaction +0.0713 (CI [-0.0936, 0.2361]); representation-limited interaction -0.1748 (CI [-0.2509, -0.0987], p=0.0000067). The implied representation-limited slope is approximately -0.0198.

Collinearity was moderate (separation VIF 2.62; purity VIF 2.62). The two most influential runs were CIFAR-10N gated runs with positive separation but negative gain (Cook's D 0.0390 and 0.0385). Leave-one-family-out separation slopes remained positive: 0.197–0.236. Excluding Permuted-MNIST retained 0.1966 (cluster-robust CI [0.0952, 0.2980]); excluding Split-MNIST retained 0.2038 (CI [0.0910, 0.3166]). Thus the pooled relationship is not disproportionately dependent on MNIST, while its predictive usefulness is regime-dependent.

## 6. AER-versus-ER bridge results

| Condition | Mammoth ER Class-IL / masked Task-IL | Mammoth AER Class-IL / masked Task-IL | Comparison status |
|---|---|---|---|
| Symmetric 20% | **Missing** | seed 0: 49.19 / 86.15; seed 1: 50.04 / 85.34; seed 2: 51.38 / 85.55 | No matched ER comparison |
| Symmetric 60% | seed 0: 17.29 / 62.47; seed 1: 17.64 / 64.62; seed 2: **missing** | seed 0: 34.27 / 77.12; seed 1: 34.75 / 77.01; seed 2: 37.86 / 79.40 | Two seed-matched comparisons only |
| Asymmetric 40% | **Missing** | seed 0: 44.48 / 84.01; seed 1: 41.73 / 82.05; seed 2: 45.37 / 85.42 | No matched ER comparison |

Only symmetric 60% has any matched Mammoth ER evidence. The two matched Class-IL AER-minus-ER differences are +16.98 and +17.11 percentage points; masked Task-IL differences are +14.65 and +12.39 points. No uncertainty claim is warranted for n=2.

## 7. Baseline-comparability matrix

| Method | Stream/boundaries | Evaluation/task identity | Memory and corrective components | Backbone/data/noise in paper | Official implementation | Comparability |
|---|---|---|---|---|---|---|
| SPR | Continual noisy stream; episodic/task progression | Task-based experiments; protocol-specific task handling | Fixed replay; self-supervised replay and graph-based self-centered purification | MNIST, CIFAR-10/100, WebVision; synthetic and real noise | Project/code reported by the [ICCV paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Continual_Learning_on_Noisy_Data_Streams_via_Self-Purified_Replay_ICCV_2021_paper.pdf) | Executable only after material protocol changes |
| PuriDivER | Online blurry continual learning; blurry boundaries and shared classes | Online class-incremental setting; no native sharp-task multihead match | Memory balances purity/diversity; semi-supervised/noise-aware learning | CIFAR benchmarks under contaminated blurry streams | Paper available from [CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Bang_Online_Continual_Learning_on_a_Contaminated_Data_Stream_With_Blurry_CVPR_2022_paper.pdf); no verified official checkout in this release | Executable only after material protocol changes; not meaningful as a direct native multihead comparison |
| CNLL | Continual stream with task-free sample-separation stage | Continual noisy-label protocol; task identity not the central admission signal | Delay, clean/noisy replay buffers; label refinement, pseudo-labels, strong/weak augmentation, SSL mixup | MNIST, CIFAR-10/100; symmetric/asymmetric noise | Code availability stated by the [official paper](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Karim_CNLL_A_Semi-Supervised_Approach_for_Continual_Noisy_Label_Learning_CVPRW_2022_paper.pdf), but absent locally | Executable only after material protocol changes |
| AER | Task-based Seq-CIFAR-10 in released bridge | Class-IL plus masked Task-IL from shared head; no separate task heads | Buffer 500 here; alternate learning/forgetting and asymmetric balanced sampling | ResNet-18, Seq-CIFAR-10; synthetic symmetric/asymmetric noise | Official [Mammoth repository](https://github.com/aimagelab/mammoth) and [BMVC paper](https://bmvc2024.org/proceedings/680/) | Directly executable and fair only against matched Mammoth ER; native comparison requires material protocol changes |

## 8. Missing data and analytical limitations

- No eligible established stronger recipe; no strong-recipe measurements or interactions.
- No local Mammoth checkout, pinned source tree, raw Mammoth logs, per-task trajectories, buffer purity, gate separation, forgetting, or admission metrics for the AER/ER bridge; only `summary.csv` remains.
- Mammoth ER missing for symmetric 20%, asymmetric 40%, and seed 2 of symmetric 60%.
- Native and Mammoth task orders/corruption masks cannot be verified as identical.
- Mammoth transforms and effective sample count cannot be verified from the absent pinned checkout; released command does not reproduce native sample caps.
- Fold-result bootstrap used 10,000 resamples with seed 20260712. The units were the already-valid held-out condition folds and held-out benchmark-family folds; cross-validation was not refitted inside bootstrap replicates.
- Regime labels in the multivariable model operationalize Seq-CIFAR-10/100 as representation-limited; this is a manuscript-derived grouping, not a newly estimated latent regime.
- Exact baseline implementation versions for SPR, PuriDivER, and CNLL were not vendored, so direct executability was not tested.

## 9. Reviewer-facing conclusion

Grouped validation supports gate separation as a real but regime-dependent diagnostic: it predicts held-out outcomes above chance, survives exclusion of every benchmark family, and is not driven solely by MNIST. Its held-out discrimination is weaker and less stable than the pooled manuscript value, and it is effectively attenuated in class-incremental CIFAR. The matched AER evidence supports a large AER advantage over Mammoth ER only at symmetric 60% and only for two seeds. The central stronger-recipe sensitivity remains unanswered because the release contains neither an eligible predeclared stronger configuration nor the runner support needed to apply one identically across methods.

## Grouped-bootstrap rerun record

- Command: `python reviewer_risk_analysis/grouped_diagnostic_analysis.py`
- Runtime: 1.596 seconds
- Random seed: 20260712
- Bootstrap repetitions: 10,000
- Bootstrap units: 26 held-out condition folds and six held-out benchmark-family folds
- Threshold estimation: training folds only in the unchanged leave-one-group-out point-estimate procedure
- Scope of regenerated files: `reviewer_risk_analysis/` only
