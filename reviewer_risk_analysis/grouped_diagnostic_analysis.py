from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve)
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "results/neural_networks_submission/csv/final_metrics.csv"
RNG = np.random.default_rng(20260712)


def metrics(y, score, threshold):
    pred = score > threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
    }


def train_threshold(y, score):
    fpr, tpr, cuts = roc_curve(y, score)
    objective = tpr - fpr
    candidates = np.flatnonzero(objective == objective.max())
    best = candidates[np.argmax(tpr[candidates])]
    return float(cuts[best])


def cross_validate(df, group_col):
    rows = []
    for held, test in df.groupby(group_col, sort=True):
        train = df[df[group_col] != held]
        threshold = train_threshold(train.target.to_numpy(), train.gate_separation.to_numpy())
        m = metrics(test.target.to_numpy(), test.gate_separation.to_numpy(), threshold)
        rows.append({"held_out": held, "n": len(test), "threshold": threshold, **m})
    return pd.DataFrame(rows)


def bootstrap_fold_results(folds, reps=10000):
    metric_cols = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "ppv"]
    values = folds[metric_cols].to_numpy()
    sampled_rows = RNG.integers(0, len(folds), size=(reps, len(folds)))
    replicate_means = values[sampled_rows].mean(axis=1)
    q = pd.DataFrame(replicate_means, columns=metric_cols).quantile([.025, .975]).T
    q.columns = ["ci_low", "ci_high"]
    return q.reset_index(names="metric")


def main():
    d = pd.read_csv(SRC)
    gates = d[d.gate_separation.notna() & ~d.method.eq("oracle")].copy()
    er = d[d.method.eq("er")][["benchmark", "condition", "seed", "average_accuracy"]].rename(columns={"average_accuracy": "er_accuracy"})
    x = gates.merge(er, on=["benchmark", "condition", "seed"], validate="many_to_one")
    x["gain"] = x.average_accuracy - x.er_accuracy
    x["target"] = (x.gain > 0).astype(int)
    x["cluster"] = x.benchmark.astype(str) + "::" + x.condition.astype(str)
    x["regime"] = np.where(x.benchmark.isin(["seq_cifar10", "seq_cifar100"]), "representation_limited",
                    np.where(x.gate_separation < 0, "inverted", "aligned"))
    assert len(x) == 530 and x.er_accuracy.notna().all()
    x.to_csv(OUT / "diagnostic_run_pairs.csv", index=False)

    summaries = []
    for label, col in [("leave_one_condition_out", "cluster"), ("leave_one_benchmark_out", "benchmark")]:
        folds = cross_validate(x, col)
        folds.to_csv(OUT / f"{label}_folds.csv", index=False)
        avg = folds[["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "ppv", "threshold"]].mean().to_dict()
        avg.update({"scheme": label, "n_folds": len(folds), "threshold_sd": folds.threshold.std(ddof=1)})
        summaries.append(avg)
        bootstrap_fold_results(folds).assign(scheme=label).to_csv(OUT / f"{label}_cluster_bootstrap_ci.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "cross_validation_summary.csv", index=False)

    ordinary = ols("gain ~ gate_separation + buffer_purity", data=x).fit()
    robust = ols("gain ~ gate_separation * C(regime) + buffer_purity + C(method)", data=x).fit(
        cov_type="cluster", cov_kwds={"groups": x.cluster})
    coef = []
    for name, model in [("ordinary", ordinary), ("cluster_robust", robust)]:
        for term in model.params.index:
            coef.append({"model": name, "term": term, "estimate": model.params[term],
                         "se": model.bse[term], "p": model.pvalues[term],
                         "ci_low": model.conf_int().loc[term, 0], "ci_high": model.conf_int().loc[term, 1],
                         "n": int(model.nobs), "r2": model.rsquared})
    pd.DataFrame(coef).to_csv(OUT / "regression_models.csv", index=False)

    vif_frame = pd.DataFrame({"gate_separation": x.gate_separation, "buffer_purity": x.buffer_purity})
    vif_frame.insert(0, "intercept", 1.0)
    pd.DataFrame({"term": vif_frame.columns,
                  "vif": [variance_inflation_factor(vif_frame.values, i) for i in range(vif_frame.shape[1])]})\
      .to_csv(OUT / "collinearity_vif.csv", index=False)

    influence = ordinary.get_influence().summary_frame()
    inf = x[["run_id", "cluster", "benchmark", "method", "gain", "gate_separation"]].reset_index(drop=True).join(influence.reset_index(drop=True))
    inf.sort_values("cooks_d", ascending=False).head(30).to_csv(OUT / "top_influential_runs.csv", index=False)

    exclusions = []
    for family in [None] + sorted(x.benchmark.unique().tolist()):
        z = x if family is None else x[x.benchmark != family]
        m = ols("gain ~ gate_separation + buffer_purity", data=z).fit(cov_type="cluster", cov_kwds={"groups": z.cluster})
        exclusions.append({"excluded": "none" if family is None else family, "n": len(z),
                           "conditions": z.cluster.nunique(), "gate_coef": m.params["gate_separation"],
                           "gate_se": m.bse["gate_separation"], "gate_p": m.pvalues["gate_separation"],
                           "ci_low": m.conf_int().loc["gate_separation", 0], "ci_high": m.conf_int().loc["gate_separation", 1],
                           "pearson_r": pearsonr(z.gain, z.gate_separation).statistic,
                           "spearman_r": spearmanr(z.gain, z.gate_separation).statistic})
    pd.DataFrame(exclusions).to_csv(OUT / "leave_one_benchmark_regression.csv", index=False)

    residual = pd.DataFrame({"fitted": robust.fittedvalues, "residual": robust.resid,
                             "cluster": x.cluster, "benchmark": x.benchmark})
    residual.groupby("cluster").agg(n=("residual", "size"), mean_residual=("residual", "mean"),
        sd_residual=("residual", "std"), mean_abs_residual=("residual", lambda s: np.abs(s).mean()))\
        .reset_index().to_csv(OUT / "residuals_by_condition.csv", index=False)
    (OUT / "analysis_metadata.json").write_text(json.dumps({
        "source": str(SRC.relative_to(ROOT)), "seed": 20260712, "bootstrap_reps": 10000,
        "bootstrap_unit": {"leave_one_condition_out": "held-out condition fold",
                           "leave_one_benchmark_out": "held-out benchmark-family fold"},
        "bootstrap_method": "sample already-valid held-out fold rows with replacement and average fold metrics; no cross-validation refitting inside bootstrap replicates",
        "threshold_estimation": "training folds only in the unchanged leave-one-group-out point-estimate procedure",
        "target": "average_accuracy(gate) > seed-matched average_accuracy(ER)",
        "included": "non-oracle runs with nonmissing gate separation", "n": len(x),
        "conditions": x.cluster.nunique(), "benchmarks": x.benchmark.nunique()}, indent=2))


if __name__ == "__main__":
    main()
