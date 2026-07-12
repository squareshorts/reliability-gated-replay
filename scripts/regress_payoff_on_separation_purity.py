"""Two-predictor regression: does gate separation predict replay payoff
beyond final buffer purity?

Regresses the paired accuracy gain over ER (d_acc) on standardized
gate_separation and buffer_purity across all gated runs, reporting
standardized coefficients, semipartial correlations, and a cluster
bootstrap (resampling benchmark x condition cells) for the coefficients.

Output: results/neural_networks_submission/stats/two_predictor_regression.csv
"""

import numpy as np
import pandas as pd
from scipy import stats as st

RNG = np.random.default_rng(0)
N_BOOT = 2000

FINAL = "results/neural_networks_submission/csv/final_metrics.csv"
OUT = "results/neural_networks_submission/stats/two_predictor_regression.csv"


def build_pairs(df):
    er = df[df.method == "er"].set_index(["benchmark", "condition", "seed"])[
        "average_accuracy"
    ]
    g = df[df.gate_separation.notna() & (df.method != "oracle")].copy()
    g["er_acc"] = g.set_index(["benchmark", "condition", "seed"]).index.map(er)
    g = g.dropna(subset=["er_acc", "buffer_purity"])
    g["d_acc"] = g.average_accuracy - g.er_acc
    g["cond_key"] = g.benchmark + "|" + g.condition
    return g


def std_ols(g):
    n = len(g)
    X = np.column_stack(
        [np.ones(n), st.zscore(g.gate_separation), st.zscore(g.buffer_purity)]
    )
    y = st.zscore(g.d_acc)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, r2, X, y


def semipartial(y, x, control):
    resid = x - np.poly1d(np.polyfit(control, x, 1))(control)
    return st.pearsonr(y, resid)


def cluster_boot(g):
    keys = g.cond_key.unique()
    betas = []
    for _ in range(N_BOOT):
        ks = RNG.choice(keys, len(keys), replace=True)
        s = pd.concat([g[g.cond_key == k] for k in ks])
        b, _, _, _ = std_ols(s)
        betas.append(b[1:])
    return np.percentile(np.array(betas), [2.5, 97.5], axis=0)


def main():
    df = pd.read_csv(FINAL)
    rows = []
    scopes = {
        "pooled_all": None,
        "purity_limited (MNIST + split_cifar10)": [
            "permuted_mnist",
            "split_mnist",
            "split_cifar10",
        ],
    }
    for name, benches in scopes.items():
        g = build_pairs(df)
        if benches is not None:
            g = g[g.benchmark.isin(benches)]
        beta, r2, X, y = std_ols(g)
        sp_sep = semipartial(y, X[:, 1], X[:, 2])
        sp_pur = semipartial(y, X[:, 2], X[:, 1])
        ci = cluster_boot(g) if benches is None else np.full((2, 2), np.nan)
        rows.append(
            dict(
                scope=name,
                n=len(g),
                beta_separation=beta[1],
                beta_purity=beta[2],
                r2=r2,
                semipartial_sep_given_purity=sp_sep[0],
                semipartial_sep_p=sp_sep[1],
                semipartial_purity_given_sep=sp_pur[0],
                semipartial_purity_p=sp_pur[1],
                boot_ci_sep_low=ci[0, 0],
                boot_ci_sep_high=ci[1, 0],
                boot_ci_purity_low=ci[0, 1],
                boot_ci_purity_high=ci[1, 1],
                collinearity_r_sep_purity=st.pearsonr(X[:, 1], X[:, 2])[0],
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
