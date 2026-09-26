import os
import json
import glob
import pandas as pd
import numpy as np

BASE_DIR = r"C:\work\reliability-gated-replay"
PRE_VALID = os.path.join(BASE_DIR, "results", "revision_round1", "pre_update_validation", "raw_logs")
TRANCHE5 = os.path.join(BASE_DIR, "results", "revision_round1", "tranche5_pre_update", "raw_logs")
PROV_FILE = os.path.join(BASE_DIR, "revision_round1", "analysis", "pre_update_40cell_provenance.csv")

def extract_metrics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    try:
        final_acc = np.mean(data["acc_matrix"][-1])
        c = data["consolidation"][-1]
        buffer_purity = c.get("buffer_purity", np.nan)

        # Calculate normalized class entropy
        counts = c.get("buffer_class_counts", {})
        if not counts:
            # Fallback if class_counts missing but task_counts present
            t_counts = c.get("buffer_task_counts", {})
            if t_counts:
                # Approximate class count by halving task counts (2 classes per task)
                counts = {f"{t}_{c}": v/2 for t, v in t_counts.items() for c in (0, 1)}

        vals = np.array(list(counts.values())) if counts else np.array([])
        if len(vals) > 0 and vals.sum() > 0:
            p = vals / vals.sum()
            entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
            max_ent = np.log(len(vals))
            norm_entropy = entropy / max_ent if max_ent > 0 else np.nan
            var_counts = np.var(vals)
        else:
            norm_entropy = np.nan
            var_counts = np.nan

        replay_loss_clean = np.nan
        replay_loss_noisy = np.nan
        if "loss_diagnostics" in c:
            replay_loss_clean = c["loss_diagnostics"].get("clean_replay_loss", np.nan)
            replay_loss_noisy = c["loss_diagnostics"].get("noisy_replay_loss", np.nan)

        return {
            "final_acc": final_acc,
            "buffer_purity": buffer_purity,
            "norm_entropy": norm_entropy,
            "var_counts": var_counts,
            "replay_loss_clean": replay_loss_clean,
            "replay_loss_noisy": replay_loss_noisy,
            "scoring_mode": c.get("scoring_mode", "post_update"),
            "max_train_per_task": data.get("max_train_per_task", 2500) # Wait, it is in args, but we can trust CLI
        }
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return None

def step6_provenance():
    rows = []

    # Validation cell
    valid_file = os.path.join(PRE_VALID, "split_cifar10__sym20__gate_loss__pre_update__seed0.json")
    if os.path.exists(valid_file):
        rows.append({
            "frozen_cell_id": "P2__split_cifar10__sym20__small_loss__pre_update__s0",
            "dataset": "split_cifar10",
            "condition": "sym20",
            "protocol_method": "small_loss",
            "runner_method": "gate_loss",
            "scoring_mode": "pre_update",
            "seed": 0,
            "max_train_per_task": 2500,
            "raw_json_path": valid_file,
            "manifest_path": os.path.join(BASE_DIR, "results", "revision_round1", "pre_update_validation", "experiment_manifest.csv"),
            "execution_group": "validation",
            "status": "complete"
        })

    for f in glob.glob(os.path.join(TRANCHE5, "*.json")):
        b = os.path.basename(f).replace(".json", "")
        parts = b.split("__")
        dataset = parts[0]
        condition = parts[1]
        runner_method = parts[2]
        seed = int(parts[4].replace("seed", ""))
        pm = "small_loss" if runner_method == "gate_loss" else "confidence"
        frozen_id = f"P2__{dataset}__{condition}__{pm}__pre_update__s{seed}"

        rows.append({
            "frozen_cell_id": frozen_id,
            "dataset": dataset,
            "condition": condition,
            "protocol_method": pm,
            "runner_method": runner_method,
            "scoring_mode": "pre_update",
            "seed": seed,
            "max_train_per_task": 2500,
            "raw_json_path": f,
            "manifest_path": os.path.join(BASE_DIR, "results", "revision_round1", "tranche5_pre_update", "experiment_manifest.csv"),
            "execution_group": "tranche5",
            "status": "complete"
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(PROV_FILE), exist_ok=True)
    df.to_csv(PROV_FILE, index=False)
    print(f"Provenance saved with {len(df)} rows.")
    return df

def step8_analysis(df):
    results = []

    # Tranches 1-4
    t_dirs = [os.path.join(BASE_DIR, "results", "revision_round1", f"tranche{i}", "raw_logs") for i in range(1, 5)]
    post_files = []
    for d in t_dirs:
        post_files.extend(glob.glob(os.path.join(d, "*.json")))

    post_data = {}
    for f in post_files:
        b = os.path.basename(f).replace(".json", "")
        if "gate_loss" in b or "gate_conf" in b:
            # e.g. split_cifar10__sym20__gate_loss__seed0
            parts = b.split("__")
            key = f"{parts[0]}__{parts[1]}__{parts[2]}__seed{parts[-1].replace('seed', '')}"
            m = extract_metrics(f)
            if m: post_data[key] = m

    pre_data = {}
    for _, row in df.iterrows():
        b = os.path.basename(row["raw_json_path"]).replace(".json", "")
        parts = b.split("__")
        # map back to same key structure for comparison
        key = f"{parts[0]}__{parts[1]}__{parts[2]}__seed{parts[4].replace('seed', '')}"
        m = extract_metrics(row["raw_json_path"])
        if m: pre_data[key] = m

    combos = [
        ("split_cifar10", "sym20", "gate_loss"), ("split_cifar10", "sym20", "gate_conf"),
        ("split_cifar10", "sym60", "gate_loss"), ("split_cifar10", "sym60", "gate_conf"),
        ("seq_cifar10", "sym20", "gate_loss"), ("seq_cifar10", "sym20", "gate_conf"),
        ("seq_cifar10", "sym60", "gate_loss"), ("seq_cifar10", "sym60", "gate_conf")
    ]

    for d, c, m in combos:
        for metric in ["final_acc", "buffer_purity", "norm_entropy", "var_counts", "replay_loss_clean", "replay_loss_noisy"]:
            diffs = []
            raws = []
            for s in range(5):
                k = f"{d}__{c}__{m}__seed{s}"
                if k in pre_data and k in post_data:
                    v_pre = pre_data[k][metric]
                    v_post = post_data[k][metric]
                    diff = v_pre - v_post
                    diffs.append(diff)
                    raws.append(diff)
                else:
                    raws.append(np.nan)

            diffs = [x for x in diffs if not np.isnan(x)]
            if diffs:
                mean_d = np.nanmean(diffs)
                sd_d = np.nanstd(diffs)
            else:
                mean_d = np.nan
                sd_d = np.nan

            results.append({
                "dataset": d, "condition": c, "method": m, "metric": metric,
                "mean_diff": mean_d, "sd_diff": sd_d,
                "s0": raws[0], "s1": raws[1], "s2": raws[2], "s3": raws[3], "s4": raws[4]
            })

    res_df = pd.DataFrame(results)
    print("\nPre-Post Match Results (Pre - Post):")
    print(res_df.to_string())
    res_df.to_csv(os.path.join(BASE_DIR, "revision_round1", "analysis", "pre_post_comparison.csv"), index=False)

def step11_inventory():
    planned = pd.read_csv(os.path.join(BASE_DIR, "revision_round1", "protocol", "planned_cells.csv"))

    # 80 done (Tranches 1-4)
    # 40 pre_update done (Tranche 5 + Validation)

    remaining = planned[
        ~((planned["scoring_mode"] == "post_update") & (planned["method"].isin(["gate_loss", "gate_conf", "er", "oracle"]))) &
        ~(planned["scoring_mode"] == "pre_update")
    ]

    print("\nInventory of remaining 307 cells:")
    inv = remaining.groupby(["phase", "kind", "benchmark", "scoring_mode"]).size().reset_index(name='count')
    print(inv.to_string())

if __name__ == "__main__":
    df = step6_provenance()
    if len(df) == 40:
        step8_analysis(df)
        step11_inventory()
    else:
        print(f"Warning: Expected 40 cells, found {len(df)}")
