#!/usr/bin/env python
"""Analyze IRGR Submission results."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "neural_networks_submission"

def load_results():
    manifest = RESULTS_DIR / "experiment_manifest.csv"
    if not manifest.exists():
        print(f"No manifest found at {manifest}")
        return []
        
    runs = []
    with open(manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "done":
                continue
            run_id = row["run_id"]
            fp = RESULTS_DIR / "raw_logs" / f"{run_id}.json"
            if fp.exists():
                try:
                    with open(fp) as fj:
                        data = json.load(fj)
                    runs.append({"meta": row, "data": data})
                except json.JSONDecodeError:
                    print(f"Skipping corrupted JSON: {fp}")
    return runs

def compute_stats(baseline_accs, irgr_accs):
    n = len(baseline_accs)
    if n == 0 or stats is None:
        return {"mean_diff": np.nan, "p_value": np.nan, "cohens_dz": np.nan}
    
    diffs = np.array(irgr_accs) - np.array(baseline_accs)
    mean_diff = float(np.mean(diffs))
    
    if n < 2 or np.std(diffs) < 1e-8:
        return {"mean_diff": mean_diff, "p_value": 1.0, "cohens_dz": 0.0}
    
    # Wilcoxon signed-rank test
    try:
        _, p_val = stats.wilcoxon(diffs)
    except Exception:
        p_val = 1.0
        
    # Cohen's dz
    dz = mean_diff / np.std(diffs, ddof=1)
    
    return {"mean_diff": mean_diff, "p_value": float(p_val), "cohens_dz": float(dz)}

def main():
    runs = load_results()
    if not runs:
        return
        
    # Group by dataset and condition
    groups = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        ds = r["meta"]["dataset"]
        cond = r["meta"]["noise_type"] + ":" + str(r["meta"]["noise_rate"])
        meth = r["meta"]["method"]
        seed = int(r["meta"]["seed"])
        
        accs = r["data"]["acc_matrix"][-1]
        final_acc = np.mean(accs)
        
        # Overwrite to keep only the latest run per seed
        groups[(ds, cond)][meth][seed] = final_acc

    print("=== FINAL ACCURACY REPORT ===")
    for (ds, cond), methods in groups.items():
        print(f"\\nDataset: {ds} | Noise: {cond}")
        
        baseline_accs = methods.get("er", {})
        irgr_accs = methods.get("irgr", {})
        
        matched_seeds = set(baseline_accs.keys()) & set(irgr_accs.keys())
        b_vals = [baseline_accs[s] for s in matched_seeds]
        i_vals = [irgr_accs[s] for s in matched_seeds]
        
        for meth, vals_dict in methods.items():
            arr = list(vals_dict.values())
            print(f"  {meth:15s}: {np.mean(arr):.4f} +/- {np.std(arr):.4f} (n={len(arr)})")
            
            # Print diagnostics for irgr variants
            if "irgr" in meth:
                adm, quar, inv = [], [], []
                for seed_val in vals_dict.keys():
                    # find the run data for this seed (searching backwards to get the latest)
                    run_data = next((r["data"] for r in reversed(runs) if r["meta"]["dataset"] == ds and r["meta"]["noise_type"] + ":" + str(r["meta"]["noise_rate"]) == cond and r["meta"]["method"] == meth and int(r["meta"]["seed"]) == seed_val), None)
                    if run_data and "consolidation" in run_data:
                        cs = run_data["consolidation"][-1]  # get last task's state
                        adm.append(cs.get("admission_rate", 0))
                        quar.append(cs.get("quarantine_fraction", 0))
                        inv.append(cs.get("inversion_trips", 0))
                if adm:
                    print(f"      -> admission_rate: {np.mean(adm):.4f}, quarantine: {np.mean(quar):.4f}, guard_trips: {np.mean(inv):.1f}")

            
        if b_vals and i_vals:
            stat = compute_stats(b_vals, i_vals)
            print(f"  --> IRGR vs ER Diff: {stat['mean_diff']:.4f}, p-val: {stat['p_value']:.4f}, Cohen's dz: {stat['cohens_dz']:.4f}")

if __name__ == "__main__":
    main()
