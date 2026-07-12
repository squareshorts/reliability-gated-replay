import json
import pandas as pd
from pathlib import Path

def main():
    results_dir = Path("results/neural_networks_submission/raw_logs")
    if not results_dir.exists():
        print("No raw logs found.")
        return

    data = []
    for f in results_dir.glob("*.json"):
        with open(f, "r") as fp:
            try:
                run_data = json.load(fp)
            except Exception:
                continue

            acc = run_data.get("acc_matrix", [[0]])[-1]
            avg_acc = sum(acc) / len(acc) if acc else 0.0
            
            purity = float("nan")
            cons_state = run_data.get("consolidation", [{}])[-1]
            if cons_state:
                purity = cons_state.get("buffer_purity", float("nan"))
            
            method = run_data.get("label", "unknown")
            dataset = run_data.get("dataset", "unknown")
            condition = run_data.get("condition", "unknown")
            seed = run_data.get("seed", 0)

            data.append({
                "dataset": dataset,
                "condition": condition,
                "method": method,
                "seed": seed,
                "accuracy": avg_acc,
                "purity": purity
            })

    if not data:
        print("No data collected.")
        return

    df = pd.DataFrame(data)
    
    # Simple aggregation
    agg = df.groupby(["dataset", "condition", "method"]).agg({
        "accuracy": ["mean", "std"],
        "purity": ["mean", "std"]
    }).reset_index()

    # Save to CSV
    out_dir = Path("results/arc_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "arc_main_table.csv", index=False)
    print(f"Report saved to {out_dir / 'arc_main_table.csv'}")

    import glob
    import os
    # Also output a Markdown table for the agent's artifact directory
    # Find the current artifact directory for the agent if possible
    # We will just write it to out_dir and let the agent move it
    md_content = f"# ARC-Replay Experimental Results\n\n"
    md_content += "This report compares ARC-Replay (Lite and Full) against standard Replay (ER, DER++) and Gated Replay methods.\n\n"
    md_content += df.groupby(["dataset", "condition", "method"])["accuracy"].mean().unstack("method").to_markdown()
    md_content += "\n\n## Buffer Purity\n\n"
    md_content += df.groupby(["dataset", "condition", "method"])["purity"].mean().unstack("method").to_markdown()
    
    with open(out_dir / "arc_report.md", "w") as f:
        f.write(md_content)
    print(f"Markdown report generated at {out_dir / 'arc_report.md'}")

if __name__ == "__main__":
    main()
