#!/bin/bash
set -e

# Small smoke test first
echo "Running smoke test..."
python scripts/run_nn_submission.py --datasets split_cifar10 --conditions sym20,sym60 --methods arc_lite,arc_full --seeds 0 --epochs 1 --max-train 100

echo "Smoke test passed. Running ARC full grid..."
# Run the ARC grid defined in the run_nn_submission.py preset
python scripts/run_nn_submission.py --preset arc_grid

echo "ARC full grid complete. Generating report..."
python scripts/report_arc.py
