import os
import csv
from pathlib import Path
from scripts.run_nn_submission import append_manifest, MANIFEST_COLS

def test_append_manifest_deduplication(tmp_path):
    manifest_path = tmp_path / "experiment_manifest.csv"

    # Write initial row
    row1 = {"run_id": "test_run_1", "status": "failed", "notes": "first attempt"}
    append_manifest(manifest_path, row1)

    # Verify it exists
    with open(manifest_path, "r") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    assert reader[0]["run_id"] == "test_run_1"
    assert reader[0]["status"] == "failed"

    # Write duplicate run_id but different status
    row2 = {"run_id": "test_run_1", "status": "done", "notes": "second attempt"}
    append_manifest(manifest_path, row2)

    # Verify it replaced the first one
    with open(manifest_path, "r") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    assert reader[0]["run_id"] == "test_run_1"
    assert reader[0]["status"] == "done"
    assert reader[0]["notes"] == "second attempt"

    # Write different run_id
    row3 = {"run_id": "test_run_2", "status": "done"}
    append_manifest(manifest_path, row3)

    with open(manifest_path, "r") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 2
    assert reader[0]["run_id"] == "test_run_1"
    assert reader[1]["run_id"] == "test_run_2"
