TITLE
Protocol Amendment 004 — Phase-4 R1 Recipe

DATE
2026-09-25

SCOPE
This amendment formalizes the R1 specification for the existing 90 frozen
Phase-4 R1 cells. It preserves their exact cell identities and does not add,
remove, tune, or select cells.

TRAINING
--------
- Use the same scratch ResNet-18 architecture and evaluation regime as R0.
- Train for 50 epochs per task.
- Use SGD with lr=0.1, momentum=0.9, weight_decay=0.0005, and
  Nesterov=False.
- Apply cosine learning-rate decay to zero over each task's 50 epochs; step
  the scheduler after each epoch.
- Reset the optimizer and scheduler at each task boundary, but do not reset
  model weights.
- Use current batch=128, replay batch=32, buffer capacity=500, and
  replay-loss coefficient=1.
- Set max_train_per_task=2500.
- Preserve R0 sample IDs, splits, task/class order, corruption masks,
  normalization, evaluation cap, and seeds 0--4.

VIEWS AND ADMISSION
-------------------
- For current-stream and replay training, apply random crop 32 with padding
  4, then horizontal flip with p=0.5, then R0 normalization.
- Store canonical unaugmented images.
- For gate scoring, validation, and testing, apply normalization only.
- Use independent deterministic augmentation RNG streams. They must not
  perturb data-order, corruption, or admission RNG streams.
- Preserve existing gate thresholds, admission and replacement rules, and
  replay sampling.
- small_loss_pre and confidence_pre use Amendment-001 pre_update semantics.
- Resolve none_or_oracle explicitly: ER has no gate; oracle uses clean-label
  equality for admission only. Oracle does not correct current-stream or
  replay training labels.

DESIGN AND RECORDS
------------------
- Preserve the exact 90 frozen R1 cells; do not tune or select cells.
- Record the full resolved configuration, recipe ID and hash, source version,
  seeds, data and corruption fingerprints, runtime, and existing required
  outcomes.
- Record optimization_steps as actual optimizer.step counts per task and in
  total.
- Record clean-data acquisition and retention as the full task-accuracy
  matrix, acquisition from its diagonal, and final retention from its final
  row.
- R1 run identities must differ from R0 run identities.

TRANSPARENCY
------------
- R0 and Phase-3 results were already reviewed, but no R1 result has been
  collected.
- This formalizes a prior conversation proposal; it is not a recovered
  original protocol definition or an already validated stronger baseline.
- No previous result changes.
- R1 tests the combined recipe, not individual component effects.
