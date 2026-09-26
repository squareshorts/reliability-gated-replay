TITLE
Protocol Amendment 001 — Operational Definition of Scoring Timing

DATE
2026-09-23

REASON
The frozen protocol specified scoring_mode labels but did not define their
runtime semantics. This ambiguity was discovered before execution of any
pre_update cell.

IMPORTANT TRANSPARENCY
- post_update experiments had already been executed when the ambiguity was discovered.
- no pre_update experiment had yet been executed.
- this amendment therefore defines the previously unspecified timing contrast before collection of any pre_update results.
- the amendment does not modify datasets, methods, seeds, thresholds, optimization, replay policy, sample caps, or existing completed runs.

OPERATIONAL DEFINITIONS

post_update
-----------
For the current incoming batch:

1. compute the normal current-stream/replay training objective;
2. backpropagate;
3. execute optimizer.step();
4. score that SAME incoming batch for replay admission using the updated live
   model;
5. apply the resulting admission decision to the replay buffer.

This is the historical behavior already used by the completed post_update
runs.

pre_update
----------
For the current incoming batch:

1. using the live model state immediately BEFORE optimizer.step() for that
   batch, compute the gate score/admission information for that SAME incoming
   batch;
2. cache the score/admission information;
3. execute the otherwise unchanged normal current-stream/replay training
   objective, backpropagation, and optimizer.step();
4. after optimizer.step(), apply the CACHED pre-update admission result to the
   replay buffer;
5. do NOT recompute the gate score after the update.

The replay-buffer insertion time therefore remains matched to post_update.
The intended experimental factor is solely the parameter state used to compute
the gate score.

Clarify explicitly:
- pre_update does NOT mean early_snapshot;
- pre_update does NOT use a frozen/peer model;
- pre_update does NOT clean labels;
- pre_update does NOT alter when an admitted item becomes available for future
  replay;
- pre_update does NOT alter replay sampling or optimizer behavior.

OTHER SCORING_MODE LABELS
-------------------------
The following additional scoring_mode labels appearing in planned_cells.csv or freeze_protocol.py remain underspecified:
- pre_update_shadow_small_loss_and_matched_random
- pre_update_at_task1_admission
- none_or_oracle

These labels require separate operational clarification before execution.
