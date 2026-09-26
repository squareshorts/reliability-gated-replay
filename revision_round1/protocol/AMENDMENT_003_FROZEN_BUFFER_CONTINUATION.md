TITLE
Protocol Amendment 003 — Phase-3 Frozen-Buffer Continuation Semantics

DATE
2026-09-24

SCOPE
This amendment defines the runtime semantics of the 160 Phase-3
frozen-buffer continuation cells whose scoring_mode is:

pre_update_at_task1_admission

It applies only to continuations that use method=frozen_buffer_replay. It does
not change the P3H common-history construction defined by Amendment 002, and
does not apply to the already-defined P3A no-replay continuations.

STARTING STATE
--------------
For each frozen-buffer continuation:

1. start from its corresponding completed P3H artifact;
2. restore the exact saved task-1 model state and optimizer state; and
3. select the saved task-1 shadow buffer specified by buffer_policy:
   - small_loss_pre; or
   - random_count_matched.

MEANING OF pre_update_at_task1_admission
----------------------------------------
Buffer membership was decided during task 1 using the already-defined
pre-update admission semantics. During continuation there is no rescoring and
no new admission decision. The selected task-1 buffer is frozen for the entire
continuation.

CONTINUATION
------------
- Train tasks 2 onward using recipe R0.
- Current-stream training remains unchanged.
- Replay samples come only from the selected frozen task-1 buffer.
- No task-2+ example may be inserted, replaced, or removed from that buffer.
- Reuse the existing R0 replay sampling and batch mechanics; introduce no new
  sampling hyperparameters.

REPLAY COEFFICIENT
------------------
- lambda=1 or lambda=2 multiplies the existing replay-loss term at the same
  point and with the same normalization already used by the R0 replay code.
- It must not scale the current-stream loss.
- No other optimizer or loss setting changes.

REPLAY LABEL FACTOR
-------------------
For the same frozen examples:

- observed: replay using the observed task-1 training labels stored with the
  buffer.
- clean_reference_oracle: replay the identical frozen inputs and membership,
  but replace only the replay target with the corresponding clean-reference
  label.

The clean-reference label must not alter:

- buffer membership;
- replay sampling;
- current-stream labels;
- corruption masks; or
- continuation data order.

RNG AND BRANCH ISOLATION
------------------------
- Every branch starts from the same saved P3H training and data RNG state.
- Replay sampling uses a separate deterministic RNG stream derived from frozen
  seed + buffer policy + replay-label policy + lambda.
- The replay RNG must not perturb the continuation data-order RNG.
- One branch must never modify another branch's saved P3H artifact.

REQUIRED PROVENANCE
-------------------
Record for every continuation:

- parent P3H cell and artifact hash;
- buffer policy;
- replay-label policy;
- lambda;
- seed;
- replay RNG seed and state;
- frozen-buffer fingerprint;
- confirmation that buffer membership remained unchanged; and
- continuation task range.

TRANSPARENCY
------------
- These semantics were missing from the original frozen protocol.
- They are defined prospectively before any frozen-buffer continuation run.
- No previous result is changed.
