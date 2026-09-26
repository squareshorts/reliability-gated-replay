TITLE
Protocol Amendment 002 — Phase-3 Persistence Common-History Semantics

DATE
2026-09-23

SCOPE
This amendment defines the runtime semantics of the Phase-3 P3H common-history
cells whose scoring_mode is:

pre_update_shadow_small_loss_and_matched_random

It applies only to creation of the task-1 common history and its two shadow
buffers. It does not define the downstream continuation scoring mode
pre_update_at_task1_admission.

IMPORTANT TRANSPARENCY
- these runtime semantics were not specified in the original frozen protocol;
- the ambiguity was identified before any P3H cell was executed;
- this amendment defines the missing semantics prospectively;
- no completed result is changed.

COMMON TRAINING HISTORY
-----------------------
For each frozen P3H benchmark/noise/seed cell:

1. train task 1 exactly once with method ordinary_current_batch and recipe R0;
2. neither shadow buffer may affect model training, gradients, optimizer state,
   replay, or data order;
3. both shadow policies observe the same ordered incoming minibatches.

The common-history run contains no replay from either shadow buffer.

SMALL-LOSS SHADOW
-----------------
The small-loss shadow uses the already implemented Phase-2 small_loss policy.

For every incoming minibatch:

1. use scoring_mode=pre_update exactly as defined in Amendment 001;
2. score using the observed training label, exactly as the existing small_loss
   gate does;
3. reuse the same threshold, gate configuration, buffer capacity, and admission
   mechanics as the existing Phase-2 pre_update small_loss implementation;
4. introduce no new hyperparameters.

The resulting admission decision is observational with respect to the common
training history: it may update only the small-loss shadow buffer and its
admission log.

COUNT-MATCHED RANDOM SHADOW
---------------------------
For each incoming minibatch:

1. let k be the number of examples admitted by the small-loss shadow for that
   minibatch;
2. uniformly select exactly k examples without replacement from that same
   incoming minibatch;
3. admit those selected examples to the random shadow buffer;
4. use an independent deterministic RNG stream derived from the frozen seed;
5. random-shadow decisions must not alter either the small-loss RNG stream or
   the training RNG stream.

Matching is exact per minibatch. It is not expected-rate matching.

BUFFER HANDLING
---------------
- both shadow buffers use the same capacity and replacement mechanics as the
  corresponding existing replay-buffer implementation;
- after their admission decisions, the two buffers evolve independently;
- neither buffer is replayed during the common-history task-1 run.

STABLE IDS AND ADMISSION MULTIPLICITY
-------------------------------------
- every incoming training example must carry a stable example ID;
- every admission event must be logged by stable ID;
- admission_multiplicity is the number of admission events for each stable ID
  across repeated presentations;
- repeated admissions must not be silently collapsed.

TASK-1 SAVE BOUNDARY
--------------------
Immediately after task-1 training finishes, save:

- model state;
- optimizer state;
- both shadow buffers;
- stable IDs and admission multiplicities;
- the corruption mask and labels needed to reproduce task 1;
- relevant RNG states;
- fingerprints sufficient to verify common-history identity.

CONTINUATION FORKS
------------------
All downstream continuations for the same benchmark/noise/seed must start from
the exact same saved task-1 model state and optimizer state.

The two shadow buffers differ only by admission policy. No continuation may
retroactively change the common task-1 training history or either saved shadow
buffer.

DEFERRED CONTINUATION SEMANTICS
-------------------------------
The scoring mode pre_update_at_task1_admission remains undefined. It requires
Amendment 003 before any dependent continuation cell can run.
