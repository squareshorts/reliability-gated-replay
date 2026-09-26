# Final execution status

## Required frozen cells

| Phase | Frozen required cells | Completed |
|---|---:|---:|
| Phase 2 scoring | 120 | 120/120 |
| Phase 3 persistence | 200 | 200/200 |
| Phase 4 recipe sensitivity | 100 | 100/100 |
| **Required total** | **420** | **420/420** |

- Required frozen cells executed: **420/420**
- Failed required cells: **0**

## Conditional Phase 8 Mammoth ER cells

The frozen ledger contains seven explicitly conditional P8 Mammoth ER cells:

- `P8__mammoth_seq_cifar10__sym20__er__s0`
- `P8__mammoth_seq_cifar10__sym20__er__s1`
- `P8__mammoth_seq_cifar10__sym20__er__s2`
- `P8__mammoth_seq_cifar10__asym40__er__s0`
- `P8__mammoth_seq_cifar10__asym40__er__s1`
- `P8__mammoth_seq_cifar10__asym40__er__s2`
- `P8__mammoth_seq_cifar10__sym60__er__s2`

The trigger condition was not satisfied because the required pinned Mammoth checkout (`pinned_mammoth_e75a491c`) and matched raw AER provenance are unavailable locally. Under the frozen ledger rule `conditional_only_if_existing_AER_provenance_can_be_matched`, all seven are:

**CONDITIONAL — NOT TRIGGERED**

They are not missing, failed, or incomplete required runs.

## Final ledger

- Total frozen cells: **427**
- Required executed: **420**
- Conditional not triggered: **7**
- Failed required: **0**
