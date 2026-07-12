# ARC-Replay Experimental Results

This report compares ARC-Replay (Lite and Full) against standard Replay (ER, DER++) and Gated Replay methods.

|                            |   arc_full |   arc_lite |
|:---------------------------|-----------:|-----------:|
| ('split_cifar10', 'sym20') |     0.5096 |     0.5267 |
| ('split_cifar10', 'sym60') |     0.5006 |     0.4704 |

## Buffer Purity

|                            |   arc_full |   arc_lite |
|:---------------------------|-----------:|-----------:|
| ('split_cifar10', 'sym20') |   0.817143 |   0.793388 |
| ('split_cifar10', 'sym60') |   0.321192 |   0.33123  |