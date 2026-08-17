# Training - `policy_unpaired1`

- data: 48941 decisions from 270 elite games (tags: bootstrap, selfplay1)
- elite score mean **275.4**, max 554.0
- network: state 109 -> 256/128, candidate scorer over 48 slots, 287,024 params
- best validation policy loss: **1.0451**
- final val top-1 **0.447**, top-3 **0.806**

| epoch | train loss | val loss | val top-1 | val top-3 | val value MSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.2799 | 1.1835 | 0.447 | 0.809 | 0.456 |
| 2 | 1.2340 | 1.1940 | 0.447 | 0.800 | 0.453 |
| 3 | 1.2129 | 1.2345 | 0.441 | 0.796 | 0.555 |
| 4 | 1.1986 | 1.2042 | 0.451 | 0.810 | 0.520 |
| 5 | 1.1816 | 1.2215 | 0.449 | 0.806 | 0.559 |
| 6 | 1.1698 | 1.2175 | 0.450 | 0.808 | 0.535 |
| 7 | 1.1598 | 1.2469 | 0.447 | 0.805 | 0.592 |
| 8 | 1.1482 | 1.2186 | 0.447 | 0.806 | 0.530 |
