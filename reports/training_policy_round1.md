# Training - `policy_round1`

- data: 50075 decisions from 270 elite games (tags: bootstrap, selfplay1)
- elite score mean **265.3**, max 552.0
- network: state 109 -> 256/128, candidate scorer over 48 slots, 287,024 params
- best validation policy loss: **1.1488**
- final val top-1 **0.453**, top-3 **0.809**

| epoch | train loss | val loss | val top-1 | val top-3 | val value MSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.3593 | 1.2528 | 0.450 | 0.812 | 0.342 |
| 2 | 1.3191 | 1.2687 | 0.447 | 0.807 | 0.348 |
| 3 | 1.2970 | 1.2607 | 0.452 | 0.808 | 0.311 |
| 4 | 1.2829 | 1.2745 | 0.440 | 0.802 | 0.311 |
| 5 | 1.2684 | 1.2889 | 0.449 | 0.802 | 0.404 |
| 6 | 1.2544 | 1.2708 | 0.451 | 0.810 | 0.350 |
| 7 | 1.2443 | 1.2800 | 0.448 | 0.804 | 0.372 |
| 8 | 1.2307 | 1.2684 | 0.453 | 0.809 | 0.351 |
