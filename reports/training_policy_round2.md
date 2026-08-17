# Training - `policy_round2`

- data: 30157 decisions from 175 elite games (tags: selfplay1, selfplay2)
- elite score mean **256.3**, max 575.0
- network: state 109 -> 256/128, candidate scorer over 48 slots, 287,024 params
- best validation policy loss: **1.2235**
- final val top-1 **0.520**, top-3 **0.846**

| epoch | train loss | val loss | val top-1 | val top-3 | val value MSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.3602 | 1.4309 | 0.528 | 0.845 | 0.662 |
| 2 | 1.3089 | 1.4292 | 0.524 | 0.841 | 0.661 |
| 3 | 1.2861 | 1.4205 | 0.527 | 0.846 | 0.652 |
| 4 | 1.2663 | 1.4553 | 0.515 | 0.842 | 0.709 |
| 5 | 1.2604 | 1.4465 | 0.516 | 0.846 | 0.698 |
| 6 | 1.2370 | 1.4408 | 0.523 | 0.846 | 0.690 |
| 7 | 1.2270 | 1.4713 | 0.517 | 0.843 | 0.757 |
| 8 | 1.2173 | 1.4541 | 0.520 | 0.846 | 0.709 |
