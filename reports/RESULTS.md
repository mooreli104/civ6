# Results

Source data: `civ6-roguelike-ai@2.1.0` from https://registry.npmjs.org (sha256 `241e4cafafc7653c...`, fetched 2026-08-17T08:20:46Z).

Every number below is produced by the scripts in `src/`; rerunning the pipeline regenerates this file.

## 1. Cleaned game data

| table                     |   rows |   columns |
|:--------------------------|-------:|----------:|
| cards.csv                 |     36 |         9 |
| district_adjacency.csv    |     20 |        10 |
| districts.csv             |     12 |        15 |
| eurekas.csv               |     15 |         5 |
| features.csv              |     12 |        10 |
| improvement_adjacency.csv |     14 |        10 |
| improvements.csv          |     18 |        15 |
| resources.csv             |     18 |        10 |
| tech_nodes.csv            |     90 |        13 |
| terrains.csv              |     14 |        10 |

## 2. Generated gameplay data

| tag       |   games |   mean score |   p90 |   max |   elite games |   elite mean |
|:----------|--------:|-------------:|------:|------:|--------------:|-------------:|
| bootstrap |    1200 |         81.1 | 213.1 |   519 |           180 |        266.8 |
| selfplay1 |     640 |        135.2 | 248.1 |   552 |            96 |        262.8 |
| selfplay2 |     640 |        134   | 233   |   575 |            96 |        245.4 |
| unpaired1 |     600 |        139.5 | 248.1 |   554 |            90 |        293.8 |

## 3. Trained models

| model            |   decisions |   elite games |   val top-1 |   val top-3 |   best val loss |
|:-----------------|------------:|--------------:|------------:|------------:|----------------:|
| policy_bootstrap |       33994 |           180 |       0.438 |       0.787 |          1.0865 |
| policy_round1    |       50075 |           270 |       0.453 |       0.809 |          1.1488 |
| policy_round2    |       30157 |           175 |       0.52  |       0.846 |          1.2235 |
| policy_unpaired1 |       48941 |           270 |       0.447 |       0.806 |          1.0451 |

## 4. Evaluation (held-out seeds, paired)


### `final`

| spec                                 |   games |   mean |   median |   p90 |   best |   goal_rate |   win_vs_heuristic |
|:-------------------------------------|--------:|-------:|---------:|------:|-------:|------------:|-------------------:|
| model:models/policy_round2.keras:0.0 |      60 |  192.9 |    183.5 | 257.7 |    364 |           0 |              0.783 |
| heuristic                            |      60 |  144.2 |    150.5 | 196   |    249 |           0 |              0     |
| scripted                             |      60 |   20.1 |     21.5 |  45.1 |     83 |           0 |              0     |
| random                               |      60 |    2.2 |      2   |   5   |      6 |           0 |              0     |

### `policy_bootstrap`

| spec                                    |   games |   mean |   median |   p90 |   best |   goal_rate |   win_vs_heuristic |
|:----------------------------------------|--------:|-------:|---------:|------:|-------:|------------:|-------------------:|
| model:models/policy_bootstrap.keras:0.0 |      40 |  167.6 |    164   | 233.3 |    329 |           0 |              0.7   |
| model:models/policy_bootstrap.keras:0.4 |      40 |  165.2 |    151.5 | 254.2 |    386 |           0 |              0.625 |
| heuristic                               |      40 |  146.6 |    157.5 | 196.7 |    249 |           0 |              0     |
| scripted                                |      40 |   18   |     19   |  41.1 |     49 |           0 |              0     |
| random                                  |      40 |    2.3 |      1.5 |   6   |      6 |           0 |              0     |

### `policy_round1`

| spec                                 |   games |   mean |   median |   p90 |   best |   goal_rate |   win_vs_heuristic |
|:-------------------------------------|--------:|-------:|---------:|------:|-------:|------------:|-------------------:|
| model:models/policy_round1.keras:0.0 |      40 |  185.2 |    184   | 262   |    340 |           0 |              0.725 |
| heuristic                            |      40 |  146.6 |    157.5 | 196.7 |    249 |           0 |              0     |
| scripted                             |      40 |   18   |     19   |  41.1 |     49 |           0 |              0     |
| random                               |      40 |    2.3 |      1.5 |   6   |      6 |           0 |              0     |

### `policy_round2`

| spec                                 |   games |   mean |   median |   p90 |   best |   goal_rate |   win_vs_heuristic |
|:-------------------------------------|--------:|-------:|---------:|------:|-------:|------------:|-------------------:|
| model:models/policy_round2.keras:0.0 |      40 |  190.3 |    169.5 | 258.5 |    364 |           0 |               0.75 |
| heuristic                            |      40 |  146.6 |    157.5 | 196.7 |    249 |           0 |               0    |
| scripted                             |      40 |   18   |     19   |  41.1 |     49 |           0 |               0    |
| random                               |      40 |    2.3 |      1.5 |   6   |      6 |           0 |               0    |

### `policy_unpaired1`

| spec                                    |   games |   mean |   median |   p90 |   best |   goal_rate |   win_vs_heuristic |
|:----------------------------------------|--------:|-------:|---------:|------:|-------:|------------:|-------------------:|
| model:models/policy_unpaired1.keras:0.0 |      40 |  150.8 |    140.5 | 195.3 |    296 |           0 |              0.525 |
| heuristic                               |      40 |  146.6 |    157.5 | 196.7 |    249 |           0 |              0     |
| scripted                                |      40 |   18   |     19   |  41.1 |     49 |           0 |              0     |
| random                                  |      40 |    2.3 |      1.5 |   6   |      6 |           0 |              0     |
