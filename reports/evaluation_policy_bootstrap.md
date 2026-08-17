# Evaluation - `policy_bootstrap`

40 games per agent on held-out seeds 900000..900039 (every agent sees the identical boards).

| spec                                    |   games |   mean |   std |   median |   p10 |   p90 |   best |   goal_rate |   mean_pop |   mean_tech_nodes |   win_vs_heuristic |   lift_vs_heuristic |
|:----------------------------------------|--------:|-------:|------:|---------:|------:|------:|-------:|------------:|-----------:|------------------:|-------------------:|--------------------:|
| model:models/policy_bootstrap.keras:0.0 |      40 |  167.6 |  54.8 |    164   | 108.4 | 233.3 |    329 |           0 |        4.1 |               3.7 |              0.7   |                21.1 |
| model:models/policy_bootstrap.keras:0.4 |      40 |  165.2 |  71.7 |    151.5 |  84   | 254.2 |    386 |           0 |        5.7 |               4.8 |              0.625 |                18.6 |
| heuristic                               |      40 |  146.6 |  50.8 |    157.5 |  75.2 | 196.7 |    249 |           0 |        4.3 |               3.6 |              0     |                 0   |
| scripted                                |      40 |   18   |  15.9 |     19   |   0   |  41.1 |     49 |           0 |        7.8 |               1.3 |              0     |              -128.6 |
| random                                  |      40 |    2.3 |   2   |      1.5 |   0   |   6   |      6 |           0 |        1.2 |               0.5 |              0     |              -144.3 |

Score is the game's own final score: accumulated science + culture + faith after 40 turns.
