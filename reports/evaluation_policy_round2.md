# Evaluation - `policy_round2`

40 games per agent on held-out seeds 900000..900039 (every agent sees the identical boards).

| spec                                 |   games |   mean |   std |   median |   p10 |   p90 |   best |   goal_rate |   mean_pop |   mean_tech_nodes |   win_vs_heuristic |   lift_vs_heuristic |
|:-------------------------------------|--------:|-------:|------:|---------:|------:|------:|-------:|------------:|-----------:|------------------:|-------------------:|--------------------:|
| model:models/policy_round2.keras:0.0 |      40 |  190.3 |  55.6 |    169.5 | 141   | 258.5 |    364 |           0 |        3.6 |               3.9 |               0.75 |                43.8 |
| heuristic                            |      40 |  146.6 |  50.8 |    157.5 |  75.2 | 196.7 |    249 |           0 |        4.3 |               3.6 |               0    |                 0   |
| scripted                             |      40 |   18   |  15.9 |     19   |   0   |  41.1 |     49 |           0 |        7.8 |               1.3 |               0    |              -128.6 |
| random                               |      40 |    2.3 |   2   |      1.5 |   0   |   6   |      6 |           0 |        1.2 |               0.5 |               0    |              -144.3 |

Score is the game's own final score: accumulated science + culture + faith after 40 turns.
