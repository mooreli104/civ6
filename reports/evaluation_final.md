# Evaluation - `final`

60 games per agent on held-out seeds 900000..900059 (every agent sees the identical boards).

| spec                                 |   games |   mean |   std |   median |   p10 |   p90 |   best |   goal_rate |   mean_pop |   mean_tech_nodes |   win_vs_heuristic |   lift_vs_heuristic |
|:-------------------------------------|--------:|-------:|------:|---------:|------:|------:|-------:|------------:|-----------:|------------------:|-------------------:|--------------------:|
| model:models/policy_round2.keras:0.0 |      60 |  192.9 |  52.3 |    183.5 | 141   | 257.7 |    364 |           0 |        3.5 |               4   |              0.783 |                48.8 |
| heuristic                            |      60 |  144.2 |  45.9 |    150.5 |  90.1 | 196   |    249 |           0 |        3.9 |               3.5 |              0     |                 0   |
| scripted                             |      60 |   20.1 |  19.8 |     21.5 |   0   |  45.1 |     83 |           0 |        7.3 |               1.3 |              0     |              -124.1 |
| random                               |      60 |    2.2 |   1.8 |      2   |   0   |   5   |      6 |           0 |        1.2 |               0.5 |              0     |              -142   |

Score is the game's own final score: accumulated science + culture + faith after 40 turns.
