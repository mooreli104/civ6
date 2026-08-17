# Civ 6 game data - cleaned dataset summary

Source: `civ6-roguelike-ai@2.1.0` from https://registry.npmjs.org (fetched 2026-08-17T08:20:46Z)

| table | rows | columns |
|---|---:|---:|
| `terrains.csv` | 14 | 10 |
| `features.csv` | 12 | 10 |
| `resources.csv` | 18 | 10 |
| `districts.csv` | 12 | 15 |
| `district_adjacency.csv` | 20 | 10 |
| `improvements.csv` | 18 | 15 |
| `improvement_adjacency.csv` | 14 | 10 |
| `tech_nodes.csv` | 90 | 13 |
| `cards.csv` | 36 | 9 |
| `eurekas.csv` | 15 | 5 |

## Highlights

- 14 terrain types; mean total base yield 1.64
- 12 districts with 20 adjacency rules
- 18 improvements with 14 adjacency rules
- 90 tech nodes across 3 trees, layers [1, 2, 3, 4]
- 36 shop cards over tiers [1, 2, 3], cost range 1-6

## Validation

- All referential-integrity checks passed.
