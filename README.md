# Training a model to play Civilization VI

An end-to-end pipeline — **fetch → clean → generate → train → evaluate** — that
produces a TensorFlow policy network which plays a Civilization VI city-builder
and beats hand-written baselines on held-out games.

```
npm install                              # Node side (game engine bridge)
# python 3.11 + ~2 GB disk for TensorFlow
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python src/fetch_data.py                    # 1. fetch game data + engine
.venv/bin/python src/clean_data.py --strict           # 2. normalize + validate
.venv/bin/python src/generate_episodes.py --tag bootstrap --games 600
.venv/bin/python src/train.py --tag bootstrap --epochs 12
.venv/bin/python src/evaluate.py --model models/policy_bootstrap.keras --games 50
.venv/bin/python src/selfplay_loop.py --rounds 2      # optional self-improvement
```

## What the model actually plays

Civilization VI itself is closed-source and needs the retail client, so there is
no way to drive the shipping game headlessly from a container. The environment
used here is [`civ6-roguelike-ai`](https://www.npmjs.com/package/civ6-roguelike-ai),
a hex-grid Civ VI city-builder published on npm specifically for AI agents. It
reimplements the Civ VI systems that matter for city play:

| Civ VI system | in the environment |
|---|---|
| Hex map with terrain/feature/resource yields | 61 hexes, 14 terrains, 12 features, 18 resources |
| Districts with adjacency bonuses | 12 districts, 20 adjacency rules (campus +1 science per adjacent mountain, ...) |
| Tile improvements | 18 improvements (farm, mine, lighthouse, ...) with upgrade levels |
| Tech / civic / religion trees + eurekas | 3 trees x 4 layers, 90 nodes, 15 eurekas |
| Citizen assignment | population = workers; only worked tiles yield |
| Victory conditions | score / population / gold / faith / science / culture goals |

A game is 40 turns; the final score is accumulated science + culture + faith.

## Pipeline

### 1. Fetch (`src/fetch_data.py`)
Downloads the package tarball straight from `registry.npmjs.org`, verifies and
records its SHA-256, extracts it to `vendor/`, and writes
`data/raw/fetch_manifest.json` with full provenance.

### 2. Clean (`src/clean_data.py`)
`src/env/dump_gamedata.ts` imports the game's own TypeScript modules and emits
their rule tables as JSON; the Python side flattens them into validated CSVs
(`data/processed/*.csv`), a vocabulary of stable id→index maps
(`vocab.json`) used by the feature encoder, and `reports/data_summary.md`.
Validation covers duplicate ids, referential integrity (cards → terrains,
tech unlocks → districts/improvements, node → parent) and value ranges.

### 3. Generate (`src/generate_episodes.py`)
The Node bridge (`src/env/bridge.ts`) exposes the engine over a JSON-lines
protocol with a **seeded** PRNG, so a game is reproducible from its seed alone.
`CivEnv` (`src/env/game_env.py`) wraps it in a gym-like API. Generation runs in
two passes:

1. **search** — play many games with noisy agents, storing only the outcome of each;
2. **record** — replay the *elite* games (identical, because both engine and
   agent are seeded) and encode every decision into padded tensors.

### 4. Train (`src/train.py`, `src/model.py`)
The legal-action set changes shape at every decision, so the network scores
candidates instead of using a fixed action index:

```
state (S)      -> trunk -> h_s
actions (C, A) -> trunk -> h_a          logits = MLP([h_s, h_a, h_s * h_a])
masked softmax over C candidates        + value head predicting the final score
```

Training is self-imitation on elite games, weighted by episode quality, with
whole games (never individual decisions) held out for validation.

### 5. Evaluate (`src/evaluate.py`)
Every agent plays the *same* held-out seeds, so comparisons are paired: identical
boards, shop rolls and thresholds. Reports mean/median/p90 score, goal rate, and
paired win-rate against the heuristic baseline.

### 6. Self-improvement (`src/selfplay_loop.py`)
Generate with the current policy → keep elites → retrain warm-started → evaluate.

## Results

Score is the game's own final score: accumulated science + culture + faith after
40 turns. Every agent plays the *same* held-out seeds, so the comparison is
paired — identical boards, shop rolls and thresholds.

**Final model (`models/policy_round2.keras`), 60 held-out games per agent:**

| agent | mean | median | p90 | best | win rate vs heuristic |
|---|---:|---:|---:|---:|---:|
| **trained policy** | **192.9** | 183.5 | 257.7 | 364 | **78%** |
| heuristic baseline | 144.2 | 150.5 | 196.0 | 249 | — |
| scripted (the package's own example strategy) | 20.1 | 21.5 | 45.1 | 83 | 0% |
| random | 2.2 | 2.0 | 5.0 | 6 | 0% |

**Progression** (40 games per agent, same seeds throughout):

| model | trained on | mean score | win rate vs heuristic |
|---|---|---:|---:|
| heuristic teacher | — | 146.6 | — |
| `policy_bootstrap` | top 15% of 1200 heuristic games | 167.6 | 70% |
| `policy_unpaired1` | + self-play, elites picked globally | 150.8 | 53% |
| `policy_round1` | + self-play, elites picked per board | 185.2 | 73% |
| `policy_round2` | + a second paired self-play round | 190.3 | 75% |

The policy is trained only by imitating good games — no reward gradients — and
still ends up well ahead of the heuristic it learned from.

### What the self-play rounds taught us

The first self-play round made the policy *worse* (150.8 vs the bootstrap's
167.6). The cause is visible in the data: score variance between boards
(p10 141 to p90 258 for one fixed policy) is much larger than the variance
between policies, so taking the global top 15% of games mostly selects easy
boards and trains the policy on whatever it happened to do there.

Fixing that means selecting per board: `--group K` plays each board K times with
different agent randomness (`agent_seed` is separate from the board seed), and
elites are ranked by their advantage over their own board's mean. Kept runs beat
their own board's average by ~110 points, and the rounds started improving
again. Both models are kept in `models/` so the comparison is reproducible.

Everything above is regenerated by the pipeline; `src/make_report.py` collates
it into `reports/RESULTS.md`.

## Layout

```
src/fetch_data.py          stage 1  fetch from npm registry
src/clean_data.py          stage 2  normalize + validate rule tables
src/env/dump_gamedata.ts   stage 2  dump the game's own data modules as JSON
src/env/bridge.ts          seeded JSONL game server
src/env/flatten.ts         grouped legal actions -> atomic action list
src/env/game_env.py        gym-like Python wrapper
src/env/features.py        state/action feature encoders
src/agents.py              random / scripted / heuristic / model agents
src/rollout.py             episode driver
src/generate_episodes.py   stage 3  search + record training data
src/model.py               stage 4  candidate-scoring policy + value network
src/train.py               stage 4  training loop
src/evaluate.py            stage 5  paired evaluation
src/selfplay_loop.py       stage 6  iterated self-imitation
src/make_report.py         collate everything into reports/RESULTS.md
tests/test_pipeline.py     encoder shapes, seed reproducibility, baseline ordering
```
