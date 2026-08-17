"""Fast checks for the pieces the training run depends on.

    .venv/bin/python tests/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import make_agent  # noqa: E402
from env.features import FeatureEncoder  # noqa: E402
from env.game_env import CivEnv  # noqa: E402
from rollout import play_episode  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def main() -> None:
    # --- cleaned data is present and self-consistent ---
    processed = ROOT / "data" / "processed"
    vocab = json.loads((processed / "vocab.json").read_text())
    check("vocab has every entity family",
          all(k in vocab for k in ("terrain", "feature", "resource", "district",
                                   "improvement", "tech_node", "faction")))
    check("tech tree has 90 nodes", len(vocab["tech_node"]) == 90, str(len(vocab["tech_node"])))

    enc = FeatureEncoder()
    check("encoder dims are stable", enc.state_dim > 0 and enc.action_dim > 0,
          f"state {enc.state_dim}, action {enc.action_dim}")

    with CivEnv() as env:
        obs = env.reset(seed=1234)
        s = enc.encode_state(obs.state)
        a = enc.encode_actions(obs.state, obs.actions)
        check("state vector is finite and correctly shaped",
              s.shape == (enc.state_dim,) and bool(np.isfinite(s).all()))
        check("action matrix is finite and correctly shaped",
              a.shape == (len(obs.actions), enc.action_dim) and bool(np.isfinite(a).all()))
        check("legal actions are executable",
              all("type" in x and "params" in x for x in obs.actions))

        # --- same seed => same game ---
        agent = make_agent("heuristic")
        r1 = play_episode(env, agent, seed=555)
        r2 = play_episode(env, agent, seed=555)
        r3 = play_episode(env, agent, seed=556)
        check("episodes are reproducible from the seed", r1.score == r2.score,
              f"{r1.score} vs {r2.score}")
        check("different seeds give different games", r1.score != r3.score or r1.turns != r3.turns,
              f"seed 555 -> {r1.score}, seed 556 -> {r3.score}")
        check("games run the full 40 turns", r1.turns == 40, str(r1.turns))

        # --- baselines are ordered as expected ---
        rnd = play_episode(env, make_agent("random"), seed=777)
        heur = play_episode(env, make_agent("heuristic"), seed=777)
        check("heuristic beats random on the same board", heur.score > rnd.score,
              f"heuristic {heur.score} vs random {rnd.score}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
