"""Stage 5 - EVALUATE.

Plays the trained policy and the baselines over the *same* held-out seeds, so
the comparison is paired: every agent gets the identical board, shop rolls and
tech thresholds.

    python src/evaluate.py --model models/policy_bootstrap.keras --games 60

Outputs:
    reports/evaluation_<name>.md   per-agent table + paired win rate vs heuristic
    reports/evaluation_<name>.csv  raw per-game results
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

_WORKER: dict = {}


def _init_worker() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from env.features import FeatureEncoder
    from env.game_env import CivEnv
    _WORKER["env"] = CivEnv(max_actions=192)
    _WORKER["encoder"] = FeatureEncoder()


def _agent_for(spec: str):
    from agents import make_agent
    if spec.startswith("model:"):
        from agents import ModelAgent
        from model import load_policy
        _, path, temp = spec.split(":", 2)
        if _WORKER.get("model_path") != path:
            _WORKER["model"] = load_policy(path)
            _WORKER["model_path"] = path
        return ModelAgent(_WORKER["model"], _WORKER["encoder"], temperature=float(temp))
    return make_agent(spec)


def _play(job: tuple[int, str]) -> dict:
    from rollout import play_episode
    seed, spec = job
    agent = _agent_for(spec)
    r = play_episode(_WORKER["env"], agent, seed=seed)
    row = asdict(r)
    row.pop("records")
    row["spec"] = spec
    return row


def evaluate(specs: list[str], games: int, seed0: int, workers: int) -> pd.DataFrame:
    jobs = [(seed0 + i, spec) for spec in specs for i in range(games)]
    # interleave so progress reflects all agents, not one agent at a time
    jobs.sort(key=lambda j: (j[0], j[1]))
    ctx = mp.get_context("spawn")
    t0 = time.time()
    print(f"[eval] {len(jobs)} games ({games} per agent x {len(specs)} agents) on {workers} workers",
          flush=True)
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(_play, jobs, chunksize=2), 1):
            rows.append(row)
            if i % max(len(jobs) // 10, 1) == 0:
                print(f"  {i}/{len(jobs)} games ({time.time() - t0:.0f}s)", flush=True)
    print(f"[eval] {len(jobs)} games in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    agg = df.groupby("spec")["score"].agg(
        games="count", mean="mean", std="std", median="median", p10=lambda s: s.quantile(0.1),
        p90=lambda s: s.quantile(0.9), best="max",
    ).round(1)
    agg["goal_rate"] = df.groupby("spec")["goal_achieved"].mean().round(3)
    agg["mean_pop"] = df.groupby("spec")["population"].mean().round(1)
    agg["mean_tech_nodes"] = df.groupby("spec")["tech_nodes"].mean().round(1)

    # paired comparison on identical seeds
    pivot = df.pivot_table(index="seed", columns="spec", values="score")
    if reference in pivot.columns:
        wins, lifts = {}, {}
        ref = pivot[reference]
        for spec in pivot.columns:
            pair = pd.concat([pivot[spec], ref], axis=1, keys=["a", "b"]).dropna()
            wins[spec] = round(float((pair["a"] > pair["b"]).mean()), 3)
            lifts[spec] = round(float((pair["a"] - pair["b"]).mean()), 1)
        agg[f"win_vs_{reference}"] = pd.Series(wins)
        agg[f"lift_vs_{reference}"] = pd.Series(lifts)
    return agg.sort_values("mean", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the trained policy against baselines.")
    ap.add_argument("--model", default=None, help="path to a .keras policy")
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0],
                    help="sampling temperatures to evaluate the model at")
    ap.add_argument("--baselines", nargs="+", default=["random", "scripted", "heuristic"])
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=900_000, help="held-out seed block")
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 1, 1))
    ap.add_argument("--reference", default="heuristic")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    specs = list(args.baselines)
    if args.model:
        specs += [f"model:{args.model}:{t}" for t in args.temps]
    name = args.name or (Path(args.model).stem if args.model else "baselines")

    df = evaluate(specs, args.games, args.seed0, args.workers)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / f"evaluation_{name}.csv", index=False)
    summary = summarize(df, args.reference)
    print(summary.to_string())

    lines = [
        f"# Evaluation - `{name}`", "",
        f"{args.games} games per agent on held-out seeds "
        f"{args.seed0}..{args.seed0 + args.games - 1} (every agent sees the identical boards).",
        "", summary.to_markdown(), "",
        "Score is the game's own final score: accumulated science + culture + faith after 40 turns.",
    ]
    (REPORTS / f"evaluation_{name}.md").write_text("\n".join(lines) + "\n")
    print(f"[eval] wrote reports/evaluation_{name}.md")


if __name__ == "__main__":
    main()
