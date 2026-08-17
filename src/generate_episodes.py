"""Stage 3 - GENERATE + CLEAN gameplay data.

Two passes, which keeps the on-disk footprint small:

  1. search  - play a lot of games with noisy agents, storing only the outcome of
               each (seed, agent spec, final score, ...). Cheap: a few hundred
               bytes per game.
  2. record  - keep the elite fraction (best games by final score), replay them
               *exactly* - engine and agent are both seeded, so a replay is
               bit-identical - and this time encode every decision into
               training tensors.

Output:
    data/episodes/<tag>_index.csv        one row per played game (all games)
    data/episodes/<tag>_elite.csv        the games selected for training
    data/episodes/<tag>_shard###.npz     padded (state, candidates, choice) tensors
    reports/generation_<tag>.md          score distribution + elite cut-off
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

from agents import make_agent  # noqa: E402
from env.features import FeatureEncoder  # noqa: E402
from env.game_env import CivEnv  # noqa: E402
from rollout import play_episode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "data" / "episodes"
REPORTS = ROOT / "reports"

MAX_CANDIDATES = 48  # padded action slots per decision

_WORKER: dict = {}


def _build_agent(spec: str):
    """Agent specs are strings so they survive the trip to worker processes."""
    if spec.startswith("model:"):
        _, model_path, temp, eps = spec.split(":")
        from agents import ModelAgent  # imported lazily; TF only loads in model runs
        from model import load_policy

        model = _WORKER.get("model")
        if model is None or _WORKER.get("model_path") != model_path:
            model = load_policy(model_path)
            _WORKER["model"] = model
            _WORKER["model_path"] = model_path
        return ModelAgent(model, _WORKER["encoder"], temperature=float(temp), epsilon=float(eps))
    return make_agent(spec)


def _init_worker() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    _WORKER["env"] = CivEnv(max_actions=192)
    _WORKER["encoder"] = FeatureEncoder()


def _search_one(job: tuple[int, int, str]) -> dict:
    seed, agent_seed, spec = job
    agent = _build_agent(spec)
    result = play_episode(_WORKER["env"], agent, seed=seed, agent_seed=agent_seed)
    row = asdict(result)
    row.pop("records")
    row["spec"] = spec
    return row


def _record_one(job: tuple[int, int, str]) -> dict:
    """Replay one elite game and encode every decision."""
    seed, agent_seed, spec = job
    enc: FeatureEncoder = _WORKER["encoder"]
    agent = _build_agent(spec)
    rng = np.random.default_rng(agent_seed)

    states: list[np.ndarray] = []
    cands: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    choices: list[int] = []
    turns: list[int] = []

    def on_decision(state: dict, actions, choice: int) -> None:
        if len(actions) < 2:
            return  # nothing was actually decided
        idxs = list(range(len(actions)))
        if len(idxs) > MAX_CANDIDATES:
            others = [i for i in idxs if i != choice]
            keep = rng.choice(len(others), size=MAX_CANDIDATES - 1, replace=False)
            idxs = [choice] + [others[k] for k in keep]
            rng.shuffle(idxs)
        subset = [actions[i] for i in idxs]
        mat = enc.encode_actions(state, subset)
        pad = np.zeros((MAX_CANDIDATES, enc.action_dim), dtype=np.float32)
        pad[: len(subset)] = mat
        mask = np.zeros(MAX_CANDIDATES, dtype=np.float32)
        mask[: len(subset)] = 1.0
        states.append(enc.encode_state(state))
        cands.append(pad)
        masks.append(mask)
        choices.append(idxs.index(choice))
        turns.append(int(state.get("turn", 0)))

    result = play_episode(_WORKER["env"], agent, seed=seed, agent_seed=agent_seed,
                          on_decision=on_decision)
    return {
        "seed": seed, "spec": spec, "score": result.score, "turns": result.turns,
        "states": np.asarray(states, dtype=np.float16),
        "candidates": np.asarray(cands, dtype=np.float16),
        "masks": np.asarray(masks, dtype=np.float16),
        "choices": np.asarray(choices, dtype=np.int16),
        "turn_ids": np.asarray(turns, dtype=np.int16),
    }


def run(tag: str, specs: list[str], n_games: int, elite_frac: float, seed0: int,
        workers: int, shard_size: int, group: int = 1) -> pd.DataFrame:
    EPISODES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # With group > 1 the same board is played several times with different agent
    # randomness. Scores can then be compared within a board, so elite selection
    # picks better *play* rather than easier boards - the variance between seeds
    # is larger than the variance between policies, and unpaired selection just
    # collects lucky maps.
    n_boards = max(n_games // group, 1)
    jobs = []
    for b in range(n_boards):
        for k in range(group):
            i = b * group + k
            jobs.append((seed0 + b, seed0 + 7_919 * (k + 1) + b, specs[i % len(specs)]))
    n_games = len(jobs)
    print(f"[gen:{tag}] search pass: {n_games} games ({n_boards} boards x {group} runs) "
          f"over {len(specs)} agent specs on {workers} workers")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(_search_one, jobs, chunksize=4), 1):
            rows.append(row)
            if i % max(n_games // 20, 1) == 0:
                best = max(r["score"] for r in rows)
                print(f"  {i}/{n_games} games  best={best:.0f}  "
                      f"mean={np.mean([r['score'] for r in rows]):.0f}  ({time.time() - t0:.0f}s)")
    index = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    index.to_csv(EPISODES / f"{tag}_index.csv", index=False)

    if group > 1:
        # keep the runs that beat their own board's average, best first
        index["board_mean"] = index.groupby("seed")["score"].transform("mean")
        index["advantage"] = index["score"] - index["board_mean"]
        n_elite = max(int(len(index) * elite_frac), 1)
        elite = index.sort_values("advantage", ascending=False).head(n_elite)
        print(f"[gen:{tag}] paired selection: mean advantage of kept runs "
              f"{elite['advantage'].mean():.1f} over their own boards")
    else:
        n_elite = max(int(len(index) * elite_frac), 1)
        elite = index.head(n_elite)
    elite.to_csv(EPISODES / f"{tag}_elite.csv", index=False)
    cutoff = float(elite["score"].min())
    print(f"[gen:{tag}] search done in {time.time() - t0:.0f}s | "
          f"mean={index['score'].mean():.1f} p90={index['score'].quantile(0.9):.1f} "
          f"max={index['score'].max():.1f} | elite cut-off {cutoff:.1f} ({n_elite} games)")

    print(f"[gen:{tag}] record pass: replaying {n_elite} elite games")
    t1 = time.time()
    shard, shard_id, n_samples = [], 0, 0
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        elite_jobs = [(int(r.seed), int(r.agent_seed), str(r.spec)) for r in elite.itertuples()]
        for i, out in enumerate(pool.imap_unordered(_record_one, elite_jobs, chunksize=2), 1):
            if len(out["states"]) == 0:
                continue
            shard.append(out)
            n_samples += len(out["states"])
            if sum(len(s["states"]) for s in shard) >= shard_size:
                _write_shard(tag, shard_id, shard)
                shard, shard_id = [], shard_id + 1
            if i % max(n_elite // 10, 1) == 0:
                print(f"  replayed {i}/{n_elite} ({n_samples} decisions, {time.time() - t1:.0f}s)")
    if shard:
        _write_shard(tag, shard_id, shard)

    _write_report(tag, index, elite, n_samples)
    print(f"[gen:{tag}] recorded {n_samples} decisions in {shard_id + 1} shard(s) "
          f"({time.time() - t1:.0f}s)")
    return index


def _write_shard(tag: str, shard_id: int, parts: list[dict]) -> None:
    path = EPISODES / f"{tag}_shard{shard_id:03d}.npz"
    np.savez_compressed(
        path,
        states=np.concatenate([p["states"] for p in parts]),
        candidates=np.concatenate([p["candidates"] for p in parts]),
        masks=np.concatenate([p["masks"] for p in parts]),
        choices=np.concatenate([p["choices"] for p in parts]),
        turn_ids=np.concatenate([p["turn_ids"] for p in parts]),
        returns=np.concatenate([np.full(len(p["states"]), p["score"], dtype=np.float32) for p in parts]),
        episode_ids=np.concatenate([np.full(len(p["states"]), p["seed"], dtype=np.int32) for p in parts]),
    )
    size_mb = path.stat().st_size / 1e6
    print(f"  wrote {path.name}: {sum(len(p['states']) for p in parts)} decisions, {size_mb:.1f} MB")


def _write_report(tag: str, index: pd.DataFrame, elite: pd.DataFrame, n_samples: int) -> None:
    by_spec = index.groupby("spec")["score"].agg(["count", "mean", "std", "max"]).round(1)
    lines = [
        f"# Episode generation - `{tag}`", "",
        f"- games played: **{len(index)}**",
        f"- score: mean **{index['score'].mean():.1f}**, median {index['score'].median():.1f}, "
        f"p90 {index['score'].quantile(0.9):.1f}, max **{index['score'].max():.1f}**",
        f"- elite games kept: **{len(elite)}** (cut-off {elite['score'].min():.1f}, "
        f"mean {elite['score'].mean():.1f})",
        f"- training decisions recorded: **{n_samples}**",
        "", "## Score by agent spec", "", by_spec.to_markdown(), "",
    ]
    (REPORTS / f"generation_{tag}.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate and encode Civ 6 gameplay training data.")
    ap.add_argument("--tag", default="bootstrap")
    ap.add_argument("--specs", nargs="+", default=["heuristic:0.3", "heuristic:0.6", "heuristic:1.0"])
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--elite-frac", type=float, default=0.2)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 1, 1))
    ap.add_argument("--shard-size", type=int, default=40000)
    ap.add_argument("--group", type=int, default=1,
                    help="runs per board; >1 enables paired (board-normalized) elite selection")
    args = ap.parse_args()
    run(args.tag, args.specs, args.games, args.elite_frac, args.seed0, args.workers,
        args.shard_size, args.group)


if __name__ == "__main__":
    main()
