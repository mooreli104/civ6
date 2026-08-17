"""Stage 6 - SELF-IMPROVEMENT LOOP.

One round =
    generate games with the current best policy (plus exploration noise)
      -> keep the elite games
      -> retrain (warm-started from the previous round)
      -> evaluate against the baselines on held-out seeds

which is the cross-entropy / self-imitation recipe: the policy is always trained
on the best games it can currently produce, so the bar rises each round.

    python src/selfplay_loop.py --rounds 2 --games 400
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
REPORTS = ROOT / "reports"


def sh(args: list[str]) -> None:
    print("\n$ " + " ".join(str(a) for a in args), flush=True)
    proc = subprocess.run([str(a) for a in args], cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"step failed: {' '.join(str(a) for a in args)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Iterated self-imitation training loop.")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--elite-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--base-model", default="models/policy_bootstrap.keras")
    ap.add_argument("--base-tag", default="bootstrap")
    ap.add_argument("--temps", nargs="+", default=["0.3", "0.6"],
                    help="sampling temperatures used when generating self-play games")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    model = args.base_model
    tags = [args.base_tag]
    history = []

    for rnd in range(1, args.rounds + 1):
        tag = f"selfplay{rnd}"
        specs = [f"model:{model}:{t}:0.05" for t in args.temps]
        sh([PY, "src/generate_episodes.py", "--tag", tag, "--games", args.games,
            "--elite-frac", args.elite_frac, "--seed0", 100_000 * rnd,
            "--workers", args.workers, "--specs", *specs])

        tags.append(tag)
        name = f"policy_round{rnd}"
        # Train on the newest elites plus the bootstrap set, warm-starting from
        # the previous round so earlier knowledge is not thrown away.
        sh([PY, "src/train.py", "--tag", *tags[-2:], "--name", name,
            "--epochs", args.epochs, "--init-from", model])
        model = f"models/{name}.keras"

        sh([PY, "src/evaluate.py", "--model", model, "--games", args.eval_games,
            "--workers", args.workers, "--name", name])
        summary = REPORTS / f"evaluation_{name}.csv"
        history.append({"round": rnd, "tag": tag, "model": model, "evaluation": str(summary)})

    (REPORTS / "selfplay_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"\n[loop] finished {args.rounds} round(s); final model: {model}")


if __name__ == "__main__":
    main()
