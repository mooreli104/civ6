"""Collate everything the pipeline produced into reports/RESULTS.md."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
EPISODES = ROOT / "data" / "episodes"
PROCESSED = ROOT / "data" / "processed"


def section(title: str) -> str:
    return f"\n## {title}\n"


def main() -> None:
    manifest = json.loads((ROOT / "data" / "raw" / "fetch_manifest.json").read_text())
    out = [
        "# Results", "",
        f"Source data: `{manifest['package']}@{manifest['version']}` from {manifest['registry']} "
        f"(sha256 `{manifest['tarball_sha256'][:16]}...`, fetched {manifest['fetched_at']}).", "",
        "Every number below is produced by the scripts in `src/`; rerunning the pipeline "
        "regenerates this file.",
    ]

    # ---- data ----
    out.append(section("1. Cleaned game data"))
    rows = []
    for csv in sorted(PROCESSED.glob("*.csv")):
        df = pd.read_csv(csv)
        rows.append({"table": csv.name, "rows": len(df), "columns": df.shape[1]})
    out.append(pd.DataFrame(rows).to_markdown(index=False))

    # ---- generation ----
    out.append(section("2. Generated gameplay data"))
    gen_rows = []
    for idx_csv in sorted(EPISODES.glob("*_index.csv")):
        tag = idx_csv.name.replace("_index.csv", "")
        idx = pd.read_csv(idx_csv)
        elite_csv = EPISODES / f"{tag}_elite.csv"
        elite = pd.read_csv(elite_csv) if elite_csv.exists() else idx
        gen_rows.append({
            "tag": tag, "games": len(idx), "mean score": round(idx["score"].mean(), 1),
            "p90": round(idx["score"].quantile(0.9), 1), "max": round(idx["score"].max(), 1),
            "elite games": len(elite), "elite mean": round(elite["score"].mean(), 1),
        })
    if gen_rows:
        out.append(pd.DataFrame(gen_rows).to_markdown(index=False))

    # ---- training ----
    out.append(section("3. Trained models"))
    train_rows = []
    for meta_path in sorted(MODELS.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text())
        last = meta["history"][-1] if meta.get("history") else {}
        train_rows.append({
            "model": meta["name"], "decisions": meta["n_decisions"], "elite games": meta["n_games"],
            "val top-1": round(last.get("val_top1", float("nan")), 3),
            "val top-3": round(last.get("val_top3", float("nan")), 3),
            "best val loss": round(meta.get("best_val_policy_loss", float("nan")), 4),
        })
    if train_rows:
        out.append(pd.DataFrame(train_rows).to_markdown(index=False))

    # ---- evaluation ----
    out.append(section("4. Evaluation (held-out seeds, paired)"))
    for csv in sorted(REPORTS.glob("evaluation_*.csv")):
        df = pd.read_csv(csv)
        name = csv.name.replace("evaluation_", "").replace(".csv", "")
        agg = df.groupby("spec")["score"].agg(games="count", mean="mean", median="median",
                                              p90=lambda s: s.quantile(0.9), best="max").round(1)
        agg["goal_rate"] = df.groupby("spec")["goal_achieved"].mean().round(3)
        pivot = df.pivot_table(index="seed", columns="spec", values="score")
        if "heuristic" in pivot.columns:
            agg["win_vs_heuristic"] = pd.Series({
                spec: round(float((pivot[[spec, "heuristic"]].dropna().iloc[:, 0]
                                   > pivot[[spec, "heuristic"]].dropna().iloc[:, 1]).mean()), 3)
                for spec in pivot.columns})
        out.append(f"\n### `{name}`\n")
        out.append(agg.sort_values("mean", ascending=False).to_markdown())

    (REPORTS / "RESULTS.md").write_text("\n".join(out) + "\n")
    print(f"wrote {(REPORTS / 'RESULTS.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
