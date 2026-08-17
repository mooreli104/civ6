"""Stage 2 - CLEAN.

Turns the fetched game's TypeScript rule tables into flat, validated datasets:

    data/processed/terrains.csv            14 terrain types x 6 yields
    data/processed/features.csv            terrain features (hills, forest, oasis, ...)
    data/processed/resources.csv           bonus/luxury/strategic/knowledge resources
    data/processed/districts.csv           districts + build cost + upgrade path
    data/processed/district_adjacency.csv  one row per adjacency rule
    data/processed/improvements.csv        improvements (farm, mine, ...)
    data/processed/improvement_adjacency.csv
    data/processed/tech_nodes.csv          3 trees x 4 layers of tech/policy/faith nodes
    data/processed/cards.csv               shop card pool with tier + faction weights
    data/processed/vocab.json              stable id -> index maps for the encoder
    data/processed/game_config.json        default rules config + victory goals
    reports/data_summary.md                human-readable summary + validation results

The dump itself comes from `src/env/dump_gamedata.ts`, which imports the real
modules, so nothing here is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
YIELD_KEYS = ["gold", "food", "production", "science", "culture", "faith"]


def dump_gamedata() -> dict:
    """Run the TS dumper via tsx and parse its JSON."""
    tsx = ROOT / "node_modules" / ".bin" / "tsx"
    if not tsx.exists():
        raise SystemExit("node_modules/.bin/tsx missing - run `npm install` first")
    if not (ROOT / "vendor" / "civ6-roguelike-ai").exists():
        raise SystemExit("vendor/civ6-roguelike-ai missing - run `python src/fetch_data.py` first")
    proc = subprocess.run(
        [str(tsx), str(ROOT / "src" / "env" / "dump_gamedata.ts")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit("game data dump failed")
    return json.loads(proc.stdout)


def _yields(prefix: str, y: dict | None) -> dict:
    y = y or {}
    return {f"{prefix}_{k}": float(y.get(k, 0) or 0) for k in YIELD_KEYS}


def _tags(tags: list[str] | None) -> str:
    return "|".join(sorted(tags or []))


def build_tables(gd: dict) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    tables["terrains"] = pd.DataFrame([
        {"id": t["id"], "name": t["name"], "buildable": bool(t["buildable"]),
         "tags": _tags(t.get("tags")), **_yields("yield", t.get("baseYields"))}
        for t in gd["terrains"]
    ])

    tables["features"] = pd.DataFrame([
        {"id": f["id"], "name": f["name"], "removable": bool(f.get("removable")),
         "tags": _tags(f.get("tags")), **_yields("mod", f.get("yieldModifier"))}
        for f in gd["features"]
    ])

    tables["resources"] = pd.DataFrame([
        {"id": r["id"], "name": r["name"], "category": r.get("category"),
         "tags": _tags(r.get("tags")), **_yields("bonus", r.get("yieldBonus"))}
        for r in gd["resources"]
    ])

    base_d = set(gd["district_availability"]["base"])
    tables["districts"] = pd.DataFrame([
        {"id": d["id"], "name": d["name"], "production_cost": d["productionCost"],
         "max_level": d["maxLevel"], "upgrade_primary_yield": d["upgradePrimaryYield"],
         "available_from_start": d["id"] in base_d,
         "requires_tags": _tags(d.get("placementRequireTags")),
         "tags": _tags(d.get("tags")), "n_adjacency_rules": len(d.get("adjacencyRules", [])),
         **_yields("yield", d.get("baseYields"))}
        for d in gd["districts"]
    ])
    tables["district_adjacency"] = pd.DataFrame([
        {"district_id": d["id"], "match_tag": rule["matchTag"], "mode": rule["mode"],
         "description": rule.get("description", ""), **_yields("bonus", rule.get("bonus"))}
        for d in gd["districts"] for rule in d.get("adjacencyRules", [])
    ])

    base_i = set(gd["improvement_availability"]["base"])
    tables["improvements"] = pd.DataFrame([
        {"id": i["id"], "name": i["name"], "production_cost": i["productionCost"],
         "max_level": i["maxLevel"], "upgrade_primary_yield": i["upgradePrimaryYield"],
         "available_from_start": i["id"] in base_i,
         "requires_tags": _tags(i.get("placementRequireTags")),
         "tags": _tags(i.get("tags")), "n_adjacency_rules": len(i.get("adjacencyRules", [])),
         **_yields("yield", i.get("yields"))}
        for i in gd["improvements"]
    ])
    tables["improvement_adjacency"] = pd.DataFrame([
        {"improvement_id": i["id"], "match_tag": rule["matchTag"], "mode": rule["mode"],
         "description": rule.get("description", ""), **_yields("bonus", rule.get("bonus"))}
        for i in gd["improvements"] for rule in i.get("adjacencyRules", [])
    ])

    tech_rows = []
    for tree in gd["tech_trees"]:
        for node in tree["nodes"]:
            boost = node.get("factionBoost") or {}
            effects = node.get("effects", [])
            tech_rows.append({
                "id": node["id"], "tree": tree["id"], "tree_resource": tree["resource"],
                "layer": node["layer"], "parent_id": node.get("parentId"),
                "name": node["name"], "description": node.get("description", ""),
                "n_effects": len(effects),
                "effect_types": "|".join(sorted({e["type"] for e in effects})),
                "unlocks": "|".join(sorted(str(e["unlockId"]) for e in effects if e.get("unlockId"))),
                "faction_boost": boost.get("faction"), "faction_boost_amount": boost.get("amount", 0),
                "layer_threshold": tree["thresholds"][node["layer"] - 1],
            })
    tables["tech_nodes"] = pd.DataFrame(tech_rows)

    tables["cards"] = pd.DataFrame([
        {"name": c["name"], "tier": c["tier"], "cost": c["cost"], "weight": c["weight"],
         "terrain_id": c["terrainId"], "feature_id": c.get("featureId"),
         "resource_id": c.get("resourceId"), "faction": c["faction"],
         "description": c.get("description", "")}
        for c in gd["cards"]
    ])

    tables["eurekas"] = pd.DataFrame([
        {"id": e["id"], "description": e["description"], "target_tree": e["targetTree"],
         "target_layer": e["targetLayer"], "reduction_percent": e["reductionPercent"]}
        for e in gd["eurekas"]
    ])
    return tables


def validate(tables: dict[str, pd.DataFrame], gd: dict) -> list[str]:
    """Referential-integrity + sanity checks. Returns a list of problem strings."""
    problems: list[str] = []
    terrain_ids = set(tables["terrains"]["id"])
    feature_ids = set(tables["features"]["id"])
    resource_ids = set(tables["resources"]["id"])
    district_ids = set(tables["districts"]["id"])
    improvement_ids = set(tables["improvements"]["id"])
    tech_ids = set(tables["tech_nodes"]["id"])

    for name, df, key in [("terrains", tables["terrains"], "id"), ("districts", tables["districts"], "id"),
                          ("improvements", tables["improvements"], "id"), ("tech_nodes", tables["tech_nodes"], "id")]:
        dupes = df[df.duplicated(key)][key].tolist()
        if dupes:
            problems.append(f"{name}: duplicate ids {dupes}")

    cards = tables["cards"]
    for col, valid in [("terrain_id", terrain_ids), ("feature_id", feature_ids), ("resource_id", resource_ids)]:
        bad = sorted({v for v in cards[col].dropna() if v not in valid})
        if bad:
            problems.append(f"cards.{col} references unknown ids: {bad}")

    tech = tables["tech_nodes"]
    bad_parents = sorted({p for p in tech["parent_id"].dropna() if p not in tech_ids})
    if bad_parents:
        problems.append(f"tech_nodes.parent_id references unknown nodes: {bad_parents}")
    unlockable = district_ids | improvement_ids
    bad_unlocks = sorted({u for row in tech["unlocks"] if row for u in row.split("|") if u not in unlockable})
    if bad_unlocks:
        problems.append(f"tech unlock targets not found in districts/improvements: {bad_unlocks}")

    if not (tech["layer"].between(1, 4).all()):
        problems.append("tech_nodes.layer outside 1..4")
    for name in ("districts", "improvements"):
        neg = tables[name][tables[name]["production_cost"] <= 0]
        if len(neg):
            problems.append(f"{name}: non-positive production_cost for {neg['id'].tolist()}")
    if cards["weight"].le(0).any():
        problems.append("cards: non-positive shop weight")
    return problems


def build_vocab(tables: dict[str, pd.DataFrame], gd: dict) -> dict:
    """Stable id -> index maps. The feature encoder depends on this ordering."""
    def idx(values) -> dict[str, int]:
        return {v: i for i, v in enumerate(sorted(values))}

    all_tags = set()
    for name in ("terrains", "features", "resources", "districts", "improvements"):
        for row in tables[name]["tags"]:
            all_tags.update(t for t in row.split("|") if t)

    return {
        "yield_keys": YIELD_KEYS,
        "terrain": idx(tables["terrains"]["id"]),
        "feature": idx(tables["features"]["id"]),
        "resource": idx(tables["resources"]["id"]),
        "district": idx(tables["districts"]["id"]),
        "improvement": idx(tables["improvements"]["id"]),
        "tech_node": idx(tables["tech_nodes"]["id"]),
        "tech_tree": idx(t["id"] for t in gd["tech_trees"]),
        "faction": idx(gd["factions"]),
        "tag": idx(all_tags),
    }


def write_summary(tables: dict[str, pd.DataFrame], gd: dict, problems: list[str]) -> None:
    manifest_path = ROOT / "data" / "raw" / "fetch_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    lines = [
        "# Civ 6 game data - cleaned dataset summary", "",
        f"Source: `{manifest.get('package', '?')}@{manifest.get('version', '?')}` "
        f"from {manifest.get('registry', '?')} (fetched {manifest.get('fetched_at', '?')})", "",
        "| table | rows | columns |", "|---|---:|---:|",
    ]
    for name, df in tables.items():
        lines.append(f"| `{name}.csv` | {len(df)} | {df.shape[1]} |")

    t = tables["terrains"]
    lines += [
        "", "## Highlights", "",
        f"- {len(t)} terrain types; mean total base yield "
        f"{t[[f'yield_{k}' for k in YIELD_KEYS]].sum(axis=1).mean():.2f}",
        f"- {len(tables['districts'])} districts with {len(tables['district_adjacency'])} adjacency rules",
        f"- {len(tables['improvements'])} improvements with {len(tables['improvement_adjacency'])} adjacency rules",
        f"- {len(tables['tech_nodes'])} tech nodes across "
        f"{tables['tech_nodes']['tree'].nunique()} trees, layers "
        f"{sorted(tables['tech_nodes']['layer'].unique().tolist())}",
        f"- {len(tables['cards'])} shop cards over tiers "
        f"{sorted(tables['cards']['tier'].unique().tolist())}, "
        f"cost range {tables['cards']['cost'].min()}-{tables['cards']['cost'].max()}",
        "", "## Validation", "",
    ]
    lines += [f"- FAIL: {p}" for p in problems] or ["- All referential-integrity checks passed."]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize fetched Civ 6 game data into flat tables.")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if validation finds problems")
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    gd = dump_gamedata()
    tables = build_tables(gd)

    problems = validate(tables, gd)
    for name, df in tables.items():
        out = PROCESSED / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"[clean] {out.relative_to(ROOT)}: {len(df)} rows x {df.shape[1]} cols")

    (PROCESSED / "vocab.json").write_text(json.dumps(build_vocab(tables, gd), indent=2, sort_keys=True) + "\n")
    (PROCESSED / "game_config.json").write_text(json.dumps(
        {"default_config": gd["default_config"], "victory_goals": gd["victory_goals"],
         "shop_levels": gd["shop_levels"], "factions": gd["factions"],
         "tech_thresholds": {t["id"]: t["thresholds"] for t in gd["tech_trees"]}},
        indent=2) + "\n")
    write_summary(tables, gd, problems)

    if problems:
        print("[clean] validation problems:")
        for p in problems:
            print(f"  - {p}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("[clean] validation: all checks passed")
    print(f"[clean] wrote {PROCESSED.relative_to(ROOT)}/vocab.json, game_config.json and reports/data_summary.md")


if __name__ == "__main__":
    main()
