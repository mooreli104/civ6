"""Feature encoders: game state -> fixed vector, legal actions -> matrix.

Everything is driven by `data/processed/vocab.json`, so the encoding stays in
sync with whatever version of the game data was fetched and cleaned. Two public
entry points:

    enc = FeatureEncoder()
    s = enc.encode_state(state)                 # (state_dim,)  float32
    a = enc.encode_actions(state, actions)      # (n_actions, action_dim) float32
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

YIELDS = ["gold", "food", "production", "science", "culture", "faith"]
PHASES = ["shopping", "placing", "game_over"]
ACTION_TYPES = [
    "select_card", "place_card", "deselect_card", "reroll_shop",
    "build_improvement", "build_district", "upgrade_tile",
    "assign_worker", "unassign_worker", "reassign_all_workers",
    "reassign_workers_food_priority", "unlock_hex", "sell_tile",
    "select_tech_node", "end_turn",
]
# tags that drive district adjacency scoring in the fetched rules
ADJ_TAGS = ["mountain", "water", "vegetation", "district", "hills", "fertile", "arid", "cold"]


def _log(x: float) -> float:
    return math.log1p(max(float(x), 0.0))


def _onehot(index: int | None, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= index < size:
        v[index] = 1.0
    return v


class FeatureEncoder:
    def __init__(self, processed_dir: Path = PROCESSED) -> None:
        self.vocab = json.loads((processed_dir / "vocab.json").read_text())
        self.config = json.loads((processed_dir / "game_config.json").read_text())

        self.terrain_ix: dict[str, int] = self.vocab["terrain"]
        self.feature_ix: dict[str, int] = self.vocab["feature"]
        self.resource_ix: dict[str, int] = self.vocab["resource"]
        self.district_ix: dict[str, int] = self.vocab["district"]
        self.improvement_ix: dict[str, int] = self.vocab["improvement"]
        self.tech_ix: dict[str, int] = self.vocab["tech_node"]
        self.tree_ix: dict[str, int] = self.vocab["tech_tree"]
        self.faction_ix: dict[str, int] = self.vocab["faction"]

        # static rule tables, used to describe action targets
        terr = pd.read_csv(processed_dir / "terrains.csv").set_index("id")
        self.terrain_tags = {tid: set(str(t).split("|")) for tid, t in terr["tags"].items()}
        self.terrain_yields = {
            tid: np.array([row[f"yield_{k}"] for k in YIELDS], dtype=np.float32)
            for tid, row in terr.iterrows()
        }
        dist = pd.read_csv(processed_dir / "districts.csv").set_index("id")
        self.district_stats = {
            did: np.array([row["production_cost"] / 20.0, row["max_level"] / 10.0,
                           row["n_adjacency_rules"] / 3.0, float(row["available_from_start"])]
                          + [row[f"yield_{k}"] for k in YIELDS], dtype=np.float32)
            for did, row in dist.iterrows()
        }
        imp = pd.read_csv(processed_dir / "improvements.csv").set_index("id")
        self.improvement_stats = {
            iid: np.array([row["production_cost"] / 20.0, row["max_level"] / 10.0,
                           row["n_adjacency_rules"] / 3.0, float(row["available_from_start"])]
                          + [row[f"yield_{k}"] for k in YIELDS], dtype=np.float32)
            for iid, row in imp.iterrows()
        }
        tech = pd.read_csv(processed_dir / "tech_nodes.csv").set_index("id")
        self.tech_stats = {}
        for nid, row in tech.iterrows():
            fb = _onehot(self.faction_ix.get(row["faction_boost"]), len(self.faction_ix))
            effects = str(row["effect_types"]) if pd.notna(row["effect_types"]) else ""
            self.tech_stats[nid] = np.concatenate([
                _onehot(self.tree_ix.get(row["tree"]), len(self.tree_ix)),
                np.array([row["layer"] / 4.0, row["n_effects"] / 3.0,
                          row["faction_boost_amount"] / 30.0,
                          float("unlock_district" in effects), float("unlock_improvement" in effects),
                          float("buff_improvement" in effects or "buff_terrain" in effects),
                          float("reduce_cost" in effects), float("shop_level" in effects),
                          _log(row["layer_threshold"])], dtype=np.float32),
                fb,
            ]).astype(np.float32)
        self.tech_stat_dim = len(next(iter(self.tech_stats.values())))

        self.state_dim = len(self.encode_state(_EMPTY_STATE))
        self.action_dim = len(self._encode_action(_EMPTY_STATE, _EMPTY_ACTION, _empty_board_index()))

    # ------------------------------------------------------------------ state

    def encode_state(self, state: dict) -> np.ndarray:
        res = state.get("resources", {})
        acc = state.get("accumulated", {})
        ypt = state.get("perTurnYields", {})
        shop = state.get("shop", {})
        goal = state.get("goal", {})
        score = state.get("score", {})
        tech = state.get("techState", {})
        board = state.get("board", [])
        turn = state.get("turn", 1)
        max_turns = max(state.get("maxTurns", 40), 1)

        parts: list[np.ndarray] = []
        parts.append(np.array([
            turn / max_turns,
            (max_turns - turn) / max_turns,
            _log(res.get("gold", 0)),
            _log(res.get("production", 0)),
            _log(res.get("foodProgress", 0)),
            res.get("netFoodPerTurn", 0) / 10.0,
            _log(state.get("population", 0)),
            _log(state.get("growthThreshold", 0)),
            _log(state.get("availableWorkers", 0)),
        ], dtype=np.float32))

        parts.append(np.array([_log(ypt.get(k, 0)) for k in YIELDS], dtype=np.float32))
        parts.append(np.array([_log(acc.get(k, 0)) for k in ("science", "culture", "faith")], dtype=np.float32))
        parts.append(np.array([_log(score.get(k, 0)) for k in ("science", "culture", "faith", "total")],
                              dtype=np.float32))
        parts.append(np.array([
            min(goal.get("ratio", 0.0), 2.0), float(goal.get("achieved", False)),
            _log(goal.get("target", 0)),
        ], dtype=np.float32))

        parts.append(np.array([
            shop.get("level", 1) / 4.0,
            len(shop.get("cards", [])) / 4.0,
            shop.get("rerollsUsed", 0) / max(shop.get("maxRerolls", 3), 1),
            _log(shop.get("rerollCost", 0)),
            float(shop.get("selectedCard") is not None),
        ], dtype=np.float32))
        # cheapest / mean shop card cost and tier mix
        cards = shop.get("cards", [])
        costs = [c.get("cost", 0) for c in cards] or [0]
        tiers = [c.get("tier", 1) for c in cards] or [1]
        parts.append(np.array([_log(min(costs)), _log(sum(costs) / len(costs)),
                               sum(tiers) / len(tiers) / 3.0], dtype=np.float32))

        parts.append(_onehot(PHASES.index(state["phase"]) if state.get("phase") in PHASES else None, len(PHASES)))

        fw = state.get("factionWeights", {})
        total_fw = sum(fw.values()) or 1.0
        parts.append(np.array([fw.get(f, 0) / total_fw for f in sorted(self.faction_ix)], dtype=np.float32))

        # tech progress: nodes taken per tree, plus progress toward the next layer
        selected = tech.get("selectedNodes", {}) or {}
        thresholds = tech.get("effectiveThresholds", {}) or {}
        tech_feats: list[float] = []
        for tree in sorted(self.tree_ix):
            taken = selected.get(tree, [])
            tech_feats.append(len(taken) / 10.0)
            resource = {"science": "science", "policy": "culture", "faith": "faith"}[tree]
            have = acc.get(resource, 0)
            thr = thresholds.get(tree, [0, 0, 0, 0])
            nxt = next((t for t in thr if t > have), thr[-1] if thr else 1)
            tech_feats.append(min(have / max(nxt, 1), 2.0))
            tech_feats.append(_log(nxt))
        tech_feats.append(len(tech.get("triggeredEurekas", [])) / 15.0)
        parts.append(np.array(tech_feats, dtype=np.float32))

        # board aggregates
        n_tiles = len(board) or 1
        unlocked = [t for t in board if t.get("unlocked")]
        placed = [t for t in board if t.get("terrainId")]
        worked = [t for t in placed if t.get("isWorked")]
        districts = [t for t in placed if t.get("districtId")]
        improvements = [t for t in placed if t.get("improvementId")]
        tile_yields = np.zeros(len(YIELDS), dtype=np.float32)
        for t in placed:
            y = t.get("yields", {})
            tile_yields += np.array([y.get(k, 0) for k in YIELDS], dtype=np.float32)
        parts.append(np.array([
            len(unlocked) / n_tiles, len(placed) / n_tiles, len(worked) / max(len(placed), 1),
            _log(len(districts)), _log(len(improvements)),
            _log(sum(t.get("districtLevel", 0) for t in districts)),
            _log(sum(t.get("improvementLevel", 0) for t in improvements)),
            len([t for t in unlocked if not t.get("terrainId")]) / n_tiles,
        ], dtype=np.float32))
        parts.append(np.log1p(tile_yields))

        # composition histograms: what has actually been built
        terrain_hist = np.zeros(len(self.terrain_ix), dtype=np.float32)
        district_hist = np.zeros(len(self.district_ix), dtype=np.float32)
        improvement_hist = np.zeros(len(self.improvement_ix), dtype=np.float32)
        for t in placed:
            ti = self.terrain_ix.get(t.get("terrainId"))
            if ti is not None:
                terrain_hist[ti] += 1
            di = self.district_ix.get(t.get("districtId"))
            if di is not None:
                district_hist[di] += 1
            ii = self.improvement_ix.get(t.get("improvementId"))
            if ii is not None:
                improvement_hist[ii] += 1
        parts += [np.log1p(terrain_hist), np.log1p(district_hist), np.log1p(improvement_hist)]

        return np.concatenate(parts).astype(np.float32)

    # ----------------------------------------------------------------- actions

    def encode_actions(self, state: dict, actions: list[dict]) -> np.ndarray:
        if not actions:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        board_index = {(t["q"], t["r"]): t for t in state.get("board", [])}
        return np.stack([self._encode_action(state, a, board_index) for a in actions]).astype(np.float32)

    def _encode_action(self, state: dict, action: dict, board_index: dict) -> np.ndarray:
        meta = action.get("meta", {}) or {}
        atype = action.get("type", "")
        res = state.get("resources", {})
        gold = float(res.get("gold", 0))
        prod = float(res.get("production", 0))

        gold_cost = float(meta.get("goldCost", 0) or 0)
        prod_cost = float(meta.get("productionCost", 0) or 0)
        parts: list[np.ndarray] = [
            _onehot(ACTION_TYPES.index(atype) if atype in ACTION_TYPES else None, len(ACTION_TYPES)),
            np.array([
                _log(gold_cost), _log(prod_cost),
                gold_cost / (gold + 1.0), prod_cost / (prod + 1.0),
            ], dtype=np.float32),
        ]

        # --- target tile ---
        tile = board_index.get((meta.get("q"), meta.get("r"))) if "q" in meta else None
        parts.append(self._tile_features(tile, board_index))

        # --- what is being built ---
        did = meta.get("districtId")
        parts.append(np.concatenate([
            _onehot(self.district_ix.get(did), len(self.district_ix)),
            self.district_stats.get(did, np.zeros(4 + len(YIELDS), dtype=np.float32)),
        ]).astype(np.float32))
        iid = meta.get("improvementId")
        parts.append(np.concatenate([
            _onehot(self.improvement_ix.get(iid), len(self.improvement_ix)),
            self.improvement_stats.get(iid, np.zeros(4 + len(YIELDS), dtype=np.float32)),
        ]).astype(np.float32))

        # --- tech node ---
        nid = meta.get("nodeId")
        parts.append(self.tech_stats.get(nid, np.zeros(self.tech_stat_dim, dtype=np.float32)))

        # --- shop card being selected ---
        parts.append(self._card_features(state, meta))

        # --- upgrade context ---
        parts.append(np.array([
            float(meta.get("currentLevel", 0) or 0) / 5.0,
            float(meta.get("maxLevel", 0) or 0) / 10.0,
            float(meta.get("remaining", 0) or 0) / 3.0,
        ], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def _tile_features(self, tile: dict | None, board_index: dict) -> np.ndarray:
        if tile is None:
            n = (len(self.terrain_ix) + len(self.feature_ix) + len(self.resource_ix)
                 + len(self.district_ix) + len(self.improvement_ix) + 8 + len(YIELDS) + len(ADJ_TAGS) + 1)
            return np.zeros(n, dtype=np.float32)

        y = tile.get("yields", {})
        parts = [
            _onehot(self.terrain_ix.get(tile.get("terrainId")), len(self.terrain_ix)),
            _onehot(self.feature_ix.get(tile.get("featureId")), len(self.feature_ix)),
            _onehot(self.resource_ix.get(tile.get("resourceId")), len(self.resource_ix)),
            _onehot(self.district_ix.get(tile.get("districtId")), len(self.district_ix)),
            _onehot(self.improvement_ix.get(tile.get("improvementId")), len(self.improvement_ix)),
            np.array([
                float(tile.get("unlocked", False)),
                float(tile.get("isWorked", False)),
                float(tile.get("terrainId") is not None),
                tile.get("ring", 0) / 4.0,
                tile.get("improvementLevel", 0) / 5.0,
                tile.get("districtLevel", 0) / 5.0,
                _log(tile.get("goldInvested", 0)),
                _log(tile.get("productionInvested", 0)),
            ], dtype=np.float32),
            np.array([y.get(k, 0) for k in YIELDS], dtype=np.float32),
        ]
        # neighbourhood: the adjacency tags that drive district bonuses
        counts = np.zeros(len(ADJ_TAGS), dtype=np.float32)
        n_neighbors = 0.0
        for nb in _neighbors(tile["q"], tile["r"]):
            ntile = board_index.get(nb)
            if not ntile or not ntile.get("terrainId"):
                continue
            n_neighbors += 1
            tags = set(self.terrain_tags.get(ntile["terrainId"], ()))
            if ntile.get("districtId"):
                tags.add("district")
            if ntile.get("featureId"):
                tags.add(ntile["featureId"])
            for i, tag in enumerate(ADJ_TAGS):
                if tag in tags:
                    counts[i] += 1
        parts.append(counts)
        parts.append(np.array([n_neighbors / 6.0], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def _card_features(self, state: dict, meta: dict) -> np.ndarray:
        size = len(self.terrain_ix) + len(self.feature_ix) + len(self.resource_ix) + 4
        idx = meta.get("cardIndex")
        cards = state.get("shop", {}).get("cards", [])
        if idx is None or idx >= len(cards):
            return np.zeros(size, dtype=np.float32)
        card = cards[idx]
        return np.concatenate([
            _onehot(self.terrain_ix.get(card.get("terrainId")), len(self.terrain_ix)),
            _onehot(self.feature_ix.get(card.get("featureId")), len(self.feature_ix)),
            _onehot(self.resource_ix.get(card.get("resourceId")), len(self.resource_ix)),
            np.array([card.get("tier", 1) / 3.0, _log(card.get("cost", 0)),
                      float(card.get("locked", False)),
                      float(np.sum(self.terrain_yields.get(card.get("terrainId"),
                                                           np.zeros(len(YIELDS), dtype=np.float32))))],
                     dtype=np.float32),
        ]).astype(np.float32)


@lru_cache(maxsize=4096)
def _neighbors(q: int, r: int) -> tuple[tuple[int, int], ...]:
    return ((q + 1, r), (q + 1, r - 1), (q, r - 1), (q - 1, r), (q - 1, r + 1), (q, r + 1))


def _empty_board_index() -> dict:
    return {}


_EMPTY_STATE: dict = {
    "turn": 1, "maxTurns": 40, "phase": "shopping",
    "resources": {}, "accumulated": {}, "perTurnYields": {},
    "shop": {"cards": []}, "goal": {}, "score": {}, "techState": {}, "board": [],
    "factionWeights": {}, "population": 1,
}
_EMPTY_ACTION: dict = {"type": "end_turn", "params": {}, "meta": {}}
