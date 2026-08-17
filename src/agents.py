"""Agents that pick one action out of the engine's legal-action list.

All agents share a single interface so the rollout driver, the data generator
and the evaluator can be pointed at any of them:

    agent.reset(seed)                       # start of episode
    idx = agent.select(state, actions)      # index into `actions`

`HeuristicAgent` is the hand-written baseline used to bootstrap training data.
`ModelAgent` wraps the trained TensorFlow policy.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

# Districts weighted by how much they feed the score (science+culture+faith).
SCORE_DISTRICTS = {
    "campus": 1.6, "holy_site": 1.5, "theater_square": 1.5, "university": 1.7,
    "monastery_district": 1.4, "government_plaza": 1.2, "commercial_hub": 1.0,
    "industrial_zone": 1.1, "harbor": 0.9, "aqueduct": 0.7, "barracks": 0.5,
    "entertainment": 0.6,
}


class BaseAgent:
    name = "base"

    def reset(self, seed: int = 0) -> None:  # noqa: D102
        self.rng = random.Random(seed)

    def select(self, state: dict, actions: Sequence[dict]) -> int:  # noqa: D102
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Uniform over legal actions - the floor for evaluation."""

    name = "random"

    def select(self, state: dict, actions: Sequence[dict]) -> int:
        return self.rng.randrange(len(actions))


class GreedyEndTurnAgent(BaseAgent):
    """Buys the cheapest card, places it, auto-assigns workers, ends the turn.

    This mirrors the reference strategy shipped with the game package.
    """

    name = "scripted"

    def reset(self, seed: int = 0) -> None:
        super().reset(seed)
        self._turn = -1
        self._reassigned = False

    def select(self, state: dict, actions: Sequence[dict]) -> int:
        if state.get("turn") != self._turn:
            self._turn = state.get("turn")
            self._reassigned = False

        by_type: dict[str, list[int]] = {}
        for i, a in enumerate(actions):
            by_type.setdefault(a["type"], []).append(i)

        for kind in ("select_tech_node", "place_card"):
            if kind in by_type:
                return by_type[kind][0]
        if "select_card" in by_type:
            return min(by_type["select_card"], key=lambda i: actions[i]["meta"].get("goldCost", 0))
        if "build_improvement" in by_type:
            return by_type["build_improvement"][0]
        if "reassign_all_workers" in by_type and not self._reassigned:
            self._reassigned = True
            return by_type["reassign_all_workers"][0]
        if "end_turn" in by_type:
            return by_type["end_turn"][0]
        return 0


class HeuristicAgent(BaseAgent):
    """Domain-knowledge scoring of every legal action, sampled with a temperature.

    The rules that matter in this game, and that the scoring below encodes:

    * only *worked* tiles produce anything, and the number of workers equals the
      population, so food (growth) compounds into every other yield;
    * the final score is accumulated science + culture + faith, which mostly
      comes from districts (campus / holy site / theater square) and from tech
      nodes, not from raw terrain;
    * tech nodes are free once the accumulated resource passes the layer
      threshold, so they should be taken the moment they appear.

    Temperature 0 is deterministic and reasonably strong; positive values inject
    the exploration that makes elite filtering meaningful.
    """

    name = "heuristic"

    # value of each district as a score engine (science/culture/faith producers)
    DISTRICT_VALUE = SCORE_DISTRICTS
    # improvements that mainly feed growth vs. everything else
    FOOD_IMPROVEMENTS = {"farm", "fishery", "terrace", "hot_spring"}
    # population to aim for before yields matter more than growth
    POP_TARGET = 8
    # production to bank before spending it on anything but a district
    DISTRICT_SAVINGS = 22
    SCORE_IMPROVEMENTS = {"observatory", "monastery", "sacred_grove", "lighthouse"}

    def __init__(self, temperature: float = 0.0) -> None:
        self.temperature = float(temperature)
        self.rng = random.Random(0)
        self._turn = -1
        self._used: set[tuple] = set()

    def reset(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._turn = -1
        self._used = set()

    @staticmethod
    def _signature(action: dict) -> tuple:
        m = action.get("meta", {})
        return (action["type"], m.get("q"), m.get("r"), m.get("nodeId"), m.get("cardIndex"),
                m.get("districtId"), m.get("improvementId"))

    @staticmethod
    def _tile_value(tile: dict | None, need_food: bool) -> float:
        if not tile:
            return 0.0
        y = tile.get("yields", {})
        score_yield = y.get("science", 0) + y.get("culture", 0) + y.get("faith", 0)
        return (2.0 * score_yield
                + (1.4 if need_food else 0.5) * y.get("food", 0)
                + 0.7 * y.get("production", 0)
                + 0.3 * y.get("gold", 0))

    def score_actions(self, state: dict, actions: Sequence[dict]) -> np.ndarray:
        turn = state.get("turn", 1)
        max_turns = max(state.get("maxTurns", 40), 1)
        progress = turn / max_turns
        res = state.get("resources", {})
        gold = res.get("gold", 0)
        prod = res.get("production", 0)
        net_food = res.get("netFoodPerTurn", 0)
        pop = state.get("population", 1)
        workers = state.get("availableWorkers", 0)
        board = {(t["q"], t["r"]): t for t in state.get("board", [])}
        empty_unlocked = sum(1 for t in board.values() if t.get("unlocked") and not t.get("terrainId"))
        unworked = sum(1 for t in board.values() if t.get("terrainId") and not t.get("isWorked"))
        need_food = net_food <= 2 or pop < self.POP_TARGET
        # production is the bottleneck for districts, which are the score engine
        saving_for_district = prod < self.DISTRICT_SAVINGS and progress < 0.7
        late = progress > 0.8

        if turn != self._turn:
            self._turn = turn
            self._used = set()

        scores = np.zeros(len(actions), dtype=np.float32)
        for i, a in enumerate(actions):
            t = a["type"]
            m = a.get("meta", {})
            s = 0.0

            if t == "select_tech_node":
                # Free once the threshold is crossed, and the effects compound.
                s = 12.0 - 0.3 * float(m.get("layer", 1))

            elif t == "build_district":
                # The main source of score: build them as early as affordable.
                value = self.DISTRICT_VALUE.get(str(m.get("districtId")), 0.7)
                tile = board.get((m.get("q"), m.get("r")))
                s = 7.5 + 2.2 * value - 0.04 * float(m.get("productionCost", 0))
                s += 1.0 if (tile and tile.get("isWorked")) else 0.0
                s -= 5.0 * max(progress - 0.75, 0.0) * 4  # no payback in the last turns

            elif t == "build_improvement":
                iid = str(m.get("improvementId"))
                bonus = 1.6 if (need_food and iid in self.FOOD_IMPROVEMENTS) else 0.0
                bonus += 1.4 if iid in self.SCORE_IMPROVEMENTS else 0.0
                s = 5.0 + bonus - 0.03 * float(m.get("productionCost", 0)) - 3.0 * max(progress - 0.7, 0.0) * 3
                if saving_for_district:
                    s -= 3.5  # bank the production for a campus instead

            elif t == "place_card":
                tile = board.get((m.get("q"), m.get("r")))
                ring = tile.get("ring", 2) if tile else 2
                s = 7.0 - 0.4 * ring

            elif t == "select_card":
                cost = float(m.get("goldCost", 0))
                if empty_unlocked > 0:
                    s = 6.0 - 0.25 * cost + 0.03 * gold
                else:
                    s = 1.5 - 0.25 * cost  # nowhere to put it yet
                s -= 2.0 if late else 0.0

            elif t == "unlock_hex":
                cost = float(m.get("goldCost", 0))
                room = 4.5 if empty_unlocked <= 1 else 1.0
                s = room - 0.12 * cost - 4.0 * max(progress - 0.7, 0.0) * 3

            elif t == "assign_worker":
                tile = board.get((m.get("q"), m.get("r")))
                s = (2.5 + self._tile_value(tile, need_food)) if workers > 0 else -5.0

            elif t == "reassign_all_workers":
                # Cheap way to keep every worker on the best tile after a build.
                s = 3.4 if (unworked > 0 or workers > 0) else 1.2

            elif t == "reassign_workers_food_priority":
                s = 3.8 if need_food else -1.0

            elif t == "upgrade_tile":
                cost = float(m.get("productionCost", 0))
                target = str(m.get("targetId", ""))
                s = 3.0 + 0.4 * float(m.get("currentLevel", 0)) - 0.02 * cost
                s += 2.0 * self.DISTRICT_VALUE.get(target, 0.0)  # levelling a campus adds science
                if prod > 2 * cost:
                    s += 2.5  # banked production earns nothing

            elif t == "reroll_shop":
                s = 1.2 if (gold > 15 and empty_unlocked > 0) else -3.0

            elif t == "end_turn":
                s = 3.2

            elif t == "deselect_card":
                s = -1.0
            elif t == "unassign_worker":
                s = -3.0
            elif t == "sell_tile":
                s = -8.0

            if self._signature(a) in self._used:
                s -= 6.0  # already did this exact thing this turn
            scores[i] = s
        return scores

    def select(self, state: dict, actions: Sequence[dict]) -> int:
        scores = self.score_actions(state, actions)
        if self.temperature <= 1e-6:
            best = float(scores.max())
            ties = [i for i, s in enumerate(scores) if s >= best - 1e-6]
            choice = self.rng.choice(ties)
        else:
            z = (scores - scores.max()) / self.temperature
            p = np.exp(z)
            p /= p.sum()
            choice = int(self.rng.choices(range(len(actions)), weights=p.tolist(), k=1)[0])
        self._used.add(self._signature(actions[choice]))
        return choice


class ModelAgent(BaseAgent):
    """Plays with the trained TensorFlow policy.

    `temperature=0` gives the greedy policy used for evaluation; positive values
    sample from it, which is how later self-improvement rounds explore.

    The network is memoryless: it sees a state and a legal-action list, so
    nothing stops it from re-picking an idempotent action (a worker reshuffle
    that changes nothing) forever. The training data never contains such
    repeats - the heuristic teacher refuses them - so the same constraint is
    applied here, keeping inference inside the distribution the policy learned.
    """

    name = "model"

    # one traced graph per (model, candidate-count bucket) - re-tracing for every
    # distinct action count would cost more than it saves
    _TRACED: dict[int, object] = {}

    def __init__(self, model, encoder, temperature: float = 0.0, epsilon: float = 0.0,
                 avoid_repeats: bool = True) -> None:
        self.model = model
        self._predict = self._traced_predict(model, encoder.state_dim, encoder.action_dim)
        self.encoder = encoder
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.avoid_repeats = avoid_repeats
        self.rng = random.Random(0)
        self._turn = -1
        self._used: set[tuple] = set()

    def reset(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._turn = -1
        self._used = set()

    @classmethod
    def _traced_predict(cls, model, state_dim: int, action_dim: int):
        """Wrap the model in a tf.function with a fully dynamic candidate axis.

        Calling a Keras model directly costs tens of milliseconds of Python
        overhead per decision, and an agent makes hundreds of decisions per
        game. Tracing once, with `None` only for the candidate axis (the feature
        dimensions have to stay static for LayerNormalization), lets every
        action count reuse the same graph.
        """
        key = id(model)
        traced = cls._TRACED.get(key)
        if traced is None:
            import tensorflow as tf

            @tf.function(reduce_retracing=True, input_signature=[{
                "state": tf.TensorSpec([1, state_dim], tf.float32),
                "actions": tf.TensorSpec([1, None, action_dim], tf.float32),
                "mask": tf.TensorSpec([1, None], tf.float32),
            }])
            def _fn(inputs):
                return model(inputs, training=False)

            traced = _fn
            cls._TRACED[key] = traced
        return traced

    def logits(self, state: dict, actions: Sequence[dict]) -> np.ndarray:
        s = self.encoder.encode_state(state)[None, :]
        a = self.encoder.encode_actions(state, list(actions))[None, :, :]
        mask = np.ones((1, a.shape[1]), dtype=np.float32)
        out = self._predict({"state": s, "actions": a, "mask": mask})
        return np.asarray(out["logits"])[0, : len(actions)].astype(np.float64)

    def value(self, state: dict, actions: Sequence[dict]) -> float:
        """The value head's estimate of the final score, in normalized units."""
        s = self.encoder.encode_state(state)[None, :]
        a = self.encoder.encode_actions(state, list(actions[:1]))[None, :, :]
        out = self._predict({"state": s, "actions": a, "mask": np.ones((1, 1), dtype=np.float32)})
        return float(np.asarray(out["value"]).reshape(-1)[0])

    def select(self, state: dict, actions: Sequence[dict]) -> int:
        turn = state.get("turn", 0)
        if turn != self._turn:
            self._turn = turn
            self._used = set()

        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            choice = self.rng.randrange(len(actions))
        else:
            logits = self.logits(state, actions)
            if self.avoid_repeats:
                blocked = [i for i, a in enumerate(actions)
                           if HeuristicAgent._signature(a) in self._used]
                if len(blocked) < len(actions):
                    logits[blocked] = -np.inf
            if self.temperature <= 1e-6:
                choice = int(np.argmax(logits))
            else:
                z = logits / self.temperature
                z -= z.max()
                p = np.exp(z)
                total = p.sum()
                if not np.isfinite(total) or total <= 0:
                    choice = int(np.argmax(logits))
                else:
                    p /= total
                    choice = int(self.rng.choices(range(len(actions)), weights=p.tolist(), k=1)[0])
        self._used.add(HeuristicAgent._signature(actions[choice]))
        return choice


def make_agent(spec: str, **kwargs) -> BaseAgent:
    """`random` | `scripted` | `heuristic` | `heuristic:0.8`"""
    name, _, param = spec.partition(":")
    if name == "random":
        return RandomAgent()
    if name == "scripted":
        return GreedyEndTurnAgent()
    if name == "heuristic":
        return HeuristicAgent(temperature=float(param) if param else 0.0)
    raise ValueError(f"unknown agent spec: {spec!r}")
