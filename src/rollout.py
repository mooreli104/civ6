"""Episode driver shared by data generation, evaluation and the self-play loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from env.game_env import CivEnv, Observation

# A turn is a bundle of micro-actions; without a cap an agent can loop forever
# on no-op toggles (assign/unassign) and never advance the game. The cap also
# bounds episode cost: every action costs a full engine round-trip, and the
# engine's legal-action computation gets more expensive as the city fills up.
MAX_ACTIONS_PER_TURN = 25
MAX_FAILED_ACTIONS = 8
# Hard ceiling on a single game. Reaching it means the agent is stuck; the
# driver then plays only turn-advancing actions until the game ends.
MAX_DECISIONS_PER_EPISODE = 2000


def _pick_escape(actions, failing_type: str | None) -> int:
    """Choose an action that gets the game moving again.

    The engine can offer an action that no longer executes - the classic case is
    a card selected while affordable and then made unaffordable by an unlock, so
    the game sits in the `placing` phase where `end_turn` is not even offered and
    every `place_card` fails. Ending the turn, or failing that dropping the
    selected card, breaks the cycle.
    """
    for kind in ("end_turn", "deselect_card"):
        for i, a in enumerate(actions):
            if a["type"] == kind:
                return i
    for i, a in enumerate(actions):
        if a["type"] != failing_type:
            return i
    return 0


@dataclass
class EpisodeResult:
    seed: int
    agent: str
    score: float
    science: float
    culture: float
    faith: float
    turns: int
    population: int
    decisions: int
    failed: int
    goal_achieved: bool
    tech_nodes: int
    records: list[dict[str, Any]] = field(default_factory=list)


def play_episode(
    env: CivEnv,
    agent,
    seed: int,
    *,
    record: bool = False,
    on_decision: Callable[[dict, Sequence[dict], int], None] | None = None,
) -> EpisodeResult:
    """Play one full game. Optionally record every (state, actions, choice)."""
    agent.reset(seed)
    obs: Observation = env.reset(seed=seed)

    records: list[dict[str, Any]] = []
    decisions = failed = 0
    actions_this_turn = 0
    consecutive_failures = 0
    current_turn = obs.turn
    last_type: str | None = None

    while not obs.done:
        actions = obs.actions
        if not actions:
            break

        if obs.turn != current_turn:
            current_turn = obs.turn
            actions_this_turn = 0

        # Force the turn to advance if the agent is spinning its wheels.
        stuck = (actions_this_turn >= MAX_ACTIONS_PER_TURN
                 or consecutive_failures >= MAX_FAILED_ACTIONS
                 or decisions >= MAX_DECISIONS_PER_EPISODE)
        if stuck:
            idx = _pick_escape(actions, last_type)
        else:
            idx = agent.select(obs.state, actions)
            idx = max(0, min(int(idx), len(actions) - 1))

        if record:
            records.append({"state": obs.state, "actions": actions, "choice": idx, "turn": obs.turn})
        if on_decision is not None:
            on_decision(obs.state, actions, idx)

        last_type = actions[idx]["type"]
        obs = env.step(idx)
        decisions += 1
        actions_this_turn += 1
        if not obs.success:
            failed += 1
            consecutive_failures += 1
        else:
            consecutive_failures = 0

    challenge = obs.challenge or {}
    stats = challenge.get("stats", {})
    goal = challenge.get("goal", {})
    score = obs.score or {}
    return EpisodeResult(
        seed=seed,
        agent=getattr(agent, "name", "agent"),
        score=float(score.get("total", 0.0)),
        science=float(score.get("science", 0.0)),
        culture=float(score.get("culture", 0.0)),
        faith=float(score.get("faith", 0.0)),
        turns=int(stats.get("turns", obs.turn)),
        population=int(stats.get("population", 0)),
        decisions=decisions,
        failed=failed,
        goal_achieved=bool(goal.get("achieved", False)),
        tech_nodes=int(stats.get("techNodesSelected", 0)),
        records=records,
    )
