"""Python-side wrapper around the Node game bridge.

Owns a `tsx src/env/bridge.ts` subprocess and speaks the JSONL protocol, giving
a small gym-like surface:

    env = CivEnv()
    obs = env.reset(seed=0)          # obs.state, obs.actions, obs.done
    obs = env.step(action_index)
    env.close()

Games are fully reproducible for a given seed: the bridge replaces Math.random
with a seeded mulberry32 PRNG before starting the game.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "src" / "env" / "bridge.ts"
TSX = ROOT / "node_modules" / ".bin" / "tsx"


@dataclass
class Observation:
    state: dict[str, Any]
    actions: list[dict[str, Any]]
    done: bool
    turn: int
    phase: str
    score: dict[str, float]
    success: bool = True
    message: str = ""
    challenge: dict[str, Any] | None = None

    @property
    def total_score(self) -> float:
        return float(self.score.get("total", 0.0))


class BridgeError(RuntimeError):
    pass


class CivEnv:
    """One game engine subprocess, reusable across episodes."""

    def __init__(self, agent_name: str = "py-agent", max_actions: int | None = 256,
                 config: dict | None = None) -> None:
        if not TSX.exists():
            raise BridgeError("node_modules/.bin/tsx not found - run `npm install`")
        if not (ROOT / "vendor" / "civ6-roguelike-ai").exists():
            raise BridgeError("vendor/civ6-roguelike-ai not found - run `python src/fetch_data.py`")
        self.agent_name = agent_name
        self.max_actions = max_actions
        self.config = config
        self._req_id = 0
        self._rng = random.Random(0)
        env = dict(os.environ, NODE_OPTIONS="--max-old-space-size=1024")
        # stderr goes to a file, not a pipe: nothing drains a stderr pipe during
        # the episode loop, and a full pipe buffer would deadlock the engine.
        self._stderr = tempfile.TemporaryFile(mode="w+")
        self._proc = subprocess.Popen(
            [str(TSX), str(BRIDGE)], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, text=True, bufsize=1, env=env,
        )
        self.last_obs: Observation | None = None

    # ---- protocol plumbing ----

    def _rpc(self, **payload: Any) -> dict:
        if self._proc.poll() is not None:
            raise BridgeError(f"bridge exited with code {self._proc.returncode}")
        self._req_id += 1
        payload["id"] = self._req_id
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise BridgeError(f"bridge closed unexpectedly: {self.stderr_tail()}")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise BridgeError(resp.get("error", "unknown bridge error"))
        return resp

    def _to_obs(self, resp: dict) -> Observation:
        actions = resp.get("actions", [])
        if self.max_actions is not None and len(actions) > self.max_actions:
            actions = self._subsample(actions, self.max_actions)
        obs = Observation(
            state=resp.get("state", self.last_obs.state if self.last_obs else {}),
            actions=actions,
            done=bool(resp.get("done")),
            turn=int(resp.get("turn", 0)),
            phase=str(resp.get("phase", "")),
            score=resp.get("score", {}),
            success=bool(resp.get("success", True)),
            message=str(resp.get("message", "")),
            challenge=resp.get("challenge"),
        )
        self.last_obs = obs
        return obs

    def _subsample(self, actions: list[dict], cap: int) -> list[dict]:
        """Bound the action set while always keeping the turn-advancing options.

        Without a cap the mid-game legal set can run into the many hundreds
        (every tile x every district), which bloats the padded training tensors
        for very little decision value.
        """
        always = [a for a in actions if a["type"] in ("end_turn", "reassign_all_workers",
                                                      "reassign_workers_food_priority", "deselect_card")]
        rest = [a for a in actions if a not in always]
        keep = cap - len(always)
        if keep > 0 and len(rest) > keep:
            rest = self._rng.sample(rest, keep)
        return always + rest

    # ---- gym-ish surface ----

    def reset(self, seed: int = 0) -> Observation:
        self._rng = random.Random(seed)
        self.last_obs = None
        payload: dict[str, Any] = {"cmd": "reset", "seed": int(seed), "agent": self.agent_name}
        if self.config:
            payload["config"] = self.config
        return self._to_obs(self._rpc(**payload))

    def step(self, action: int | dict) -> Observation:
        if isinstance(action, int):
            if self.last_obs is None:
                raise BridgeError("call reset() before step()")
            action = self.last_obs.actions[action]
        return self._to_obs(self._rpc(cmd="step", action={"type": action["type"],
                                                          "params": action.get("params", {})}))

    def observe(self) -> Observation:
        return self._to_obs(self._rpc(cmd="observe"))

    def stderr_tail(self, limit: int = 2000) -> str:
        """Whatever the Node side printed to stderr - for diagnostics."""
        try:
            self._stderr.seek(0)
            return self._stderr.read()[-limit:]
        except Exception:  # noqa: BLE001 - diagnostics only
            return ""

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._rpc(cmd="close")
            except Exception:  # noqa: BLE001 - closing is best effort
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        try:
            self._stderr.close()
        except Exception:  # noqa: BLE001 - best effort
            pass

    def __enter__(self) -> "CivEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
