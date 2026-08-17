/**
 * bridge.ts - a headless, seeded JSONL worker around the fetched game engine.
 *
 * Protocol: one JSON request per line on stdin, one JSON response per line on
 * stdout. Python owns the agent loop and TensorFlow inference; this process
 * owns the game rules.
 *
 *   -> {"id":1,"cmd":"reset","seed":123}
 *   <- {"id":1,"ok":true,"state":{...},"actions":[...],"done":false}
 *   -> {"id":2,"cmd":"step","action":{"type":"end_turn","params":{}}}
 *   <- {"id":2,"ok":true,"success":true,"state":{...},"actions":[...],"done":false}
 *
 * The engine's only source of randomness is Math.random (shop rolls), so
 * overriding it with a seeded PRNG makes whole games exactly reproducible.
 */
import * as readline from 'node:readline';
import { CivGame } from '../../vendor/civ6-roguelike-ai/src/api/sdk';
import type { IGameConfig } from '../../vendor/civ6-roguelike-ai/src/core/config';
import { flattenActions, type FlatAction } from './flatten';

// ---- deterministic RNG (mulberry32) ----
function makeRng(seed: number): () => number {
  let a = (seed >>> 0) || 1;
  return function rng(): number {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let game: CivGame | null = null;

function snapshot(includeState: boolean): Record<string, unknown> {
  if (!game) throw new Error('no active game - call reset first');
  const done = game.isGameOver();
  const flat: FlatAction[] = done ? [] : flattenActions(game.getValidActions());
  const out: Record<string, unknown> = {
    turn: game.getTurn(),
    phase: game.getPhase(),
    done,
    actions: flat,
    score: game.getScore(),
  };
  if (includeState) out.state = game.getState();
  if (done) out.challenge = game.createChallenge();
  return out;
}

const rl = readline.createInterface({ input: process.stdin });

rl.on('line', (line: string) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let req: any;
  try {
    req = JSON.parse(trimmed);
  } catch (err) {
    process.stdout.write(JSON.stringify({ ok: false, error: `bad json: ${String(err)}` }) + '\n');
    return;
  }

  try {
    switch (req.cmd) {
      case 'ping':
        respond(req, { pong: true });
        break;

      case 'reset': {
        Math.random = makeRng(Number(req.seed ?? 0));
        const config: Partial<IGameConfig> | undefined = req.config ?? undefined;
        game = new CivGame(String(req.agent ?? 'py-agent'), config);
        respond(req, snapshot(true));
        break;
      }

      case 'step': {
        if (!game) throw new Error('no active game - call reset first');
        const action = req.action ?? {};
        const result = game.act({ type: String(action.type), params: action.params ?? {} });
        respond(req, {
          success: result.success,
          message: result.message,
          ...snapshot(req.include_state !== false),
        });
        break;
      }

      case 'observe':
        respond(req, snapshot(true));
        break;

      case 'close':
        respond(req, { closed: true });
        rl.close();
        process.exit(0);
        break;

      default:
        throw new Error(`unknown cmd: ${req.cmd}`);
    }
  } catch (err) {
    process.stdout.write(
      JSON.stringify({ id: req?.id ?? null, ok: false, error: String((err as Error).message ?? err) }) + '\n',
    );
  }
});

function respond(req: any, payload: Record<string, unknown>): void {
  process.stdout.write(JSON.stringify({ id: req.id ?? null, ok: true, ...payload }) + '\n');
}
