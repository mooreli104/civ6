/**
 * flatten.ts - turn the engine's grouped `getValidActions()` payload into a flat
 * list of atomic, directly-executable actions.
 *
 * The engine reports actions grouped by kind ("build_district" with a list of
 * tiles, each with a list of options). A policy network needs one entry per
 * concrete choice, each carrying the numeric fields the encoder consumes.
 * Unaffordable options are dropped: they are not executable this instant.
 */
import type { ValidActionsResponse } from '../../vendor/civ6-roguelike-ai/src/api/valid-actions';

export interface FlatAction {
  /** engine action type */
  type: string;
  /** params passed straight to CivGame.act() */
  params: Record<string, unknown>;
  /** numeric/id fields for feature encoding */
  meta: Record<string, unknown>;
}

export function flattenActions(resp: ValidActionsResponse): FlatAction[] {
  const out: FlatAction[] = [];
  const push = (type: string, params: Record<string, unknown>, meta: Record<string, unknown> = {}) =>
    out.push({ type, params, meta });

  for (const action of resp.actions) {
    switch (action.type) {
      case 'select_card': {
        for (const o of (action.options ?? []) as any[]) {
          if (!o.affordable) continue;
          push('select_card', { cardIndex: o.index }, { cardIndex: o.index, goldCost: o.cost });
        }
        break;
      }

      case 'place_card': {
        const card = (action.card ?? null) as { name?: string; cost?: number } | null;
        for (const c of (action.coords ?? []) as any[]) {
          push('place_card', { q: c.q, r: c.r }, { q: c.q, r: c.r, goldCost: card?.cost ?? 0 });
        }
        break;
      }

      case 'deselect_card':
        push('deselect_card', {}, {});
        break;

      case 'reroll_shop':
        if (action.affordable) {
          push('reroll_shop', {}, { goldCost: action.cost, remaining: action.remaining });
        }
        break;

      case 'build_improvement': {
        for (const t of (action.tiles ?? []) as any[]) {
          for (const o of t.options as any[]) {
            if (!o.affordable) continue;
            push('build_improvement', { q: t.q, r: t.r, improvementId: o.id },
                 { q: t.q, r: t.r, improvementId: o.id, productionCost: o.productionCost });
          }
        }
        break;
      }

      case 'build_district': {
        for (const t of (action.tiles ?? []) as any[]) {
          for (const o of t.options as any[]) {
            if (!o.affordable) continue;
            push('build_district', { q: t.q, r: t.r, districtId: o.id },
                 { q: t.q, r: t.r, districtId: o.id, productionCost: o.productionCost });
          }
        }
        break;
      }

      case 'upgrade_tile': {
        for (const t of (action.tiles ?? []) as any[]) {
          if (!t.affordable) continue;
          push('upgrade_tile', { q: t.q, r: t.r },
               { q: t.q, r: t.r, productionCost: t.productionCost, targetId: t.targetId,
                 currentLevel: t.currentLevel, maxLevel: t.maxLevel });
        }
        break;
      }

      case 'assign_worker': {
        for (const c of (action.coords ?? []) as any[]) {
          push('assign_worker', { q: c.q, r: c.r }, { q: c.q, r: c.r });
        }
        break;
      }

      case 'unassign_worker': {
        for (const c of (action.coords ?? []) as any[]) {
          push('unassign_worker', { q: c.q, r: c.r }, { q: c.q, r: c.r });
        }
        break;
      }

      case 'reassign_all_workers':
      case 'reassign_workers_food_priority':
        push(action.type, {}, {});
        break;

      case 'unlock_hex': {
        for (const t of (action.tiles ?? []) as any[]) {
          if (!t.affordable) continue;
          push('unlock_hex', { q: t.q, r: t.r }, { q: t.q, r: t.r, goldCost: t.cost });
        }
        break;
      }

      case 'sell_tile': {
        for (const t of (action.tiles ?? []) as any[]) {
          push('sell_tile', { q: t.q, r: t.r }, { q: t.q, r: t.r, terrainId: t.terrainId });
        }
        break;
      }

      case 'select_tech_node': {
        for (const o of (action.options ?? []) as any[]) {
          push('select_tech_node', { treeId: o.treeId, nodeId: o.nodeId },
               { treeId: o.treeId, nodeId: o.nodeId, layer: o.layer });
        }
        break;
      }

      case 'end_turn':
        push('end_turn', {}, {});
        break;

      // toggle_shop_lock and new_game are deliberately excluded: the former is a
      // no-op for a single-episode agent, the latter is driven by the harness.
      default:
        break;
    }
  }

  return out;
}
