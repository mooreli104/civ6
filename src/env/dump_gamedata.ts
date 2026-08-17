/**
 * Dumps the fetched game's static rule tables to JSON on stdout.
 *
 * Importing the TypeScript modules (rather than regex-scraping them) means the
 * numbers we train on are exactly the numbers the engine plays with.
 */
import {
  TERRAIN_REGISTRY,
  FEATURE_REGISTRY,
  RESOURCE_REGISTRY,
} from '../../vendor/civ6-roguelike-ai/src/data/terrains';
import { DISTRICT_REGISTRY, BASE_DISTRICTS, UNLOCK_DISTRICTS } from '../../vendor/civ6-roguelike-ai/src/data/districts';
import {
  IMPROVEMENT_REGISTRY,
  BASE_IMPROVEMENTS,
  UNLOCK_IMPROVEMENTS,
} from '../../vendor/civ6-roguelike-ai/src/data/improvements';
import {
  ALL_TECH_TREES,
  EUREKA_DEFS,
} from '../../vendor/civ6-roguelike-ai/src/data/tech-trees';
import {
  ALL_CARD_TEMPLATES,
  ALL_FACTIONS,
  FACTION_META,
  SHOP_LEVEL_CONFIGS,
} from '../../vendor/civ6-roguelike-ai/src/data/card-pool';
import { DEFAULT_CONFIG, VICTORY_GOAL_PRESETS } from '../../vendor/civ6-roguelike-ai/src/core/config';

const payload = {
  terrains: Object.values(TERRAIN_REGISTRY),
  features: Object.values(FEATURE_REGISTRY),
  resources: Object.values(RESOURCE_REGISTRY),
  districts: Object.values(DISTRICT_REGISTRY),
  district_availability: { base: BASE_DISTRICTS, tech_unlocked: UNLOCK_DISTRICTS },
  improvements: Object.values(IMPROVEMENT_REGISTRY),
  improvement_availability: { base: BASE_IMPROVEMENTS, tech_unlocked: UNLOCK_IMPROVEMENTS },
  tech_trees: ALL_TECH_TREES,
  eurekas: EUREKA_DEFS,
  cards: ALL_CARD_TEMPLATES,
  factions: ALL_FACTIONS,
  faction_meta: FACTION_META,
  shop_levels: SHOP_LEVEL_CONFIGS,
  default_config: DEFAULT_CONFIG,
  victory_goals: VICTORY_GOAL_PRESETS,
};

process.stdout.write(JSON.stringify(payload));
