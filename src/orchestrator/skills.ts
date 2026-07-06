/**
 * The skill registry: each capability wrapped behind the uniform Skill interface.
 * TS-native skills (search, market) call our code directly; Python skills
 * (recommend, knowledge) go through the bridge. The router dispatches by intent.
 */
import { handleSearchTurn } from '../agent/conversation.js';
import { getMarketStats, formatMarketStats } from '../market/marketStats.js';
import { SkillRegistry } from './skill.js';
import type { PythonBridge } from './bridge.js';

function extractId(message: string): number | undefined {
  const m = message.match(/\b(\d{6,})\b/);
  return m ? Number(m[1]) : undefined;
}

export function buildRegistry(bridge: PythonBridge): SkillRegistry {
  return new SkillRegistry()
    .register({
      name: 'search',
      description: 'Search active listings by city, beds/baths, budget, type, pool, or keywords.',
      async run(ctx) {
        const turn = await handleSearchTurn(ctx.userId, ctx.message);
        return { skill: 'search', reply: turn.reply, data: turn };
      },
    })
    .register({
      name: 'market',
      description: 'City market stats: median price, $/sqft, days on market, sold-to-list, trend.',
      async run(ctx) {
        if (!ctx.filter.city) {
          return { skill: 'market', reply: 'Which city do you want market stats for?' };
        }
        const stats = await getMarketStats(ctx.filter.city, 12);
        return { skill: 'market', reply: formatMarketStats(stats), data: stats };
      },
    })
    .register({
      name: 'recommend',
      description: 'Given a listing you like (by id/MLS), recommend similar homes with a price check.',
      async run(ctx) {
        const id = extractId(ctx.message);
        if (!id) {
          return { skill: 'recommend', reply: 'Which listing? Send its id / MLS number.' };
        }
        return { skill: 'recommend', reply: await bridge.recommend(id) };
      },
    })
    .register({
      name: 'knowledge',
      description: 'Answer real-estate questions (DOM, $/sqft, comps, field meanings) with sources.',
      async run(ctx) {
        return { skill: 'knowledge', reply: await bridge.rag(ctx.message) };
      },
    });
}
