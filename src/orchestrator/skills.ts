/**
 * The skill registry: each capability wrapped behind the uniform Skill interface.
 * TS-native skills (search, market) call our code directly; Python skills
 * (recommend, knowledge) go through the bridge. The router dispatches by intent.
 */
import { handleSearchTurn } from '../agent/conversation.js';
import { getMarketStats, formatMarketStats } from '../market/marketStats.js';
import { MySqlDraftStore, type DraftStore } from '../email/drafts.js';
import { draftEmail, previewDraft } from '../email/email.js';
import { weeklyMarketReport } from '../email/templates.js';
import { SkillRegistry } from './skill.js';
import type { PythonBridge } from './bridge.js';

function extractId(message: string): number | undefined {
  const m = message.match(/\b(\d{6,})\b/);
  return m ? Number(m[1]) : undefined;
}

function extractEmails(message: string): string[] {
  return message.match(/[^@\s]+@[^@\s]+\.[^@\s]+/g) ?? [];
}

export function buildRegistry(bridge: PythonBridge, draftStore: DraftStore = new MySqlDraftStore()): SkillRegistry {
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
    })
    .register({
      name: 'email',
      description: 'Draft an outbound email (e.g. a market report) to a recipient — always pending human approval, never auto-sent.',
      async run(ctx) {
        const recipients = extractEmails(ctx.message);
        if (!recipients.length) return { skill: 'email', reply: 'Who should I email? Include a recipient address.' };
        if (!ctx.filter.city) return { skill: 'email', reply: 'Which city\'s market report? e.g. "email the Irvine report to client@x.com".' };
        const { subject, body } = await weeklyMarketReport(ctx.filter.city);
        const r = await draftEmail({ createdBy: ctx.userId, recipients, subject, body }, draftStore);
        if (!r.ok) return { skill: 'email', reply: `Couldn't draft the email: ${r.error}` };
        return {
          skill: 'email',
          reply: `📝 Drafted (pending your approval — I won't send it myself):\n\n${previewDraft(r.draft!)}\n\nApprove: npm run drafts -- approve ${r.draft!.id}`,
          data: r.draft,
        };
      },
    });
}
