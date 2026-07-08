/**
 * The skill registry: each capability wrapped behind the uniform Skill interface.
 * TS-native skills (search, market) call our code directly; Python skills
 * (recommend, knowledge) go through the bridge. The router dispatches by intent.
 */
import { handleSearchTurn } from '../agent/conversation.js';
import { getMarketStats, formatMarketStats } from '../market/marketStats.js';
import { formatListingCard, type ListingRow } from '../search/listingRow.js';
import { summarizeFilter, type SearchFilter } from '../search/filters.js';
import { MySqlDraftStore, type DraftStore } from '../email/drafts.js';
import { draftEmail, previewDraft } from '../email/email.js';
import { weeklyMarketReport } from '../email/templates.js';
import { SkillRegistry } from './skill.js';
import type { PythonBridge, SemanticListing } from './bridge.js';

/** Normalized propertyType -> physical L_Type_ value (matches the Qdrant payload). */
const TYPE_DB: Record<string, string> = {
  'condo': 'Condominium', 'townhouse': 'Townhouse', 'single-family': 'SingleFamilyResidence',
};

function extractId(message: string): number | undefined {
  const m = message.match(/\b(\d{6,})\b/);
  return m ? Number(m[1]) : undefined;
}

function extractEmails(message: string): string[] {
  // TLD must end in letters so trailing punctuation (commas, periods) isn't captured.
  return message.match(/[^\s@,;<>]+@[^\s@,;<>]+\.[a-zA-Z]{2,}/g) ?? [];
}

const EN_FILLER = /\b(find|show|me|i|want|looking|for|homes?|houses?|property|properties|in|near|around|with|under|over|below|above|between|and|a|an|the|please|less|than|more|of|to)\b/gi;
const ZH_FILLER = /找|帮我|想要?|房子|套|在|的|一下|附近|以下|以上|以内|左右|大概|带|要|有没有|给我/g;

/** The "soft" part of a query — what's left after removing structured/filler tokens.
 * Non-empty => the user wants semantic matching (route to the Qdrant hybrid). */
export function extractSemanticText(message: string, filter: SearchFilter): string {
  let s = ` ${message} `;
  if (filter.city) s = s.replace(new RegExp(filter.city, 'gi'), ' ');
  // numbers + units (beds/baths/price/sqft, incl. 中文)
  s = s.replace(/\$?\d[\d.,]*\s*(?:万|million|mil|m|k|thousand|bed(?:room)?s?|br|bd|bath(?:room)?s?|ba|sqft|sq\.?\s?ft|square feet|平方英尺|居室|室|卧室|卧|房|卫|平)?/gi, ' ');
  // property-type + pool words
  s = s.replace(/\b(?:single[\s-]?family|sfr|detached|town\s?houses?|town\s?homes?|condos?|condominiums?|apartments?|apts?)\b|独栋|单户|联排|公寓|pool|泳池|游泳池/gi, ' ');
  s = s.replace(EN_FILLER, ' ').replace(ZH_FILLER, ' ');
  s = s.replace(/[，,、。.!?？!:：;；]/g, ' ').replace(/\s+/g, ' ').trim();
  const content = s.match(/[A-Za-z]{2,}|[一-鿿]/g) ?? [];
  return content.length >= 1 ? s : '';
}

function toListingRow(s: SemanticListing): ListingRow {
  return {
    id: s.listing_id ?? 0, listingId: null, mls: s.mls ?? null, address: s.address ?? null,
    city: s.city ?? null, zip: null, type: s.type ?? null, beds: s.beds ?? null,
    baths: s.baths ?? null, sqft: s.sqft ?? null, price: s.price ?? null,
    photoCount: null, yearBuilt: null, pool: s.pool ?? false,
  };
}

function formatSemanticResults(results: SemanticListing[], filter: SearchFilter, semantic: string): string {
  const cards = results.map((r) => formatListingCard(toListingRow(r))).join('\n\n');
  return `🔎 ${summarizeFilter(filter)}  ·  semantic: "${semantic}"\nTop ${results.length} by relevance:\n\n${cards}`;
}

export function buildRegistry(bridge: PythonBridge, draftStore: DraftStore = new MySqlDraftStore()): SkillRegistry {
  return new SkillRegistry()
    .register({
      name: 'search',
      description: 'Search active listings by city, beds/baths, budget, type, pool, or free-text style (e.g. "ocean view craftsman").',
      async run(ctx) {
        // Soft/semantic content -> hybrid (Qdrant: hard filters + dense+BM25).
        const semantic = extractSemanticText(ctx.message, ctx.filter);
        if (semantic) {
          try {
            const results = await bridge.search({
              text: semantic,
              city: ctx.filter.city ?? null,
              max_price: ctx.filter.maxPrice ?? null,
              min_price: ctx.filter.minPrice ?? null,
              min_beds: ctx.filter.beds ?? null,
              pool: ctx.filter.pool ?? false,
              ptype: ctx.filter.propertyType ? TYPE_DB[ctx.filter.propertyType] ?? null : null,
              k: 5,
            });
            if (results.length) {
              return { skill: 'search', reply: formatSemanticResults(results, ctx.filter, semantic), data: { semantic: true, results } };
            }
            // no semantic matches -> fall through to structured search
          } catch {
            // Qdrant/service down -> degrade to MySQL structured search (乙)
          }
        }
        // Pure structured (or fallback): multi-turn MySQL search (LLM-aware parse).
        const turn = await handleSearchTurn(ctx.userId, ctx.message, { llm: ctx.llm });
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
          reply: `📝 Drafted (pending your approval — I won't send it myself):\n\n${previewDraft(r.draft!)}\n\nReply "approve ${r.draft!.id}" to send, or "cancel ${r.draft!.id}".`,
          data: r.draft,
        };
      },
    });
}
