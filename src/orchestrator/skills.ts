/**
 * The skill registry: each capability wrapped behind the uniform Skill interface.
 * TS-native skills (search, market) call our code directly; Python skills
 * (recommend, knowledge) go through the bridge. The router dispatches by intent.
 */
import { handleSearchTurn } from '../agent/conversation.js';
import { defaultSessionStore, freshSession } from '../agent/session.js';
import { getMarketStats, formatMarketStats } from '../market/marketStats.js';
import { formatListingCard, type ListingRow } from '../search/listingRow.js';
import { summarizeFilter, type SearchFilter } from '../search/filters.js';
import { getMapsClient, type MapsClient } from '../maps/mapsClient.js';
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

const CN_NUM: Record<string, number> = { 一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9, 十: 10 };
/** "第2个" / "#3" / "the first" -> a 1-based position. */
function ordinalOf(message: string): number | undefined {
  let m: RegExpMatchArray | null;
  if ((m = message.match(/第\s*([一二两三四五六七八九十]|\d+)\s*(?:个|套|条|间)?/))) {
    return /\d/.test(m[1]!) ? Number(m[1]) : CN_NUM[m[1]!];
  }
  if ((m = message.match(/#\s*(\d+)/))) return Number(m[1]);
  for (const [re, n] of [[/\b(first|1st)\b/i, 1], [/\b(second|2nd)\b/i, 2], [/\b(third|3rd)\b/i, 3],
                         [/\b(fourth|4th)\b/i, 4], [/\b(fifth|5th)\b/i, 5]] as Array<[RegExp, number]>) {
    if (re.test(message)) return n;
  }
  return undefined;
}
/** Resolve "跟第一个/那套 Canterbury 类似的" against the last shown listings -> a listing id. */
function resolveListingRef(message: string, rows?: ListingRow[]): number | undefined {
  if (!rows?.length) return undefined;
  const ord = ordinalOf(message);
  if (ord && rows[ord - 1]) return rows[ord - 1]!.id;
  for (const r of rows) {                                   // else match an address token
    for (const w of (r.address ?? '').split(/\s+/)) {
      if (w.length >= 3 && /[A-Za-z]/.test(w) && message.includes(w)) return r.id;
    }
  }
  return undefined;
}

/** Persist the just-shown listings so a later "跟第一个类似的" can reference them. */
async function rememberResults(userId: string, rows: ListingRow[]): Promise<void> {
  const s = (await defaultSessionStore.get(userId)) ?? freshSession();
  s.lastResults = rows;
  await defaultSessionStore.set(userId, s);
}

function extractEmails(message: string): string[] {
  // TLD must end in letters so trailing punctuation (commas, periods) isn't captured.
  return message.match(/[^\s@,;<>]+@[^\s@,;<>]+\.[a-zA-Z]{2,}/g) ?? [];
}

const EN_FILLER = /\b(find|show|me|i|want|looking|for|homes?|houses?|property|properties|in|near|around|with|under|over|below|above|between|and|a|an|the|please|less|than|more|of|to)\b/gi;
const ZH_FILLER = /找|帮我|想要?|房子|套|在|的|一下|附近|以下|以上|以内|左右|大概|带|要|有没有|给我|再|这个?|这座|城市/g;
// Cross-intent action/query words — never property FEATURES, so they must not pollute
// the semantic search text (matters when a multi-skill plan hands the full message to search).
const ACTION_WORDS = /\b(market|trend|appreciat\w*|similar|recommend|comparable|worth)\b|行情|怎么样|看看|走势|趋势|市场|贵不贵|值不值|类似|相似|推荐|均价|中位|多少钱|房价|涨了?|跌了?|升值|贬值|另外|房子/gi;

/** The "soft" part of a query — what's left after removing structured/filler tokens.
 * Non-empty => the user wants semantic matching (route to the Qdrant hybrid). */
export function extractSemanticText(message: string, filter: SearchFilter): string {
  let s = ` ${message} `;
  if (filter.city) s = s.replace(new RegExp(filter.city, 'gi'), ' ');
  // numbers + units (beds/baths/price/sqft, incl. 中文)
  s = s.replace(/\$?\d[\d.,]*\s*(?:万|million|mil|m|k|thousand|bed(?:room)?s?|br|bd|bath(?:room)?s?|ba|sqft|sq\.?\s?ft|square feet|平方英尺|居室|室|卧室|卧|房|卫|平)?/gi, ' ');
  // property-type + pool words
  s = s.replace(/\b(?:single[\s-]?family|sfr|detached|town\s?houses?|town\s?homes?|condos?|condominiums?|apartments?|apts?)\b|独栋|单户|联排|公寓|pool|泳池|游泳池/gi, ' ');
  s = s.replace(ACTION_WORDS, ' ').replace(EN_FILLER, ' ').replace(ZH_FILLER, ' ');
  s = s.replace(/[，,、。.!?？!:：;；]/g, ' ').replace(/\s+/g, ' ').trim();
  const content = s.match(/[A-Za-z]{2,}|[一-鿿]/g) ?? [];
  return content.length >= 1 ? s : '';
}

/** Apply a proximity constraint to listings via the maps client (CODE-driven: only
 * called because filter.proximity exists). Geocodes the destination, computes commute
 * for listings that have coordinates, annotates commuteMinutes, filters by withinMinutes
 * (if given) and sorts by commute. Degrades to the original rows on any failure (乙). */
export async function applyProximity(
  rows: ListingRow[],
  prox: NonNullable<SearchFilter['proximity']>,
  maps: MapsClient = getMapsClient(),   // injectable for tests
): Promise<ListingRow[]> {
  if (!maps.available) return rows;                       // no key -> skip silently
  const dest = await maps.geocode(prox.to);
  if (!dest) return rows;                                 // can't resolve destination -> skip
  const geo = rows.filter((r) => r.lat != null && r.lng != null);
  if (!geo.length) return rows;
  const commutes = await maps.commuteMinutes(
    geo.map((r) => ({ id: r.id, lat: r.lat!, lng: r.lng! })), dest, prox.mode ?? 'driving');
  const byId = new Map(commutes.map((c) => [c.id, c.minutes]));
  let out = rows.map((r) => ({ ...r, commuteMinutes: byId.get(r.id) ?? null }));
  if (prox.withinMinutes != null) {
    out = out.filter((r) => r.commuteMinutes != null && r.commuteMinutes <= prox.withinMinutes!);
  }
  out.sort((a, b) => (a.commuteMinutes ?? Infinity) - (b.commuteMinutes ?? Infinity));
  return out;
}

function toListingRow(s: SemanticListing): ListingRow {
  return {
    id: s.listing_id ?? 0, listingId: null, mls: s.mls ?? null, address: s.address ?? null,
    city: s.city ?? null, zip: null, type: s.type ?? null, beds: s.beds ?? null,
    baths: s.baths ?? null, sqft: s.sqft ?? null, price: s.price ?? null,
    photoCount: null, yearBuilt: null, pool: s.pool ?? false,
    lat: null, lng: null,
  };
}

function formatSemanticResults(results: SemanticListing[], filter: SearchFilter, semantic: string): string {
  const cards = results.map((r, i) => `${i + 1}. ${formatListingCard(toListingRow(r))}`).join('\n\n');
  return `🔎 ${summarizeFilter(filter)}  ·  semantic: "${semantic}"\nTop ${results.length} by relevance:\n\n${cards}`;
}

export function buildRegistry(bridge: PythonBridge, draftStore: DraftStore = new MySqlDraftStore()): SkillRegistry {
  return new SkillRegistry()
    .register({
      name: 'search', parallelSafe: true,
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
              pool: ctx.filter.pool,   // tri-state: true=has / false=no / undefined=don't care
              ptype: ctx.filter.propertyType ? TYPE_DB[ctx.filter.propertyType] ?? null : null,
              k: 5,
            });
            if (results.length) {
              const rows = results.map(toListingRow);
              await rememberResults(ctx.userId, rows);   // so "跟第一个类似的" can reference these
              return { skill: 'search', reply: formatSemanticResults(results, ctx.filter, semantic), data: { semantic: true, results } };
            }
            // no semantic matches -> fall through to structured search
          } catch {
            // Qdrant/service down -> degrade to MySQL structured search (乙)
          }
        }
        // Pure structured: deterministic mode re-parses (multi-turn); auto mode passes the
        // filter the LLM (+ memory) already extracted, so it isn't re-parsed / no regex fallback.
        const turn = await handleSearchTurn(ctx.userId, ctx.message, {
          llm: ctx.llm, filter: ctx.args ? ctx.filter : undefined,
        });
        // proximity: CODE decides to call maps because the slot exists (not the LLM).
        if (ctx.filter.proximity && turn.rows?.length) {
          const ranked = await applyProximity(turn.rows, ctx.filter.proximity);
          if (ranked.length) {
            const cards = ranked.map((r, i) => {
              const c = r.commuteMinutes != null ? `  ⏱ ${r.commuteMinutes}min to ${ctx.filter.proximity!.to}` : '';
              return `${i + 1}. ${formatListingCard(r)}${c}`;
            }).join('\n\n');
            const head = `🔎 ${summarizeFilter(ctx.filter)}\nTop ${ranked.length} by commute:`;
            return { skill: 'search', reply: `${head}\n\n${cards}`, data: { ...turn, rows: ranked, proximity: true } };
          }
        }
        return { skill: 'search', reply: turn.reply, data: turn };
      },
    })
    .register({
      name: 'market', parallelSafe: true,
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
      name: 'recommend', parallelSafe: true,
      description: 'Recommend homes similar to one the user references — by position ("the first one" / "第2个"), address, or an explicit MLS id — with a price check.',
      async run(ctx) {
        // resolve WITHOUT making the user type a raw id: an explicit id, else a reference
        // ("第一个" / an address) into the last shown listings (session memory).
        let id = extractId(ctx.message);
        if (!id) {
          const session = await defaultSessionStore.get(ctx.userId);
          id = resolveListingRef(ctx.message, session?.lastResults);
        }
        if (!id) {
          return { skill: 'recommend', reply: 'Which listing? Search first, then say e.g. "跟第一个类似的" / "more like #2" — or send its MLS number.' };
        }
        return { skill: 'recommend', reply: await bridge.recommend(id) };
      },
    })
    .register({
      name: 'knowledge', parallelSafe: true,
      description: 'Answer real-estate questions (DOM, $/sqft, comps, field meanings) with sources.',
      async run(ctx) {
        return { skill: 'knowledge', reply: await bridge.rag(ctx.message) };
      },
    })
    .register({
      name: 'email', parallelSafe: true,
      description: 'Draft an outbound email to a recipient — always pending human approval, never auto-sent. '
        + 'Provide a custom subject+body to author any email, or omit them + give a city to use the market-report template.',
      // Bespoke params: an agent can author subject/body directly (router mode leaves ctx.args undefined -> template).
      paramSchema: {
        type: 'object',
        properties: {
          recipients: { type: 'array', items: { type: 'string' }, description: 'Recipient email addresses.' },
          subject: { type: 'string', description: 'Email subject. Provide (with body) to author a custom email.' },
          body: { type: 'string', description: 'Full email body text. Provide (with subject) to author a custom email; omit to use the city market-report template.' },
          city: { type: 'string', description: 'City for the market-report template (used only when body is omitted).' },
          query: { type: 'string', description: 'What the email is about (fallback only; prefer subject/body or city).' },
        },
      },
      async run(ctx) {
        const a = ctx.args ?? {};
        const argRecipients = Array.isArray(a.recipients)
          ? a.recipients.filter((x): x is string => typeof x === 'string') : [];
        const recipients = argRecipients.length ? argRecipients : extractEmails(ctx.message);
        if (!recipients.length) return { skill: 'email', reply: 'Who should I email? Include a recipient address.' };
        const argSubject = typeof a.subject === 'string' && a.subject.trim() ? a.subject.trim() : undefined;
        const argBody = typeof a.body === 'string' && a.body.trim() ? a.body.trim() : undefined;
        let subject: string, body: string;
        if (argSubject && argBody) {
          ({ subject, body } = { subject: argSubject, body: argBody });   // agent-authored email
        } else {
          if (!ctx.filter.city) return { skill: 'email', reply: 'Which city\'s market report? e.g. "email the Irvine report to client@x.com" (or give me a subject + body).' };
          ({ subject, body } = await weeklyMarketReport(ctx.filter.city));  // template fallback
        }
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
