/**
 * Deterministic email templates (Week 11). Content is filled from real data
 * (market stats / listing rows) — facts come from the data, not an LLM (Q5). An
 * LLM "polish the prose" layer is a reserved slot for later; the template stands
 * on its own with no key.
 */
import { getMarketStats, formatMarketStats } from '../market/marketStats.js';
import { formatListingCard, type ListingRow } from '../search/listingRow.js';

export interface EmailContent {
  subject: string;
  body: string;
}

/** Weekly market report for a city, filled from california_sold stats. */
export async function weeklyMarketReport(city: string): Promise<EmailContent> {
  const stats = await getMarketStats(city, 12);
  const subject = stats.count
    ? `${city} market update — median $${(stats.medianPrice ?? 0).toLocaleString('en-US')}`
    : `${city} market update`;
  const body = [
    `Hi,`,
    ``,
    `Here's this week's ${city} market snapshot:`,
    ``,
    formatMarketStats(stats),
    ``,
    `Reply if you'd like listings that match your criteria.`,
  ].join('\n');
  return { subject, body };
}

/** Property summary / recommendation digest from listing rows. */
export function listingDigest(rows: ListingRow[], intro = 'Here are some listings you might like:'): EmailContent {
  const city = rows[0]?.city ?? '';
  const subject = `${rows.length} listing${rows.length === 1 ? '' : 's'}${city ? ` in ${city}` : ''} for you`;
  const body = [`Hi,`, ``, intro, ``, ...rows.map((r) => formatListingCard(r)), ``, `Let me know which you'd like to tour.`].join('\n\n');
  return { subject, body };
}
