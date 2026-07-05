/**
 * Interactive demo CLI — see the whole pipeline on real data, no black box.
 *
 *   npm run search -- "在 Irvine 找 3 居室带泳池 250万以下"
 *   npm run search -- "3 bed 2 bath condo in San Diego under 900k"
 *   npm run search -- --comps "San Diego"     # recent sold market snapshot
 *
 * Shows: the parsed filter, match count, structural signal, and real listing cards.
 */
import { parseQuery } from '../search/parseQuery.js';
import { searchActiveListings, searchSignal, getSoldComps } from '../search/searchListings.js';
import { formatListingCard } from '../search/listingRow.js';
import { summarizeFilter } from '../search/filters.js';
import { getMarketStats, formatMarketStats } from '../market/marketStats.js';
import { closePool } from '../db.js';

function median(nums: number[]): number {
  const s = [...nums].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : Math.round((s[m - 1]! + s[m]!) / 2);
}

async function showComps(city: string) {
  const comps = await getSoldComps(city, 6);
  console.log(`\n📊 ${city} — sold in the last 6 months: ${comps.length} sales`);
  if (!comps.length) return;
  const prices = comps.map((c) => c.closePrice!).filter(Boolean);
  const ppsf = comps.filter((c) => c.closePrice && c.sqft).map((c) => c.closePrice! / c.sqft!);
  console.log(`   median sold price: $${median(prices).toLocaleString('en-US')}`);
  if (ppsf.length) console.log(`   median $/sqft:     $${median(ppsf).toLocaleString('en-US')}`);
  console.log(`   recent examples:`);
  for (const c of comps.slice(0, 3)) {
    console.log(`     ${c.closeDate}  $${c.closePrice?.toLocaleString('en-US')}  ${c.beds ?? '?'}bd/${c.baths ?? '?'}ba  ${c.sqft ?? '?'}sqft`);
  }
}

async function showSearch(query: string) {
  console.log(`\n💬 Query: ${query}`);
  const { filter, confidence, clarification } = await parseQuery(query);
  console.log(`🔎 Parsed: ${summarizeFilter(filter)}   [confidence: ${confidence}]`);
  if (clarification) { console.log(`❓ ${clarification}`); return; }

  const { rows, total } = await searchActiveListings(filter, 1, 5);
  const sig = searchSignal(total);
  console.log(`📦 ${total} active matches${sig ? `  (${sig.signal})` : ''}`);
  if (sig) console.log(`❓ ${sig.clarification}`);
  console.log(`--- showing top ${rows.length} ---`);
  for (const r of rows) console.log('\n' + formatListingCard(r));
}

async function main() {
  const args = process.argv.slice(2);
  if (args[0] === '--market') {
    console.log('\n' + formatMarketStats(await getMarketStats(args.slice(1).join(' ') || 'Irvine', 12)));
  } else if (args[0] === '--comps') {
    await showComps(args.slice(1).join(' ') || 'San Diego');
  } else if (args.length) {
    await showSearch(args.join(' '));
  } else {
    console.log('Usage: npm run search -- "<your query>"   |   npm run search -- --comps "<city>"');
  }
  await closePool();
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
