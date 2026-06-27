/**
 * Parser eval harness — measures regex+normalization coverage and the LLM-escalation
 * decision, WITHOUT calling a real LLM (a spy records whether escalation would fire).
 *
 * This is the "use an LLM as a sample source" idea done right: the diverse cases below
 * are the test corpus; the runtime stays deterministic. Run weekly to catch regressions.
 *
 * Run: npm run eval
 */
import { parseQuery } from './parseQuery.js';
import type { SearchFilter } from './filters.js';
import type { LLMClient } from '../llm/client.js';

interface EvalCase {
  q: string;
  cat: string;
  expect?: Partial<SearchFilter>; // fields regex (after normalization) should capture
  escalate?: boolean;             // should it consult the LLM?
}

const CASES: EvalCase[] = [
  // explicit EN — should NOT escalate
  { q: '3 bed 2 bath in Irvine under 900k', cat: 'explicit-en', expect: { city: 'Irvine', beds: 3, baths: 2, maxPrice: 900_000 }, escalate: false },
  { q: 'condo in San Diego between 500k and 800k', cat: 'explicit-en', expect: { city: 'San Diego', propertyType: 'condo', minPrice: 500_000, maxPrice: 800_000 }, escalate: false },
  { q: 'single family home in Tustin over 1m', cat: 'explicit-en', expect: { city: 'Tustin', propertyType: 'single-family', minPrice: 1_000_000 }, escalate: false },
  // explicit 中文
  { q: '在 Irvine 找 3 居室带泳池、250万以下', cat: 'explicit-zh', expect: { city: 'Irvine', beds: 3, pool: true, maxPrice: 2_500_000 }, escalate: false },
  { q: '在 San Jose 找 200万以上的独栋', cat: 'explicit-zh', expect: { city: 'San Jose', propertyType: 'single-family', minPrice: 2_000_000 }, escalate: false },
  // normalization wins (would fail without normalize.ts)
  { q: 'four bedroom house in Pasadena', cat: 'spelled-num', expect: { city: 'Pasadena', beds: 4 }, escalate: false },
  { q: '在 Irvine 找３居室', cat: 'fullwidth', expect: { city: 'Irvine', beds: 3 }, escalate: false },
  { q: '3 bd w/ pool in Irvine', cat: 'synonym', expect: { city: 'Irvine', beds: 3, pool: true }, escalate: false },
  { q: 'townhome in Irvine under 1.2m', cat: 'synonym', expect: { city: 'Irvine', propertyType: 'townhouse', maxPrice: 1_200_000 }, escalate: false },
  // missing required slot -> escalate (and clarify if no LLM)
  { q: '3 bedroom under 1M', cat: 'missing-city', expect: { beds: 3, maxPrice: 1_000_000 }, escalate: true },
  { q: 'condo with a pool', cat: 'missing-city', expect: { propertyType: 'condo', pool: true }, escalate: true },
  // fuzzy / under-parsed -> escalate
  { q: 'a cozy family home near the beach around a million', cat: 'fuzzy', escalate: true },
  { q: 'modern place with an ocean view in Irvine', cat: 'fuzzy-soft', expect: { city: 'Irvine' }, escalate: true },
  // edge / control
  { q: 'hello', cat: 'noise', escalate: true },
];

const spyLLM = (): LLMClient & { called: boolean } => {
  const c = { called: false, available: true, async parseFilters() { c.called = true; return {}; } };
  return c as LLMClient & { called: boolean };
};

function fieldsMatch(got: SearchFilter, exp: Partial<SearchFilter>): boolean {
  return Object.entries(exp).every(([k, v]) => (got as Record<string, unknown>)[k] === v);
}

(async () => {
  let extractOK = 0, extractTotal = 0, escalateOK = 0;
  const fails: string[] = [];

  for (const c of CASES) {
    const llm = spyLLM();
    const r = await parseQuery(c.q, { llm });

    if (c.expect) {
      extractTotal++;
      if (fieldsMatch(r.filter, c.expect)) extractOK++;
      else fails.push(`extract  [${c.cat}] "${c.q}" -> ${JSON.stringify(r.filter)} (want ${JSON.stringify(c.expect)})`);
    }
    if (c.escalate !== undefined) {
      if (llm.called === c.escalate) escalateOK++;
      else fails.push(`escalate [${c.cat}] "${c.q}" -> called=${llm.called} (want ${c.escalate})`);
    }
  }

  console.log(`Extraction:  ${extractOK}/${extractTotal} fields-correct`);
  console.log(`Escalation:  ${escalateOK}/${CASES.filter((c) => c.escalate !== undefined).length} decisions-correct`);
  if (fails.length) {
    console.log('\nMisses:');
    for (const f of fails) console.log('  -', f);
  } else {
    console.log('\nAll eval cases passed ✓');
  }
  // regression gate
  if (extractOK < extractTotal || escalateOK < CASES.filter((c) => c.escalate !== undefined).length) {
    process.exitCode = 1;
  }
})();
