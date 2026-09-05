/**
 * Multi-query retrieval (query rewrite to lift recall).
 *
 * Design: don't pick "the best rewrite" — generate variants, retrieve each, and FUSE by rank
 * (RRF), the same fusion already used for dense+BM25. Any rewrite method (multi-query variants,
 * a future HyDE doc) is just "one more list" into the same RRF, so items ranked high across
 * variants win. LLM unavailable -> degrade to the original query only (乙).
 *
 * Opt-in (a flag) because N variants = N retrieval calls (latency/cost); enable only once eval
 * shows recall actually improves — same discipline as tuning RRF's k.
 */
import type { LLMClient } from '../llm/client.js';

/** Generate up to N phrasings (incl. the original) of a real-estate semantic query. */
export async function expandQuery(query: string, llm?: LLMClient, n = 3): Promise<string[]> {
  const original = query.trim();
  if (!original || !llm?.chatWithTools || n <= 1) return original ? [original] : [];
  try {
    const turn = await llm.chatWithTools([
      { role: 'system', content: `Rewrite the real-estate search phrase into ${n - 1} alternative phrasings that mean the SAME thing `
        + 'but use different words/angles (synonyms, related styles/features). Return ONLY a JSON array of strings.' },
      { role: 'user', content: original },
    ], []);
    const m = turn.content.match(/\[[\s\S]*\]/);
    const arr = m ? (JSON.parse(m[0]) as unknown[]) : [];
    const variants = arr.filter((s): s is string => typeof s === 'string' && s.trim().length > 0)
      .map((s) => s.trim()).slice(0, n - 1);
    return [...new Set([original, ...variants])];
  } catch {
    return [original];   // LLM down -> degrade to the original query only
  }
}

/**
 * Reciprocal Rank Fusion of multiple ranked lists. Fuses by RANK (no score normalization
 * needed) so it works across heterogeneous rewrite methods. k dampens the top-rank weight.
 */
export function rrfFuse<T>(lists: T[][], idOf: (x: T) => string | number, k = 60): T[] {
  const score = new Map<string | number, number>();
  const item = new Map<string | number, T>();
  for (const list of lists) {
    list.forEach((x, rank) => {
      const id = idOf(x);
      score.set(id, (score.get(id) ?? 0) + 1 / (k + rank + 1));
      if (!item.has(id)) item.set(id, x);
    });
  }
  return [...score.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => item.get(id)!);
}
