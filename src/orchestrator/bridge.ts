/**
 * Bridge from the TS orchestrator to the warm Python retrieval service (Week 10).
 * HTTP to a hot FastAPI process (model preloaded) instead of spawning python per
 * call — ~7ms vs ~1.2s. Injectable so tests use a fake and route logic is verified
 * without the service/Qdrant.
 */
import { config } from '../config.js';
import { withResilience, CircuitBreaker } from '../resilience/resilience.js';

export interface IntentGuess {
  skill: string;
  score: number;
  /** top1 - top2 score gap. Low margin = the query is "half-like" several intents,
   * a signal of an out-of-domain / ambiguous input even when top1 isn't that low. */
  margin: number;
}

export interface SemanticSearchParams {
  text: string;
  city?: string | null;
  max_price?: number | null;
  min_price?: number | null;
  min_beds?: number | null;
  pool?: boolean;
  ptype?: string | null;
  k?: number;
}

export interface SemanticListing {
  score: number;
  listing_id?: number;
  mls?: string | null;
  address?: string | null;
  city?: string | null;
  type?: string | null;
  beds?: number | null;
  baths?: number | null;
  sqft?: number | null;
  price?: number | null;
  pool?: boolean;
}

export interface PythonBridge {
  classify(message: string): Promise<IntentGuess>;
  rag(question: string): Promise<string>;
  recommend(listingId: number): Promise<string>;
  validate(listing: { city: string | null; sqft: number | null; price: number | null }): Promise<string>;
  search(params: SemanticSearchParams): Promise<SemanticListing[]>;
}

// one circuit breaker for the whole retrieval service (all paths share the same process)
const retrievalBreaker = new CircuitBreaker(5, 15_000);

async function post<T>(path: string, body: unknown): Promise<T> {
  return withResilience(async () => {
    const res = await fetch(`${config.retrieval.url}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(20_000),
    });
    if (!res.ok) throw new Error(`retrieval ${path} -> HTTP ${res.status}`);
    return res.json() as Promise<T>;
    // no fallback here — degradation is the caller's job (e.g. search skill falls back to MySQL);
    // bridge just adds timeout + retry + breaker so a slow/down service doesn't hang or hammer.
  }, { name: `retrieval${path}`, timeoutMs: 20_000, retries: 2, breaker: retrievalBreaker });
}

/** Real bridge: HTTP to the warm retrieval service. */
export const pythonBridge: PythonBridge = {
  classify: async (message) => {
    const r = await post<{ skill: string; score: number; ranked: Array<[string, number]> }>('/classify', { message });
    const margin = r.ranked.length >= 2 ? r.ranked[0]![1] - r.ranked[1]![1] : r.score;
    return { skill: r.skill, score: r.score, margin };
  },
  rag: async (question) => {
    const r = await post<{ answer: string; sources: string[] }>('/rag', { question });
    return `${r.answer}\n\nSources: ${r.sources.join('; ')}`;
  },
  recommend: async (listingId) => {
    const r = await post<{ reply?: string; error?: string }>('/recommend', { listing_id: listingId });
    return r.reply ?? r.error ?? 'No recommendations found.';
  },
  validate: async (l) => JSON.stringify(await post('/validate', l)),
  search: async (params) => {
    const r = await post<{ results: SemanticListing[] }>('/search', params);
    return r.results ?? [];
  },
};
