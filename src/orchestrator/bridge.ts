/**
 * Bridge from the TS orchestrator to the warm Python retrieval service (Week 10).
 * HTTP to a hot FastAPI process (model preloaded) instead of spawning python per
 * call — ~7ms vs ~1.2s. Injectable so tests use a fake and route logic is verified
 * without the service/Qdrant.
 */
import { config } from '../config.js';

export interface IntentGuess {
  skill: string;
  score: number;
}

export interface PythonBridge {
  classify(message: string): Promise<IntentGuess>;
  rag(question: string): Promise<string>;
  recommend(listingId: number): Promise<string>;
  validate(listing: { city: string | null; sqft: number | null; price: number | null }): Promise<string>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${config.retrieval.url}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(60_000),
  });
  if (!res.ok) throw new Error(`retrieval ${path} -> HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

/** Real bridge: HTTP to the warm retrieval service. */
export const pythonBridge: PythonBridge = {
  classify: (message) => post<IntentGuess>('/classify', { message }),
  rag: async (question) => {
    const r = await post<{ answer: string; sources: string[] }>('/rag', { question });
    return `${r.answer}\n\nSources: ${r.sources.join('; ')}`;
  },
  recommend: async (listingId) => {
    const r = await post<{ error?: string }>('/recommend', { listing_id: listingId });
    return r.error ?? JSON.stringify(r);
  },
  validate: async (l) => JSON.stringify(await post('/validate', l)),
};
