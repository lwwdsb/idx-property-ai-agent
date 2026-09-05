/**
 * Config loader + fail-fast validation.
 * Loads .env, validates required keys at startup, and exposes a typed config.
 * Missing required config throws immediately so the app never starts half-configured.
 * (TS mirror of the Week 0 scripts/check_env.py philosophy, for the live path.)
 */
import 'dotenv/config';

function required(name: string): string {
  const v = process.env[name];
  if (!v || v.trim() === '') {
    throw new Error(`Missing required env var: ${name} (copy .env.example -> .env)`);
  }
  return v.trim();
}

function optional(name: string, fallback = ''): string {
  return (process.env[name] ?? fallback).trim();
}

export interface Config {
  db: {
    host: string;
    port: number;
    user: string;
    password: string;
    database: string;
  };
  openai: {
    /** May be empty until Week 6 (embeddings). Guard before use. */
    apiKey: string;
    embeddingModel: string;
    chatModel: string;
  };
  /**
   * Provider-agnostic chat LLM (OpenAI-compatible /chat/completions).
   * Swap provider by changing baseUrl + apiKey + model in .env — e.g. point at
   * MiniMax or OpenAI without touching code. Empty apiKey => LLM path disabled
   * (callers fall back to the regex fast-path).
   */
  llm: {
    baseUrl: string;
    apiKey: string;
    model: string;
  };
  whatsapp: {
    sessionName: string;
  };
  /** Warm Python retrieval service (Week 10). */
  retrieval: {
    url: string;
    /** Multi-query rewrite: expand the semantic query into variants + RRF-fuse the hits.
     * Opt-in (N variants = N retrieval calls); enable only once eval shows recall improves. */
    multiQuery: boolean;
    /** HyDE rewrite for knowledge RAG: retrieve on an LLM hypothetical passage (+blend).
     * Opt-in (extra LLM call); eval showed +0.10 hit@k but the refusal gate must re-tune up. */
    hyde: boolean;
  };
  /** Google Maps (commute/proximity filtering). Empty key => proximity filtering
   * silently skipped (乙: search still returns results, just without commute ranking). */
  maps: {
    apiKey: string;
  };
  /** Email agent (Week 11). Outbound always requires human approval. */
  email: {
    from: string;
    smtpHost: string;
    smtpPort: number;
    user: string;
    password: string;
    /** Operator numbers allowed to draft/approve emails. */
    allowlist: string[];
  };
}

export const config: Config = {
  db: {
    host: required('DB_HOST'),
    port: Number(required('DB_PORT')),
    user: required('DB_USER'),
    password: required('DB_PASSWORD'),
    database: required('DB_NAME'),
  },
  openai: {
    apiKey: optional('OPENAI_API_KEY'),
    embeddingModel: optional('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    chatModel: optional('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
  },
  llm: {
    baseUrl: optional('LLM_BASE_URL', 'https://api.openai.com/v1'),
    apiKey: optional('LLM_API_KEY') || optional('OPENAI_API_KEY'),
    model: optional('LLM_MODEL', 'gpt-4o-mini'),
  },
  whatsapp: {
    sessionName: optional('WHATSAPP_SESSION_NAME', 'idx-assistant'),
  },
  retrieval: {
    url: optional('RETRIEVAL_URL', 'http://localhost:8099'),
    multiQuery: optional('RETRIEVAL_MULTI_QUERY', 'false') === 'true',
    hyde: optional('RAG_HYDE', 'false') === 'true',
  },
  maps: {
    apiKey: optional('GOOGLE_MAPS_API_KEY'),
  },
  email: {
    from: optional('EMAIL_FROM'),
    smtpHost: optional('EMAIL_SMTP_HOST'),
    smtpPort: Number(optional('EMAIL_SMTP_PORT', '587')),
    user: optional('EMAIL_USER') || optional('EMAIL_FROM'),
    password: optional('EMAIL_PASSWORD'),
    allowlist: optional('EMAIL_ALLOWLIST').split(',').map((s) => s.trim()).filter(Boolean),
  },
};

/** True only when real SMTP creds are present; otherwise sends run as dry-run. */
export function emailConfigured(): boolean {
  return Boolean(config.email.smtpHost && config.email.user && config.email.password);
}

/** Throws if a feature's config is missing. Call before using OpenAI (Week 6+). */
export function requireOpenAI(): string {
  if (!config.openai.apiKey) {
    throw new Error('OPENAI_API_KEY is not set — required for embeddings/chat (Week 6+).');
  }
  return config.openai.apiKey;
}
