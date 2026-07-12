/**
 * Orchestrator HTTP server (inbound wiring, Week 12+).
 *
 * Exposes our deterministic orchestrator over HTTP so the OpenClaw agent (acting as
 * a thin front door) can forward each inbound message here via a registered tool.
 * The real work stays in our system; OpenClaw just relays.
 *
 * This is the live boundary where OpenClaw hands off to us, so the comms guardrails
 * live HERE (not only in the offline handler.ts): per-user rate limiting, idempotency
 * (a redelivered request returns the cached reply instead of re-running), and 乙
 * degradation (errors become a friendly reply, never a raw 500 to the user).
 *
 *   npm run serve         # listens on ORCH_PORT (default 8100)
 *   curl -s localhost:8100/orchestrate -d '{"userId":"+1","message":"Irvine 行情"}'
 */
import 'dotenv/config';
import { createServer } from 'node:http';
import { orchestrate } from '../orchestrator/orchestrate.js';
import { RateLimiter } from '../whatsapp/handler.js';
import { logger } from '../logger.js';

const PORT = Number(process.env.ORCH_PORT ?? 8100);
const JSON_HEADERS = { 'Content-Type': 'application/json' };
// Shared secret with the OpenClaw plugin. When set, only callers presenting it may
// invoke /orchestrate — so a random local process can't forge a request with a spoofed
// userId (which would bypass OpenClaw's sender authentication and break 丙).
const AUTH_TOKEN = (process.env.ORCH_TOKEN ?? '').trim();

// Protects OUR resources (DB/LLM/retrieval) — 20 requests/min per user.
const limiter = new RateLimiter(20, 60_000);

// Idempotency: a redelivered request (same id, or same user+text within the window)
// returns the cached reply instead of re-running the whole pipeline.
const IDEMPOTENCY_TTL = 30_000;
const replyCache = new Map<string, { reply: string; at: number }>();
function cachedReply(key: string): string | null {
  const e = replyCache.get(key);
  if (!e) return null;
  if (Date.now() - e.at > IDEMPOTENCY_TTL) { replyCache.delete(key); return null; }
  return e.reply;
}
function cacheReply(key: string, reply: string): void {
  replyCache.set(key, { reply, at: Date.now() });
  if (replyCache.size > 1000) {
    const cutoff = Date.now() - IDEMPOTENCY_TTL;
    for (const [k, v] of replyCache) if (v.at < cutoff) replyCache.delete(k);
  }
}

function readBody(req: import('node:http').IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => { data += c; if (data.length > 1_000_000) req.destroy(); });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

const server = createServer(async (req, res) => {
  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, JSON_HEADERS);
    res.end(JSON.stringify({ ok: true }));
    return;
  }
  if (req.method === 'POST' && req.url === '/orchestrate') {
    // auth: reject anything not carrying the shared token (only the plugin has it)
    if (AUTH_TOKEN && req.headers.authorization !== `Bearer ${AUTH_TOKEN}`) {
      logger.warn('orchestrate unauthorized request rejected');
      res.writeHead(401, JSON_HEADERS);
      res.end(JSON.stringify({ error: 'unauthorized' }));
      return;
    }
    let userId = 'anon';
    try {
      const body = JSON.parse(await readBody(req) || '{}');
      const message = body.message;
      userId = String(body.userId ?? 'anon');
      const id = body.id ? String(body.id) : null;
      if (!message) {
        res.writeHead(400, JSON_HEADERS);
        res.end(JSON.stringify({ error: 'message required' }));
        return;
      }

      // rate limit (protect our resources) — 200 so the agent relays the friendly note
      if (!limiter.allow(userId)) {
        logger.info('orchestrate rate limited', { userId });
        res.writeHead(200, JSON_HEADERS);
        res.end(JSON.stringify({ reply: "You're sending messages very fast — give me a moment 🙏", rateLimited: true }));
        return;
      }

      // idempotency: serve a redelivered request from cache, don't re-run the pipeline
      const key = id ?? `${userId}|${message}`;
      const cached = cachedReply(key);
      if (cached !== null) {
        logger.info('orchestrate duplicate served from cache', { userId });
        res.writeHead(200, JSON_HEADERS);
        res.end(JSON.stringify({ reply: cached, duplicate: true }));
        return;
      }

      const result = await orchestrate(userId, String(message));
      cacheReply(key, result.reply);
      res.writeHead(200, JSON_HEADERS);
      res.end(JSON.stringify(result));
    } catch (err) {
      // 乙: degrade to a friendly reply (200), never a raw 500 to the user
      logger.error('orchestrate server error', { userId, error: String(err) });
      res.writeHead(200, JSON_HEADERS);
      res.end(JSON.stringify({ reply: 'Sorry — I hit a problem handling that. Please try again.', error: true }));
    }
    return;
  }
  res.writeHead(404); res.end();
});

// bind to loopback only — not reachable from other machines on the network
server.listen(PORT, '127.0.0.1', () =>
  logger.info('orchestrate server listening', { port: PORT, host: '127.0.0.1', auth: AUTH_TOKEN ? 'token required' : 'OPEN — set ORCH_TOKEN' }),
);
