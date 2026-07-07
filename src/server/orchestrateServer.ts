/**
 * Orchestrator HTTP server (inbound wiring, Week 12+).
 *
 * Exposes our deterministic orchestrator over HTTP so the OpenClaw agent (acting as
 * a thin front door) can forward each inbound message here via a registered tool.
 * The real work stays in our system; OpenClaw just relays.
 *
 *   npm run serve         # listens on ORCH_PORT (default 8100)
 *   curl -s localhost:8100/orchestrate -d '{"userId":"+1","message":"Irvine 行情"}'
 */
import { createServer } from 'node:http';
import { orchestrate } from '../orchestrator/orchestrate.js';
import { logger } from '../logger.js';

const PORT = Number(process.env.ORCH_PORT ?? 8100);

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
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }
  if (req.method === 'POST' && req.url === '/orchestrate') {
    try {
      const { userId, message } = JSON.parse(await readBody(req) || '{}');
      if (!message) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'message required' }));
        return;
      }
      const result = await orchestrate(String(userId ?? 'anon'), String(message));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    } catch (err) {
      logger.error('orchestrate server error', { error: String(err) });
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'internal error' }));
    }
    return;
  }
  res.writeHead(404); res.end();
});

server.listen(PORT, () => logger.info('orchestrate server listening', { port: PORT }));
