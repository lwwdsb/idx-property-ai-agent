/**
 * IDX assistant as an MCP server (stdio). OpenClaw's agent (thin front door)
 * connects to this and gets one tool that forwards a user message to our
 * deterministic orchestrator.
 *
 * Design: this MCP server is a THIN forwarder to the orchestrate HTTP service
 * (:8100) with the shared token — mirroring the native plugin — so BOTH integration
 * paths hit the same boundary and get the same guardrails (token auth, rate limit,
 * idempotency, 乙 degradation). One guardrail implementation, not two.
 *
 * Wired into OpenClaw via CLI:
 *   openclaw mcp add idx --command <abs>/node_modules/.bin/tsx \
 *     --arg <abs>/src/mcp/idxServer.ts --cwd <abs>
 * (reads ORCH_URL / ORCH_TOKEN from .env via loadenv)
 */
import './loadenv.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const ORCH_URL = process.env.ORCH_URL ?? 'http://localhost:8100';
const ORCH_TOKEN = (process.env.ORCH_TOKEN ?? '').trim();

const server = new McpServer({ name: 'idx-assistant', version: '1.0.0' });

server.tool(
  'ask_idx_assistant',
  'Real-estate assistant over live California MLS data. Use this for ANY user request about: '
    + 'property search, market stats, recommendations with price checks, term/field questions, '
    + 'or drafting an email. Pass the user message verbatim; return the tool output verbatim.',
  { message: z.string().describe('the user message, verbatim'),
    userId: z.string().optional().describe('sender id (e.g. WhatsApp E.164)') },
  async ({ message, userId }) => {
    try {
      const res = await fetch(`${ORCH_URL}/orchestrate`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          ...(ORCH_TOKEN ? { authorization: `Bearer ${ORCH_TOKEN}` } : {}),
        },
        body: JSON.stringify({ userId: userId ?? 'whatsapp', message }),
        signal: AbortSignal.timeout(60_000),
      });
      const data = (await res.json()) as { reply?: string };
      return { content: [{ type: 'text', text: data.reply ?? 'No answer.' }] };
    } catch (e) {
      return { content: [{ type: 'text', text: 'The IDX assistant service is unavailable right now — please try again shortly.' }], isError: false };
    }
  },
);

await server.connect(new StdioServerTransport());
