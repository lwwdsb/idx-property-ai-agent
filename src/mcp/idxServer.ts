/**
 * IDX assistant as an MCP server (stdio). OpenClaw's agent (thin front door)
 * connects to this and gets one tool that forwards a user message to our
 * deterministic orchestrator. The real work stays in our system.
 *
 * Wired into OpenClaw via config:
 *   mcp.servers.idx = { command: "npx", args: ["tsx", "<abs>/src/mcp/idxServer.ts"] }
 *   tools.sandbox.tools.alsoAllow = ["bundle-mcp"]
 */
import './loadenv.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { orchestrate } from '../orchestrator/orchestrate.js';

const server = new McpServer({ name: 'idx-assistant', version: '1.0.0' });

server.tool(
  'ask_idx_assistant',
  'Real-estate assistant over live California MLS data. Use this for ANY user request about: '
    + 'property search, market stats, recommendations with price checks, term/field questions, '
    + 'or drafting an email. Pass the user message verbatim; return the tool output verbatim.',
  { message: z.string().describe('the user message, verbatim'),
    userId: z.string().optional().describe('sender id (e.g. WhatsApp E.164)') },
  async ({ message, userId }) => {
    const r = await orchestrate(userId ?? 'whatsapp', message);
    return { content: [{ type: 'text', text: r.reply }] };
  },
);

await server.connect(new StdioServerTransport());
