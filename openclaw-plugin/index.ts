/**
 * IDX OpenClaw plugin — the "thin front door".
 *
 * Registers ONE tool (`ask_idx_assistant`) that forwards a user message to our
 * deterministic orchestrator (HTTP, ORCH_URL) and returns its reply. A
 * before_prompt_build guidance tells the channel agent to always route real-estate
 * requests through this tool rather than answering itself — so the OpenClaw agent
 * is just a relay and our system does the actual work.
 */
import { Type } from '@sinclair/typebox';

const ORCH_URL = process.env.ORCH_URL ?? 'http://localhost:8100';

const idxTool = {
  name: 'ask_idx_assistant',
  label: 'IDX Real-Estate Assistant',
  description:
    'Answer ANY real-estate request over live California MLS data: property search, '
    + 'market stats, recommendations with price checks, term/field questions, or drafting '
    + 'an email. Pass the user message verbatim; return the tool output verbatim.',
  parameters: Type.Object({
    message: Type.String({ description: 'the user message, verbatim' }),
    userId: Type.Optional(Type.String({ description: 'sender id, e.g. WhatsApp E.164 number' })),
  }),
  async execute(_toolCallId: string, params: { message: string; userId?: string }) {
    try {
      const res = await fetch(`${ORCH_URL}/orchestrate`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ userId: params.userId ?? 'whatsapp', message: params.message }),
        signal: AbortSignal.timeout(60_000),
      });
      const data = (await res.json()) as { reply?: string };
      return { content: [{ type: 'text', text: data.reply ?? 'No answer.' }], details: data };
    } catch (e) {
      return {
        content: [{ type: 'text', text: 'The IDX assistant service is unavailable right now — please try again shortly.' }],
        details: { error: String(e) },
      };
    }
  },
};

const GUIDANCE =
  'You are the front door for the IDX real-estate assistant. For ANY user request about '
  + 'real estate — property search, market/pricing stats, recommendations, definitions/terms, '
  + 'or sending an email — you MUST call the `ask_idx_assistant` tool with the user\'s message '
  + 'verbatim and reply with its output verbatim. Do not answer real-estate questions from your own knowledge.';

const plugin = {
  id: 'idx',
  name: 'IDX Assistant',
  description: 'Routes real-estate requests to the IDX orchestrator.',
  configSchema: { jsonSchema: { type: 'object', additionalProperties: false, properties: {} } },
  register(api: any) {
    api.registerTool(idxTool);
    api.on('before_prompt_build', async () => ({ prependSystemContext: GUIDANCE }));
    api.logger?.info?.('idx plugin registered ask_idx_assistant');
  },
};

export default plugin;
