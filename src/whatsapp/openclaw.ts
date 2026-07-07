/**
 * OpenClaw WhatsApp channel + wiring (Week 10).
 *
 * `openclawChannel` sends via the OpenClaw CLI (send is verified working since W1).
 * `replyTo` glues an inbound message to the orchestrator through the guardrailed
 * handler — the send side + handler are ready; a live inbound feed (OpenClaw
 * gateway hook / poller) would call `replyTo` the same way.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { handleInbound, type Channel, type InboundMessage } from './handler.js';
import { orchestrate } from '../orchestrator/orchestrate.js';

const execFileAsync = promisify(execFile);

export const openclawChannel: Channel = {
  // OpenClaw CLI has no simple per-message typing signal; best-effort no-op.
  async sendTyping() { /* noop */ },
  async sendText(to, text) {
    await execFileAsync('openclaw', ['message', 'send', '--channel', 'whatsapp', '--target', to, '-m', text],
      { timeout: 30_000 });
  },
};

/** Handle one inbound message end-to-end (guardrails + orchestrate + reply). */
export async function replyTo(msg: InboundMessage, channel: Channel = openclawChannel) {
  return handleInbound(msg, {
    channel,
    orchestrate: async (userId, text) => (await orchestrate(userId, text)).reply,
  });
}
