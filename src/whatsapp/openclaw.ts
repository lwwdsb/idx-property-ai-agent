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
import { buildRegistry } from '../orchestrator/skills.js';
import { pythonBridge } from '../orchestrator/bridge.js';
import { getLLMClient } from '../llm/client.js';
import { MySqlDraftStore } from '../email/drafts.js';
import { MySqlAgentRunStore } from '../agent/auto/runStore.js';
import { handleAgentMessage } from '../agent/auto/entry.js';
import { loadProfile, saveProfile, learnFromFilter, preferredFilter } from '../memory/profile.js';

const execFileAsync = promisify(execFile);

// Shared, persistent singletons so agent runs + their drafts survive across messages
// (and restarts) — required for the async WhatsApp HITL suspend/resume.
const draftStore = new MySqlDraftStore();
const runStore = new MySqlAgentRunStore();
const llm = getLLMClient();
const registry = buildRegistry(pythonBridge, draftStore);

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
    orchestrate: async (userId, text) => {
      // Auto/agent mode first: `/auto <task>` runs, and `approve/cancel #N` of a
      // SUSPENDED run's draft resumes it. Non-agent messages return null -> orchestrate.
      const agentReply = await handleAgentMessage(userId, text, { registry, llm, draftStore, runStore });
      if (agentReply !== null) return agentReply;
      // Long-term memory: soft-default missing fields from prefs, and learn from what the user gave.
      const profile = loadProfile(userId);
      const result = await orchestrate(userId, text, {
        registry, draftStore, llm,
        filterDefaults: preferredFilter(profile),
        onFilter: (uf) => {
          if (Object.values(uf).some((v) => v != null)) { learnFromFilter(profile, uf); saveProfile(profile); }
        },
      });
      return result.reply;
    },
  });
}
