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
      // 确定性路径【没有跨会话记忆】(2026-09-28 起)。所有槽位值只来自当轮解析或本会话前几轮的
      // 携带,因此 filter 里的每一个值都是用户在这段可见的对话里说过的 —— 不会出现"这个值是三周前
      // 学来的还是你刚说的"这种下游无法区分的歧义。跨会话偏好只保留在 auto 路径(语义记忆的声明式
      // slots + 散文注入),那里值的来源在 trace 的 effectiveFilter 里可查。详见 memory/profile.ts。
      const result = await orchestrate(userId, text, { registry, draftStore, llm });
      return result.reply;
    },
  });
}
