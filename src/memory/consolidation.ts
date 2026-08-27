/**
 * Periodic memory-consolidation sub-agent (#53).
 *
 * A ReAct loop with a DELIBERATELY ISOLATED capability domain — it can ONLY read this
 * user's recent sessions and write to this user's profile. It has NO business tools, so
 * it structurally cannot search, email, or take any outward action.
 *
 * TWO isolation mechanisms (see the exported constants/closure):
 *  1. TOOL-LIST ISOLATION — `MEMORY_TOOLS` is a separate whitelist, disjoint from the
 *     business SkillRegistry's tools. The loop is handed ONLY `MEMORY_TOOLS`; there is no
 *     path from here to `search`/`email`/etc.
 *  2. CONTEXT ISOLATION — `userId` is bound in the executor CLOSURE, NOT exposed as a tool
 *     parameter. The model cannot address another user's data; reads are limited to this
 *     user's most-recent N runs (user + recency bounded).
 *
 * Triggered periodically (cron / CLI), low-frequency. Output: new semantic/episodic
 * memories via addMemory (a human can review the rendered profile.md).
 */
import type { LLMClient, ChatMessage, ToolSpec } from '../llm/client.js';
import type { AgentRunStore } from '../agent/auto/runStore.js';
import { logger } from '../logger.js';
import { loadProfile, saveProfile, addMemory, forgetMemory, compactMemories, memoryIndex, type MemoryType } from './profile.js';

/** The memory capability domain — a whitelist DISJOINT from the business tool set. */
export const MEMORY_TOOLS: ToolSpec[] = [
  {
    name: 'read_recent_sessions',
    description: "Read THIS user's recent conversation snippets to find durable preferences/events.",
    parameters: { type: 'object', properties: { limit: { type: 'number', description: 'how many recent runs (max 10)' } } },
  },
  {
    name: 'list_memories',
    description: 'List memories already stored (name + description + type), to avoid duplicates.',
    parameters: { type: 'object', properties: {} },
  },
  {
    name: 'add_memory',
    description: 'Store a durable memory: a generalized preference (semantic) or a notable event (episodic). Skip one-off/transient details.',
    parameters: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'short kebab-case id' },
        description: { type: 'string', description: 'one line, used later to decide relevance' },
        type: { type: 'string', enum: ['semantic', 'episodic'] },
        content: { type: 'string' },
        salience: { type: 'number', description: '0-1 importance' },
        mergedFrom: { type: 'array', items: { type: 'string' }, description: 'names of memories this consolidates (promotion/merge lineage)' },
      },
      required: ['name', 'description', 'type', 'content'],
    },
  },
  {
    name: 'forget_memory',
    description: 'Delete an outdated/contradicted/redundant memory by name (e.g. after promoting episodics into a semantic).',
    parameters: { type: 'object', properties: { name: { type: 'string' } }, required: ['name'] },
  },
];

const SYSTEM = [
  'You are a MEMORY CONSOLIDATION assistant. Your ONLY job: read the user\'s recent sessions and',
  'distill DURABLE long-term memories about THIS user — generalized preferences (semantic) and',
  'notable events (episodic). You have NO other capabilities: you cannot search, email, or act.',
  'Process: read_recent_sessions -> list_memories (avoid duplicates) -> add_memory for each new,',
  'durable memory (skip one-off/transient details; keep it concise).',
  'CONSOLIDATE when useful: if several episodics point to one durable preference, add_memory a',
  'semantic (list them in mergedFrom) and forget_memory the redundant episodics; forget_memory',
  'anything outdated or contradicted. Reply with a short summary when done.',
].join('\n');

/**
 * CONTEXT ISOLATION: userId is captured here — it is NOT a tool argument, so the model
 * cannot address a different user. Tools touch only this user's runs (read) + profile (write).
 */
function makeExecutor(userId: string, runStore: AgentRunStore) {
  return async function exec(name: string, args: Record<string, unknown>): Promise<string> {
    if (name === 'read_recent_sessions') {
      const limit = Math.min(10, Math.max(1, Number(args.limit) || 5));
      const runs = await runStore.recentForUser(userId, limit);
      if (!runs.length) return 'No recent sessions for this user.';
      return runs.map((r) => {
        const turns = r.state.messages
          .filter((m) => m.role === 'user' || m.role === 'assistant')
          .map((m) => `${m.role}: ${String(m.content).slice(0, 200)}`).join('\n');
        return `--- run #${r.id} (${new Date(r.createdAt).toISOString().slice(0, 10)}) ---\n${turns}`;
      }).join('\n\n').slice(0, 4000);
    }
    if (name === 'list_memories') {
      return JSON.stringify(memoryIndex(loadProfile(userId)));
    }
    if (name === 'add_memory') {
      const profile = loadProfile(userId);
      addMemory(profile, {
        name: String(args.name),
        description: String(args.description),
        type: (args.type === 'episodic' ? 'episodic' : 'semantic') as MemoryType,
        content: String(args.content),
        salience: typeof args.salience === 'number' ? args.salience : undefined,
        mergedFrom: Array.isArray(args.mergedFrom) ? args.mergedFrom.filter((x): x is string => typeof x === 'string') : undefined,
      });
      saveProfile(profile);
      return `stored ${args.type} memory "${String(args.name)}"`;
    }
    if (name === 'forget_memory') {
      const profile = loadProfile(userId);
      const ok = forgetMemory(profile, String(args.name));
      if (ok) saveProfile(profile);
      return ok ? `forgot memory "${String(args.name)}"` : `no memory named "${String(args.name)}"`;
    }
    return `error: unknown tool "${name}" — this agent only has memory tools.`;
  };
}

export interface ConsolidateDeps { llm: LLMClient; runStore: AgentRunStore; maxSteps?: number; }

/** Run one consolidation pass for a user. Returns the agent's final summary. */
export async function runConsolidation(userId: string, deps: ConsolidateDeps): Promise<string> {
  const { llm, runStore } = deps;
  if (!llm.chatWithTools) throw new Error('consolidation needs an LLM with tool-calling');
  const exec = makeExecutor(userId, runStore);
  const budget = deps.maxSteps ?? 6;
  const messages: ChatMessage[] = [
    { role: 'system', content: SYSTEM },
    { role: 'user', content: `Consolidate durable long-term memory for this user from their recent sessions.` },
  ];
  for (let step = 0; step < budget; step++) {
    const turn = await llm.chatWithTools(messages, MEMORY_TOOLS);   // ONLY the isolated memory tools
    if (!turn.toolCalls.length) {
      // capacity trigger: code-level backstop bounds size after the LLM's promotion/merge/forget
      const profile = loadProfile(userId);
      const { removed } = compactMemories(profile);
      if (removed.length) saveProfile(profile);
      logger.info('memory consolidation done', { userId, steps: step, compacted: removed.length });
      return turn.content;
    }
    messages.push(turn.raw);
    for (const call of turn.toolCalls) {
      const observation = await exec(call.name, call.arguments);
      messages.push({ role: 'tool', tool_call_id: call.id, content: observation });
    }
  }
  logger.warn('memory consolidation hit step budget', { userId, budget });
  return 'consolidation reached step budget';
}
