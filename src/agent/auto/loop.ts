/**
 * Auto/agent mode — the ReAct loop (M2: persistent + interrupt/resume).
 *
 * A real autonomous loop (NOT the deterministic router): the LLM decides each step
 * whether to call another tool or finish, looking at accumulated observations. The loop
 * is just a `while` — "should I continue" is the model's call (it keeps emitting
 * tool_calls) until it returns a final answer, hits a guardrail, or reaches a human-in-
 * the-loop point.
 *
 * HITL (M2): when the agent drafts an email (an outbound action), the loop CHECKPOINTS
 * its whole state to the run store and SUSPENDS (returns `awaiting_approval`). The
 * process is released; later `approve <draftId>` sends the draft and calls
 * `resumeAgentRun`, which restores the state and continues to the final answer.
 *
 * Guardrails (runaway/cost, not correctness): step budget, identical-call dedupe,
 * per-tool call cap.
 */
import { logger } from '../../logger.js';
import type { LLMClient, ChatMessage, ToolSpec } from '../../llm/client.js';
import type { SkillRegistry } from '../../orchestrator/skill.js';
import { toolSpecs, executeTool, FIND_TOOLS_SPEC, findTools } from './tools.js';
import { freshMemory, recordStep, renderMemory, isEmpty, type WorkingMemory } from './memory.js';
import type { SearchFilter } from '../../search/filters.js';
import { MySqlAgentRunStore, type AgentRunStore, type AgentRunState, type AgentTraceStep } from './runStore.js';
import { groundFinal } from './grounding.js';
export type { AgentTraceStep };

const MAX_STEPS = 8;
const MAX_PER_TOOL = 3;      // soft cap: same tool called this many times -> nudge to finish

const SYSTEM = [
  'You are an autonomous real-estate assistant. Use the provided tools to FULLY complete',
  "the user's task. Call tools to gather facts (search listings, market stats, knowledge,",
  'recommendations) and to draft emails. Work step by step: after each tool result, decide',
  'whether to call another tool or to give the final answer.',
  'Rules:',
  '- Each tool call handles ONE sub-task; pass its self-contained request in "query".',
  '- Ground every claim in tool results — never invent listings, prices, or stats.',
  '- Emails are DRAFTED and require human approval; you never send directly.',
  '- When the task is done, reply in natural language WITHOUT calling any tool.',
].join('\n');

const PROGRESSIVE_HINT = '\nTools are NOT all loaded up front. Call find_tools(query) to '
  + 'discover and enable the tools you need before you can call them.';

/** Runtime metrics for measuring the agent system (all auto-recorded from the loop). */
export interface AgentMetrics {
  steps: number;               // loop iterations
  toolCalls: number;           // total tool invocations
  toolErrors: number;          // hallucinated/unknown tools or skill errors
  loopGuards: number;          // times a repeat / per-tool-cap guard fired (thrashing signal)
  llmCalls: number;            // model calls (cost proxy)
  groundingRewrites: number;   // grounding gate LLM-rewrite passes
  groundingStripped: number;   // grounding gate deterministic strips (rewrite couldn't fix)
  budgetExhausted: boolean;
  suspended: boolean;          // HITL suspend
  elapsedMs: number;           // wall-clock latency of this drive (for p50/p99)
}

export interface AgentResult {
  reply: string;
  trace: AgentTraceStep[];
  steps: number;
  stopReason: 'final' | 'budget' | 'awaiting_approval';
  memory: WorkingMemory;
  runId?: number;
  pendingDraftId?: number;
  metrics: AgentMetrics;
}

export interface RunAgentOptions {
  userId: string;
  registry: SkillRegistry;
  llm: LLMClient;
  maxSteps?: number;
  /** Progressive tool loading: expose only find_tools up front; the agent discovers +
   * enables real tools on demand. Default false (all tools exposed). */
  progressive?: boolean;
  /** Persistence for suspend/resume. Defaults to MySQL. */
  store?: AgentRunStore;
  /** Long-term memory (soft): profile summary appended to the system prompt. */
  profileHint?: string;
  /** Long-term memory (soft): seed the slot working-memory constraints from preferences. */
  seedFilter?: Partial<SearchFilter>;
}

interface DriveDeps {
  userId: string;
  registry: SkillRegistry;
  llm: LLMClient;
  store: AgentRunStore;
  runId: number;
  budget: number;
}

/** Rebuild the enabled ToolSpecs from persisted names (so resume restores the tool set). */
function rebuildActive(registry: SkillRegistry, names: string[], progressive: boolean): ToolSpec[] {
  const all = progressive ? [FIND_TOOLS_SPEC, ...toolSpecs(registry)] : toolSpecs(registry);
  return all.filter((t) => names.includes(t.name));
}

/** Core loop over a (possibly resumed) state. Mutates state in place and checkpoints it. */
async function driveLoop(state: AgentRunState, deps: DriveDeps): Promise<AgentResult> {
  const { userId, registry, llm, store, runId, budget } = deps;
  if (!llm.chatWithTools) throw new Error('auto mode needs an LLM with tool-calling (chatWithTools)');
  const activeTools = rebuildActive(registry, state.activeToolNames, state.progressive);
  const messages = state.messages;
  const mem = state.memory;
  const trace = state.trace;                     // persisted; accumulates across resumes
  const seen = new Set<string>();               // reset per drive (resume acceptable)
  const perTool = new Map<string, number>();
  let toolCalls = 0, toolErrors = 0, loopGuards = 0, llmCalls = 0;
  const t0 = Date.now();
  const withMemory = (): ChatMessage[] =>
    isEmpty(mem) ? messages : [...messages, { role: 'system', content: renderMemory(mem) }];
  const M = (extra: Partial<AgentMetrics> = {}): AgentMetrics => ({
    steps: state.step, toolCalls, toolErrors, loopGuards, llmCalls,
    groundingRewrites: 0, groundingStripped: 0, budgetExhausted: false, suspended: false,
    elapsedMs: Date.now() - t0, ...extra,
  });

  while (state.step < budget) {
    state.step++;
    const turn = await llm.chatWithTools!(withMemory(), activeTools);
    llmCalls++;

    if (!turn.toolCalls.length) {               // LLM chose to finish -> grounding gate -> done
      const g = await groundFinal(turn.content, messages, llm);
      llmCalls += g.calls;
      trace.push({ step: state.step, thought: g.reply });
      await store.save(runId, { state, status: 'done' });
      logger.info('auto agent final', { userId, runId, steps: state.step });
      return { reply: g.reply, trace, steps: state.step, stopReason: 'final', memory: mem, runId,
        metrics: M({ groundingRewrites: g.rewrites, groundingStripped: g.stripped }) };
    }

    messages.push(turn.raw);
    let pendingDraft: number | undefined;
    for (const call of turn.toolCalls) {
      toolCalls++;
      const sig = `${call.name}:${JSON.stringify(call.arguments)}`;
      const count = perTool.get(call.name) ?? 0;
      let observation: string;
      if (seen.has(sig)) {
        loopGuards++;
        observation = `error: you already made this exact call to "${call.name}". `
          + 'Use the previous result, try a different action, or finish.';
      } else if (count >= MAX_PER_TOOL) {
        loopGuards++;
        observation = `error: "${call.name}" has been called ${count} times already; it won't `
          + 'return anything new. Use the results you have or finish with a final answer.';
      } else if (call.name === 'find_tools') {
        const q = typeof call.arguments.query === 'string' ? call.arguments.query : '';
        const found = findTools(registry, q);
        for (const t of found) if (!state.activeToolNames.includes(t.name)) {
          activeTools.push(t);
          state.activeToolNames.push(t.name);   // persist so resume keeps them enabled
        }
        perTool.set(call.name, count + 1);
        observation = `Enabled ${found.length} tool(s): `
          + found.map((t) => `${t.name} — ${t.description}`).join(' | ') + '. You can now call them.';
      } else {
        seen.add(sig);
        perTool.set(call.name, count + 1);
        const res = await executeTool(registry, call.name, call.arguments, { userId, llm, memConstraints: mem.constraints });
        observation = res.observation;
        if (observation.startsWith('error')) toolErrors++;
        recordStep(mem, call.name, call.arguments, observation);
        if (res.draftId !== undefined) pendingDraft = res.draftId;   // outbound -> HITL interrupt
      }
      trace.push({ step: state.step, tool: call.name, args: call.arguments, observation: observation.slice(0, 500) });
      messages.push({ role: 'tool', tool_call_id: call.id, content: observation });
    }

    if (pendingDraft !== undefined) {
      // Human-in-the-loop: checkpoint the whole loop state and suspend until approval.
      await store.save(runId, { state, status: 'awaiting_approval', pendingDraftId: pendingDraft });
      logger.info('auto agent awaiting approval', { userId, runId, draftId: pendingDraft });
      return {
        reply: `⏸ I drafted email #${pendingDraft} and paused for your approval. `
          + `Reply "approve ${pendingDraft}" to send it and let me finish, or "cancel ${pendingDraft}".`,
        trace, steps: state.step, stopReason: 'awaiting_approval', memory: mem, runId, pendingDraftId: pendingDraft,
        metrics: M({ suspended: true }),
      };
    }
  }

  // budget exhausted -> force a best-effort summary with no tools
  const wrap = await llm.chatWithTools!(
    [...withMemory(), { role: 'user', content: 'Stop using tools now and summarize your findings for the user.' }],
    [],
  );
  llmCalls++;
  await store.save(runId, { state, status: 'done' });
  logger.warn('auto agent hit step budget', { userId, runId, budget });
  return {
    reply: wrap.content || 'I ran out of steps before finishing — could you narrow the task?',
    trace, steps: state.step, stopReason: 'budget', memory: mem, runId,
    metrics: M({ budgetExhausted: true }),
  };
}

/** Start a fresh autonomous run over a task. */
export async function runAgent(task: string, opts: RunAgentOptions): Promise<AgentResult> {
  const { userId, registry, llm } = opts;
  if (!llm.chatWithTools) throw new Error('auto mode needs an LLM with tool-calling (chatWithTools)');
  const progressive = opts.progressive ?? false;
  const store = opts.store ?? new MySqlAgentRunStore();
  const activeToolNames = progressive ? [FIND_TOOLS_SPEC.name] : toolSpecs(registry).map((t) => t.name);
  const mem0 = freshMemory();
  if (opts.seedFilter) mem0.constraints = { ...opts.seedFilter } as SearchFilter;   // soft seed from prefs
  const sys = SYSTEM + (progressive ? PROGRESSIVE_HINT : '') + (opts.profileHint ? `\n\n${opts.profileHint}` : '');
  const state: AgentRunState = {
    task,
    progressive,
    step: 0,
    memory: mem0,
    activeToolNames,
    trace: [],
    messages: [
      { role: 'system', content: sys },
      { role: 'user', content: task },
    ],
  };
  const run = await store.create(userId, state);
  return driveLoop(state, { userId, registry, llm, store, runId: run.id, budget: opts.maxSteps ?? MAX_STEPS });
}

export interface ResumeAgentOptions {
  approved: boolean;
  registry: SkillRegistry;
  llm: LLMClient;
  store?: AgentRunStore;
  maxSteps?: number;
  /** Recipients the draft was sent to (for the resume note). */
  sentTo?: string[];
}

/** Resume a suspended run after the human approved (sent) or cancelled its pending email. */
export async function resumeAgentRun(runId: number, opts: ResumeAgentOptions): Promise<AgentResult> {
  const { registry, llm } = opts;
  const store = opts.store ?? new MySqlAgentRunStore();
  const run = await store.get(runId);
  if (!run) throw new Error(`agent run ${runId} not found`);
  if (run.status !== 'awaiting_approval') {
    return { reply: `Run #${runId} is not awaiting approval (status: ${run.status}).`,
      trace: [], steps: run.state.step, stopReason: 'final', memory: run.state.memory, runId,
      metrics: { steps: run.state.step, toolCalls: 0, toolErrors: 0, loopGuards: 0, llmCalls: 0,
        groundingRewrites: 0, groundingStripped: 0, budgetExhausted: false, suspended: false, elapsedMs: 0 } };
  }
  const state = run.state;
  const draftId = run.pendingDraftId;
  state.messages.push({
    role: 'user',
    content: opts.approved
      ? `[system] The human approved and SENT email draft #${draftId}`
        + `${opts.sentTo?.length ? ' to ' + opts.sentTo.join(', ') : ''}. Continue and finish the task.`
      : `[system] The human CANCELLED email draft #${draftId}. Do not resend it; finish or adjust the task.`,
  });
  await store.save(runId, { status: 'running', pendingDraftId: null });
  return driveLoop(state, {
    userId: run.userId, registry, llm, store, runId,
    budget: state.step + (opts.maxSteps ?? MAX_STEPS),
  });
}
