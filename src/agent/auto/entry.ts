/**
 * Auto/agent mode — entry wrapper (M2 wiring).
 *
 * Sits IN FRONT of the deterministic orchestrator without polluting it. Handles two
 * things and returns a reply, or null to fall through to normal orchestration:
 *   1. `approve/cancel #N` where #N is the pending email of a SUSPENDED agent run ->
 *      send (or cancel) the draft, then resume the run to completion. A normal email
 *      draft (not tied to a run) returns null so the existing orchestrator handles it.
 *   2. `/auto <task>` -> start a fresh autonomous run.
 *
 * The approval channel is the SAME `approve <id>` the operator already uses — the HITL
 * gate is deterministic CODE, never an LLM tool (丙).
 */
import { approveAndSend, cancelDraft, type SendFn } from '../../email/email.js';
import type { DraftStore } from '../../email/drafts.js';
import type { SkillRegistry } from '../../orchestrator/skill.js';
import type { LLMClient } from '../../llm/client.js';
import { runAgent, resumeAgentRun } from './loop.js';
import { loadProfile, saveProfile, profileHint, preferredFilter, selectMemories, episodicMemories, touchMemory, learnFromFilter } from '../../memory/profile.js';
import { parseQuery } from '../../search/parseQuery.js';
import { isKnownCity } from '../../search/cityDictionary.js';
import type { AgentRunStore, AgentRun } from './runStore.js';

const APPROVE = /^\s*(?:approve|send it|批准|通过|确认发送|确认)\s*#?(\d+)\s*$/i;
const CANCEL = /^\s*(?:cancel|discard|取消|作废|不发)\s*#?(\d+)\s*$/i;
const AUTO = /^\s*\/auto\s+([\s\S]+)$/i;
const STATUS = /^\s*(?:status|进度|状态)\s*#?(\d+)?\s*$/i;

/** Human-readable progress of a run — the persisted trace makes this observable/replayable. */
export function renderAgentStatus(run: AgentRun): string {
  const s = run.state;
  const lines = [`🤖 Run #${run.id} — ${run.status} · ${s.step} step(s)`];
  const tools = s.trace.filter((t) => t.tool).map((t) => t.tool);
  if (tools.length) lines.push(`tools: ${tools.join(' → ')}`);
  if (s.memory.facts.length) {
    lines.push('progress:');
    s.memory.facts.forEach((f, i) => lines.push(`  ${i + 1}. ${f}`));
  }
  if (run.status === 'awaiting_approval') lines.push(`⏸ waiting on "approve ${run.pendingDraftId}"`);
  return lines.join('\n');
}

export interface AgentEntryDeps {
  registry: SkillRegistry;
  llm: LLMClient;
  draftStore: DraftStore;
  runStore: AgentRunStore;
  send?: SendFn;        // real SMTP in prod; injectable for tests
  progressive?: boolean;
}

/** Returns a reply if this message is an agent-mode command, else null (-> orchestrate). */
export async function handleAgentMessage(
  userId: string,
  message: string,
  deps: AgentEntryDeps,
): Promise<string | null> {
  const { registry, llm, draftStore, runStore, send } = deps;

  // 0. status query (observability): latest run, or a specific #id
  const st = message.match(STATUS);
  if (st) {
    const id = st[1] ? Number(st[1]) : undefined;
    const run = id ? await runStore.get(id) : await runStore.latestForUser(userId);
    if (!run) return id ? `No agent run #${id}.` : 'No agent runs yet.';
    return renderAgentStatus(run);
  }

  // 1. approve/cancel that belongs to a suspended run -> resume it
  const ap = message.match(APPROVE);
  const cn = message.match(CANCEL);
  if (ap || cn) {
    const id = Number((ap ?? cn)![1]);
    const run = await runStore.byPendingDraft(id);
    if (!run) return null;            // not an agent draft -> let the orchestrator handle approve/cancel
    if (ap) {
      const r = await approveAndSend(id, userId, draftStore, send);
      if (r.status !== 'sent' && r.status !== 'sent_dryrun') return `Couldn't send draft #${id} (${r.status}).`;
      const res = await resumeAgentRun(run.id, { approved: true, registry, llm, store: runStore, sentTo: r.draft?.recipients });
      return `✅ Sent draft #${id}. ${res.reply}`;
    }
    await cancelDraft(id, draftStore);
    const res = await resumeAgentRun(run.id, { approved: false, registry, llm, store: runStore });
    return `Draft #${id} cancelled. ${res.reply}`;
  }

  // 2. explicit auto task
  const auto = message.match(AUTO);
  if (auto) {
    const task = auto[1]!.trim();
    const profile = loadProfile(userId);   // long-term memory: facts + semantic + selected episodic
    // immediate FACT learning (symmetric with the deterministic mode's onFilter): parse the
    // user's OWN words (not the injected defaults) so we don't self-reinforce our own seeds.
    const userFilter = (await parseQuery(task, { isKnownCity })).filter;
    if (Object.values(userFilter).some((v) => v != null)) learnFromFilter(profile, userFilter);
    // episodic memories are SELECTIVELY loaded: LLM picks the ones relevant to this task by desc
    const episodic = await selectMemories(episodicMemories(profile), task, llm);
    episodic.forEach((m) => touchMemory(profile, m.name));
    saveProfile(profile);   // persist learned facts + memory touches
    const res = await runAgent(task, {
      userId, registry, llm, store: runStore, progressive: deps.progressive ?? true,
      profileHint: profileHint(profile, episodic), seedFilter: preferredFilter(profile),
    });
    return res.reply;
  }

  return null;
}
