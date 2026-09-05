/** Memory full-chain e2e: fake sessions -> real-DeepSeek consolidation -> profile ->
 * profile-injected auto task. Real LLM + real DB (search). Writes /tmp/memory_e2e.txt. */
import { runConsolidation } from './src/memory/consolidation.js';
import { loadProfile, saveProfile, learnFromFilter, renderMd } from './src/memory/profile.js';
import { handleAgentMessage } from './src/agent/auto/entry.js';
import { InMemoryAgentRunStore, type AgentRunState } from './src/agent/auto/runStore.js';
import { buildRegistry } from './src/orchestrator/skills.js';
import { pythonBridge } from './src/orchestrator/bridge.js';
import { getLLMClient } from './src/llm/client.js';
import { InMemoryDraftStore } from './src/email/drafts.js';
import type { SearchFilter } from './src/search/filters.js';
import { closePool } from './src/db.js';
import { writeFileSync, rmSync } from 'node:fs';

const uid = 'e2e-memory-user';
const clean = () => { rmSync(`data/profiles/${uid}.json`, { force: true }); rmSync(`data/profiles/${uid}.md`, { force: true }); };
clean();

const llm = getLLMClient();
const runStore = new InMemoryAgentRunStore();
const draftStore = new InMemoryDraftStore();
const registry = buildRegistry(pythonBridge, draftStore);
const send = async () => {};
const out: string[] = [];

const mk = (u: string, a: string): AgentRunState => ({
  task: u, progressive: false, step: 1, activeToolNames: [], trace: [],
  memory: { constraints: {}, facts: [], drafts: [] },
  messages: [{ role: 'user', content: u }, { role: 'assistant', content: a }],
});

// 1) fake history: user repeatedly looks for Irvine 3-bed near good schools
await runStore.create(uid, mk('find 3 bed homes in Irvine under 2M near good schools', 'Here are Irvine listings...'));
await runStore.create(uid, mk('any 3-bedroom in Irvine with a good school district?', 'Irvine school-zone homes...'));
await runStore.create(uid, mk('show me Irvine 3-bed again', 'More Irvine 3-bed...'));

// 2) memory sub-agent (REAL DeepSeek) distills semantic/episodic memories
const summary = await runConsolidation(uid, { llm, runStore });
out.push('=== consolidation summary ===', summary.slice(0, 200));
let p = loadProfile(uid);
out.push('=== memories written by sub-agent ===');
p.memories.forEach((m) => out.push(`  [${m.type}] ${m.name}: ${m.description} (salience ${m.salience})`));

// 3) immediate fact layer (simulate repeated turns learning city/beds)
for (let i = 0; i < 3; i++) p = learnFromFilter(p, { city: 'Irvine', beds: 3 } as SearchFilter);
saveProfile(p);
out.push('=== facts (prefs) ===', JSON.stringify(p.prefs));

// 4) profile-injected auto task WITHOUT a city -> should lean Irvine via seed + hint
const reply = await handleAgentMessage(uid, '/auto find me a 3-bedroom home', { registry, llm, draftStore, runStore, send });
out.push('=== auto reply (no city given; profile-injected) ===', (reply ?? '').slice(0, 350));
out.push('=== leaned Irvine? ===', String((reply ?? '').includes('Irvine')));

out.push('=== profile.md ===', renderMd(loadProfile(uid)).slice(0, 700));
writeFileSync('/tmp/memory_e2e.txt', out.join('\n') + '\n');
clean();
await closePool();
