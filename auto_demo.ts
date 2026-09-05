import { handleAgentMessage } from './src/agent/auto/entry.js';
import { InMemoryAgentRunStore } from './src/agent/auto/runStore.js';
import { InMemoryDraftStore } from './src/email/drafts.js';
import { buildRegistry } from './src/orchestrator/skills.js';
import { pythonBridge } from './src/orchestrator/bridge.js';
import { getLLMClient } from './src/llm/client.js';
import { config } from './src/config.js';
import { closePool } from './src/db.js';
import { writeFileSync } from 'node:fs';

const llm = getLLMClient();
const draftStore = new InMemoryDraftStore();
const runStore = new InMemoryAgentRunStore();
const registry = buildRegistry(pythonBridge, draftStore);
const sentBox: string[][] = [];
const send = async (m: { recipients: string[] }) => { sentBox.push(m.recipients); };
const userId = config.email.allowlist[0] ?? 'op';
const deps = { registry, llm, draftStore, runStore, send };
const out: string[] = [];

const r1 = await handleAgentMessage(userId, '/auto Check the Irvine market and email agent@example.com the stats.', deps);
out.push('=== /auto ===', (r1 ?? '').slice(0, 100));
out.push('=== status (mid, should be awaiting) ===', (await handleAgentMessage(userId, 'status', deps)) ?? '(null)');

const m = r1?.match(/#(\d+)/);
if (m) {
  await handleAgentMessage(userId, `approve ${m[1]}`, deps);
  out.push(`=== (approved #${m[1]}, emailsSent=${sentBox.length}) ===`);
  out.push('=== status (after resume, should be done + more steps) ===', (await handleAgentMessage(userId, 'status', deps)) ?? '(null)');
}
writeFileSync('/tmp/auto_demo.txt', out.join('\n') + '\n');
await closePool();
