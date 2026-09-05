/**
 * Auto/agent mode — slot working memory.
 *
 * Reuses the SearchFilter slot+patch idea as the agent's structured scratchpad, so the
 * model reads a compact STATE instead of re-scanning a growing wall of raw observations:
 *   - constraints: the SearchFilter accumulated across steps (mergeFilter, not append)
 *   - facts:       one short line per completed action (progress log)
 *   - drafts:      ids of email drafts created (so it doesn't re-draft)
 * A snapshot is injected each turn (see loop.ts). This curbs context growth and lightens
 * the (DeepSeek) model's memory burden — the same reason the router uses filter slots.
 */
import { sanitizeFilter, type SearchFilter } from '../../llm/client.js';
import { mergeFilter, summarizeFilter, type FilterPatch } from '../../search/filters.js';

export interface WorkingMemory {
  constraints: SearchFilter;
  facts: string[];
  drafts: number[];
}

export function freshMemory(): WorkingMemory {
  return { constraints: {}, facts: [], drafts: [] };
}

export function isEmpty(mem: WorkingMemory): boolean {
  return mem.facts.length === 0 && mem.drafts.length === 0
    && Object.keys(mem.constraints).length === 0;
}

/** First non-empty line of an observation — enough as a progress marker. */
function firstLine(text: string): string {
  const line = text.split('\n').map((l) => l.trim()).find((l) => l.length > 0) ?? '';
  return line.slice(0, 120);
}

/** Fold one completed tool step into memory: accumulate constraints, log a fact, track drafts. */
export function recordStep(mem: WorkingMemory, name: string, args: Record<string, unknown>, observation: string): void {
  mem.constraints = mergeFilter(mem.constraints, sanitizeFilter(args) as FilterPatch);
  mem.facts.push(`${name}: ${firstLine(observation)}`);
  if (name === 'email') {
    const m = observation.match(/Draft #(\d+)/);
    if (m) mem.drafts.push(Number(m[1]));
  }
}

/** Render a compact snapshot for injection into the model's context. */
export function renderMemory(mem: WorkingMemory): string {
  const lines = ['Working memory (your progress so far — build on it, do not repeat done work):'];
  const c = summarizeFilter(mem.constraints);
  if (c) lines.push(`- known constraints: ${c}`);
  if (mem.facts.length) {
    lines.push('- done:');
    mem.facts.forEach((f, i) => lines.push(`  ${i + 1}. ${f}`));
  }
  if (mem.drafts.length) lines.push(`- email drafts created: ${mem.drafts.map((d) => `#${d}`).join(', ')}`);
  return lines.join('\n');
}
