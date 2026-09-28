/**
 * memory_dynamics 用例的执行 + 断言,单一实现。
 *
 * 主集 runner(evalMemoryDynamics.ts)和回归池 runner(regression.ts)共用这一份。抽出来的理由是
 * 项目一贯的那条:两份实现会漂移,而这里漂移的后果是主集和副池对"同一条用例算不算过"给出
 * 不同答案 —— 副池是一票否决闸,它和主集判得不一样比两边都错更糟。
 */
import { freshProfile, addMemory, touchMemory, compactMemories, slotConflicts, memoryDerivedFilter,
  type UserProfile } from '../../../src/memory/profile.js';

/** compScore is private to profile.ts, so it is reproduced here — and md-006..008 exist to catch
 *  the two implementations drifting: if the real one changes shape, the ordering probes break. */
const daysSince = (d: string) => Math.max(0, Math.round((Date.now() - Date.parse(d)) / 86_400_000));
function score(m: { salience: number; useCount: number; lastUsed: string; updatedAt: string }) {
  const fresh = Math.min(daysSince(m.lastUsed), daysSince(m.updatedAt));
  return m.salience * (1 / (1 + fresh / 30)) * (1 + Math.log1p(m.useCount) / 5);
}
function rankOnly(m: { salience: number; lastUsed: string; updatedAt: string }) {
  const fresh = Math.min(daysSince(m.lastUsed), daysSince(m.updatedAt));
  return m.salience * (1 / (1 + fresh / 30));
}
const shiftDays = (iso: string, days: number) =>
  new Date(Date.parse(iso) - days * 86_400_000).toISOString().slice(0, 10);

function run(ops: any[], p: UserProfile) {
  let last: { removed: string[] } | null = null;
  for (const o of ops) {
    if (o.op === 'add') {
      p = addMemory(p, { name: o.name, description: o.description ?? 'd', type: o.type,
        content: o.content, salience: o.salience, sourceRuns: o.sourceRuns, mergedFrom: o.mergedFrom,
        slots: o.slots });
    } else if (o.op === 'touch') {
      // touchMemory mutates in place and does NOT return the profile — reassigning its
      // result wiped the profile on the first call.
      for (let i = 0; i < (o.times ?? 1); i++) touchMemory(p, o.name);
    } else if (o.op === 'age') {
      const m = p.memories.find((x) => x.name === o.name)!;
      m.lastUsed = shiftDays(m.lastUsed, o.days);
      m.updatedAt = shiftDays(m.updatedAt, o.days);
      m.createdAt = shiftDays(m.createdAt, o.days);
    } else if (o.op === 'compact') {
      last = compactMemories(p, { maxSemantic: o.maxSemantic, maxEpisodic: o.maxEpisodic, minScore: o.minScore });
    }
  }
  return { p, last };
}


/** 跑完一条用例的 ops,再逐条求断言。返回 checks + 存活名单。 */
export function evalDynamicsCase(c: any): { checks: Record<string, boolean>; ok: boolean; survivors: string[]; removed: string[] } {
  const { p, last } = run(c.ops, freshProfile('eval'));
  const names = p.memories.map((m) => m.name).sort();
  const byName = new Map(p.memories.map((m) => [m.name, m]));
  const e = c.expect;
  const checks: Record<string, boolean> = {};

  if (e.survivors !== undefined) checks.survivors = JSON.stringify(names) === JSON.stringify([...e.survivors].sort());
  if (e.removed !== undefined) checks.removed = (e.removed as string[]).every((n) => !byName.has(n));
  if (e.salience !== undefined) {
    checks.salience = Object.entries(e.salience).every(([n, v]) => byName.get(n)?.salience === v);
  }
  if (e.content_contains !== undefined) {
    checks.content = Object.entries(e.content_contains).every(([n, sub]) => byName.get(n)?.content.includes(sub as string));
  }
  if (e.fields !== undefined) {
    checks.fields = Object.entries(e.fields as Record<string, any>).every(([n, f]) => {
      const m = byName.get(n); if (!m) return false;
      return Object.entries(f).every(([k, v]) => JSON.stringify((m as any)[k]) === JSON.stringify(v));
    });
  }
  if (e.score_order !== undefined) {
    const ordered = [...(e.score_order as string[])];
    const got = ordered.map((n) => byName.get(n)).filter(Boolean) as any[];
    // strict >: a TIE means the varied signal is dead, which is exactly what these probe
    checks.score_order = got.length === ordered.length
      && got.every((m, i) => i === 0 || score(got[i - 1]!) > score(m));
  }
  if (e.rank_order_differs === true) {
    const [a, b] = (e.score_order as string[]).map((n) => byName.get(n)!);
    checks.rank_differs = score(a) > score(b) && !(rankOnly(a) > rankOnly(b));
  }
  // CROSS-NAME CONTRADICTION, detected off DECLARED slots (never off prose — see slotConflicts).
  // Expected as [{type, field}] pairs; order-insensitive.
  if (e.conflicts !== undefined) {
    const got = slotConflicts(p.memories).map((c) => `${c.type}.${c.field}`).sort();
    const want = (e.conflicts as Array<{ type: string; field: string }>)
      .map((c) => `${c.type}.${c.field}`).sort();
    checks.conflicts = JSON.stringify(got) === JSON.stringify(want);
  }
  // What the deterministic channel actually seeds. The point of most conflict cases is not that
  // a conflict was NOTICED but that the conflicted field stops reaching a search at all.
  if (e.derived_filter !== undefined) {
    const got = memoryDerivedFilter(p.memories) as Record<string, unknown>;
    const want = e.derived_filter as Record<string, unknown>;
    const keys = [...new Set([...Object.keys(got), ...Object.keys(want)])];
    checks.derived_filter = keys.every((k) => JSON.stringify(got[k]) === JSON.stringify(want[k]));
  }
  if (e.types !== undefined) {
    checks.types = Object.entries(e.types as Record<string, string>)
      .every(([n, t]) => byName.get(n)?.type === t);
  }
  if (e.idempotent === true) {
    const before = p.memories.map((m) => m.name).sort().join(',');
    const second = compactMemories(p, {});
    checks.idempotent = second.removed.length === 0 && p.memories.map((m) => m.name).sort().join(',') === before;
  }

  const ok = Object.values(checks).every(Boolean);
  return { checks, ok, survivors: names, removed: last?.removed ?? [] };
}
