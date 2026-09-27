/**
 * Tuning CLI — change one parameter, see what changed, roll back.
 *
 * The apply/rollback primitive the sweep loop is built on. Two properties matter:
 *   ATOMIC   — tmp file + rename, so a reader never sees a torn JSON. A half-written
 *              tuning.json would make every reader fail loudly (by design), and a sweep
 *              crash mid-write must not leave the repo unable to start.
 *   REVERSIBLE — the previous version is kept in tuning.prev.json, so "roll back the last
 *              change" is one command and needs no git.
 *
 *   npm run tune -- show
 *   npm run tune -- set shared.retrieval.prefetch 20
 *   npm run tune -- diff
 *   npm run tune -- rollback
 */
import { readFileSync, writeFileSync, renameSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CONFIG_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'config');
const CUR = process.env.IDX_TUNING?.trim() || join(CONFIG_DIR, 'tuning.json');
const PREV = join(CONFIG_DIR, 'tuning.prev.json');

/** Which eval suites a change in this section must not regress. */
const BLAST: Record<string, string> = {
  shared: 'BOTH paths — deterministic AND auto eval must not regress',
  deterministic: 'deterministic router only — intent/parse eval',
  auto: 'auto loop only — agent eval (NOT swept automatically)',
};

type Json = Record<string, unknown>;
const read = (p: string): Json => JSON.parse(readFileSync(p, 'utf8')) as Json;

function writeAtomic(path: string, value: Json): void {
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, `${JSON.stringify(value, null, 2)}\n`);
  renameSync(tmp, path);
}

/** Flatten to dotted leaf paths, so show/diff read the same as `set` takes. */
function flatten(o: unknown, prefix = ''): Array<[string, unknown]> {
  if (o === null || typeof o !== 'object' || Array.isArray(o)) return [[prefix, o]];
  return Object.entries(o as Json).flatMap(([k, v]) => flatten(v, prefix ? `${prefix}.${k}` : k));
}

function show(): void {
  const cfg = read(CUR);
  let section = '';
  for (const [path, value] of flatten(cfg)) {
    const top = path.split('.')[0]!;
    if (top !== section && BLAST[top]) {
      section = top;
      console.log(`\n[${top}]  ${BLAST[top]}`);
    }
    if (path !== 'version') console.log(`  ${path.padEnd(38)} ${JSON.stringify(value)}`);
  }
  console.log(`\nfile: ${CUR}${existsSync(PREV) ? `\nprev: ${PREV}` : ''}`);
}

function set(dotted: string, raw: string): void {
  const cfg = read(CUR);
  const keys = dotted.split('.');
  const leaf = keys.pop()!;
  let node: Json = cfg;
  for (const k of keys) {
    const next = node[k];
    if (next === undefined || typeof next !== 'object' || next === null) {
      throw new Error(`no such section "${keys.join('.')}" (run: npm run tune -- show)`);
    }
    node = next as Json;
  }
  const before = node[leaf];
  if (before === undefined) throw new Error(`no such parameter "${dotted}" (run: npm run tune -- show)`);

  // Type must match the existing value — a sweep writing "20" (string) where a number was
  // expected would change behaviour silently rather than fail.
  let parsed: unknown;
  if (typeof before === 'number') {
    parsed = Number(raw);
    if (!Number.isFinite(parsed as number)) throw new Error(`"${raw}" is not a number (${dotted} is a number)`);
  } else if (typeof before === 'boolean') {
    if (raw !== 'true' && raw !== 'false') throw new Error(`"${raw}" is not true/false (${dotted} is a boolean)`);
    parsed = raw === 'true';
  } else {
    parsed = raw;
  }
  if (parsed === before) { console.log(`unchanged: ${dotted} is already ${JSON.stringify(before)}`); return; }

  writeAtomic(PREV, read(CUR));        // snapshot BEFORE mutating
  node[leaf] = parsed;
  writeAtomic(CUR, cfg);
  const top = dotted.split('.')[0]!;
  console.log(`${dotted}: ${JSON.stringify(before)} -> ${JSON.stringify(parsed)}`);
  console.log(`blast radius: ${BLAST[top] ?? 'unknown section'}`);
  console.log('rollback with: npm run tune -- rollback');
}

function diff(): void {
  if (!existsSync(PREV)) { console.log('no previous version yet (nothing changed via this CLI)'); return; }
  const prev = new Map(flatten(read(PREV)));
  const cur = flatten(read(CUR));
  const rows = cur.filter(([p, v]) => JSON.stringify(prev.get(p)) !== JSON.stringify(v));
  if (!rows.length) { console.log('current == previous'); return; }
  for (const [p, v] of rows) console.log(`  ${p.padEnd(38)} ${JSON.stringify(prev.get(p))} -> ${JSON.stringify(v)}`);
}

function rollback(): void {
  if (!existsSync(PREV)) throw new Error('no previous version to roll back to');
  const restoring = read(PREV);
  writeAtomic(PREV, read(CUR));        // swap, so rollback is itself reversible
  writeAtomic(CUR, restoring);
  console.log('rolled back. run `npm run tune -- diff` to see the result');
}

const [cmd, ...rest] = process.argv.slice(2);
try {
  if (cmd === 'show' || cmd === undefined) show();
  else if (cmd === 'set') {
    if (rest.length !== 2) throw new Error('usage: npm run tune -- set <dotted.path> <value>');
    set(rest[0]!, rest[1]!);
  } else if (cmd === 'diff') diff();
  else if (cmd === 'rollback') rollback();
  else throw new Error(`unknown command "${cmd}" (show | set | diff | rollback)`);
} catch (e) {
  console.error(`error: ${(e as Error).message}`);
  process.exit(1);
}
