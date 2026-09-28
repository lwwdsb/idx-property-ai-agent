/**
 * Runs EVERY src/**\/*.test.ts, sequentially, and fails if any file fails.
 *
 * `npm test` used to run exactly one file (src/search/parseQuery.test.ts). The other 13 test
 * files existed, passed, and were simply never invoked — so "npm test: 29/29 passed" was a true
 * statement about the parser and said nothing about memory, email, orchestration, resilience or
 * the agent loop. A green suite that silently covers 29 of 143 assertions is worse than no suite,
 * because it is quoted as evidence.
 *
 * Sequential on purpose: several tests write real files under data/profiles/ and clean up after
 * themselves, so running them in parallel would have them clobber each other's fixtures.
 */
import { readdirSync, statSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';

const files = [];
(function walk(dir) {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isDirectory()) walk(p);
    else if (e.endsWith('.test.ts')) files.push(p);
  }
})('src');
files.sort();

let failed = 0, total = 0;
for (const f of files) {
  const r = spawnSync('npx', ['tsx', f], { encoding: 'utf8' });
  const out = (r.stdout ?? '') + (r.stderr ?? '');
  const m = out.match(/(\d+)\/(\d+) passed/);
  const n = m ? Number(m[2]) : 0;
  total += n;
  const ok = r.status === 0 && (!m || m[1] === m[2]);
  if (!ok) { failed++; process.stdout.write(out); }
  console.log(`${ok ? '✓' : '✗'} ${f.padEnd(44)} ${m ? `${m[1]}/${m[2]}` : '(no count)'}`);
}
console.log(`\n${files.length} files, ${total} assertions — ${failed ? `${failed} FILE(S) FAILED` : 'all passed'}`);
process.exit(failed ? 1 : 0);
