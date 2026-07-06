/**
 * Bridge from the TS orchestrator to the Python retrieval subsystem
 * (recommend / RAG / price-validate). Injectable so tests use a fake and the
 * router logic is verified without Python/Qdrant.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';

const execFileAsync = promisify(execFile);
const ROOT = path.resolve(import.meta.dirname, '../..');
const PY = path.join(ROOT, '.venv/bin/python');

export interface PythonBridge {
  rag(question: string): Promise<string>;
  recommend(listingId: number): Promise<string>;
  validate(listing: { city: string | null; sqft: number | null; price: number | null }): Promise<string>;
}

async function run(script: string, args: string[]): Promise<string> {
  const { stdout } = await execFileAsync(PY, [path.join(ROOT, 'retrieval', script), ...args], {
    cwd: ROOT,
    timeout: 60_000,
    maxBuffer: 4 * 1024 * 1024,
  });
  return stdout.trim();
}

/** Real bridge: shells out to the venv Python scripts. */
export const pythonBridge: PythonBridge = {
  rag: (q) => run('rag.py', [q]),
  recommend: (id) => run('recommend.py', [String(id)]),
  validate: (l) => run('recommend.py', ['--validate', l.city ?? '', String(l.sqft ?? ''), String(l.price ?? '')]),
};
