/**
 * Auto/agent mode — persistent agent-run store (M2).
 *
 * An agent run's ENTIRE resumable state is (messages + slot memory + enabled tools +
 * step). Persisting it lets the loop suspend at a human-in-the-loop point (e.g. "send
 * this email?") — the process is released, and hours later an `approve <draftId>` over
 * WhatsApp resumes from exactly here. This generalizes the existing draft→approve→send
 * primitive (which already persist-suspends a single draft) to the whole loop state.
 */
import type { RowDataPacket, ResultSetHeader } from 'mysql2/promise';
import { getPool } from '../../db.js';
import type { ChatMessage } from '../../llm/client.js';
import type { WorkingMemory } from './memory.js';

export type RunStatus = 'running' | 'awaiting_approval' | 'done' | 'cancelled';

/** One recorded loop step (defined here, not loop.ts, so state can carry it without a cycle). */
export interface AgentTraceStep {
  step: number;
  tool?: string;
  args?: Record<string, unknown>;
  observation?: string;
  thought?: string;
}

/** The full resumable snapshot of a loop. */
export interface AgentRunState {
  task: string;
  messages: ChatMessage[];
  memory: WorkingMemory;
  activeToolNames: string[];
  step: number;
  progressive: boolean;
  /** Full step trace, persisted for observability + replay (accumulates across resumes). */
  trace: AgentTraceStep[];
}

export interface AgentRun {
  id: number;
  userId: string;
  status: RunStatus;
  state: AgentRunState;
  pendingDraftId?: number;
  createdAt: number;
  updatedAt: number;
}

export interface SavePatch {
  state?: AgentRunState;
  status?: RunStatus;
  pendingDraftId?: number | null;
}

export interface AgentRunStore {
  create(userId: string, state: AgentRunState): Promise<AgentRun>;
  get(id: number): Promise<AgentRun | null>;
  save(id: number, patch: SavePatch): Promise<void>;
  /** Find the run currently awaiting approval of this draft (HITL resume lookup). */
  byPendingDraft(draftId: number): Promise<AgentRun | null>;
  /** Most recent run for a user (for the `status` progress query). */
  latestForUser(userId: string): Promise<AgentRun | null>;
  /** Recent runs for a user, newest first (for memory consolidation — bounded context). */
  recentForUser(userId: string, limit: number): Promise<AgentRun[]>;
}

// ---- MySQL-backed (persistent) ----
export class MySqlAgentRunStore implements AgentRunStore {
  private ready?: Promise<void>;

  private async ensure() {
    this.ready ??= getPool().query(`CREATE TABLE IF NOT EXISTS agent_runs (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      user_id VARCHAR(64) NOT NULL,
      status VARCHAR(32) NOT NULL DEFAULT 'running',
      state JSON NOT NULL,
      pending_draft_id BIGINT NULL,
      created_at BIGINT NOT NULL,
      updated_at BIGINT NOT NULL
    )`).then(() => undefined);
    await this.ready;
  }

  async create(userId: string, state: AgentRunState): Promise<AgentRun> {
    await this.ensure();
    const now = Date.now();
    const [res] = await getPool().execute<ResultSetHeader>(
      'INSERT INTO agent_runs (user_id, status, state, created_at, updated_at) VALUES (?,?,?,?,?)',
      [userId, 'running', JSON.stringify(state), now, now],
    );
    return { id: res.insertId, userId, status: 'running', state, createdAt: now, updatedAt: now };
  }

  async get(id: number): Promise<AgentRun | null> {
    await this.ensure();
    const [rows] = await getPool().query<RowDataPacket[]>('SELECT * FROM agent_runs WHERE id=?', [id]);
    return rows[0] ? toRun(rows[0]) : null;
  }

  async save(id: number, patch: SavePatch): Promise<void> {
    await this.ensure();
    const sets: string[] = ['updated_at=?'];
    const vals: (string | number | null)[] = [Date.now()];
    if (patch.state !== undefined) { sets.push('state=?'); vals.push(JSON.stringify(patch.state)); }
    if (patch.status !== undefined) { sets.push('status=?'); vals.push(patch.status); }
    if (patch.pendingDraftId !== undefined) { sets.push('pending_draft_id=?'); vals.push(patch.pendingDraftId); }
    vals.push(id);
    await getPool().execute(`UPDATE agent_runs SET ${sets.join(', ')} WHERE id=?`, vals);
  }

  async byPendingDraft(draftId: number): Promise<AgentRun | null> {
    await this.ensure();
    const [rows] = await getPool().query<RowDataPacket[]>(
      "SELECT * FROM agent_runs WHERE pending_draft_id=? AND status='awaiting_approval' LIMIT 1", [draftId]);
    return rows[0] ? toRun(rows[0]) : null;
  }

  async latestForUser(userId: string): Promise<AgentRun | null> {
    await this.ensure();
    const [rows] = await getPool().query<RowDataPacket[]>(
      'SELECT * FROM agent_runs WHERE user_id=? ORDER BY id DESC LIMIT 1', [userId]);
    return rows[0] ? toRun(rows[0]) : null;
  }

  async recentForUser(userId: string, limit: number): Promise<AgentRun[]> {
    await this.ensure();
    const [rows] = await getPool().query<RowDataPacket[]>(
      'SELECT * FROM agent_runs WHERE user_id=? ORDER BY id DESC LIMIT ?', [userId, Math.max(1, Math.min(50, limit))]);
    return rows.map(toRun);
  }
}

function toRun(r: RowDataPacket): AgentRun {
  return {
    id: Number(r.id),
    userId: r.user_id,
    status: r.status,
    state: typeof r.state === 'string' ? JSON.parse(r.state) : r.state,
    pendingDraftId: r.pending_draft_id != null ? Number(r.pending_draft_id) : undefined,
    createdAt: Number(r.created_at),
    updatedAt: Number(r.updated_at),
  };
}

// ---- In-memory (tests) ----
export class InMemoryAgentRunStore implements AgentRunStore {
  private readonly map = new Map<number, AgentRun>();
  private seq = 0;
  async create(userId: string, state: AgentRunState): Promise<AgentRun> {
    const now = Date.now();
    const run: AgentRun = { id: ++this.seq, userId, status: 'running', state, createdAt: now, updatedAt: now };
    this.map.set(run.id, run);
    return run;
  }
  async get(id: number) { return this.map.get(id) ?? null; }
  async save(id: number, patch: SavePatch) {
    const run = this.map.get(id);
    if (!run) return;
    if (patch.state !== undefined) run.state = patch.state;
    if (patch.status !== undefined) run.status = patch.status;
    if (patch.pendingDraftId !== undefined) run.pendingDraftId = patch.pendingDraftId ?? undefined;
    run.updatedAt = Date.now();
  }
  async byPendingDraft(draftId: number) {
    return [...this.map.values()].find((r) => r.pendingDraftId === draftId && r.status === 'awaiting_approval') ?? null;
  }
  async latestForUser(userId: string) {
    return [...this.map.values()].filter((r) => r.userId === userId).sort((a, b) => b.id - a.id)[0] ?? null;
  }
  async recentForUser(userId: string, limit: number) {
    return [...this.map.values()].filter((r) => r.userId === userId).sort((a, b) => b.id - a.id).slice(0, Math.max(1, limit));
  }
}
