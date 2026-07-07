/**
 * Email drafts + persistent store (Week 11).
 *
 * Drafts are created with status 'pending_approval' and survive restarts (MySQL),
 * so the human-approval gate spans sessions. Sending is idempotent via the status
 * (a 'sent' draft can't be sent again).
 */
import type { RowDataPacket, ResultSetHeader } from 'mysql2/promise';
import { getPool } from '../db.js';

export type DraftStatus = 'pending_approval' | 'sent' | 'cancelled';

export interface EmailDraft {
  id: number;
  recipients: string[];
  subject: string;
  body: string;
  createdBy: string;
  status: DraftStatus;
  createdAt: number;
  sentAt?: number;
}

export type NewDraft = Omit<EmailDraft, 'id' | 'status' | 'createdAt' | 'sentAt'>;

export interface DraftStore {
  create(d: NewDraft): Promise<EmailDraft>;
  get(id: number): Promise<EmailDraft | null>;
  list(status?: DraftStatus): Promise<EmailDraft[]>;
  setStatus(id: number, status: DraftStatus, sentAt?: number): Promise<void>;
}

// ---- MySQL-backed (persistent) ----
export class MySqlDraftStore implements DraftStore {
  private ready?: Promise<void>;

  private async ensure() {
    this.ready ??= getPool().query(`CREATE TABLE IF NOT EXISTS email_drafts (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      recipients JSON NOT NULL,
      subject VARCHAR(500) NOT NULL,
      body MEDIUMTEXT NOT NULL,
      created_by VARCHAR(64) NOT NULL,
      status VARCHAR(32) NOT NULL DEFAULT 'pending_approval',
      created_at BIGINT NOT NULL,
      sent_at BIGINT NULL
    )`).then(() => undefined);
    await this.ready;
  }

  async create(d: NewDraft): Promise<EmailDraft> {
    await this.ensure();
    const createdAt = Date.now();
    const [res] = await getPool().execute<ResultSetHeader>(
      'INSERT INTO email_drafts (recipients, subject, body, created_by, status, created_at) VALUES (?,?,?,?,?,?)',
      [JSON.stringify(d.recipients), d.subject, d.body, d.createdBy, 'pending_approval', createdAt],
    );
    return { ...d, id: res.insertId, status: 'pending_approval', createdAt };
  }

  async get(id: number): Promise<EmailDraft | null> {
    await this.ensure();
    const rows = await queryRows('SELECT * FROM email_drafts WHERE id=?', [id]);
    return rows[0] ? toDraft(rows[0]) : null;
  }

  async list(status?: DraftStatus): Promise<EmailDraft[]> {
    await this.ensure();
    const rows = status
      ? await queryRows('SELECT * FROM email_drafts WHERE status=? ORDER BY id DESC', [status])
      : await queryRows('SELECT * FROM email_drafts ORDER BY id DESC', []);
    return rows.map(toDraft);
  }

  async setStatus(id: number, status: DraftStatus, sentAt?: number): Promise<void> {
    await this.ensure();
    await getPool().execute('UPDATE email_drafts SET status=?, sent_at=? WHERE id=?',
      [status, sentAt ?? null, id]);
  }
}

async function queryRows(sql: string, params: unknown[]): Promise<RowDataPacket[]> {
  const [rows] = await getPool().query<RowDataPacket[]>(sql, params);
  return rows;
}

function toDraft(r: RowDataPacket): EmailDraft {
  return {
    id: Number(r.id),
    recipients: typeof r.recipients === 'string' ? JSON.parse(r.recipients) : r.recipients,
    subject: r.subject,
    body: r.body,
    createdBy: r.created_by,
    status: r.status,
    createdAt: Number(r.created_at),
    sentAt: r.sent_at != null ? Number(r.sent_at) : undefined,
  };
}

// ---- In-memory (tests) ----
export class InMemoryDraftStore implements DraftStore {
  private readonly map = new Map<number, EmailDraft>();
  private seq = 0;
  async create(d: NewDraft): Promise<EmailDraft> {
    const draft: EmailDraft = { ...d, id: ++this.seq, status: 'pending_approval', createdAt: Date.now() };
    this.map.set(draft.id, draft);
    return draft;
  }
  async get(id: number) { return this.map.get(id) ?? null; }
  async list(status?: DraftStatus) {
    return [...this.map.values()].filter((d) => !status || d.status === status).reverse();
  }
  async setStatus(id: number, status: DraftStatus, sentAt?: number) {
    const d = this.map.get(id);
    if (d) { d.status = status; if (sentAt) d.sentAt = sentAt; }
  }
}
