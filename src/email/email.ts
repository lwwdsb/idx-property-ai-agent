/**
 * Email agent (Week 11) — draft-then-approve, with outbound locked at the tool layer.
 *
 * 丙 (top priority): the assistant/LLM can ONLY call `draftEmail`, which creates a
 * `pending_approval` draft — it never sends. `approveAndSend` is the ONLY send path;
 * it is not a skill and is never reachable by the orchestrator/LLM, only by a human
 * (CLI now, WhatsApp approve later). So the model structurally cannot send.
 *
 * Guardrails (handbook red lines): operator allowlist, explicit recipients (no
 * DB-scrape), batch cap, idempotent send. No SMTP creds => dry-run (never sends).
 */
import nodemailer, { type Transporter } from 'nodemailer';
import { config, emailConfigured } from '../config.js';
import { logger } from '../logger.js';
import type { DraftStore, EmailDraft } from './drafts.js';

export const MAX_RECIPIENTS = 25;
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

function authorized(user: string): boolean {
  // empty allowlist = single-operator/dev mode (allow); otherwise must be listed
  return config.email.allowlist.length === 0 || config.email.allowlist.includes(user);
}

export interface DraftRequest {
  createdBy: string;
  recipients: string[];
  subject: string;
  body: string;
}

export interface DraftOutcome { ok: boolean; draft?: EmailDraft; error?: string; }

/** The ONLY email capability exposed to the assistant. Creates a pending draft. */
export async function draftEmail(req: DraftRequest, store: DraftStore): Promise<DraftOutcome> {
  if (!authorized(req.createdBy)) return { ok: false, error: 'not authorized to draft/send email' };

  const recipients = req.recipients.map((r) => r.trim()).filter(Boolean);
  if (recipients.length === 0) return { ok: false, error: 'no recipients' };
  if (recipients.length > MAX_RECIPIENTS) return { ok: false, error: `too many recipients (max ${MAX_RECIPIENTS})` };
  if (recipients.some((r) => !EMAIL_RE.test(r))) return { ok: false, error: 'invalid recipient email address' };
  if (!req.subject.trim() || !req.body.trim()) return { ok: false, error: 'subject and body are required' };

  const draft = await store.create({ recipients, subject: req.subject, body: req.body, createdBy: req.createdBy });
  logger.info('email drafted (pending approval)', { id: draft.id, recipients: recipients.length, by: req.createdBy });
  return { ok: true, draft }; // status = pending_approval — NOT sent
}

export type SendStatus = 'sent' | 'sent_dryrun' | 'not_found' | 'unauthorized' | 'already_sent' | 'not_pending';

let transporter: Transporter | null = null;
function getTransporter(): Transporter {
  transporter ??= nodemailer.createTransport({
    host: config.email.smtpHost,
    port: config.email.smtpPort,
    secure: config.email.smtpPort === 465,
    auth: { user: config.email.user, pass: config.email.password },
  });
  return transporter;
}

/**
 * The ONLY code path that sends. Human-invoked (approver), never the LLM.
 * Idempotent: a draft already 'sent' is not re-sent. Dry-run when SMTP is unconfigured.
 */
export async function approveAndSend(
  draftId: number,
  approver: string,
  store: DraftStore,
): Promise<{ status: SendStatus; draft?: EmailDraft }> {
  const draft = await store.get(draftId);
  if (!draft) return { status: 'not_found' };
  if (!authorized(approver)) return { status: 'unauthorized' };
  if (draft.status === 'sent') return { status: 'already_sent', draft };   // idempotency
  if (draft.status !== 'pending_approval') return { status: 'not_pending', draft };

  if (!emailConfigured()) {
    await store.setStatus(draft.id, 'sent', Date.now());
    logger.warn('email dry-run (no SMTP configured) — marked sent, not delivered', { id: draft.id });
    return { status: 'sent_dryrun', draft };
  }

  await getTransporter().sendMail({
    from: config.email.from,
    to: draft.recipients.join(', '),
    subject: draft.subject,
    text: draft.body,
  });
  await store.setStatus(draft.id, 'sent', Date.now());
  logger.info('email sent after approval', { id: draft.id, approver, recipients: draft.recipients.length });
  return { status: 'sent', draft };
}

export async function cancelDraft(draftId: number, store: DraftStore): Promise<boolean> {
  const draft = await store.get(draftId);
  if (!draft || draft.status !== 'pending_approval') return false;
  await store.setStatus(draft.id, 'cancelled');
  return true;
}

export function previewDraft(d: EmailDraft): string {
  return [
    `Draft #${d.id}  [${d.status}]`,
    `To: ${d.recipients.join(', ')}`,
    `Subject: ${d.subject}`,
    '',
    d.body,
  ].join('\n');
}
