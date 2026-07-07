/**
 * Deterministic email-approval commands (Week 12+): approve / cancel / list drafts.
 *
 * 丙 preservation: these are matched and executed by CODE — NOT exposed as an LLM
 * tool and NOT routed through intent classification. The orchestrator checks these
 * BEFORE any LLM step, so a send only fires on an explicit `approve <id>` from an
 * allowlisted operator, deterministically, and every send is logged. The OpenClaw
 * agent merely relays the operator's text; it has no "send" capability.
 */
import { config } from '../config.js';
import { approveAndSend, cancelDraft, type SendFn } from '../email/email.js';
import type { DraftStore } from '../email/drafts.js';

function isOperator(userId: string): boolean {
  return config.email.allowlist.length === 0 || config.email.allowlist.includes(userId);
}

const APPROVE = /^\s*(?:approve|send it|批准|通过|确认发送|确认)\s*#?(\d+)\s*$/i;
const CANCEL = /^\s*(?:cancel|discard|取消|作废|不发)\s*#?(\d+)\s*$/i;
const LIST = /^\s*(?:drafts|list drafts|show drafts|pending drafts?|草稿列表|待发草稿|待审草稿)\s*$/i;

/** Returns a reply string if the message is a draft command, else null. */
export async function handleDraftCommand(
  userId: string,
  message: string,
  store: DraftStore,
  send?: SendFn,
): Promise<string | null> {
  let m: RegExpMatchArray | null;

  if ((m = message.match(APPROVE))) {
    if (!isOperator(userId)) return 'Only the operator can approve and send emails.';
    const r = await approveAndSend(Number(m[1]), userId, store, send);
    switch (r.status) {
      case 'sent':
      case 'sent_dryrun':
        return `✅ Sent draft #${m[1]} to ${r.draft?.recipients.join(', ')}.`;
      case 'already_sent': return `Draft #${m[1]} was already sent.`;
      case 'not_found': return `Draft #${m[1]} not found.`;
      case 'not_pending': return `Draft #${m[1]} isn't pending approval.`;
      case 'unauthorized': return "You're not authorized to send email.";
    }
  }

  if ((m = message.match(CANCEL))) {
    if (!isOperator(userId)) return 'Only the operator can manage drafts.';
    const ok = await cancelDraft(Number(m[1]), store);
    return ok ? `Draft #${m[1]} cancelled.` : `Draft #${m[1]} can't be cancelled (not found or already sent).`;
  }

  if (LIST.test(message)) {
    if (!isOperator(userId)) return null; // don't leak drafts; fall through to normal handling
    const pending = await store.list('pending_approval');
    if (!pending.length) return 'No pending drafts.';
    return 'Pending drafts:\n'
      + pending.map((d) => `#${d.id} → ${d.recipients.join(', ')} — ${d.subject}`).join('\n')
      + '\n\nReply "approve <id>" to send, or "cancel <id>".';
  }

  return null;
}
