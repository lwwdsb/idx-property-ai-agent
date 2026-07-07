/**
 * Email draft review CLI (Week 11) — the human approval surface.
 *
 *   npm run drafts                      # list pending drafts
 *   npm run drafts -- show <id>
 *   npm run drafts -- approve <id>      # the ONLY send trigger (human)
 *   npm run drafts -- cancel <id>
 *   npm run drafts -- report <city> <to@email>   # demo: draft a weekly market report
 */
import { config } from '../config.js';
import { MySqlDraftStore } from '../email/drafts.js';
import { draftEmail, approveAndSend, cancelDraft, previewDraft } from '../email/email.js';
import { weeklyMarketReport } from '../email/templates.js';
import { closePool } from '../db.js';

const store = new MySqlDraftStore();
const operator = config.email.allowlist[0] ?? 'cli-operator';

async function main() {
  const [cmd, ...args] = process.argv.slice(2);
  switch (cmd) {
    case 'show': {
      const d = await store.get(Number(args[0]));
      console.log(d ? previewDraft(d) : 'not found');
      break;
    }
    case 'approve': {
      const r = await approveAndSend(Number(args[0]), operator, store);
      console.log(`approve #${args[0]} -> ${r.status}`);
      break;
    }
    case 'cancel': {
      console.log(await cancelDraft(Number(args[0]), store) ? 'cancelled' : 'not cancellable');
      break;
    }
    case 'report': {
      const [city, to] = args;
      const { subject, body } = await weeklyMarketReport(city!);
      const r = await draftEmail({ createdBy: operator, recipients: [to!], subject, body }, store);
      console.log(r.ok ? `drafted #${r.draft!.id} (pending approval) — review with: npm run drafts -- show ${r.draft!.id}` : `error: ${r.error}`);
      break;
    }
    default: {
      const pending = await store.list('pending_approval');
      console.log(`Pending drafts (${pending.length}):`);
      for (const d of pending) {
        console.log(`  #${d.id}  to ${d.recipients.join(', ')}  —  ${d.subject}`);
      }
      console.log('\nReview: npm run drafts -- show <id> | approve <id> | cancel <id>');
    }
  }
  await closePool();
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
