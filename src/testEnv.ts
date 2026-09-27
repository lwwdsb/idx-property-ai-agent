/**
 * Deterministic environment for the offline test suites — import this FIRST, before any
 * module that reads config.
 *
 * Why it exists: the email red-line tests inject a fake sender, but whether `approveAndSend`
 * takes the real-send path or the dry-run path came from the DEVELOPER's `.env`
 * (`emailConfigured()` needs smtpHost + user + password). With no EMAIL_* keys present the
 * send became a dry-run, the fake sender was never called, and `sentBox.length === 1`
 * failed — 4 tests across two suites went red for an environment reason, not a code reason.
 * A test baseline that depends on local secrets cannot gate anything.
 *
 * These values are set BEFORE `dotenv/config` runs, and dotenv does not override existing
 * process.env entries — so the suites behave identically whether or not `.env` has real
 * credentials. Nothing is delivered either way: every test injects its own sender.
 */
process.env.EMAIL_FROM ??= 'test-operator@example.com';
process.env.EMAIL_USER ??= 'test-operator@example.com';
process.env.EMAIL_PASSWORD ??= 'test-only-not-a-real-credential';
process.env.EMAIL_SMTP_HOST ??= 'smtp.example.invalid';
process.env.EMAIL_SMTP_PORT ??= '587';
// A non-empty allowlist is required for the authorization tests to mean anything: an EMPTY
// allowlist is single-operator/dev mode and authorizes EVERYONE, so "non-operator cannot
// approve" silently passed authorization and failed its assertion instead.
process.env.EMAIL_ALLOWLIST ??= 'op';
