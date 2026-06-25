/**
 * Structured logger with mandatory secret redaction.
 * Safety red line (handbook): never print credentials. Any field whose key looks
 * like a secret (password/key/token/secret/authorization) is masked before output,
 * so secrets can't leak into logs even if accidentally passed in a context object.
 */

type Level = 'debug' | 'info' | 'warn' | 'error';
const ORDER: Record<Level, number> = { debug: 10, info: 20, warn: 30, error: 40 };

const MIN: Level = (process.env.LOG_LEVEL as Level) in ORDER
  ? (process.env.LOG_LEVEL as Level)
  : 'info';

const SECRET_KEY = /(pass|secret|token|api[_-]?key|key|authorization|auth)/i;

function redact(value: unknown, depth = 0): unknown {
  if (depth > 4 || value == null) return value;
  if (Array.isArray(value)) return value.map((v) => redact(v, depth + 1));
  if (typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = SECRET_KEY.test(k) ? '***REDACTED***' : redact(v, depth + 1);
    }
    return out;
  }
  return value;
}

function emit(level: Level, msg: string, ctx?: Record<string, unknown>) {
  if (ORDER[level] < ORDER[MIN]) return;
  const line: Record<string, unknown> = {
    ts: new Date().toISOString(),
    level,
    msg,
    ...(ctx ? { ctx: redact(ctx) } : {}),
  };
  const out = JSON.stringify(line);
  if (level === 'error' || level === 'warn') console.error(out);
  else console.log(out);
}

export const logger = {
  debug: (msg: string, ctx?: Record<string, unknown>) => emit('debug', msg, ctx),
  info: (msg: string, ctx?: Record<string, unknown>) => emit('info', msg, ctx),
  warn: (msg: string, ctx?: Record<string, unknown>) => emit('warn', msg, ctx),
  error: (msg: string, ctx?: Record<string, unknown>) => emit('error', msg, ctx),
};
