/**
 * Message router. Routes a user message to the right handler. Week 9 replaces this
 * with the full orchestrator; for now: "time" -> getCurrentTime, else -> the
 * multi-turn search agent (Week 4).
 */
import { getCurrentTime } from './tools/getCurrentTime.js';
import { handleSearchTurn } from './agent/conversation.js';
import { logger } from './logger.js';

export interface MessageResult {
  response?: string;
  data?: unknown;
}

export async function handleMessage(message: string, userId = 'default'): Promise<MessageResult> {
  logger.info('handleMessage', { userId, message });
  if (/\btime\b/i.test(message)) {
    return { data: await getCurrentTime() };
  }
  const turn = await handleSearchTurn(userId, message);
  return { response: turn.reply, data: { kind: turn.kind, filter: turn.filter } };
}
