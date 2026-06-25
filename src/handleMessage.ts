/**
 * Minimal message router (handbook Week 1 example shape).
 * This is a placeholder spine: Week 9 replaces the keyword check with the
 * orchestrator (LLM intent + params in one call, confidence-gated clarification).
 * Kept here so the skeleton runs end-to-end before any LLM/agent exists.
 */
import { getCurrentTime } from './tools/getCurrentTime.js';
import { logger } from './logger.js';

export interface MessageResult {
  response?: string;
  data?: unknown;
}

export async function handleMessage(message: string): Promise<MessageResult> {
  logger.info('handleMessage', { message });
  if (message.toLowerCase().includes('time')) {
    return { data: await getCurrentTime() };
  }
  return { response: 'I could not understand the request.' };
}
