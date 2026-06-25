/**
 * Week 1 skeleton smoke test — proves the spine runs end-to-end:
 *   1. config loads + validates
 *   2. field dictionary resolves semantic -> physical columns
 *   3. logger redacts secrets
 *   4. DB pool connects + returns row counts
 *   5. a tool runs and the message router routes to it
 *
 * Run: npm run smoke
 */
import { config } from './config.js';
import { logger } from './logger.js';
import { healthCheck, closePool } from './db.js';
import { handleMessage } from './handleMessage.js';
import { col, FIELDS } from '../schema/columns.js';

async function main() {
  logger.info('=== IDX skeleton smoke test ===');

  // 1. config
  logger.info('config loaded', { db: config.db.database, chatModel: config.openai.chatModel });

  // 2. field dictionary (semantic -> physical)
  logger.info('field dictionary', {
    'beds@rets_property': col('beds', 'rets_property'),       // L_Keyword2
    'beds@california_sold': col('beds', 'california_sold'),   // BedroomsTotal
    'price@rets_property': col('price', 'rets_property'),     // L_SystemPrice
    fieldCount: Object.keys(FIELDS).length,
  });

  // 3. logger redaction (the password must NOT appear in output)
  logger.info('redaction check', { DB_PASSWORD: config.db.password, note: 'value should show as REDACTED' });

  // 4. DB connectivity
  const counts = await healthCheck();
  logger.info('db health', counts);

  // 5. tool + router
  const timeResult = await handleMessage('what time is it?');
  const unknownResult = await handleMessage('hello there');
  logger.info('router results', { timeResult, unknownResult });

  await closePool();
  logger.info('=== smoke test OK ===');
}

main().catch((err) => {
  logger.error('smoke test FAILED', { error: String(err) });
  process.exitCode = 1;
});
