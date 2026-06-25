/**
 * MySQL access layer: a shared connection pool + a parameterized query helper.
 * All queries MUST go through `query()` with `?` placeholders — never string-concat
 * user input into SQL (injection guard). The ≤50-row result cap (handbook guardrail)
 * is enforced in the query builders added in Week 3, on top of this primitive.
 */
import mysql, { type Pool, type RowDataPacket } from 'mysql2/promise';
import { config } from './config.js';
import { logger } from './logger.js';

let pool: Pool | null = null;

export function getPool(): Pool {
  if (!pool) {
    pool = mysql.createPool({
      host: config.db.host,
      port: config.db.port,
      user: config.db.user,
      password: config.db.password,
      database: config.db.database,
      connectionLimit: 10,
      waitForConnections: true,
      namedPlaceholders: false,
    });
    logger.info('db pool created', { host: config.db.host, database: config.db.database });
  }
  return pool;
}

/** Parameterized query. Returns typed rows. */
export async function query<T extends RowDataPacket = RowDataPacket>(
  sql: string,
  params: ReadonlyArray<unknown> = [],
): Promise<T[]> {
  const [rows] = await getPool().query<T[]>(sql, params as unknown[]);
  return rows;
}

/** Health check: returns row counts for the three core tables. */
export async function healthCheck(): Promise<Record<string, number>> {
  const tables = ['rets_property', 'california_sold', 'rets_openhouse'];
  const out: Record<string, number> = {};
  for (const t of tables) {
    const rows = await query<RowDataPacket & { n: number }>(`SELECT COUNT(*) AS n FROM \`${t}\``);
    out[t] = Number(rows[0]?.n ?? 0);
  }
  return out;
}

export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
  }
}
