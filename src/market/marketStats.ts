/**
 * Market analytics over california_sold (Week 5).
 *
 * Improvement over the handbook: it reports "median" but its SQL only computed
 * AVG. Averages get dragged by luxury/foreclosure outliers, so we compute a TRUE
 * median (and $/sqft, DOM, list-to-sold ratio) in TS from the raw rows. Also a
 * 12-month trend. Dirty future CloseDates are excluded via the <= today bound.
 * Feeds Week 7 comp-based price validation.
 */
import type { RowDataPacket } from 'mysql2/promise';
import { query } from '../db.js';
import { col } from '../../schema/columns.js';

const CS = 'california_sold' as const;

export interface MonthPoint {
  month: string;            // 'YYYY-MM'
  count: number;
  medianPrice: number | null;
}

export interface MarketStats {
  city: string;
  window: { months: number; from: string; to: string };
  count: number;
  medianPrice: number | null;
  medianPricePerSqft: number | null;
  medianDom: number | null;
  /** Median of ClosePrice/ListPrice as a percent (100 = sold at asking). */
  medianListToSoldPct: number | null;
  trend: MonthPoint[];      // chronological, up to `months` points
}

export function median(nums: number[]): number | null {
  if (!nums.length) return null;
  const s = [...nums].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

const isoToday = () => new Date().toISOString().slice(0, 10);
function isoMonthsAgo(months: number): string {
  const d = new Date();
  d.setMonth(d.getMonth() - months);
  return d.toISOString().slice(0, 10);
}

interface SaleRow extends RowDataPacket {
  closeDate: string | null;
  closePrice: number | null;
  listPrice: number | null;
  sqft: number | null;
  dom: number | null;
}

export async function getMarketStats(city: string, months = 12): Promise<MarketStats> {
  const from = isoMonthsAgo(months);
  const to = isoToday();
  const rows = await query<SaleRow>(
    `SELECT
       ${col('closeDate', CS)}    AS closeDate,
       ${col('closePrice', CS)}   AS closePrice,
       ${col('price', CS)}        AS listPrice,
       ${col('livingArea', CS)}   AS sqft,
       ${col('daysOnMarket', CS)} AS dom
     FROM ${CS}
     WHERE ${col('city', CS)} = ?
       AND ${col('closePrice', CS)} > 0
       AND ${col('closeDate', CS)} >= ?
       AND ${col('closeDate', CS)} <= ?`,
    [city, from, to],
  );

  const prices: number[] = [];
  const ppsf: number[] = [];
  const doms: number[] = [];
  const ratios: number[] = [];
  const byMonth = new Map<string, number[]>();

  for (const r of rows) {
    const close = Number(r.closePrice);
    if (!Number.isFinite(close) || close <= 0) continue;
    prices.push(close);
    const sqft = Number(r.sqft);
    if (Number.isFinite(sqft) && sqft > 0) ppsf.push(close / sqft);
    const dom = Number(r.dom);
    if (r.dom != null && Number.isFinite(dom)) doms.push(dom);
    const list = Number(r.listPrice);
    if (Number.isFinite(list) && list > 0) ratios.push((close / list) * 100);
    const month = (r.closeDate ?? '').slice(0, 7);
    if (month) (byMonth.get(month) ?? byMonth.set(month, []).get(month)!).push(close);
  }

  const trend: MonthPoint[] = [...byMonth.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([month, ps]) => ({ month, count: ps.length, medianPrice: median(ps) }))
    .slice(-months);

  const round = (n: number | null) => (n == null ? null : Math.round(n));
  return {
    city,
    window: { months, from, to },
    count: prices.length,
    medianPrice: round(median(prices)),
    medianPricePerSqft: round(median(ppsf)),
    medianDom: round(median(doms)),
    medianListToSoldPct: median(ratios) == null ? null : Math.round(median(ratios)! * 10) / 10,
    trend,
  };
}

/** Compact human summary for WhatsApp/CLI. */
export function formatMarketStats(s: MarketStats): string {
  if (s.count === 0) return `No recent sales found for ${s.city}.`;
  const money = (n: number | null) => (n == null ? 'N/A' : `$${n.toLocaleString('en-US')}`);
  const lines = [
    `📊 ${s.city} — last ${s.window.months} months (${s.count} sales)`,
    `   median sold price: ${money(s.medianPrice)}`,
    `   median $/sqft:     ${money(s.medianPricePerSqft)}`,
    `   median days on market: ${s.medianDom ?? 'N/A'}`,
    `   sold-to-list:      ${s.medianListToSoldPct != null ? s.medianListToSoldPct + '%' : 'N/A'}`,
  ];
  if (s.trend.length >= 2) {
    const first = s.trend[0]!.medianPrice;
    const last = s.trend[s.trend.length - 1]!.medianPrice;
    if (first && last) {
      const pct = Math.round(((last - first) / first) * 1000) / 10;
      lines.push(`   trend (${s.trend[0]!.month}→${s.trend[s.trend.length - 1]!.month}): ${pct >= 0 ? '+' : ''}${pct}% median`);
    }
  }
  return lines.join('\n');
}
