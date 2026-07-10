/**
 * Regex fast-path parser (no LLM): handles well-formed EN/中文 queries cheaply.
 * This is the Week-2 cost/latency optimization AND the Week-11 (乙) fallback when
 * the LLM is unavailable. Anything it can't confidently extract (e.g. city in a
 * lowercase/fuzzy query) is left undefined — a structural signal for the caller
 * to clarify or escalate to the LLM.
 */
import type { SearchFilter } from './filters.js';

// A money token must carry a currency signal so bare numbers (e.g. "2000 sqft",
// a year) are NOT mistaken for prices: $ prefix, OR k/m/万/million suffix, OR
// comma-grouped thousands.
const MONEY = String.raw`(?:\$\s*[\d.,]+(?:\s*(?:万|million|mil|m|k|thousand))?|[\d.]+\s*(?:万|million|mil|m|k|thousand)|\d{1,3}(?:,\d{3})+)`;

/** Parse a money token to a number. Handles $, commas, k/m/million, and 中文 万. */
export function parseMoney(raw: string): number | undefined {
  const s = raw.trim().toLowerCase().replace(/[$,\s]/g, '');
  let m: RegExpMatchArray | null;
  if ((m = s.match(/^(\d*\.?\d+)万$/))) return Math.round(parseFloat(m[1]!) * 1e4);
  if ((m = s.match(/^(\d*\.?\d+)(?:m|million|mil)$/))) return Math.round(parseFloat(m[1]!) * 1e6);
  if ((m = s.match(/^(\d*\.?\d+)(?:k|thousand)$/))) return Math.round(parseFloat(m[1]!) * 1e3);
  if ((m = s.match(/^(\d*\.?\d+)$/))) return Math.round(parseFloat(m[1]!));
  return undefined;
}

const TYPE_PATTERNS: Array<[RegExp, string]> = [
  [/\b(?:single[\s-]?family|sfr|detached)\b|独栋|单户/i, 'single-family'],
  [/\b(?:town\s?houses?|town\s?homes?)\b|联排/i, 'townhouse'],
  [/\b(?:condos?|condominiums?|apartments?|apts?)\b|公寓/i, 'condo'],
];

const CITY_STOPWORDS = new Set(['find', 'with', 'under', 'over', 'for', 'near', 'and', 'homes', 'home', 'house', 'houses', 'condo', 'condos']);

function extractCity(q: string): string | undefined {
  // English: "in/near/around/at <Capitalized words>"
  let m = q.match(/\b(?:in|near|around|at)\s+([A-Z][a-zA-Z.]+(?:\s+[A-Z][a-zA-Z.]+){0,2})/);
  if (m) return cleanCity(m[1]!);
  // 中文: "在 <Latin city> ..." (CA city names are latin even in zh queries)
  m = q.match(/在\s*([A-Za-z][A-Za-z\s.]*?)(?=找|的|附近|地区|有|[，,、]|\d|$)/);
  if (m && m[1]!.trim()) return cleanCity(m[1]!);
  // city-first: "Irvine market", "San Diego prices", "Irvine 行情"
  m = q.match(/\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(?:[Mm]arket|[Pp]rices?|[Hh]ousing|行情|房价)/);
  if (m) return cleanCity(m[1]!);
  return undefined;
}

function cleanCity(raw: string): string | undefined {
  const words = raw.trim().split(/\s+/).filter((w) => !CITY_STOPWORDS.has(w.toLowerCase()));
  const city = words.join(' ').trim();
  return city || undefined;
}

function priceWithDirection(q: string): { maxPrice?: number; minPrice?: number } {
  const out: { maxPrice?: number; minPrice?: number } = {};
  let m: RegExpMatchArray | null;
  // max: "under/below/less than/up to/max X"  |  "X 以下/以内/or less"  |  "<= X"
  if ((m = q.match(new RegExp(String.raw`(?:under|below|less than|up to|max(?:imum)?|cheaper than|<=?|≤)\s*(${MONEY})`, 'i')))) {
    out.maxPrice = parseMoney(m[1]!);
  } else if ((m = q.match(new RegExp(String.raw`(${MONEY})\s*(?:以下|以内|or less|or under)`, 'i')))) {
    out.maxPrice = parseMoney(m[1]!);
  }
  // min: "over/above/more than/at least/from X"  |  "X 以上/起"  |  ">= X"
  if ((m = q.match(new RegExp(String.raw`(?:over|above|more than|at least|starting (?:at|from)|min(?:imum)?|>=?|≥)\s*(${MONEY})`, 'i')))) {
    out.minPrice = parseMoney(m[1]!);
  } else if ((m = q.match(new RegExp(String.raw`(${MONEY})\s*(?:以上|起)`, 'i')))) {
    out.minPrice = parseMoney(m[1]!);
  }
  // range: "between X and/to Y" | "X-Y" | "X到/至Y"
  if (out.maxPrice === undefined && out.minPrice === undefined) {
    if ((m = q.match(new RegExp(String.raw`(?:between|from)\s*(${MONEY})\s*(?:and|to|-|~|到|至)\s*(${MONEY})`, 'i')))) {
      out.minPrice = parseMoney(m[1]!);
      out.maxPrice = parseMoney(m[2]!);
    }
  }
  return out;
}

/** Best-effort structured extraction. Missing fields are simply left out. */
export function regexParse(query: string): SearchFilter {
  const q = query.trim();
  const f: SearchFilter = {};

  const city = extractCity(q);
  if (city) f.city = city;

  let m: RegExpMatchArray | null;
  if ((m = q.match(/(\d+(?:\.\d+)?)\s*\+?\s*(?:bed(?:room)?s?|br|bd)\b/i)) ||
      (m = q.match(/(\d+)\s*(?:居室|室|卧室|卧|房)/))) {
    f.beds = Number(m[1]);
  }
  if ((m = q.match(/(\d+(?:\.\d+)?)\s*\+?\s*(?:bath(?:room)?s?|ba)\b/i)) ||
      (m = q.match(/(\d+(?:\.\d+)?)\s*卫/))) {
    f.baths = Number(m[1]);
  }
  if ((m = q.match(/(\d[\d,]*)\s*(?:sq\.?\s?ft|sqft|square\s?feet)/i)) ||
      (m = q.match(/(\d[\d,]*)\s*(?:平方英尺|平方呎)/))) {
    f.minSqft = Number(m[1]!.replace(/,/g, ''));
  }
  // pool is tri-state: honor negation so "no pool"/"不要泳池" isn't a false positive.
  const NO_POOL = /(?:no|without|not|non-?|excluding|don'?t\s+(?:want|need))\s+(?:a\s+)?(?:private\s+)?pools?\b|(?:不要|不带|不需要|没有|无|不含|去掉|去除|取消)\s*(?:私家|私人)?(?:泳池|游泳池)/i;
  const HAS_POOL = /\bpools?\b|泳池|游泳池/i;
  if (NO_POOL.test(q)) f.pool = false;
  else if (HAS_POOL.test(q)) f.pool = true;

  for (const [re, type] of TYPE_PATTERNS) {
    if (re.test(q)) { f.propertyType = type; break; }
  }

  const price = priceWithDirection(q);
  if (price.maxPrice !== undefined) f.maxPrice = price.maxPrice;
  if (price.minPrice !== undefined) f.minPrice = price.minPrice;

  return f;
}
