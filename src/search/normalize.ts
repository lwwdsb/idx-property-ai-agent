/**
 * Deterministic query normalization (no LLM) — hardens the regex parser cheaply.
 * Handles full-width chars (common from Chinese IMEs), spelled-out numbers, and a
 * few safe synonyms/abbreviations. Does NOT lowercase (English city extraction
 * relies on capitalization).
 */

// Full-width digits/punctuation -> half-width.
function toHalfWidth(s: string): string {
  return s.replace(/[！-～]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0xFEE0))
    .replace(/　/g, ' '); // ideographic space
}

// ── Chinese numerals ──────────────────────────────────────────────────────────
// The regex parser requires DIGITS: MONEY needs `[\d.]+万` and the bed pattern needs `(\d+)`.
// So "一百二十万" and "三居" extracted nothing at all, and a Chinese query reached Qdrant with
// an EMPTY hard filter plus its price and bed count still sitting in the semantic text as
// noise against an English corpus (measured: 0/14 known-item hits on the deterministic path).
// Converting here — rather than adding Chinese branches to every downstream regex — keeps the
// parser single-shaped and is the same tactic already used for English spelled-out numbers.
const CN_D: Record<string, number> = { 零: 0, 〇: 0, 一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9 };
const CN_U: Record<string, number> = { 十: 10, 百: 100, 千: 1000 };

/** "一百二十" -> 120. Returns null if the string is not a pure numeral. */
export function cnNumeral(raw: string): number | null {
  // Vague ranges: "一百五六十万" means 1.5-1.6M, and as a BUDGET CEILING the upper bound is
  // what the user means, so the second digit wins: 五六十 -> 六十.
  const s = raw.replace(/([一二三四五六七八九])([一二三四五六七八九])(?=[十百千])/g, '$2');
  let total = 0, section = 0, digit = -1;
  for (const ch of s) {
    if (ch in CN_D) { digit = CN_D[ch]!; }
    else if (ch in CN_U) { section += (digit < 0 ? 1 : digit) * CN_U[ch]!; digit = -1; }
    else return null;
  }
  if (digit >= 0) {
    // Colloquial shorthand: "一百八" is 180, not 108 — a bare digit after 百/千 fills the next
    // place down. Only applied when a unit preceded it, so "八十五" still parses as 85.
    const last = s[s.length - 2];
    const shift = last === '百' ? 10 : last === '千' ? 100 : 1;
    section += digit * shift;
  }
  return total + section;
}

const CN_NUM_CHARS = '零〇一二三四五六七八九十百千两';
/** Rewrite a Chinese numeral to digits ONLY when a unit follows, so names and addresses that
 *  merely contain these characters are left alone. */
function cnNumeralsToDigits(s: string): string {
  const unit = '万|千万|亿|居室|居|室|卧室|卧|房|卫|层|英亩|亩|平米|平方米|平';
  return s.replace(new RegExp(`([${CN_NUM_CHARS}]{1,6})(?=${unit})`, 'g'),
    (m) => { const n = cnNumeral(m); return n === null ? m : String(n); });
}

const NUM_WORDS: Record<string, string> = {
  one: '1', two: '2', three: '3', four: '4', five: '5',
  six: '6', seven: '7', eight: '8', nine: '9', ten: '10',
};

// Conservative, low-misfire synonyms/abbreviations.
const SYNONYMS: Array<[RegExp, string]> = [
  [/\btown\s?home\b/gi, 'townhouse'],
  [/\bbdrm?s?\b/gi, 'bedroom'],
  [/\bsfr\b/gi, 'single family'],
  [/\bw\/\s*/gi, 'with '],          // "3bd w/ pool"
];

// Chinese city names -> their Latin form (CA cities are stored latin in the data).
// Replaced (spaced) so the downstream city regex can extract them — e.g. 尔湾行情 -> Irvine 行情.
const CITY_ALIASES: Array<[RegExp, string]> = [
  [/尔湾/g, 'Irvine'], [/洛杉矶|洛城/g, 'Los Angeles'], [/圣地亚哥|圣地牙哥/g, 'San Diego'],
  [/旧金山|三藩市/g, 'San Francisco'], [/圣何塞|圣荷西/g, 'San Jose'], [/帕萨迪纳|帕萨迪那/g, 'Pasadena'],
  [/富勒顿/g, 'Fullerton'], [/阿凯迪亚|阿凱迪亞/g, 'Arcadia'], [/圣盖博/g, 'San Gabriel'],
  [/罗兰岗/g, 'Rowland Heights'], [/核桃市?/g, 'Walnut'], [/钻石吧/g, 'Diamond Bar'],
  [/天普市?/g, 'Temple City'], [/塔斯汀/g, 'Tustin'], [/亨廷顿海滩/g, 'Huntington Beach'],
  [/纽波特海滩/g, 'Newport Beach'], [/安纳海姆|阿纳海姆/g, 'Anaheim'], [/长滩/g, 'Long Beach'],
  [/奇诺岗/g, 'Chino Hills'], [/蒙特利公园|蒙市/g, 'Monterey Park'], [/阿罕布拉/g, 'Alhambra'],
  [/库比蒂诺/g, 'Cupertino'], [/比佛利山庄?|比华利山/g, 'Beverly Hills'], [/圣塔莫尼卡|圣莫尼卡/g, 'Santa Monica'],
  // Added after the Chinese known-item cases showed these spellings reaching Qdrant untranslated.
  // Several are ALTERNATE transliterations of cities already listed above (图斯汀 vs 塔斯汀,
  // 阿卡迪亚 vs 阿凯迪亚, 蒙特雷公园 vs 蒙特利公园) — a missing variant is indistinguishable
  // from a missing city, since the filter ends up empty either way.
  [/图斯汀/g, 'Tustin'], [/弗雷斯诺|弗雷斯诺市/g, 'Fresno'], [/弗里蒙特/g, 'Fremont'],
  [/萨克拉门托/g, 'Sacramento'], [/阿卡迪亚/g, 'Arcadia'], [/蒙特雷公园市?/g, 'Monterey Park'],
  [/圣贝纳迪诺/g, 'San Bernardino'], [/萨利纳斯/g, 'Salinas'], [/森林湖/g, 'Lake Forest'],
  [/托卢卡湖/g, 'Toluca Lake'], [/圣罗莎/g, 'Santa Rosa'], [/河滨市?/g, 'Riverside'],
  [/欧文戴尔/g, 'Irwindale'], [/惠提尔/g, 'Whittier'], [/柔斯密/g, 'Rosemead'],
  [/圣塔安娜|圣安娜/g, 'Santa Ana'], [/科罗娜/g, 'Corona'], [/奥克兰/g, 'Oakland'],
];

/**
 * The Latin city name whose Chinese alias appears in `raw`, if any.
 *
 * Re-extracting the city from normalized text does not work: the aliases splice a Latin name
 * into a Chinese sentence ("尔湾赛普拉斯村..." -> "Irvine 赛普拉斯村..."), and none of the
 * city patterns in regexParse cover that shape — they expect "in <City>", "在 <City>", or
 * "<City> 行情". So the city was silently dropped and the query reached Qdrant with an empty
 * filter AND the Latin city still sitting in its semantic text.
 *
 * Reporting the alias hit directly is both simpler and safer than another regex: the table is
 * a curated dictionary, so a match is definitive, whereas guessing "a Latin run next to CJK"
 * would also fire on ADU, HOA or an English project name — and a wrong guess becomes a
 * rejectedCity, which derails the whole turn.
 */
export function cityFromAlias(raw: string): string | undefined {
  const s = toHalfWidth(raw);
  let best: { at: number; city: string } | undefined;
  for (const [re, city] of CITY_ALIASES) {
    const m = s.match(new RegExp(re.source));       // fresh, non-global: we want the index
    if (m && m.index !== undefined && (!best || m.index < best.at)) best = { at: m.index, city };
  }
  return best?.city;   // leftmost wins, so "在尔湾或洛杉矶" takes the one the user led with
}

export function normalizeQuery(raw: string): string {
  let s = toHalfWidth(raw).replace(/\s+/g, ' ').trim();
  // resolve Chinese city names to their latin form (spaced so the city regex sees them)
  for (const [re, rep] of CITY_ALIASES) s = s.replace(re, ` ${rep} `);
  // spelled-out numbers -> digits (word-boundary, so "someone" is unaffected)
  for (const [word, digit] of Object.entries(NUM_WORDS)) {
    s = s.replace(new RegExp(`\\b${word}\\b`, 'gi'), digit);
  }
  s = cnNumeralsToDigits(s);
  for (const [re, rep] of SYNONYMS) s = s.replace(re, rep);
  return s.replace(/\s+/g, ' ').trim();
}
