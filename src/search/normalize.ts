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
  [/纽波特海滩/g, 'Newport Beach'], [/安纳海姆/g, 'Anaheim'], [/长滩/g, 'Long Beach'],
  [/奇诺岗/g, 'Chino Hills'], [/蒙特利公园|蒙市/g, 'Monterey Park'], [/阿罕布拉/g, 'Alhambra'],
  [/库比蒂诺/g, 'Cupertino'], [/比佛利山庄?|比华利山/g, 'Beverly Hills'], [/圣塔莫尼卡|圣莫尼卡/g, 'Santa Monica'],
];

export function normalizeQuery(raw: string): string {
  let s = toHalfWidth(raw).replace(/\s+/g, ' ').trim();
  // resolve Chinese city names to their latin form (spaced so the city regex sees them)
  for (const [re, rep] of CITY_ALIASES) s = s.replace(re, ` ${rep} `);
  // spelled-out numbers -> digits (word-boundary, so "someone" is unaffected)
  for (const [word, digit] of Object.entries(NUM_WORDS)) {
    s = s.replace(new RegExp(`\\b${word}\\b`, 'gi'), digit);
  }
  for (const [re, rep] of SYNONYMS) s = s.replace(re, rep);
  return s.replace(/\s+/g, ' ').trim();
}
