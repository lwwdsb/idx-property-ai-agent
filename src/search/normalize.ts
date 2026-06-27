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

export function normalizeQuery(raw: string): string {
  let s = toHalfWidth(raw).replace(/\s+/g, ' ').trim();
  // spelled-out numbers -> digits (word-boundary, so "someone" is unaffected)
  for (const [word, digit] of Object.entries(NUM_WORDS)) {
    s = s.replace(new RegExp(`\\b${word}\\b`, 'gi'), digit);
  }
  for (const [re, rep] of SYNONYMS) s = s.replace(re, rep);
  return s.replace(/\s+/g, ' ').trim();
}
