/**
 * Multi-query retrieval (query rewrite to lift recall).
 *
 * Design: don't pick "the best rewrite" — generate variants, retrieve each, and FUSE by rank
 * (RRF), the same fusion already used for dense+BM25. Any rewrite method (multi-query variants,
 * a future HyDE doc) is just "one more list" into the same RRF, so items ranked high across
 * variants win. LLM unavailable -> degrade to the original query only (乙).
 *
 * Opt-in (a flag) because N variants = N retrieval calls (latency/cost); enable only once eval
 * shows recall actually improves — same discipline as tuning RRF's k.
 */
import type { LLMClient } from '../llm/client.js';

/** Generate up to N phrasings (incl. the original) of a real-estate semantic query. */
export async function expandQuery(query: string, llm?: LLMClient, n = 3): Promise<string[]> {
  const original = query.trim();
  if (!original || !llm?.chatWithTools || n <= 1) return original ? [original] : [];
  try {
    const turn = await llm.chatWithTools([
      { role: 'system', content: `Rewrite the real-estate search phrase into ${n - 1} alternative phrasings that mean the SAME thing `
        + 'but use different words/angles (synonyms, related styles/features). Return ONLY a JSON array of strings.' },
      { role: 'user', content: original },
    ], []);
    const m = turn.content.match(/\[[\s\S]*\]/);
    const arr = m ? (JSON.parse(m[0]) as unknown[]) : [];
    const variants = arr.filter((s): s is string => typeof s === 'string' && s.trim().length > 0)
      .map((s) => s.trim()).slice(0, n - 1);
    return [...new Set([original, ...variants])];
  } catch {
    return [original];   // LLM down -> degrade to the original query only
  }
}

/**
 * Reciprocal Rank Fusion of multiple ranked lists. Fuses by RANK (no score normalization
 * needed) so it works across heterogeneous rewrite methods. k dampens the top-rank weight.
 */
export function rrfFuse<T>(lists: T[][], idOf: (x: T) => string | number, k = 60): T[] {
  const score = new Map<string | number, number>();
  const item = new Map<string | number, T>();
  for (const list of lists) {
    list.forEach((x, rank) => {
      const id = idOf(x);
      score.set(id, (score.get(id) ?? 0) + 1 / (k + rank + 1));
      if (!item.has(id)) item.set(id, x);
    });
  }
  return [...score.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => item.get(id)!);
}

/** 残余里有没有中日韩字符 —— 语料是英文的,所以这是个"这条查询用不了"的信号,不是风格问题。 */
export const hasCJK = (s: string): boolean => /[\u3400-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]/.test(s);

const trCache = new Map<string, string>();

/**
 * 把中文的语义残余翻成英文,因为房源 remark 是英文的。
 *
 * 为什么必须做:实测 44 条里有 16 条残余是中文,而 known-item 的分层召回是【英文 0.80 / 中文 0.21】。
 * 中文特征词(太阳能板 / 中古风 / 学区好 / 海边)直接进了英文索引,dense 向量和 BM25 词项都对不上,
 * 于是语义那一半完全失效,只剩硬过滤在干活。normalize 里只翻城市名,特征词一个都没翻。
 *
 * 为什么用 LLM 而不是再建一张词典:城市是闭集(972 个,可枚举),所以词典合适;而特征词汇是开放集
 * (通透、采光好、加建潜力、双车位……),建词典就是把"规则越堆越多"这个问题搬个地方重演。翻译是
 * 生成,只有 LLM 能做。
 *
 * 没有 LLM 时返回空串,让调用方降级到纯结构化搜索 —— 宁可丢掉特征,也不要拿中文去搜英文语料:
 * 后者不是"效果差一点",而是把一次语义检索变成了噪声,还顺带把路由判断带错(非空残余 = 走 hybrid)。
 */
export async function translateSemantic(text: string, llm?: LLMClient): Promise<string> {
  const q = text.trim();
  if (!q || !hasCJK(q)) return q;
  const hit = trCache.get(q);
  if (hit !== undefined) return hit;
  if (!llm?.chatWithTools) return '';
  try {
    const turn = await llm.chatWithTools([
      { role: 'system', content: 'Translate the real-estate feature/style phrase into ENGLISH terms that '
        + 'would appear in a US MLS listing remark. Keep it short, keep only feature/style words, drop any '
        + 'leftover instruction words or numbers. Return ONLY the English phrase, no quotes, no explanation.' },
      { role: 'user', content: q },
    ], []);
    const out = turn.content.trim().replace(/^["'`]|["'`]$/g, '');
    // 还带中文 = 没翻成,按失败处理,别把半成品送进索引
    const ok = out && !hasCJK(out) ? out : '';
    trCache.set(q, ok);
    return ok;
  } catch {
    return '';                        // 乙: 翻译失败 -> 降级结构化,不送噪声
  }
}
