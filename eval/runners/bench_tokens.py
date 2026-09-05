"""Token-usage probe: the retrieval path (hybrid + cross-encoder) uses ZERO LLM tokens
(both are local models). LLM tokens are spent UPSTREAM — extraction, translation, RAG
generation. This hits the DeepSeek API directly and reports usage.{prompt,completion,total}
for representative per-operation calls, so the (previously unmeasured) token cost is visible.

  python eval/runners/bench_tokens.py
Needs LLM_API_KEY. Rough: real auto tool-call extraction also ships the tool SCHEMAS, so its
prompt tokens are higher than the plain extraction probe here (noted).
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "retrieval"))
from common import load_env  # noqa: E402

env = load_env()
KEY = env.get("LLM_API_KEY", "").strip()
BASE = env.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL = env.get("LLM_MODEL", "gpt-4o-mini")


def usage(system, user):
    body = json.dumps({"model": MODEL, "temperature": 0.2,
                       "messages": ([{"role": "system", "content": system}] if system else [])
                       + [{"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        u = json.loads(r.read()).get("usage", {})
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("total_tokens", 0)


CASES = [
    ("① 跨语言翻译(#2)", None,
     "Translate this real-estate search query to natural English. Return ONLY the translation.\n\n"
     "尔湾赛普拉斯村带太阳能板的三居,一百五六十万"),
    ("② 参数提取(JSON,近似)", "Extract structured filters (city, beds, maxPrice, propertyType, semantic) "
     "as JSON from the real-estate query.",
     "looking for a mid-century place in Pasadena with ADU potential and city views, around 1.8M, 3 beds"),
    ("③ RAG 生成(带上下文)", "You are a precise real-estate assistant. Never invent facts beyond the context.",
     "Answer using ONLY the context. Cite sources in [brackets].\n\nContext:\n[terms.md > DOM]\nDays on "
     "market (DOM) is the number of days a listing is active before going under contract.\n\nQuestion: what does DOM mean?"),
]


def main():
    if not KEY:
        print("LLM_API_KEY not set"); sys.exit(1)
    print(f"token usage per operation (model={MODEL})\n")
    print(f"{'operation':26} {'prompt':>7} {'completion':>11} {'total':>7}")
    tot = 0
    for name, sysp, userp in CASES:
        p, c, t = usage(sysp, userp)
        tot += t
        print(f"{name:26} {p:>7} {c:>11} {t:>7}")
    print(f"\n注:检索(hybrid + cross-encoder 精排)= 0 LLM token(本地模型)。")
    print(f"auto 每步 tool-call 抽参还会附全部工具 schema → prompt token 高于上面②(此为近似下界)。")


if __name__ == "__main__":
    main()
