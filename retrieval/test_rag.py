"""Week 8 RAG correctness (offline & hermetic: a no-LLM chat_fn is injected so the
test is deterministic and exercises the extractive/grounded path even when a real
LLM key is configured in .env). Verifies the question types retrieve the right
source and answers stay grounded.

  python retrieval/test_rag.py
"""
from rag import RagIndex, answer

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print("✓", name)
    else:
        failed += 1; print("✗", name)


idx = RagIndex()
no_llm = lambda *a, **k: None                    # force extractive (ignore any live key)
A = lambda q: answer(q, idx, chat_fn=no_llm)     # hermetic answer helper

check("index built from KB", len(idx.chunks) > 5)

# --- concept question ---
r = A("what does days on market mean?")
check("DOM: top source is Days on Market", "Days on Market" in r["sources"][0])

# --- term/concept: list-to-sold ---
r = A("how is the sold to list ratio calculated?")
check("ratio: source mentions List-to-Sold", any("List-to-Sold" in s for s in r["sources"]))

# --- concept: comps + verifier rule embedded in KB ---
r = A("what are comparable sales and how many do you need?")
check("comps: source is Comparable Sales", any("Comparable Sales" in s for s in r["sources"]))

# --- field question (glossary generated from schema/columns) ---
r = A("which column stores the number of bedrooms?")
check("field: source from fields.md", any("fields.md" in s for s in r["sources"]))
check("field: answer mentions the bedrooms field", "Bedroom" in r["answer"] or "bed" in r["answer"].lower())

# --- grounding + fallback behavior (no key) ---
r = A("what is price per square foot?")
check("no-key -> extractive mode", r["mode"] == "extractive")
check("answer is verbatim from a KB chunk (grounded)",
      any(r["answer"][:40] in c["text"] for c in idx.chunks))
check("sources are cited", len(r["sources"]) >= 1)

# --- irrelevant question still returns a source, doesn't crash ---
r = A("what is the capital of France?")
check("off-topic: no crash, still cites nearest", len(r["sources"]) >= 1)

print(f"\n{passed}/{passed + failed} passed")
raise SystemExit(1 if failed else 0)
