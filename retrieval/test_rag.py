"""Week 8 RAG correctness (offline: no Qdrant, no LLM key -> extractive mode).
Verifies the three question types retrieve the right source and answers stay grounded.

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
check("index built from KB", len(idx.chunks) > 5)

# --- concept question ---
r = answer("what does days on market mean?", idx)
check("DOM: top source is Days on Market", "Days on Market" in r["sources"][0])

# --- term/concept: list-to-sold ---
r = answer("how is the sold to list ratio calculated?", idx)
check("ratio: source mentions List-to-Sold", any("List-to-Sold" in s for s in r["sources"]))

# --- concept: comps + verifier rule embedded in KB ---
r = answer("what are comparable sales and how many do you need?", idx)
check("comps: source is Comparable Sales", any("Comparable Sales" in s for s in r["sources"]))

# --- field question (glossary generated from schema/columns) ---
r = answer("which column stores the number of bedrooms?", idx)
check("field: source from fields.md", any("fields.md" in s for s in r["sources"]))
check("field: answer mentions the bedrooms field", "Bedroom" in r["answer"] or "bed" in r["answer"].lower())

# --- grounding + fallback behavior (no key) ---
r = answer("what is price per square foot?", idx)
check("no-key -> extractive mode", r["mode"] == "extractive")
check("answer is verbatim from a KB chunk (grounded)",
      any(r["answer"][:40] in c["text"] for c in idx.chunks))
check("sources are cited", len(r["sources"]) >= 1)

# --- irrelevant question still returns a source, doesn't crash ---
r = answer("what is the capital of France?", idx)
check("off-topic: no crash, still cites nearest", len(r["sources"]) >= 1)

print(f"\n{passed}/{passed + failed} passed")
raise SystemExit(1 if failed else 0)
