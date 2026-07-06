"""Week 8: RAG knowledge Q&A over a small curated corpus (knowledge/*.md).

The corpus (field glossary generated from schema/columns + a hand-written terms doc)
is tiny, so retrieval is exact in-memory (fastembed dense + cosine) — no Qdrant/ANN
needed. Answers are GROUNDED in retrieved chunks and cite their source. The generation
step is a reserved LLM slot (llm.chat): with a key it writes a grounded answer; without
one it falls back to an extractive answer (the top chunk text + citation) — never
invents facts (Q5).

  python retrieval/rag.py "what does DOM mean?"
  python retrieval/rag.py "which column is bedrooms?"
"""
import glob
import os
import re
import sys

import numpy as np

from common import get_dense
from llm import chat, llm_available

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")
CHUNK_SIZE, OVERLAP = 600, 100


def _split(text, size=CHUNK_SIZE, overlap=OVERLAP):
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def load_chunks():
    """Parse markdown into (source, text) chunks, split by section then by size."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
        fname = os.path.basename(path)
        raw = open(path, encoding="utf-8").read()
        # split on ## headings, keep the heading with its body
        sections = re.split(r"\n(?=## )", raw)
        for sec in sections:
            m = re.match(r"##\s+(.+)", sec)
            heading = m.group(1).strip() if m else fname
            body = sec.strip()
            if len(body) < 15:
                continue
            for piece in _split(body):
                chunks.append({"source": f"{fname} › {heading}", "text": piece})
    return chunks


class RagIndex:
    def __init__(self):
        self.chunks = load_chunks()
        vecs = list(get_dense().embed([c["text"] for c in self.chunks]))
        m = np.array([v for v in vecs], dtype=np.float32)
        self.mat = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    def retrieve(self, question, k=3):
        q = list(get_dense().embed([question]))[0].astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self.mat @ q
        idx = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in idx]


def answer(question, index=None, k=3):
    index = index or RagIndex()
    hits = index.retrieve(question, k)
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    sources = [h["source"] for h in hits]

    # reserved LLM generation slot — grounded answer with citations
    generated = chat(
        f"Answer the question using ONLY the context. Cite sources in [brackets]. "
        f"If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}",
        system="You are a precise real-estate assistant. Never invent facts beyond the context.",
    )
    if generated:
        return {"mode": "generated", "answer": generated, "sources": sources}

    # extractive fallback (no key): return the most relevant chunk, grounded + cited
    top = hits[0]
    return {"mode": "extractive", "answer": top["text"], "sources": sources,
            "note": "no LLM key — returning the most relevant passage verbatim"}


def format_answer(res):
    lines = [res["answer"], "", "Sources: " + "; ".join(res["sources"])]
    if res.get("note"):
        lines.append(f"({res['note']})")
    return "\n".join(lines)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "what is DOM?"
    print(f"Q: {q}  [llm={'on' if llm_available() else 'off — extractive'}]\n")
    print(format_answer(answer(q)))
