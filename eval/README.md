# Evaluation framework

Turns quality from "it seems to work" into measured numbers: labeled datasets +
metrics (nDCG, F1, faithfulness, latency) + a report + regression tracking.

Run everything: `make eval`  →  produces `eval/report.md`.

## ⚠️ Confidential-data rule

MLS data is confidential and must never be committed. In eval files:

- **Committable**: query text, listing **ids**, relevance grades, intent/parse labels,
  RAG questions + gold source names. These carry no confidential listing text.
- **Local-only (gitignored)**: anything embedding listing **remarks / descriptions** —
  use the `*.raw.jsonl` / `*.pool.jsonl` suffix, which `.gitignore` excludes.
- `eval/history/` and `eval/report.md` are gitignored (per-run artifacts).

## Dataset format (JSONL, one record per line)

Every record shares a common envelope:

```json
{"id": "ret-001", "input": "...", "label": {...}, "meta": {"source": "llm-gen|human|synthetic", "verified": true}}
```

- `source`: how the label was produced. `llm-gen` = DeepSeek-proposed; `human` =
  hand-labeled; `synthetic` = rule/template.
- `verified`: whether a human spot-checked this record. The runners report the
  **human/LLM agreement rate** over the verified subset (honest calibration — we do not
  pretend LLM labels are 100% trustworthy).

### Per-dataset `label` shape

| dataset | `input` | `label` |
|---|---|---|
| `intent.jsonl` | user message | `{"intents": ["search"]}` (list — supports multi-intent) |
| `parse.jsonl` | user message | `{"filter": {"city": "Irvine", "beds": 3, ...}, "escalate": false}` |
| `retrieval.jsonl` | search query | `{"relevant": {"<listing_id>": 2, ...}}` graded 0/1/2 (pooled candidates only) |
| `rag.jsonl` | question | `{"in_corpus": true, "gold_sources": ["terms.md › Days on Market"]}` |
| `recommend.jsonl` | seed listing id | `{"good": ["<id>", ...], "bad": ["<id>", ...]}` |

## Layout

```
eval/
  datasets/   # the labeled sets above (committable subset)
  metrics/    # reusable metric functions (ir.py, classification.py, ...)
  runners/    # per-capability eval scripts + LLM-assisted labeling helpers
  history/    # per-run metric JSON (gitignored) — regression tracking
  report.md   # generated report (gitignored)
```
