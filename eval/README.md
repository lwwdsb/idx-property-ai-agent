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

---

## 记忆评测体系（2026-09-27）

记忆分五层，能不能评取决于**有没有客观 gold**，而不是取决于难度。下表按
「客观性 × 成本」排序，也就是该建的顺序。

| 层 | 问的问题 | 客观性 | 状态 |
|---|---|---|---|
| **L1 事实层** | 结构化偏好学得对吗、应用时会不会覆盖用户 | 完全确定性，0 LLM | ✅ 已建 |
| **L2 选择层** | 该注入的记忆被选中了吗（注入准确率 / 召回准确率） | 客观（人工标 gold，无裁判） | ✅ 已建 |
| **L3 压缩层** | 衰减与容量淘汰保留了该保留的吗 | 完全确定性 | 单测已覆盖，未建评测集 |
| **L4 效用层** | 注入记忆**到底改变结果了吗** | 客观（配对：有记忆 / 无记忆） | ❌ 待建，见下 |
| **L5 整理层** | 子 agent 提炼得对吗、隔离成立吗 | 不变量客观，内容需人标 | 部分（9 条单测覆盖隔离与水位线） |

### L1 事实层 — `memory_facts.jsonl` / `evalMemoryFacts.ts`

两件事：**学习**（一个字段要同值出现约 3 次才过 0.5 门槛——单次提及不是偏好；异值**侵蚀**而非
替换，所以不会无脑相信最新一轮）和**应用**（记忆只能填**空白**）。

后者是**硬不变量**，与「真发邮件数 = 0」同级：*用户当轮说了的字段，记忆不得覆盖*。
一次违例就是失败，不看其他任何数字——**覆盖明确请求的偏好不是默认值，是劫持**。

> 📌 `mf-004` 这条的 gold 最初写错了：三次 200 万后改口 300 万，我以为字段仍会启用，
> 实测是 confidence 从 0.6 跌回 0.4、跌破门槛、**整个字段停用**。而这比两个替代方案都安全——
> 守着 200 万是过时，跳到 300 万是过度相信单次证据。**评测跑出来才发现是我的预期错了、代码是对的。**

### L2 选择层 — `memory_select.jsonl` / `evalMemorySelect.ts` + `report_memory_select.py`

`selectMemories` 本质是个检索问题，而且有客观 gold：给定记忆池和当前任务，人能说清哪几条相关。
所以**不需要 LLM 裁判**，标签是人工的、指标是 IR 指标、数字可以进门禁。

**两个 arm 打同一批用例**，这是重点：

| arm | 是什么 | 为什么要它 |
|---|---|---|
| `llm` | 生产路径，一次 LLM 调用读 name+description 选相关项 | 被测对象 |
| `fallback` | 无 key 时的降级：`salience × recency × frequency` | **它完全忽略任务**，所以是诚实的地板。LLM 那次调用必须显著跑赢它，否则不值 |

三条是**探针**而非普通样例：

- `ms-005` / `ms-009` —— gold 为**空**（寒暄、域外）。正确答案是**什么都不注入**，
  所以任何选择都是误注入；一个"总要选点什么"的选择器在这两条上不可能得分。
- `ms-010` —— 当轮显式给了新预算，**旧的预算记忆必须不被选中**。
  「软默认不绑架当前请求」这条性质，从注释里的断言变成了被测量的数字。

**基线（2026-09-27）**：

| arm | precision | recall | exact | 探针 P | 多选 | 漏选 |
|---|---|---|---|---|---|---|
| llm | **0.900** | 0.729 | 0.600 | **1.000** | 2 | 5 |
| fallback | 0.380 | 1.000 | 0.000 | 0.000 | 24 | 0 |

配对（逐用例 exact match）：llm **6 胜 / fallback 0 胜**，符号检验 **p=0.031** ——
那次 LLM 调用可证明值得。

> 🐛 **这个集第一次跑就抓到一个真 bug。** 探针精度原本是 **0.000**：`jsonArray` 对三种情况
> 都返回 `[]` —— 找不到数组、JSON 解析失败、**以及模型真的答了 `[]`** —— 而调用方把所有空结果
> 都当成失败，降级去**排序全部记忆**。于是在寒暄和域外请求上（正确答案是什么都不注入）
> **整份画像被注入了 prompt**。而 prompt 里明明写着 "`[]` if none"：**代码问了一个它随后就丢掉的答案。**
> 修法是把"解析失败"（→ 降级排序）和"有效的空"（→ 就是不注入）分开。
> 探针精度 **0.000 → 1.000**、整体 precision **0.700 → 0.900**、配对 p **0.125 → 0.031**。

**已知短板（不掩饰）**：`recall 0.729` —— 选择器**偏保守、漏选**。10 条里漏 5 项，
典型是 `ms-008` 漏掉了解析「这套」所必需的那条情景记忆。`ms-010` 探针仍然失败：
它保留了过时的预算记忆。那条到底该由**选择层**丢掉、还是由**应用层**（`mergeFilter` 让当轮赢）
兜住，是个尚未裁决的设计问题——当前 L1 的硬不变量保证了它不会造成错误结果，但把矛盾的记忆
放进 prompt 仍然是在邀请混乱。

### L4 效用层 — 待建，而且它是最重要的一层

**现在完全没有「注入记忆是否有用」的评测。** 后果很具体：把限额从 5/3 调成 8/5 或 3/2，
**没有任何指标会动**——所以 `auto.memory.selectSemantic` / `selectEpisodic` 目前**不可调**。

最小可行形态：一组跨会话任务，正确答案**依赖一条被记住的偏好**（例如上一轮说过"只看独栋"，
这一轮不再重复说）。配对断言两件事：

1. 注入时任务成功
2. **不注入时任务失败** ← 关键

第 2 条才是判据。**如果不注入也成功，那这条记忆没有价值**，整个限额问题都是伪问题——
该修的是记忆的**相关性**，不是**数量**。

免费的先行信号：`useCount` 已经在记录每条记忆被选中的次数。**常年为 0** 说明筛选从不选它；
**限额被顶满**说明可能挤掉了有用的。这两个观测不用建评测集就能看，应该先看。

### 进 make eval 与门禁

`evalMemoryFacts.ts` · `evalMemorySelect.ts` · `report_memory_select.py` 已接进 `make eval`。
门禁纳入 3 项指标（facts 通过率、注入 precision、注入 recall）+ **1 条硬不变量**
（`invariant_ok`：记忆不得覆盖当轮明说的字段）。

记忆单测（`profile.test.ts` 14 条 + `consolidation.test.ts` 9 条）此前**不在 `test:all` 里**，
已加 `npm run test:memory` 并接入。
