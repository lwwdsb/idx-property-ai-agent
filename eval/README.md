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

### L3 动态层 — `memory_dynamics.jsonl` / `evalMemoryDynamics.ts`（合并 · 冲突消解 · 压缩）

单测已覆盖 happy path（创建、合并、touch、淘汰）。这个集专测**容易坏且原先没覆盖**的性质：

| 性质 | 为什么重要 |
|---|---|
| **同名冲突消解** | 内容改变 = 偏好反转 → salience 跟随**新内容**而不是取 max。取 max 会把一个过时的高分挂在新内容上 |
| **压缩幂等** | `compactMemories` 跑在 `finally` 里、每条退出路径都执行——**不幂等就会每次整理都继续啃掉记忆** |
| **信号活性 ×3** | 三条探针各只变 `compScore` 的一个输入（salience / recency / frequency），其余全等。信号若是死的，排序就会打平、探针失败。**这不是假想**：曾经有个 `confidence` 字段从不被消费，只靠读代码才发现 |

**基线**：9/9 通过。三条活性探针全过 → **三个信号都活着**。

**两条记录性用例**（不是 bug，是需要被记住的事实）：

- `md-009` **已知缺口**：异名语义矛盾（两条内容相反但名字不同）**没有确定性网**——只有同名/同槽有确定消解，跨内容靠 LLM 子 agent 尽力、命名纪律承重。本条断言的是**当前行为**（两条都留下）并标记为缺口，**已知缺口不计入通过率**（与 completion 自评不进门禁同一纪律）。哪天有人补了确定性消解，这条会失败、评测会提醒。
- `md-010` **设计不一致**：`compScore`（压缩用）= salience × recency × frequency，而 `rank()`（无 LLM 选择降级用）= salience × recency，**不含 frequency**。所以同一批记忆在「压缩」和「降级选择」下排序可能不同。断言这个差异存在，以免它被当成 bug 反复发现——但值得记一笔：**被频繁用到的记忆在降级选择里得不到加分**。

### L4 效用层 — `memory_utility.jsonl` / `evalMemoryUtility.ts`

其余所有记忆指标测的是**机器能不能转**；这一层测的是**机器值不值得有**，也是唯一能给限额参数提供依据的层。

**构造上就是配对的**：同一输入跑两遍，一次带记忆一次不带。**决定性的断言不是"带记忆时成功"，而是两个 arm 必须有差异**——两边都成功说明那条记忆没起作用，而这正是本层要暴露的空结果。

三条**负对照**和正例一样重要，因为**一个什么都改的记忆系统和一个什么都不改的一样坏**：
`mu-003`（低于门槛 → 不得有影响）、`mu-004`（当轮明说 → 不得覆盖）、`mu-007`（无关记忆 → 不得给工具调用添约束）。

**基线**：7 条 6 通过 · **正例里记忆真正改变结果的比例 3/4** · 负对照 3/3。

> 🔍 **`mu-006` 失败，而这是本层最有价值的产出。** 两条 agent 用例对照：
>
> | 用例 | 记忆内容 | 有无差异 | 结果 |
> |---|---|---|---|
> | `mu-005` | 只看独栋 | ✅ differs | `propertyType` 进了工具参数 |
> | `mu-006` | 预算天花板 150 万 | ❌ **完全相同** | **`maxPrice` 从未到达工具调用** |
>
> 根因是**通道问题**，不是模型偶然抽风：`seedFilter` 只来自**事实层**（`preferredFilter` 读 `prefs`），
> 而语义记忆只能走 `profileHint` 的散文提示，**由模型自行决定要不要翻译成结构化参数**。
> 它翻了"只看独栋"，漏了"预算 150 万"。
>
> **一条映射到结构化槽位的记忆，本该走确定性通道，现在走的是模型自由裁量通道。**
> 修法方向：让语义记忆里能解析成 slot 的部分也参与 `seedFilter`，而不是只当提示词。未做。

### 门禁纳入

`make eval` 现在跑记忆四个 runner。门禁纳入 **6 项指标**（facts 通过率、注入 precision/recall、
dynamics 通过率、utility 通过率、**记忆改变结果的比例**）+ **2 条硬不变量**：

- `memory_facts.invariant_ok` —— 记忆不得覆盖当轮明说的字段（覆盖明确请求的偏好不是默认值，是劫持）
- `memory_utility.controls_held` —— 记忆**不得**在不该改的地方改变结果
