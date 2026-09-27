# config/tuning.json — 可调参数的单一来源

所有会被 sweep 改动的算法参数都在 `tuning.json` 里。**环境/部署值（DB 凭据、端口、密钥）仍在 `.env`**——
两者写者不同、生命周期不同：`.env` 由人写一次，`tuning.json` 由调参回路程序化改写。
（和 user profile 按写者拆文件是同一个理由。）

读取器：`src/tuning.ts`（TS）/ `retrieval/tuning.py`（Python），都从**文件自身位置**解析路径而非 cwd，
因为两个服务的启动目录不同（orchestrate 从仓库根，uvicorn 从 `retrieval/`）。
两个读取器都**故意不做 try/catch**——配置缺失或损坏必须在启动时大声失败，
静默回退到硬编码默认值会让 sweep 对比两次"其实用了同一组参数"的运行。

## 用另一份配置跑一次（sweep 测候选值的方式）

```bash
IDX_TUNING=/abs/path/candidate.json npm run test:orch
IDX_TUNING=/abs/path/candidate.json uvicorn service:app --port 8099   # 从 retrieval/ 起
```

## 顶层按影响面（blast radius）分区

分区不是为了整洁，是为了**让 sweep 从配置结构就知道该跑哪套评测**，而不依赖人记得。

| 区 | 含义 | 改动后必须跑 |
|---|---|---|
| `shared` | 两条路都受影响。auto 复用同一个 `SkillRegistry`（`whatsapp/openclaw.ts:28` 只建了一个），所以 skill 层参数天然共享 | **确定性 + auto 两套评测都不许退化** |
| `deterministic` | 只影响确定性路由。auto 靠 function calling 选工具，从不经过意图分类 | 意图/解析评测 |
| `auto` | 只影响 ReAct 循环 | agent 评测（**但不进自动回路**，见下） |

## 参数清单

### shared.retrieval

| 键 | 当前值 | 原来位置 | 说明 / 来源 |
|---|---|---|---|
| `rerankCoarse` | 20 | `service.py` | 交给 cross-encoder 的粗排池大小。70 条分级集上扫出：20 带来 +0.14 nDCG@10，30 只有 +0.06；超过 ~50 精排转负（从太深的位置提上来的是噪声） |
| `prefetch` | 30 | `search.py` | RRF 融合前每条路径（dense / bm25）各取多深。与 `rerankCoarse` 是**两个独立旋钮** |
| `rerankEnabled` | true | `service.py` | `/search` 是否开精排。代价：p50 从 ~10ms 涨到 ~158ms，换 P@5 0.76→0.92 |
| `topK` | 5 | `service.py` | `SearchReq.k` 默认值 |

### shared.search / shared.rag / shared.facts

| 键 | 当前值 | 原来位置 | 说明 |
|---|---|---|---|
| `search.maxResults` | 50 | `searchListings.ts` | 手册护栏，查询层集中强制，调用方改不了 |
| `search.tooMany` | 200 | `searchListings.ts` | 超过就触发澄清而不是甩一屏结果。**注意它影响澄清触发率，不只是检索** |
| `rag.chunkSize` / `chunkOverlap` | 600 / 100 | `rag.py` | 知识库切块。改动后需重建 RAG 索引 |
| `rag.topK` | 3 | `rag.py` | 检索几块喂给生成。**与拒答阈值联动**——HyDE 实验里改写会抬高相似度，阈值门要跟着上调 |
| `facts.confidenceThreshold` | 0.5 | `profile.ts` | 事实层启用门槛（约需同值出现 3 次）。**事实是两模式共享层** |

### deterministic.intent

| 键 | 当前值 | 说明 / 来源 |
|---|---|---|
| `embedThreshold` | 0.58 | 从 0.55 扫上来的。取的是「**域内误伤 ≤5% 约束下**」的最优，不是 F1 最优——F1 最优 0.74 会误伤 36% 域内查询 |
| `embedMargin` | 0.05 | top1−top2 的最小差距。拍的保守值 + 二维扫描（score × margin）验证在安全区。这两个参数把域外拒识从 17% 提到 62.9%、macro-F1 0.63→0.80 |

### auto.loop / auto.memory

| 键 | 当前值 | 说明 |
|---|---|---|
| `loop.maxSteps` | 8 | 步数预算（防跑飞/成本） |
| `loop.maxPerTool` | 3 | 同一工具的软上限，超了回灌提示让它收尾 |
| `memory.selectSemantic` | 5 | 语义记忆注入上限 |
| `memory.selectEpisodic` | 3 | 情景记忆注入上限。分类型限额是为了**防情景挤掉偏好** |

## 不进自动调参回路的参数

| 排除项 | 理由 |
|---|---|
| **RRF 的 k** | 生产用 Qdrant **内置** RRF（`search.py:46` 的 `FusionQuery`），k 固定在服务端、API 不暴露。要让它可调必须把融合挪到客户端自己做——多一次数据传输、自己实现、延迟变化。**这是架构改动，不是旋钮**，所以压根没进这个文件 |
| **所有 prompt** | 方差远大于数值参数、影响面横跨全部用例、回滚不干净。而且 LLM 非确定：agent 评测要 pass@3 连跑三遍才敢说稳定 |
| **`auto.*` 全部** | 在文件里是为了集中管理，**但不自动扫**：评测集只有 14 条，auto p50 3.5s / p99 15.4s，每个候选要 14×3 次真实 LLM 调用，效应量会被噪声淹掉 |

## 改参数的方式

```bash
npm run tune -- show                          # 看当前值 + 影响面
npm run tune -- set shared.retrieval.prefetch 20
npm run tune -- diff                          # 与上一版对比
npm run tune -- rollback                      # 回到上一版
```

`set` 和 `rollback` 都是 tmp + rename 的原子写，并把上一版存到 `tuning.prev.json`。
