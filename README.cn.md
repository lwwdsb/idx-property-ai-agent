# IDX 房产 AI 助手

> [English](README.md) · **中文**

一个生产级的房地产 AI 助手,构建在 [OpenClaw](https://github.com/openclaw/openclaw)
运行时之上。它基于约 14.5 万条真实加州 MLS 数据回答自然语言问题——房源搜索、行情分析、
语义推荐、RAG 知识问答——并通过 WhatsApp 和邮件沟通,**每一个对外动作都需人工审批**。

> **AI Agentic Engineer 实习 — IDX Exchange · 2026 暑期 · 12 周**

系统本质是**一组专用 skill,由一个确定性编排器统一调度**(并保留了 LLM 推理层)——
而不是一个自主 agent 循环。这个取舍是刻意的:**用鲁棒性和可预测性,换掉自由发挥的自主性**。

---

## 架构

```
WhatsApp  ⇄  OpenClaw 网关  ⇄  agent(DeepSeek —— 薄前台)
                                   │  只调一个工具:ask_idx_assistant
                                   ▼
                       编排服务(HTTP :8100)
                                   │  确定性路由(判意图 + 解析)
        ┌────────────┬────────────┼─────────────┬─────────────┐
      搜房          行情         推荐           RAG            邮件
   MySQL / Qdrant  成交中位数/   结构化+        有据可查        起草 → 人工
   (硬过滤 +        趋势         语义融合        + 引用出处      审批 → 发送
    dense+BM25)                  + comp 验证                   (工具层锁死)
        │            │            │              │              │
      MySQL  ◄───────┴────────────┴──── Qdrant(向量)───────────┘
                                   ▲
                    常驻 Python 检索服务(:8099)
                    fastembed bge-small(本地)· 无需 embedding key
```

**设计原则 —— LLM 只放在边缘。** 路由、SQL、价格验证、发送闸都是确定性代码。LLM(DeepSeek)
只用在它不可替代的地方:解析模糊表达、组织有据可查的措辞、做细致的推理判断。**事实永不由模型
杜撰**——价格和地址都来自数据库行。

**鲁棒性优先级 —— 丙 > 甲 > 乙**(一个可机器执行的裁决规则):绝不误发对外消息 >
绝不自信地犯错(幻觉)> 永远保持响应(优雅降级)。

**技术栈:** OpenClaw · TypeScript · Python · MySQL · Qdrant · fastembed(本地 `bge-small`)
· DeepSeek(chat)· WhatsApp · Nodemailer

---

## 数据

本地 schema(`idx_exchange`)中的三张精选 MySQL 表——来自实习 FTP 的 `sql/` 目录:

| 表 | 行数 | 作用 |
|---|---|---|
| `rets_property` | 53,122 条在售房源(加州),130+ 个晦涩的 `L_*` 字段 | 实时搜索与发现 |
| `california_sold` | 87,157 条成交记录(加州),46 个字段 | 历史 comp 与分析 |
| `rets_openhouse` | 4,282 条开放日活动 | 开放日查询 |

**关联方式:** `CAST(rets_property.L_ListingID AS UNSIGNED) = california_sold.ListingKey`,
或按 城市 + 邮编 做市场级分析。一份**字段字典**(`schema/columns.ts`)把语义名映射到晦涩的
物理列(如 `beds → L_Keyword2`),作为单一事实来源。

> MLS 数据是机密的,**不**提交到本仓库(见 `.gitignore`)。数据只导入本地 MySQL 实例。

---

## 能力

- **自然语言搜房** —— `归一化 → 正则快路径 → DeepSeek 兜底 → 澄清`。中英双语。
  纯结构化查询走 MySQL;带软/语义内容的查询(如*"有山景、带大院子的工匠风房子"*)走 Qdrant 混合检索。
  解析**能识别否定**(泳池三态:要 / 不要 / 无所谓),并**对提取的城市查字典验真**;可疑解析——
  假城市("USC""Gotham")或正则难处理的否定——会升级到 LLM 当**按需 critic**(或反问真实城市),
  而不是拿一个错查询去搜。
- **语义搜索** —— dense(`bge-small`)+ BM25,用倒数排名融合(RRF)合并,硬条件作为
  **Qdrant 内部的 payload 过滤器**(先过滤、再排序)。Qdrant 挂了则降级到 MySQL。
- **行情分析** —— 真·**中位数**价格、$/sqft、在市天数、成交/挂牌比,以及 `california_sold` 上的 12 个月趋势。
- **推荐** —— 结构化相似度(60%)+ 语义相似度(40%)融合,附**comp 价格核查**和确定性验证门槛
  (可比成交少于 3 条时不给结论——数据太薄绝不瞎猜)。
- **RAG 知识问答** —— 基于精选语料的有据可查、带引用的回答;无 key 时摘录降级。模型绝不在检索
  上下文之外杜撰事实。
- **WhatsApp 入站自动回复** —— 真实消息经编排器处理并自动回复。
- **邮件** —— 起草报告、预览,再 **`approve <id>`**(在 WhatsApp 或 CLI)发送。单发/群发共用一条
  路径(多收件人用 BCC);发送**锁死在工具层**,LLM 结构上无法自行发送。

---

## 项目结构

```
idx-property-ai-agent/
├── schema/            # 字段字典(columns.ts)+ SQL DDL/索引(是代码,不是数据)
├── src/               # TypeScript 实时链路
│   ├── search/          # 自然语言解析(正则+归一化+LLM)→ 过滤器 → 参数化 SQL
│   ├── market/          # 基于中位数的行情统计
│   ├── agent/           # 多轮会话
│   ├── orchestrator/    # 路由、skill 注册表、bridge、审批命令
│   ├── email/           # 起草-再审批、持久化草稿库、模板
│   ├── whatsapp/        # 入站处理(幂等、限流、降级)
│   ├── server/          # 编排 HTTP 服务(:8100)
│   └── llm/             # 与厂商无关的 chat 客户端
├── retrieval/         # Python:fastembed + Qdrant 混合检索、推荐、RAG、常驻服务
├── openclaw-plugin/   # OpenClaw 插件:注册 ask_idx_assistant(薄前台)
├── knowledge/         # RAG 语料(字段词表 + 手写术语)
├── scripts/           # import.sh、check_env.py、start-local.sh(make up)、stop-local.sh
├── Makefile           # import | indexes | check | up | down | test:all
└── .env.example
```

---

## 安装

### 前置条件
- Node.js(v20+)与 npm · Python 3.10+ · MySQL · Docker(用于 Qdrant)
- 已安装 OpenClaw(`npm install -g openclaw`,再 `openclaw onboard`)
- **无需 OpenAI 付费** —— embedding 本地运行(fastembed);chat key(推荐 DeepSeek)是可选的,
  没有也能优雅降级。

### 1. 克隆与配置
```bash
git clone https://github.com/lwwdsb/idx-property-ai-agent.git
cd idx-property-ai-agent
cp .env.example .env          # 设 DB_PASSWORD;可选 LLM_API_KEY、EMAIL_*
npm install
python3 -m venv .venv && source .venv/bin/activate && pip install -r retrieval/requirements.txt
```

### 2. 数据与索引
把 SQL 表从实习 FTP 下载到仓库根目录,然后:
```bash
make import     # 幂等导入 idx_exchange(对零日期用 SET sql_mode='')
make indexes    # 高频过滤字段的索引
make check      # 校验 .env + 行数
```

### 3. 向量(用于语义搜索/推荐)
```bash
docker start idx-qdrant                      # 或:docker run -p 6333:6333 qdrant/qdrant
source .venv/bin/activate
python retrieval/ingest.py --resume          # embedding 并 upsert 房源到 Qdrant
```

### 4. 连接 WhatsApp
```bash
openclaw channels login --channel whatsapp   # 用"关联设备"扫码
```

---

## 运行

```bash
# 一键拉起整套本地服务(MySQL、Qdrant、检索 :8099、编排 :8100)
make up
make down                                     # 停应用服务

# 直接问编排器(即 WhatsApp 工具所调用的)
curl -s localhost:8100/orchestrate -H 'Content-Type: application/json' \
  -d '{"userId":"me","message":"在 Irvine 找有山景的 3 居室 300万以下"}'

# 命令行工具
npm run search  -- "在 Irvine 找 3 居室带泳池 250万以下"
npm run drafts  -- report Irvine client@x.com     # 起草;然后 "approve <id>" 发送
python retrieval/search.py "craftsman with a big backyard" --city Irvine --max-price 2500000
python retrieval/rag.py "what is DOM?"
```

服务起来、WhatsApp 关联后,给关联号码发消息即可收到走完整流水线的自动回复。

---

## 12 周路线图

| 周 | 模块 | 状态 |
|------|--------|--------|
| 0 | 环境、MySQL 导入、索引、env 校验 | ✅ |
| 1 | 骨架 + 字段字典 + 共享模块 | ✅ |
| 2 | 自然语言搜房(归一化 → 正则 → LLM 兜底) | ✅ |
| 3 | MySQL 查询层(参数化、≤50 上限、FULLTEXT、DTO) | ✅ |
| 4 | 会话 agent(多轮、patch 合并、可插拔存储) | ✅ |
| 5 | `california_sold` 行情分析(真中位数、趋势) | ✅ |
| 6 | 混合检索 —— Qdrant 里 dense + BM25 + RRF | ✅ |
| 7 | 推荐 + comp 价格验证 + 验证门槛 | ✅ |
| 8 | RAG 问答(有据可查 + 引用,摘录降级) | ✅ |
| 9 | 编排器 —— skill 注册表 + 确定性路由 + 配方 | ✅ |
| 10 | 常驻检索服务(约 160×)+ embedding 意图 + WhatsApp 处理 | ✅ |
| 11 | 邮件 agent —— 起草-再审批,发送锁死在工具层 | ✅ |
| 12 | 收尾:完整 skill 接线、测试套件、文档 | ✅ |

**第 12 周之后(已上线、端到端验证):**
- WhatsApp **入站自动回复**,经 OpenClaw 插件(薄前台 → 我们的工具)。
- **DeepSeek** 接入所有保留的 LLM 槽位(解析 / RAG / 验证 / 邮件)以及前台 agent。
- **邮件真发上线**(Gmail SMTP),可在 WhatsApp 审批(`approve <id>` / `cancel <id>`)。
- **语义搜房接入编排器**(软查询 → Qdrant 混合检索)。

*保留 / 可选(优雅降级):* `LLM_API_KEY` 为空 → 规则/模板/摘录;`EMAIL_*` 为空 → dry-run。
*尚未做的增强:* RRF 之后的 cross-encoder 精排。

---

## 安全与护栏

- **无自主对外动作。** 邮件先起草预览;人工 `approve <id>` 是唯一发送路径,发送能力不暴露给 LLM(丙)。
- **操作者白名单**管控起草/审批;**批次上限**(≤25 收件人,群发用 BCC)。
- **只给有据可查的答案。** 事实来自数据库行;数据太薄时验证器不给结论。
- **git 里无密钥、无批量数据。** 查询结果集有上限(≤50 行);`.env`、SQL 转储、向量都被 gitignore。

---

## 测试

```bash
npm run test:all                                          # 全部 TS 套件 + 类型检查 + eval
python retrieval/test_recommend.py && python retrieval/test_rag.py
```
全绿:TS `unit / eval / agent / market / orchestrator / whatsapp / email`;
Python `recommend 20 · rag 10`。

---

## 许可

实习课程作业 —— 不授权再分发。MLS 数据机密,归 IDX Exchange 所有。
