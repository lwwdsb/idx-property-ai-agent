# IDX property AI agent — Week 0 data-layer tasks.
# Reads DB creds from .env. Usage: `make help`

SHELL := /bin/bash
ENV_FILE := .env

# Python for the eval/retrieval steps. The repo's Python deps (pymysql, qdrant-client,
# fastembed) live ONLY in .venv, so a bare `python` cannot run eval_retrieval.py and
# `make eval` died partway every time — which also meant the regression gate at the end of
# the target had never actually run as part of it. Fall back to `python` if the venv is gone,
# so the failure is a clear ImportError rather than a missing-file error.
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

# export every var defined in .env to recipe environments
ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
export
endif

.PHONY: help db-up db-down import indexes check rebuild up down eval eval-datasets

help:           ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

db-up:          ## start MySQL (brew services)
	HOMEBREW_NO_AUTO_UPDATE=1 brew services start mysql

db-down:        ## stop MySQL
	HOMEBREW_NO_AUTO_UPDATE=1 brew services stop mysql

import:         ## (re)import the three SQL tables into idx_exchange
	bash scripts/import.sh

indexes:        ## add high-frequency filter indexes (safe if already present -> errors are ok)
	MYSQL_PWD="$(DB_PASSWORD)" mysql -h $(DB_HOST) -P $(DB_PORT) -u $(DB_USER) \
		--init-command="SET sql_mode=''" $(DB_NAME) < schema/indexes.sql || true

check:          ## validate env + DB connectivity + OpenAI key
	python3 scripts/check_env.py

rebuild: import indexes check   ## full data-layer rebuild then validate

up:             ## start the full local stack (MySQL, Qdrant, services)
	bash scripts/start-local.sh

down:           ## stop the app services (orchestrate + retrieval)
	bash scripts/stop-local.sh

eval:           ## run the full evaluation suite -> eval/report.md (needs Qdrant + LLM key)
	$(PY) eval/runners/manifest.py
	$(PY) eval/runners/provenance.py
	$(PY) eval/metrics/test_metrics.py
	$(PY) eval/runners/objective_selftest.py
	npx tsx eval/runners/evalIntentParse.ts
	$(PY) eval/runners/report_intent_parse.py
	npx tsx eval/runners/evalE2E.ts
	$(PY) eval/runners/report_e2e.py
	npx tsx eval/runners/evalModeRetrieval.ts
	$(PY) eval/runners/report_mode_intent.py
	npx tsx eval/runners/evalAgent.ts
	$(PY) eval/runners/report_agent.py
	npx tsx eval/runners/evalMemoryFacts.ts
	npx tsx eval/runners/evalMemoryDynamics.ts
	npx tsx eval/runners/evalMemorySelect.ts
	$(PY) eval/runners/report_memory_select.py
	npx tsx eval/runners/evalMemoryUtility.ts
	npx tsx eval/runners/evalMemoryConsolidate.ts
	$(PY) eval/runners/eval_retrieval.py
	$(PY) eval/runners/eval_rag.py
	$(PY) eval/runners/tune_retrieval.py
	$(PY) eval/runners/bench_latency.py
	npx tsx eval/runners/regression.ts
	$(PY) eval/runners/make_report.py
	@echo "\n==> eval/report.md"
	@$(PY) eval/runners/gate.py   # non-zero exit on a regression vs eval/baseline.json

pool:           ## regression pool — the boolean veto (fails if a pinned behaviour broke)
	npx tsx eval/runners/regression.ts

manifest:       ## check the frozen main sets have not changed without a version bump
	$(PY) eval/runners/manifest.py

arm:            ## N repeats of the agent eval under the CURRENT config -> an arm (LABEL=x N=5)
	$(PY) eval/runners/repeat_agent.py --label $(LABEL) --n $(or $(N),5)

objective:      ## compare two arms: quality as constraint, tokens as objective (BASE=x CAND=y)
	$(PY) eval/runners/objective.py --baseline $(BASE) --candidate $(CAND)

eval-datasets:  ## rebuild the LLM-assisted labeled sets (retrieval); needs Qdrant + LLM key
	$(PY) eval/runners/build_retrieval_set.py
