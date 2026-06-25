# IDX property AI agent — Week 0 data-layer tasks.
# Reads DB creds from .env. Usage: `make help`

SHELL := /bin/bash
ENV_FILE := .env

# export every var defined in .env to recipe environments
ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
export
endif

.PHONY: help db-up db-down import indexes check rebuild

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
