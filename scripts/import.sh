#!/usr/bin/env bash
# Import the three MLS tables into the idx_exchange database.
# Idempotent: drops each table before re-importing. Safe to re-run.
#
# Usage:  DB_PASSWORD=root123 bash scripts/import.sh
# Env:    DB_HOST(127.0.0.1) DB_PORT(3306) DB_USER(root) DB_PASSWORD DB_NAME(idx_exchange)
#         DATA_DIR(dir holding the .sql files, default: parent of this script)
set -euo pipefail

DB_HOST="${DB_HOST:-127.0.0.1}"
DB_PORT="${DB_PORT:-3306}"
DB_USER="${DB_USER:-root}"
DB_PASSWORD="${DB_PASSWORD:-root123}"
DB_NAME="${DB_NAME:-idx_exchange}"
DATA_DIR="${DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export MYSQL_PWD="$DB_PASSWORD"
# Clear sql_mode for the whole session: the MariaDB dumps use zero-date
# timestamp defaults ('0000-00-00 00:00:00') that MySQL 9.x rejects by default.
MYSQL=(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" --init-command="SET sql_mode=''")

# table_name -> sql file (imported small -> large; rets_property last, FULLTEXT build is slow)
TABLES=("rets_openhouse:rets_openhouse.sql" "california_sold:california_sold.sql" "rets_property:rets_property.sql")

echo ">> Target: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME   data dir: $DATA_DIR"
"${MYSQL[@]}" -e "CREATE DATABASE IF NOT EXISTS \`$DB_NAME\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

for entry in "${TABLES[@]}"; do
  tbl="${entry%%:*}"; file="${entry##*:}"; path="$DATA_DIR/$file"
  if [[ ! -f "$path" ]]; then echo "!! MISSING: $path  (skip)"; continue; fi
  size=$(du -h "$path" | cut -f1)
  echo ">> [$tbl] dropping if exists, then importing $file ($size) ..."
  "${MYSQL[@]}" "$DB_NAME" -e "DROP TABLE IF EXISTS \`$tbl\`;"
  t0=$SECONDS
  "${MYSQL[@]}" "$DB_NAME" < "$path"
  echo "   done in $((SECONDS - t0))s"
done

echo ">> Row counts:"
for entry in "${TABLES[@]}"; do
  tbl="${entry%%:*}"
  cnt=$("${MYSQL[@]}" -N "$DB_NAME" -e "SELECT COUNT(*) FROM \`$tbl\`;" 2>/dev/null || echo "ERR")
  printf "   %-18s %s\n" "$tbl" "$cnt"
done
echo ">> Import complete."
