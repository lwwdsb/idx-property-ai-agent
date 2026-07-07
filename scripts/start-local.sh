#!/usr/bin/env bash
# Bring up the full local stack (idempotent — safe to re-run).
# Run from a Terminal (has Desktop file access; macOS blocks launchd from ~/Desktop).
#   bash scripts/start-local.sh   |   make up
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

ok(){ printf "  \033[32m✓\033[0m %s\n" "$1"; }
warn(){ printf "  \033[33m•\033[0m %s\n" "$1"; }

echo "Starting IDX local stack…"

# 1. MySQL
if MYSQL_PWD=root123 mysqladmin -u root ping 2>/dev/null | grep -q alive; then ok "MySQL already up"
else brew services start mysql >/dev/null 2>&1 && ok "MySQL started"; fi

# 2. Docker + Qdrant (optional — semantic search / recommend need it)
if docker ps >/dev/null 2>&1; then
  docker start idx-qdrant >/dev/null 2>&1 && ok "Qdrant up" || warn "Qdrant container missing"
else
  warn "Docker not running — semantic search/recommend will be unavailable (open -a Docker to enable)"
fi

# 3. Retrieval service (:8099)
if curl -s localhost:8099/health >/dev/null 2>&1; then ok "retrieval service already up"
else
  ( cd retrieval && nohup "$ROOT/.venv/bin/uvicorn" service:app --port 8099 --log-level warning \
      > "$ROOT/logs/retrieval.log" 2>&1 & )
  warn "retrieval service starting (model warmup ~6s)…"
fi

# 4. Orchestrate service (:8100)
if curl -s localhost:8100/health >/dev/null 2>&1; then ok "orchestrate service already up"
else
  nohup "$ROOT/node_modules/.bin/tsx" src/server/orchestrateServer.ts \
      > "$ROOT/logs/orchestrate.log" 2>&1 &
  warn "orchestrate service starting…"
fi

# 5. OpenClaw gateway (launchd-managed; just report)
curl -s http://localhost:18789/health >/dev/null 2>&1 && ok "OpenClaw gateway up (WhatsApp)" || warn "gateway down — run: openclaw gateway start"

# wait for the two services
for i in $(seq 1 20); do
  curl -s localhost:8099/health >/dev/null 2>&1 && curl -s localhost:8100/health >/dev/null 2>&1 && break
  sleep 2
done
echo "Status:"
curl -s localhost:8100/health >/dev/null 2>&1 && ok "orchestrate :8100" || warn "orchestrate :8100 not ready"
curl -s localhost:8099/health >/dev/null 2>&1 && ok "retrieval  :8099" || warn "retrieval :8099 not ready"
echo "Done. WhatsApp assistant is live if all ✓ above."
