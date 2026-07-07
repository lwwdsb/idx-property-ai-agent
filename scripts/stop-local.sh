#!/usr/bin/env bash
# Stop the app services (leaves MySQL + OpenClaw gateway running).
#   bash scripts/stop-local.sh   |   make down
pkill -f 'orchestrateServer' 2>/dev/null && echo "stopped orchestrate (:8100)" || echo "orchestrate not running"
pkill -f 'uvicorn service:app' 2>/dev/null && echo "stopped retrieval (:8099)" || echo "retrieval not running"
echo "(MySQL + OpenClaw gateway left running; 'brew services stop mysql' / 'openclaw gateway stop' to stop those)"
