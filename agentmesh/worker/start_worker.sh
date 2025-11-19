#!/bin/bash
# Start Celery worker

cd "$(dirname "$0")/.."

echo "🚀 Starting AgentMesh Celery worker..."

celery -A agentmesh.worker.celery_app worker \
  --loglevel=info \
  --concurrency=4 \
  --max-tasks-per-child=50
