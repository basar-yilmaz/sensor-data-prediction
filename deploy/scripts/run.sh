#!/usr/bin/env bash
# Launch the BFRB deploy service via Uvicorn.
#
# Honors the standard BFRB_* env vars; see deploy/README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

HOST="${BFRB_HOST:-127.0.0.1}"
PORT="${BFRB_PORT:-8000}"
LOG_LEVEL="${BFRB_LOG_LEVEL:-info}"
RELOAD_FLAG=""
if [[ "${BFRB_RELOAD:-false}" == "true" ]]; then
  RELOAD_FLAG="--reload"
fi

exec uv --project "${REPO_ROOT}/deploy" run uvicorn app.main:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --log-level "${LOG_LEVEL}" \
  ${RELOAD_FLAG}
