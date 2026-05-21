#!/usr/bin/env bash

# Validate DVC cache behavior by changing a tracked training parameter and
# confirming only downstream stages rerun.

set -euo pipefail

readonly PARAMS_FILE="params.yaml"
readonly LOG_FILE="repro_log.txt"
readonly BACKUP_FILE="${PARAMS_FILE}.bak.caching-test"
readonly ORIGINAL_N_ESTIMATORS=100
readonly UPDATED_N_ESTIMATORS=200

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: required command '$1' was not found in PATH"
    exit 127
  fi
}

resolve_dvc_command() {
  local candidates=(
    "./venv/Scripts/dvc.exe"
    "./venv/Scripts/dvc"
    "./venv/bin/dvc"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if command -v dvc >/dev/null 2>&1; then
    command -v dvc
    return 0
  fi

  return 1
}

backup_params() {
  cp "$PARAMS_FILE" "$BACKUP_FILE"
}

restore_params() {
  if [[ -f "$BACKUP_FILE" ]]; then
    mv "$BACKUP_FILE" "$PARAMS_FILE"
  fi
}

cleanup() {
  restore_params
}

modify_params() {
  local python_bin=""
  if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  elif command -v python >/dev/null 2>&1; then
    python_bin="python"
  else
    log "ERROR: neither python3 nor python was found in PATH"
    exit 127
  fi

  "$python_bin" - <<'PY'
from pathlib import Path

path = Path("params.yaml")
text = path.read_text(encoding="utf-8")
old = '  n_estimators: 100\n'
new = '  n_estimators: 200\n'
if old not in text:
    raise SystemExit("Expected n_estimators value not found in params.yaml")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
}

assert_log_contains() {
  local pattern="$1"
  local description="$2"
  if grep -qE "$pattern" "$LOG_FILE"; then
    log "PASS: $description"
  else
    log "ERROR: $description"
    log "--- repro_log.txt tail ---"
    tail -n 40 "$LOG_FILE" || true
    exit 1
  fi
}

main() {
  trap cleanup EXIT

  require_command grep
  require_command tail

  readonly DVC_BIN="$(resolve_dvc_command || true)"
  if [[ -z "${DVC_BIN:-}" ]]; then
    log "ERROR: could not locate a DVC executable in PATH or the local venv"
    exit 127
  fi

  if [[ ! -f "$PARAMS_FILE" ]]; then
    log "ERROR: $PARAMS_FILE does not exist"
    exit 1
  fi

  log "Starting DVC cache validation"
  log "Running baseline dvc repro to ensure the pipeline is up to date"
  "$DVC_BIN" repro >/dev/null

  log "Backing up $PARAMS_FILE"
  backup_params

  log "Updating train.n_estimators from $ORIGINAL_N_ESTIMATORS to $UPDATED_N_ESTIMATORS"
  modify_params

  log "Running dvc repro again and capturing output to $LOG_FILE"
  "$DVC_BIN" repro > "$LOG_FILE" 2>&1

  log "Validating cache behavior from $LOG_FILE"
  assert_log_contains "prepare.*didn't change, skipping|Skipping stage 'prepare'|Stage 'prepare'.*skipping" "prepare stage was skipped"
  assert_log_contains "featurize.*didn't change, skipping|Skipping stage 'featurize'|Stage 'featurize'.*skipping" "featurize stage was skipped"
  assert_log_contains "Running stage 'train'|stage 'train'" "train stage executed"
  assert_log_contains "Running stage 'evaluate'|stage 'evaluate'" "evaluate stage executed"

  log "DVC cache validation completed successfully"
  log "Generated artifact: $LOG_FILE"
}

main "$@"
