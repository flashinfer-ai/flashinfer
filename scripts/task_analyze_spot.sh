#!/bin/bash
# Copyright (c) 2026 by FlashInfer team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

JOB_FILTER="${1:-}"
REPOSITORY="${2:-}"
RUN_ID="${3:-}"

if [ -z "$JOB_FILTER" ] || [ -z "$REPOSITORY" ] || [ -z "$RUN_ID" ]; then
  echo "Usage: $0 <job_filter> <repository> <run_id>"
  echo "Example: $0 'startswith(\"AOT\")' 'flashinfer-ai/flashinfer' '12345'"
  exit 1
fi

SPOT_TERMINATION=false

# Temp file for job logs (cleaned up on exit)
LOG_FILE=$(mktemp)
LOG_FILE_ZIP="${LOG_FILE}.zip"
cleanup() { rm -f "$LOG_FILE" "$LOG_FILE_ZIP"; }
trap cleanup EXIT

# Include both failed and cancelled jobs (spot termination can cause either)
FAILED_JOBS=$(gh api --paginate "/repos/${REPOSITORY}/actions/runs/${RUN_ID}/jobs" \
  --jq ".jobs[] | select(.name | ${JOB_FILTER}) | select(.conclusion == \"failure\" or .conclusion == \"cancelled\") | .id")

if [ -z "$FAILED_JOBS" ]; then
  echo "No failed jobs matching filter: ${JOB_FILTER}"
  echo "is_spot_termination=false" >> "$GITHUB_OUTPUT"
  exit 0
fi

for JOB_ID in $FAILED_JOBS; do
  JOB_INFO=$(gh api "/repos/${REPOSITORY}/actions/jobs/${JOB_ID}" 2>/dev/null || true)
  JOB_CONCLUSION=$(echo "$JOB_INFO" | jq -r '.conclusion // empty' 2>/dev/null || echo "")

  # Skip jobs cancelled by fail-fast (they're not the root cause)
  if [ "$JOB_CONCLUSION" == "cancelled" ]; then
    continue
  fi

  # Check job metadata for runner communication errors
  if echo "$JOB_INFO" | grep -qiE "runner.*lost|lost communication"; then
    echo "Detected: Runner lost communication (job $JOB_ID)"
    SPOT_TERMINATION=true
    break
  fi

  # Try to download job logs to /tmp
  if ! gh api "/repos/${REPOSITORY}/actions/jobs/${JOB_ID}/logs" > "$LOG_FILE_ZIP" 2>/dev/null; then
    echo "Detected: Cannot download logs, likely infrastructure failure (job $JOB_ID)"
    SPOT_TERMINATION=true
    break
  fi

  # Handle both zip and plain text log formats
  if file "$LOG_FILE_ZIP" | grep -q "Zip archive"; then
    unzip -p "$LOG_FILE_ZIP" > "$LOG_FILE" 2>/dev/null || mv "$LOG_FILE_ZIP" "$LOG_FILE"
  else
    mv "$LOG_FILE_ZIP" "$LOG_FILE"
  fi

  # Check for spot termination marker from task_monitor_spot.sh
  if grep -q "FLASHINFER_SPOT_TERMINATION_DETECTED" "$LOG_FILE"; then
    echo "Detected: AWS spot termination marker (job $JOB_ID)"
    SPOT_TERMINATION=true
    break
  fi

  # Check for infrastructure error patterns
  if grep -qiE "connection reset by peer|context canceled|The operation was canceled|grpc.*closing|The self-hosted runner.*lost" "$LOG_FILE"; then
    echo "Detected: infrastructure error pattern (job $JOB_ID)"
    SPOT_TERMINATION=true
    break
  fi
done

echo "is_spot_termination=$SPOT_TERMINATION"
echo "is_spot_termination=$SPOT_TERMINATION" >> "$GITHUB_OUTPUT"
