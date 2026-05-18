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

# Spot Termination Monitor for AWS EC2 Spot Instances
# Usage: ./scripts/task_monitor_spot.sh &

set -euo pipefail

IMDS_URL="http://169.254.169.254/latest/meta-data/spot/instance-action"
TOKEN_URL="http://169.254.169.254/latest/api/token"
CHECK_INTERVAL=5

while true; do
  # Try IMDSv2 first (token-based)
  TOKEN=$(curl -s --max-time 2 -X PUT "$TOKEN_URL" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)

  if [ -n "$TOKEN" ]; then
    # IMDSv2: Use token in header
    META=$(curl -sf --max-time 2 -H "X-aws-ec2-metadata-token: $TOKEN" "$IMDS_URL" 2>/dev/null || true)
  else
    # IMDSv1: Direct access (fallback)
    META=$(curl -sf --max-time 2 "$IMDS_URL" 2>/dev/null || true)
  fi

  if echo "$META" | grep -q "terminate"; then
    # Output GitHub Actions error annotation for visibility
    echo "::error::FLASHINFER_SPOT_TERMINATION_DETECTED"
    echo "AWS Spot Termination Notice received at $(date)"
    echo "Instance will be terminated soon. Job will be rerun on on-demand instance."
    exit 0
  fi

  sleep $CHECK_INTERVAL
done
