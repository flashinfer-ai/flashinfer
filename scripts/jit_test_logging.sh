#!/bin/bash

# Keep collection/JIT time alongside each file's case timings.
_jit_result_dir="junit/$(basename "${BASH_SOURCE[1]}" .sh)"

# A targeted run sets JIT_TEST_SCOPE to the comma-separated test files it covers
# (from scripts/pr_checks/targeted_lane_scope.py). Invocations of other files are
# skipped, so the shard keeps its own -k selections and compile settings.
pytest() {
  local target result
  for target in "$@"; do
    if [[ "$target" == tests/*.py ]]; then
      if [ -n "${JIT_TEST_SCOPE:-}" ] && [[ ",${JIT_TEST_SCOPE}," != *",${target},"* ]]; then
        echo "SKIP (outside targeted scope): pytest $*"
        return 0
      fi
      result="${_jit_result_dir}/$(basename "$target" .py)"
      mkdir -p "${_jit_result_dir}"
      python -m pytest -p scripts.test_sharding.pytest_plugin \
        --junitxml="${result}.xml" \
        --flashinfer-telemetry-json="${result}.telemetry.json" \
        --flashinfer-result-json="${result}.meta.json" \
        --durations=20 "$@"
      return
    fi
  done
  echo "ERROR: pytest invocation has no test file: $*" >&2
  return 2
}
