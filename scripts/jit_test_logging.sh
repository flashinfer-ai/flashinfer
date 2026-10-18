#!/bin/bash

# Keep collection/JIT time alongside each file's case timings.
_jit_result_dir="junit/$(basename "${BASH_SOURCE[1]}" .sh)"

pytest() {
  local target result
  for target in "$@"; do
    if [[ "$target" == tests/*.py ]]; then
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
