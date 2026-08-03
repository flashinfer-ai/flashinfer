#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/unit_test_runner.py"

# Help must not prepare the environment, install dependencies, or collect tests.
for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        exec python "${RUNNER}" run --help
    fi
done

if [ -n "${PYTEST_FILE_TIMEOUT_SECONDS+x}" ]; then
    echo "ERROR: PYTEST_FILE_TIMEOUT_SECONDS is obsolete; use UNIT_TEST_TIMEOUT_SECONDS or --unit-timeout-seconds." >&2
    exit 3
fi
if [ -n "${PYTEST_FILE_TIMEOUT_KILL_AFTER_SECONDS+x}" ]; then
    echo "ERROR: PYTEST_FILE_TIMEOUT_KILL_AFTER_SECONDS is obsolete; use UNIT_TEST_TIMEOUT_GRACE_SECONDS or --timeout-grace-seconds." >&2
    exit 3
fi

# The old shared helper randomized this value while being sourced. The sharding
# manifest freezes sampling, so establish the documented deterministic default.
: "${SAMPLE_OFFSET:=0}"
export SAMPLE_OFFSET
export PARALLEL_TESTS=true

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/test_utils.sh"

main() {
    local runner_settings_output
    local -a runner_settings
    runner_settings_output=$(python "${RUNNER}" __shell-settings "$@")
    mapfile -t runner_settings <<< "${runner_settings_output}"
    local operation="${runner_settings[0]}"
    TEST_PATH="${runner_settings[1]}"

    export JUNIT_DIR SAMPLE_RATE SAMPLE_OFFSET TEST_PATH

    if [ "$operation" = "plan" ]; then
        echo "🔍 DRY RUN MODE - collecting and writing a deterministic plan only"
    else
        echo "📋 DETERMINISTIC SHARD MODE - finalized batches resume automatically"

        # nvshmem4py-cu12 pins cuda-python<=12.9. The CI image already carries
        # compatible cuda-python and NVIDIA NVSHMEM libraries, so avoid allowing
        # pip to replace the CUDA flavor while filling this one missing module.
        python -c "import nvshmem.core" 2>/dev/null || pip install --no-deps nvshmem4py-cu12

        install_and_verify

        # Apply dependency overrides after installation since pip may overwrite
        # the versions established by the CI image.
        # shellcheck disable=SC1091
        source "${SCRIPT_DIR}/setup_test_env.sh"

        # tests/moe_ep needs its additional runtime stack only on CUDA 13 images.
        local cuda_major
        cuda_major=$(python -c \
            'import torch; v=torch.version.cuda; print(v.split(".")[0] if v else "0")' \
            2>/dev/null || echo 0)
        if [[ "${TEST_PATH:-}" == *moe_ep* ]] && [ "${cuda_major}" -ge 13 ]; then
            FI_SRC="$(pwd)" bash docker/install/build_flashinfer_ep_pytorch.sh
        fi
    fi

    # test_utils.sh defines the obsolete variables for its legacy callers. They
    # are deliberately not forwarded to the replacement runner.
    unset PYTEST_FILE_TIMEOUT_SECONDS PYTEST_FILE_TIMEOUT_KILL_AFTER_SECONDS

    local runner_exit_code
    local runner_status
    if python "${RUNNER}" "${operation}" "$@"; then
        runner_exit_code=0
    else
        runner_exit_code=$?
    fi

    case "${runner_exit_code}" in
        0)
            runner_status="complete-without-failures"
            ;;
        1)
            runner_status="complete-with-failures"
            ;;
        2)
            runner_status="incomplete-and-resumable"
            ;;
        3)
            runner_status="configuration-collection-or-infrastructure-error"
            ;;
        *)
            echo "UNIT TEST RUNNER ABNORMAL EXIT: exit_code=${runner_exit_code} wrapper_exit_code=unchanged" >&2
            return "${runner_exit_code}"
            ;;
    esac

    echo "UNIT TEST RUNNER RESULT: exit_code=${runner_exit_code} status=${runner_status} wrapper_exit_code=0"
    return 0
}

main "$@"
