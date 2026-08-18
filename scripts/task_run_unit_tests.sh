#!/bin/bash

set -eo pipefail
exec 2>&1

printf -v UNIT_TEST_RUN_STARTED_AT '%(%s)T' -1
UNIT_TEST_RUN_STARTED_SECONDS=${SECONDS}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/unit_test_runner.py"
SUMMARY_PRINTED=false
CURRENT_PHASE="preflight"
LAST_ERROR_COMMAND=""

record_error_command() {
    LAST_ERROR_COMMAND=${BASH_COMMAND}
}

format_elapsed() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    if [ "${hours}" -gt 0 ]; then
        printf '%dh%02dm%02ds' "${hours}" "${minutes}" "${seconds}"
    elif [ "${minutes}" -gt 0 ]; then
        printf '%dm%02ds' "${minutes}" "${seconds}"
    else
        printf '%ds' "${seconds}"
    fi
}

print_fallback_summary() {
    local shell_status=$?
    if [ "$#" -gt 0 ]; then
        shell_status=$1
    fi
    if [ "${SUMMARY_PRINTED}" = "true" ]; then
        return
    fi
    local started_at_iso
    local ended_at_iso
    TZ=UTC printf -v started_at_iso '%(%Y-%m-%dT%H:%M:%SZ)T' "${UNIT_TEST_RUN_STARTED_AT}"
    TZ=UTC printf -v ended_at_iso '%(%Y-%m-%dT%H:%M:%SZ)T' -1
    local elapsed_seconds=$((SECONDS - UNIT_TEST_RUN_STARTED_SECONDS))
    echo "=========================================="
    echo "TEST SUMMARY"
    echo "=========================================="
    echo "Start time: ${started_at_iso}"
    echo "End time: ${ended_at_iso}"
    echo "Time elapsed: $(format_elapsed "${elapsed_seconds}")"
    echo "Scope: unavailable"
    echo "Planned nodes: unavailable"
    echo "Finalized nodes: unavailable"
    echo ""
    echo "STOP CAUSE"
    echo "  phase=${CURRENT_PHASE} shell_exit_code=${shell_status} command=${LAST_ERROR_COMMAND:-unknown}"
    echo ""
    echo "TEST RUN RESOURCE SUMMARY"
    echo "  No finalized source data."
    echo "=========================================="
    echo "Result: status=preflight-setup-or-abnormal-error python_exit_code=unavailable shell_exit_code=${shell_status}"
    echo "=========================================="
}

handle_signal() {
    local signal_name=$1
    local signal_status=$2
    CURRENT_PHASE="signal"
    LAST_ERROR_COMMAND="received ${signal_name}"
    print_fallback_summary "${signal_status}"
    SUMMARY_PRINTED=true
    exit "${signal_status}"
}

trap record_error_command ERR
trap print_fallback_summary EXIT
trap 'handle_signal SIGINT 130' INT
trap 'handle_signal SIGTERM 143' TERM

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
    CURRENT_PHASE="argument-preflight"
    local runner_settings_output
    local -a runner_settings
    if runner_settings_output=$(python "${RUNNER}" __shell-settings "$@"); then
        :
    else
        local preflight_exit_code=$?
        printf '%s\n' "${runner_settings_output}"
        return "${preflight_exit_code}"
    fi
    mapfile -t runner_settings <<< "${runner_settings_output}"
    local operation="${runner_settings[0]}"
    TEST_PATH="${runner_settings[1]}"

    export JUNIT_DIR SAMPLE_RATE SAMPLE_OFFSET TEST_PATH

    if [ "$operation" = "plan" ]; then
        echo "🔍 DRY RUN MODE - collecting and writing a deterministic plan only"
    else
        CURRENT_PHASE="dependency-setup"
        echo "📋 DETERMINISTIC SHARD MODE - finalized batches resume automatically"

        # nvshmem4py-cu12 pins cuda-python<=12.9. The CI image already carries
        # compatible cuda-python and NVIDIA NVSHMEM libraries, so avoid allowing
        # pip to replace the CUDA flavor while filling this one missing module.
        python -c "import nvshmem.core" 2>/dev/null || pip install --no-deps nvshmem4py-cu12

        install_and_verify

        # tests/moe_ep needs its additional runtime stack only on CUDA 13 images.
        local cuda_major
        cuda_major=$(python -c \
            'import torch; v=torch.version.cuda; print(v.split(".")[0] if v else "0")' \
            2>/dev/null || echo 0)
        if [[ "${TEST_PATH:-}" == *moe_ep* ]] && [ "${cuda_major}" -ge 13 ]; then
            FI_SRC="$(pwd)" bash docker/install/build_flashinfer_ep_pytorch.sh
            # The EP setup performs another dependency-resolving editable
            # install, so restore the source-CI DSL baseline afterward.
            # shellcheck disable=SC1091
            source "${SCRIPT_DIR}/setup_ci_test_env.sh"
        fi
    fi

    # test_utils.sh defines the obsolete variables for its legacy callers. They
    # are deliberately not forwarded to the replacement runner.
    unset PYTEST_FILE_TIMEOUT_SECONDS PYTEST_FILE_TIMEOUT_KILL_AFTER_SECONDS

    CURRENT_PHASE="python-runner"
    local runner_exit_code
    if python "${RUNNER}" "${operation}" "$@" \
        --wrapper-started-at "${UNIT_TEST_RUN_STARTED_AT}"; then
        runner_exit_code=0
    else
        runner_exit_code=$?
    fi

    case "${runner_exit_code}" in
        0|1|2|3) ;;
        *)
            echo "ERROR: UNIT TEST RUNNER ABNORMAL EXIT: exit_code=${runner_exit_code} wrapper_exit_code=unchanged"
            return "${runner_exit_code}"
            ;;
    esac

    SUMMARY_PRINTED=true
    case "${runner_exit_code}" in
        0|2)
            return 0
            ;;
        *)
            return "${runner_exit_code}"
            ;;
    esac
}

main "$@"
