#!/bin/bash

set -eo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set MPI command prefix for multi-GPU tests
: "${PYTEST_COMMAND_PREFIX:=mpirun -np 4}"

# Disable sanity testing for multi-GPU tests (always run full suite)
# shellcheck disable=SC2034  # Used by common_test_functions.sh
DISABLE_SANITY_TEST=true

# Source common test functions
# shellcheck disable=SC1091  # File exists, checked separately
source "${SCRIPT_DIR}/test_utils.sh"

# Define the specific test files for multi-GPU comm tests (single-node)
# MPI_TEST_FILES="tests/comm/test_allreduce_unified_api.py tests/comm/test_allreduce_negative.py tests/comm/test_trtllm_allreduce_fusion.py"
# Add others back once they are fixed
MPI_TEST_FILES="tests/comm/test_allreduce_unified_api.py"

# These tests create their own distributed workers with torch.multiprocessing.spawn.
# Running them under mpirun would start multiple pytest parents and can collide on
# TCPStore ports.
SPAWN_MANAGED_TEST_FILES="tests/comm/test_quantized_allreduce.py tests/comm/test_trtllm_allreduce_fusion_pdl.py tests/comm/test_ulysses_a2a.py tests/comm/test_ulysses_communicator.py"

# Tests that require torchrun instead of mpirun
TORCHRUN_TEST_FILES="tests/attention/test_parallel_attention.py tests/gemm/test_multi_gpu_cute_dsl_blockscaled_gemm_fusion.py tests/comm/test_mnnvl_cutedsl_numerical_contract.py tests/comm/test_mnnvl_cutedsl_no_rms_norm.py tests/comm/test_mnnvl_cutedsl_h7168.py tests/comm/test_mnnvl_cutedsl_static_fp8.py tests/comm/test_mnnvl_cutedsl_dynamic_fp8.py tests/comm/test_mnnvl_cutedsl_allreduce_api.py"
SM107_TEST_FILES="tests/moe_ep/test_moe_ep_sm107_block_scaled_mega_multirank.py tests/moe_ep/test_sm107_routing_graph_multirank.py tests/moe_ep/test_sm107_auto_multirank.py tests/moe_ep/test_sm107_variant_selection_multirank.py"
: "${TORCHRUN_PREFIX:=torchrun --nproc_per_node=4}"

# BI qualification needs an allocation-wide torchrun, not a one-GPU pytest
# shard. Enroll only explicitly selected targets; ordinary bare comm runs keep
# their existing scope. The strict qualifier supplies per-rank JUnit results.
MEGAMOE_BI_TEST_FILES="tests/moe_ep/test_bf16_megamoe_batch_invariant.py tests/moe_ep/test_nvfp4_megamoe_batch_invariant.py tests/moe_ep/test_mxfp4_megamoe_batch_invariant.py tests/moe_ep/test_mxfp8_megamoe_batch_invariant.py tests/moe_ep/test_a16_megamoe_batch_invariant.py tests/moe_ep/test_megamoe_quantized_tiles_batch_invariant.py tests/moe_ep/test_megamoe_numerical_variants.py tests/moe_ep/test_blackwell_bi_outer_autoconfig.py"
: "${MEGAMOE_BI_WORLD_SIZE:=4}"

_run_pytest_group() {
    local files="$1"
    if [ -z "$files" ]; then
        echo "(none selected by TEST_PATH)"
        echo ""
        return 0
    fi
    if [ "$DRY_RUN" == "true" ]; then
        execute_dry_run "$files"
    else
        execute_tests "$files"
    fi
}

# Main execution
main() {
    test_status_begin
    parse_args "$@"
    print_test_mode_banner

    MPI_TEST_FILES="$(filter_files_by_test_path "$MPI_TEST_FILES")"
    SPAWN_MANAGED_TEST_FILES="$(filter_files_by_test_path "$SPAWN_MANAGED_TEST_FILES")"
    TORCHRUN_TEST_FILES="$(filter_files_by_test_path "$TORCHRUN_TEST_FILES")"
    SM107_TEST_FILES="$(filter_files_by_test_path "$SM107_TEST_FILES")"
    if [ -n "$SM107_TEST_FILES" ]; then
        local sm107_supported
        sm107_supported=$(python -c '
import torch
print(int(bool(torch.version.cuda)
          and int(torch.version.cuda.split(".")[0]) >= 13
          and torch.cuda.is_available()
          and torch.cuda.get_device_capability() == (10, 7)))')
        if [ "$sm107_supported" = "1" ]; then
            TORCHRUN_TEST_FILES+=" ${SM107_TEST_FILES}"
        else
            echo "Skipping SM107 MegaMoE suites: requires Rubin and CUDA 13+."
            SM107_TEST_FILES=""
        fi
    fi
    if [ -n "${TEST_PATH:-}" ]; then
        MEGAMOE_BI_TEST_FILES="$(filter_files_by_test_path "$MEGAMOE_BI_TEST_FILES")"
    else
        MEGAMOE_BI_TEST_FILES=""
    fi
    local -a bi_families=() bi_command=()
    local bi_file family
    for bi_file in $MEGAMOE_BI_TEST_FILES; do
        if [[ "$bi_file" == *test_blackwell_bi_outer_autoconfig.py ]]; then
            for family in bf16 nvfp4 a16; do
                [[ " ${bi_families[*]} " == *" $family "* ]] || bi_families+=("$family")
            done
            continue
        fi
        if [[ "$bi_file" == *test_megamoe_numerical_variants.py ]]; then
            for family in bf16 nvfp4 mxfp4 mxfp8; do
                [[ " ${bi_families[*]} " == *" $family "* ]] || bi_families+=("$family")
            done
            continue
        fi
        if [[ "$bi_file" == *test_megamoe_quantized_tiles_batch_invariant.py ]]; then
            for family in nvfp4 mxfp4 mxfp8; do
                [[ " ${bi_families[*]} " == *" $family "* ]] || bi_families+=("$family")
            done
            continue
        fi
        family=${bi_file#tests/moe_ep/test_}
        family=${family%_megamoe_batch_invariant.py}
        [[ " ${bi_families[*]} " == *" $family "* ]] || bi_families+=("$family")
    done
    if [ "${#bi_families[@]}" -gt 0 ]; then
        bi_command=("${PYTHON:-python}" tests/moe_ep/qualify_batch_invariant.py
            --suite multi --families "${bi_families[@]}"
            --world-size "${MEGAMOE_BI_WORLD_SIZE}" --output-dir "${JUNIT_DIR}/megamoe-bi")
        # Scope/launcher inspection must not install packages or touch GPUs.
        if [ "$DRY_RUN" == "true" ] && [ -z "${MPI_TEST_FILES}${SPAWN_MANAGED_TEST_FILES}${TORCHRUN_TEST_FILES}" ]; then
            "${bi_command[@]}" --plan
            test_status_complete 0
            exit 0
        fi
    fi
    if [ -n "${TEST_PATH:-}" ] && [ -z "${MPI_TEST_FILES}${SPAWN_MANAGED_TEST_FILES}${TORCHRUN_TEST_FILES}${MEGAMOE_BI_TEST_FILES}" ]; then
        echo "No multi-GPU files overlap TEST_PATH=${TEST_PATH}; skipping."
        test_status_complete 0
        exit 0
    fi

    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/setup_test_env.sh"
    install_and_verify
    if [ -n "$SM107_TEST_FILES" ]; then
        setup_moe_ep_test_env "$SM107_TEST_FILES"
    fi

    echo "Multi-GPU comm kernel test files (running with: ${PYTEST_COMMAND_PREFIX}):"
    for test_file in $MPI_TEST_FILES; do
        echo "  $test_file"
    done
    echo ""
    _run_pytest_group "$MPI_TEST_FILES"

    echo "Spawn-managed multi-GPU comm test files (running with plain pytest):"
    for test_file in $SPAWN_MANAGED_TEST_FILES; do
        echo "  $test_file"
    done
    echo ""
    PYTEST_COMMAND_PREFIX= _run_pytest_group "$SPAWN_MANAGED_TEST_FILES"

    echo "Multi-GPU torchrun test files:"
    for test_file in $TORCHRUN_TEST_FILES; do
        echo "  $test_file"
    done

    if [ "${#bi_families[@]}" -gt 0 ]; then
        echo "MegaMoE BI qualification (physical EP${MEGAMOE_BI_WORLD_SIZE}): ${bi_families[*]}"
        if [ "$DRY_RUN" == "true" ]; then
            "${bi_command[@]}" --plan
        else
            local bi_status=0 bi_receipt
            bi_receipt=$(test_status_start)
            TOTAL_TESTS=$((TOTAL_TESTS + 1))
            "${bi_command[@]}" || bi_status=$?
            if [ "$bi_status" -eq 0 ]; then
                PASSED_TESTS=$((PASSED_TESTS + 1))
            else
                record_failed_test "MegaMoE BI qualification"
                EXIT_CODE=1
            fi
            test_status_finish "${bi_receipt}" "${bi_status}"
        fi
    fi
    echo ""

    for test_file in $TORCHRUN_TEST_FILES; do
        local junit_file
        junit_file=$(junit_file_for_test "$test_file")
        echo "=========================================="
        echo "Running: ${TORCHRUN_PREFIX} -m pytest ${test_file} -v --junitxml=${junit_file}"
        echo "=========================================="
        if [ "$DRY_RUN" != "true" ]; then
            local receipt pytest_status=0
            receipt=$(test_status_start)
            local -a TEST_STATUS_PYTEST_COMMAND
            local -a torchrun_env=()
            if [[ " $SM107_TEST_FILES " == *" $test_file "* ]]; then
                torchrun_env+=(FAIL_ON_DISTRIBUTED_SKIP=1 FLASHINFER_STRICT_MOE_EP_TESTS=1 CUTE_DSL_ARCH=sm_107a)
            fi
            test_status_pytest_command "${receipt}" torchrun
            rm -f "$junit_file"
            TOTAL_TESTS=$((TOTAL_TESTS + 1))
            if env "${torchrun_env[@]}" ${TORCHRUN_PREFIX} "${TEST_STATUS_PYTEST_COMMAND[@]}" "${test_file}" -v --junitxml="${junit_file}"; then
                if [ -f "$junit_file" ]; then
                    echo "PASSED: $test_file"
                    PASSED_TESTS=$((PASSED_TESTS + 1))
                else
                    echo "NO RESULT: $test_file (missing JUnit XML: $junit_file)"
                    record_no_result_test "$test_file"
                fi
            else
                pytest_status=$?
                if [ -f "$junit_file" ]; then
                    echo "FAILED: $test_file"
                    record_failed_test "$test_file"
                else
                    echo "NO RESULT: $test_file (missing JUnit XML: $junit_file)"
                    record_no_result_test "$test_file"
                fi
            fi
            test_status_finish "${receipt}" "${pytest_status}"
        fi
        echo ""
    done

    # Counters are cumulative across all groups, so this is the overall summary.
    if [ "$DRY_RUN" != "true" ] && [ -n "${TORCHRUN_TEST_FILES}${MEGAMOE_BI_TEST_FILES}" ]; then
        print_execution_summary
    fi

    test_status_complete "$EXIT_CODE"
    exit "$EXIT_CODE"
}

main "$@"
