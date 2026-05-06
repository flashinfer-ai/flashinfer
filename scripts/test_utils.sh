#!/bin/bash
# Common test functions for FlashInfer test scripts
# This file is meant to be sourced by test runner scripts

# Default environment variables
: "${JUNIT_DIR:=$(realpath ./junit)}"

# Cap ninja parallelism by available RAM (~12 GB per nvcc process) to avoid OOM
# during JIT compilation.  Exported because flashinfer/jit/cpp_ext.py reads it
# from the environment (os.environ) inside the child Python process.
if [ -z "${MAX_JOBS:-}" ]; then
    _num_cpus=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    _mem_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null)
    _mem_gb=${_mem_gb:-0}
    if [ "$_mem_gb" -gt 0 ]; then
        MAX_JOBS=$(( (_mem_gb - 8) / 12 ))
        [ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1
        [ "$MAX_JOBS" -gt "$_num_cpus" ] && MAX_JOBS=$_num_cpus
    else
        MAX_JOBS=$_num_cpus
    fi
    unset _num_cpus _mem_gb
fi
export MAX_JOBS

# CUDA_VISIBLE_DEVICES: Not set by default - let detect_gpus() auto-detect via nvidia-smi
: "${SAMPLE_RATE:=5}"  # Run every Nth test in sanity mode (5 = ~20% coverage)
: "${PARALLEL_TESTS:=false}"  # Disable parallel test execution by default

# Randomize starting offset (0 to SAMPLE_RATE-1) for sampling variety
if [ -z "${SAMPLE_OFFSET:-}" ]; then
    SAMPLE_OFFSET=$((RANDOM % SAMPLE_RATE))
fi

# Pytest configuration flags
PYTEST_FLAGS="--continue-on-collection-errors"

# Command prefix for pytest (e.g., "mpirun -np 4" for multi-GPU tests)
: "${PYTEST_COMMAND_PREFIX:=}"

# Global variables for test execution
FAILED_TESTS=""
NO_RESULT_TESTS=""
TOTAL_TESTS=0
PASSED_TESTS=0
NO_RESULT_COUNT=0
TOTAL_TEST_CASES=0
SAMPLED_TEST_CASES=0
# shellcheck disable=SC2034  # EXIT_CODE is used by calling scripts
EXIT_CODE=0

# Parse command line arguments
# Set DISABLE_SANITY_TEST=true before sourcing to disable sanity testing
: "${DISABLE_SANITY_TEST:=false}"

parse_args() {
    DRY_RUN=false
    SANITY_TEST=false
    for arg in "$@"; do
        case $arg in
            --dry-run)
                DRY_RUN=true
                ;;
            --sanity-test)
                if [ "$DISABLE_SANITY_TEST" = "true" ]; then
                    echo "⚠️  WARNING: Sanity testing is disabled for this test suite"
                    echo "    Running full tests instead"
                    echo ""
                else
                    SANITY_TEST=true
                fi
                ;;
        esac
    done
}

# Print test mode banner
print_test_mode_banner() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "🔍 DRY RUN MODE - No tests will be executed"
        echo ""
    fi

    if [ "$SANITY_TEST" = "true" ]; then
        echo "🔬 SANITY TEST MODE - Running every ${SAMPLE_RATE}th test (~$((100 / SAMPLE_RATE))% coverage)"
        echo "   Sampling pattern: offset=${SAMPLE_OFFSET} (tests #${SAMPLE_OFFSET}, #$((SAMPLE_OFFSET + SAMPLE_RATE)), #$((SAMPLE_OFFSET + SAMPLE_RATE * 2))...)"
        echo ""
    else
        echo "📋 FULL TEST MODE - Running all tests from each test file"
        echo ""
    fi
}

# Install precompiled kernels (CI build artifacts)
install_precompiled_kernels() {
    if [ "$DRY_RUN" = "true" ]; then
        return
    fi

    JIT_ARCH_EFFECTIVE=""
    # Map CUDA_VERSION to CUDA_STREAM for artifact lookup
    if [[ "${CUDA_VERSION}" == cu* ]]; then
        CUDA_STREAM="${CUDA_VERSION}"
    elif [ "${CUDA_VERSION}" = "12.9.0" ]; then
        CUDA_STREAM="cu129"
    else
        CUDA_STREAM="cu130"
    fi
    echo "Using CUDA stream: ${CUDA_STREAM}"
    echo ""

    if [ -n "${JIT_ARCH}" ]; then
        # 12.0a for CUDA 12.8, 12.0f for CUDA 12.9+
        if [ "${JIT_ARCH}" = "12.0" ]; then
            if [ "${CUDA_STREAM}" = "cu128" ]; then
                JIT_ARCH_EFFECTIVE="12.0a"
            else
                JIT_ARCH_EFFECTIVE="12.0f"
            fi
        else
            JIT_ARCH_EFFECTIVE="${JIT_ARCH}"
        fi

        echo "Using JIT_ARCH from environment: ${JIT_ARCH_EFFECTIVE}"
        DIST_CUBIN_DIR="../dist/${CUDA_STREAM}/${JIT_ARCH_EFFECTIVE}/cubin"
        DIST_JIT_CACHE_DIR="../dist/${CUDA_STREAM}/${JIT_ARCH_EFFECTIVE}/jit-cache"

        echo "==== Debug: listing artifact directories ===="
        echo "Tree under ../dist:"
        (cd .. && ls -al dist) || true
        echo ""
        echo "Tree under ../dist/${CUDA_STREAM}:"
        (cd .. && ls -al "dist/${CUDA_STREAM}") || true
        echo ""
        echo "Contents of ${DIST_CUBIN_DIR}:"
        ls -al "${DIST_CUBIN_DIR}" || true
        echo ""
        echo "Contents of ${DIST_JIT_CACHE_DIR}:"
        ls -al "${DIST_JIT_CACHE_DIR}" || true
        echo "============================================="

        if [ -d "${DIST_CUBIN_DIR}" ] && ls "${DIST_CUBIN_DIR}"/*.whl >/dev/null 2>&1; then
            echo "Installing flashinfer-cubin from ${DIST_CUBIN_DIR} ..."
            pip install -q "${DIST_CUBIN_DIR}"/*.whl
        else
            echo "ERROR: flashinfer-cubin wheel not found in ${DIST_CUBIN_DIR}. Ensure the CI build stage produced the artifact." >&2
        fi

        if [ -d "${DIST_JIT_CACHE_DIR}" ] && ls "${DIST_JIT_CACHE_DIR}"/*.whl >/dev/null 2>&1; then
            echo "Installing flashinfer-jit-cache from ${DIST_JIT_CACHE_DIR} ..."
            pip install -q "${DIST_JIT_CACHE_DIR}"/*.whl
        else
            echo "ERROR: flashinfer-jit-cache wheel not found in ${DIST_JIT_CACHE_DIR} for ${CUDA_VERSION}. Ensure the CI build stage produced the artifact." >&2
        fi
        echo ""
    fi
}

# Install and verify FlashInfer
install_and_verify() {
    if [ "$DRY_RUN" != "true" ]; then
        echo "Using CUDA version: ${CUDA_VERSION}"
        echo ""

        # Install precompiled kernels if enabled
        install_precompiled_kernels

        # Install local python sources
        pip install -e . -v --no-deps
        echo ""

        # Verify installation
        echo "Verifying installation..."
        (cd /tmp && python -m flashinfer show-config)
        echo ""
    fi
}

# Collect tests from a file
collect_tests() {
    local test_file=$1

    # Temporarily disable exit on error for collection
    set +e
    COLLECTION_OUTPUT=$(pytest --collect-only -q "$test_file" 2>&1)
    COLLECTION_EXIT_CODE=$?
    set -e

    ALL_NODE_IDS=$(echo "$COLLECTION_OUTPUT" | grep "::" || true)
}

# Return the expected JUnit XML path for a test file
junit_file_for_test() {
    local test_file=$1
    local flattened_test_file=${test_file//\//_}
    local test_hash
    test_hash=$(printf '%s' "$test_file" | cksum | awk '{print $1}')
    echo "${JUNIT_DIR}/${flattened_test_file}.${test_hash}.xml"
}

# Record a failed test file in the execution summary
record_failed_test() {
    local test_file=$1
    FAILED_TESTS="$FAILED_TESTS\n  - $test_file"
    # shellcheck disable=SC2034  # EXIT_CODE is used by calling scripts
    EXIT_CODE=1
}

# Record a test file that produced no result artifacts
record_no_result_test() {
    local test_file=$1
    NO_RESULT_TESTS="$NO_RESULT_TESTS\n  - $test_file"
    NO_RESULT_COUNT=$((NO_RESULT_COUNT + 1))
}

# Describe which execution artifacts are missing for a test file
describe_missing_artifacts() {
    local result_file=$1
    local junit_file=$2
    local -a missing=()

    if [ -n "$result_file" ] && [ ! -f "$result_file" ]; then
        missing+=("result marker")
    fi
    if [ ! -f "$junit_file" ]; then
        missing+=("JUnit XML: $junit_file")
    fi

    local IFS=', '
    echo "${missing[*]}"
}

# Sample tests based on SAMPLE_RATE and SAMPLE_OFFSET
sample_tests() {
    local all_node_ids=$1

    # Sample every Nth test with random offset
    SAMPLED_NODE_IDS=$(echo "$all_node_ids" | awk "NR % $SAMPLE_RATE == $SAMPLE_OFFSET")
    # Fallback: if no tests sampled (offset missed all tests), take the first test
    if [ -z "$SAMPLED_NODE_IDS" ] || [ "$(echo "$SAMPLED_NODE_IDS" | wc -l)" -eq 0 ]; then
        SAMPLED_NODE_IDS=$(echo "$all_node_ids" | head -1)
    fi
}

# Process a single test file for dry run (sanity mode)
dry_run_sanity_file() {
    local test_file=$1
    local file_count=$2

    echo ""
    echo "[$file_count] Collecting tests from: $test_file"

    collect_tests "$test_file"

    if [ -z "$ALL_NODE_IDS" ]; then
        if [ $COLLECTION_EXIT_CODE -ne 0 ]; then
            echo "  ⚠️  Collection failed for $test_file (skipping)"
        else
            echo "  ⚠️  No tests found in $test_file"
        fi
        return
    fi

    # Count total tests
    TOTAL_IN_FILE=$(echo "$ALL_NODE_IDS" | wc -l)
    TOTAL_TEST_CASES=$((TOTAL_TEST_CASES + TOTAL_IN_FILE))

    sample_tests "$ALL_NODE_IDS"
    SAMPLED_IN_FILE=$(echo "$SAMPLED_NODE_IDS" | wc -l)
    SAMPLED_TEST_CASES=$((SAMPLED_TEST_CASES + SAMPLED_IN_FILE))

    echo "  Total test cases: $TOTAL_IN_FILE"
    echo "  Sampled test cases: $SAMPLED_IN_FILE (every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET})"
    echo "  Sample of tests that would run:"
    echo "$SAMPLED_NODE_IDS" | head -5 | sed 's/^/    /' || true
    if [ "$SAMPLED_IN_FILE" -gt 5 ]; then
        echo "    ... and $((SAMPLED_IN_FILE - 5)) more"
    fi
}

# Process a single test file for dry run (full mode)
dry_run_full_file() {
    local test_file=$1

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    local junit_file
    junit_file=$(junit_file_for_test "$test_file")
    JUNIT_FLAG="--junitxml=${junit_file}"
    # shellcheck disable=SC2086  # PYTEST_COMMAND_PREFIX needs word splitting
    echo "$TOTAL_TESTS. ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS ${JUNIT_FLAG} \"${test_file}\""
}

# Print dry run summary
print_dry_run_summary() {
    if [ "$SANITY_TEST" == "true" ]; then
        echo ""
        echo "=========================================="
        echo "DRY RUN SUMMARY (SANITY MODE)"
        echo "=========================================="
        echo "Total test files: $FILE_COUNT"
        echo "Total test cases (full suite): $TOTAL_TEST_CASES"
        echo "Sampled test cases (sanity): $SAMPLED_TEST_CASES"
        if [ "$TOTAL_TEST_CASES" -gt 0 ]; then
            echo "Coverage: ~$((SAMPLED_TEST_CASES * 100 / TOTAL_TEST_CASES))%"
        else
            echo "Coverage: N/A (no tests collected)"
        fi
        echo "Sample rate: every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET}"
        echo ""
        echo "To reproduce this exact run:"
        echo "  SAMPLE_RATE=${SAMPLE_RATE} SAMPLE_OFFSET=${SAMPLE_OFFSET} $0 --sanity-test"
    else
        echo ""
        echo "=========================================="
        echo "DRY RUN SUMMARY"
        echo "=========================================="
        echo "Total test files that would be executed: $TOTAL_TESTS"
    fi

    echo ""
    echo "To actually run the tests, execute without --dry-run:"
    if [ "$SANITY_TEST" == "true" ]; then
        echo "  $0 --sanity-test"
        echo ""
        echo "To reproduce this exact sampling pattern:"
        echo "  SAMPLE_RATE=${SAMPLE_RATE} SAMPLE_OFFSET=${SAMPLE_OFFSET} $0 --sanity-test"
    else
        echo "  $0"
    fi
}

# Run a single test file in sanity mode
run_sanity_test_file() {
    local test_file=$1
    local file_count=$2
    local junit_file
    junit_file=$(junit_file_for_test "$test_file")

    echo "=========================================="
    echo "[$file_count] Processing: $test_file"
    echo "=========================================="

    echo "Collecting test cases..."

    collect_tests "$test_file"

    if [ -z "$ALL_NODE_IDS" ]; then
        if [ $COLLECTION_EXIT_CODE -ne 0 ]; then
            echo "⚠️  Collection failed for $test_file (skipping)"
        else
            echo "⚠️  No tests found in $test_file"
        fi
        echo ""
        return
    fi

    # Count total tests
    TOTAL_IN_FILE=$(echo "$ALL_NODE_IDS" | wc -l)
    TOTAL_TEST_CASES=$((TOTAL_TEST_CASES + TOTAL_IN_FILE))

    sample_tests "$ALL_NODE_IDS"
    SAMPLED_IN_FILE=$(echo "$SAMPLED_NODE_IDS" | wc -l)
    SAMPLED_TEST_CASES=$((SAMPLED_TEST_CASES + SAMPLED_IN_FILE))

    echo "Total test cases in file: $TOTAL_IN_FILE"
    echo "Running sampled test cases: $SAMPLED_IN_FILE (every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET})"

    if [ "$SAMPLED_IN_FILE" -eq 0 ]; then
        echo "⚠️  No tests sampled from $test_file, skipping"
        echo ""
        return
    fi

    # Create a bash array with the node IDs
    mapfile -t SAMPLED_NODE_IDS_ARRAY <<< "$SAMPLED_NODE_IDS"

    JUNIT_FLAG="--junitxml=${junit_file}"

    # Run pytest with the sampled node IDs
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    rm -f "$junit_file"

    # shellcheck disable=SC2086  # PYTEST_COMMAND_PREFIX and PYTEST_FLAGS need word splitting
    if ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${SAMPLED_NODE_IDS_ARRAY[@]}"; then
        if [ -f "$junit_file" ]; then
            echo "✅ PASSED: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests)"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo "⚠️  NO RESULT: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests, missing JUnit XML: $junit_file)"
            record_no_result_test "$test_file"
        fi
    else
        if [ -f "$junit_file" ]; then
            echo "❌ FAILED: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests)"
            record_failed_test "$test_file"
        else
            echo "⚠️  NO RESULT: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests, missing JUnit XML: $junit_file)"
            record_no_result_test "$test_file"
        fi
    fi

    echo ""
}

# Run a single test file in full mode
run_full_test_file() {
    local test_file=$1
    local junit_file
    junit_file=$(junit_file_for_test "$test_file")

    echo "=========================================="
    JUNIT_FLAG="--junitxml=${junit_file}"
    # shellcheck disable=SC2086  # PYTEST_COMMAND_PREFIX needs word splitting
    echo "Running: ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS ${JUNIT_FLAG} \"${test_file}\""
    echo "=========================================="

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    rm -f "$junit_file"

    # shellcheck disable=SC2086  # PYTEST_COMMAND_PREFIX and PYTEST_FLAGS need word splitting
    if ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${test_file}"; then
        if [ -f "$junit_file" ]; then
            echo "✅ PASSED: $test_file"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo "⚠️  NO RESULT: $test_file (missing JUnit XML: $junit_file)"
            record_no_result_test "$test_file"
        fi
    else
        if [ -f "$junit_file" ]; then
            echo "❌ FAILED: $test_file"
            record_failed_test "$test_file"
        else
            echo "⚠️  NO RESULT: $test_file (missing JUnit XML: $junit_file)"
            record_no_result_test "$test_file"
        fi
    fi

    echo ""
}

# Detect available GPUs from CUDA_VISIBLE_DEVICES or nvidia-smi
detect_gpus() {
    if [ "$PARALLEL_TESTS" != "true" ]; then
        echo "0"
        return
    fi

    # Parse CUDA_VISIBLE_DEVICES if set
    if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "-1" ]; then
        # Handle various formats: "0,1,2,3" or "0 1 2 3"
        AVAILABLE_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ' | tr -s ' ')
        echo "$AVAILABLE_GPUS"
        return
    fi

    # Fallback to nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | awk '{print NR-1}' | tr '\n' ' ' | sed 's/ $//')
        if [ -n "$AVAILABLE_GPUS" ]; then
            echo "$AVAILABLE_GPUS"
            return
        fi
    fi

    # Default to single GPU
    echo "0"
}

# Run tests in parallel across multiple GPUs
run_tests_parallel() {
    local test_files=$1
    local mode=$2  # "sanity" or "full"

    # Detect available GPUs
    local gpu_string
    gpu_string=$(detect_gpus)
    local -a GPU_LIST
    IFS=' ' read -r -a GPU_LIST <<< "$gpu_string"
    local NUM_GPUS=${#GPU_LIST[@]}

    # Auto-fallback to sequential if only one GPU
    if [ "$NUM_GPUS" -eq 1 ]; then
        echo "=========================================="
        echo "Only 1 GPU detected - using sequential execution"
        echo "=========================================="
        echo "GPU: ${GPU_LIST[0]}"
        echo ""
        # Run sequentially instead
        if [ "$mode" = "sanity" ]; then
            FILE_COUNT=0
            for test_file in $test_files; do
                FILE_COUNT=$((FILE_COUNT + 1))
                run_sanity_test_file "$test_file" "$FILE_COUNT"
            done
        else
            for test_file in $test_files; do
                run_full_test_file "$test_file"
            done
        fi
        return
    fi

    echo "=========================================="
    echo "PARALLEL EXECUTION MODE"
    echo "=========================================="
    echo "Available GPUs: ${GPU_LIST[*]}"
    echo "Number of GPUs: $NUM_GPUS"
    echo "Test mode: $mode"
    echo ""

    # Create a temporary directory for parallel job state
    PARALLEL_TMP_DIR=$(mktemp -d)

    # Preserve existing EXIT trap and add cleanup
    PREV_EXIT_TRAP=$(trap -p EXIT | sed -E "s/^trap -- '(.*)' EXIT$/\1/")
    trap 'rm -rf "$PARALLEL_TMP_DIR"; '"$PREV_EXIT_TRAP" EXIT

    # Convert test files to array
    local -a test_files_array
    IFS=' ' read -r -a test_files_array <<< "$test_files"
    local total_files=${#test_files_array[@]}

    echo "Total test files to execute: $total_files"
    echo ""

    # Create a results file for each test
    declare -A test_result_files
    declare -A test_pid_map
    declare -A test_gpu_map

    # Free GPU queue for proper GPU assignment
    local -a available_gpus=("${GPU_LIST[@]}")

    # Function to run a single test file
    run_single_test_background() {
        local test_file=$1
        local gpu_id=$2
        local file_index=$3
        local result_file="$PARALLEL_TMP_DIR/result_${file_index}"
        local log_file="$PARALLEL_TMP_DIR/log_${file_index}"
        local junit_file
        junit_file=$(junit_file_for_test "$test_file")

        (
            # Set GPU for this test
            export CUDA_VISIBLE_DEVICES=$gpu_id

            # Redirect output to log file
            exec > "$log_file" 2>&1
            rm -f "$junit_file"

            echo "=========================================="
            echo "[$file_index/$total_files] Processing: $test_file"
            echo "GPU: $gpu_id"
            echo "=========================================="

            if [ "$mode" = "sanity" ]; then
                # Run sanity test
                collect_tests "$test_file"

                if [ -z "$ALL_NODE_IDS" ]; then
                    if [ $COLLECTION_EXIT_CODE -ne 0 ]; then
                        echo "⚠️  Collection failed for $test_file (skipping)"
                    else
                        echo "⚠️  No tests found in $test_file"
                    fi
                    echo "SKIPPED" > "$result_file"
                    exit 0
                fi

                TOTAL_IN_FILE=$(echo "$ALL_NODE_IDS" | wc -l)
                sample_tests "$ALL_NODE_IDS"
                SAMPLED_IN_FILE=$(echo "$SAMPLED_NODE_IDS" | wc -l)

                if [ "$SAMPLED_IN_FILE" -eq 0 ]; then
                    echo "⚠️  No tests sampled from $test_file, skipping"
                    echo "SKIPPED" > "$result_file"
                    exit 0
                fi

                mapfile -t SAMPLED_NODE_IDS_ARRAY <<< "$SAMPLED_NODE_IDS"
                JUNIT_FLAG="--junitxml=${junit_file}"

                # shellcheck disable=SC2086
                if ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${SAMPLED_NODE_IDS_ARRAY[@]}"; then
                    echo "✅ PASSED: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests)"
                    echo "PASSED:$TOTAL_IN_FILE:$SAMPLED_IN_FILE" > "$result_file"
                else
                    echo "❌ FAILED: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests)"
                    echo "FAILED:$TOTAL_IN_FILE:$SAMPLED_IN_FILE" > "$result_file"
                fi
            else
                # Run full test
                JUNIT_FLAG="--junitxml=${junit_file}"

                # shellcheck disable=SC2086
                if ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${test_file}"; then
                    echo "✅ PASSED: $test_file"
                    echo "PASSED" > "$result_file"
                else
                    echo "❌ FAILED: $test_file"
                    echo "FAILED" > "$result_file"
                fi
            fi
        ) &

        local pid=$!
        echo "$pid:$test_file:$result_file:$log_file:$file_index:$junit_file"
    }

    # Launch tests in parallel with GPU queue
    echo "Launching tests in parallel..."
    local test_idx=0
    local completed=0
    while [ $test_idx -lt $total_files ]; do
        # Wait for a GPU to become available
        while [ ${#available_gpus[@]} -eq 0 ]; do
            # Check for finished jobs and reclaim their GPUs
            for pid in "${!test_pid_map[@]}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    # Job finished, reclaim its GPU
                    wait "$pid" 2>/dev/null || true
                    local freed_gpu="${test_gpu_map[$pid]}"
                    local finished_test="${test_pid_map[$pid]}"
                    available_gpus+=("$freed_gpu")
                    completed=$((completed + 1))
                    local running=$((${#test_pid_map[@]} - 1))
                    echo "[Progress: ${completed}/${total_files} completed, ${running} running] Finished: $(basename "$finished_test") (GPU ${freed_gpu})"
                    unset "test_pid_map[$pid]"
                    unset "test_gpu_map[$pid]"
                fi
            done
            # Small sleep to avoid busy-waiting
            [ ${#available_gpus[@]} -eq 0 ] && sleep 0.1
        done

        # Get next available GPU
        local gpu_id="${available_gpus[0]}"
        available_gpus=("${available_gpus[@]:1}")  # Remove first element

        # Launch test on this GPU
        local test_file="${test_files_array[$test_idx]}"
        local file_index=$((test_idx + 1))
        echo "[${file_index}/${total_files}] Launching: $(basename "$test_file") on GPU ${gpu_id}"
        local job_info
        job_info=$(run_single_test_background "$test_file" "$gpu_id" "$file_index")

        # Parse job info
        local pid result_file log_file junit_file
        IFS=':' read -r pid test_file result_file log_file file_index junit_file <<< "$job_info"
        test_result_files[$pid]="$result_file:$test_file:$log_file:$file_index:$junit_file"
        test_pid_map[$pid]="$test_file"
        test_gpu_map[$pid]="$gpu_id"

        test_idx=$((test_idx + 1))
    done

    # Wait for all remaining jobs with progress
    echo ""
    echo "All tests launched. Waiting for remaining ${#test_pid_map[@]} tests to complete..."
    for pid in "${!test_pid_map[@]}"; do
        wait "$pid" 2>/dev/null || true
        local finished_test="${test_pid_map[$pid]}"
        local freed_gpu="${test_gpu_map[$pid]}"
        completed=$((completed + 1))
        echo "[Progress: ${completed}/${total_files} completed] Finished: $(basename "$finished_test") (GPU ${freed_gpu})"
    done

    echo ""
    echo "All tests completed. Processing results..."
    echo ""

    # Sort results by file_index for deterministic output
    local -a sorted_pids=()
    for pid in "${!test_result_files[@]}"; do
        local result_file test_file log_file file_index junit_file
        IFS=':' read -r result_file test_file log_file file_index junit_file <<< "${test_result_files[$pid]}"
        sorted_pids+=("$file_index:$pid")
    done
    local sorted_list
    sorted_list=$(printf '%s\n' "${sorted_pids[@]}" | sort -n)
    mapfile -t sorted_pids <<< "$sorted_list"

    # Process results in order
    for entry in "${sorted_pids[@]}"; do
        local pid="${entry#*:}"
        local result_file test_file log_file file_index junit_file
        IFS=':' read -r result_file test_file log_file file_index junit_file <<< "${test_result_files[$pid]}"

        # Show log output
        if [ -f "$log_file" ]; then
            cat "$log_file"
            echo ""
        fi

        # Process result
        if [ -f "$result_file" ]; then
            local result
            result=$(cat "$result_file")
            if [[ "$result" == SKIPPED* ]]; then
                continue
            fi

            TOTAL_TESTS=$((TOTAL_TESTS + 1))

            if [ "$mode" = "sanity" ] && [[ "$result" == PASSED:* || "$result" == FAILED:* ]]; then
                local total_in_file sampled_in_file
                # shellcheck disable=SC2034  # status is part of the read but unused
                IFS=':' read -r _ total_in_file sampled_in_file <<< "$result"
                TOTAL_TEST_CASES=$((TOTAL_TEST_CASES + total_in_file))
                SAMPLED_TEST_CASES=$((SAMPLED_TEST_CASES + sampled_in_file))
            fi

            if [ ! -f "$junit_file" ]; then
                echo "⚠️  NO RESULT: $test_file (missing JUnit XML: $junit_file)"
                record_no_result_test "$test_file"
                continue
            fi

            if [[ "$result" == PASSED* ]]; then
                PASSED_TESTS=$((PASSED_TESTS + 1))
            elif [[ "$result" == FAILED* ]]; then
                record_failed_test "$test_file"
            fi
        else
            TOTAL_TESTS=$((TOTAL_TESTS + 1))
            local missing_artifacts
            missing_artifacts=$(describe_missing_artifacts "$result_file" "$junit_file")
            echo "⚠️  NO RESULT: $test_file (missing ${missing_artifacts})"
            record_no_result_test "$test_file"
        fi
    done
}

# Print execution summary
print_execution_summary() {
    if [ "$SANITY_TEST" == "true" ]; then
        local failed_count=$((TOTAL_TESTS - PASSED_TESTS - NO_RESULT_COUNT))
        echo "=========================================="
        echo "SANITY TEST SUMMARY"
        echo "=========================================="
        echo "Total test files executed: $TOTAL_TESTS"
        echo "Test files passed: $PASSED_TESTS"
        echo "Test files failed: $failed_count"
        echo "Test files with no result: $NO_RESULT_COUNT"
        echo ""
        echo "Total test cases (full suite): $TOTAL_TEST_CASES"
        echo "Sampled test cases (executed): $SAMPLED_TEST_CASES"
        if [ "$TOTAL_TEST_CASES" -gt 0 ]; then
            echo "Coverage: ~$((SAMPLED_TEST_CASES * 100 / TOTAL_TEST_CASES))%"
        else
            echo "Coverage: N/A (no tests collected)"
        fi
        echo "Sample rate: every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET}"
        echo ""
        echo "To reproduce this exact run:"
        echo "  SAMPLE_RATE=${SAMPLE_RATE} SAMPLE_OFFSET=${SAMPLE_OFFSET} $0 --sanity-test"
    else
        local failed_count=$((TOTAL_TESTS - PASSED_TESTS - NO_RESULT_COUNT))
        echo "=========================================="
        echo "TEST SUMMARY"
        echo "=========================================="
        echo "Total test files executed: $TOTAL_TESTS"
        echo "Passed: $PASSED_TESTS"
        echo "Failed: $failed_count"
        echo "No result: $NO_RESULT_COUNT"
    fi

    if [ -n "$FAILED_TESTS" ]; then
        echo ""
        echo "Failed test files:"
        echo -e "$FAILED_TESTS"
    fi

    if [ -n "$NO_RESULT_TESTS" ]; then
        echo ""
        echo "Test files with no result:"
        echo -e "$NO_RESULT_TESTS"
    fi
}

# Main execution function for dry run mode
execute_dry_run() {
    local test_files=$1

    echo "=========================================="
    echo "DRY RUN: Tests that would be executed"
    echo "=========================================="

    if [ "$SANITY_TEST" == "true" ]; then
        FILE_COUNT=0
        for test_file in $test_files; do
            FILE_COUNT=$((FILE_COUNT + 1))
            dry_run_sanity_file "$test_file" "$FILE_COUNT"
        done
    else
        for test_file in $test_files; do
            dry_run_full_file "$test_file"
        done
    fi

    print_dry_run_summary
}

# Main execution function for actual test run
execute_tests() {
    local test_files=$1

    mkdir -p "${JUNIT_DIR}"

    # Check if parallel execution is enabled
    if [ "$PARALLEL_TESTS" == "true" ]; then
        # Run tests in parallel
        if [ "$SANITY_TEST" == "true" ]; then
            run_tests_parallel "$test_files" "sanity"
        else
            run_tests_parallel "$test_files" "full"
        fi
    else
        # Original sequential execution
        if [ "$SANITY_TEST" == "true" ]; then
            FILE_COUNT=0
            for test_file in $test_files; do
                FILE_COUNT=$((FILE_COUNT + 1))
                run_sanity_test_file "$test_file" "$FILE_COUNT"
            done
        else
            for test_file in $test_files; do
                run_full_test_file "$test_file"
            done
        fi
    fi

    print_execution_summary
}
