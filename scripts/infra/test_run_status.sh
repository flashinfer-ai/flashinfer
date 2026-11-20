#!/usr/bin/env bash
# Optional machine-readable status reporting. With no result directory set,
# existing local test invocations retain their behavior and incur no subprocesses.

test_status_report() {
    [ -n "${UNIT_TEST_RUN_STATUS_DIR:-}" ] || return 0
    local helper
    helper="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_run_status.py"
    python "${helper}" --directory "${UNIT_TEST_RUN_STATUS_DIR}" "$@"
}

test_status_begin() { test_status_report begin; }
test_status_start() { test_status_report start; }
test_status_finish() {
    [ -n "$1" ] || return 0
    test_status_report finish --receipt "$1" --status "$2"
}
test_status_complete() { test_status_report complete --status "$1"; }

# Populates an array rather than shell text, preserving paths and pytest args.
# torchrun starts Python itself; MPI and ordinary shells need the interpreter.
test_status_pytest_command() {
    local receipt=$1 launcher=${2:-shell} helper
    TEST_STATUS_PYTEST_COMMAND=(pytest)
    if [ -n "${UNIT_TEST_RUN_STATUS_DIR:-}" ]; then
        helper="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_run_status.py"
        TEST_STATUS_PYTEST_COMMAND=("${helper}" --directory "${UNIT_TEST_RUN_STATUS_DIR}"
            run --receipt "${receipt}" --pytest-args)
        if [ "${launcher}" != torchrun ]; then
            TEST_STATUS_PYTEST_COMMAND=(python "${TEST_STATUS_PYTEST_COMMAND[@]}")
        fi
    elif [ "${launcher}" = torchrun ]; then
        TEST_STATUS_PYTEST_COMMAND=(-m pytest)
    fi
}
