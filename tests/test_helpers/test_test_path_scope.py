import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPE_SH = REPO_ROOT / "scripts" / "test_path_scope.sh"


def _filter(test_path: str, files: str) -> str:
    env = os.environ.copy()
    env["TEST_PATH"] = test_path
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source {SCOPE_SH} && filter_files_by_test_path "$1"',
            "filter_files_by_test_path",
            files,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout


def test_empty_test_path_keeps_all_files() -> None:
    files = (
        "tests/comm/test_allreduce_unified_api.py "
        "tests/attention/test_parallel_attention.py"
    )
    assert _filter("", files) == files


def test_directory_keeps_nested_files_only() -> None:
    files = (
        "tests/comm/test_allreduce_unified_api.py "
        "tests/comm/test_quantized_allreduce.py "
        "tests/attention/test_parallel_attention.py "
        "tests/gemm/test_multi_gpu_cute_dsl_blockscaled_gemm_fusion.py"
    )
    assert _filter("tests/comm", files) == (
        "tests/comm/test_allreduce_unified_api.py "
        "tests/comm/test_quantized_allreduce.py"
    )


def test_file_token_selects_only_that_file() -> None:
    files = (
        "tests/comm/test_allreduce_unified_api.py "
        "tests/comm/test_quantized_allreduce.py"
    )
    assert (
        _filter("tests/comm/test_quantized_allreduce.py", files)
        == "tests/comm/test_quantized_allreduce.py"
    )


def test_unrelated_paths_select_nothing() -> None:
    files = "tests/comm/test_allreduce_unified_api.py tests/attention/test_parallel_attention.py"
    assert _filter("tests/cli tests/trace_apply", files) == ""


def test_multi_token_union() -> None:
    files = (
        "tests/comm/test_allreduce_unified_api.py "
        "tests/attention/test_parallel_attention.py "
        "tests/gemm/test_multi_gpu_cute_dsl_blockscaled_gemm_fusion.py"
    )
    assert _filter("tests/cli tests/attention", files) == (
        "tests/attention/test_parallel_attention.py"
    )


def test_prefix_does_not_match_sibling_name() -> None:
    files = "tests/comm/foo.py tests/communication/bar.py"
    assert _filter("tests/comm", files) == "tests/comm/foo.py"
