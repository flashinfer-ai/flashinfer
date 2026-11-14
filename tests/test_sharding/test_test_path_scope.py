import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_task_scripts_source_setup_after_scope_skip() -> None:
    for name in (
        "task_test_multi_gpu_comm_kernels.sh",
        "task_test_multi_node_comm_kernels.sh",
    ):
        text = (REPO_ROOT / "scripts" / name).read_text()
        skip = text.index("skipping.")
        setup = text.index('source "${SCRIPT_DIR}/setup_test_env.sh"')
        assert setup > skip, name
        assert text.count("setup_test_env.sh") == 1, name


def test_megamoe_distributed_files_are_registered() -> None:
    text = (REPO_ROOT / "scripts/task_test_multi_gpu_comm_kernels.sh").read_text()
    for name in (
        "test_moe_ep_sm107_block_scaled_mega_multirank.py",
        "test_sm107_routing_graph_multirank.py",
        "test_sm107_auto_multirank.py",
        "test_sm107_variant_selection_multirank.py",
    ):
        assert (REPO_ROOT / "tests/moe_ep" / name).is_file()
        assert "tests/moe_ep/" + name in text


@pytest.mark.parametrize(
    "cuda,capability,test_path,expected_count",
    [
        ("13.4", (10, 7), "", 4),
        ("13.4", (10, 7), "tests/moe_ep", 4),
        ("13.4", (10, 7), "tests/moe_ep/test_sm107_auto_multirank.py", 1),
        ("12.9", (10, 7), "tests/moe_ep", 0),
        ("13.4", (10, 0), "tests/moe_ep", 0),
        ("13.4", (10, 3), "tests/moe_ep", 0),
        ("13.4", (9, 0), "tests/moe_ep", 0),
        (None, None, "tests/moe_ep", 0),
        ("13.4", (10, 7), "tests/cli", 0),
        ("13.4", (10, 7), "tests/attention", 0),
        ("12.9", (10, 3), "tests/attention tests/moe_ep", 0),
    ],
)
def test_sm107_runner_selects_supported_lanes(
    tmp_path, cuda, capability, test_path, expected_count
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("task_test_multi_gpu_comm_kernels.sh", "test_path_scope.sh"):
        shutil.copy(REPO_ROOT / "scripts" / name, scripts)
    # Exercise selection through the real wrapper before package setup or launch.
    (scripts / "test_utils.sh").write_text(
        'source "${SCRIPT_DIR}/test_path_scope.sh"\n'
        "test_status_begin() { :; }\n"
        "test_status_complete() { :; }\n"
        "parse_args() { DRY_RUN=true; EXIT_CODE=0; }\n"
        "print_test_mode_banner() { :; }\n"
        "execute_dry_run() { :; }\n"
        "junit_file_for_test() { printf '%s.xml' \"$1\"; }\n"
    )
    (scripts / "setup_test_env.sh").write_text(
        "install_and_verify() { :; }\n"
        'setup_moe_ep_test_env() { printf "%s" "$1" > "$EP_SETUP_LOG"; }\n'
    )
    (tmp_path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        f"version = SimpleNamespace(cuda={cuda!r})\n"
        f"cuda = SimpleNamespace(is_available=lambda: {capability is not None}, "
        f"get_device_capability=lambda: {capability!r})\n"
    )
    setup_log = tmp_path / "ep-setup"
    env = os.environ.copy()
    env.update(
        PATH=f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}",
        TEST_PATH=test_path,
        EP_SETUP_LOG=str(setup_log),
    )
    result = subprocess.run(
        ["bash", str(scripts / "task_test_multi_gpu_comm_kernels.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    launched = [
        line for line in result.stdout.splitlines() if line.startswith("Running:")
    ]
    assert sum("tests/moe_ep/" in line for line in launched) == expected_count
    assert setup_log.exists() is (expected_count > 0)
    if expected_count:
        assert len(setup_log.read_text().split()) == expected_count
    if "tests/attention" in test_path:
        assert any(
            "tests/attention/test_parallel_attention.py" in line for line in launched
        )


@pytest.mark.parametrize(
    "paths,cuda,installed",
    [
        ("tests/moe_ep", "13", True),
        ("tests/comm/a.py tests/moe_ep/test_sm107_auto_multirank.py", "13", True),
        ("tests/comm", "13", False),
        ("tests/moe_ep_extra", "13", False),
        ("tests/moe_ep", "12", False),
    ],
)
def test_ep_runtime_setup_matches_the_selected_files(tmp_path, paths, cuda, installed):
    source = (REPO_ROOT / "scripts/setup_test_env.sh").read_text()
    start = source.index("setup_moe_ep_test_env() {")
    end = source.index("\n}\n", start) + 3
    definition = source[start:end]
    script = (
        definition
        + r"""
set -e
REPO_ROOT=/source
TEST_PATH=
LD_LIBRARY_PATH=/existing
python() {
  case "$*" in
    *'import torch'*) printf '%s\n' "$MOCK_CUDA" ;;
    *'import nvidia.nccl'*) printf '%s\n' /wheel/nccl/lib ;;
    *) return 1 ;;
  esac
}
bash() {
  test "$FI_SRC" = /source
  test "$1" = /source/docker/install/build_flashinfer_ep_pytorch.sh
  printf '%s\n' installed > "$INSTALL_LOG"
}
setup_moe_ep_test_env "$SELECTED_FILES"
printf '%s\n' "$LD_LIBRARY_PATH"
"""
    )
    log = tmp_path / "installed"
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "MOCK_CUDA": cuda,
            "SELECTED_FILES": paths,
            "INSTALL_LOG": str(log),
        },
    )
    assert log.exists() is installed
    assert result.stdout.strip() == (
        "/wheel/nccl/lib:/existing" if installed else "/existing"
    )
