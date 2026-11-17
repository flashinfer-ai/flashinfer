"""CPU tests for the AOT spec CUDA-target lint (no torch, nvcc or GPU needed)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture
def cuda_architecture():
    module = _load(
        "_test_cuda_architecture", REPO_ROOT / "flashinfer/jit/_cuda_architecture.py"
    )
    yield module
    sys.modules.pop("_test_cuda_architecture", None)


@pytest.fixture
def lint_script():
    module = _load(
        "_test_check_jit_spec_gencodes",
        REPO_ROOT / "scripts/check_jit_spec_gencodes.py",
    )
    yield module
    sys.modules.pop("_test_check_jit_spec_gencodes", None)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("flags", "sass", "ptx"),
    [
        (["-O3", "-DNDEBUG"], [], []),
        (["-gencode=arch=compute_89,code=sm_89"], ["sm89"], []),
        (
            [
                "-gencode=arch=compute_89,code=sm_89",
                "-gencode=arch=compute_120a,code=sm_120a",
            ],
            ["sm120a", "sm89"],
            [],
        ),
        (["-gencode", "arch=compute_90a,code=sm_90a"], ["sm90a"], []),
        (["--generate-code=arch=compute_120f,code=sm_120f"], ["sm120f"], []),
        (
            ["-gencode=arch=compute_89,code=[sm_89,compute_89]"],
            ["sm89"],
            ["compute89"],
        ),
        (['-gencode=arch=compute_89,code="sm_89,compute_89"'], ["sm89"], ["compute89"]),
        (["-arch=sm_90a"], ["sm90a"], ["compute90a"]),
        (["--gpu-architecture=compute_80"], [], ["compute80"]),
        (["-arch=native"], [], []),
        (["-gencode=arch=compute_80,code=lto_80"], [], []),
        (["-gencode"], [], []),
    ],
)
def test_cuda_targets_from_nvcc_flags(cuda_architecture, flags, sass, ptx):
    assert cuda_architecture.cuda_targets_from_nvcc_flags(flags) == (sass, ptx)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("provider", "flags", "expected"),
    [
        # No explicit targets: the provider-filtered context decides.
        ("sm89", ["-O3"], []),
        ("sm89", ["-gencode=arch=compute_89,code=sm_89"], []),
        ("sm90a", ["-gencode=arch=compute_90a,code=sm_90a"], []),
        # An unsuffixed or family target runs on newer parts of its major.
        ("sm86", ["-gencode=arch=compute_80,code=sm_80"], []),
        ("sm121a", ["-gencode=arch=compute_120f,code=sm_120f"], []),
    ],
)
def test_provider_flag_issues_accepts_compatible_targets(
    cuda_architecture, provider, flags, expected
):
    assert cuda_architecture.provider_flag_issues(flags, provider) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize("provider", ["sm89", "sm120f"])
def test_provider_flag_issues_rejects_fat_sm89_sm120a_module(
    cuda_architecture, provider
):
    # gen_ulysses_lowp_module() once emitted both targets unconditionally, so the
    # SM89 and SM120 providers each carried a binary the other could not run.
    flags = [
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_120a,code=sm_120a",
    ]

    issues = cuda_architecture.provider_flag_issues(flags, provider)

    foreign = "sm120a" if provider == "sm89" else "sm89"
    assert len(issues) == 1
    assert foreign in issues[0] and provider in issues[0]


@pytest.mark.cpu_only
def test_provider_flag_issues_rejects_other_architecture_and_ptx(cuda_architecture):
    assert cuda_architecture.provider_flag_issues(
        ["-gencode=arch=compute_100a,code=sm_100a"], "sm90a"
    ) == ["emits sm100a, which sm90a cannot run"]

    issues = cuda_architecture.provider_flag_issues(
        ["-gencode=arch=compute_90a,code=[sm_90a,compute_90a]"], "sm90a"
    )
    assert len(issues) == 1 and "PTX" in issues[0] and "compute90a" in issues[0]


@pytest.mark.cpu_only
def test_spec_flag_lists_covers_shared_and_per_source_flags(lint_script):
    spec = SimpleNamespace(
        name="module",
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_89,code=sm_89"],
        extra_cuda_cflags_by_source={
            Path("/csrc/special.cu"): ["-gencode=arch=compute_120a,code=sm_120a"]
        },
    )

    assert lint_script.spec_flag_lists(spec) == [
        ("module", ["-O3", "-gencode=arch=compute_89,code=sm_89"]),
        ("module (special.cu)", ["-gencode=arch=compute_120a,code=sm_120a"]),
    ]


@pytest.mark.cpu_only
def test_spec_flag_lists_ignores_specs_without_nvcc_flags(lint_script):
    assert lint_script.spec_flag_lists(SimpleNamespace(name="cute")) == []
    assert (
        lint_script.spec_flag_lists(
            SimpleNamespace(
                name="plain", extra_cuda_cflags=None, extra_cuda_cflags_by_source=None
            )
        )
        == []
    )


@pytest.mark.cpu_only
def test_aot_script_lints_specs_before_building_and_inspects_wheels_after():
    text = (
        REPO_ROOT / "scripts/task_test_jit_cache_package_build_import.sh"
    ).read_text()
    common = (REPO_ROOT / "scripts/jit_cache_build_common.sh").read_text()

    lint = text.index("run_provider_spec_lint ")
    build_started = text.index("PROVIDER_BUILD_STARTED_AT=")
    inspect = text.index('"${SCRIPT_DIR}/verify_jit_cache_provider_architectures.py"')
    assert lint < build_started < inspect
    assert "check_jit_spec_gencodes.py" in common


@pytest.mark.cpu_only
def test_standalone_lint_script_covers_both_cpu_architectures():
    text = (REPO_ROOT / "scripts/task_check_jit_spec_gencodes.sh").read_text()

    assert "x86_64_provider_architectures" in text
    assert "aarch64_provider_architectures" in text
    assert "run_provider_spec_lint" in text


@pytest.mark.cpu_only
def test_pr_test_runs_spec_lint_and_requires_it():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/pr-test.yml").read_text())
    jobs = workflow["jobs"]

    lint = jobs["spec-lint"]
    assert lint["needs"] == ["gate", "setup"]
    # Authorized runs only: the lane executes the PR's code on a self-hosted runner.
    assert "needs.gate.outputs.authorized == 'true'" in lint["if"]
    assert "task_check_jit_spec_gencodes.sh" in str(lint["steps"])
    assert "spec-lint" in jobs["test-results-summary"]["needs"]
