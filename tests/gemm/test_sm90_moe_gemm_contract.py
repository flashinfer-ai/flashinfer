"""FFI and workspace contracts for the private SM90 push FP8 MoE GEMM."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.usefixtures("isolated_deep_gemm_cache")

_FP8_SOURCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "flashinfer"
    / "moe_ep"
    / "kernel_src"
    / "sm90_push_megamoe"
    / "src"
    / "fp8_gemm"
)


def _private_source(name: str) -> str:
    return (_FP8_SOURCE_DIR / name).read_text(encoding="utf-8")


def _assert_in_order(source: str, fragments: tuple[str, ...]) -> None:
    offset = 0
    for fragment in fragments:
        offset = source.index(fragment, offset) + len(fragment)


def test_private_fp8_host_headers_resolve_deep_gemm_symbols() -> None:
    scheduler = _private_source("fp8_moe_scheduler.cuh")
    binding = _private_source("fp8_moe_binding.cu")
    launcher = _private_source("fp8_moe_launcher.cuh")
    jit = _private_source("fp8_moe_jit.cuh")

    assert "deep_gemm::ceil_div" not in scheduler
    assert "deep_gemm::ceil_div" not in binding
    assert "deep_gemm::div_up" not in launcher
    assert launcher.count("deep_gemm::jit::div_up") == 2
    assert '#include "fp8_moe_scheduler.cuh"' in launcher
    assert "#include <tensorrt_llm/deep_gemm/compiler.cuh>" in jit


def test_private_fp8_jit_requires_a_source_digest() -> None:
    jit = _private_source("fp8_moe_jit.cuh")

    assert '#include "fp8_moe_build_config.h"' in jit
    assert '#error "fp8_moe_build_config.h is required' in jit
    assert '"source-tree"' not in jit


def test_private_fp8_jit_checks_exact_caches_before_nvcc() -> None:
    binding = _private_source("fp8_moe_binding.cu")
    launcher = _private_source("fp8_moe_launcher.cuh")
    jit = _private_source("fp8_moe_jit.cuh")
    runner = (_FP8_SOURCE_DIR.parents[1] / "shim" / "runner.py").read_text(
        encoding="utf-8"
    )

    for name in (
        "get_deepgemm_cache_dir",
        "get_deepgemm_nvcc_compiler",
        "is_deepgemm_jit_enabled",
        "is_moe_gemm_jit_cache_ready",
        "is_moe_gemm_fc1_fused_jit_cache_ready",
    ):
        assert f'if (name == "{name}")' in binding

    assert "grouped_kernel_cache_path" in jit
    assert "FLASHINFER_SM90_PUSH_FP8_MOE_SOURCE_DIGEST" in jit
    assert "getGlobalRuntimeCache()[path.string()] != nullptr" in jit
    assert "fp8_moe_gemm_jit_cache_ready" in launcher
    assert "fp8_moe_fc1_fused_jit_cache_ready" in launcher

    project_root = Path(__file__).resolve().parents[2]
    compiler = (
        project_root / "csrc/nv_internal/tensorrt_llm/deep_gemm/compiler.cuh"
    ).read_text(encoding="utf-8")
    upstream_start = compiler.index(
        'std::string name = std::string(swapAB ? "gemm_swapAB_" : "gemm_")'
    )
    private_start = jit.index(
        "std::string const name =", jit.index("grouped_kernel_cache_path")
    )
    upstream_key = compiler[upstream_start : compiler.index(";", upstream_start) + 1]
    private_key = jit[private_start : jit.index(";", private_start) + 1]

    def normalize_key(source: str) -> str:
        return (
            "".join(source.split())
            .replace("const", "")
            .replace("swap_ab", "swapAB")
            .replace("gemm_type_to_string(gemm_type)", '"GroupedWithOffset"')
            .replace('+"_"+"GroupedWithOffset"', '+"_GroupedWithOffset"')
        )

    assert normalize_key(private_key) == normalize_key(upstream_key)

    _assert_in_order(
        runner,
        (
            "def _missing_gemm_jit_caches",
            "def _require_gemm_jit_runtime",
            "self._nvcc_available(compiler)",
            "def _prepare_gemm_jit_collective",
            '"gemm-jit-cache-warm"',
            '"gemm-jit-cache-load"',
        ),
    )


def test_private_fp8_jit_tactic_probe_matches_grouped_dispatch() -> None:
    project_root = Path(__file__).resolve().parents[2]
    upstream = (
        project_root
        / "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels"
        / "fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh"
    ).read_text(encoding="utf-8")
    private = _private_source("fp8_moe_launcher.cuh")
    upstream_body = upstream[
        upstream.index("void grouped_gemm_dispatch(") : upstream.index(
            "void fp8_grouped_gemm_run("
        )
    ]
    private_body = private[
        private.index("inline bool fp8_moe_gemm_jit_cache_ready(") : private.index(
            "inline bool fp8_moe_fc1_fused_jit_cache_ready("
        )
    ]

    def normalize_body(source: str) -> str:
        source = re.sub(r"static_cast<uint32_t>\(([^()]*)\)", r"\1", source)
        return "".join(source.split()).replace("kNumDeviceSMs", "num_device_sms")

    def config_calls(source: str) -> list[str]:
        source = normalize_body(source)
        return re.findall(r"deep_gemm::jit::get_best_gemm_config\(([^()]*)\)", source)

    normalized_upstream = normalize_body(upstream_body)
    normalized_private = normalize_body(private_body)
    assert (
        normalized_private.count("constexpruint32_tblock_k=128;")
        == normalized_upstream.count("constexpruint32_tblock_k=128;")
        == 1
    )
    threshold = "m_per_expert_threshold=num_device_sms==78?64:32;"
    assert threshold in normalized_upstream
    assert threshold in normalized_private
    assert "if(expected_m>=m_per_expert_threshold)" in normalized_upstream
    assert "swap_ab=expected_m<m_per_expert_threshold;" in normalized_private
    assert "swap_ab?deep_gemm::jit::get_best_gemm_config(" in normalized_private

    upstream_calls = config_calls(upstream_body)
    private_calls = config_calls(private_body)
    assert len(upstream_calls) == len(private_calls) == 2
    assert private_calls == [upstream_calls[1], upstream_calls[0]]


def test_private_fc1_kernel_requires_its_paired_scheduler() -> None:
    scheduler = _private_source("fp8_moe_scheduler.cuh")
    kernel = _private_source("fp8_moe_fc1_fused.cuh")

    assert "static constexpr bool kIsFp8MoeFc1Scheduler = true;" in scheduler
    assert "IsFp8MoeFc1Scheduler<SchedulerType>::value" in kernel
    assert "requires Fp8MoeFc1Scheduler" in kernel


def test_fused_offsets_preflight_runs_once_only_for_untrusted_offsets() -> None:
    binding = _private_source("fp8_moe_binding.cu")
    kernel = _private_source("fp8_moe_fc1_fused.cuh")
    launcher = _private_source("fp8_moe_launcher.cuh")
    fused_runner = binding[
        binding.index("void run_fc1_fused") : binding.index("bool queried_")
    ]

    assert binding.count("if (!trusted_offsets)") == 2
    assert binding.count("offsets_preflight_kernel<<<1, 32, 0, stream>>>") == 2
    assert fused_runner.index("if (!trusted_offsets)") < fused_runner.index(
        "fp8_moe_fc1_fused("
    )
    assert "cudaGetLastError()" in fused_runner
    assert "(void)trusted_offsets" not in binding
    assert "for (uint32_t g = 1; g <= kNumGroups" not in kernel
    assert "fc1_fused: bad offsets" not in kernel
    assert "int64_t a_rows" not in kernel
    assert "int64_t a_rows" not in launcher


def test_private_fc1_launch_abi_matches_generated_kernel() -> None:
    kernel = " ".join(_private_source("fp8_moe_fc1_fused.cuh").split())
    launcher = " ".join(_private_source("fp8_moe_launcher.cuh").split())
    kernel_start = kernel.index("fp8_gemm_kernel_sm90_push_fc1_fused(")
    kernel_end = kernel.index(") {", kernel_start)
    launch_start = launcher.index("cudaLaunchKernelEx(")
    launch_end = launcher.index(");", launch_start)

    _assert_in_order(
        kernel[kernel_start:kernel_end],
        (
            "gmem_d_fp8",
            "d_rows",
            "gmem_sfa_out",
            "sfa_out_stride",
            "scales_b",
            "problem_input",
            "tensor_map_a",
            "tensor_map_b",
            "tensor_map_scales_a",
        ),
    )
    _assert_in_order(
        launcher[launch_start:launch_end],
        (
            "&config",
            "kernel",
            "mat_d_fp8",
            "d_rows",
            "sfa_out",
            "max_shape_m_padded",
            "scales_b",
            "input",
            "tma_a_desc",
            "tma_b_desc",
            "tma_scales_a_desc",
        ),
    )


def test_production_runner_marks_protocol_offsets_trusted() -> None:
    runner_path = _FP8_SOURCE_DIR.parents[1] / "shim" / "runner.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"moe_gemm", "moe_gemm_fc1_fused"}
    ]

    assert sum(call.func.attr == "moe_gemm" for call in calls) == 4
    assert sum(call.func.attr == "moe_gemm_fc1_fused" for call in calls) == 2
    assert all(
        call.args
        and isinstance(call.args[-1], ast.Constant)
        and call.args[-1].value is True
        for call in calls
    )


def test_capacity_factor_bounds_private_gemm_storage() -> None:
    binding = _private_source("fp8_moe_binding.cu")
    package_dir = _FP8_SOURCE_DIR.parents[1]
    runner = (package_dir / "shim" / "runner.py").read_text(encoding="utf-8")
    protocol = (package_dir / "shim" / "protocol.py").read_text(encoding="utf-8")

    assert "expected_m_ = expected_m;" in binding
    assert "max_rows_ = ceil_div(max_rows, int64_t{4}) * 4;" in binding
    assert "expected_m exceeds int32" in binding
    assert "padded max_rows exceeds int32" in binding
    assert (
        "padded_rows_ = deep_gemm::compute_padded_offset(max_rows, num_problems);"
        in binding
    )

    resources = runner[
        runner.index("def _init_gemm_resources") : runner.index(
            "def configure_workspace"
        )
    ]
    assert "m_buf = (m_cap + 127) // 128 * 128" in resources
    assert "p_ws = max((m_cap + E * 31) // 32 * 32, 1)" in resources
    assert "m_ws" not in resources

    configure = runner[
        runner.index("def configure_workspace") : runner.index("_FC1_FUSED_FAIL_HELP")
    ]
    _assert_in_order(
        configure,
        (
            "pipe.token_capacity * pipe.K",
            "pipe.m_cap",
            "max(two_i, pipe.H)",
            "max(pipe.H, self.I)",
            "pipe.E",
        ),
    )

    wait_prefix = protocol[
        protocol.index("def proto_wait_prefix") : protocol.index("def proto_compact")
    ]
    assert "self.m_cap," in wait_prefix
    assert "self.m_ws," not in wait_prefix

    full_rows = 8 * 4096 * 8
    capped_rows = full_rows // 4
    hidden = 7168
    full_y_bytes = full_rows * hidden * 2
    capped_y_bytes = capped_rows * hidden * 2
    assert capped_y_bytes * 4 == full_y_bytes
    assert capped_y_bytes == 896 * 1024**2


def _sm90_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from flashinfer.jit.cpp_ext import is_cuda_version_at_least
        from flashinfer.utils import is_sm90a_supported

        return is_cuda_version_at_least("12.8") and is_sm90a_supported(
            torch.device("cuda")
        )
    except Exception:
        return False


requires_sm90 = pytest.mark.skipif(
    not _sm90_available(),
    reason="requires an SM90 GPU and CUDA Toolkit 12.8+",
)


def _create_runner():
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim.gemm import (
        create_sm90_push_fp8_moe_gemm_runner,
    )

    return create_sm90_push_fp8_moe_gemm_runner()


def _fp8_case() -> SimpleNamespace:
    device = torch.device("cuda", 0)
    groups, rows, hidden, intermediate = 2, 64, 256, 256
    runner = _create_runner()
    workspace_size = runner.get_moe_workspace_size(
        rows,
        rows,
        max(2 * intermediate, hidden),
        max(hidden, intermediate),
        groups,
        True,
        True,
    )
    runner.configure_workspace(
        torch.empty(max(int(workspace_size), 1), device=device, dtype=torch.uint8)
    )
    padded_rows = (rows + groups * 31) // 32 * 32
    return SimpleNamespace(
        runner=runner,
        groups=groups,
        rows=rows,
        hidden=hidden,
        intermediate=intermediate,
        a=torch.zeros(rows, hidden, dtype=torch.float8_e4m3fn, device=device),
        b=torch.zeros(
            groups,
            2 * intermediate,
            hidden,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        scales_a=torch.zeros(
            (hidden // 128) * padded_rows + 128,
            dtype=torch.float32,
            device=device,
        ),
        scales_b=torch.zeros(
            groups,
            2 * intermediate // 128,
            hidden // 128,
            dtype=torch.float32,
            device=device,
        ),
        output=torch.zeros(
            rows,
            2 * intermediate,
            dtype=torch.bfloat16,
            device=device,
        ),
        fused_output=torch.zeros(
            rows,
            intermediate,
            dtype=torch.uint8,
            device=device,
        ),
        fused_scales=torch.zeros(
            (intermediate // 128) * padded_rows + 128,
            dtype=torch.float32,
            device=device,
        ),
        offsets=torch.zeros(groups + 1, dtype=torch.int64, device=device),
    )


@requires_sm90
def test_moe_gemm_rejects_invalid_ffi_arguments() -> None:
    case = _fp8_case()

    def call(
        *,
        output=None,
        a=None,
        b=None,
        offsets=None,
        n=None,
        k=None,
        scales_a="default",
        scales_b="default",
    ) -> None:
        case.runner.moe_gemm(
            case.output if output is None else output,
            case.a if a is None else a,
            case.b if b is None else b,
            case.offsets if offsets is None else offsets,
            2 * case.intermediate if n is None else n,
            case.hidden if k is None else k,
            case.scales_a if scales_a == "default" else scales_a,
            case.scales_b if scales_b == "default" else scales_b,
            False,
        )

    call()
    torch.cuda.synchronize()
    with pytest.raises(Exception, match="int64"):
        call(offsets=case.offsets.to(torch.int32))
    with pytest.raises(Exception, match="contiguous"):
        call(offsets=torch.zeros(6, dtype=torch.int64, device="cuda")[::2])
    with pytest.raises(Exception, match="positive"):
        call(n=0)
    with pytest.raises(Exception, match="bfloat16"):
        call(output=case.output.float())
    with pytest.raises(Exception, match="input K does not match"):
        call(
            a=torch.zeros(
                case.rows,
                case.hidden // 2,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
        )
    with pytest.raises(Exception, match="weight must be"):
        call(b=case.b[:, : case.intermediate].contiguous())
    with pytest.raises(Exception, match="float32"):
        call(scales_a=case.scales_a.double())
    with pytest.raises(Exception, match="scales_a"):
        call(scales_a=case.scales_a[: case.scales_a.numel() // 2])
    with pytest.raises(Exception, match="require scales_a"):
        call(scales_a=None)
    with pytest.raises(Exception, match="input rows"):
        call(a=case.a[: case.rows // 2])
    with pytest.raises(Exception, match="rows"):
        call(output=case.output[: case.rows // 2])
    with pytest.raises(Exception, match="N exceeds the workspace query"):
        call(
            output=torch.zeros(
                case.rows,
                3 * case.intermediate,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            b=torch.zeros(
                case.groups,
                3 * case.intermediate,
                case.hidden,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            n=3 * case.intermediate,
        )
    with pytest.raises(Exception, match="K exceeds the workspace query"):
        call(
            a=torch.zeros(
                case.rows,
                case.hidden + 128,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            b=torch.zeros(
                case.groups,
                2 * case.intermediate,
                case.hidden + 128,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            k=case.hidden + 128,
        )
    with pytest.raises(Exception, match="runtime group count"):
        call(
            b=torch.zeros(
                case.groups + 1,
                2 * case.intermediate,
                case.hidden,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            offsets=torch.zeros(
                case.groups + 2,
                dtype=torch.int64,
                device="cuda",
            ),
        )


@requires_sm90
def test_fc1_fused_gemm_rejects_invalid_ffi_arguments() -> None:
    case = _fp8_case()

    def call(
        *,
        output=None,
        output_scales=None,
        a=None,
        offsets=None,
        n=None,
        scales_a=None,
    ) -> None:
        case.runner.moe_gemm_fc1_fused(
            case.fused_output if output is None else output,
            case.fused_scales if output_scales is None else output_scales,
            case.a if a is None else a,
            case.b,
            case.offsets if offsets is None else offsets,
            2 * case.intermediate if n is None else n,
            case.hidden,
            case.scales_a if scales_a is None else scales_a,
            case.scales_b,
            False,
        )

    call()
    torch.cuda.synchronize()
    with pytest.raises(Exception, match="int64"):
        call(offsets=case.offsets.to(torch.int32))
    with pytest.raises(Exception, match="contiguous"):
        call(offsets=torch.zeros(6, dtype=torch.int64, device="cuda")[::2])
    with pytest.raises(Exception, match="positive"):
        call(n=0)
    with pytest.raises(Exception, match="divisible by 256"):
        call(n=2 * case.intermediate + 128)
    with pytest.raises(Exception, match="float32"):
        call(scales_a=case.scales_a.double())
    with pytest.raises(Exception, match="scales_a"):
        call(scales_a=case.scales_a[: case.scales_a.numel() // 2])
    with pytest.raises(Exception, match="output_scales"):
        call(output_scales=case.fused_scales[: case.fused_scales.numel() // 2])
    with pytest.raises(Exception, match="input rows"):
        call(a=case.a[: case.rows // 2])
    with pytest.raises(Exception, match="rows"):
        call(output=case.fused_output[: case.rows // 2])


@requires_sm90
def test_moe_gemm_workspace_contract() -> None:
    runner = _create_runner()
    with pytest.raises(Exception, match="get_moe_workspace_size"):
        runner.configure_workspace(torch.empty(1, dtype=torch.uint8, device="cuda"))
    for arguments in (
        (0, 256, 256, 1, 2),
        (64, 0, 256, 1, 2),
        (64, 256, 0, 1, 2),
        (64, 256, 256, 0, 2),
        (64, 256, 256, 1, 0),
    ):
        with pytest.raises(Exception, match="must be positive"):
            runner.get_moe_workspace_size(*arguments, False, False)

    with pytest.raises(Exception, match="accepts FP8 A and FP8 B"):
        runner.get_moe_workspace_size(64, 64, 256, 256, 2, False, False)

    size = int(runner.get_moe_workspace_size(64, 64, 256, 256, 2, True, True))
    assert size == 0
    with pytest.raises(Exception, match="uint8"):
        runner.configure_workspace(
            torch.empty(size, dtype=torch.float32, device="cuda")
        )
    with pytest.raises(Exception, match="must be a CUDA tensor"):
        runner.configure_workspace(torch.empty(size, dtype=torch.uint8))
    runner.configure_workspace(torch.empty(1, dtype=torch.uint8, device="cuda"))


def _assert_device_trap(result: subprocess.CompletedProcess[str], marker: str) -> None:
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined[-1500:]
    assert "UNEXPECTED-SURVIVAL" not in result.stdout, combined[-1500:]
    assert marker in combined, combined[-1500:]
    for bad_marker in ("ImportError", "ModuleNotFoundError"):
        assert bad_marker not in combined, combined[-1500:]


@requires_sm90
@pytest.mark.parametrize(
    "fused,offset_values,marker",
    [
        (False, [0, 48, 16], "sm90_push_fp8_moe: bad offsets"),
        (False, [1, 32, 64], "sm90_push_fp8_moe: bad offsets"),
        (True, [0, 48, 16], "bad offsets"),
        (True, [1, 32, 64], "bad offsets"),
    ],
)
def test_moe_gemm_invalid_offsets_trap(
    fused: bool,
    offset_values: list[int],
    marker: str,
) -> None:
    invocation = (
        "runner.moe_gemm_fc1_fused(d, sfd, a, b, offsets, 2 * i, h, sfa, sfb, False)"
        if fused
        else "runner.moe_gemm(d, a, b, offsets, 2 * i, h, sfa, sfb, False)"
    )
    output_setup = (
        "d = torch.zeros(m, i, dtype=torch.uint8, device=device)\n"
        "sfd = torch.zeros((i // 128) * p + 128, dtype=torch.float32, device=device)"
        if fused
        else "d = torch.zeros(m, 2 * i, dtype=torch.bfloat16, device=device)"
    )
    code = f"""
import torch
from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim.gemm import create_sm90_push_fp8_moe_gemm_runner
device = torch.device("cuda", 0)
groups, m, h, i = 2, 64, 256, 256
runner = create_sm90_push_fp8_moe_gemm_runner()
size = runner.get_moe_workspace_size(m, m, max(2 * i, h), max(h, i), groups, True, True)
runner.configure_workspace(torch.empty(max(int(size), 1), device=device, dtype=torch.uint8))
p = (m + groups * 31) // 32 * 32
a = torch.zeros(m, h, dtype=torch.float8_e4m3fn, device=device)
b = torch.zeros(groups, 2 * i, h, dtype=torch.float8_e4m3fn, device=device)
sfa = torch.zeros((h // 128) * p + 128, dtype=torch.float32, device=device)
sfb = torch.zeros(groups, 2 * i // 128, h // 128, dtype=torch.float32, device=device)
{output_setup}
offsets = torch.tensor({offset_values!r}, dtype=torch.int64, device=device)
{invocation}
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    _assert_device_trap(result, marker)


@requires_sm90
@pytest.mark.parametrize("fused", [False, True])
def test_moe_gemm_offsets_exceed_input_capacity_trap(fused: bool) -> None:
    if fused:
        flags = "True, True"
        tensors = """
a = torch.zeros(m, h, dtype=torch.float8_e4m3fn, device=device)
b = torch.zeros(groups, 2 * i, h, dtype=torch.float8_e4m3fn, device=device)
p = (m + groups * 31) // 32 * 32
sfa = torch.zeros((h // 128) * p + 128, dtype=torch.float32, device=device)
sfb = torch.zeros(groups, 2 * i // 128, h // 128, dtype=torch.float32, device=device)
d = torch.zeros(2 * m, i, dtype=torch.uint8, device=device)
sfd = torch.zeros((i // 128) * p + 128, dtype=torch.float32, device=device)
offsets = torch.tensor([0, m, 2 * m], dtype=torch.int64, device=device)
runner.moe_gemm_fc1_fused(d, sfd, a, b, offsets, 2 * i, h, sfa, sfb, False)
"""
    else:
        flags = "True, True"
        tensors = """
a = torch.zeros(m, h, dtype=torch.float8_e4m3fn, device=device)
b = torch.zeros(groups, 2 * i, h, dtype=torch.float8_e4m3fn, device=device)
p = (m + groups * 31) // 32 * 32
sfa = torch.zeros((h // 128) * p + 128, dtype=torch.float32, device=device)
sfb = torch.zeros(groups, 2 * i // 128, h // 128, dtype=torch.float32, device=device)
d = torch.zeros(2 * m, 2 * i, dtype=torch.bfloat16, device=device)
offsets = torch.tensor([0, m, 2 * m], dtype=torch.int64, device=device)
runner.moe_gemm(d, a, b, offsets, 2 * i, h, sfa, sfb, False)
"""
    code = f"""
import torch
from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim.gemm import create_sm90_push_fp8_moe_gemm_runner
device = torch.device("cuda", 0)
groups, m, h, i = 2, 64, 256, 256
runner = create_sm90_push_fp8_moe_gemm_runner()
size = runner.get_moe_workspace_size(m, m, max(2 * i, h), max(h, i), groups, {flags})
runner.configure_workspace(torch.empty(max(int(size), 1), device=device, dtype=torch.uint8))
{tensors}
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    _assert_device_trap(result, "bad offsets")
