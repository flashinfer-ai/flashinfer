"""The dense CuTe-DSL GEMMs persist their kernels in the CuTe-DSL disk cache.

Each case compiles its kernel into an empty cache, drops the in-memory kernel, and runs again: the second run must load
the artifact without compiling, and both outputs must equal, bit for bit, the output of a kernel compiled in-process with
the disk cache disabled.
"""

import pytest
import torch

pytest.importorskip("cutlass")

from flashinfer.jit import cute_dsl_core  # noqa: E402
from flashinfer.jit import env as jit_env  # noqa: E402


def _require_sm100_family():
    from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("the dense CuTe-DSL GEMMs require SM100 or SM103")
    if not is_cute_dsl_experimental_available():
        pytest.skip("nvidia-cutlass-dsl with cutlass.experimental is not available")


def _gen(shape, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, generator=g, device="cuda") * scale).to(torch.bfloat16)


def _bf16_case(kind):
    from flashinfer.gemm.kernels import (
        dense_bf16_gemm_direct as direct,
        dense_bf16_gemm_sm100_splitk as sm100,
        dense_bf16_gemm_warp_splitk as warp,
    )

    m, n, k = 8, 1024, 2048
    a = _gen((m, k), 0)
    b = _gen((n, k), 1, 0.05).T
    bias = _gen((n,), 2)

    def out():
        return torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    if kind == "bf16_direct":
        tactic = direct.default_tactic(m, n, k)

        def run():
            direct.run_direct_dense(a, b, o := out(), True, tactic)
            return o

        return run, direct._get_compiled_direct_kernel.cache_clear
    if kind == "bf16_warp_splitk":
        tactic = warp.default_tactic(m, n, k)

        def run():
            warp.run_warp_splitk_dense(a, b, o := out(), True, tactic, bias)
            return o

        return run, warp._compile.cache_clear
    tactic = sm100.default_tactic(m, n, k)

    def run():
        sm100.run_splitk_dense(a, b, bias, o := out(), True, tactic)
        return o

    return run, sm100._get_compiled_splitk_kernel.cache_clear


def _mxfp8_case(kind):
    from flashinfer import mxfp8_quantize
    from flashinfer.gemm import gemm_base

    m, n, k = 4, 2048, 4096
    a_mx, a_sf = mxfp8_quantize(_gen((m, k), 3), True)
    b_mx, b_sf = mxfp8_quantize(_gen((n, k), 4, 0.05), True)
    workspace = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")

    def inputs():
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        return [a_mx, b_mx.t(), a_sf, b_sf, torch.bfloat16, out, workspace]

    major, minor = torch.cuda.get_device_capability()
    if kind == "mxfp8_warp_splitk":
        from flashinfer.gemm.kernels import dense_mxfp8_gemm_warp_splitk

        runner = gemm_base._cute_dsl_warp_splitk_mxfp8_gemm_runner(True)
        clear = dense_mxfp8_gemm_warp_splitk._compile.cache_clear
    else:
        runner = gemm_base._cutedsl_low_latency_blockscaled_gemm_runner(
            major * 10 + minor, True
        )
        clear = gemm_base._CUTEDSL_LOW_LATENCY_BLOCK_SCALED_KERNEL_CACHE.clear
    tactic = runner.get_valid_tactics(inputs(), None)[0]

    def run():
        x = inputs()
        runner.forward(x, tactic=tactic)
        return x[5]

    return run, clear


@pytest.mark.parametrize(
    "kind",
    [
        "bf16_direct",
        "bf16_warp_splitk",
        "bf16_sm100_splitk",
        "mxfp8_warp_splitk",
        "mxfp8_low_latency",
    ],
)
def test_dense_gemm_kernel_reloads_from_disk_bitwise(kind, tmp_path, monkeypatch):
    _require_sm100_family()
    # Operands first: their quantization kernels stay in the regular JIT cache.
    run, clear = (_bf16_case if kind.startswith("bf16") else _mxfp8_case)(kind)
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "cached_ops")

    def artifacts():
        return sorted(tmp_path.rglob("*_cute_dsl/*.o"))

    def fresh(disable_cache=False):
        clear()
        if disable_cache:
            monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
        else:
            monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
        out = run()
        torch.cuda.synchronize()
        return out.view(torch.uint8).clone()

    try:
        in_process = fresh(disable_cache=True)
        assert not artifacts()
        built = fresh()
        persisted = artifacts()
        assert persisted, "the kernel was not persisted"
        with monkeypatch.context() as m:
            m.setattr(
                cute_dsl_core.JitSpecCuteDsl,
                "build",
                lambda self: pytest.fail(f"{self.name} recompiled"),
            )
            loaded = fresh()
        assert artifacts() == persisted
    finally:
        clear()
    assert torch.equal(built, in_process)
    assert torch.equal(loaded, in_process)
