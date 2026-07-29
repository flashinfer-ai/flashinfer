"""Arch/tactic guards of the trtllm-gen GEMM runner (#4107).

Two layers of defense are exercised:

1. On a GPU without compatible cubins the runner constructor raises an
   explicit architecture error instead of dumping the GEMM option list.
2. On a supported GPU, a tactic that is a valid manifest index ("in range")
   but not in the runner's passing config set is rejected before any kernel
   work. ``trtllm_gemm`` reaches ``getWorkspaceSizeInBytes()`` first, so that
   is the guard observed here; ``run()`` performs the same membership check
   at entry for callers that reach it directly.
"""

import pytest
import torch

from flashinfer.tllm_enums import DtypeTrtllmGen

M, N, K = 128, 256, 512


def _arch():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _load_op():
    from flashinfer.jit.gemm.core import gen_trtllm_gen_gemm_module

    return gen_trtllm_gen_gemm_module().build_and_load()


def test_trtllm_gemm_arch_error_on_unsupported_sm():
    """Constructing any trtllm-gen GEMM runner on an unsupported GPU must
    raise the explicit arch error, not the GEMM option dump."""
    arch = _arch()
    if arch in (100, 103):
        pytest.skip("needs a GPU without trtllm-gen cubins (e.g. sm90/sm120)")

    op = _load_op()
    with pytest.raises(Exception, match="contains no kernels runnable on sm"):
        op.trtllm_gemm_tactics(
            M, N, K, int(DtypeTrtllmGen.E4m3), int(DtypeTrtllmGen.Bfloat16), True
        )


def test_trtllm_gemm_rejects_in_range_tactic_outside_config_set():
    """A manifest index that belongs to a different runner configuration must
    be rejected by the tactic-membership guard, not silently dispatched."""
    arch = _arch()
    # Explicit device-arch gate: skip only on hardware that is not sm100/103.
    # On sm100/103 this test must run and must not skip.
    if arch not in (100, 103):
        pytest.skip(f"requires an sm100/sm103 GPU, detected sm{arch}")

    op = _load_op()
    fp8_tactics = set(
        op.trtllm_gemm_tactics(
            M, N, K, int(DtypeTrtllmGen.E4m3), int(DtypeTrtllmGen.Bfloat16), False
        )
    )
    fp4_tactics = list(
        op.trtllm_gemm_tactics(
            M, N, K, int(DtypeTrtllmGen.E2m1), int(DtypeTrtllmGen.Bfloat16), True
        )
    )
    foreign = [t for t in fp4_tactics if t not in fp8_tactics]
    # On supported hardware this test must not skip: the E2m1 and E4m3 cubin
    # families are disjoint in the manifest, so a foreign tactic must exist.
    assert foreign, (
        "expected at least one E2m1 tactic outside the E4m3 set; "
        f"E4m3={sorted(fp8_tactics)} E2m1={sorted(fp4_tactics)}"
    )

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    a_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    b_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(4 * 1024 * 1024, device="cuda", dtype=torch.int8)

    with pytest.raises(Exception, match="not in this runner's compatible config set"):
        op.trtllm_gemm(
            int(DtypeTrtllmGen.E4m3),
            int(DtypeTrtllmGen.Bfloat16),
            workspace,
            a,
            b,
            a_scale,
            b_scale,
            None,
            out,
            False,
            int(foreign[0]),
        )
