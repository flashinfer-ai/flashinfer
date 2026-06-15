"""Reference correctness test for the mm_w4a16_fp4 trace API."""

import pytest
import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


@pytest.mark.parametrize("backend", ["cudnn", "cute-dsl"])
@pytest.mark.parametrize(
    "shape_kwargs", [dict(M=32, N=1024, K=1024), dict(M=16, N=2048, K=512)]
)
def test_mm_w4a16_fp4_reference_correctness(backend, shape_kwargs):
    """flashinfer.mm_fp4 W4A16 mode kernel vs reference (dequant + matmul).

    The trace inits build *prepared* (backend-specific) weights via
    ``prepare_w4a16_fp4_weights``; each backend's reference dequantizes
    that prepared layout directly (the cute-dsl one inverts the MMA tile
    permutation and decodes S0E5M3 scales).
    """
    import flashinfer
    from flashinfer.trace.templates.gemm import (
        mm_w4a16_fp4_cudnn_trace,
        mm_w4a16_fp4_cute_dsl_trace,
    )

    tpl = {
        "cudnn": mm_w4a16_fp4_cudnn_trace,
        "cute-dsl": mm_w4a16_fp4_cute_dsl_trace,
    }[backend]
    try:
        inputs = tpl.init(**shape_kwargs)
        api = flashinfer.mm_fp4(
            inputs["a"],
            inputs["b"],
            inputs["a_descale"],
            inputs["b_descale"],
            inputs["alpha"],
            backend=backend,
            block_size=inputs["block_size"],
        )
    except Exception as exc:
        pytest.skip(f"W4A16 mm_fp4 ({backend}) unavailable: {exc}")
    _assert_finite(inputs["a"])
    ref = tpl.reference(
        inputs["a"],
        inputs["b"],
        inputs["b_descale"],
        inputs["alpha"],
        block_size=inputs["block_size"],
    )
    _assert_finite(api, ref)
    _check(
        tpl,
        ref.to(api.dtype),
        api,
        max_mismatch_pct=100.0,
        min_cos_sim=0.99,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
