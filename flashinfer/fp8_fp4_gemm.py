"""Experimental FP8 E4M3 x FP4 E2M1 GEMM using prepacked UE8M0 scales."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_fp8_fp4_gemm(a, b, a_scales, b_scales, *, m=None, out=None, gran_k_a=32):
    """Prepare ``out[:m] = A @ B.T`` with FP32 accumulation and BF16 output.

    A is ``[M, K]`` E4M3 FP8 or raw uint8; B is ``[N, K/2]`` packed E2M1 with the
    even K element in the low nibble. Scales are prepacked UE8M0 words: four
    bytes per int32/uint32 word, packed-K major with the 4-aligned MN extent
    contiguous (``[words, aligned_MN]``). ``gran_k_a`` is the A scale
    granularity (32 or 128 K elements per byte); B scales are per 32.
    Any ``m`` in ``[1, rows of A]`` is accepted; K must be a multiple of 128.
    Preparation owns allocations and route selection; ``run()`` submits one
    kernel on the current stream without allocating.
    """
    from .experimental.deepgemm_mixed_gemm.mixed_gemm import MixedGemmPlan

    return MixedGemmPlan(a, b, a_scales, b_scales, m=m, out=out, gran_k_a=gran_k_a)
