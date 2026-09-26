"""Experimental FP8 E4M3 × FP4 E2M1 GEMM using prepacked UE8M0 scales."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_fp8_fp4_gemm(
    a,
    b,
    a_scales,
    b_scales,
    *,
    m,
    out=None,
    block_n=128,
    gran_k_a=32,
    variant=None,
    descriptor_workspace=None,
):
    """Prepare A @ B.T with FP32 accumulation and BF16 output.

    A is [storage_M,K] E4M3 FP8 or raw uint8. B is [N,K/2] packed E2M1,
    with even K in the low nibble. Scale tensors store four UE8M0 bytes per
    word, packed-K major with aligned-MN contiguous. Model routes use per-32
    A/B scales. Explicit variants require gran_k_a=128: BK128 uses native
    packing; BK256 repeats each A scale byte four times before packing.
    Preparation owns allocations; run() submits on the current stream.
    """
    from .experimental.deepgemm_mixed_gemm.mixed_gemm import MixedGemmPlan

    return MixedGemmPlan(
        a,
        b,
        a_scales,
        b_scales,
        m=m,
        out=out,
        block_n=block_n,
        gran_k_a=gran_k_a,
        variant=variant,
        descriptor_workspace=descriptor_workspace,
    )
