"""Experimental native FP4 GEMM using prepacked E2M1 operands and UE8M0 scales."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_fp4_gemm(
    a,
    b,
    a_scales,
    b_scales,
    *,
    m,
    alpha=1.0,
    out=None,
    num_stages=None,
    block_n=128,
    epilogue_store_n=32,
    descriptor_workspace=None,
):
    """Prepare A @ B.T, scaling FP32 accumulators by alpha before BF16 output.

    Packed A/B are [storage_M,K/2]/[N,K/2]; scales are uint32[K/128,storage_MN].
    Four granularity-32 UE8M0 bytes occupy each word. m is logical output M;
    required physical rows follow the exported production route. No data
    conversion occurs during run(). See Fp4GemmPlan for the full tensor ABI.
    """
    from .experimental.deepgemm_fp4_gemm.fp4_gemm import Fp4GemmPlan

    return Fp4GemmPlan(
        a,
        b,
        a_scales,
        b_scales,
        m=m,
        alpha=alpha,
        out=out,
        num_stages=num_stages,
        block_n=block_n,
        epilogue_store_n=epilogue_store_n,
        descriptor_workspace=descriptor_workspace,
    )
