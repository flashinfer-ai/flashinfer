"""Experimental per-head FP8 projection with fused BF16 or dynamic FP8 output."""
from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_fp8_batched_gemm(a, b, *, output_fp8=True, alpha=None, out=None,
                             output_scales=None, descriptor_workspace=None):
    """Prepare A[T,H,K] @ B[H,N,K] -> [T,H,N] once, then call plan.run().

    Pass a=(E4M3 values, FP32 scales[T,H,K/128]) and
    b=(E4M3 values, FP32 scales[H,N/128,K/128]). Scales are positive powers
    of two. Preparation packs them; prepare again when their values change.
    BF16 output accepts an optional alpha scalar. Dynamic FP8 output returns
    (E4M3 values, int32 packed per-32 UE8M0 scale words[T,H*N/128]); scale
    storage is column-major with T padded to4. Alpha and FP8 do not compose.
    run() submits the prepared projection on the current PyTorch stream,
    including descriptor updates and launch overhead, without allocation.
    """
    from .experimental.deepgemm_batched_gemm.batched_gemm import BatchedGemmPlan
    return BatchedGemmPlan(a, b, output_fp8=output_fp8, alpha=alpha, out=out,
        output_scales=output_scales, descriptor_workspace=descriptor_workspace)
