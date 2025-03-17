import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

@triton.jit
def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret

def matmul_tma_persistent_get_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8, "EPILOGUE_SUBTILE" : SUBTILE}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BN in [128, 256] \
        for BK in [64, 128] \
        for s in ([3, 4]) \
        for w in [4, 8] \
        for SUBTILE in [True, False] \
    ]

@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def batched_matmul_kernel(
    a_ptr, b_ptr, d_ptr, a_scale_ptr, b_scale_ptr, d_scale_ptr, batch_size, M, N, K,
    stride_a_batch, stride_b_batch, stride_d_batch,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr
):
    pass # TODO(yingyi): update triton to support bmm fp8 with scale