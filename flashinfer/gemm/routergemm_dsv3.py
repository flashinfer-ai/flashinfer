from ..api_logging import flashinfer_api
from flashinfer.jit import gen_dsv3_router_gemm_module
import functools
from types import SimpleNamespace
import torch
from flashinfer.utils import (
    register_custom_op,
    supported_compute_capability,
    backend_requirement,
)


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([100])
def _mm_M1_16_K7168_N256_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    # Dimension checks
    if mat_a.dim() != 2:
        raise ValueError("mat_a must be a 2D tensor")
    if mat_b.dim() != 2:
        raise ValueError("mat_b must be a 2D tensor")
    if out.dim() != 2:
        raise ValueError("out must be a 2D tensor")

    # Stride checks (check these before dimension checks to give better error messages)
    if mat_a.stride(1) != 1:
        raise ValueError("mat_a must be row-major")
    if out.stride(1) != 1:
        raise ValueError("out must be row-major")
    if mat_b.stride(0) != 1:
        raise ValueError("mat_b must be column-major")

    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError("mat_a.shape[1] must be equal to mat_b.shape[0]")
    if out.shape[0] != mat_a.shape[0]:
        raise ValueError("out.shape[0] must be equal to mat_a.shape[0]")
    if out.shape[1] != mat_b.shape[1]:
        raise ValueError("out.shape[1] must be equal to mat_b.shape[1]")

    # Problem size checks
    expected_hidden_dim = 7168
    expected_num_experts = 256
    min_tokens = 1
    max_tokens = 16
    if mat_a.shape[0] < min_tokens or mat_a.shape[0] > max_tokens:
        raise ValueError(
            f"mat_a.shape[0] (num_tokens) must be between {min_tokens} and {max_tokens}"
        )
    if mat_a.shape[1] != expected_hidden_dim:
        raise ValueError(
            f"mat_a.shape[1] (hidden_dim) must be equal to {expected_hidden_dim}"
        )
    if mat_b.shape[1] != expected_num_experts:
        raise ValueError(
            f"mat_b.shape[1] (num_experts) must be equal to {expected_num_experts}"
        )

    # Data type checks
    if mat_a.dtype != torch.bfloat16:
        raise ValueError("mat_a must be a bfloat16 tensor")
    if mat_b.dtype != torch.bfloat16:
        raise ValueError("mat_b must be a bfloat16 tensor")
    if out.dtype != torch.float32:
        raise ValueError("out must be a float32 tensor")

    return True


@functools.cache
def get_dsv3_router_gemm_module():
    module = gen_dsv3_router_gemm_module().build_and_load()

    @register_custom_op(
        "flashinfer::dsv3_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N256(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = False,
    ) -> None:
        module.dsv3_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    return SimpleNamespace(
        mm_M1_16_K7168_N256=mm_M1_16_K7168_N256,
    )


@flashinfer_api
@backend_requirement({}, common_check=_mm_M1_16_K7168_N256_shape_checks)
def mm_M1_16_K7168_N256(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = False,
) -> None:
    """Optimized GEMM for the router operation in DeepSeek-V3.

    This function performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in DeepSeek-V3's Mixture of Experts (MoE) architecture.
    It computes out = mat_a @ mat_b where mat_a contains token embeddings and mat_b
    contains expert routing weights.

    The implementation is optimized for the specific problem dimensions used in DeepSeek-V3:
    - Hidden dimension (K): 7168
    - Number of experts (N): 256
    - Number of tokens (M): 1-16

    Args:
        mat_a (torch.Tensor): Input token embeddings of shape (M, K) where M is the number
            of tokens (1-16) and K is the hidden dimension (7168). Must be bfloat16,
            row-major (contiguous).
        mat_b (torch.Tensor): Expert routing weights of shape (K, N) where K is the hidden
            dimension (7168) and N is the number of experts (256). Must be bfloat16,
            column-major (transposed layout).
        out (torch.Tensor): Pre-allocated output tensor of shape (M, N) containing the
            routing scores. Must be float32, row-major (contiguous). This tensor is
            mutated in-place.
        launch_with_pdl (bool, optional): Whether to launch the kernel using Persistent
            Device-side Launch. Defaults to False.

    Returns:
        None: The result is written directly to the `out` tensor.

    Raises:
        ValueError: If tensor dimensions, strides, or data types do not match the
            expected DeepSeek-V3 router configuration.

    Note:
        This kernel is specialized for compute capability 10.0 (Blackwell architecture).
        The specific problem size optimization makes this significantly faster than
        general-purpose GEMM implementations for the router operation.
    """
    get_dsv3_router_gemm_module().mm_M1_16_K7168_N256(
        mat_a, mat_b, out, launch_with_pdl
    )
