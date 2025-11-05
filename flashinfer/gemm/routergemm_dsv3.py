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
def _dsv3_router_gemm_shape_checks(mat_a, mat_b, out, launch_with_pdl):
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
    def routergemm_dsv3_hidden_7168_experts_256_tokens_lt16(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = False,
    ) -> None:
        module.dsv3_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    return SimpleNamespace(
        routergemm_dsv3_hidden_7168_experts_256_tokens_lt16=routergemm_dsv3_hidden_7168_experts_256_tokens_lt16,
    )


@backend_requirement({}, common_check=_dsv3_router_gemm_shape_checks)
def routergemm_dsv3_hidden_7168_experts_256_tokens_lt16(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = False,
) -> None:
    get_dsv3_router_gemm_module().routergemm_dsv3_hidden_7168_experts_256_tokens_lt16(
        mat_a, mat_b, out, launch_with_pdl
    )
