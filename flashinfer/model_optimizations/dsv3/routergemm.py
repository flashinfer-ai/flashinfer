from flashinfer.jit import gen_dsv3_router_gemm_module
import functools
from types import SimpleNamespace
import torch
from flashinfer.utils import register_custom_op
from typing import Optional


@functools.cache
def get_dsv3_router_gemm_module():
    module = gen_dsv3_router_gemm_module().build_and_load()

    @register_custom_op(
        "flashinfer::dsv3_router_gemm_op",
        mutates_args=[
            "mat_a",
            "mat_b",
            "out",
            "launch_with_pdl",
            "bias",
        ],
    )
    def dsv3_router_gemm_op(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool,
        bias: Optional[torch.Tensor],
    ) -> None:
        module.dsv3_router_gemm_op(mat_a, mat_b, out, launch_with_pdl, bias)

    return SimpleNamespace(
        dsv3_router_gemm_op=dsv3_router_gemm_op,
    )


# TODO: Add decorator for support checks: compute capability and type checks
def dsv3_router_gemm_op(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool,
    bias: Optional[torch.Tensor],
) -> None:
    get_dsv3_router_gemm_module().dsv3_router_gemm_op(
        mat_a, mat_b, out, launch_with_pdl, bias
    )
