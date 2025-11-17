from flashinfer.jit import gen_dsv3_fused_routing_module
import functools
from types import SimpleNamespace
from flashinfer.utils import (
    register_custom_op,
    # supported_compute_capability,
    # backend_requirement,
)


@functools.cache
def get_dsv3_fused_routing_module():
    module = gen_dsv3_fused_routing_module().build_and_load()

    @register_custom_op(
        "flashinfer::NoAuxTc",
        mutates_args=["topk_values", "topk_indices"],
    )
    def NoAuxTc(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values,
        topk_indices,
        launch_with_pdl=True,
    ):
        module.NoAuxTc(
            scores,
            bias,
            n_group,
            topk_group,
            topk,
            routed_scaling_factor,
            topk_values,
            topk_indices,
            launch_with_pdl,
        )

    return SimpleNamespace(
        NoAuxTc=NoAuxTc,
    )


def NoAuxTc(
    scores,
    bias,
    n_group,
    topk_group,
    topk,
    routed_scaling_factor,
    topk_values,
    topk_indices,
    launch_with_pdl=True,
):
    get_dsv3_fused_routing_module().NoAuxTc(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values,
        topk_indices,
        launch_with_pdl,
    )
