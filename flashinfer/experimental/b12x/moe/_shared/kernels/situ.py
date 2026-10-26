"""Kimi K3 SiTU wrappers for the activation-specialized fused MoE backends.

The shared backend implements ``beta * tanh(gate / beta) * sigmoid(gate)``
times ``linear_beta * tanh(up / linear_beta)`` with K3's beta values.
"""

from __future__ import annotations

from typing import Tuple

from b12x.moe._shared.kernels.activations import SITU
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from b12x.moe._shared.kernels.micro import MoEMicroKernelBackend


class MoEMicroKernelSitu(MoEMicroKernelBackend):
    """SiTU micro specialization, trellis-only.

    The compact NVFP4 intermediate quantization has not passed its
    correctness gate, so ``is_supported`` stays False and the nvfp4-family
    dispatch never selects this class. The trellis_t256 arm quantizes its
    intermediate through per-32 UE8M0 E4M3 (a8_mx) instead; the w4a8_mx
    trellis band dispatches it explicitly and constructs this class with
    ``weight_layout="trellis_t256"``.
    """

    @classmethod
    def is_supported(
        cls,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
    ) -> bool:
        del m, k, n, num_topk, weight_E
        return False

    def __init__(self, *args: object, **kwargs: object):
        if kwargs.get("weight_layout") != "trellis_t256":
            raise NotImplementedError(
                "SiTU compact micro serves only the trellis_t256 weight "
                "layout; select the fused dynamic kernel"
            )
        kwargs["activation"] = SITU
        super().__init__(*args, **kwargs)


class MoEDynamicKernelSitu(MoEDynamicKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        fast_math: bool = False,
        dynamic_down_scale: bool = False,
        share_input_across_experts: bool = False,
        deterministic_output: bool = False,
        num_topk: int = 1,
        swap_ab: bool = False,
        separate_w13_halves: bool = False,
        quant_recipe: str = "nvfp4",
        w4a8_repacked: bool = False,
        trellis_bits: int | None = None,
        trellis_intermediate_hadamard: bool = False,
        direct_routing: bool = False,
        work_source: str = "materialized_queue",
        materialize_intermediate: bool = False,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            fast_math=fast_math,
            activation=SITU,
            dynamic_down_scale=dynamic_down_scale,
            share_input_across_experts=share_input_across_experts,
            deterministic_output=deterministic_output,
            num_topk=num_topk,
            swap_ab=swap_ab,
            separate_w13_halves=separate_w13_halves,
            quant_recipe=quant_recipe,
            w4a8_repacked=w4a8_repacked,
            trellis_bits=trellis_bits,
            trellis_intermediate_hadamard=trellis_intermediate_hadamard,
            direct_routing=direct_routing,
            work_source=work_source,
            materialize_intermediate=materialize_intermediate,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )


__all__ = ["MoEDynamicKernelSitu", "MoEMicroKernelSitu"]
