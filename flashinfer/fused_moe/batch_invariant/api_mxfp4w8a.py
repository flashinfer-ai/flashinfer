# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The mixed MXFP4-weight x MXFP8-activation MoE classes ``moe.mxfp4w8a.<act>``: the expert weights quantized by the
``mxfp4.quant`` class (e2m1 values, two per byte, one UE8M0 scale per 32 elements; :class:`.api_fp4.MoEWeightsFp4` of the
pair :data:`.api_fp4.MXFP4W8A`), the activations by the ``mxfp8.quant`` class (the canonical MXFP8 pack of :mod:`.api_mxfp8`),
the two GEMMs as grouped placements of the dense mixed class of the part (both operands' block scales inside the mixed-format
block-scaled step), the bf16 classes' chains, routing, deterministic permutation and slot-order combine. The quant pair
TRTLLM-Gen serves as ``TrtllmFp4Config`` with ``QuantConfig(weight=MXFP4, activation=MXFP8)``.

The op is the MX op (:mod:`.api_mxfp8`: the padded and the dense packings, the fused requantization of the intermediate, the
unified grouped and swapped bodies on the block-scaled operand block) at a mixed instruction class: what the pair adds is the
byte-packed e2m1 weight (the swapped body on the twin plan with the operand formats exchanged), the class's K granule (32 on
the SM107 class, 128 on the SM100 / SM103 class, whose TMA unpacks the e2m1 weight in 128-element runs), the swapped
records' own stage width and its dispatcher (:mod:`.schedule_mxfp4w8a`); the gather and decode forms are the one-format
classes'. The specification is :mod:`.spec_mxfp4w8a`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch

from ...batch_invariant.support import compute_capability
from ...gemm.batch_invariant.generator import Knobs
from ...gemm.batch_invariant.heuristic import family_of
from ...gemm.batch_invariant.schedule import mx_class_for
from .api_mxfp8 import DENSE_TAIL, GROUP_ALIGN, PACKINGS, _part
from .api_mxfp8 import fused_moe_mxfp8, grouped_gemm_mxfp8
from .blocks.epilogue import GroupedEpilogueBlock
from .spec_mxfp4w8a import BLOCK, CLASS, MX_CLASSES, granule_of

__all__ = [
    "BLOCK",
    "CLASS",
    "DENSE_TAIL",
    "GROUP_ALIGN",
    "MX_CLASSES",
    "PACKINGS",
    "expert_classes_mixed",
    "fused_moe_mxfp4w8a",
    "granule_of",
    "grouped_gemm_mxfp4w8a",
    "mx_class_of",
    "_part",
]


def mx_class_of(device: torch.device) -> str:
    """The instruction class a device runs: a function of the architecture family only (``mxfp4w8a_k64`` on SM107,
    ``mxfp4w8a`` on SM100 / SM103), never of the shape."""
    return mx_class_for("mxfp4w8a", family_of(compute_capability(device)))


def expert_classes_mixed(family: Optional[str]) -> Tuple[str, str]:
    """The mixed instruction class of both expert weights: the part's (``mx_class_for``), the same for GEMM1 and GEMM2."""
    c = mx_class_for("mxfp4w8a", family)
    return c, c


def grouped_gemm_mxfp4w8a(
    a: torch.Tensor,
    a_sf: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: GroupedEpilogueBlock,
    in_fmt: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
    dense_a: bool = False,
    sf_rows: Optional[int] = None,
    n_live: Optional[int] = None,
) -> Any:
    """:func:`.api_mxfp8.grouped_gemm_mxfp8` at a mixed instruction class (the device's by default): ``w [E, N, K / 2]`` the
    byte-packed e2m1 weight (uint8), ``a`` the e4m3 rows, both operands' 128x4 UE8M0 scales; the record selects the body
    (a dense record the grouped body on either packing, a swapped record the swapped body on the dense packing)."""
    return grouped_gemm_mxfp8(
        a,
        a_sf,
        w,
        w_sf,
        counts,
        config,
        chain,
        in_fmt or mx_class_of(a.device),
        out,
        dense_a,
        sf_rows,
        n_live=n_live,
    )


def fused_moe_mxfp4w8a(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: Any,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    in_fmt: Optional[str] = None,
    return_intermediates: bool = False,
    requant: str = "fused",
    plan: Optional[Tuple[torch.Tensor, ...]] = None,
    swap: Optional[bool] = None,
    path: Optional[str] = None,
    w_round: bool = False,
) -> Any:
    """The mixed MXFP4 x MXFP8 MoE FFN for pre-routed tokens: :func:`.api_mxfp8.fused_moe_mxfp8` on the mixed pair's weights
    (``w`` a :class:`.api_fp4.MoEWeightsFp4` of :data:`.api_fp4.MXFP4W8A`) at a mixed instruction class (``in_fmt``: the
    device's by default). ``x`` a bf16 ``[T, H]`` (quantized here per token by the ``mxfp8.quant`` class) or the quantized
    pack ``(xq [T, H] e4m3, xsf [T, H / 32] uint8 / UE8M0)`` taken as given; the records, the packing (``path``), the plan,
    the requantization spelling and ``w_round`` as for the MXFP8 op; ``swap``: the dispatcher's record spaces when the records
    are not given (None ranks both, True the swapped decode records only, False the grouped records only; placements)."""
    from .api_fp4 import MXFP4W8A

    if getattr(w, "pair", None) is not MXFP4W8A:
        raise TypeError(
            "fused_moe_mxfp4w8a takes the MXFP4 x MXFP8 pair's weights (MoEWeightsFp4.prepare(MXFP4W8A, ...))"
        )
    dev = x.device if isinstance(x, torch.Tensor) else x[0].device
    in_fmt = in_fmt or mx_class_of(dev)
    if swap is not None and (config1 is None or config2 is None):
        from .api_mxfp8 import _part
        from .schedule_mxfp4w8a import default_configs_mx

        T = x.shape[0] if isinstance(x, torch.Tensor) else x[0].shape[0]
        k = token_selected_experts.shape[1]
        sm_count, family = _part(dev)
        c1, c2 = default_configs_mx(
            T, k, w.hidden_size, w.intermediate_size, w.num_experts, w.chain1_quant,
            in_fmt, sm_count, family, swap=False if (requant != "fused" or path == "padded") else swap,
        )  # fmt: skip
        config1, config2 = config1 or c1, config2 or c2
    return fused_moe_mxfp8(
        x,
        token_selected_experts,
        token_final_scales,
        w,
        out=out,
        config1=config1,
        config2=config2,
        in_fmt=in_fmt,
        return_intermediates=return_intermediates,
        requant=requant,
        plan=plan,
        path=path,
        w_round=w_round,
    )
