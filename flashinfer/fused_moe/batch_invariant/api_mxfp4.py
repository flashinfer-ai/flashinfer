# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP4 MoE classes ``moe.mxfp4.<act>`` (W4A4 with UE8M0 block scales on both operands): MXFP4 expert weights and
MXFP4 activations, one UE8M0 scale per 32 elements and no global or per-token scale, on the bf16 classes' routing,
permutation and combine (:mod:`.api`) and the NVFP4 classes' grouped block-scaled bodies (:mod:`.api_nvfp4`).

Pipeline: the MXFP4 quantization of ``x`` (the class ``mxfp4.quant`` of :mod:`flashinfer.batch_invariant.quantization`, or
the caller's quantized pair ``(x4, x_sf)``) -> the deterministic permutation of the expanded rows (expert order, then
token-slot order) with the scale factors placed at 128-row atoms -> the grouped block-scaled GEMM1 with the bf16 class's chain
on the raw accumulators (the block scales of both operands are applied inside the instruction: ``cvt_bf16(glu_pre(g, u))``) ->
the quantization of the materialised bf16 intermediate by the same class -> the grouped block-scaled GEMM2
(``cvt_bf16(acc)``) -> the slot-order combine. Every output bit is a function of the token's own row, its ``k`` experts and
routing weights and the expert weights (the specification is :mod:`.spec_mxfp4`; the Lean statements the ``MoEMxfp4`` module
of the mode).

The two packings of the expanded rows are the NVFP4 classes' (``path``: ``"dense"`` / ``"padded"``, placements of the same
bits); the quantizer and the scatter are two launches each on this pair (the quantizer stores the class's layout, the scatter
moves the codes and re-tiles the scale bytes to the grouped positions: the same bytes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from ...batch_invariant.quantization.layout import padded_sf_cols, sf_numel
from ...batch_invariant.quantization.mx_spec import MXFP4
from ...batch_invariant.support import unsupported
from ...gemm.batch_invariant.api import persistent_clusters
from ...gemm.batch_invariant.generator import Knobs
from ...gemm.batch_invariant.schedule import chunks_for
from ...gemm.batch_invariant.spec import MXFP4_FMTS
from . import spec as S
from .api import interleave_fc1, route  # noqa: F401  (route re-exported for the class's callers)
from .api_nvfp4 import (
    gather_form_ok,
    DENSE_TAIL,
    GROUP_ALIGN,
    MAX_TOP_K,
    PATHS,
    _part,
    padded_rows_bound,
    permute_plan_dual,
    route_plan_dual,
)
from .blocks.epilogue import MXFP4_GEMM2_CHAIN, Mxfp4Chain
from .schedule import launch_tile_bound
from .spec_mxfp4 import CLASS, MX_BLOCK, expert_classes_mx

#: 16-byte rows of the packed fp4 operands (the TMA alignment) and whole 32-element scale blocks.
ALIGN_MXFP4 = 32
_OUT = {"bf16": torch.bfloat16, "f32": torch.float32}


def check_shape_mxfp4(
    H: Optional[int], I: int, E: int, k: Optional[int] = None, gated: bool = True
) -> None:
    """The complete acceptance rule of the class on ``(H, I, E, k)``: ``H`` and ``I`` positive multiples of 32 (16-byte rows
    of the packed fp4 operands and whole scale blocks), ``1 <= E <= 1024``, ``1 <= k <= min(E, 16)``, and the expert weight
    shapes unsplit by the dense MX class's chunk rule. Everything else about a shape is placement. ``H=None`` checks the rules
    the layer's configuration fixes. Raises :class:`BatchInvariantUnsupportedError`; the class never routes a shape elsewhere."""
    from .blocks.scheduler import MAX_E

    def bad(what: str, limit: str):
        raise unsupported(
            CLASS,
            f"{what} ("
            + (f"H={H}, " if H is not None else "")
            + f"I={I}, E={E}"
            + (f", k={k}" if k is not None else "")
            + ")",
            limit,
        )

    if I <= 0 or I % ALIGN_MXFP4 or (H is not None and (H <= 0 or H % ALIGN_MXFP4)):
        bad(
            f"hidden size {H} / intermediate size {I}"
            if H is not None
            else f"intermediate size {I}",
            f"positive multiples of {ALIGN_MXFP4} (16-byte rows of the quantized activations and weights, whole scale blocks)",
        )
    if not 1 <= E <= MAX_E:
        bad(f"{E} experts", f"1 <= num_experts <= {MAX_E}")
    if k is not None and not 1 <= k <= min(E, MAX_TOP_K):
        bad(f"top_k {k}", f"1 <= top_k <= min(num_experts, {MAX_TOP_K})")
    if H is None:
        return
    n1 = 2 * I if gated else I
    if chunks_for(n1, H, "mxfp4") != 1 or chunks_for(H, I, "mxfp4") != 1:
        bad(
            "the expert weight shapes",
            "unsplit expert weights (the dense class would K-split this shape)",
        )


def mx_sf_numel(rows: int, K: int) -> int:
    """Bytes of the class's 128x4 swizzled UE8M0 scale buffer of ``rows`` rows of ``K`` elements."""
    return sf_numel(rows, K, MX_BLOCK, "128x4")


def mx_sf_cols(K: int) -> int:
    """Scale columns of a row of ``K`` elements in the 128x4 layout (``K / 32`` padded to whole 4-column groups)."""
    return padded_sf_cols(K, MX_BLOCK, "128x4")


@dataclass
class MoEWeightsMXFP4:
    """The MXFP4 expert weights as the kernels consume them: ``w1 [E, N1, H/2]`` packed fp4 (``N1 = 2I``, gate / up
    interleaved, for a gated chain; ``I`` as given for an ungated one), ``w1_sf [E, mx_sf_numel(N1, H)]`` 128x4 swizzled
    UE8M0 scales per expert, likewise ``w2 [E, H, I/2]``, ``w2_sf``; the GEMM1 chain ``act`` with its ``params``."""

    w1: torch.Tensor
    w1_sf: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    hidden_size: int
    intermediate_size: int
    gated: bool = True
    act: str = "silu"
    params: Tuple[Optional[float], ...] = ()
    #: the chain's arithmetic (``spec.ARITHS``): ``exact`` (`moe.mxfp4.<act>`) or ``hw`` (`moe.mxfp4.<act>.hw`: the
    #: hardware-arithmetic class; the chains without an exponential or a division stay the exact class)
    arith: str = "exact"

    @staticmethod
    def prepare(
        fc1: torch.Tensor,
        fc2: torch.Tensor,
        interleaved: bool = False,
        gated: bool = True,
        act: Optional[str] = None,
        params: Sequence[Optional[float]] = (),
        arith: str = "exact",
    ) -> "MoEWeightsMXFP4":
        """Quantize bf16 experts (``fc1 [E, 2I, H]`` rows ``[up, gate]``, or ``[E, I, H]`` ungated; ``fc2 [E, H, I]``) with the
        batch-invariant MXFP4 class (one UE8M0 scale per 32 elements of a row, TransformerEngine's recipe). The gate / up
        interleave is a row permutation applied before the quantization (a block scale is a function of its row). ``act``
        defaults to the class's default chain of the gating."""
        act = act if act is not None else ("silu" if gated else "relu2")
        if arith not in S.ARITHS:
            raise ValueError(f"arith must be one of {S.ARITHS}, not {arith!r}")
        if act not in (S.ACTS if gated else S.UNGATED_ACTS):
            raise ValueError(f"no {'gated' if gated else 'ungated'} chain {act!r}")
        key: Tuple[Optional[float], ...] = S.params_key(params)
        if len(key) != len(S.ACT_PARAMS.get(act, ())):
            raise ValueError(
                f"chain {act!r} takes parameters {S.ACT_PARAMS.get(act, ())}, got {key!r}"
            )
        if (
            fc1.dtype != torch.bfloat16
            or fc2.dtype != torch.bfloat16
            or fc1.dim() != 3
            or fc2.dim() != 3
        ):
            raise TypeError(
                "the MXFP4 MoE classes take bf16 expert weights fc1 [E, N1, H], fc2 [E, H, I]"
            )
        E, N1, H = fc1.shape
        I = N1 // 2 if gated else N1
        if tuple(fc2.shape) != (E, H, I) or (gated and N1 % 2):
            raise ValueError(
                f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)} (gated={gated})"
            )
        check_shape_mxfp4(H, I, E, gated=gated)
        w1i = fc1.contiguous() if (interleaved or not gated) else interleave_fc1(fc1)
        fc2 = fc2.contiguous()
        w1s, sf1s, w2s, sf2s = [], [], [], []
        for e in range(E):
            q1, sf1 = quantize_rows_mx(w1i[e])
            q2, sf2 = quantize_rows_mx(fc2[e])
            w1s.append(q1)
            sf1s.append(sf1)
            w2s.append(q2)
            sf2s.append(sf2)
        return MoEWeightsMXFP4(
            torch.stack(w1s).contiguous(),
            torch.stack(sf1s).contiguous(),
            torch.stack(w2s).contiguous(),
            torch.stack(sf2s).contiguous(),
            H,
            I,
            gated,
            act,
            key,
            arith,
        )

    @property
    def num_experts(self) -> int:
        return self.w1.shape[0]

    @property
    def n1(self) -> int:
        """GEMM1's weight rows: ``2I`` gated, ``I`` ungated."""
        return self.w1.shape[1]

    def classes(self, family: Optional[str]) -> Tuple[str, str]:
        """The MXFP4 instruction class of each GEMM (the part's: the sm_107 K = 128 step on SM107, the K = 64 step elsewhere)."""
        return expert_classes_mx(family)

    @property
    def chain1(self) -> Mxfp4Chain:
        """The GEMM1 chain of the class: the gated ``act`` on the (gate, up) pair or the ungated ``act`` on the accumulator, with
        the parameters; bf16 output."""
        return Mxfp4Chain(
            "glu" if self.gated else "act",
            self.act,
            "bf16",
            self.params,
            arith=self.arith,
        )

    @property
    def chain2(self) -> Mxfp4Chain:
        """The GEMM2 chain: ``cvt_bf16(acc)``."""
        return MXFP4_GEMM2_CHAIN

    def view(self) -> Dict[str, object]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for this backend (``quant`` names the pair:
        the fp4 backend's pairs share the view key)."""
        return {
            "quant": "mxfp4",
            "gemm1_weights": self.w1,
            "gemm1_weights_scale": self.w1_sf,
            "gemm2_weights": self.w2,
            "gemm2_weights_scale": self.w2_sf,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
            "activation_arith": self.arith,
        }

    @staticmethod
    def from_view(view: Dict[str, object]) -> "MoEWeightsMXFP4":
        gated = bool(view.get("gated", True))
        return MoEWeightsMXFP4(
            view["gemm1_weights"],  # type: ignore[arg-type]
            view["gemm1_weights_scale"],  # type: ignore[arg-type]
            view["gemm2_weights"],  # type: ignore[arg-type]
            view["gemm2_weights_scale"],  # type: ignore[arg-type]
            int(view["hidden_size"]),  # type: ignore[call-overload]
            int(view["intermediate_size"]),  # type: ignore[call-overload]
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),  # type: ignore[arg-type]
            str(view.get("activation_arith", "exact")),
        )


def quantize_rows_mx(
    x: torch.Tensor, out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The class's MXFP4 quantization of bf16 rows ``[M, K]`` (``mxfp4.quant``: one UE8M0 scale per 32 elements of a row):
    ``(x4 [M, K/2] uint8, sf flat 128x4 uint8 of mx_sf_numel(M, K) bytes)``. ``out = (x4, sf)`` preallocated buffers."""
    from ...batch_invariant.quantization.mx_api import mx_quantize

    M, K = x.shape
    dev = x.device
    if M == 0:
        return (
            torch.empty(0, K // 2, dtype=torch.uint8, device=dev),
            torch.zeros(mx_sf_numel(0, K), dtype=torch.uint8, device=dev),
        )
    x4, sf = mx_quantize(x, MXFP4, "128x4", out=out, enable_pdl=True)
    return x4, sf.reshape(-1)


def grouped_gemm_mxfp4(
    a4: torch.Tensor,
    a_sf: torch.Tensor,
    w4: torch.Tensor,
    w_sf: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: Mxfp4Chain,
    in_fmt: str,
    out: Optional[torch.Tensor] = None,
    dense_a: bool = False,
    rows_sf: Optional[int] = None,
    *,
    src: Optional[torch.Tensor] = None,
    k_div: int = 1,
) -> torch.Tensor:
    """``out[p] = chain(a4[p] @ w4[e(p)].T)`` on the expert segments ``counts`` (int32 ``[E]``, group order = expert order),
    the UE8M0 block scales of both operands applied inside the instruction. Padded packing (default): ``a4 [rows, K/2]``
    uint8 with every group's rows starting at a multiple of 128 (``rows`` a multiple of 128), ``a_sf`` its 128x4 swizzled
    scales. Dense packing (``dense_a``): ``a4 [rows, K/2]`` the groups' rows back to back (any row count), ``a_sf`` at the
    padded positions of ``rows_sf`` rows (a multiple of 128, at least the live rows), ``out [rows]`` dense. ``w4 [E, N, K/2]``
    uint8, ``w_sf [E, mx_sf_numel(N, K)]``. The record selects the body as :func:`.api_nvfp4.grouped_gemm_nvfp4`: a dense
    record runs the padded body on either packing, a swapped record the swapped decode body on the dense packing (``a4``
    carries at least the token tile's rows past the live rows: the op's ``DENSE_TAIL``). The gather record (``config.gather_a``,
    a swapped record): ``a4 [T, K/2]`` the unpermuted token rows, ``a_sf`` their scale factors (the flat 128x4 buffer of the
    ``T`` rows, or the linear ``[T, K/32]`` bytes), ``src [>= Tk]`` int32 the plan's slot of every dense row (``src[p] //
    k_div`` its token), ``out`` given (it sizes the dense rows); the body reads the rows itself."""
    from .api import _check_grouped_chain, _check_grouped_operands, _grouped_out
    from .blocks.compile import (
        can_implement_grouped_blockscaled,
        can_implement_swapped_blockscaled,
        get_compiled_grouped,
        get_compiled_swapped,
    )

    if config.gather_a:
        return _grouped_gemm_mxfp4_gather(
            a4, a_sf, w4, w_sf, counts, config, chain, in_fmt, out, src, k_div
        )
    if src is not None:
        raise ValueError(
            "grouped_gemm_mxfp4: src belongs to the gather record (config.gather_a)"
        )
    rows, Kh, E, N, _ = _check_grouped_operands(
        "grouped_gemm_mxfp4",
        a4,
        w4,
        counts,
        a4.shape[0] if a4.dim() == 2 else None,
        None,
        torch.uint8,
        ALIGN_MXFP4 // 2,
        "packed fp4 uint8",
        n_align=8,
    )
    K = 2 * Kh
    if dense_a:
        if rows_sf is None or rows_sf % GROUP_ALIGN or rows_sf < rows - DENSE_TAIL:
            raise ValueError(
                "grouped_gemm_mxfp4: dense A takes the scale factors at the padded positions (rows_sf a multiple of 128)"
            )
    else:
        if rows % GROUP_ALIGN or (rows_sf is not None and rows_sf != rows):
            raise ValueError(
                "grouped_gemm_mxfp4: the padded rows are a multiple of the 128-row scale-factor atom"
            )
        rows_sf = rows
    if a_sf.dtype != torch.uint8 or a_sf.numel() < mx_sf_numel(rows_sf, K):
        raise ValueError(
            "grouped_gemm_mxfp4: a_sf must hold the 128x4 swizzled scales of the (padded) rows"
        )
    if w_sf.dtype != torch.uint8 or w_sf.numel() < E * mx_sf_numel(N, K):
        raise ValueError(
            "grouped_gemm_mxfp4: w_sf must hold E consecutive 128x4 swizzled scale buffers of (N, K)"
        )
    if not isinstance(chain, Mxfp4Chain):
        raise ValueError("grouped_gemm_mxfp4: chain must be an Mxfp4Chain")
    if in_fmt not in MXFP4_FMTS or chunks_for(N, K, in_fmt) != 1:
        raise ValueError(
            "grouped_gemm_mxfp4: the grouped class takes unsplit expert weight shapes"
        )
    if config.swap_ab:
        if not dense_a:
            raise ValueError(
                "grouped_gemm_mxfp4: the swapped decode body reads the dense packing (dense_a=True)"
            )
        if not can_implement_swapped_blockscaled(config, N, K, E, chain, in_fmt):
            raise ValueError(
                f"grouped_gemm_mxfp4: swapped record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
            )
    elif not can_implement_grouped_blockscaled(config, N, K, E, chain, in_fmt):
        raise ValueError(
            f"grouped_gemm_mxfp4: record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
        )
    n_out = _check_grouped_chain("grouped_gemm_mxfp4", N, chain)
    out_dtype = _OUT[chain.out_fmt]
    if out is None:
        out = torch.zeros(rows, n_out, dtype=out_dtype, device=a4.device)
    out = _grouped_out("grouped_gemm_mxfp4", out, rows, n_out, out_dtype, a4.device)
    if rows == 0:
        return out
    grid = min(
        persistent_clusters(a4.device, config), launch_tile_bound(config, rows, E, N)
    )
    if config.swap_ab:
        get_compiled_swapped(config, chain.at(in_fmt))(
            a4.unsqueeze(-1),
            a_sf.reshape(-1),
            w4.permute(1, 2, 0),
            w_sf.reshape(-1),
            out,
            counts,
            grid,
        )
        return out
    get_compiled_grouped(
        config, chain.at(in_fmt), packing="dense" if dense_a else "padded"
    )(
        a4.unsqueeze(-1),
        w4.permute(1, 2, 0),
        a_sf.reshape(-1),
        w_sf.reshape(-1),
        out,
        counts,
        rows_sf,
        grid,
    )
    return out


def _grouped_gemm_mxfp4_gather(
    a4, a_sf, w4, w_sf, counts, config, chain, in_fmt, out, src, k_div
) -> torch.Tensor:
    """The gather record of :func:`grouped_gemm_mxfp4` (its operand rules, the token rows in place of the permuted rows)."""
    from .api import _check_grouped_chain, _check_grouped_operands, _grouped_out
    from .api_nvfp4 import gather_form_ok, gather_sf_layout
    from .blocks.compile import can_implement_swapped_blockscaled, get_compiled_swapped

    T, Kh, E, N, _ = _check_grouped_operands(
        "grouped_gemm_mxfp4",
        a4,
        w4,
        counts,
        a4.shape[0] if a4.dim() == 2 else None,
        None,
        torch.uint8,
        ALIGN_MXFP4 // 2,
        "packed fp4 uint8",
        n_align=8,
    )
    K = 2 * Kh
    if out is None or src is None:
        raise ValueError(
            "grouped_gemm_mxfp4: the gather record reads the token rows through the plan's map: src and out"
        )
    if (
        src.dtype != torch.int32
        or src.dim() != 1
        or not src.is_contiguous()
        or k_div < 1
    ):
        raise ValueError(
            "grouped_gemm_mxfp4: src must be a contiguous int32 [>= Tk] and k_div >= 1"
        )
    sf_src = gather_sf_layout(a_sf, T, K, MX_BLOCK)
    if a_sf.dtype != torch.uint8 or (
        sf_src == "128x4" and a_sf.numel() < mx_sf_numel(T, K)
    ):
        raise ValueError(
            "grouped_gemm_mxfp4: a_sf must hold the 128x4 swizzled scales of the T token rows or their linear [T, K/32] bytes"
        )
    if not gather_form_ok(K, sf_src, MX_BLOCK):
        raise ValueError(
            "grouped_gemm_mxfp4: the linear scale layout's rows must be whole 4-byte groups (K a multiple of 128)"
        )
    if w_sf.dtype != torch.uint8 or w_sf.numel() < E * mx_sf_numel(N, K):
        raise ValueError(
            "grouped_gemm_mxfp4: w_sf must hold E consecutive 128x4 swizzled scale buffers of (N, K)"
        )
    if not isinstance(chain, Mxfp4Chain):
        raise ValueError("grouped_gemm_mxfp4: chain must be an Mxfp4Chain")
    if in_fmt not in MXFP4_FMTS or chunks_for(N, K, in_fmt) != 1:
        raise ValueError(
            "grouped_gemm_mxfp4: the grouped class takes unsplit expert weight shapes"
        )
    if not config.swap_ab or not can_implement_swapped_blockscaled(
        config, N, K, E, chain, in_fmt
    ):
        raise ValueError(
            f"grouped_gemm_mxfp4: gather record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
        )
    n_out = _check_grouped_chain("grouped_gemm_mxfp4", N, chain)
    rows = out.shape[0]
    out = _grouped_out(
        "grouped_gemm_mxfp4", out, rows, n_out, _OUT[chain.out_fmt], a4.device
    )
    if rows == 0 or T == 0:
        return out
    grid = min(
        persistent_clusters(a4.device, config), launch_tile_bound(config, rows, E, N)
    )
    get_compiled_swapped(config, chain.at(in_fmt), sf_src=sf_src)(
        a4.unsqueeze(-1),
        a_sf.reshape(-1),
        src,
        k_div,
        w4.permute(1, 2, 0),
        w_sf.reshape(-1),
        out,
        counts,
        grid,
    )
    return out


_UNIT: Dict[str, torch.Tensor] = {}


def _placeholder(device: torch.device) -> torch.Tensor:
    """A one-element fp32 tensor for the scatter kernels' unread per-token-scale operands, per device (no fill per call)."""
    key = str(torch.device(device))
    t = _UNIT.get(key)
    if t is None:
        t = _UNIT[key] = torch.ones(1, dtype=torch.float32, device=device)
    return t


def fused_moe_mxfp4(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: MoEWeightsMXFP4,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    in_fmts: Optional[Tuple[str, str]] = None,
    path: str = "dense",
    return_intermediates: bool = False,
    swap: Optional[bool] = None,
    plan: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    round_scales: bool = False,
    gather: Optional[bool] = None,
):
    """The MXFP4 MoE FFN for pre-routed tokens: ``x`` bf16 ``[T, H]`` (quantized here with the class) or the quantized pair
    ``(x4 [T, H/2] uint8, x_sf uint8: the 128x4 swizzled buffer of ``mxfp4_quantize(..., batch_invariant=True)`` or the linear
    ``[T, H/32]`` bytes)``; ``token_selected_experts [T, k]`` int32, ``token_final_scales [T, k]`` f32 (``round_scales``: the
    combine takes their bf16 rounding); the output bf16 ``[T, H]`` in token order. ``in_fmts`` names the instruction class per
    GEMM (default: the part's); ``config1`` / ``config2`` the grouped records (any record of the space gives the same bits);
    ``path`` the packing of the expanded rows; ``swap`` the dense path's body (None: the dispatcher's ranking, True / False:
    the swapped / the padded body on both GEMMs); ``plan`` the dense packing's precomputed plan (:func:`.api_nvfp4.route_plan_dual`'s
    outputs after the ids and weights); ``gather`` GEMM1's swapped records in their gather form (the body reads the token rows
    through the plan's map: this pair's scatter launch gone) -- False excludes it, True forces it, None (the default) applies
    the structural rule (:func:`~.schedule_nvfp4.gather_form_rule`: the narrow swapped tile); an explicit ``config1`` selects
    the form itself (``gather_a``)."""
    from .combine import get_compiled_combine, warps_per_token
    from .permute import warps_per_row
    from .permute_nvfp4 import (
        get_compiled_scatter_fp4,
        get_compiled_scatter_sf,
        scatter_lanes,
        scatter_sf_lanes,
    )
    from .schedule_mxfp4 import default_configs_mxfp4
    from .schedule_nvfp4 import gather_form_rule

    if path not in PATHS:
        raise ValueError(f"fused_moe_mxfp4: path must be one of {PATHS}, not {path!r}")
    H, I, E = w.hidden_size, w.intermediate_size, w.num_experts
    if isinstance(x, torch.Tensor):
        if (
            x.dtype != torch.bfloat16
            or x.dim() != 2
            or x.stride(1) != 1
            or x.stride(0) % 8
        ):
            raise TypeError(
                "fused_moe_mxfp4: x must be a row-major, 16-byte-aligned bf16 [T, H] or a quantized pair"
            )
        if x.shape[1] != H:
            raise ValueError(
                f"fused_moe_mxfp4: x has H = {x.shape[1]}, the weights H = {H}"
            )
        T = x.shape[0]
        dev = x.device
        x4, x_sf = quantize_rows_mx(x.contiguous())
        sf_src = "128x4"
    else:
        x4, x_sf = x
        T = x4.shape[0]
        dev = x4.device
        # the pack's scale factors: the class's flat 128x4 buffer, or the row-major ``[T, H / 32]`` bytes (the scatter assembles
        # the atoms from them: no re-tiling pass)
        sf_src = (
            "linear"
            if x_sf.dim() == 2 and tuple(x_sf.shape) == (T, H // MX_BLOCK)
            else "128x4"
        )
        x_sf = x_sf.contiguous().reshape(-1)
        if (
            x4.dtype != torch.uint8
            or tuple(x4.shape) != (T, H // 2)
            or not x4.is_contiguous()
            or x_sf.dtype != torch.uint8
            or (sf_src == "128x4" and x_sf.numel() < mx_sf_numel(T, H))
        ):
            raise TypeError(
                "fused_moe_mxfp4: the quantized pair is (x4 [T, H/2] uint8, x_sf 128x4 uint8 of >= mx_sf_numel(T, H) bytes or the linear [T, H/32] bytes)"
            )
    k = token_selected_experts.shape[1]
    check_shape_mxfp4(H, I, E, k, w.gated)
    if (
        tuple(token_selected_experts.shape) != (T, k)
        or token_selected_experts.dtype != torch.int32
    ):
        raise TypeError(
            "fused_moe_mxfp4: token_selected_experts must be an int32 [T, k]"
        )
    if (
        tuple(token_final_scales.shape) != (T, k)
        or token_final_scales.dtype != torch.float32
    ):
        raise TypeError("fused_moe_mxfp4: token_final_scales must be a float32 [T, k]")
    experts = token_selected_experts.contiguous()
    scales = token_final_scales.contiguous()
    if out is None:
        out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    if (
        tuple(out.shape) != (T, H)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "fused_moe_mxfp4: out must be a row-major, 16-byte-aligned bf16 [T, H]"
        )
    sm_count, family = _part(dev)
    in1, in2 = in_fmts if in_fmts is not None else w.classes(family)
    chain1, chain2 = w.chain1, w.chain2
    if config1 is None or config2 is None:
        ranked1 = config1 is None
        c1, c2 = default_configs_mxfp4(
            T,
            k,
            H,
            I,
            E,
            (in1, in2),
            chain1,
            chain2,
            sm_count,
            family,
            swap=swap if path == "dense" else False,
            gather=(gather or False) if path == "dense" else False,
        )
        config1, config2 = config1 or c1, config2 or c2
        if ranked1 and gather is None and path == "dense":
            # the rule: this pair's scatter launch folded into the narrow swapped tile's gather form
            config1 = gather_form_rule(config1, True)
    if path != "dense" and (swap or config1.swap_ab or config2.swap_ab):
        raise ValueError(
            "fused_moe_mxfp4: the swapped decode body reads the dense packing (path='dense')"
        )
    if config2.gather_a or (path != "dense" and (gather or config1.gather_a)):
        raise ValueError(
            "fused_moe_mxfp4: the gather form is GEMM1's on the dense packing (path='dense')"
        )
    gather1 = config1.gather_a
    if gather1 and not gather_form_ok(H, sf_src, MX_BLOCK):
        # the linear scale layout's rows are not whole 4-byte groups: the box form through the scatter
        import dataclasses

        config1 = dataclasses.replace(config1, gather_a=False)
        gather1 = False
    if path != "dense" and plan is not None:
        raise ValueError(
            "fused_moe_mxfp4: the precomputed plan is the dense packing's (path='dense')"
        )
    Tk = T * k
    Tk_pad = padded_rows_bound(T, k, E)
    pcols_h, pcols_i = mx_sf_cols(H), mx_sf_cols(I)
    unit = _placeholder(dev)
    wpr = warps_per_row(Tk, H // 32, sm_count)
    if path == "dense":
        counts, dest, src, dest_pad, pos_pad = (
            plan if plan is not None else permute_plan_dual(experts, E)
        )
        # the fp4 rows back to back (a tail past Tk for the last tiles' TMA boxes); the scale factors at the padded positions
        # (one 128x4 atom per (expert, 128-row tile); a tail past Tk_pad likewise)
        Tk_a, Tk_sf = Tk + DENSE_TAIL, Tk_pad + DENSE_TAIL
        x_perm4 = x_perm_sf = None
        if gather1:
            # the gather form: GEMM1 reads the token rows through the plan's map (no scatter launch)
            g_a = x4
            g_sf = x_sf.view(T, H // MX_BLOCK) if sf_src == "linear" else x_sf
        else:
            x_perm4 = torch.empty(Tk_a, H // 2, dtype=torch.uint8, device=dev)
            x_perm_sf = torch.empty(
                mx_sf_numel(Tk_sf, H), dtype=torch.uint8, device=dev
            )
            if Tk:
                get_compiled_scatter_fp4(
                    k,
                    wpr,
                    sf_src,
                    False,
                    True,
                    MX_BLOCK,
                    **scatter_lanes(H // 2, pcols_h, wpr),
                )(
                    x4,
                    x_sf,
                    unit,
                    dest.reshape(-1),
                    dest_pad.reshape(-1),
                    x_perm4,
                    x_perm_sf,
                    unit,
                    Tk,
                    H // 2,
                    pcols_h,
                )
            g_a, g_sf = x_perm4, x_perm_sf
        a = torch.empty(Tk_a, I, dtype=torch.bfloat16, device=dev)
        grouped_gemm_mxfp4(
            g_a,
            g_sf,
            w.w1,
            w.w1_sf,
            counts,
            config1,
            chain1,
            in1,
            out=a,
            dense_a=True,
            rows_sf=Tk_sf,
            src=src if gather1 else None,
            k_div=k,
        )
        # the class's second pass over the live rows: the codes stay dense; the scale factors are re-tiled to the padded
        # positions for GEMM2's atom-aligned scale boxes (a byte move)
        a4 = torch.empty(Tk_a, I // 2, dtype=torch.uint8, device=dev)
        a_sf = torch.empty(mx_sf_numel(Tk, I), dtype=torch.uint8, device=dev)
        a_sf_pad = torch.empty(mx_sf_numel(Tk_sf, I), dtype=torch.uint8, device=dev)
        if Tk:
            quantize_rows_mx(a[:Tk], out=(a4[:Tk], a_sf))
            get_compiled_scatter_sf(1, True, spl=scatter_sf_lanes(pcols_i))(
                a_sf, unit, pos_pad, a_sf_pad, unit, Tk, pcols_i
            )
        y = torch.empty(Tk_a, H, dtype=torch.bfloat16, device=dev)
        grouped_gemm_mxfp4(
            a4,
            a_sf_pad,
            w.w2,
            w.w2_sf,
            counts,
            config2,
            chain2,
            in2,
            out=y,
            dense_a=True,
            rows_sf=Tk_sf,
        )
        if T:
            get_compiled_combine(
                k, "bf16", "bf16", warps_per_token(T, H, sm_count), round_scales
            )(y, dest, scales, out, T, H)
        if return_intermediates:
            return out, dict(
                counts=counts,
                dest=dest,
                src=src,
                dest_pad=dest_pad,
                pos_pad=pos_pad,
                x4=x4,
                x_sf=x_sf,
                x_perm4=None if x_perm4 is None else x_perm4[:Tk],
                x_perm_sf=None
                if x_perm_sf is None
                else x_perm_sf[: mx_sf_numel(Tk_pad, H)],
                a=a[:Tk],
                a4=a4[:Tk],
                a_sf=a_sf,
                a_sf_pad=a_sf_pad[: mx_sf_numel(Tk_pad, I)],
                y=y[:Tk],
                config1=config1,
                config2=config2,
                in_fmts=(in1, in2),
                Tk_pad=Tk_pad,
                path=path,
            )
        return out
    # the padded packing: every group's rows start at a 128-row atom; the padding rows are zero (computed, never stored)
    from .api import PLAN_MULTI_MIN_N
    from .permute import (
        get_compiled_plan,
        get_compiled_plan_multi,
        PLAN_BLOCK,
        plan_threads,
    )

    n = Tk
    dest = torch.empty(n, dtype=torch.int32, device=dev)
    src = torch.full((Tk_pad,), -1, dtype=torch.int32, device=dev)
    counts = torch.empty(E, dtype=torch.int32, device=dev)
    flat = experts.reshape(-1)
    if n == 0:
        counts.zero_()
    elif n >= PLAN_MULTI_MIN_N:
        nb = (n + PLAN_BLOCK - 1) // PLAN_BLOCK
        hist_blk = torch.empty(nb * E, dtype=torch.int32, device=dev)
        get_compiled_plan_multi(GROUP_ALIGN, False)(
            flat, counts, dest, src, dest, dest, hist_blk, n, E, nb
        )
    else:
        get_compiled_plan(GROUP_ALIGN, False, threads=plan_threads(n))(
            flat, counts, dest, src, dest, dest, n, E
        )
    dest = dest.view(T, k)
    x_perm4 = torch.zeros(Tk_pad, H // 2, dtype=torch.uint8, device=dev)
    x_perm_sf = torch.zeros(mx_sf_numel(Tk_pad, H), dtype=torch.uint8, device=dev)
    if Tk:
        flat_dest = dest.reshape(-1)
        get_compiled_scatter_fp4(
            k, wpr, sf_src, False, True, MX_BLOCK, **scatter_lanes(H // 2, pcols_h, wpr)
        )(
            x4,
            x_sf,
            unit,
            flat_dest,
            flat_dest,
            x_perm4,
            x_perm_sf,
            unit,
            Tk,
            H // 2,
            pcols_h,
        )
    a = grouped_gemm_mxfp4(
        x_perm4, x_perm_sf, w.w1, w.w1_sf, counts, config1, chain1, in1
    )
    a4, a_sf = quantize_rows_mx(
        a
    )  # the class, second pass (padding rows included: zeros)
    y = grouped_gemm_mxfp4(a4, a_sf, w.w2, w.w2_sf, counts, config2, chain2, in2)
    if T:
        get_compiled_combine(
            k, "bf16", "bf16", warps_per_token(T, H, sm_count), round_scales
        )(y, dest, scales, out, T, H)
    if return_intermediates:
        return out, dict(
            counts=counts,
            dest=dest,
            src=src,
            x4=x4,
            x_sf=x_sf,
            x_perm4=x_perm4,
            x_perm_sf=x_perm_sf,
            a=a,
            a4=a4,
            a_sf=a_sf,
            y=y,
            config1=config1,
            config2=config2,
            in_fmts=(in1, in2),
            Tk_pad=Tk_pad,
            path=path,
        )
    return out


def moe_ffn_mxfp4(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    router_logits: torch.Tensor,
    w: MoEWeightsMXFP4,
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
) -> torch.Tensor:
    """Routing (the bf16 classes' kernel) + the MXFP4 FFN."""
    experts, weights = route(router_logits, top_k, method, bias, scaling_factor)
    return fused_moe_mxfp4(x, experts, weights, w, out, config1, config2)


__all__ = [
    "ALIGN_MXFP4",
    "MoEWeightsMXFP4",
    "check_shape_mxfp4",
    "fused_moe_mxfp4",
    "grouped_gemm_mxfp4",
    "moe_ffn_mxfp4",
    "mx_sf_cols",
    "mx_sf_numel",
    "quantize_rows_mx",
    "route_plan_dual",
]
