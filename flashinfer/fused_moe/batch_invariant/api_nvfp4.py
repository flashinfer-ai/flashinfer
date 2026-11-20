# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 MoE classes ``moe.nvfp4.<act>`` (W4A4): NVFP4 expert weights with one per-tensor global scale per expert and
GEMM, per-token NVFP4 activations, on the bf16 classes' routing, permutation and combine (:mod:`.api`).

Pipeline: the per-token NVFP4 quantization of ``x`` (the class ``nvfp4.quant.pertoken`` of
:mod:`flashinfer.batch_invariant.quantization`, or the caller's quantized triple ``(x4, x_sf, x_pts)``) -> the deterministic
permutation of the expanded rows (expert order, then token-slot order) with its scale factors placed at 128-row atoms ->
the grouped block-scaled GEMM1 with the W4A4 chain (``cvt_bf16(act(mul(g, s)) * mul(u, s))``, ``s = mul(alpha1[e],
pts[row])``) -> the per-token quantization of the materialised bf16 intermediate (the same class, its second pass) -> the
grouped block-scaled GEMM2 (``cvt_bf16(mul(acc, mul(alpha2[e], pts2[row])))``) -> the slot-order combine. Every output bit
is a function of the token's own row, its ``k`` experts and routing weights and the expert weights (the specification is
:mod:`.spec_nvfp4`; the Lean statements the ``MoENvfp4`` module of the mode).

Two packings of the expanded rows are placements of the same bits (``path``): ``"dense"`` (the default: the fp4 rows back
to back, the scale factors and per-token scales at the 128-row-padded positions, a dense output; the intermediate quantized
over the live rows and its scales re-tiled to the padded positions) and ``"padded"`` (every group's rows start at a 128-row
atom; the padding rows are zero-filled, computed and never stored).
"""

from __future__ import annotations

import dataclasses

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from ...batch_invariant.quantization.nvfp4_schedule import padded_sf_cols, sf_numel
from ...batch_invariant.support import BatchInvariantUnsupportedError, unsupported
from ...gemm.batch_invariant.api import part_of, persistent_clusters, sm_count_of
from .schedule import launch_tile_bound
from ...gemm.batch_invariant.generator import Knobs
from ...gemm.batch_invariant.schedule import chunks_for, fp4_class_for
from ...gemm.batch_invariant.spec import NVFP4_FMTS
from . import spec as S
from .api import interleave_fc1, permute_plan, route  # noqa: F401  (route re-exported for the class's callers)
from .blocks.epilogue import W4A4Chain
from .spec_nvfp4 import CLASS, E2M1_MAX, E4M3_MAX, INTERMEDIATES, TE_GS_INV

#: 16-byte rows of the packed fp4 operands (the TMA alignment) and whole 64-element scale-factor column groups.
ALIGN_NVFP4 = 32


def expert_classes(
    H: int, I: int, family: Optional[str], gated: bool = True
) -> Tuple[str, str]:
    """The NVFP4 instruction class of each expert weight by the dense rule (`schedule.fp4_class_for`): GEMM1 on ``(2I, H)``
    (``(I, H)`` ungated), GEMM2 on ``(H, I)``; a function of the weight shapes and the architecture family only."""
    return fp4_class_for(2 * I if gated else I, H, family), fp4_class_for(H, I, family)


MAX_TOP_K = 16
PATHS = ("dense", "padded")
#: Rows past the live rows in the dense packing's A buffers (and past the padded rows in its scale-factor buffers) so the
#: last tiles' TMA boxes stay in bounds: a box past the end is zero-filled by the engine at a cost. The tail rows are never
#: stored. A placement.
DENSE_TAIL = 256
# the intermediate's scale factors reach the padded positions by the row-mapped quantizer (one launch) or by the quantizer
# and a re-tiling move (two launches; the same bytes): the two spellings of one placement
INTERMEDIATE_RETILE = "mapped"
RETILES = ("mapped", "scatter")
# a bf16 input reaches the grouped layout by the quantizer and the fp4 scatter (two launches, the dense codes round-tripped)
# or by the quantizer storing into the grouped layout itself (one launch; the same bytes): the two spellings of one placement
INPUT_SCATTER = "fused"
INPUT_SCATTERS = ("fused", "scatter")
GROUP_ALIGN = 128  # the scale-factor atom: the group alignment of the scale factors
_OUT = {"bf16": torch.bfloat16, "f32": torch.float32}


def check_shape_nvfp4(
    H: Optional[int], I: int, E: int, k: Optional[int] = None, gated: bool = True
) -> None:
    """The complete acceptance rule of the class on ``(H, I, E, k)``: ``H`` and ``I`` positive multiples of 32 (16-byte rows
    of the packed fp4 operands and whole scale-factor column groups), ``1 <= E <= 1024`` (the routing kernel's widest bucket
    and the group tables), ``1 <= k <= min(E, 16)``, and the expert weight shapes unsplit by the dense class's chunk rule
    (``chunks_for(N1, H) == chunks_for(H, I) == 1``: the grouped classes take one fold per weight). Everything else about a
    shape is placement. ``H=None`` checks the rules the layer's configuration fixes (the hidden size enters at the first
    call, with the full rule). Raises :class:`BatchInvariantUnsupportedError`; the class never routes a shape elsewhere."""
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

    if I <= 0 or I % ALIGN_NVFP4 or (H is not None and (H <= 0 or H % ALIGN_NVFP4)):
        bad(
            f"hidden size {H} / intermediate size {I}"
            if H is not None
            else f"intermediate size {I}",
            f"positive multiples of {ALIGN_NVFP4} (16-byte rows of the quantized activations and weights)",
        )
    if not 1 <= E <= MAX_E:
        bad(f"{E} experts", f"1 <= num_experts <= {MAX_E}")
    if k is not None and not 1 <= k <= min(E, MAX_TOP_K):
        bad(f"top_k {k}", f"1 <= top_k <= min(num_experts, {MAX_TOP_K})")
    if H is None:
        return
    n1 = 2 * I if gated else I
    if chunks_for(n1, H, "nvfp4") != 1 or chunks_for(H, I, "nvfp4") != 1:
        bad(
            "the expert weight shapes",
            "unsplit expert weights (the dense class would K-split this shape)",
        )


def _scale_vector(t, name: str, E: int, device: torch.device) -> torch.Tensor:
    """A per-expert vector of ``E`` multipliers as a contiguous fp32 ``[E]`` on ``device``: finite, positive, normal values
    (their reciprocals are factors of the class's chains, so a zero, subnormal, infinite or NaN entry is refused here instead
    of reaching the kernels as an infinite or NaN scale); a tensor of ``E`` elements of any shape is reshaped, a one-element
    tensor broadcast to the experts."""
    v = torch.as_tensor(t, dtype=torch.float32, device=device)
    if v.numel() == 1:
        v = v.reshape(1).expand(E)
    if v.numel() != E:
        raise ValueError(
            f"{CLASS}: {name} must hold one fp32 multiplier per expert ({E}), got shape {tuple(v.shape)}"
        )
    v = v.reshape(E).contiguous()
    tiny = torch.finfo(torch.float32).tiny
    if not bool(torch.isfinite(v).all()) or not bool((v >= tiny).all()):
        raise ValueError(
            f"{CLASS}: {name} must be finite positive normal fp32 multipliers (at least {tiny:.4g})"
        )
    return v


def fp4_global_scale(w: torch.Tensor) -> torch.Tensor:
    """FlashInfer's per-tensor global scale of a weight: ``448 * 6 / amax(w)`` (a 1-element f32 tensor; a static property of
    the model, computed once at preparation)."""
    amax = (
        w.detach()
        .to(torch.float32)
        .abs()
        .amax()
        .clamp_min(torch.finfo(torch.float32).tiny)
    )
    return (E4M3_MAX * E2M1_MAX / amax).reshape(1).to(torch.float32)


@dataclass
class MoEWeightsNVFP4:
    """The W4A4 expert weights as the kernels consume them: ``w1 [E, N1, H/2]`` packed fp4 (``N1 = 2I``, gate / up
    interleaved, for a gated chain; ``I`` as given for an ungated one), ``w1_sf [E, sf_bytes(N1, H)]`` 128x4 swizzled e4m3
    scales per expert, ``w1_alpha [E]`` f32 = ``1 / global_scale`` (as given to the kernel: a multiplier), likewise ``w2 [E,
    H, I/2]``, ``w2_sf``, ``w2_alpha``; the GEMM1 chain ``act`` with its ``params``."""

    w1: torch.Tensor
    w1_sf: torch.Tensor
    w1_alpha: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_alpha: torch.Tensor
    hidden_size: int
    intermediate_size: int
    gated: bool = True
    act: str = "silu"
    params: Tuple[Optional[float], ...] = ()
    #: the intermediate's quantization: ``per_token`` (`moe.nvfp4.<act>`) or ``static`` (`moe.nvfp4.<act>.static`: GEMM1's
    #: epilogue quantizes the fp32 gated value per 16-column block with the expert's static scale ``gs2 [E]``, a multiplier in
    #: FlashInfer's ``fc2_input_scale`` convention; ``inv2 [E] = 1 / gs2`` formed here once, read by GEMM2 as given)
    intermediate: str = "per_token"
    gs2: Optional[torch.Tensor] = None
    inv2: Optional[torch.Tensor] = None
    #: the chain's arithmetic (``spec.ARITHS``): ``exact`` (`moe.nvfp4.<act>`) or ``hw`` (`moe.nvfp4.<act>.hw`: the
    #: hardware-arithmetic class, the part's ``ex2`` / ``rcp`` in the chain; the chains without them stay the exact class)
    arith: str = "exact"

    @staticmethod
    def prepare(
        fc1: torch.Tensor,
        fc2: torch.Tensor,
        interleaved: bool = False,
        gated: bool = True,
        act: Optional[str] = None,
        params: Sequence[Optional[float]] = (),
        global_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        intermediate: str = "per_token",
        intermediate_scales: Optional[torch.Tensor] = None,
        arith: str = "exact",
    ) -> "MoEWeightsNVFP4":
        """Quantize bf16 experts (``fc1 [E, 2I, H]`` rows ``[up, gate]``, or ``[E, I, H]`` ungated; ``fc2 [E, H, I]``) with the
        batch-invariant per-tensor NVFP4 class, one global scale per expert and GEMM (``global_scales = (gs1 [E], gs2 [E])``,
        or FlashInfer's ``448 * 6 / amax``). The gate / up interleave is a row permutation applied before the quantization (a
        per-tensor global scale commutes with it). ``act`` defaults to the class's default chain of the gating.
        ``intermediate="static"``: the static class, whose GEMM1 quantizes the intermediate in place with the per-expert static
        scales ``intermediate_scales [E]`` (f32 multipliers; default ones, every open backend's unified-layer value)."""
        from ...batch_invariant.quantization.nvfp4_api import nvfp4_quantize_rows

        if intermediate not in INTERMEDIATES:
            raise ValueError(
                f"intermediate must be one of {INTERMEDIATES}, not {intermediate!r}"
            )
        if arith not in S.ARITHS:
            raise ValueError(f"arith must be one of {S.ARITHS}, not {arith!r}")
        act = act if act is not None else ("silu" if gated else "relu2")
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
                "the NVFP4 MoE classes take bf16 expert weights fc1 [E, N1, H], fc2 [E, H, I]"
            )
        E, N1, H = fc1.shape
        I = N1 // 2 if gated else N1
        if tuple(fc2.shape) != (E, H, I) or (gated and N1 % 2):
            raise ValueError(
                f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)} (gated={gated})"
            )
        check_shape_nvfp4(H, I, E, gated=gated)
        w1i = fc1.contiguous() if (interleaved or not gated) else interleave_fc1(fc1)
        fc2 = fc2.contiguous()
        if global_scales is not None:
            # the caller's per-tensor global scales: one finite positive normal multiplier per expert and GEMM
            global_scales = (
                _scale_vector(global_scales[0], "global_scales[0]", E, w1i.device),
                _scale_vector(global_scales[1], "global_scales[1]", E, w1i.device),
            )
        w1s, sf1s, a1s, w2s, sf2s, a2s = [], [], [], [], [], []
        for e in range(E):
            gs1 = (
                global_scales[0][e].reshape(1)
                if global_scales is not None
                else fp4_global_scale(w1i[e])
            )
            gs2 = (
                global_scales[1][e].reshape(1)
                if global_scales is not None
                else fp4_global_scale(fc2[e])
            )
            q1, sf1 = nvfp4_quantize_rows(w1i[e], gs1, "128x4")
            q2, sf2 = nvfp4_quantize_rows(fc2[e], gs2, "128x4")
            w1s.append(q1)
            sf1s.append(sf1.reshape(-1))
            a1s.append(
                torch.reciprocal(gs1.to(w1i.device))
            )  # div.rn: the alpha as mm_fp4 takes it (a multiplier)
            w2s.append(q2)
            sf2s.append(sf2.reshape(-1))
            a2s.append(torch.reciprocal(gs2.to(w1i.device)))
        gs_inter = inv_inter = None
        if intermediate == "static":
            gs_inter = (
                torch.ones(E, dtype=torch.float32, device=w1i.device)
                if intermediate_scales is None
                else _scale_vector(
                    intermediate_scales, "intermediate_scales", E, w1i.device
                )
            )
            inv_inter = torch.reciprocal(
                gs_inter
            ).contiguous()  # div.rn: the quantizer's correctly rounded reciprocal
        return MoEWeightsNVFP4(
            torch.stack(w1s).contiguous(),
            torch.stack(sf1s).contiguous(),
            torch.cat(a1s).contiguous(),
            torch.stack(w2s).contiguous(),
            torch.stack(sf2s).contiguous(),
            torch.cat(a2s).contiguous(),
            H,
            I,
            gated,
            act,
            key,
            intermediate,
            gs_inter,
            inv_inter,
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
        """The NVFP4 instruction class of each GEMM by the dense rule on the expert weight shape and the architecture family."""
        return expert_classes(
            self.hidden_size, self.intermediate_size, family, self.gated
        )

    @property
    def static(self) -> bool:
        return self.intermediate == "static"

    @property
    def chain1(self) -> W4A4Chain:
        """The GEMM1 chain of the class: the gated ``act`` on the scaled (gate, up) pair or the ungated ``act`` on the scaled
        accumulator, with the parameters; bf16 output, or the fused block quantization (``nvfp4``) of the static class."""
        return W4A4Chain(
            "glu" if self.gated else "act",
            self.act,
            "nvfp4" if self.static else "bf16",
            self.params,
            "w4a4",
            arith=self.arith,
        )

    @property
    def chain2(self) -> W4A4Chain:
        """The GEMM2 chain: ``cvt_bf16(mul(acc, mul(alpha2[e], pts2)))``, or the static class's with the per-expert ``inv2[e]``."""
        return GEMM2_STATIC_CHAIN if self.static else GEMM2_CHAIN

    def view(self) -> Dict[str, object]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for this backend (``quant`` names the pair:
        the fp4 backend's pairs share the view key)."""
        return {
            "quant": "nvfp4",
            "gemm1_weights": self.w1,
            "gemm1_weights_scale": self.w1_sf,
            "gemm1_alpha": self.w1_alpha,
            "gemm2_weights": self.w2,
            "gemm2_weights_scale": self.w2_sf,
            "gemm2_alpha": self.w2_alpha,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
            "intermediate": self.intermediate,
            "intermediate_scale": self.gs2,
            "intermediate_inv_scale": self.inv2,
            "activation_arith": self.arith,
        }

    @staticmethod
    def from_view(view: Dict[str, object]) -> "MoEWeightsNVFP4":
        gated = bool(view.get("gated", True))
        return MoEWeightsNVFP4(
            view["gemm1_weights"],  # type: ignore[arg-type]
            view["gemm1_weights_scale"],  # type: ignore[arg-type]
            view["gemm1_alpha"],  # type: ignore[arg-type]
            view["gemm2_weights"],  # type: ignore[arg-type]
            view["gemm2_weights_scale"],  # type: ignore[arg-type]
            view["gemm2_alpha"],  # type: ignore[arg-type]
            int(view["hidden_size"]),  # type: ignore[call-overload]
            int(view["intermediate_size"]),  # type: ignore[call-overload]
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),  # type: ignore[arg-type]
            str(view.get("intermediate", "per_token")),
            view.get("intermediate_scale"),  # type: ignore[arg-type]
            view.get("intermediate_inv_scale"),  # type: ignore[arg-type]
            str(view.get("activation_arith", "exact")),
        )


GEMM2_CHAIN = W4A4Chain("plain", "silu", "bf16", (), "w4a4")
#: GEMM2 of the static class: ``cvt_bf16(mul(acc, mul(alpha2[e], inv2[e])))``, the scale once per tile.
GEMM2_STATIC_CHAIN = W4A4Chain("plain", "silu", "bf16", (), "w4a4s")
F32_CHAIN = W4A4Chain(
    "plain", "silu", "f32", (), "none"
)  # the unscaled oracle: the raw fp32 fold


def grouped_gemm_nvfp4(
    a4: torch.Tensor,
    a_sf: torch.Tensor,
    a_pts: torch.Tensor,
    w4: torch.Tensor,
    w_sf: torch.Tensor,
    alpha: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: W4A4Chain,
    in_fmt: str,
    out: Optional[torch.Tensor] = None,
    dense_a: bool = False,
    out_sf: Optional[torch.Tensor] = None,
    aux: Optional[torch.Tensor] = None,
    *,
    src: Optional[torch.Tensor] = None,
    k_div: int = 1,
) -> torch.Tensor:
    """``out[p] = chain(a4[p] @ w4[e(p)].T, alpha[e(p)], pts[p])`` on the expert segments ``counts`` (int32 ``[E]``, group
    order = expert order). Padded packing (default): ``a4 [rows, K/2]`` uint8 with every group's rows starting at a multiple
    of 128 (``rows`` a multiple of 128), ``a_sf`` its 128x4 swizzled scales, ``a_pts [rows]`` f32. Dense packing (``dense_a``):
    ``a4 [rows, K/2]`` the groups' rows back to back (any row count), ``a_sf`` and ``a_pts`` at the padded positions
    (``a_pts`` of the padded row count), ``out [rows]`` dense. ``w4 [E, N, K/2]`` uint8, ``w_sf [E, sf_bytes(N, K)]``,
    ``alpha [E]`` f32. ``chain``: the W4A4 GEMM1 / GEMM2 chains or the unscaled oracle (``scale == "none"``: ``alpha`` and
    ``pts`` ignored). The record selects the body: a dense record runs the padded body on either packing, a swapped record
    (``config.swap_ab``: 128 weight rows x 8 / 16 / 32 tokens, the weights on the MMA's M side, :mod:`.blocks.swap`) runs
    the swapped decode body on the dense packing (``a4`` carries at least the token tile's rows past the live rows: the op's
    ``DENSE_TAIL``). The persistent grid is a launch argument: the device's co-resident cluster capacity for the record. The
    static class: GEMM1 with ``chain.out_fmt == "nvfp4"`` writes the packed codes to ``out`` (uint8 ``[rows, N_out/2]``) and the
    scale bytes to ``out_sf`` (the flat 128x4 buffer of the (padded) rows) with ``aux = gs2 [E]``; GEMM2 with ``chain.scale ==
    "w4a4s"`` takes ``aux = inv2 [E]`` (``a_pts`` unread). The gather record (``config.gather_a``: a swapped record whose
    body reads the token rows itself): ``a4 [T, K/2]`` the unpermuted token rows, ``a_sf`` their scale factors (the flat 128x4
    buffer of the ``T`` rows, or the linear ``[T, K/16]`` bytes), ``a_pts [T]`` (or one entry: the unit scale), ``src [>= Tk]``
    int32 the plan's slot of every dense row (``src[p] // k_div`` its token), ``out [rows, n_out]`` given (it sizes the dense
    rows); no permuted tensor, the same bits."""
    from .blocks.compile import (
        can_implement_grouped_blockscaled,
        can_implement_swapped_blockscaled,
        get_compiled_grouped,
        get_compiled_swapped,
    )

    from .api import _check_grouped_chain, _check_grouped_operands, _grouped_out

    if config.gather_a:
        return _grouped_gemm_nvfp4_gather(
            a4,
            a_sf,
            a_pts,
            w4,
            w_sf,
            alpha,
            counts,
            config,
            chain,
            in_fmt,
            out,
            out_sf,
            aux,
            src,
            k_div,
        )
    if src is not None:
        raise ValueError(
            "grouped_gemm_nvfp4: src belongs to the gather record (config.gather_a)"
        )
    # the shared operand rules at the packed fp4 formats' alignment (16-byte rows of codes: whole scale-factor column groups);
    # the weight rows N in whole 8-row units (the 16-bit output's rows), the expert count under the group tables' bound
    rows, Kh, E, N, _ = _check_grouped_operands(
        "grouped_gemm_nvfp4",
        a4,
        w4,
        counts,
        a4.shape[0] if a4.dim() == 2 else None,
        None,
        torch.uint8,
        ALIGN_NVFP4 // 2,
        "packed fp4 uint8",
        n_align=8,
    )
    K = 2 * Kh
    rows_sf = a_pts.numel()
    if dense_a:
        if rows_sf % GROUP_ALIGN or rows_sf < rows - DENSE_TAIL:
            raise ValueError(
                "grouped_gemm_nvfp4: dense A takes the scale factors and row scales at the padded positions"
            )
    elif rows % GROUP_ALIGN or rows_sf != rows:
        raise ValueError(
            "grouped_gemm_nvfp4: the padded rows are a multiple of the 128-row scale-factor atom"
        )
    if a_sf.dtype != torch.uint8 or a_sf.numel() < sf_numel(rows_sf, K):
        raise ValueError(
            "grouped_gemm_nvfp4: a_sf must hold the 128x4 swizzled scales of the (padded) rows"
        )
    if w_sf.dtype != torch.uint8 or w_sf.numel() < E * sf_numel(N, K):
        raise ValueError(
            "grouped_gemm_nvfp4: w_sf must hold E consecutive 128x4 swizzled scale buffers of (N, K)"
        )
    if (
        alpha.dtype != torch.float32
        or alpha.numel() != E
        or a_pts.dtype != torch.float32
    ):
        raise ValueError("grouped_gemm_nvfp4: alpha [E] and pts must be float32")
    if in_fmt not in NVFP4_FMTS or chunks_for(N, K, in_fmt) != 1:
        raise ValueError(
            "grouped_gemm_nvfp4: the grouped class takes unsplit expert weight shapes"
        )
    if config.swap_ab:
        if not dense_a:
            raise ValueError(
                "grouped_gemm_nvfp4: the swapped decode body reads the dense packing (dense_a=True)"
            )
        if not can_implement_swapped_blockscaled(config, N, K, E, chain, in_fmt):
            raise ValueError(
                f"grouped_gemm_nvfp4: swapped record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
            )
    elif not can_implement_grouped_blockscaled(config, N, K, E, chain, in_fmt):
        raise ValueError(
            f"grouped_gemm_nvfp4: record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
        )
    # the chain's shape rules on the host before any record is consulted: whole gate / up pairs, the output width in whole
    # 16-byte rows of the output format (whole scale blocks for the packed codes)
    n_out = _check_grouped_chain("grouped_gemm_nvfp4", N, chain)
    static = chain.needs_aux
    if chain.static_quant:
        n_out //= 2  # the packed codes: two per byte
        out_dtype = torch.uint8
        if (
            out_sf is None
            or out_sf.dtype != torch.uint8
            or out_sf.numel() < sf_numel(rows_sf, N // 2 if chain.pair else N)
        ):
            raise ValueError(
                "grouped_gemm_nvfp4: the static class's GEMM1 takes out_sf, the flat 128x4 scale bytes of the (padded) rows"
            )
    else:
        out_dtype = _OUT[chain.out_fmt]
    if static:
        if aux is None or aux.dtype != torch.float32 or aux.numel() != E:
            raise ValueError(
                "grouped_gemm_nvfp4: the static class's chains take aux, the per-expert static scales [E] (gs2 for GEMM1, inv2 for GEMM2)"
            )
        if out_sf is None:
            out_sf = _dummy_sf(
                a4.device
            )  # GEMM2 of the static class: the scale-byte operand is unread
    if out is None:
        out = torch.zeros(rows, n_out, dtype=out_dtype, device=a4.device)
    out = _grouped_out("grouped_gemm_nvfp4", out, rows, n_out, out_dtype, a4.device)
    if rows == 0:
        return out
    # the persistent grid clipped to the launch's tile bound (a placement: which cluster computes a tile, never the bits)
    grid = min(
        persistent_clusters(a4.device, config), launch_tile_bound(config, rows, E, N)
    )
    if config.swap_ab:
        kernel = get_compiled_swapped(config, chain.at(in_fmt))
        if static:
            kernel(
                a4.unsqueeze(-1),
                a_sf.reshape(-1),
                w4.permute(1, 2, 0),
                w_sf.reshape(-1),
                out,
                out_sf.reshape(-1),
                counts,
                alpha.reshape(-1).contiguous(),
                a_pts.reshape(-1).contiguous(),
                aux.reshape(-1).contiguous(),
                grid,
            )
        else:
            kernel(
                a4.unsqueeze(-1),
                a_sf.reshape(-1),
                w4.permute(1, 2, 0),
                w_sf.reshape(-1),
                out,
                counts,
                alpha.reshape(-1).contiguous(),
                a_pts.reshape(-1).contiguous(),
                grid,
            )
        return out
    kernel = get_compiled_grouped(
        config, chain.at(in_fmt), packing="dense" if dense_a else "padded"
    )
    if static:
        kernel(
            a4.unsqueeze(-1),
            w4.permute(1, 2, 0),
            a_sf.reshape(-1),
            w_sf.reshape(-1),
            out,
            out_sf.reshape(-1),
            counts,
            alpha.reshape(-1).contiguous(),
            a_pts.reshape(-1).contiguous(),
            aux.reshape(-1).contiguous(),
            grid,
        )
    else:
        kernel(
            a4.unsqueeze(-1),
            w4.permute(1, 2, 0),
            a_sf.reshape(-1),
            w_sf.reshape(-1),
            out,
            counts,
            alpha.reshape(-1).contiguous(),
            a_pts.reshape(-1).contiguous(),
            grid,
        )
    return out


def gather_sf_layout(a_sf: torch.Tensor, T: int, K: int, sf_vec: int = 16) -> str:
    """The layout of a gather record's scale operand: the row-major linear bytes ``[T, K / sf_vec]`` or the flat 128x4 buffer."""
    return (
        "linear"
        if a_sf.dim() == 2 and tuple(a_sf.shape) == (T, K // sf_vec)
        else "128x4"
    )


def gather_form_ok(K: int, sf_src: str, sf_vec: int = 16) -> bool:
    """The gather form's host rule on the scale source: the linear layout's rows are whole 4-byte groups (the kernel copies a
    row's scales in 4-byte groups), the 128x4 buffer always."""
    return sf_src == "128x4" or (K // sf_vec) % 4 == 0


def _grouped_gemm_nvfp4_gather(
    a4,
    a_sf,
    a_pts,
    w4,
    w_sf,
    alpha,
    counts,
    config,
    chain,
    in_fmt,
    out,
    out_sf,
    aux,
    src,
    k_div,
) -> torch.Tensor:
    """The gather record of :func:`grouped_gemm_nvfp4` (its operand rules, the token rows in place of the permuted rows; the
    swapped body)."""
    from .api import _check_grouped_chain, _check_grouped_operands, _grouped_out
    from .blocks.compile import can_implement_swapped_blockscaled, get_compiled_swapped

    T, Kh, E, N, _ = _check_grouped_operands(
        "grouped_gemm_nvfp4",
        a4,
        w4,
        counts,
        a4.shape[0] if a4.dim() == 2 else None,
        None,
        torch.uint8,
        ALIGN_NVFP4 // 2,
        "packed fp4 uint8",
        n_align=8,
    )
    K = 2 * Kh
    if out is None or src is None:
        raise ValueError(
            "grouped_gemm_nvfp4: the gather record reads the token rows through the plan's map: src and out"
        )
    if (
        src.dtype != torch.int32
        or src.dim() != 1
        or not src.is_contiguous()
        or k_div < 1
    ):
        raise ValueError(
            "grouped_gemm_nvfp4: src must be a contiguous int32 [>= Tk] and k_div >= 1"
        )
    sf_src = gather_sf_layout(a_sf, T, K)
    if a_sf.dtype != torch.uint8 or (
        sf_src == "128x4" and a_sf.numel() < sf_numel(T, K)
    ):
        raise ValueError(
            "grouped_gemm_nvfp4: a_sf must hold the 128x4 swizzled scales of the T token rows or their linear [T, K/16] bytes"
        )
    if not gather_form_ok(K, sf_src):
        raise ValueError(
            "grouped_gemm_nvfp4: the linear scale layout's rows must be whole 4-byte groups (K a multiple of 64)"
        )
    if w_sf.dtype != torch.uint8 or w_sf.numel() < E * sf_numel(N, K):
        raise ValueError(
            "grouped_gemm_nvfp4: w_sf must hold E consecutive 128x4 swizzled scale buffers of (N, K)"
        )
    if (
        alpha.dtype != torch.float32
        or alpha.numel() != E
        or a_pts.dtype != torch.float32
        or a_pts.numel() not in (1, T)
    ):
        raise ValueError(
            "grouped_gemm_nvfp4: alpha [E] and pts [T] (or one unit entry) must be float32"
        )
    if in_fmt not in NVFP4_FMTS or chunks_for(N, K, in_fmt) != 1:
        raise ValueError(
            "grouped_gemm_nvfp4: the grouped class takes unsplit expert weight shapes"
        )
    if not config.swap_ab or not can_implement_swapped_blockscaled(
        config, N, K, E, chain, in_fmt
    ):
        raise ValueError(
            f"grouped_gemm_nvfp4: gather record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
        )
    n_out = _check_grouped_chain("grouped_gemm_nvfp4", N, chain)
    static = chain.needs_aux
    rows = out.shape[0]
    if chain.static_quant:
        n_out //= 2
        out_dtype = torch.uint8
        if out_sf is None or out_sf.dtype != torch.uint8:
            raise ValueError(
                "grouped_gemm_nvfp4: the static class's GEMM1 takes out_sf, the flat 128x4 scale bytes of the padded rows"
            )
    else:
        out_dtype = _OUT[chain.out_fmt]
    if static:
        if aux is None or aux.dtype != torch.float32 or aux.numel() != E:
            raise ValueError(
                "grouped_gemm_nvfp4: the static class's chains take aux, the per-expert static scales [E]"
            )
        if out_sf is None:
            out_sf = _dummy_sf(a4.device)
    out = _grouped_out("grouped_gemm_nvfp4", out, rows, n_out, out_dtype, a4.device)
    if rows == 0 or T == 0:
        return out
    grid = min(
        persistent_clusters(a4.device, config), launch_tile_bound(config, rows, E, N)
    )
    kernel = get_compiled_swapped(config, chain.at(in_fmt), sf_src=sf_src)
    if static:
        kernel(
            a4.unsqueeze(-1),
            a_sf.reshape(-1),
            src,
            k_div,
            w4.permute(1, 2, 0),
            w_sf.reshape(-1),
            out,
            out_sf.reshape(-1),
            counts,
            alpha.reshape(-1).contiguous(),
            a_pts.reshape(-1).contiguous(),
            aux.reshape(-1).contiguous(),
            grid,
        )
    else:
        kernel(
            a4.unsqueeze(-1),
            a_sf.reshape(-1),
            src,
            k_div,
            w4.permute(1, 2, 0),
            w_sf.reshape(-1),
            out,
            counts,
            alpha.reshape(-1).contiguous(),
            a_pts.reshape(-1).contiguous(),
            grid,
        )
    return out


def padded_rows_bound(T: int, k: int, E: int, align: int = GROUP_ALIGN) -> int:
    """An upper bound of the padded expanded row count ``sum_e roundup(counts[e], align)`` that does not read the device
    (CUDA-graph safe): every active expert pads at most ``align - 1`` rows."""
    n = T * k + (align - 1) * min(E, T * k)
    return -(-n // align) * align


def permute_plan_dual(
    experts: torch.Tensor, E: int, pad_align: int = GROUP_ALIGN
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(counts [E], dest [T, k], src [Tk], dest_pad [T, k], pos_pad [Tk])``: the dense plan (:func:`.api.permute_plan`)
    plus the positions with the groups padded to ``pad_align`` rows (``dest_pad[i]``) and the padded position of every dense
    row (``pos_pad[p]``): the dense packing's row map (A rows dense, scale factors and row scales at the padded positions)."""
    from .api import PLAN_MULTI_MIN_N
    from .permute import (
        PLAN_BLOCK,
        get_compiled_plan,
        get_compiled_plan_multi,
        plan_threads,
    )

    T, k = experts.shape
    n = T * k
    dev = experts.device
    dest = torch.empty(n, dtype=torch.int32, device=dev)
    src = torch.empty(n, dtype=torch.int32, device=dev)
    dest_pad = torch.empty(n, dtype=torch.int32, device=dev)
    pos_pad = torch.empty(n, dtype=torch.int32, device=dev)
    counts = torch.empty(E, dtype=torch.int32, device=dev)
    flat = experts.reshape(-1)
    if not flat.is_contiguous() or flat.dtype != torch.int32:
        raise TypeError("permute_plan_dual: experts must be a contiguous int32 [T, k]")
    if n == 0:
        counts.zero_()
    elif n >= PLAN_MULTI_MIN_N:
        nb = (n + PLAN_BLOCK - 1) // PLAN_BLOCK
        hist_blk = torch.empty(nb * E, dtype=torch.int32, device=dev)
        get_compiled_plan_multi(1, True, pad_align)(
            flat, counts, dest, src, dest_pad, pos_pad, hist_blk, n, E, nb
        )
    else:
        get_compiled_plan(1, True, pad_align, plan_threads(n))(
            flat, counts, dest, src, dest_pad, pos_pad, n, E
        )
    return counts, dest.view(T, k), src, dest_pad.view(T, k), pos_pad


_GS_INV: Dict[
    Tuple[str, float], torch.Tensor
] = {}  # the recipe's constant per device, filled once (no fill launch per call)
_DUMMY_SF: Dict[str, torch.Tensor] = {}


def _dummy_sf(device: torch.device) -> torch.Tensor:
    """A 512-byte placeholder for an unread scale-byte operand (one atom: the kernels' alignment assumption), per device."""
    key = str(torch.device(device))
    t = _DUMMY_SF.get(key)
    if t is None:
        t = _DUMMY_SF[key] = torch.zeros(512, dtype=torch.uint8, device=device)
    return t


def route_plan_dual(
    logits: torch.Tensor,
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    pad_align: int = GROUP_ALIGN,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """:func:`.api.route` and :func:`permute_plan_dual` in one launch for small token counts: the shared fused kernel's dual form
    (:func:`.api.route_plan` with ``dual=True``): ``(experts [T, k] int32, weights [T, k] f32, counts [E] int32, dest [T, k] int32,
    src [T k] int32, dest_pad [T, k] int32, pos_pad [T k] int32)``, the same bits and integers as the two calls."""
    from .api import route_plan

    return route_plan(
        logits, top_k, method, bias, scaling_factor, dual=True, pad_align=pad_align
    )


def _gs_inv(gs_inv: Union[float, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(gs_inv, torch.Tensor):
        return gs_inv.to(device=device, dtype=torch.float32).reshape(1)
    key = (str(torch.device(device)), float(gs_inv))
    t = _GS_INV.get(key)
    if t is None:
        t = _GS_INV[key] = torch.full(
            (1,), float(gs_inv), dtype=torch.float32, device=device
        )
    return t


def quantize_activations(
    x: torch.Tensor, gs_inv: Union[float, torch.Tensor] = TE_GS_INV, out=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The class's per-token NVFP4 quantization of bf16 rows ``[M, K]``: ``(x4 [M, K/2] uint8, sf flat 128x4 uint8,
    pts [M] f32)``; ``gs_inv`` the recipe's inverse base scale. ``out = (x4, sf)`` preallocated buffers."""
    from ...batch_invariant.quantization.nvfp4_api import nvfp4_quantize_rows

    M, K = x.shape
    dev = x.device
    if M == 0:
        return (
            torch.empty(0, K // 2, dtype=torch.uint8, device=dev),
            torch.zeros(sf_numel(0, K), dtype=torch.uint8, device=dev),
            torch.empty(0, dtype=torch.float32, device=dev),
        )
    x4, sf, pts = nvfp4_quantize_rows(
        x, _gs_inv(gs_inv, dev), "128x4", per_token=True, out=out, enable_pdl=True
    )
    return x4, sf.reshape(-1), pts


def quantize_rows_mapped(
    x: torch.Tensor,
    gs_inv: Union[float, torch.Tensor],
    row_map: torch.Tensor,
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The class's per-token NVFP4 quantization of bf16 rows ``[M, K]`` with the scale factors and per-row scales stored at
    row ``row_map[r]`` of the padded buffers ``out = (x4 [M, K/2], sf_pad flat 128x4, pts_pad)`` while the codes stay at row
    ``r``: the quantizer's kernel term for term (:mod:`.blocks.quant_nvfp4`), one launch in place of the quantizer and the
    scale re-tiling."""
    from ...batch_invariant.quantization.mx_api import device_sm_count
    from ...batch_invariant.quantization.nvfp4_schedule import default_config
    from .blocks.quant_nvfp4 import get_compiled_rowmap_quant

    M, K = x.shape
    x4, sf_pad, pts_pad = out
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0
    assert x4.shape == (M, K // 2) and x4.is_contiguous() and x4.dtype == torch.uint8
    assert (
        row_map.dtype == torch.int32
        and row_map.is_contiguous()
        and row_map.numel() >= M
    )
    assert sf_pad.dtype == torch.uint8 and pts_pad.dtype == torch.float32
    if M == 0:
        return
    cfg = default_config(
        K, M, x.element_size(), True, device_sm_count(x.device), "128x4"
    )
    kernel = get_compiled_rowmap_quant("bf16", K, cfg, True)
    kernel(x, x4, sf_pad.view(-1), pts_pad, _gs_inv(gs_inv, x.device), row_map, M, K)


def quantize_scatter_rows(
    x: torch.Tensor,
    gs_inv: Union[float, torch.Tensor],
    dest: torch.Tensor,
    dest_pad: torch.Tensor,
    k: int,
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The class's per-token NVFP4 quantization of bf16 rows ``[T, K]`` stored into the grouped layout: token ``t``'s codes at
    the dense rows ``dest[t k + j]`` of ``out[0]`` (``[rows, K/2]``), its scale factors and per-token scale at the padded
    positions ``dest_pad[t k + j]`` of ``out[1]`` (flat 128x4) and ``out[2]``: the quantizer's kernel term for term
    (:mod:`.blocks.quant_nvfp4`), one launch in place of the quantizer and the fp4 scatter."""
    from ...batch_invariant.quantization.mx_api import device_sm_count
    from ...batch_invariant.quantization.nvfp4_schedule import default_config
    from .blocks.quant_nvfp4 import get_compiled_scatter_quant

    T, K = x.shape
    x_perm4, sf_pad, pts_pad = out
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0
    assert (
        x_perm4.dtype == torch.uint8
        and x_perm4.is_contiguous()
        and x_perm4.shape[1] == K // 2
    )
    assert (
        dest.dtype == torch.int32
        and dest_pad.dtype == torch.int32
        and dest.numel() >= T * k
    )
    assert dest.is_contiguous() and dest_pad.is_contiguous()
    assert sf_pad.dtype == torch.uint8 and pts_pad.dtype == torch.float32
    if T == 0:
        return
    cfg = default_config(
        K, T, x.element_size(), True, device_sm_count(x.device), "128x4"
    )
    kernel = get_compiled_scatter_quant("bf16", K, cfg, k, True)
    kernel(
        x,
        x_perm4,
        sf_pad.view(-1),
        pts_pad,
        _gs_inv(gs_inv, x.device),
        dest,
        dest_pad,
        T,
        K,
    )


def _part(device: torch.device) -> Tuple[int, str]:
    return part_of(device)


def fused_moe_nvfp4(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: MoEWeightsNVFP4,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    in_fmts: Optional[Tuple[str, str]] = None,
    path: str = "dense",
    gs_inv: Union[float, torch.Tensor] = TE_GS_INV,
    return_intermediates: bool = False,
    swap: Optional[bool] = None,
    plan: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    round_scales: bool = False,
    gather: Optional[bool] = None,
):
    """The W4A4 MoE FFN for pre-routed tokens: ``x`` bf16 ``[T, H]`` (quantized here per token with the class and
    ``gs_inv``) or the quantized triple ``(x4 [T, H/2] uint8, x_sf (128x4 swizzled, uint8), x_pts [T] f32)``;
    ``token_selected_experts [T, k]`` int32, ``token_final_scales [T, k]`` f32 (``round_scales``: the combine takes their
    bf16 rounding, the pre-routed packed-word convention); the output bf16 ``[T, H]`` in token order.
    ``in_fmts`` names the instruction class per GEMM (default: the dense rule on each expert weight shape and the device's
    family); ``config1`` / ``config2`` the grouped records (any record of the space gives the same bits; the dispatcher's pick
    by default); ``path`` the packing of the expanded rows (``"dense"`` / ``"padded"``: placements); ``swap`` the dense path's
    body -- None: the dispatcher's ranking (the swapped decode body's records ranked with the grouped records by the dense
    estimate at the grouped shape), True / False: the swapped / the padded body on both GEMMs (placements: the same bits); explicit records select the
    body themselves (a swapped record needs the dense path). ``plan``: the dense packing's precomputed plan, the outputs of
    :func:`route_plan_dual` after the ids and weights (``counts, dest, src, dest_pad, pos_pad``: the same integers as
    :func:`permute_plan_dual`'s, one launch saved). ``gather``: GEMM1's swapped records in their gather form (the body reads
    the token rows through the plan's map: no scatter launch, no permuted tensor; the dense path) -- False excludes it, True
    forces it, None (the default) applies the structural rule (:func:`~.schedule_nvfp4.gather_form_rule`: the narrow swapped
    tile's gather form when the box form would scatter a pre-quantized input in a launch of its own); an explicit ``config1``
    selects the form itself (``gather_a``)."""
    from .combine import get_compiled_combine, warps_per_token
    from .permute import warps_per_row
    from .permute_nvfp4 import (
        get_compiled_scatter_fp4,
        get_compiled_scatter_sf,
        scatter_lanes,
        scatter_sf_lanes,
    )
    from .schedule_nvfp4 import default_configs_nvfp4, gather_form_rule

    if path not in PATHS:
        raise ValueError(f"fused_moe_nvfp4: path must be one of {PATHS}, not {path!r}")
    H, I, E = w.hidden_size, w.intermediate_size, w.num_experts
    if isinstance(x, torch.Tensor):
        if (
            x.dtype != torch.bfloat16
            or x.dim() != 2
            or x.stride(1) != 1
            or x.stride(0) % 8
        ):
            raise TypeError(
                "fused_moe_nvfp4: x must be a row-major, 16-byte-aligned bf16 [T, H] or a quantized triple"
            )
        if x.shape[1] != H:
            raise ValueError(
                f"fused_moe_nvfp4: x has H = {x.shape[1]}, the weights H = {H}"
            )
        T = x.shape[0]
        dev = x.device
        x_rows: Optional[torch.Tensor] = x.contiguous()
        sf_src, unit_pts = "128x4", False
        if path == "dense" and INPUT_SCATTER == "fused":
            x4 = x_sf = x_pts = None  # quantized straight into the grouped layout below
        else:
            x4, x_sf, x_pts = quantize_activations(x, gs_inv)
    else:
        x_rows = None
        x4, x_sf, x_pts = x
        T = x4.shape[0]
        dev = x4.device
        # the pack's scale factors: the class's flat 128x4 buffer, or the row-major ``[T, H / 16]`` bytes of the per-tensor pack
        # of the TRTLLM backends (the scatter assembles the atoms from them: no re-tiling pass); ``x_pts = None``: unit per-token
        # scales (the per-tensor pack: nothing to read, no fill)
        sf_src = (
            "linear"
            if x_sf.dim() == 2 and tuple(x_sf.shape) == (T, H // 16)
            else "128x4"
        )
        x_sf = x_sf.contiguous().reshape(-1)
        unit_pts = x_pts is None
        if (
            x4.dtype != torch.uint8
            or tuple(x4.shape) != (T, H // 2)
            or not x4.is_contiguous()
            or x_sf.dtype != torch.uint8
            or (sf_src == "128x4" and x_sf.numel() < sf_numel(T, H))
            or (
                not unit_pts
                and (tuple(x_pts.shape) != (T,) or x_pts.dtype != torch.float32)
            )
        ):
            raise TypeError(
                "fused_moe_nvfp4: the quantized triple is (x4 [T, H/2] uint8, x_sf 128x4 uint8 of >= sf_numel(T, H) bytes or the linear [T, H/16] bytes, x_pts [T] f32 or None)"
            )
        if unit_pts:
            x_pts = _gs_inv(
                1.0, dev
            )  # a placeholder the unit-scale scatter never reads
    k = token_selected_experts.shape[1]
    check_shape_nvfp4(H, I, E, k, w.gated)
    if (
        tuple(token_selected_experts.shape) != (T, k)
        or token_selected_experts.dtype != torch.int32
    ):
        raise TypeError(
            "fused_moe_nvfp4: token_selected_experts must be an int32 [T, k]"
        )
    if (
        tuple(token_final_scales.shape) != (T, k)
        or token_final_scales.dtype != torch.float32
    ):
        raise TypeError("fused_moe_nvfp4: token_final_scales must be a float32 [T, k]")
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
            "fused_moe_nvfp4: out must be a row-major, 16-byte-aligned bf16 [T, H]"
        )
    sm_count, family = _part(dev)
    in1, in2 = in_fmts if in_fmts is not None else w.classes(family)
    chain1, chain2 = w.chain1, w.chain2
    static = w.static
    if config1 is None or config2 is None:
        ranked1 = config1 is None
        c1, c2 = default_configs_nvfp4(
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
            # the rule: a standalone scatter launch (a pre-quantized input) folded into the narrow swapped tile's gather form
            config1 = gather_form_rule(config1, x4 is not None)
    if path != "dense" and (swap or config1.swap_ab or config2.swap_ab):
        raise ValueError(
            "fused_moe_nvfp4: the swapped decode body reads the dense packing (path='dense')"
        )
    if config2.gather_a or (path != "dense" and (gather or config1.gather_a)):
        raise ValueError(
            "fused_moe_nvfp4: the gather form is GEMM1's on the dense packing (path='dense')"
        )
    gather1 = config1.gather_a
    if gather1 and not gather_form_ok(H, "linear" if x_rows is None else "128x4"):
        # the linear scale layout's rows are not whole 4-byte groups: the box form through the scatter
        config1 = dataclasses.replace(config1, gather_a=False)
        gather1 = False
    if path != "dense" and plan is not None:
        raise ValueError(
            "fused_moe_nvfp4: the precomputed plan is the dense packing's (path='dense')"
        )
    Tk = T * k
    Tk_pad = padded_rows_bound(T, k, E)
    pcols_h, pcols_i = padded_sf_cols(H), padded_sf_cols(I)
    gsi = _gs_inv(gs_inv, dev)
    if path == "dense":
        counts, dest, src, dest_pad, pos_pad = (
            plan if plan is not None else permute_plan_dual(experts, E)
        )  # ``plan``: ``route_plan_dual``'s outputs, the same integers
        # the fp4 rows back to back (a tail past Tk for the last tiles' TMA boxes); the scale factors and the row scales at
        # the padded positions (one 128x4 atom per (expert, 128-row tile); a tail past Tk_pad likewise)
        Tk_a, Tk_sf = Tk + DENSE_TAIL, Tk_pad + DENSE_TAIL
        x_perm4 = x_perm_sf = x_perm_pts = None
        if gather1:
            # the gather form: GEMM1 reads the token rows through the plan's map; a bf16 input is quantized over its T rows
            if x4 is None:
                x4, x_sf, x_pts = quantize_activations(x_rows, gsi)
            g_a, g_pts = x4, x_pts
            g_sf = x_sf.view(T, H // 16) if sf_src == "linear" else x_sf
        else:
            x_perm4 = torch.empty(Tk_a, H // 2, dtype=torch.uint8, device=dev)
            x_perm_sf = torch.empty(sf_numel(Tk_sf, H), dtype=torch.uint8, device=dev)
            x_perm_pts = torch.empty(Tk_sf, dtype=torch.float32, device=dev)
            if Tk and x4 is None:
                quantize_scatter_rows(
                    x_rows,
                    gsi,
                    dest.reshape(-1),
                    dest_pad.reshape(-1),
                    k,
                    (x_perm4, x_perm_sf, x_perm_pts),
                )
            elif Tk:
                wpr = warps_per_row(Tk, H // 32, sm_count)
                get_compiled_scatter_fp4(
                    k, wpr, sf_src, unit_pts, **scatter_lanes(H // 2, pcols_h, wpr)
                )(
                    x4,
                    x_sf,
                    x_pts,
                    dest.reshape(-1),
                    dest_pad.reshape(-1),
                    x_perm4,
                    x_perm_sf,
                    x_perm_pts,
                    Tk,
                    H // 2,
                    pcols_h,
                )
            g_a, g_sf, g_pts = x_perm4, x_perm_sf, x_perm_pts
        g_src = src if gather1 else None
        a4 = torch.empty(Tk_a, I // 2, dtype=torch.uint8, device=dev)
        a_sf_pad = torch.empty(sf_numel(Tk_sf, I), dtype=torch.uint8, device=dev)
        a = a_sf = a_pts = None
        if static:
            # the static class: GEMM1's epilogue writes the intermediate's codes (dense rows) and scale bytes (the padded
            # positions) itself -- no bf16 intermediate, no quantizer launch; GEMM2's scale is mul(alpha2[e], inv2[e])
            grouped_gemm_nvfp4(
                g_a,
                g_sf,
                g_pts,
                w.w1,
                w.w1_sf,
                w.w1_alpha,
                counts,
                config1,
                chain1,
                in1,
                out=a4,
                dense_a=True,
                out_sf=a_sf_pad,
                aux=w.gs2,
                src=g_src,
                k_div=k,
            )
            a_pts_pad = (
                x_perm_pts  # GEMM2's row-scale operand is unread by the static chain
                if x_perm_pts is not None
                else torch.empty(Tk_sf, dtype=torch.float32, device=dev)
            )
        else:
            a = torch.empty(Tk_a, I, dtype=torch.bfloat16, device=dev)
            grouped_gemm_nvfp4(
                g_a,
                g_sf,
                g_pts,
                w.w1,
                w.w1_sf,
                w.w1_alpha,
                counts,
                config1,
                chain1,
                in1,
                out=a,
                dense_a=True,
                src=g_src,
                k_div=k,
            )
            # the class's second pass over the live rows: the codes stay dense; the scale factors and the row scales are re-tiled
            # to the padded positions for GEMM2's atom-aligned scale boxes (a byte move)
            a_pts_pad = torch.empty(Tk_sf, dtype=torch.float32, device=dev)
            if Tk and INTERMEDIATE_RETILE == "mapped":
                quantize_rows_mapped(
                    a[:Tk], gsi, pos_pad, (a4[:Tk], a_sf_pad, a_pts_pad)
                )
            elif Tk:
                a_sf = torch.empty(sf_numel(Tk, I), dtype=torch.uint8, device=dev)
                _, _, a_pts = quantize_activations(a[:Tk], gsi, out=(a4[:Tk], a_sf))
                get_compiled_scatter_sf(1, spl=scatter_sf_lanes(pcols_i))(
                    a_sf, a_pts, pos_pad, a_sf_pad, a_pts_pad, Tk, pcols_i
                )
        y = torch.empty(Tk_a, H, dtype=torch.bfloat16, device=dev)
        grouped_gemm_nvfp4(
            a4,
            a_sf_pad,
            a_pts_pad,
            w.w2,
            w.w2_sf,
            w.w2_alpha,
            counts,
            config2,
            chain2,
            in2,
            out=y,
            dense_a=True,
            aux=w.inv2 if static else None,
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
                x_pts=x_pts,
                x_perm4=None if x_perm4 is None else x_perm4[:Tk],
                x_perm_sf=None
                if x_perm_sf is None
                else x_perm_sf[: sf_numel(Tk_pad, H)],
                x_perm_pts=None if x_perm_pts is None else x_perm_pts[:Tk_pad],
                a=None if static else a[:Tk],
                a4=a4[:Tk],
                a_sf=a_sf,
                a_pts=a_pts,
                a_sf_pad=a_sf_pad[: sf_numel(Tk_pad, I)],
                a_pts_pad=a_pts_pad[:Tk_pad],
                y=y[:Tk],
                config1=config1,
                config2=config2,
                in_fmts=(in1, in2),
                Tk_pad=Tk_pad,
                path=path,
            )
        return out
    # the padded packing: every group's rows start at a 128-row atom; the padding rows are zero (computed, never stored)
    from .permute import (
        PLAN_BLOCK,
        get_compiled_plan,
        get_compiled_plan_multi,
        plan_threads,
    )
    from .api import PLAN_MULTI_MIN_N

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
    x_perm_sf = torch.zeros(sf_numel(Tk_pad, H), dtype=torch.uint8, device=dev)
    x_perm_pts = torch.zeros(Tk_pad, dtype=torch.float32, device=dev)
    if Tk:
        flat_dest = dest.reshape(-1)
        wpr = warps_per_row(Tk, H // 32, sm_count)
        get_compiled_scatter_fp4(
            k, wpr, sf_src, unit_pts, **scatter_lanes(H // 2, pcols_h, wpr)
        )(
            x4,
            x_sf,
            x_pts,
            flat_dest,
            flat_dest,
            x_perm4,
            x_perm_sf,
            x_perm_pts,
            Tk,
            H // 2,
            pcols_h,
        )
    if static:
        # the static class on the padded packing: the codes and scale bytes at the padded rows (the group's rows start at an atom)
        a4 = torch.zeros(Tk_pad, I // 2, dtype=torch.uint8, device=dev)
        a_sf = torch.zeros(sf_numel(Tk_pad, I), dtype=torch.uint8, device=dev)
        grouped_gemm_nvfp4(
            x_perm4,
            x_perm_sf,
            x_perm_pts,
            w.w1,
            w.w1_sf,
            w.w1_alpha,
            counts,
            config1,
            chain1,
            in1,
            out=a4,
            out_sf=a_sf,
            aux=w.gs2,
        )
        a, a_pts = None, x_perm_pts
        y = grouped_gemm_nvfp4(
            a4,
            a_sf,
            a_pts,
            w.w2,
            w.w2_sf,
            w.w2_alpha,
            counts,
            config2,
            chain2,
            in2,
            aux=w.inv2,
        )
    else:
        a = grouped_gemm_nvfp4(
            x_perm4,
            x_perm_sf,
            x_perm_pts,
            w.w1,
            w.w1_sf,
            w.w1_alpha,
            counts,
            config1,
            chain1,
            in1,
        )
        a4, a_sf, a_pts = quantize_activations(
            a, gsi
        )  # the class, second pass (padding rows included: zeros)
        y = grouped_gemm_nvfp4(
            a4,
            a_sf,
            a_pts,
            w.w2,
            w.w2_sf,
            w.w2_alpha,
            counts,
            config2,
            GEMM2_CHAIN,
            in2,
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
            x4=x4,
            x_sf=x_sf,
            x_pts=x_pts,
            x_perm4=x_perm4,
            x_perm_sf=x_perm_sf,
            x_perm_pts=x_perm_pts,
            a=a,
            a4=a4,
            a_sf=a_sf,
            a_pts=a_pts,
            y=y,
            config1=config1,
            config2=config2,
            in_fmts=(in1, in2),
            Tk_pad=Tk_pad,
            path=path,
        )
    return out


def moe_ffn_nvfp4(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    router_logits: torch.Tensor,
    w: MoEWeightsNVFP4,
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
) -> torch.Tensor:
    """Routing (the bf16 classes' kernel) + the W4A4 FFN."""
    experts, weights = route(router_logits, top_k, method, bias, scaling_factor)
    return fused_moe_nvfp4(x, experts, weights, w, out, config1, config2)


__all__ = [
    "ALIGN_NVFP4",
    "BatchInvariantUnsupportedError",
    "DENSE_TAIL",
    "F32_CHAIN",
    "GEMM2_CHAIN",
    "GEMM2_STATIC_CHAIN",
    "GROUP_ALIGN",
    "MAX_TOP_K",
    "MoEWeightsNVFP4",
    "PATHS",
    "TE_GS_INV",
    "check_shape_nvfp4",
    "fp4_global_scale",
    "fused_moe_nvfp4",
    "grouped_gemm_nvfp4",
    "moe_ffn_nvfp4",
    "padded_rows_bound",
    "permute_plan_dual",
    "route_plan_dual",
    "quantize_activations",
    "sm_count_of",
]
