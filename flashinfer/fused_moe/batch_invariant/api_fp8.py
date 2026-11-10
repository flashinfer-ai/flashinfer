# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8 per-tensor MoE classes ``moe.e4m3.<act>`` (W8A8): the weight view, the static input quantization, the grouped fp8
GEMMs (placements of the dense fp8 class) and the fused FFN on pre-routed tokens.

The routing, the deterministic permutation and the slot-order combine are the bf16 classes' (:mod:`.api`). The scales follow
FlashInfer's per-tensor convention: the caller's ``hidden_states_scale_global`` and ``intermediate_scale_global`` are
multipliers (``q = cvt(mul(x, m))``), the expert weights are quantized here with one multiplier per expert and GEMM
(``448 / amax``, as :func:`flashinfer.fused_moe.prepare.prepare_trtllm_fp8_per_tensor_weights` forms it), and the class's
dequantization scales are the correctly rounded reciprocals of the multipliers, formed once here
(:mod:`.spec_fp8`). Every output bit is a function of the token's own e4m3 row, its ``k`` experts and routing weights, the
expert bytes and the scales: batch size and composition, row position, expert load and the grouped records never enter
the dataflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from ...batch_invariant.support import unsupported
from ...gemm.batch_invariant.api import persistent_clusters, sm_count_of
from ...gemm.batch_invariant.generator import Knobs
from . import spec as S
from .api import interleave_fc1, permute_plan
from .blocks.fp8_epilogue import GEMM2_CHAIN, Fp8Chain
from .blocks.scheduler import MAX_E
from .schedule import launch_tile_bound
from .reference_fp8 import dequant_scale, quantize_e4m3_static
from .spec_fp8 import E4M3_MAX

CLASS = "moe.e4m3"
ALIGN_E4M3 = 16  # e4m3 elements per 16-byte row: the TMA alignment of every operand
_OUT_DTYPE = {"bf16": torch.bfloat16, "f32": torch.float32, "e4m3": torch.float8_e4m3fn}


def weight_multipliers(w: torch.Tensor) -> torch.Tensor:
    """FlashInfer's per-expert E4M3 multipliers of ``w [E, N, K]``: ``448 / amax`` (1 for an all-zero expert), fp32."""
    amax = w.detach().to(torch.float32).abs().amax(dim=(-1, -2))
    if not bool(torch.isfinite(amax).all()):
        raise ValueError(f"{CLASS}: the expert weights must be finite")
    return torch.where(amax > 0, E4M3_MAX / amax, torch.ones_like(amax)).to(
        torch.float32
    )


def _scale_tensor(m, name: str, device: torch.device) -> torch.Tensor:
    """A per-tensor multiplier as a 1-element fp32 tensor on ``device``."""
    t = torch.as_tensor(m, dtype=torch.float32, device=device)
    if t.numel() != 1:
        raise ValueError(
            f"{CLASS}: {name} must be one fp32 value, got shape {tuple(t.shape)}"
        )
    return t.reshape(1).contiguous()


@dataclass
class MoEWeightsE4M3:
    """The W8A8 expert weights as the kernels consume them: ``w1q [E, 2I, H]`` e4m3 (gate / up interleaved; ``[E, I, H]`` as
    given for an ungated activation), ``w2q [E, H, I]`` e4m3; the dequantization scales ``s_w1 [E]``, ``s_w2 [E]``, ``s_x [1]``,
    ``s_inter [1]`` (fp32); the multipliers ``m_x``, ``m_inter``, ``m_w1``, ``m_w2`` as given / formed (``m_inter`` is GEMM1's
    requantization multiplier); the GEMM1 chain ``act`` with its ``params`` and gating."""

    w1q: torch.Tensor
    w2q: torch.Tensor
    s_w1: torch.Tensor
    s_w2: torch.Tensor
    s_x: torch.Tensor
    s_inter: torch.Tensor
    m_inter: torch.Tensor
    gated: bool = True
    act: str = "silu"
    params: Tuple[Optional[float], ...] = ()
    m_x: Optional[torch.Tensor] = None
    m_w1: Optional[torch.Tensor] = None
    m_w2: Optional[torch.Tensor] = None

    @staticmethod
    def prepare(
        fc1: torch.Tensor,
        fc2: torch.Tensor,
        hidden_states_scale_global,
        intermediate_scale_global,
        gated: bool = True,
        act: Optional[str] = None,
        params: Sequence[Optional[float]] = (),
        interleaved: bool = False,
    ) -> "MoEWeightsE4M3":
        """Quantize the bf16 experts ``fc1 [E, 2I, H]`` (rows ``[up, gate]``; ``[E, I, H]`` ungated) and ``fc2 [E, H, I]`` to e4m3
        with one multiplier per expert and GEMM (``cvt_e4m3(mul(w, 448 / amax))``), interleave the gate / up rows, and form the
        class's dequantization scales (``div(1, m)`` of every multiplier)."""
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
                f"{CLASS}: the classes take bf16 expert weights fc1 [E, N1, H], fc2 [E, H, I]"
            )
        E, N1, H = fc1.shape
        I = N1 // 2 if gated else N1
        if tuple(fc2.shape) != (E, H, I):
            raise ValueError(
                f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)} (gated={gated})"
            )
        check_shape_e4m3(H, I, E)
        dev = fc1.device
        w1i = fc1.contiguous() if (interleaved or not gated) else interleave_fc1(fc1)
        m_w1, m_w2 = weight_multipliers(w1i), weight_multipliers(fc2)
        # per expert: the vectorised rounding holds several fp32-sized temporaries, so a whole [E, N, K] at once would need
        # a multiple of the bf16 weights' memory
        w1q = torch.stack(
            [quantize_e4m3_static(w1i[e], m_w1[e]) for e in range(E)]
        ).contiguous()
        w2q = torch.stack(
            [quantize_e4m3_static(fc2[e], m_w2[e]) for e in range(E)]
        ).contiguous()
        m_x = _scale_tensor(
            hidden_states_scale_global, "hidden_states_scale_global", dev
        )
        m_inter = _scale_tensor(
            intermediate_scale_global, "intermediate_scale_global", dev
        )
        return MoEWeightsE4M3(
            w1q,
            w2q,
            dequant_scale(m_w1).contiguous(),
            dequant_scale(m_w2).contiguous(),
            dequant_scale(m_x).contiguous(),
            dequant_scale(m_inter).contiguous(),
            m_inter,
            gated,
            act,
            key,
            m_x,
            m_w1.contiguous(),
            m_w2.contiguous(),
        )

    @property
    def num_experts(self) -> int:
        return self.w1q.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.w1q.shape[2]

    @property
    def intermediate_size(self) -> int:
        return self.w1q.shape[1] // 2 if self.gated else self.w1q.shape[1]

    @property
    def chain1(self) -> Fp8Chain:
        """GEMM1's chain: the gated or the ungated activation's fused e4m3 quantization, with the parameters."""
        return Fp8Chain(
            "glu" if self.gated else "act", self.act, "e4m3", self.params, True
        )

    def view(self) -> Dict[str, Any]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for this backend."""
        return {
            "w1q": self.w1q,
            "w2q": self.w2q,
            "s_w1": self.s_w1,
            "s_w2": self.s_w2,
            "s_x": self.s_x,
            "s_inter": self.s_inter,
            "m_x": self.m_x,
            "m_inter": self.m_inter,
            "m_w1": self.m_w1,
            "m_w2": self.m_w2,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
        }

    @staticmethod
    def from_view(view: Dict[str, Any]) -> "MoEWeightsE4M3":
        gated = bool(view.get("gated", True))
        return MoEWeightsE4M3(
            view["w1q"],
            view["w2q"],
            view["s_w1"],
            view["s_w2"],
            view["s_x"],
            view["s_inter"],
            view["m_inter"],
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),
            view.get("m_x"),
            view.get("m_w1"),
            view.get("m_w2"),
        )


@dataclass
class MoEWeightsE4M3PerChannel:
    """The per-token class's expert weights as the kernels consume them (``moe.e4m3.<act>.pertoken``): ``w1q [E, 2I, H]`` e4m3 (gate / up
    interleaved; ``[E, I, H]`` ungated), ``w2q [E, H, I]`` e4m3; the dequantization scales ``s_w1 [E, 2I]`` and ``s_w2 [E, H]`` (fp32, one
    per output channel in the stored row order); no static activation or intermediate scale (the activations and the intermediate carry
    one dynamic scale per row). Quantized by the class's own term per channel: ``m = div(448, nzor(amax, 448))`` over the row,
    ``q = cvt_e4m3(mul(w, m))``, ``s = div(1, m)``."""

    w1q: torch.Tensor
    w2q: torch.Tensor
    s_w1: torch.Tensor
    s_w2: torch.Tensor
    gated: bool = True
    act: str = "silu"
    params: Tuple[Optional[float], ...] = ()

    @staticmethod
    def prepare(
        fc1: torch.Tensor,
        fc2: torch.Tensor,
        gated: bool = True,
        act: Optional[str] = None,
        params: Sequence[Optional[float]] = (),
        interleaved: bool = False,
    ) -> "MoEWeightsE4M3PerChannel":
        """Quantize the bf16 experts ``fc1 [E, 2I, H]`` (rows ``[up, gate]``; ``[E, I, H]`` ungated) and ``fc2 [E, H, I]`` to e4m3 with
        one scale per output channel (:func:`quantize_e4m3_pertoken` on every expert's rows), interleave the gate / up rows."""
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
                f"{CLASS}.pertoken: the class takes bf16 expert weights fc1 [E, N1, H], fc2 [E, H, I]"
            )
        E, N1, H = fc1.shape
        I = N1 // 2 if gated else N1
        if tuple(fc2.shape) != (E, H, I):
            raise ValueError(
                f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)} (gated={gated})"
            )
        check_shape_e4m3(H, I, E)
        if not bool(torch.isfinite(fc1.float()).all()) or not bool(
            torch.isfinite(fc2.float()).all()
        ):
            raise ValueError(f"{CLASS}.pertoken: the expert weights must be finite")
        dev = fc1.device
        w1i = fc1.contiguous() if (interleaved or not gated) else interleave_fc1(fc1)
        w1q = torch.empty(E, N1, H, dtype=torch.float8_e4m3fn, device=dev)
        s_w1 = torch.empty(E, N1, dtype=torch.float32, device=dev)
        w2q = torch.empty(E, H, I, dtype=torch.float8_e4m3fn, device=dev)
        s_w2 = torch.empty(E, H, dtype=torch.float32, device=dev)
        for e in range(E):  # one expert at a time: the fp32 working copies stay small
            w1q[e], s_w1[e] = quantize_e4m3_pertoken(w1i[e])
            w2q[e], s_w2[e] = quantize_e4m3_pertoken(fc2[e].contiguous())
        return MoEWeightsE4M3PerChannel(w1q, w2q, s_w1, s_w2, gated, act, key)

    @property
    def num_experts(self) -> int:
        return self.w1q.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.w1q.shape[2]

    @property
    def intermediate_size(self) -> int:
        return self.w1q.shape[1] // 2 if self.gated else self.w1q.shape[1]

    @property
    def chain1(self) -> Fp8Chain:
        """GEMM1's chain: the gated or the ungated activation under the (token x channel) scales, bf16 out."""
        return Fp8Chain(
            "glu" if self.gated else "act", self.act, "bf16", self.params, True, True
        )

    def view(self) -> Dict[str, Any]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for this backend."""
        return {
            "w1q": self.w1q,
            "w2q": self.w2q,
            "s_w1": self.s_w1,
            "s_w2": self.s_w2,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
            "pertoken": True,
        }

    @staticmethod
    def from_view(view: Dict[str, Any]) -> "MoEWeightsE4M3PerChannel":
        gated = bool(view.get("gated", True))
        return MoEWeightsE4M3PerChannel(
            view["w1q"],
            view["w2q"],
            view["s_w1"],
            view["s_w2"],
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),
        )


#: GEMM2 of the per-token class: the accumulator under the (row x channel) scale, converted to bf16.
GEMM2_CHAIN_PERTOKEN = Fp8Chain("plain", "silu", "bf16", (), True, True)


def quantize_e4m3_pertoken(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The per-token class's quantization of the rows of ``x [M, K]`` as torch ops (the exact model of
    :func:`.reference_fp8.pertoken_quant_cpu` on the device): ``(q float8_e4m3fn [M, K], s fp32 [M])`` with ``m = div(448, nzor(amax,
    448))``, ``q = cvt.rn.satfinite.e4m3(mul(x, m))``, ``s = div(1, m)``; every op one IEEE rounding (torch's ``amax`` is exact, its
    division and multiply are ``.rn``); graph-capturable."""
    from .reference import TorchF32

    xf = x.to(torch.float32)
    amax = xf.abs().amax(dim=1)
    m = torch.div(
        torch.full_like(amax, E4M3_MAX),
        torch.where(amax != 0, amax, torch.full_like(amax, E4M3_MAX)),
    )
    q = TorchF32.cvt_out(xf * m.view(-1, 1), "e4m3").to(torch.float8_e4m3fn)
    return q, torch.div(torch.ones_like(m), m)


def quantize_e4m3_pertoken_kernel(
    x: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    dest: Optional[torch.Tensor] = None,
    k: int = 0,
) -> None:
    """The class's quantizer kernel (:mod:`.blocks.fp8_pertoken_quant`) on the bf16 rows ``x [M, K]``: the codes into ``q [R, K]``
    (``float8_e4m3fn``) and the scales into ``s [R]`` at the rows themselves (``k = 0``, ``R >= M``) or at the ``k`` dense rows ``dest[t
    k + j]`` of every token (``dest [M k]`` int32, ``-1`` for a dropped slot): the input quantization fused with the scatter (a placement
    of the same bytes)."""
    from .blocks.fp8_pertoken_quant import (
        get_compiled_pertoken_quant,
        warps_per_row_quant,
    )

    M, K = x.shape
    if x.dtype != torch.bfloat16 or x.stride(1) != 1 or x.stride(0) % 8 or K % 8:
        raise TypeError(
            f"{CLASS}.pertoken: the quantizer takes 16-byte-aligned bf16 rows (got {x.dtype}, strides {tuple(x.stride())})"
        )
    if (
        q.dtype != torch.float8_e4m3fn
        or q.shape[1] != K
        or q.stride(1) != 1
        or q.stride(0) % ALIGN_E4M3
        or s.dtype != torch.float32
        or s.numel() != q.shape[0]
    ):
        raise ValueError(
            f"{CLASS}.pertoken: q [R, K] e4m3 row-major 16-byte-aligned and s [R] fp32"
        )
    if k == 0:
        if q.shape[0] < M:
            raise ValueError(
                f"{CLASS}.pertoken: q holds {q.shape[0]} rows for {M} inputs"
            )
        d = torch.zeros(1, dtype=torch.int32, device=x.device)
    else:
        if dest is None or dest.dtype != torch.int32 or dest.numel() != M * k:
            raise ValueError(
                f"{CLASS}.pertoken: dest [M k] int32 for the scatter placement"
            )
        d = dest.reshape(-1).contiguous()
    if M:
        get_compiled_pertoken_quant(K, warps_per_row_quant(K), k)(
            x, q.view(torch.uint8), s, d, M
        )


def grouped_gemm_fp8_pertoken(
    a: torch.Tensor,
    w: torch.Tensor,
    s_w: torch.Tensor,
    pts: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: Fp8Chain = GEMM2_CHAIN_PERTOKEN,
    out: Optional[torch.Tensor] = None,
    Tk: Optional[int] = None,
) -> torch.Tensor:
    """``out[p, n] = chain(a[p] @ w[e(p)].T, s = mul(pts[p], s_w[e(p), n]))`` for the expert segments ``counts``: the per-token class's GEMM
    (``chain.pertoken``) -- ``a [R >= Tk, K]`` e4m3 row-major, ``w [E, N, K]`` e4m3, ``s_w [E, N]`` fp32 the per-channel weight scales in
    the stored row order, ``pts [R]`` fp32 the per-row scales of ``a``. The record selects the body as :func:`grouped_gemm_fp8`; bf16
    (or the f32 oracle) out."""
    from .blocks.fp8_compile import (
        can_implement_fp8,
        get_compiled_grouped_fp8,
        get_compiled_swapped_fp8,
    )

    if not chain.pertoken:
        raise ValueError("grouped_gemm_fp8_pertoken takes the per-token class's chains")
    if (
        a.dtype != torch.float8_e4m3fn
        or w.dtype != torch.float8_e4m3fn
        or a.dim() != 2
        or w.dim() != 3
    ):
        raise TypeError(
            "grouped_gemm_fp8_pertoken takes e4m3 a [Tk, K] and w [E, N, K]"
        )
    R, K = a.shape
    E, N, K2 = w.shape
    if Tk is None:
        Tk = out.shape[0] if out is not None else R
    if not 0 <= Tk <= R:
        raise ValueError(
            f"grouped_gemm_fp8_pertoken: Tk {Tk} must lie in [0, {R}] (the rows of a)"
        )
    if (
        K != K2
        or a.stride(1) != 1
        or not w.is_contiguous()
        or a.stride(0) % ALIGN_E4M3
        or K % ALIGN_E4M3
    ):
        raise ValueError(
            "grouped_gemm_fp8_pertoken: K-contiguous, 16-byte-aligned operands of one K"
        )
    if (
        counts.dtype != torch.int32
        or tuple(counts.shape) != (E,)
        or not counts.is_contiguous()
    ):
        raise ValueError(
            "grouped_gemm_fp8_pertoken: counts must be a contiguous int32 [E]"
        )
    if (
        s_w.dtype != torch.float32
        or tuple(s_w.shape) != (E, N)
        or not s_w.is_contiguous()
    ):
        raise ValueError(
            "grouped_gemm_fp8_pertoken: s_w must be a contiguous fp32 [E, N]"
        )
    if pts.dtype != torch.float32 or pts.numel() < R or not pts.is_contiguous():
        raise ValueError(
            "grouped_gemm_fp8_pertoken: pts must be a contiguous fp32 [R] (the rows of a)"
        )
    if not can_implement_fp8(config, N, K, E, chain):
        raise ValueError(
            f"grouped_gemm_fp8_pertoken: {config.describe()} does not implement (N={N}, K={K}, E={E}, {chain.describe()})"
        )
    n_out = N // 2 if chain.pair else N
    out_dtype = _OUT_DTYPE[chain.out_fmt]
    if out is None:
        out = torch.empty(Tk, n_out, dtype=out_dtype, device=a.device)
    oal = 16 // out.element_size()
    if (
        tuple(out.shape) != (Tk, n_out)
        or out.dtype != out_dtype
        or out.stride(1) != 1
        or out.stride(0) % oal
    ):
        raise ValueError(
            "grouped_gemm_fp8_pertoken: out must be a 16-byte-aligned row-major [Tk, n_out] of the chain's dtype"
        )
    if Tk == 0:
        return out
    kernel = (
        get_compiled_swapped_fp8(config, chain)
        if config.swap_ab
        else get_compiled_grouped_fp8(config, chain)
    )
    if not config.swap_ab and Tk != R:
        a = a[:Tk]
    si = torch.ones(
        1, dtype=torch.float32, device=a.device
    )  # unread by the per-token chains
    kernel(
        a.unsqueeze(-1),
        w.permute(1, 2, 0),
        out,
        counts,
        s_w,
        pts.reshape(-1),
        si,
        min(persistent_clusters(a.device, config), launch_tile_bound(config, Tk, E, N)),
    )
    return out


def check_shape_e4m3(H: int, I: int, E: int) -> None:
    """The class's shape rule: ``H`` and ``I`` positive multiples of 16 (16-byte e4m3 rows of the activations, of the
    intermediate and of both expert weights), ``1 <= E <= MAX_E`` (the group tables). Everything else about a shape is
    placement."""
    if H <= 0 or I <= 0 or H % ALIGN_E4M3 or I % ALIGN_E4M3:
        raise unsupported(
            CLASS,
            f"hidden size {H}, intermediate size {I}",
            f"positive multiples of {ALIGN_E4M3} (16-byte e4m3 rows)",
        )
    if not 1 <= E <= MAX_E:
        raise unsupported(CLASS, f"num_experts {E}", f"1 <= num_experts <= {MAX_E}")


def grouped_gemm_fp8(
    a: torch.Tensor,
    w: torch.Tensor,
    s_w: torch.Tensor,
    s_a: torch.Tensor,
    inv: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: Fp8Chain = GEMM2_CHAIN,
    out: Optional[torch.Tensor] = None,
    Tk: Optional[int] = None,
) -> torch.Tensor:
    """``out[p] = chain(a[p] @ w[e(p)].T, alpha = mul(s_a, s_w[e(p)]))`` for the expert segments ``counts``: ``a [R >= Tk, K]`` e4m3
    row-major, ``w [E, N, K]`` e4m3 row-major, ``s_w [E]`` fp32, ``s_a [1]`` fp32 (the A operand's dequantization scale), ``inv [1]``
    fp32 (the intermediate multiplier of the e4m3 chains; ignored by the others). The e4m3 chains return ``float8_e4m3fn
    [Tk, N / 2]`` (gated) or ``[Tk, N]``; the plain chain bf16 / fp32 ``[Tk, N]``. The record selects the body: a dense record runs
    the grouped fp8 body, a swapped record (``config.swap_ab``: 128 or 64 weight rows x 16 / 32 tokens, the weights on the MMA's M
    side, :mod:`.blocks.fp8_swap_kernel`) the swapped body on the same dense rows -- a placement, the same bits. ``Tk`` (default:
    the rows of ``out``, else of ``a``) is the expanded row count; ``a`` may carry more rows (the swapped body's last token box then
    stays inside the buffer instead of being zero-filled past it). The persistent grid is a launch argument."""
    from .blocks.fp8_compile import (
        can_implement_fp8,
        get_compiled_grouped_fp8,
        get_compiled_swapped_fp8,
    )

    if (
        a.dtype != torch.float8_e4m3fn
        or w.dtype != torch.float8_e4m3fn
        or a.dim() != 2
        or w.dim() != 3
    ):
        raise TypeError("grouped_gemm_fp8 takes e4m3 a [Tk, K] and w [E, N, K]")
    R, K = a.shape
    E, N, K2 = w.shape
    if Tk is None:
        Tk = out.shape[0] if out is not None else R
    if not 0 <= Tk <= R:
        raise ValueError(
            f"grouped_gemm_fp8: Tk {Tk} must lie in [0, {R}] (the rows of a)"
        )
    if (
        K != K2
        or a.stride(1) != 1
        or not w.is_contiguous()
        or a.stride(0) % ALIGN_E4M3
        or K % ALIGN_E4M3
    ):
        raise ValueError(
            "grouped_gemm_fp8: K-contiguous, 16-byte-aligned operands of one K "
            f"(a {tuple(a.shape)} strides {tuple(a.stride())}, w {tuple(w.shape)})"
        )
    if (
        counts.dtype != torch.int32
        or tuple(counts.shape) != (E,)
        or not counts.is_contiguous()
    ):
        raise ValueError("grouped_gemm_fp8: counts must be a contiguous int32 [E]")
    if (
        s_w.dtype != torch.float32
        or s_w.numel() != E
        or s_a.numel() != 1
        or inv.numel() != 1
    ):
        raise ValueError("grouped_gemm_fp8: s_w [E] fp32, s_a and inv 1-element fp32")
    if not can_implement_fp8(config, N, K, E, chain):
        raise ValueError(
            f"grouped_gemm_fp8: {config.describe()} does not implement (N={N}, K={K}, E={E}, {chain.describe()})"
        )
    n_out = N // 2 if chain.pair else N
    out_dtype = _OUT_DTYPE[chain.out_fmt]
    if out is None:
        out = torch.empty(Tk, n_out, dtype=out_dtype, device=a.device)
    oal = 16 // out.element_size()
    if (
        tuple(out.shape) != (Tk, n_out)
        or out.dtype != out_dtype
        or out.stride(1) != 1
        or out.stride(0) % oal
    ):
        raise ValueError(
            "grouped_gemm_fp8: out must be a 16-byte-aligned row-major [Tk, n_out] of the chain's dtype"
        )
    out_k = out.view(torch.uint8) if chain.out_fmt == "e4m3" else out
    if Tk == 0:
        return out
    kernel = (
        get_compiled_swapped_fp8(config, chain)
        if config.swap_ab
        else get_compiled_grouped_fp8(config, chain)
    )
    if not config.swap_ab and Tk != R:
        a = a[
            :Tk
        ]  # the grouped body's A has the output's rows (one symbol); the swapped body's R is free
    kernel(
        a.unsqueeze(-1),
        w.permute(1, 2, 0),
        out_k,
        counts,
        s_w.reshape(-1).to(torch.float32),
        s_a.reshape(1).to(torch.float32),
        inv.reshape(1).to(torch.float32),
        min(persistent_clusters(a.device, config), launch_tile_bound(config, Tk, E, N)),
    )
    return out


def fused_moe_fp8(
    xq: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: MoEWeightsE4M3,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    return_intermediates: bool = False,
    plan: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """The W8A8 MoE FFN for pre-routed tokens: ``xq [T, H]`` e4m3 (the caller's static quantization, ``x = q / m_x``),
    ``token_selected_experts [T, k]`` int32, ``token_final_scales [T, k]`` fp32; ``out[t] = sum_r w[t, r] * y_r`` with the class's
    chains and rounding (:mod:`.spec_fp8`), the ``k`` slots folded in slot order in fp32 and converted once. ``config1`` /
    ``config2``: the grouped records (any record of the space gives the same bits; the dispatcher's pick by default)."""
    from .combine import get_compiled_combine, warps_per_token
    from .permute import get_compiled_gather, warps_per_row
    from .schedule_fp8 import default_configs_fp8

    if (
        xq.dtype != torch.float8_e4m3fn
        or xq.dim() != 2
        or xq.stride(1) != 1
        or xq.stride(0) % ALIGN_E4M3
    ):
        raise TypeError(
            "fused_moe_fp8: xq must be a row-major, 16-byte-aligned e4m3 [T, H]"
        )
    T, H = xq.shape
    if w.hidden_size != H:
        raise ValueError(
            f"fused_moe_fp8: xq has H = {H}, the weights H = {w.hidden_size}"
        )
    k = token_selected_experts.shape[1]
    if (
        tuple(token_selected_experts.shape) != (T, k)
        or token_selected_experts.dtype != torch.int32
    ):
        raise TypeError("fused_moe_fp8: token_selected_experts must be an int32 [T, k]")
    if (
        tuple(token_final_scales.shape) != (T, k)
        or token_final_scales.dtype != torch.float32
    ):
        raise TypeError("fused_moe_fp8: token_final_scales must be a float32 [T, k]")
    experts = token_selected_experts.contiguous()
    scales = token_final_scales.contiguous()
    E, I = w.num_experts, w.intermediate_size
    check_shape_e4m3(H, I, E)
    dev = xq.device
    if out is None:
        out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    if (
        tuple(out.shape) != (T, H)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "fused_moe_fp8: out must be a row-major, 16-byte-aligned bf16 [T, H]"
        )
    if config1 is None or config2 is None:
        from ...gemm.batch_invariant.api import part_of

        sm_count, family = part_of(dev)
        c1, c2 = default_configs_fp8(
            T, k, H, I, E, sm_count, family, gated=w.gated, chain1=w.chain1
        )
        config1, config2 = config1 or c1, config2 or c2
    counts, dest, src = (
        plan if plan is not None else permute_plan(experts, E)
    )  # ``plan``: ``route_plan``'s outputs, the same integers
    Tk = T * k
    # a swapped record reads a token box of its tile's rows at every group's row: the dense buffer it reads carries that many
    # rows past the live ones, so the last tiles' boxes stay inside it (rows past the live ones are computed and never stored)
    pad1 = config1.tile_n if config1.swap_ab else 0
    pad2 = config2.tile_n if config2.swap_ab else 0
    x_buf = torch.empty(Tk + pad1, H, dtype=torch.float8_e4m3fn, device=dev)
    x_perm = x_buf[:Tk]
    if Tk:
        get_compiled_gather(k, "e4m3", warps_per_row(Tk, H // 16, sm_count_of(dev)))(
            xq, src, x_perm, Tk, H
        )
    a_buf = torch.empty(Tk + pad2, I, dtype=torch.float8_e4m3fn, device=dev)
    a = grouped_gemm_fp8(
        x_buf,
        w.w1q,
        w.s_w1,
        w.s_x,
        w.m_inter,
        counts,
        config1,
        w.chain1,
        out=a_buf[:Tk],
        Tk=Tk,
    )  # [Tk, I] e4m3
    y = grouped_gemm_fp8(
        a_buf, w.w2q, w.s_w2, w.s_inter, w.m_inter, counts, config2, GEMM2_CHAIN, Tk=Tk
    )  # [Tk, H] bf16
    if T:
        get_compiled_combine(
            k, "bf16", "bf16", warps_per_token(T, H, sm_count_of(dev))
        )(y, dest, scales, out, T, H)
    if return_intermediates:
        return out, dict(
            counts=counts,
            dest=dest,
            src=src,
            x_perm=x_perm,
            a=a,
            y=y,
            config1=config1,
            config2=config2,
        )
    return out


def fused_moe_fp8_pertoken(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: MoEWeightsE4M3PerChannel,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    return_intermediates: bool = False,
    plan: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    x_scale: Optional[torch.Tensor] = None,
):
    """The per-token fp8 MoE FFN ``moe.e4m3.<act>.pertoken`` for pre-routed tokens: ``x [T, H]`` bf16, quantized per token by the
    class's quantizer with its stores at the token's ``k`` dense rows (the input quantization fused with the scatter), or ``x`` e4m3
    with ``x_scale [T]`` fp32 the caller's per-token scales (the class then starts at the bytes: the rows are gathered and the scales
    taken per slot). GEMM1 scales every accumulator by the token's scale times the channel's, applies the activation and emits bf16;
    the intermediate is requantized per row by the same quantizer; GEMM2 scales by the row's scale times the channel's; the combine is
    the bf16 class's. ``config1`` / ``config2``: records of the fp8 classes' spaces (dense or swapped: the record selects the body);
    the dispatcher's picks by default."""
    from .combine import get_compiled_combine, warps_per_token
    from .permute import get_compiled_gather, warps_per_row
    from .schedule_fp8 import default_configs_fp8

    pre = x.dtype == torch.float8_e4m3fn
    if (
        x.dim() != 2
        or x.stride(1) != 1
        or (x.dtype not in (torch.bfloat16, torch.float8_e4m3fn))
    ):
        raise TypeError(
            "fused_moe_fp8_pertoken: x must be a row-major bf16 [T, H] (or e4m3 with x_scale)"
        )
    T, H = x.shape
    if w.hidden_size != H:
        raise ValueError(
            f"fused_moe_fp8_pertoken: x has H = {H}, the weights H = {w.hidden_size}"
        )
    k = token_selected_experts.shape[1]
    if (
        tuple(token_selected_experts.shape) != (T, k)
        or token_selected_experts.dtype != torch.int32
    ):
        raise TypeError(
            "fused_moe_fp8_pertoken: token_selected_experts must be an int32 [T, k]"
        )
    if (
        tuple(token_final_scales.shape) != (T, k)
        or token_final_scales.dtype != torch.float32
    ):
        raise TypeError(
            "fused_moe_fp8_pertoken: token_final_scales must be a float32 [T, k]"
        )
    if pre and (
        x_scale is None
        or tuple(x_scale.shape) != (T,)
        or x_scale.dtype != torch.float32
    ):
        raise TypeError(
            "fused_moe_fp8_pertoken: an e4m3 x comes with x_scale, a float32 [T]"
        )
    experts = token_selected_experts.contiguous()
    scales = token_final_scales.contiguous()
    E, I = w.num_experts, w.intermediate_size
    check_shape_e4m3(H, I, E)
    dev = x.device
    if out is None:
        out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    if (
        tuple(out.shape) != (T, H)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "fused_moe_fp8_pertoken: out must be a row-major, 16-byte-aligned bf16 [T, H]"
        )
    if config1 is None or config2 is None:
        from ...gemm.batch_invariant.api import part_of

        sm_count, family = part_of(dev)
        c1, c2 = default_configs_fp8(
            T, k, H, I, E, sm_count, family, gated=w.gated, chain1=w.chain1
        )
        config1, config2 = config1 or c1, config2 or c2
    counts, dest, src = plan if plan is not None else permute_plan(experts, E)
    Tk = T * k
    pad1 = config1.tile_n if config1.swap_ab else 0
    pad2 = config2.tile_n if config2.swap_ab else 0
    # GEMM1's operand: the tokens' codes at their dense rows and the per-row scales (the pad rows are computed by the swapped body
    # and never stored)
    x_buf = torch.empty(Tk + pad1, H, dtype=torch.float8_e4m3fn, device=dev)
    pts1 = torch.zeros(Tk + pad1, dtype=torch.float32, device=dev)
    if pre:
        if Tk:
            get_compiled_gather(
                k, "e4m3", warps_per_row(Tk, H // 16, sm_count_of(dev))
            )(x.contiguous(), src, x_buf[:Tk], Tk, H)
            pts1[:Tk] = x_scale.reshape(-1)[(src // k).long()]
    else:
        quantize_e4m3_pertoken_kernel(x.contiguous(), x_buf, pts1, dest.reshape(-1), k)
    h = grouped_gemm_fp8_pertoken(
        x_buf, w.w1q, w.s_w1, pts1, counts, config1, w.chain1, Tk=Tk
    )  # [Tk, I] bf16
    a_buf = torch.empty(Tk + pad2, I, dtype=torch.float8_e4m3fn, device=dev)
    pts2 = torch.zeros(Tk + pad2, dtype=torch.float32, device=dev)
    quantize_e4m3_pertoken_kernel(h, a_buf, pts2)
    y = grouped_gemm_fp8_pertoken(
        a_buf, w.w2q, w.s_w2, pts2, counts, config2, GEMM2_CHAIN_PERTOKEN, Tk=Tk
    )  # [Tk, H] bf16
    if T:
        get_compiled_combine(
            k, "bf16", "bf16", warps_per_token(T, H, sm_count_of(dev))
        )(y, dest, scales, out, T, H)
    if return_intermediates:
        return out, dict(
            counts=counts,
            dest=dest,
            src=src,
            x_perm=x_buf[:Tk],
            pts1=pts1[:Tk],
            h=h,
            a=a_buf[:Tk],
            pts2=pts2[:Tk],
            y=y,
            config1=config1,
            config2=config2,
        )
    return out
