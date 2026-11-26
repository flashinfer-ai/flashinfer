# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The block-scale fp8 MoE classes ``moe.e4m3bs.<act>`` (DeepSeek-style fp8): the quantizer entry points, the weight view, the
grouped block-scale GEMMs (placements of the dense class ``gemm.e4m3.kblock128``) and the fused FFN on pre-routed tokens.

The routing, the deterministic permutation and the slot-order combine are the bf16 classes' (:mod:`.api`). The operands follow
FlashInfer's DeepSeek conventions: e4m3 activations with one fp32 scale per 1x128 block (``hidden_states_scale [H / 128, T]``,
the one layout :meth:`TrtllmFp8BlockConfig.prepare_activations` hands around, taken as given and never transposed), e4m3
experts with one fp32 scale per 128x128 block (``[E, N / 128, K / 128]``). Both quantizations are the ``fp8bs.quant`` class
(:mod:`.spec_fp8bs`); the class takes a caller's quantization as given (it starts at the bytes and the scales).

Pipeline: the dense permutation plan -> the e4m3 row gather and the block-scale gather (``sx_perm [H / 128, Tk]``: the MN-major
layout the grouped GEMM's row-scale ring reads) -> grouped block-scale GEMM1 (the dense class per expert: the per-block fold
promoted with the token row's and the weight block's scales; the bf16 class's activation chain, one bf16 rounding) -> the
intermediate's requantization per 1x128 block by the ``fp8bs.quant`` class, either fused into GEMM1's epilogue (the wide-tile
records) or as the standalone pass on the materialised bf16 intermediate -- two placements of one term, the same bytes ->
grouped block-scale GEMM2 (bf16 out) -> the bf16 class's slot-order combine. Every output bit is a function of the token's own
e4m3 row and block scales, its ``k`` experts and routing weights, the expert bytes and block scales: batch size and
composition, row position, expert load, the grouped records and the requantization spelling never enter the dataflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from ...batch_invariant.fp_interface import exact as fp32
from ...batch_invariant.support import LazyNames, check_device, unsupported
from ...gemm.batch_invariant.api import sm_count_of
from ...gemm.batch_invariant.generator import Knobs
from ...gemm.batch_invariant.spec import BLOCK_K
from . import spec as S
from .api import (
    GATHER_TOKEN_TILE_MAX,
    _check_fused_inputs,
    _check_grouped_chain,
    _check_grouped_operands,
    _grouped_out,
    _permute_plan_run,
    combine_rows,
    interleave_fc1,
    launch_handle,
    permute_plan,
)
from .blocks.epilogue_fp8bs import GEMM2_CHAIN, OUT_FMT_QUANT, Fp8BsChain, gemm1_chain
from .blocks.scheduler import MAX_E
from .reference import TorchF32
from .spec_fp8bs import BLOCK as QBLOCK
from .spec_fp8bs import (
    BLOCKS,
    CLASS,
    CLASS_OF,
    E4M3_MAX,
    IN_FMT_OF,
    block_chain,
    check_block,
    quant_pre,
)

ALIGN_E4M3 = 16  # e4m3 elements per 16-byte row: the TMA alignment of every operand
#: The compute capabilities the classes are stated for (the dense block-scale GEMM class's parts).
MOE_FP8BS_ARCHS: Tuple[Tuple[int, int], ...] = ((10, 0), (10, 3), (10, 7))
#: The weight view's pair tag (``quant``): the backend's views share one key, so a view names its pair.
VIEW_PAIR = "deepseek_fp8"
TAIL_ROWS = 256  # rows past the live rows of every packed buffer: the largest cluster tile's TMA box stays in bounds
_OUT_DTYPE = {"bf16": torch.bfloat16, "f32": torch.float32}
_DTYPE_STR = {torch.bfloat16: "bf16"}  # the quantizer's input: the layer's bf16 rows
SCALE_LAYOUTS = ("MN", "K")
#: Below this many warps per SM the quantizer's launch is latency-bound: more warps share a row (empirical; a placement).
ROW_WARPS_PER_SM_SMALL = 16
#: The route kernel's top-k bound; the class's bound on the slots.
K_MAX = 16


def _load_kernels() -> Dict[str, Any]:
    from .blocks.compile import get_compiled_grouped, get_compiled_swapped
    from .blocks.fp8bs_small import get_compiled_gather_rows_scales
    from .permute import warps_per_row
    from .schedule import launch_tile_bound
    from .schedule_fp8bs import (
        can_implement_grouped_fp8bs,
        can_implement_swapped_fp8bs,
        default_configs_fp8bs,
        fused_quant_default,
    )

    return dict(
        get_compiled_grouped=get_compiled_grouped,
        get_compiled_swapped=get_compiled_swapped,
        get_compiled_gather_rows_scales=get_compiled_gather_rows_scales,
        warps_per_row=warps_per_row,
        launch_tile_bound=launch_tile_bound,
        can_implement_grouped_fp8bs=can_implement_grouped_fp8bs,
        can_implement_swapped_fp8bs=can_implement_swapped_fp8bs,
        default_configs_fp8bs=default_configs_fp8bs,
        fused_quant_default=fused_quant_default,
    )


_k = LazyNames(_load_kernels)


# ----------------------------------------------------------------------------- the quantizer
def quant_warps_per_row(
    M: int, K: int, sms: Optional[int] = None, block: int = QBLOCK
) -> int:
    """The quantizer's placement at (rows, K): warps per row, a power of two of at most 8 and at most the row's block groups (the
    blocks of one warp pass: two at 128, eight at 32), doubled while the launch holds fewer than ``ROW_WARPS_PER_SM_SMALL`` warps
    per SM (a placement: the block trees never move)."""
    bpp = 32 // (check_block(block) // 8)
    pairs = max(1, (K // block + bpp - 1) // bpp)
    cap = 1
    while cap * 2 <= min(8, pairs):
        cap *= 2
    sms = sms or 1
    w = 1
    while w * 2 <= cap and M * w * 2 <= ROW_WARPS_PER_SM_SMALL * sms:
        w *= 2
    return w


def _check_quant_input(x: torch.Tensor, block: int) -> Tuple[int, int]:
    """The quantizer's input rule: a 2-D bf16 tensor with 16-byte-aligned contiguous rows, ``K`` whole blocks."""
    if x.dim() != 2 or x.dtype not in _DTYPE_STR:
        raise unsupported(
            "fp8bs.quant", f"input {tuple(x.shape)} of {x.dtype}", "a 2-D bf16 tensor"
        )
    if block not in BLOCKS:
        raise unsupported(
            "fp8bs.quant", f"deepseek_fp8_block_size {block}", f"one of {BLOCKS}"
        )
    M, K = x.shape
    if K <= 0 or K % block:
        raise unsupported(
            "fp8bs.quant", f"K = {K}", f"a whole number of {block}-element blocks"
        )
    if x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
        raise unsupported(
            "fp8bs.quant",
            f"rows with strides {tuple(x.stride())}",
            "16-byte-aligned contiguous rows",
        )
    return M, K


def fp8_block_quantize_plan(
    x: torch.Tensor,
    experts: torch.Tensor,
    E: int,
    block: int = QBLOCK,
    wpr: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(q, s, counts, dest, src)``: :func:`fp8_block_quantize` of ``x [T, K]`` (the ``"MN"`` scale layout) and the dense
    :func:`~.api.permute_plan` of ``experts [T, k]`` in one launch (the plan's CTAs, then the quantizer's: a placement, the
    same bits and integers as the two launches); above the plan's multi-CTA threshold the two launches run."""
    from .api import PLAN_MULTI_MIN_N
    from .blocks.fp8bs_small import get_compiled_plan_quant

    M, K = _check_quant_input(x, block)
    if experts.dim() != 2 or experts.shape[0] != M:
        raise ValueError(
            f"fp8_block_quantize_plan: experts {tuple(experts.shape)} must be [T, k] for x of {M} rows"
        )
    n = experts.numel()
    if n == 0 or n >= PLAN_MULTI_MIN_N:
        q, s = fp8_block_quantize(x, "MN", block=block, wpr=wpr)
        return (q, s) + tuple(permute_plan(experts, E))
    if not 1 <= E <= MAX_E:
        raise ValueError(
            f"fp8_block_quantize_plan: E {E} must lie in [1, {MAX_E}] (the plan kernel's expert tables)"
        )
    if not 1 <= experts.shape[1] <= S.MAX_TOP_K:
        raise ValueError(
            f"fp8_block_quantize_plan: {experts.shape[1]} slots per token; the kernels fold at most {S.MAX_TOP_K}"
        )
    flat = experts.reshape(-1)
    if not flat.is_contiguous() or flat.dtype != torch.int32:
        raise TypeError(
            "fp8_block_quantize_plan: experts must be a contiguous int32 [T, k]"
        )
    dev = x.device
    KB = K // block
    q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty(KB, M, dtype=torch.float32, device=dev)
    dest = torch.empty(n, dtype=torch.int32, device=dev)
    src = torch.empty(n, dtype=torch.int32, device=dev)
    counts = torch.empty(E, dtype=torch.int32, device=dev)
    w = wpr if wpr is not None else quant_warps_per_row(M, K, sm_count_of(dev), block)
    get_compiled_plan_quant(_DTYPE_STR[x.dtype], w, block)(
        flat, counts, dest, src, x, q.view(torch.uint8), s.t(), n, E, M, K
    )
    return q, s, counts, dest.view(experts.shape), src


def fp8_block_quantize(
    x: torch.Tensor,
    scale_layout: str = "MN",
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    wpr: Optional[int] = None,
    enable_pdl: bool = True,
    block: int = QBLOCK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(q [M, K] float8_e4m3fn, s fp32)`` of ``x [M, K]`` (bf16, 16-byte-aligned rows, ``K`` a whole number of
    ``block``-element blocks; ``block`` 128, or 32 for the ``moe.e4m3bs32`` classes): per block ``s = fmaxc(div(amax |x|, 448),
    1e-12)``, ``q = cvt.rn.satfinite.e4m3(mul(x, div(1, s)))``. ``s`` is ``[K / block, M]`` (``"MN"``, FlashInfer's DeepSeek
    activation layout) or ``[M, K / block]`` (``"K"``). ``out = (q, s)`` writes into the caller's tensors (``s`` in the requested
    layout, any strides). Bitwise identical for every ``wpr`` (:func:`quant_warps_per_row`), every batch composition and across
    runs."""
    from .blocks.fp8bs_small import get_compiled_fp8bs_quant

    M, K = _check_quant_input(x, block)
    if scale_layout not in SCALE_LAYOUTS:
        raise unsupported(
            "fp8bs.quant", f"scale_layout {scale_layout!r}", '"MN" or "K"'
        )
    KB = K // block
    want = (KB, M) if scale_layout == "MN" else (M, KB)
    if out is None:
        q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty(want, dtype=torch.float32, device=x.device)
    else:
        q, s = out
        if (
            tuple(q.shape) != (M, K)
            or q.dtype != torch.float8_e4m3fn
            or q.stride(1) != 1
            or q.stride(0) % 16
        ):
            raise ValueError(
                "fp8_block_quantize: out[0] must be a 16-byte-aligned row-major e4m3 [M, K]"
            )
        if s.dtype != torch.float32 or tuple(s.shape) != want:
            raise ValueError(
                f"fp8_block_quantize: out[1] must be an fp32 {want} for scale_layout={scale_layout!r}"
            )
    if M == 0:
        return q, s
    s_view = (
        s.t() if scale_layout == "MN" else s
    )  # the kernel addresses (M, KB) through the tensor's strides
    w = (
        wpr
        if wpr is not None
        else quant_warps_per_row(M, K, sm_count_of(x.device), block)
    )
    get_compiled_fp8bs_quant(_DTYPE_STR[x.dtype], w, enable_pdl, block)(
        x, q.view(torch.uint8), s_view, M, K
    )
    return q, s


def _on_device(dev: torch.device):
    """``TorchF32`` with its constants on ``dev``: a tensor divided by a CPU scalar is multiplied by the scalar's reciprocal by
    torch's CUDA kernel (two roundings); a 0-d device tensor divisor takes the IEEE division."""

    class OnDevice(TorchF32):
        @staticmethod
        def const(v):
            return torch.tensor(fp32.to_f32(v), dtype=torch.float32, device=dev)

    return OnDevice


def quantize_weight_blocks(
    w: torch.Tensor, block: int = QBLOCK
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The class per ``block x block`` block of ``w [N, K]`` (bf16): ``(q [N, K] float8_e4m3fn, s [N / block, K / block] fp32)``
    as torch ops on the device, each a single IEEE rounding (``amax`` exact, ``div`` / ``mul`` correctly rounded, the saturating
    RNE conversion of the exact model); the rounded values are e4m3 values, so the final cast is exact. The same chain as the
    activations' blocks."""
    if w.dim() != 2 or w.dtype != torch.bfloat16:
        raise unsupported(
            "fp8bs.quant", f"weight {tuple(w.shape)} of {w.dtype}", "a 2-D bf16 weight"
        )
    check_block(block)
    N, K = w.shape
    if N % block or K % block or N == 0 or K == 0:
        raise unsupported(
            "fp8bs.quant", f"weight ({N}, {K})", f"whole {block}x{block} blocks"
        )
    blocks = (
        w.to(torch.float32)
        .view(N // block, block, K // block, block)
        .permute(0, 2, 1, 3)
    )
    amax = blocks.abs().amax(dim=(-1, -2), keepdim=True)
    s, inv = block_chain(_on_device(w.device), amax)
    # the conversion: torch's RNE cast after the clamp to the e4m3 range (the clamp makes it the saturating conversion on
    # finite values)
    y = quant_pre(TorchF32, blocks, inv).clamp(-float(E4M3_MAX), float(E4M3_MAX))
    q = y.permute(0, 2, 1, 3).reshape(N, K).to(torch.float8_e4m3fn)
    return q, s[..., 0, 0].contiguous()


# ----------------------------------------------------------------------------- the weights
def check_shape_fp8bs(
    H: int, I: int, E: int, k: Optional[int] = None, block: int = QBLOCK
) -> None:
    """The classes' shape rule: ``H`` and ``I`` positive multiples of 128 (whole 128-element scale blocks of the quantized
    activations (K = H), of the quantized intermediate (K = I) and whole 128-row scale blocks of both expert weights, the
    gated weight's up and gate halves each; the 16-byte e4m3 rows and the gate / up pairing are implied; the 32-wide class
    keeps the rule: its smem stage is four blocks), ``1 <= E <= MAX_E`` (the group tables), ``1 <= k <= min(E, 16)``.
    Everything else about a shape is placement; the classes have no split variant."""
    if block not in BLOCKS:
        raise unsupported(CLASS, f"deepseek_fp8_block_size {block}", f"one of {BLOCKS}")
    name = CLASS_OF[block]
    if H <= 0 or I <= 0 or H % BLOCK_K or I % BLOCK_K:
        raise unsupported(
            name,
            f"hidden size {H}, intermediate size {I}",
            f"positive multiples of {BLOCK_K} (whole smem stages of scale blocks of every operand)",
        )
    if not 1 <= E <= MAX_E:
        raise unsupported(name, f"num_experts {E}", f"1 <= num_experts <= {MAX_E}")
    if k is not None and not 1 <= k <= min(E, K_MAX):
        raise unsupported(
            name, f"top_k {k}", f"1 <= top_k <= min(num_experts, {K_MAX})"
        )


def check_block_scales(
    sf: torch.Tensor, what: str, name: str, own: bool = False
) -> None:
    """A weight's block scales ``sf [E, NB, KB]`` must be finite, positive and normal fp32 values (the reciprocal and the
    promotion stay finite); the class's own chain gives them for every finite block, so only an infinite weight fails
    (``own``), while a caller's scales may fail any way. The refusal names the first offending expert."""
    good = torch.isfinite(sf) & (sf >= torch.finfo(torch.float32).tiny)
    bad = (~good).flatten(1).any(dim=1)
    if bool(bad.any()):
        e = int(bad.nonzero()[0])
        raise unsupported(
            name,
            f"the {what} block scales of expert {e}"
            + (" (an infinite weight)" if own else "")
            + " that are not finite, positive, normal fp32 values",
            "finite, positive, normal block scales of every expert (the reciprocal and the promotion stay finite)",
        )


@dataclass
class MoEWeightsE4M3BS:
    """The block-scale experts as the kernels consume them: ``w1 [E, N1, H]`` e4m3 (``N1 = 2I`` gate / up interleaved in groups of 8
    rows for the gated chains, ``I`` as given for the ungated ones), ``w1_sf [E, N1 / block, H / block]`` fp32 -- the scales of the
    ORIGINAL rows' square blocks (the interleave permutes the quantized rows only; the kernel reads an interleaved column's
    scale from its original block), ``w2 [E, H, I]`` e4m3, ``w2_sf [E, H / block, I / block]``; ``act`` / ``params`` name the GEMM1
    chain; ``block`` the class (``deepseek_fp8_block_size``: 128 for ``moe.e4m3bs.<act>``, 32 for ``moe.e4m3bs32.<act>``)."""

    w1: torch.Tensor
    w1_sf: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    gated: bool = True
    act: str = "silu"
    params: Tuple[Optional[float], ...] = ()
    block: int = QBLOCK

    @staticmethod
    def prepare(
        fc1: torch.Tensor,
        fc2: torch.Tensor,
        gated: bool = True,
        act: Optional[str] = None,
        params: Sequence[Optional[float]] = (),
        quantized: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        block: int = QBLOCK,
        device: Optional[torch.device] = None,
    ) -> "MoEWeightsE4M3BS":
        """Quantize the experts (``fc1 [E, 2I, H]`` rows ``[up, gate]``, or ``[E, I, H]`` (``gated=False``), ``fc2 [E, H, I]``;
        bf16, or a wider floating dtype cast to bf16 one expert at a time) per ``block x block`` block with the ``fp8bs.quant``
        chain (:func:`quantize_weight_blocks`) and interleave the gated weight's quantized rows per expert (the scales stay
        the canonical rows'). The pack lives on ``device`` (the weights' by default), a CUDA device of the class: it is
        preallocated and filled one expert at a time, each expert's slice moved and cast right before its quantization, so
        the peak memory beyond the inputs is the pack plus one expert's temporaries; a block whose scale is not finite (an
        infinite weight) is refused naming the expert, a NaN weight gives NaN codes in its block. ``quantized = (w1_q, w1_sf,
        w2_q, w2_sf)``: a caller's block quantization of the canonical (non-interleaved) weights, e4m3 ``[E, N1, H]`` with
        fp32 ``[E, N1 / block, H / block]`` scales (FlashInfer's DeepSeek weight view: ``gemm1_weights`` /
        ``gemm1_weights_scale`` / ``gemm2_weights`` / ``gemm2_weights_scale``), taken as given: the codes interleaved into the
        pack per expert, the scales copied onto the pack's device and refused unless finite, positive and normal (the
        reciprocal and the promotion stay finite). ``block``: 128 (the DeepSeek recipe) or 32 (the re-blocked recipe of the
        ``kblock32`` class)."""
        if block not in BLOCKS:
            raise unsupported(
                CLASS, f"deepseek_fp8_block_size {block}", f"one of {BLOCKS}"
            )
        name = CLASS_OF[block]
        act = act if act is not None else ("silu" if gated else "relu2")
        if act not in (S.ACTS if gated else S.UNGATED_ACTS):
            raise ValueError(f"no {'gated' if gated else 'ungated'} chain {act!r}")
        key: Tuple[Optional[float], ...] = S.params_key(params)
        if len(key) != (len(S.ACT_PARAMS.get(act, ())) if gated else 0):
            raise ValueError(
                f"chain {act!r} takes parameters {S.ACT_PARAMS.get(act, ())}, got {key!r}"
            )
        if quantized is None:
            if (
                not fc1.is_floating_point()
                or not fc2.is_floating_point()
                or fc1.dim() != 3
                or fc2.dim() != 3
            ):
                raise TypeError(
                    f"{CLASS}: the classes take floating expert weights fc1 [E, N1, H], fc2 [E, H, I] (bf16, or cast "
                    "to bf16 one expert at a time)"
                )
            src1, src2 = fc1, fc2
        else:
            src1, src1_sf, src2, src2_sf = quantized
            if (
                src1.dtype != torch.float8_e4m3fn
                or src2.dtype != torch.float8_e4m3fn
                or src1.dim() != 3
                or src2.dim() != 3
            ):
                raise TypeError(
                    f"{CLASS}: the quantized experts are e4m3 [E, N1, H] and [E, H, I]"
                )
        E, N1, H = src1.shape
        I = N1 // 2 if gated else N1
        check_shape_fp8bs(H, I, E, block=block)
        if tuple(src2.shape) != (E, H, I) or (gated and N1 % 2):
            raise ValueError(
                f"fc2 {tuple(src2.shape)} does not match fc1 {tuple(src1.shape)} (gated={gated})"
            )
        NB1, KB1, NB2, KB2 = N1 // block, H // block, H // block, I // block
        if quantized is not None:
            # the caller's scales: their shape and values before any device is consulted
            for sf, shape, what in (
                (src1_sf, (E, NB1, KB1), "w1"),
                (src2_sf, (E, NB2, KB2), "w2"),
            ):
                if tuple(sf.shape) != shape or sf.dtype != torch.float32:
                    raise ValueError(
                        f"{name}: the block scales are fp32 [E, N / {block}, K / {block}] of each weight "
                        f"({what}: {tuple(sf.shape)} for {shape})"
                    )
                check_block_scales(sf, what, name)
        dev = src1.device if device is None else torch.device(device)
        if dev.type != "cuda":
            raise unsupported(
                name,
                f"expert weights quantized on {dev}",
                "a CUDA device for the pack (the weights' device, or device=...): the kernels read it there",
            )
        check_device(dev, name, MOE_FP8BS_ARCHS)
        # the pack, preallocated and filled one expert at a time: the peak beyond the inputs and the pack is one expert's
        # temporaries (the fp32 blocks of the quantizer, or the caller's codes and scales moved to the pack's device)
        w1q = torch.empty(E, N1, H, dtype=torch.float8_e4m3fn, device=dev)
        w1s = torch.empty(E, NB1, KB1, dtype=torch.float32, device=dev)
        w2q = torch.empty(E, H, I, dtype=torch.float8_e4m3fn, device=dev)
        w2s = torch.empty(E, NB2, KB2, dtype=torch.float32, device=dev)

        def expert(w: torch.Tensor, e: int) -> torch.Tensor:
            """Expert ``e``'s slice on the pack's device as contiguous bf16, moved and cast right before its quantization."""
            return w[e].to(device=dev, dtype=torch.bfloat16).contiguous()

        for e in range(E):
            if quantized is None:
                q1, s1 = quantize_weight_blocks(expert(src1, e), block)
                q2, s2 = quantize_weight_blocks(expert(src2, e), block)
            else:
                q1, q2 = src1[e].to(device=dev), src2[e].to(device=dev)
                s1 = src1_sf[e].to(device=dev, dtype=torch.float32, copy=True)
                s2 = src2_sf[e].to(device=dev, dtype=torch.float32, copy=True)
            # the interleave is a row permutation of the quantized rows: no arithmetic, the scales the canonical rows'
            w1q[e] = interleave_fc1(q1.unsqueeze(0))[0] if gated else q1
            w1s[e], w2q[e], w2s[e] = s1, q2, s2
        if quantized is None:
            check_block_scales(w1s, "w1", name, own=True)
            check_block_scales(w2s, "w2", name, own=True)
        return MoEWeightsE4M3BS(w1q, w1s, w2q, w2s, gated, act, key, block)

    @staticmethod
    def block_of(w1q: torch.Tensor, w1_sf: torch.Tensor) -> int:
        """The scale block size a quantized weight's scales imply (``N / w1_sf.shape[1]``), one of ``BLOCKS``."""
        if w1_sf.dim() != 3 or w1_sf.shape[1] == 0 or w1q.shape[1] % w1_sf.shape[1]:
            raise ValueError(
                f"{CLASS}: weight scales {tuple(w1_sf.shape)} do not tile the weight {tuple(w1q.shape)}"
            )
        block = w1q.shape[1] // w1_sf.shape[1]
        if block not in BLOCKS:
            raise unsupported(
                CLASS,
                f"a scale block of {block} rows",
                f"one of {BLOCKS} (deepseek_fp8_block_size)",
            )
        return block

    @property
    def num_experts(self) -> int:
        return self.w1.shape[0]

    @property
    def in_fmt(self) -> str:
        """The class's format key: ``"e4m3_bs"`` (128-wide blocks) or ``"e4m3_bs32"``."""
        return IN_FMT_OF[self.block]

    @property
    def hidden_size(self) -> int:
        return self.w1.shape[2]

    @property
    def intermediate_size(self) -> int:
        return self.w1.shape[1] // 2 if self.gated else self.w1.shape[1]

    def chain1(self, fused: bool) -> Fp8BsChain:
        """GEMM1's chain: the activation's bf16 output, or its fused requantization."""
        return gemm1_chain(self.act, self.params, self.gated, fused)

    def view(self) -> Dict[str, Any]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for this backend (``quant`` names the pair:
        the backend's views share one key; ``deepseek_fp8_block_size`` the class)."""
        return {
            "quant": VIEW_PAIR,
            "w1": self.w1,
            "w1_sf": self.w1_sf,
            "w2": self.w2,
            "w2_sf": self.w2_sf,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
            "deepseek_fp8_block_size": self.block,
        }

    @staticmethod
    def from_view(view: Dict[str, Any]) -> "MoEWeightsE4M3BS":
        """The weights of a view :meth:`view` wrote; the pair and the block are read from the ``quant`` and
        ``deepseek_fp8_block_size`` keys, never assumed, and the block must be the scales' own."""
        pair = view.get("quant")
        if pair != VIEW_PAIR:
            raise unsupported(
                CLASS,
                "a weight view without its quantization pair (no `quant` key)"
                if pair is None
                else f"a weight view of the pair {pair!r}",
                "the view of BatchInvariantFp8BlockConfig.prepare_weights(..., quant=QuantConfig(weight=DeepSeekFp8, "
                f"activation=DeepSeekFp8)) or of from_trtllm_view, which names the pair (`quant` = {VIEW_PAIR!r})",
            )
        if "deepseek_fp8_block_size" not in view:
            raise unsupported(
                CLASS,
                "a weight view without its scale block (no `deepseek_fp8_block_size` key)",
                "the view of prepare_weights / from_trtllm_view, which records the block",
            )
        block = int(view["deepseek_fp8_block_size"])
        own = MoEWeightsE4M3BS.block_of(view["w1"], view["w1_sf"])
        if block != own:
            raise unsupported(
                CLASS,
                f"a weight view whose scales tile {own}-row blocks under deepseek_fp8_block_size {block}",
                "a view whose block key and scale shapes agree",
            )
        gated = bool(view.get("gated", True))
        return MoEWeightsE4M3BS(
            view["w1"],
            view["w1_sf"],
            view["w2"],
            view["w2_sf"],
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),
            block,
        )


# ----------------------------------------------------------------------------- the grouped GEMM
def scale_mn_view(
    s: torch.Tensor, KB: int, rows: int, name: str, block: int = QBLOCK
) -> torch.Tensor:
    """The block scales of ``rows`` rows in the DeepSeek layout ``[KB, rows]`` (one fp32 scale per 1x``block`` block of each
    row: the layout ``TrtllmFp8BlockConfig.prepare_activations`` hands around and the grouped GEMM's row-scale ring reads),
    taken as given, as the kernel's ``(rows, KB)`` view with the tensor's own strides; another shape -- a ``[rows, KB]``
    tensor among them, never transposed into the layout -- or another dtype is refused naming the layout. The empty pack
    (``rows = 0``) takes any empty tensor."""
    if not isinstance(s, torch.Tensor) or s.dtype != torch.float32:
        raise unsupported(
            CLASS,
            f"{name} of {type(s).__name__ if not isinstance(s, torch.Tensor) else s.dtype}",
            f"an fp32 {name} [{KB}, {rows}]",
        )
    if rows == 0 and s.numel() == 0:
        return s.reshape(KB, 0).t()
    if tuple(s.shape) != (KB, rows):
        raise unsupported(
            CLASS,
            f"{name} of shape {tuple(s.shape)} for {rows} rows",
            f"[{KB}, {rows}] (one fp32 scale per 1x{block} block of each row: the DeepSeek layout, taken as given; a "
            f"[{rows}, {KB}] tensor is not transposed into it)",
        )
    return s.t()


def grouped_gemm_fp8bs(
    a: torch.Tensor,
    sfa: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: Fp8BsChain = GEMM2_CHAIN,
    out: Optional[torch.Tensor] = None,
    sfo: Optional[torch.Tensor] = None,
    interleaved: Optional[bool] = None,
    src: Optional[torch.Tensor] = None,
    k_div: int = 1,
    rows_out: Optional[int] = None,
):
    """``out[p] = chain( fold_kb fma(mul(sfa[p, kb], w_sf[e(p), blk(n), kb]), P_kb, acc) )`` on the dense expert segments
    ``counts``: ``a [Tk, K]`` e4m3, ``sfa [K / block, Tk]`` fp32 (the DeepSeek layout, as given), ``w [E, N, K]`` e4m3, ``w_sf
    [E, N / block, K / block]`` fp32 -- the block (128 or 32: the class) is read off the weight scales' shape. ``chain``: the bf16 / f32
    chains (one tensor returned) or the fused requantization (``out_fmt = "e4m3bs"``, the 128-wide class: ``(codes [Tk, Nout]
    float8_e4m3fn, scales [Nout / 128, Tk] fp32)``). ``interleaved``: the weight's rows are gate / up interleaved (default: the
    chain's pairing; the f32 oracle output of GEMM1 reads the interleaved weight with the plain chain). ``src``: the gather form
    of a swapped record (the bf16 / f32 chains): ``a [T, K]`` and ``sfa`` are then the unpermuted token rows and their scales, dense
    row ``p`` reads token ``src[p] // k_div`` (the plan's slot map, ``k_div`` the slots per token), ``rows_out`` the dense rows of
    ``out`` (``Tk`` plus the tail). The persistent grid is a launch argument, the device's capacity clipped to the launch's tile
    bound (`schedule.launch_tile_bound`: which CTA computes a tile, never the bits)."""
    gather = src is not None
    if gather and (
        not config.swap_ab
        or chain.out_fmt == OUT_FMT_QUANT
        or rows_out is None
        or src.dtype != torch.int32
        or not src.is_contiguous()
        or k_div < 1
    ):
        raise ValueError(
            "grouped_gemm_fp8bs: the gather form takes a swapped record, a bf16 / f32 chain, an int32 contiguous src, "
            "k_div >= 1 and rows_out"
        )
    # the shared operand rules at the class's alignment: K in whole smem stages of scale blocks, every row 16 bytes
    _, K, E, N, Tk = _check_grouped_operands(
        "grouped_gemm_fp8bs",
        a,
        w,
        counts,
        a.shape[0],
        None,
        torch.float8_e4m3fn,
        ALIGN_E4M3,
        "e4m3",
        k_align=BLOCK_K,
        n_align=ALIGN_E4M3,
    )
    if gather:
        # the gather form's dense rows are the output's (``a`` holds the tokens, fewer than the expanded rows)
        if rows_out < 0 or src.shape[0] < rows_out - TAIL_ROWS:
            raise ValueError("grouped_gemm_fp8bs: rows_out and src disagree")
        Tk = int(rows_out)
    block = MoEWeightsE4M3BS.block_of(w, w_sf)
    KB, NB = K // block, N // block
    if (
        tuple(w_sf.shape) != (E, NB, KB)
        or w_sf.dtype != torch.float32
        or not w_sf.is_contiguous()
        or w_sf.data_ptr() % 16
    ):
        raise ValueError(
            f"grouped_gemm_fp8bs: w_sf must be a contiguous, 16-byte-aligned fp32 {(E, NB, KB)}, got {tuple(w_sf.shape)}"
        )
    accepted = (
        _k.can_implement_swapped_fp8bs(config, N, K, E, chain, interleaved, block)
        if config.swap_ab
        else _k.can_implement_grouped_fp8bs(config, N, K, E, chain, interleaved, block)
    )
    if not accepted:
        raise ValueError(
            f"grouped_gemm_fp8bs: {config.describe()} does not implement (N={N}, K={K}, E={E}, {chain.describe()}, interleaved={interleaved}, block={block})"
        )
    sfa_v = scale_mn_view(sfa, KB, a.shape[0] if gather else Tk, "sfa", block)
    n_out = _check_grouped_chain("grouped_gemm_fp8bs", N, chain)
    quant = chain.out_fmt == OUT_FMT_QUANT
    if quant:
        out = _grouped_out(
            "grouped_gemm_fp8bs", out, Tk, n_out, torch.float8_e4m3fn, a.device
        )
        if sfo is None:
            sfo = torch.empty(n_out // QBLOCK, Tk, dtype=torch.float32, device=a.device)
        if (
            tuple(sfo.shape) != (n_out // QBLOCK, Tk)
            or sfo.dtype != torch.float32
            or not sfo.is_contiguous()
        ):
            raise ValueError(
                f"grouped_gemm_fp8bs: sfo must be a contiguous fp32 [{n_out // QBLOCK}, {Tk}]"
            )
        out_k, sfo_k = out.view(torch.uint8), sfo
    else:
        out = _grouped_out(
            "grouped_gemm_fp8bs", out, Tk, n_out, _OUT_DTYPE[chain.out_fmt], a.device
        )
        out_k, sfo_k = (
            out,
            w_sf[0, :1, :1].reshape(1, 1),
        )  # the scale output is unread: any fp32 (1, 1) tensor
    # the unified bodies on the promoted operand block: the chain carries the class's block and the weight interleave
    _grouped_fp8bs_launch(
        a, sfa_v, w.permute(1, 2, 0), w_sf, out_k, sfo_k, counts, config, chain.at(block, interleaved),
        quant, Tk, E, N, src=src, k_div=k_div,
    )  # fmt: skip
    return (out, sfo) if quant else out


def _body_kernel(config: Knobs, chain: Fp8BsChain) -> Any:
    """The compiled body of a block-scale fp8 record: the swapped body for a swapped record, the grouped body otherwise."""
    if config.swap_ab:
        return _k.get_compiled_swapped(config, chain)
    return _k.get_compiled_grouped(config, chain)


def _gather_kernel(config: Knobs, chain: Fp8BsChain) -> Any:
    return _k.get_compiled_swapped(config, chain, gather=True)


def _grouped_fp8bs_launch(
    a: torch.Tensor,
    sfa_v: torch.Tensor,
    w3: torch.Tensor,
    w_sf: torch.Tensor,
    out_k: torch.Tensor,
    sfo_k: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    body_chain: Fp8BsChain,
    quant: bool,
    Tk: int,
    E: int,
    N: int,
    src: Optional[torch.Tensor] = None,
    k_div: int = 1,
) -> None:
    """The body of ``config`` on prepared operands (:func:`grouped_gemm_fp8bs`'s launch): the gather form (``src`` given: the
    unpermuted rows and their scales through the plan's map), the swapped body or the grouped body (with the fused
    requantization's scale output ``sfo_k`` when ``quant``); ``w3 (N, K, E)`` the weight's kernel view, ``sfa_v`` the row
    scales as the kernel's ``(rows, KB)`` view; the persistent grid is the handle's cluster count clipped to the launch's tile
    bound. The callers validated the operands."""
    if Tk == 0:
        return
    dev = a.device
    if src is not None:
        h = launch_handle(
            dev, config, ("gather", body_chain), _gather_kernel, config, body_chain
        )
        grid = min(h.clusters, _k.launch_tile_bound(config, Tk, E, N))
        h.kernel(a, sfa_v, src, k_div, w3, w_sf, out_k, counts, grid)
        return
    h = launch_handle(
        dev, config, ("body", body_chain), _body_kernel, config, body_chain
    )
    grid = min(h.clusters, _k.launch_tile_bound(config, Tk, E, N))
    if config.swap_ab:
        h.kernel(a.unsqueeze(-1), sfa_v, w3, w_sf, out_k, counts, grid)
    elif quant:
        h.kernel(a.unsqueeze(-1), w3, sfa_v, w_sf, out_k, sfo_k, counts, Tk, grid, h.raster)  # fmt: skip
    else:
        h.kernel(a.unsqueeze(-1), w3, sfa_v, w_sf, out_k, counts, Tk, grid, h.raster)


def check_aligned(block: int, x: torch.Tensor, sx: Optional[torch.Tensor]) -> None:
    """The rows and the scales start at 16-byte boundaries (the kernels' vector loads and TMA boxes): the base pointer, not
    only the strides, which the shared entry rules check; the op's and the runner's one rule."""
    if x.data_ptr() % 16 or (sx is not None and sx.data_ptr() % 16):
        raise unsupported(
            CLASS_OF[block],
            "hidden_states_q or hidden_states_scale at a storage offset that is not a multiple of 16 bytes",
            "16-byte aligned rows and scale bytes (the kernels' vector loads and TMA boxes)",
        )


# ----------------------------------------------------------------------------- the op
def fused_moe_fp8bs(
    xq: torch.Tensor,
    sx: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: MoEWeightsE4M3BS,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    fuse_quant: Optional[bool] = None,
    return_intermediates: bool = False,
    gather: Optional[bool] = None,
    plan: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    w_round: bool = False,
):
    """The block-scale fp8 MoE FFN for pre-routed tokens: ``xq [T, H]`` e4m3 with ``sx`` its block scales (``[H / block, T]``,
    the DeepSeek layout as given; rows and scales 16-byte aligned) -- or the bf16 activations themselves with ``sx = None``:
    the class quantizes them by its own chain,
    in one launch with the permutation plan (:func:`fp8_block_quantize_plan`) --, ``token_selected_experts [T, k]`` int32,
    ``token_final_scales [T, k]`` fp32; ``out[t] = sum_r w[t, r] * y_r``
    with the class's chains and rounding (:mod:`.spec_fp8bs`), the ``k`` slots folded in slot order in fp32 and converted once.
    ``config1`` / ``config2``: the grouped records (any record of the space gives the same bits; the dispatcher's pick by
    default); ``fuse_quant``: the intermediate's requantization in GEMM1's epilogue (True) or by the standalone pass (False):
    a placement, the same bytes either way (:func:`.schedule_fp8bs.fused_quant_default` by default). ``gather``: GEMM1's
    gather form (the swapped body reads the token rows and their scales through the plan's map; no routed gather launch) --
    True / False force it, None applies the bf16 op's rule: a swapped record whose token tile is at most
    ``GATHER_TOKEN_TILE_MAX``, without the fused requantization (the gather4 instructions per stage grow with the tile; the
    box form's one launch is the cheaper placement at the wider tiles). ``plan``: ``route_plan``'s ``(counts, dest, src)``.
    ``w_round``: the f32 weights enter the fold as their bf16 rounding, by the combine kernel (the pre-routed packed-word
    convention), as in the bf16 op."""
    if xq.dtype in _DTYPE_STR:
        # bf16 activations: the class quantizes them itself -- with the dense plan in one launch when the plan is this
        # call's (the pre-routed form), by the standalone quantizer when the plan is given (the routed form's route + plan)
        if sx is not None:
            raise ValueError(
                "fused_moe_fp8bs: bf16 activations take no hidden_states_scale (the class quantizes them)"
            )
        T0, _, _, _, _, out = _check_fused_inputs(
            "fused_moe_fp8bs",
            xq,
            token_selected_experts,
            token_final_scales,
            out,
            w.hidden_size,
            xq.dtype,
            8,
            _DTYPE_STR[xq.dtype],
        )
        check_aligned(w.block, xq, None)
        if T0 == 0:
            return out  # the empty output: no plan, no quantizer, no record
        xq, sx, plan = _quantize_rows(xq, token_selected_experts.contiguous(), w, plan)
    # the shared fused-entry rules (the rows' alignment, the ids and weights, the output)
    T, H, k, experts, scales, out = _check_fused_inputs(
        "fused_moe_fp8bs",
        xq,
        token_selected_experts,
        token_final_scales,
        out,
        w.hidden_size,
        torch.float8_e4m3fn,
        ALIGN_E4M3,
        "e4m3",
    )
    E, I, block = w.num_experts, w.intermediate_size, w.block
    check_shape_fp8bs(H, I, E, k, block)
    dev = xq.device
    KB = H // block
    sx_kt = scale_mn_view(
        sx, KB, T, "hidden_states_scale", block
    ).t()  # as [KB, T] (a view; the gather reads its strides)
    check_aligned(block, xq, sx_kt)
    if T == 0:
        return out  # the empty output: no record is consulted
    if fuse_quant is None:
        fuse_quant = (
            _k.fused_quant_default(T, k, E, gated=w.gated) and block == QBLOCK
            if config1 is None
            else False
        )  # the 32-wide class requantizes by the standalone pass
    if fuse_quant and block != QBLOCK:
        raise unsupported(
            CLASS_OF[block],
            "the fused requantization",
            "the standalone requantization pass (the fused spelling is the 128-wide class's)",
        )
    if config1 is None or config2 is None:
        from ...gemm.batch_invariant.api import part_of

        sm_count, family = part_of(dev)
        c1, c2 = _k.default_configs_fp8bs(
            T,
            k,
            H,
            I,
            E,
            sm_count,
            family,
            w.act,
            w.gated,
            w.params,
            fuse_quant,
            block=block,
        )
        config1, config2 = config1 or c1, config2 or c2
    # GEMM1's gather form: a narrow swapped record reads the token rows and their scales through the plan's map itself (the
    # routed gather launch is not run); the box form (a grouped record, a wide token tile, or ``gather=False``) reads the
    # permuted copies -- the bf16 op's rule
    if gather is None:
        gather = _gather_form(config1, fuse_quant)
    if gather and (not config1.swap_ab or fuse_quant):
        raise ValueError(
            "fused_moe_fp8bs: the gather form belongs to GEMM1's swapped records without the fused requantization"
        )
    # the records' acceptance (the grouped entry's rule, for a caller's records)
    for cfg, N, K, chain in (
        (config1, w.w1.shape[1], H, w.chain1(fuse_quant)),
        (config2, H, I, GEMM2_CHAIN),
    ):
        accepted = (
            _k.can_implement_swapped_fp8bs(cfg, N, K, E, chain, None, block)
            if cfg.swap_ab
            else _k.can_implement_grouped_fp8bs(cfg, N, K, E, chain, None, block)
        )
        if not accepted:
            raise ValueError(
                f"fused_moe_fp8bs: {cfg.describe()} does not implement (N={N}, K={K}, E={E}, {chain.describe()}, block={block})"
            )
    return _fused_moe_fp8bs_run(
        xq, sx_kt, experts, scales, w, out, config1, config2, fuse_quant, gather, T, H, k, E, I,
        plan=plan, return_intermediates=return_intermediates, w_round=w_round,
    )  # fmt: skip


def _quantize_rows(
    x: torch.Tensor, experts: torch.Tensor, w: MoEWeightsE4M3BS, plan
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """The bf16 rows quantized by the class: with the dense plan in one launch when the plan is this call's, by the standalone
    quantizer when the plan is given (the routed form's route + plan). Returns ``(xq, sx, plan)``."""
    if plan is None:
        xq, sx, counts_q, dest_q, src_q = fp8_block_quantize_plan(
            x, experts, w.num_experts, block=w.block
        )
        return xq, sx, (counts_q, dest_q, src_q)
    xq, sx = fp8_block_quantize(x, "MN", block=w.block)
    return xq, sx, plan


def _gather_form(config1: Knobs, fuse_quant: bool) -> bool:
    """GEMM1's gather form (the bf16 op's rule): a swapped record of a narrow token tile without the fused requantization."""
    return (
        bool(config1.swap_ab)
        and not fuse_quant
        and config1.tile_n <= GATHER_TOKEN_TILE_MAX
    )


def _fused_moe_fp8bs_run(
    xq: torch.Tensor,
    sx_kt: torch.Tensor,
    experts: torch.Tensor,
    scales: torch.Tensor,
    w: MoEWeightsE4M3BS,
    out: torch.Tensor,
    config1: Knobs,
    config2: Knobs,
    fuse_quant: bool,
    gather: bool,
    T: int,
    H: int,
    k: int,
    E: int,
    I: int,
    *,
    plan: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    return_intermediates: bool = False,
    w_round: bool = False,
):
    """:func:`fused_moe_fp8bs` on validated operands (``xq`` e4m3 ``[T, H]``, ``sx_kt`` its ``[H / block, T]`` scales as given)
    with the records and the spelling chosen: the runner's launch (its pack proved every rule the public entry checks, so
    nothing is checked twice per call) and the public entry's body."""
    block = w.block
    dev = xq.device
    KB = H // block
    chain1 = w.chain1(fuse_quant)
    counts, dest, src = (
        plan if plan is not None else _permute_plan_run(experts.reshape(-1), T, k, E, 1, False, 128)
    )  # fmt: skip  # ``plan``: ``route_plan``'s outputs
    Tk = T * k
    # the packed row buffers (and their scale columns) carry TAIL_ROWS past Tk so every tile's TMA box stays in bounds (an
    # out-of-bounds box is served far more slowly than the rows it zero-fills); the tail rows are never stored or read back
    Tk_a = Tk + TAIL_ROWS if Tk else 0
    x_perm = torch.empty(Tk_a, H, dtype=torch.float8_e4m3fn, device=dev)
    sx_perm = torch.empty(KB, Tk_a, dtype=torch.float32, device=dev)
    if Tk and (not gather or return_intermediates):
        # the e4m3 rows and their scale columns to the expanded rows in one launch (a copy)
        _k.get_compiled_gather_rows_scales(
            k, _k.warps_per_row(Tk, H // 16, sm_count_of(dev))
        )(
            xq.view(torch.uint8),
            src.reshape(-1),
            x_perm.view(torch.uint8),
            sx_kt,
            sx_perm,
            Tk,
            H,
            KB,
        )
    w1_3, w2_3 = w.w1.permute(1, 2, 0), w.w2.permute(1, 2, 0)
    n1 = w.w1.shape[1]
    body1 = chain1.at(block, None)
    a = None
    if fuse_quant:
        # [Tk_a, I] e4m3 + [I / 128, Tk_a]: the fused requantization's codes and scales
        aq = torch.empty(Tk_a, I, dtype=torch.float8_e4m3fn, device=dev)
        sa = torch.empty(I // QBLOCK, Tk_a, dtype=torch.float32, device=dev)
        _grouped_fp8bs_launch(
            x_perm, sx_perm.t(), w1_3, w.w1_sf, aq.view(torch.uint8), sa, counts, config1, body1,
            True, Tk_a, E, n1,
        )  # fmt: skip
    else:
        a = torch.empty(Tk_a, I, dtype=torch.bfloat16, device=dev)
        unread = w.w1_sf[0, :1, :1].reshape(
            1, 1
        )  # the scale output is unread: any fp32 (1, 1) tensor
        if gather:
            # [Tk_a, I] bf16: the unpermuted rows through the plan's map
            _grouped_fp8bs_launch(
                xq, sx_kt.t(), w1_3, w.w1_sf, a, unread, counts, config1, body1, False, Tk_a, E, n1,
                src=src.reshape(-1), k_div=k,
            )  # fmt: skip
        else:
            _grouped_fp8bs_launch(
                x_perm, sx_perm.t(), w1_3, w.w1_sf, a, unread, counts, config1, body1, False, Tk_a, E, n1,
            )  # fmt: skip  # [Tk_a, I] bf16
        aq = torch.empty(Tk_a, I, dtype=torch.float8_e4m3fn, device=dev)
        sa = torch.empty(I // block, Tk_a, dtype=torch.float32, device=dev)
        if Tk:
            fp8_block_quantize(
                a[:Tk], "MN", out=(aq[:Tk], sa[:, :Tk]), block=block
            )  # the live rows only
    y = torch.empty(Tk_a, H, dtype=torch.bfloat16, device=dev)
    _grouped_fp8bs_launch(
        aq, sa.t(), w2_3, w.w2_sf, y, w.w2_sf[0, :1, :1].reshape(1, 1), counts, config2,
        GEMM2_CHAIN.at(block, None), False, Tk_a, E, H,
    )  # fmt: skip  # [Tk_a, H] bf16
    combine_rows(y[:Tk], dest, scales, out, k, w_round)
    if return_intermediates:
        # the live rows under the common keys; the tail-carrying buffers the kernels consume under ``*_buf`` (a kernel timed
        # on a live-row view reintroduces the out-of-bounds boxes)
        return out, dict(
            counts=counts,
            dest=dest,
            src=src,
            x_perm=x_perm[:Tk],
            sx_perm=sx_perm[:, :Tk],
            a=None if a is None else a[:Tk],
            gather=gather,
            aq=aq[:Tk],
            sa=sa[:, :Tk],
            y=y[:Tk],
            x_perm_buf=x_perm,
            sx_perm_buf=sx_perm,
            a_buf=a,
            aq_buf=aq,
            sa_buf=sa,
            y_buf=y,
            config1=config1,
            config2=config2,
            fused=fuse_quant,
            chain1=chain1,
        )
    return out


__all__ = [
    "ALIGN_E4M3",
    "CLASS",
    "K_MAX",
    "MOE_FP8BS_ARCHS",
    "ROW_WARPS_PER_SM_SMALL",
    "SCALE_LAYOUTS",
    "TAIL_ROWS",
    "VIEW_PAIR",
    "MoEWeightsE4M3BS",
    "check_aligned",
    "check_block_scales",
    "check_shape_fp8bs",
    "fp8_block_quantize",
    "fp8_block_quantize_plan",
    "fused_moe_fp8bs",
    "grouped_gemm_fp8bs",
    "quant_warps_per_row",
    "quantize_weight_blocks",
    "scale_mn_view",
]
