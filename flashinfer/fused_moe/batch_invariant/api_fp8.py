# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8 per-tensor MoE classes ``moe.e4m3.<act>`` (W8A8): the weight view, the static input quantization, the grouped fp8
GEMMs (placements of the dense fp8 class) and the fused FFN on pre-routed tokens.

The routing, the deterministic permutation, the slot-order combine and the launch glue (the operand rules, the gather of the
permuted rows, the grid) are the bf16 classes' (:mod:`.api`); the kernels are the shared grouped and swapped bodies at the
fp8 chains (:mod:`.blocks.epilogue.Fp8Chain`). The scales follow
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
from ...gemm.batch_invariant.generator import Knobs
from . import spec as S
from .api import (
    _check_fused_inputs,
    _check_grouped_operands,
    _grouped_out,
    combine_rows,
    gather_rows,
    interleave_fc1,
    launch_grid,
    permute_plan,
    swapped_pads,
)
from .blocks.epilogue import GEMM2_CHAIN, Fp8Chain
from .blocks.scheduler import MAX_E
from .reference_fp8 import dequant_scale, quantize_e4m3_static
from .spec_fp8 import E4M3_MAX

CLASS = "moe.e4m3"
ALIGN_E4M3 = 16  # e4m3 elements per 16-byte row: the TMA alignment of every operand
_OUT_DTYPE = {"bf16": torch.bfloat16, "f32": torch.float32, "e4m3": torch.float8_e4m3fn}


def weight_multipliers(w: torch.Tensor) -> torch.Tensor:
    """FlashInfer's per-expert E4M3 multipliers of ``w [E, N, K]``: ``448 / amax`` (1 for an all-zero expert), fp32; an
    expert whose amax is below ``448 / FLT_MAX`` takes the largest finite multiplier, so finite weights never quantize through
    an infinite multiplier."""
    amax = w.detach().to(torch.float32).abs().amax(dim=(-1, -2))
    if not bool(torch.isfinite(amax).all()):
        raise ValueError(f"{CLASS}: the expert weights must be finite")
    finite_floor = E4M3_MAX / torch.finfo(torch.float32).max
    return torch.where(
        amax > 0, E4M3_MAX / amax.clamp_min(finite_floor), torch.ones_like(amax)
    ).to(torch.float32)


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
    the grouped body, a swapped record (``config.swap_ab``: 128 or 64 weight rows x 16 / 32 tokens, the weights on the MMA's M
    side, :mod:`.blocks.swap`) the swapped body on the same dense rows -- a placement, the same bits. ``Tk`` (default:
    the rows of ``out``, else of ``a``) is the expanded row count; ``a`` may carry more rows (the swapped body's last token box then
    stays inside the buffer instead of being zero-filled past it). The persistent grid is a launch argument."""
    from .blocks.compile import (
        can_implement,
        get_compiled_grouped,
        get_compiled_swapped,
    )

    R, K, E, N, Tk = _check_grouped_operands(
        "grouped_gemm_fp8",
        a,
        w,
        counts,
        Tk,
        out,
        torch.float8_e4m3fn,
        ALIGN_E4M3,
        "e4m3",
    )
    if (
        s_w.dtype != torch.float32
        or s_w.numel() != E
        or s_a.numel() != 1
        or inv.numel() != 1
    ):
        raise ValueError("grouped_gemm_fp8: s_w [E] fp32, s_a and inv 1-element fp32")
    if not can_implement(config, N, K, E, chain):
        raise ValueError(
            f"grouped_gemm_fp8: {config.describe()} does not implement (N={N}, K={K}, E={E}, {chain.describe()})"
        )
    n_out = N // 2 if chain.pair else N
    out = _grouped_out(
        "grouped_gemm_fp8", out, Tk, n_out, _OUT_DTYPE[chain.out_fmt], a.device
    )
    out_k = out.view(torch.uint8) if chain.out_fmt == "e4m3" else out
    if Tk == 0:
        return out
    kernel = (
        get_compiled_swapped(config, chain)
        if config.swap_ab
        else get_compiled_grouped(config, chain)
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
        launch_grid(a.device, config, Tk, E, N),
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
    w_round: bool = False,
):
    """The W8A8 MoE FFN for pre-routed tokens: ``xq [T, H]`` e4m3 (the caller's static quantization, ``x = q / m_x``),
    ``token_selected_experts [T, k]`` int32, ``token_final_scales [T, k]`` fp32; ``out[t] = sum_r w[t, r] * y_r`` with the class's
    chains and rounding (:mod:`.spec_fp8`), the ``k`` slots folded in slot order in fp32 and converted once. ``config1`` /
    ``config2``: the grouped records (any record of the space gives the same bits; the dispatcher's pick by default).
    ``w_round``: the f32 weights enter the fold as their bf16 rounding (the pre-routed packed-word convention), by the combine
    kernel. An id outside ``[0, E)`` is a dropped slot (the term ``+0``), as in the bf16 op."""
    from .schedule_fp8 import default_configs_fp8

    T, H, k, experts, scales, out = _check_fused_inputs(
        "fused_moe_fp8",
        xq,
        token_selected_experts,
        token_final_scales,
        out,
        w.hidden_size,
        torch.float8_e4m3fn,
        ALIGN_E4M3,
        "e4m3",
    )
    E, I = w.num_experts, w.intermediate_size
    check_shape_e4m3(H, I, E)
    dev = xq.device
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
    pad1, pad2 = swapped_pads(config1, config2)
    x_buf, x_perm = gather_rows(xq, src, k, Tk, pad1, "e4m3")
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
    combine_rows(y, dest, scales, out, k, w_round)
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
