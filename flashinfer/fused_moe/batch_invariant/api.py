# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The bf16 MoE classes ``moe.bf16.swiglu`` and ``moe.bf16.relu2``: routing, the deterministic permutation, the grouped
GEMMs (placements of the dense GEMM class), the slot-order combine.

``route`` turns router logits into ``(experts, weights)`` in slot order; ``fused_moe`` runs the FFN on pre-routed tokens
(the pre-routed convention of FlashInfer's MoE operators: ``token_selected_experts [T, k]`` int32,
``token_final_scales [T, k]`` fp32); ``moe_ffn`` composes the two. Every output bit is a function of the token's own row,
its ``k`` experts and routing weights, and the expert weights: batch size, batch composition, row position, expert load
and the configuration chosen for the grouped GEMMs never enter the dataflow (the specification is :mod:`.spec`; the
proofs are the ``MoE`` Lean module of the mode).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from ...gemm.batch_invariant.api import persistent_clusters, sm_count_of
from ...gemm.batch_invariant.generator import Knobs
from . import spec as S
from .blocks.epilogue import GroupedChain
from .blocks.scheduler import MAX_E

_CONST: Dict[Tuple[int, int, float], torch.Tensor] = {}
_CONST_VALUES = (
    0.0,
    1.0,
)  # the cached constants: the absent bias (per expert count) and the unit scale


def _const(dev: torch.device, n: int, value: float) -> torch.Tensor:
    """An fp32 constant tensor (``value`` repeated ``n`` times) the route kernel reads for an absent bias or scale. The
    defaults (``+0`` per expert count, ``1``) are cached per device, so the cache is bounded by the devices and the expert
    counts seen; a cached constant is built on the host and copied with a blocking copy, so its bytes are on the device
    before the call returns and any stream may read it. Any other value, and every call under CUDA graph capture (a
    graph-pool allocation whose fill runs only at replay: it serves that capture alone), is a fresh device fill, one more
    kernel in the caller's stream or graph (a tensor argument avoids it)."""
    cacheable = value in _CONST_VALUES and math.copysign(1.0, value) > 0
    if not cacheable or torch.cuda.is_current_stream_capturing():
        return torch.full((n,), value, dtype=torch.float32, device=dev)
    key = (
        dev.index if dev.index is not None else torch.cuda.current_device(),
        n,
        value,
    )
    t = _CONST.get(key)
    if t is None:
        t = torch.full((n,), value, dtype=torch.float32).to(dev)
        _CONST[key] = t
    return t


_IN = {torch.float32: "f32", torch.bfloat16: "bf16"}
_OUT = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
_DTYPE_OF = {v: k for k, v in _OUT.items()}


def interleave_fc1(w1: torch.Tensor) -> torch.Tensor:
    """FlashInfer's canonical ``fc1 [E, 2I, H]`` (up rows first, then gate rows) -> the GEMM1 weight with gate and up
    interleaved in groups of ``GROUP`` rows (``u_0..u_7, g_0..g_7, u_8..``), so the two accumulators of one SwiGLU term
    sit in one thread's fragment. A row permutation: no arithmetic."""
    from .blocks.epilogue import GROUP

    E, N2, H = w1.shape
    I = N2 // 2
    if N2 % (2 * GROUP):
        raise ValueError(f"intermediate size {I} must be a multiple of {GROUP}")
    up, gate = w1[:, :I], w1[:, I:]
    return (
        torch.stack(
            [
                up.reshape(E, I // GROUP, GROUP, H),
                gate.reshape(E, I // GROUP, GROUP, H),
            ],
            dim=2,
        )
        .reshape(E, N2, H)
        .contiguous()
    )


@dataclass
class MoEWeights:
    """The expert weights as the kernels consume them: ``w1i [E, 2I, H]`` interleaved (gated: the (gate, up) pair in one
    thread's fragment) or ``w1i [E, I, H]`` as given (``gated=False``: no gate / up pairing), ``w2 [E, H, I]``; the GEMM1
    chain ``act`` (a gated chain of ``spec.ACTS`` or an ungated one of ``spec.UNGATED_ACTS``) with its ``params``."""

    w1i: torch.Tensor
    w2: torch.Tensor
    gated: bool = True
    act: str = "silu"
    params: Tuple[Optional[float], ...] = ()

    @staticmethod
    def prepare(
        fc1: torch.Tensor,
        fc2: torch.Tensor,
        interleaved: bool = False,
        gated: bool = True,
        act: Optional[str] = None,
        params: Sequence[Optional[float]] = (),
    ) -> "MoEWeights":
        """``act`` defaults to the class's default chain of the gating (``silu`` gated, ``relu2`` ungated)."""
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
                "the bf16 MoE classes take bf16 expert weights fc1 [E, N1, H], fc2 [E, H, I]"
            )
        E, N1, H = fc1.shape
        if not 1 <= E <= MAX_E:
            raise ValueError(
                f"the bf16 MoE classes take 1 to {MAX_E} experts (the plan's and the grouped GEMMs' tables), got {E}"
            )
        I = N1 // 2 if gated else N1
        if tuple(fc2.shape) != (E, H, I):
            raise ValueError(
                f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)} (gated={gated})"
            )
        if H % 8 or I % 8:
            raise ValueError(
                "H and I must be multiples of 8 (16-byte-aligned bf16 operands)"
            )
        if not gated:
            return MoEWeights(fc1.contiguous(), fc2.contiguous(), False, act, key)
        return MoEWeights(
            fc1.contiguous() if interleaved else interleave_fc1(fc1),
            fc2.contiguous(),
            True,
            act,
            key,
        )

    @property
    def num_experts(self) -> int:
        return self.w1i.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.w1i.shape[2]

    @property
    def intermediate_size(self) -> int:
        return self.w1i.shape[1] // 2 if self.gated else self.w1i.shape[1]

    @property
    def chain1(self) -> GroupedChain:
        """The GEMM1 epilogue chain of the class: the gated ``act`` on the (gate, up) pair or the ungated ``act`` on the
        single accumulator, with the parameters."""
        return GroupedChain(
            "glu" if self.gated else "act", self.act, "bf16", self.params
        )

    def view(self) -> Dict[str, object]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for this backend."""
        return {
            "w1i": self.w1i,
            "w2": self.w2,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
        }

    @staticmethod
    def from_view(view: Dict[str, object]) -> "MoEWeights":
        gated = bool(view.get("gated", True))
        return MoEWeights(
            view["w1i"],  # type: ignore[arg-type]
            view["w2"],  # type: ignore[arg-type]
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),  # type: ignore[arg-type]
        )


def grouped_gemm(
    a: torch.Tensor,
    w: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: Union[str, GroupedChain] = "plain",
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    Tk: Optional[int] = None,
    grid: Optional[int] = None,
) -> torch.Tensor:
    """``out[p] = chain(a[p] @ w[e(p)].T)`` for the expert segments ``counts`` (int32 ``[E]``, group order = expert order;
    rows ``[off_e, off_e + counts[e])`` of ``a`` belong to expert ``e``). ``a [R >= Tk, K]`` bf16 row-major, ``w [E, N, K]``
    bf16 row-major; ``chain`` ``plain`` -> ``[Tk, N]`` in ``out_dtype`` (bf16 / f16 / f32), ``swiglu`` -> ``[Tk, N / 2]`` bf16
    (``w`` interleaved as :func:`interleave_fc1`), ``relu2`` -> ``[Tk, N]`` bf16 (the ungated chain). The record selects the
    body: a dense record runs the grouped body, a swapped record (``config.swap_ab``: a 128- or 64-row weight tile x 8 / 16 /
    32 tokens, the weights on the MMA's M side, :mod:`.blocks.swap_bf16`) the swapped body on the same dense rows -- a
    placement, the same bits. ``Tk`` (default: the rows of ``out``, else of ``a``) is the expanded row count; ``a`` may carry more rows (the
    swapped body's last token box then stays inside the buffer instead of being zero-filled past it). The persistent grid is a
    launch argument: the device's co-resident cluster capacity for the record's cluster size."""
    from .blocks.compile import get_compiled_grouped, get_compiled_swapped
    from .schedule import launch_tile_bound

    if (
        a.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or a.dim() != 2
        or w.dim() != 3
    ):
        raise TypeError("grouped_gemm takes bf16 a [Tk, K] and w [E, N, K]")
    R, K = a.shape
    E, N, K2 = w.shape
    if E > MAX_E:
        raise ValueError(
            f"grouped_gemm: {E} experts; the group tables hold at most {MAX_E}"
        )
    if (
        K != K2
        or a.stride(1) != 1
        or w.stride(2) != 1
        or w.stride(1) != K
        or a.stride(0) % 8
        or w.stride(0) % 8
    ):
        raise ValueError(
            "grouped_gemm: K-contiguous, 16-byte-aligned operands of one K "
            f"(a {tuple(a.shape)} strides {tuple(a.stride())}, w {tuple(w.shape)} strides {tuple(w.stride())})"
        )
    if Tk is None:
        Tk = out.shape[0] if out is not None else R
    if not 0 <= Tk <= R:
        raise ValueError(f"grouped_gemm: Tk {Tk} must lie in [0, {R}] (the rows of a)")
    if (
        counts.dtype != torch.int32
        or tuple(counts.shape) != (E,)
        or not counts.is_contiguous()
    ):
        raise ValueError("grouped_gemm: counts must be a contiguous int32 [E]")
    if isinstance(chain, str):
        if chain not in ("plain", "swiglu", "relu2"):
            raise ValueError(f"unknown chain {chain!r}")
        chain = GroupedChain.legacy(chain, _OUT[out_dtype])
    n_out = N // 2 if chain.pair else N
    out_dtype = _DTYPE_OF[
        chain.out_fmt
    ]  # the chain's output format (the gated chains convert once, to bf16)
    if out is None:
        out = torch.empty(Tk, n_out, dtype=out_dtype, device=a.device)
    if (
        tuple(out.shape) != (Tk, n_out)
        or out.dtype != out_dtype
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "grouped_gemm: out must be a 16-byte-aligned row-major [Tk, n_out] of the chain's dtype"
        )
    kernel = (
        get_compiled_swapped(config, chain)
        if config.swap_ab
        else get_compiled_grouped(config, chain)
    )
    if Tk == 0:
        return out
    if not config.swap_ab and Tk != R:
        a = a[
            :Tk
        ]  # the grouped body's A has the output's rows (one symbol); the swapped body's R is free
    if grid is None:
        # the persistent grid clipped to the launch's tile bound (a placement: which CTA computes a tile)
        grid = min(
            persistent_clusters(a.device, config), launch_tile_bound(config, Tk, E, N)
        )
    kernel(
        a.unsqueeze(-1),
        w.permute(1, 2, 0),
        out,
        counts,
        grid,
    )
    return out


def grouped_gemm_gather(
    x: torch.Tensor,
    src: torch.Tensor,
    k_div: int,
    w: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: Union[str, GroupedChain] = "plain",
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    Tk: Optional[int] = None,
    grid: Optional[int] = None,
) -> torch.Tensor:
    """``out[p] = chain(x[src[p] // k_div] @ w[e(p)].T)`` for the expert segments ``counts``: :func:`grouped_gemm` on a
    swapped record with the rows gathered by the kernel (the TMA's gather4 mode) from the unpermuted ``x [T, K]`` bf16 -- no
    permuted activation tensor, no gather launch; ``src [>= Tk]`` int32 the plan's slot of every expanded row. The same bits
    as :func:`grouped_gemm` on the materialised permutation (the stage holds the same bytes, the fold is the same). The
    gather form belongs to the swapped records (a dense record raises)."""
    from .blocks.compile import get_compiled_swapped
    from .schedule import launch_tile_bound

    if (
        x.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or x.dim() != 2
        or w.dim() != 3
    ):
        raise TypeError("grouped_gemm_gather takes bf16 x [T, K] and w [E, N, K]")
    if not config.swap_ab:
        raise ValueError(
            "grouped_gemm_gather: the gather form belongs to the swapped records"
        )
    T, K = x.shape
    E, N, K2 = w.shape
    if E > MAX_E:
        raise ValueError(
            f"grouped_gemm_gather: {E} experts; the group tables hold at most {MAX_E}"
        )
    src = src.reshape(-1)
    if Tk is None:
        Tk = out.shape[0] if out is not None else int(src.shape[0])
    if (
        K != K2
        or x.stride(1) != 1
        or w.stride(2) != 1
        or w.stride(1) != K
        or x.stride(0) % 8
        or w.stride(0) % 8
    ):
        raise ValueError(
            "grouped_gemm_gather: K-contiguous, 16-byte-aligned operands of one K "
            f"(x {tuple(x.shape)} strides {tuple(x.stride())}, w {tuple(w.shape)} strides {tuple(w.stride())})"
        )
    if (
        counts.dtype != torch.int32
        or tuple(counts.shape) != (E,)
        or not counts.is_contiguous()
    ):
        raise ValueError("grouped_gemm_gather: counts must be a contiguous int32 [E]")
    if (
        src.dtype != torch.int32
        or not src.is_contiguous()
        or src.shape[0] < Tk
        or k_div < 1
    ):
        raise ValueError(
            "grouped_gemm_gather: src must be a contiguous int32 [>= Tk] and k_div >= 1"
        )
    if isinstance(chain, str):
        if chain not in ("plain", "swiglu", "relu2"):
            raise ValueError(f"unknown chain {chain!r}")
        chain = GroupedChain.legacy(chain, _OUT[out_dtype])
    n_out = N // 2 if chain.pair else N
    out_dtype = _DTYPE_OF[chain.out_fmt]
    if out is None:
        out = torch.empty(Tk, n_out, dtype=out_dtype, device=x.device)
    if (
        tuple(out.shape) != (Tk, n_out)
        or out.dtype != out_dtype
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "grouped_gemm_gather: out must be a 16-byte-aligned row-major [Tk, n_out] of the chain's dtype"
        )
    kernel = get_compiled_swapped(config, chain, gather=True)
    if Tk == 0:
        return out
    if grid is None:
        grid = min(
            persistent_clusters(x.device, config), launch_tile_bound(config, Tk, E, N)
        )
    kernel(
        x,
        src,
        int(k_div),
        w.permute(1, 2, 0),
        out,
        counts,
        grid,
    )
    return out


def expert_bucket(E: int) -> int:
    return max(32, S.pow2ceil(E))


def _routing_args(
    name: str,
    logits: torch.Tensor,
    top_k: int,
    method: Union[S.RoutingMethod, str],
    bias: Optional[torch.Tensor],
    scaling_factor: Optional[Union[float, torch.Tensor]],
) -> Tuple[S.RoutingMethod, int, int, torch.Tensor, torch.Tensor]:
    """The routing entry points' shared validation and constants: ``(method, T, E, bias, scale)`` for :func:`route` and
    :func:`route_plan` (``name`` prefixes the errors). The bounds (``E <= MAX_E``: the route kernels' widest bucket is the
    plan's table size; ``1 <= top_k <= min(E, MAX_TOP_K)``), the group rule of a group-limited method
    (:func:`.spec.group_rule_error`: the groups a term of the DeepSeekV3 method alone, its shape rule), a bias exactly for the
    methods that select on ``sigmoid(l) + bias`` and a scaling factor exactly for the scaled methods; a method name resolves
    through :func:`.spec.method_named` (a grouped name as ``<method>_g<n>x<t>``); the absent bias and the unit scale are the
    cached constants (:func:`_const`), a Python scaling factor other than ``1`` a fill per call, a one-element tensor read as
    given."""
    method = S.method_named(method) if isinstance(method, str) else method
    if logits.dim() != 2 or logits.stride(1) != 1 or logits.dtype not in _IN:
        raise TypeError(f"{name}: logits must be a row-major f32 or bf16 [T, E]")
    T, E = logits.shape
    if E > MAX_E:
        raise ValueError(
            f"{name}: {E} experts; the routing kernels score at most {MAX_E} experts"
        )
    if not 1 <= top_k <= min(E, S.MAX_TOP_K):
        raise ValueError(f"{name}: top_k {top_k} must be in [1, min(E, {S.MAX_TOP_K})]")
    rule = S.group_rule_error(method, E, top_k)
    if rule is not None:
        raise ValueError(
            f"{name}: n_group {method.n_group}, topk_group {method.topk_group} over {E} experts for top_k "
            f"{top_k}: {rule}"
        )
    if (bias is not None) != method.bias:
        raise ValueError(
            f"{name}: a bias is given iff the method selects on sigmoid(l) + bias"
        )
    if (scaling_factor is not None) != method.scaled:
        raise ValueError(
            f"{name}: a scaling factor is given iff the method scales the weights"
        )
    dev = logits.device
    b = (
        bias.to(device=dev, dtype=torch.float32).contiguous()
        if bias is not None
        else _const(dev, E, 0.0)
    )
    if tuple(b.shape) != (E,):
        raise ValueError(f"{name}: bias must be [E] = [{E}]")
    if scaling_factor is None:
        s = _const(dev, 1, 1.0)
    elif isinstance(scaling_factor, torch.Tensor):
        s = scaling_factor.to(device=dev, dtype=torch.float32).reshape(1)
    else:
        s = _const(dev, 1, float(scaling_factor))
    return method, T, E, b, s


def route(
    logits: torch.Tensor,
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    experts: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(experts [T, k] int32, weights [T, k] f32)`` in slot (rank) order from ``logits [T, E]`` (f32 or bf16): the
    class's routing (:mod:`.spec`: the declared ``exp``, the strict selection order ``(score desc, index asc)`` on exact
    binary32 scores, the method's weight formula). A NaN logit is never selected and contributes nothing to the softmax
    normalisation (its term is ``+0``; the row maximum skips it); when fewer than ``top_k`` experts of a row have a
    non-NaN score, each remaining slot takes the lowest expert index the row has not taken, with a NaN weight (the row's
    output is NaN; every id stays in ``[0, E)``). A Python ``scaling_factor`` other than ``1`` costs a fill per call; a
    one-element tensor is read as given."""
    from .route import get_compiled_route

    method, T, E, b, s = _routing_args(
        "route", logits, top_k, method, bias, scaling_factor
    )
    dev = logits.device
    if experts is None:
        experts = torch.empty(T, top_k, dtype=torch.int32, device=dev)
    if weights is None:
        weights = torch.empty(T, top_k, dtype=torch.float32, device=dev)
    if tuple(experts.shape) != (T, top_k) or tuple(weights.shape) != (T, top_k):
        raise ValueError("route: experts and weights must be [T, top_k]")
    if not experts.is_contiguous() or not weights.is_contiguous():
        raise ValueError("route: experts and weights must be contiguous")
    if T:
        get_compiled_route(expert_bucket(E), top_k, method, _IN[logits.dtype])(
            logits, b, s, experts, weights, T, E
        )
    return experts, weights


def route_plan(
    logits: torch.Tensor,
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    align: int = 1,
    dual: bool = False,
    pad_align: int = 128,
    local: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, ...]:
    """``route`` and ``permute_plan`` in one launch for small token counts: ``(experts [T, k] int32, weights [T, k] f32,
    counts [E] int32, dest [T, k] int32, src [T k] int32)``, the same bits and integers as the two calls (the routing is the
    route kernel's per-token code, the plan the two-launch plan's ranking); ``T k <= route_plan.ROUTE_PLAN_MAX_SLOTS``.
    ``local = (offset, E_local)``: an expert-parallel shard -- the routing over the ``E`` global experts as given (global
    ids in ``experts``), the plan over the local experts ``[offset, offset + E_local)`` by their local ids (``counts
    [E_local]``), every other slot dropped (``dest -1``): the integers of ``permute_plan(experts - offset, E_local)``.
    ``align > 1``: the group offsets rounded up to ``align`` rows (the block-scaled classes' padded packing) and ``src`` of the
    padded bound's rows with ``-1`` in every padding slot, as the padded two-launch plan gives them. ``dual`` (with ``align = 1``):
    ``(..., dest_pad [T, k] int32, pos_pad [T k] int32)`` follow -- the positions with the groups padded to ``pad_align`` rows and
    the padded position of every dense row, ``permute_plan_dual``'s integers (the block-scaled classes' dense packing)."""
    from .route_plan import ROUTE_PLAN_MAX_SLOTS, get_compiled_route_plan

    method, T, E, b, s = _routing_args(
        "route_plan", logits, top_k, method, bias, scaling_factor
    )
    dev = logits.device
    if T * top_k > ROUTE_PLAN_MAX_SLOTS:
        raise ValueError(
            f"route_plan: {T * top_k} expanded rows; the fused plan holds at most {ROUTE_PLAN_MAX_SLOTS}"
        )
    if align < 1 or align & (align - 1):
        raise ValueError(f"route_plan: align {align} must be a power of two")
    if dual and (align > 1 or pad_align < 2):
        raise ValueError(
            "route_plan: the dual form takes align 1 and a padded alignment above 1"
        )
    e_off, E_plan = (0, E) if local is None else (int(local[0]), int(local[1]))
    if not (e_off >= 0 and E_plan >= 1 and e_off + E_plan <= E):
        raise ValueError(
            f"route_plan: local experts [{e_off}, {e_off + E_plan}) must lie in [0, {E})"
        )
    n = T * top_k
    n_src = n if align == 1 else -(-(n + (align - 1) * min(E_plan, n)) // align) * align
    experts = torch.empty(T, top_k, dtype=torch.int32, device=dev)
    weights = torch.empty(T, top_k, dtype=torch.float32, device=dev)
    counts = torch.empty(E_plan, dtype=torch.int32, device=dev)
    dest = torch.empty(n, dtype=torch.int32, device=dev)
    src = torch.empty(n_src, dtype=torch.int32, device=dev)
    dest_pad = torch.empty(n, dtype=torch.int32, device=dev) if dual else dest
    pos_pad = torch.empty(n, dtype=torch.int32, device=dev) if dual else dest
    if T:
        get_compiled_route_plan(
            expert_bucket(E), top_k, method, _IN[logits.dtype], align, dual, pad_align
        )(
            logits,
            b,
            s,
            experts,
            weights,
            counts,
            dest,
            src,
            dest_pad,
            pos_pad,
            T,
            E,
            E_plan,
            e_off,
        )
    else:
        counts.zero_()
    if dual:
        return (
            experts,
            weights,
            counts,
            dest.view(T, top_k),
            src,
            dest_pad.view(T, top_k),
            pos_pad,
        )
    return experts, weights, counts, dest.view(T, top_k), src


PLAN_MULTI_MIN_N = 16384  # from this many slots the plan runs as the two-launch block form (the same outputs)


def permute_plan(
    experts: torch.Tensor, E: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(counts [E] int32, dest [T, k] int32, src [Tk] int32)``: the deterministic permutation of the expanded rows,
    grouped by expert in expert order and, inside a group, in token-slot order (a function of the routing alone). An id
    outside ``[0, E)`` is a dropped slot: ``counts`` excludes it, its ``dest`` is ``-1`` and ``src`` holds ``-1`` past
    the valid rows."""
    from .permute import PLAN_BLOCK, get_compiled_plan, get_compiled_plan_multi

    if not 1 <= E <= MAX_E:
        raise ValueError(
            f"permute_plan: E {E} must lie in [1, {MAX_E}] (the plan kernel's expert tables)"
        )
    T, k = experts.shape
    if not 1 <= k <= S.MAX_TOP_K:
        raise ValueError(
            f"permute_plan: {k} slots per token; the kernels fold at most {S.MAX_TOP_K}"
        )
    flat = experts.reshape(-1)
    if not flat.is_contiguous() or flat.dtype != torch.int32:
        raise TypeError("permute_plan: experts must be a contiguous int32 [T, k]")
    n = T * k
    dev = experts.device
    dest = torch.empty(n, dtype=torch.int32, device=dev)
    src = torch.empty(n, dtype=torch.int32, device=dev)
    counts = torch.empty(E, dtype=torch.int32, device=dev)
    if n == 0:
        counts.zero_()
        return counts, dest.view(T, k), src
    if n >= PLAN_MULTI_MIN_N:
        nb = (n + PLAN_BLOCK - 1) // PLAN_BLOCK
        hist_blk = torch.empty(nb * E, dtype=torch.int32, device=dev)
        get_compiled_plan_multi(1, False, 128)(
            flat, counts, dest, src, dest, dest, hist_blk, n, E, nb
        )
    else:
        get_compiled_plan(1, False)(flat, counts, dest, src, dest, dest, n, E)
    return counts, dest.view(T, k), src


# The gather form's token tiles: the swapped GEMM1 gathers the tokens' rows itself (no gather launch) when its token tile is
# at most this wide. The gather stage streams one swizzle atom of K at a time where the box form's stage streams two, and a
# wide token tile exposes that slower stream; the narrow tiles of the decode regime hide it behind the launch they save
# (declared empirical: the in-op A/B of the two forms per token tile).
GATHER_TOKEN_TILE_MAX = 16


def fused_moe(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: Union[MoEWeights, Tuple[torch.Tensor, torch.Tensor]],
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    return_intermediates: bool = False,
    plan: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    finalize: bool = True,
    gather: Optional[bool] = None,
    grid_clip: bool = True,
    w_round: bool = False,
):
    """The MoE FFN for pre-routed tokens: ``out[t] = sum_r w[t, r] * act(x_t W1[e]^T) W2[e]^T`` with the class's rounding
    (gated: ``act`` = SwiGLU on the gate / up pair; ungated: ReLU^2 on the single accumulator), the ``k`` slots folded in
    slot order in fp32 and converted once. An expert id outside ``[0, E)`` is a dropped slot: neither gathered nor
    computed, the term ``+0`` in its token's fold whatever its weight (a token whose slots are all dropped gives a zero
    row). ``config1`` / ``config2``: the grouped GEMMs' records (any record of the space gives the same bits; the
    dispatcher's pick by default). ``finalize=False`` skips the combine launch and leaves ``out`` as given: the per-slot
    rows ``y`` and the positions ``dest`` of ``return_intermediates`` are then the unfinalized form of the op (a placement:
    the rows are the class's own intermediates, a function of the token, its expert and the routing alone). ``gather``:
    whether GEMM1 gathers the tokens' rows itself (the swapped body's gather form: no gather launch, no permuted tensor;
    ``x_perm`` of the intermediates is then None) -- by default when GEMM1's record is a swapped one with a narrow token
    tile (`GATHER_TOKEN_TILE_MAX`), a placement with the same bits either way; ``True`` on a dense record raises. ``grid_clip``: the GEMMs' persistent grids clipped to
    the launch's tile bound (the default), or the device's capacity (a placement either way). ``w_round``: the f32 ``weights`` enter the fold as their bf16
    rounding, done by the combine kernel (the pre-routed packed-word convention: the bits of rounding them on the host first,
    without the host's passes over the slots)."""
    from .combine import get_compiled_combine, warps_per_token
    from .permute import get_compiled_gather, warps_per_row
    from .schedule import default_configs

    if not isinstance(w, MoEWeights):
        w = MoEWeights.prepare(*w)
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.stride(1) != 1 or x.stride(0) % 8:
        raise TypeError("fused_moe: x must be a row-major, 16-byte-aligned bf16 [T, H]")
    T, H = x.shape
    if w.hidden_size != H:
        raise ValueError(f"fused_moe: x has H = {H}, the weights H = {w.hidden_size}")
    k = token_selected_experts.shape[1]
    if (
        tuple(token_selected_experts.shape) != (T, k)
        or token_selected_experts.dtype != torch.int32
    ):
        raise TypeError("fused_moe: token_selected_experts must be an int32 [T, k]")
    if not 1 <= k <= S.MAX_TOP_K:
        # the bound of the slots per token (a shard's k is the global routing's, so E does not bound it here)
        raise ValueError(
            f"fused_moe: {k} slots per token; the plan, gather and combine kernels fold at most {S.MAX_TOP_K}"
        )
    if (
        tuple(token_final_scales.shape) != (T, k)
        or token_final_scales.dtype != torch.float32
    ):
        raise TypeError("fused_moe: token_final_scales must be a float32 [T, k]")
    experts = token_selected_experts.contiguous()
    scales = token_final_scales.contiguous()
    E, I = w.num_experts, w.intermediate_size
    if E > MAX_E:
        raise ValueError(
            f"fused_moe: {E} experts; the plan and the grouped GEMMs hold at most {MAX_E}"
        )
    if out is None:
        out = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
    if (
        tuple(out.shape) != (T, H)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "fused_moe: out must be a row-major, 16-byte-aligned bf16 [T, H]"
        )
    if config1 is None or config2 is None:
        c1, c2 = default_configs(T, k, H, I, E, *_part(x.device), gated=w.gated)
        config1, config2 = config1 or c1, config2 or c2
    counts, dest, src = (
        plan if plan is not None else permute_plan(experts, E)
    )  # ``plan``: ``route_plan``'s outputs, the same integers
    Tk = T * k
    # a swapped record reads a token box of its tile's rows at every group's row: the dense buffer it reads carries that many
    # rows past the live ones, so the last tiles' boxes stay inside it (rows past the live ones are computed and never stored)
    pad1 = config1.tile_n if config1.swap_ab else 0
    pad2 = config2.tile_n if config2.swap_ab else 0
    a_buf = torch.empty(Tk + pad2, I, dtype=torch.bfloat16, device=x.device)
    g1 = None if grid_clip else persistent_clusters(x.device, config1)
    g2 = None if grid_clip else persistent_clusters(x.device, config2)
    if gather is None:
        gather = bool(config1.swap_ab) and config1.tile_n <= GATHER_TOKEN_TILE_MAX
    if gather:
        # the swapped GEMM1 gathers the tokens' rows itself through the plan's slots: no gather launch, no permuted tensor
        x_perm = None
        a = grouped_gemm_gather(
            x, src, k, w.w1i, counts, config1, w.chain1, out=a_buf[:Tk], Tk=Tk, grid=g1
        )  # [Tk, I] bf16
    else:
        x_buf = torch.empty(Tk + pad1, H, dtype=torch.bfloat16, device=x.device)
        x_perm = x_buf[:Tk]
        if Tk:
            get_compiled_gather(
                k, "bf16", warps_per_row(Tk, H // 8, sm_count_of(x.device))
            )(x, src, x_perm, Tk, H)
        a = grouped_gemm(
            x_buf, w.w1i, counts, config1, w.chain1, out=a_buf[:Tk], Tk=Tk, grid=g1
        )  # [Tk, I] bf16
    y = grouped_gemm(
        a_buf, w.w2, counts, config2, "plain", torch.bfloat16, Tk=Tk, grid=g2
    )  # [Tk, H] bf16
    if finalize and T:
        get_compiled_combine(
            k, "bf16", "bf16", warps_per_token(T, H, sm_count_of(x.device)), w_round
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


def combine_unfinalized(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The class's combine over the layer's unfinalized triple (``MoEFinalizeConfig(do_finalize=False)``: ``[gemm2_output,
    expert_weights, expanded_idx_to_permuted_idx]``): ``out[t] = cvt_bf16(fold_r add(acc, mul(f32(y[map[t, r]]), w[t, r])))``,
    the fold in slot order from ``+0``, a dropped slot (``map = -1``) the term ``+0`` whatever its weight -- the finalized
    layer's output bit for bit, through the combine kernel the finalized forward runs. ``gemm2_output [rows, H]`` bf16 (the
    rows in the class's permutation), ``expert_weights [T, k]`` fp32 or bf16 (a bf16 weight widens exactly: the fold is the
    same), ``expanded_idx_to_permuted_idx`` int32 of ``T k`` entries (flat or ``[T, k]``), each in ``[0, rows)`` or ``-1``
    (the triple as the layer returned it; the map is not validated on the host). The receiver of expert-parallel ranks'
    triples folds the ranks' rows the same way (one row buffer, one map, every token's ``k`` slots)."""
    from .combine import get_compiled_combine, warps_per_token

    y, w, dest = gemm2_output, expert_weights, expanded_idx_to_permuted_idx
    if y.dtype != torch.bfloat16 or y.dim() != 2 or y.stride(1) != 1 or y.stride(0) % 8:
        raise TypeError(
            "combine_unfinalized: gemm2_output must be a row-major, 16-byte-aligned bf16 [rows, H]"
        )
    if w.dim() != 2 or w.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            "combine_unfinalized: expert_weights must be an fp32 or bf16 [T, k]"
        )
    T, k = w.shape
    H = y.shape[1]
    if dest.dtype != torch.int32 or dest.numel() != T * k:
        raise TypeError(
            f"combine_unfinalized: expanded_idx_to_permuted_idx must be an int32 of {T} x {k} entries"
        )
    if not 1 <= k <= S.MAX_TOP_K:
        raise ValueError(
            f"combine_unfinalized: {k} slots per token; the combine kernel folds at most {S.MAX_TOP_K}"
        )
    if out is None:
        out = torch.empty(T, H, dtype=torch.bfloat16, device=y.device)
    if (
        tuple(out.shape) != (T, H)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(
            "combine_unfinalized: out must be a row-major, 16-byte-aligned bf16 [T, H]"
        )
    if T:
        get_compiled_combine(
            k, "bf16", "bf16", warps_per_token(T, H, sm_count_of(y.device)), False
        )(
            y,
            dest.reshape(T, k).contiguous(),
            w.to(torch.float32).contiguous(),
            out,
            T,
            H,
        )
    return out


def _part(device: torch.device) -> Tuple[int, str]:
    """``(sm_count, family)`` of the device for the dispatcher (the GEMM class's part detection)."""
    from ...gemm.batch_invariant.api import part_of

    return part_of(device)


def moe_ffn(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w: Union[MoEWeights, Tuple[torch.Tensor, torch.Tensor]],
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
) -> torch.Tensor:
    """Routing + the FFN (the shape of call of the routing-fused MoE operators)."""
    experts, weights = route(router_logits, top_k, method, bias, scaling_factor)
    return fused_moe(x, experts, weights, w, out, config1, config2)
