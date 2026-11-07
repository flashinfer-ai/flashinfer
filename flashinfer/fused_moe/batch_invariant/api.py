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

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from ...gemm.batch_invariant.api import persistent_clusters, sm_count_of
from ...gemm.batch_invariant.generator import Knobs
from . import spec as S
from .blocks.epilogue import GroupedChain

_CONST: Dict[Tuple[int, int, float], torch.Tensor] = {}


def _const(dev: torch.device, n: int, value: float) -> torch.Tensor:
    """A cached fp32 constant tensor (``value`` repeated ``n`` times) per device: the route kernel reads an absent bias
    or scale from it instead of a per-call fill (a fill is one more kernel in the caller's stream or graph)."""
    key = (
        dev.index if dev.index is not None else torch.cuda.current_device(),
        n,
        value,
    )
    t = _CONST.get(key)
    if t is None:
        t = torch.full((n,), value, dtype=torch.float32, device=dev)
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
) -> torch.Tensor:
    """``out[p] = chain(a[p] @ w[e(p)].T)`` for the expert segments ``counts`` (int32 ``[E]``, group order = expert order;
    rows ``[off_e, off_e + counts[e])`` of ``a`` belong to expert ``e``). ``a [Tk, K]`` bf16 row-major, ``w [E, N, K]`` bf16
    row-major; ``chain`` ``plain`` -> ``[Tk, N]`` in ``out_dtype`` (bf16 / f16 / f32), ``swiglu`` -> ``[Tk, N / 2]`` bf16
    (``w`` interleaved as :func:`interleave_fc1`), ``relu2`` -> ``[Tk, N]`` bf16 (the ungated chain). The persistent grid is a
    launch argument: the device's co-resident cluster capacity for the record's cluster size."""
    from .blocks.compile import get_compiled_grouped

    if (
        a.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or a.dim() != 2
        or w.dim() != 3
    ):
        raise TypeError("grouped_gemm takes bf16 a [Tk, K] and w [E, N, K]")
    Tk, K = a.shape
    E, N, K2 = w.shape
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
    kernel = get_compiled_grouped(config, chain)
    if Tk == 0:
        return out
    kernel(
        a.unsqueeze(-1),
        w.permute(1, 2, 0),
        out,
        counts,
        persistent_clusters(a.device, config),
    )
    return out


def expert_bucket(E: int) -> int:
    return max(32, S.pow2ceil(E))


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
    binary32 scores, the method's weight formula)."""
    from .route import get_compiled_route

    method = S.METHODS[method] if isinstance(method, str) else method
    if logits.dim() != 2 or logits.stride(1) != 1 or logits.dtype not in _IN:
        raise TypeError("route: logits must be a row-major f32 or bf16 [T, E]")
    T, E = logits.shape
    if E > 1024:
        raise ValueError(
            f"route: {E} experts; the route kernel scores at most 1024 experts"
        )
    if not 1 <= top_k <= min(E, 16):
        raise ValueError(f"route: top_k {top_k} must be in [1, min(E, 16)]")
    if (bias is not None) != method.bias:
        raise ValueError(
            "route: a bias is given iff the method selects on sigmoid(l) + bias"
        )
    if (scaling_factor is not None) != method.scaled:
        raise ValueError(
            "route: a scaling factor is given iff the method scales the weights"
        )
    dev = logits.device
    b = (
        bias.to(device=dev, dtype=torch.float32).contiguous()
        if bias is not None
        else _const(dev, E, 0.0)
    )
    if tuple(b.shape) != (E,):
        raise ValueError(f"route: bias must be [E] = [{E}]")
    if scaling_factor is None:
        s = _const(dev, 1, 1.0)
    elif isinstance(scaling_factor, torch.Tensor):
        s = scaling_factor.to(device=dev, dtype=torch.float32).reshape(1)
    else:
        s = _const(dev, 1, float(scaling_factor))
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


PLAN_MULTI_MIN_N = 16384  # from this many slots the plan runs as the two-launch block form (the same outputs)


def permute_plan(
    experts: torch.Tensor, E: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(counts [E] int32, dest [T, k] int32, src [Tk] int32)``: the deterministic permutation of the expanded rows,
    grouped by expert in expert order and, inside a group, in token-slot order (a function of the routing alone)."""
    from .permute import PLAN_BLOCK, get_compiled_plan, get_compiled_plan_multi

    T, k = experts.shape
    n = T * k
    dev = experts.device
    dest = torch.empty(n, dtype=torch.int32, device=dev)
    src = torch.empty(n, dtype=torch.int32, device=dev)
    counts = torch.empty(E, dtype=torch.int32, device=dev)
    flat = experts.reshape(-1)
    if not flat.is_contiguous() or flat.dtype != torch.int32:
        raise TypeError("permute_plan: experts must be a contiguous int32 [T, k]")
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


def fused_moe(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: Union[MoEWeights, Tuple[torch.Tensor, torch.Tensor]],
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    return_intermediates: bool = False,
):
    """The MoE FFN for pre-routed tokens: ``out[t] = sum_r w[t, r] * act(x_t W1[e]^T) W2[e]^T`` with the class's rounding
    (gated: ``act`` = SwiGLU on the gate / up pair; ungated: ReLU^2 on the single accumulator), the ``k`` slots folded in
    slot order in fp32 and converted once. ``config1`` / ``config2``: the grouped GEMMs' records (any record of the space
    gives the same bits; the dispatcher's pick by default)."""
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
    if (
        tuple(token_final_scales.shape) != (T, k)
        or token_final_scales.dtype != torch.float32
    ):
        raise TypeError("fused_moe: token_final_scales must be a float32 [T, k]")
    experts = token_selected_experts.contiguous()
    scales = token_final_scales.contiguous()
    E, I = w.num_experts, w.intermediate_size
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
    counts, dest, src = permute_plan(experts, E)
    Tk = T * k
    x_perm = torch.empty(Tk, H, dtype=torch.bfloat16, device=x.device)
    if Tk:
        get_compiled_gather(
            k, "bf16", warps_per_row(Tk, H // 8, sm_count_of(x.device))
        )(x, src, x_perm, Tk, H)
    a = grouped_gemm(x_perm, w.w1i, counts, config1, w.chain1)  # [Tk, I] bf16
    y = grouped_gemm(a, w.w2, counts, config2, "plain", torch.bfloat16)  # [Tk, H] bf16
    if T:
        get_compiled_combine(
            k, "bf16", "bf16", warps_per_token(T, H, sm_count_of(x.device))
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
