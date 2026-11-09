"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Exact KDA decode by replay of the current chunk through the pinned prefill kernel.

The native SM100/SM103 chunked KDA prefill (``recurrent_kda`` with
``backend="cute-dsl"``, the BT = 16 source kernel, fused Q/K l2-norm, in-kernel
gate and beta sigmoid, fp32 state) is deterministic, batch-invariant and
prefix-stable, and resuming it from its own fp32 state is bitwise exact at
multiples of the 16-token chunk. A decode that keeps the fp32 state at the last
chunk boundary plus the raw inputs of the tokens since that boundary, and
re-runs the pinned kernel on that partial chunk, therefore returns for token
``p`` exactly the row a one-shot prefill of the whole sequence would produce.
:class:`KdaExactDecodeState` with :func:`kda_exact_decode` / :func:`kda_exact_verify`
is the per-request reference form; :class:`KdaExactDecodePool` is the slot-addressed,
CUDA-graph-capturable serving form with the same bits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import torch

from . import kda_prefill_cute
from .kda import recurrent_kda
from .utils import get_compute_capability

CHUNK = 16
_HEAD_DIM = 128
_SUPPORTED_CC = ((10, 0), (10, 3))
_IO_DTYPE = torch.bfloat16
_BACKEND = "cute-dsl"
# The kernel's two routes (decomp / persistent engine) differ bitwise, so every call pins one.
ROUTE = "engine"


@dataclass(frozen=True)
class KdaLayerParams:
    """Per-layer constants of the fused KDA contract: ``A_log`` fp32 ``[HV]``,
    ``dt_bias`` fp32 ``[HV, 128]``, the finite negative gate lower bound the
    pinned kernel is compiled for, and the query scale (``None`` = ``1/sqrt(128)``)."""

    A_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float
    scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.A_log.dtype != torch.float32 or self.A_log.ndim != 1:
            raise ValueError("A_log must be fp32 [HV]")
        if self.dt_bias.dtype != torch.float32 or self.dt_bias.shape != (
            self.A_log.shape[0],
            _HEAD_DIM,
        ):
            raise ValueError(f"dt_bias must be fp32 [HV, {_HEAD_DIM}]")
        if not isinstance(self.lower_bound, (int, float)) or not (
            math.isfinite(self.lower_bound) and self.lower_bound < 0.0
        ):
            raise ValueError("lower_bound must be a finite negative float")


def _check_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("exact KDA decode requires CUDA tensors")
    cc = get_compute_capability(device)
    if cc not in _SUPPORTED_CC:
        raise NotImplementedError(
            "exact KDA decode is pinned to the native SM100/SM103 cute-dsl prefill kernel "
            f"(other builds of it are not validated bitwise); got compute capability {cc}"
        )


def _check_heads(num_heads: int, num_v_heads: int) -> None:
    if num_heads != num_v_heads:
        raise NotImplementedError(
            f"exact KDA decode supports HV == H only, got H={num_heads}, HV={num_v_heads}"
        )


def _check_state(state: torch.Tensor, name: str = "state") -> None:
    if state.dtype != torch.float32:
        raise ValueError(
            f"{name} must be float32 (a bf16 state rounds at every call boundary and is not exact), got {state.dtype}"
        )
    if state.ndim != 3 or state.shape[-1] != _HEAD_DIM or state.shape[-2] != _HEAD_DIM:
        raise ValueError(
            f"{name} must be [HV, {_HEAD_DIM}, {_HEAD_DIM}], got {tuple(state.shape)}"
        )
    if not state.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _pinned_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    params: KdaLayerParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # recurrent_kda updates initial_state in place: never hand it the caller's buffer.
    initial_state = initial_state.clone()
    # tokens == sequences would route to the T=1 decode kernels; a dummy two-token sequence keeps the prefill kernel.
    n_seq = int(cu_seqlens.numel()) - 1
    padded = q.shape[0] == n_seq
    if padded:
        q = torch.cat([q, q[:1], q[:1]], 0)
        k = torch.cat([k, k[:1], k[:1]], 0)
        v = torch.cat([v, v[:1], v[:1]], 0)
        g = torch.cat([g, g[:1], g[:1]], 0)
        beta = torch.cat([beta, beta[:1], beta[:1]], 0)
        cu_seqlens = torch.cat([cu_seqlens, cu_seqlens.new_tensor([q.shape[0]])])
        initial_state = torch.cat(
            [initial_state, torch.zeros_like(initial_state[:1])], 0
        )
    with kda_prefill_cute.force_route(ROUTE):
        res = _call_pinned(q, k, v, g, beta, cu_seqlens, initial_state, params)
    out, final_state = res[0], res[1]
    if final_state is None or final_state.dtype != torch.float32:
        raise RuntimeError("the pinned KDA prefill did not return an fp32 final state")
    if padded:
        return out[0][:-2], final_state[:-1]
    return out[0], final_state


def _call_pinned(q, k, v, g, beta, cu_seqlens, initial_state, params):
    return recurrent_kda(
        q[None].contiguous(),
        k[None].contiguous(),
        v[None].contiguous(),
        g[None].contiguous(),
        beta[None].contiguous(),
        A_log=params.A_log,
        dt_bias=params.dt_bias,
        scale=params.scale,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=params.lower_bound,
        cu_seqlens=cu_seqlens,
        beta_is_logit=True,
        backend=_BACKEND,
    )


@dataclass
class KdaExactDecodeState:
    """Per-request state of the replay decode.

    ``boundary_state`` is the fp32 ``[HV, V, K]`` state at the last chunk
    boundary ``16 * (position // 16)``; ``hist_*`` hold the raw inputs of the
    ``position % 16`` tokens since that boundary (rows ``[:n_hist]`` live):
    bf16 ``q, k`` ``[16, H, 128]``, ``v`` ``[16, HV, 128]``, raw gate ``g``
    ``[16, HV, 128]`` and beta logits ``[16, HV]``. ``current_state`` is the
    fp32 state after the last consumed token.
    """

    boundary_state: torch.Tensor
    position: int = 0
    hist_q: Optional[torch.Tensor] = None
    hist_k: Optional[torch.Tensor] = None
    hist_v: Optional[torch.Tensor] = None
    hist_g: Optional[torch.Tensor] = None
    hist_beta: Optional[torch.Tensor] = None
    current_state: Optional[torch.Tensor] = field(default=None)
    # (boundary_state, position, history, draft tokens) before the open draft, for the num_accepted_tokens rollback
    _snap: Optional[tuple] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        _check_state(self.boundary_state, "boundary_state")
        if self.position < 0:
            raise ValueError("position must be non-negative")
        if self.hist_q is None:
            if self.position % CHUNK != 0:
                raise ValueError(
                    f"position {self.position} is not a chunk boundary and no in-chunk "
                    "history was given; a state saved mid-chunk cannot be resumed exactly"
                )
        elif self.hist_q.shape[0] != CHUNK:
            raise ValueError("history buffers must hold 16 rows")
        if self.current_state is None:
            self.current_state = self.boundary_state

    @property
    def n_hist(self) -> int:
        return self.position % CHUNK

    @property
    def num_v_heads(self) -> int:
        return self.boundary_state.shape[0]

    def _ensure_history(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
    ) -> None:
        if self.hist_q is not None:
            return
        dev = self.boundary_state.device
        self.hist_q = torch.zeros((CHUNK, *q.shape), dtype=q.dtype, device=dev)
        self.hist_k = torch.zeros((CHUNK, *k.shape), dtype=k.dtype, device=dev)
        self.hist_v = torch.zeros((CHUNK, *v.shape), dtype=v.dtype, device=dev)
        self.hist_g = torch.zeros((CHUNK, *g.shape), dtype=g.dtype, device=dev)
        self.hist_beta = torch.zeros((CHUNK, v.shape[0]), dtype=_IO_DTYPE, device=dev)


def _check_token_inputs(q_t, k_t, v_t, g_t, beta_t) -> int:
    B = q_t.shape[0]
    for t, name, nd in (
        (q_t, "q_t", 3),
        (k_t, "k_t", 3),
        (v_t, "v_t", 3),
        (g_t, "g_t", 3),
        (beta_t, "beta_t", 2),
    ):
        if t.ndim != nd or t.shape[0] != B:
            raise ValueError(
                f"{name} must have {nd} dims with batch {B}, got {tuple(t.shape)}"
            )
        if t.dtype != _IO_DTYPE:
            raise ValueError(
                f"{name} must be bf16 (the fused KDA contract), got {t.dtype}"
            )
    for t, name in ((q_t, "q_t"), (k_t, "k_t"), (v_t, "v_t"), (g_t, "g_t")):
        if t.shape[-1] != _HEAD_DIM:
            raise ValueError(
                f"{name} must end in head dim {_HEAD_DIM}, got {tuple(t.shape)}"
            )
    if g_t.shape[1] != v_t.shape[1] or beta_t.shape[1] != v_t.shape[1]:
        raise ValueError("g_t and beta_t must have the value-head count")
    _check_heads(q_t.shape[1], v_t.shape[1])
    return B


def kda_exact_decode(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    state: Sequence[KdaExactDecodeState],
    params: KdaLayerParams,
) -> torch.Tensor:
    """Decode one token per request by replaying each request's current chunk.

    ``q_t, k_t``: bf16 ``[B, H, 128]`` raw (un-normalized); ``v_t``: bf16 ``[B, HV, 128]``;
    ``g_t``: bf16 ``[B, HV, 128]`` raw gate pre-activation; ``beta_t``: bf16 ``[B, HV]`` logits.
    Returns bf16 ``[B, HV, 128]``, bitwise equal to row ``position`` of a one-shot pinned
    prefill of each request's full sequence from its initial state. ``state`` is updated in place.
    """
    B = _check_token_inputs(q_t, k_t, v_t, g_t, beta_t)
    if len(state) != B:
        raise ValueError(f"got {B} tokens for {len(state)} request states")
    if B == 0:
        raise ValueError("empty batch")
    _check_device(q_t.device)

    lens: List[int] = []
    for i, s in enumerate(state):
        _check_state(s.boundary_state, f"state[{i}].boundary_state")
        if s.num_v_heads != v_t.shape[1]:
            raise ValueError(
                f"state[{i}] has {s.num_v_heads} heads, v_t has {v_t.shape[1]}"
            )
        s._ensure_history(q_t[i], k_t[i], v_t[i], g_t[i])
        n = s.n_hist
        s.hist_q[n].copy_(q_t[i])
        s.hist_k[n].copy_(k_t[i])
        s.hist_v[n].copy_(v_t[i])
        s.hist_g[n].copy_(g_t[i])
        s.hist_beta[n].copy_(beta_t[i])
        lens.append(n + 1)

    q = torch.cat([s.hist_q[:n] for s, n in zip(state, lens, strict=True)], 0)
    k = torch.cat([s.hist_k[:n] for s, n in zip(state, lens, strict=True)], 0)
    v = torch.cat([s.hist_v[:n] for s, n in zip(state, lens, strict=True)], 0)
    g = torch.cat([s.hist_g[:n] for s, n in zip(state, lens, strict=True)], 0)
    beta = torch.cat([s.hist_beta[:n] for s, n in zip(state, lens, strict=True)], 0)
    initial_state = torch.stack([s.boundary_state for s in state], 0)
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + n)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=q.device)

    out, final_state = _pinned_prefill(
        q, k, v, g, beta, cu_seqlens, initial_state, params
    )
    row_idx = torch.tensor([offsets[i + 1] - 1 for i in range(B)], device=q.device)
    rows = out.index_select(0, row_idx)

    for i, s in enumerate(state):
        s.current_state = final_state[i].clone()
        s.position += 1
        if lens[i] == CHUNK:
            s.boundary_state = s.current_state
    return rows


def kda_exact_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    params: KdaLayerParams,
) -> Tuple[torch.Tensor, KdaExactDecodeState]:
    """Prefill one prompt with the pinned kernel and return a decode-ready state.

    ``[0, b)`` with ``b = 16 * (L // 16)`` runs as one call whose final state is the
    boundary state; the tail ``[b, L)`` runs as a second call from that state and is
    kept as the in-chunk history, so the returned state is exactly what token-by-token
    decoding would have built.
    """
    _check_device(q.device)
    _check_state(initial_state, "initial_state")
    _check_heads(q.shape[1], v.shape[1])
    L = q.shape[0]
    b = CHUNK * (L // CHUNK)
    dev = q.device
    st = KdaExactDecodeState(boundary_state=initial_state.contiguous().clone())
    outs = []
    if b > 0:
        cu = torch.tensor([0, b], dtype=torch.int32, device=dev)
        o, fs = _pinned_prefill(
            q[:b], k[:b], v[:b], g[:b], beta[:b], cu, st.boundary_state[None], params
        )
        outs.append(o)
        st.boundary_state = fs[0].clone()
        st.current_state = st.boundary_state
        st.position = b
    if b < L:
        n = L - b
        st._ensure_history(q[0], k[0], v[0], g[0])
        st.hist_q[:n].copy_(q[b:L])
        st.hist_k[:n].copy_(k[b:L])
        st.hist_v[:n].copy_(v[b:L])
        st.hist_g[:n].copy_(g[b:L])
        st.hist_beta[:n].copy_(beta[b:L])
        cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
        o, fs = _pinned_prefill(
            st.hist_q[:n],
            st.hist_k[:n],
            st.hist_v[:n],
            st.hist_g[:n],
            st.hist_beta[:n],
            cu,
            st.boundary_state[None],
            params,
        )
        outs.append(o)
        st.current_state = fs[0].clone()
        st.position = L
    out = torch.cat(outs, 0) if outs else q.new_empty((0, v.shape[1], _HEAD_DIM))
    return out, st


def _advance(
    state: KdaExactDecodeState, toks: dict, params: KdaLayerParams
) -> torch.Tensor:
    """Append ``toks`` (``T`` tokens: q/k/v/g ``[T, heads, 128]``, beta ``[T, HV]``) to one request, in one or two pinned
    calls, returning the ``T`` output rows; commits a new boundary state whenever a chunk completes."""
    T = toks["q"].shape[0]
    state._ensure_history(toks["q"][0], toks["k"][0], toks["v"][0], toks["g"][0])
    n = state.n_hist
    seq = {
        key: torch.cat([getattr(state, f"hist_{key}")[:n], toks[key]], 0)
        for key in ("q", "k", "v", "g", "beta")
    }
    cu = torch.tensor([0, n + T], dtype=torch.int32, device=toks["q"].device)
    out, fs = _pinned_prefill(
        seq["q"],
        seq["k"],
        seq["v"],
        seq["g"],
        seq["beta"],
        cu,
        state.boundary_state[None],
        params,
    )
    rows = out[n:]
    total = n + T
    completed = (total // CHUNK) * CHUNK
    if completed > 0:
        if completed == total:
            new_boundary = fs[0]
        else:
            cu_b = torch.tensor(
                [0, completed], dtype=torch.int32, device=toks["q"].device
            )
            _, fsb = _pinned_prefill(
                seq["q"][:completed],
                seq["k"][:completed],
                seq["v"][:completed],
                seq["g"][:completed],
                seq["beta"][:completed],
                cu_b,
                state.boundary_state[None],
                params,
            )
            new_boundary = fsb[0]
        state.boundary_state = new_boundary.clone()
        rem = total - completed
        for key in ("q", "k", "v", "g", "beta"):
            getattr(state, f"hist_{key}")[:rem].copy_(seq[key][completed:])
    else:
        for key in ("q", "k", "v", "g", "beta"):
            getattr(state, f"hist_{key}")[n:total].copy_(toks[key])
    state.current_state = fs[0].clone()
    state.position += T
    return rows


def kda_exact_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: Sequence[KdaExactDecodeState],
    params: KdaLayerParams,
    num_accepted_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MTP verify step with the existing ``recurrent_kda`` spec-decode contract: ``T = 1 + S`` tokens per request
    (``q, k`` bf16 ``[N, T, H, 128]``, ``v, g`` ``[N, T, HV, 128]``, ``beta`` ``[N, T, HV]``); ``num_accepted_tokens``
    ``[N]`` int32 is the acceptance of the PREVIOUS round, applied first: the request rolls back to its state before
    that round advanced by its first ``a`` tokens, ``1 <= a <= T`` (the existing API loads slot ``[n, a - 1]``; here the
    per-token slots are implied by the boundary state plus the history); a request with no open draft (its first round)
    takes ``0``. Returns bf16 ``[N, T, HV, 128]``: row ``t`` is bitwise what a one-shot prefill of the accepted sequence
    followed by the ``T`` draft tokens gives at that position.
    """
    N, T = q.shape[0], q.shape[1]
    if len(state) != N:
        raise ValueError(f"got {N} requests for {len(state)} states")
    for t, name in ((q, "q"), (k, "k"), (v, "v"), (g, "g")):
        if (
            t.ndim != 4
            or t.shape[1] != T
            or t.dtype != _IO_DTYPE
            or t.shape[-1] != _HEAD_DIM
        ):
            raise ValueError(
                f"{name} must be bf16 [N, T, heads, {_HEAD_DIM}], got {tuple(t.shape)} {t.dtype}"
            )
    if beta.ndim != 3 or beta.shape[:2] != (N, T) or beta.dtype != _IO_DTYPE:
        raise ValueError(
            f"beta must be bf16 [N, T, HV], got {tuple(beta.shape)} {beta.dtype}"
        )
    _check_heads(q.shape[2], v.shape[2])
    _check_device(q.device)
    acc = _check_accepted(state, num_accepted_tokens)
    outs = []
    for i, s in enumerate(state):
        if s._snap is not None:
            _rollback(s, acc[i], params)
        s._ensure_history(q[i, 0], k[i, 0], v[i, 0], g[i, 0])
        snap_hist = {
            key: getattr(s, f"hist_{key}").clone()
            for key in ("q", "k", "v", "g", "beta")
        }
        draft = dict(q=q[i], k=k[i], v=v[i], g=g[i], beta=beta[i])
        s._snap = (
            s.boundary_state.clone(),
            s.position,
            snap_hist,
            {key: t.clone() for key, t in draft.items()},
        )
        outs.append(_advance(s, draft, params))
    return torch.stack(outs, 0)


def _check_accepted(
    state: Sequence[KdaExactDecodeState], num_accepted_tokens: Optional[torch.Tensor]
) -> List[int]:
    """Validate ``num_accepted_tokens`` against every request's open draft before any request is touched."""
    if num_accepted_tokens is None:
        if any(s._snap is not None for s in state):
            raise ValueError("a request has an open draft; pass num_accepted_tokens")
        return [0] * len(state)
    if num_accepted_tokens.numel() != len(state):
        raise ValueError("num_accepted_tokens must have one entry per request")
    acc = [int(a) for a in num_accepted_tokens.tolist()]
    for i, (s, a) in enumerate(zip(state, acc, strict=True)):
        if s._snap is None:
            if a != 0:
                raise ValueError(f"state[{i}] has no open draft to accept from")
            continue
        T = s._snap[3]["q"].shape[0]
        if not 1 <= a <= T:
            raise ValueError(f"num_accepted_tokens[{i}]={a} outside [1, {T}]")
    return acc


def _rollback(s: KdaExactDecodeState, a: int, params: KdaLayerParams) -> None:
    """Close the open draft at ``a`` accepted tokens: restore the snapshot and replay the accepted prefix."""
    boundary, position, hist, draft = s._snap
    s.boundary_state = boundary.clone()
    s.position = position
    for key in ("q", "k", "v", "g", "beta"):
        getattr(s, f"hist_{key}").copy_(hist[key])
    s.current_state = None
    _advance(s, {key: draft[key][:a] for key in draft}, params)
    s._snap = None


def kda_exact_accept(
    state: Sequence[KdaExactDecodeState],
    num_accepted_tokens: torch.Tensor,
    params: KdaLayerParams,
) -> None:
    """Close the open draft of each request at ``num_accepted_tokens`` accepted tokens (rollback to the accepted prefix)
    without starting a new round; afterwards ``kda_exact_decode`` may continue token by token."""
    acc = _check_accepted(state, num_accepted_tokens)
    for i, s in enumerate(state):
        if s._snap is None:
            raise ValueError(f"state[{i}] has no open draft")
        _rollback(s, acc[i], params)


# ---------------------------------------------------------------------------
# Pooled serving form: fixed 16-row history slots, one packed call, no per-request host work.
# ---------------------------------------------------------------------------


class KdaExactDecodePool:
    """Slot-addressed state of the replay decode for ``slots`` requests.

    Each slot owns a fixed 16-row history (``hist_*[slot]``), one fp32 boundary state (``state[slot]``) and one
    scratch state (``scratch[slot]``). A step packs every active slot as a full 16-token
    sequence (rows past the live count are padding the causal chunk math never reads into earlier rows), runs
    the pinned engine once reading the boundary pool and writing a scratch pool, commits the slots that completed
    their chunk, and returns row ``count`` of each slot. A step is three launches (prep, engine, finish) and
    :meth:`capture_step` records it into a CUDA graph. Outputs are bitwise those of :func:`kda_exact_decode`.
    """

    def __init__(
        self,
        slots: int,
        num_heads: int,
        num_v_heads: int,
        params: KdaLayerParams,
        device: torch.device,
    ) -> None:
        _check_device(torch.device(device))
        if num_v_heads != params.A_log.shape[0]:
            raise ValueError("num_v_heads must match params")
        _check_heads(num_heads, num_v_heads)
        if num_v_heads & (num_v_heads - 1):
            raise NotImplementedError(
                f"the pool's Triton kernels need a power-of-two head count, got {num_v_heads}"
            )
        self.slots = int(slots)
        self.H, self.HV = int(num_heads), int(num_v_heads)
        self.params = params
        self.device = params.A_log.device
        if torch.device(device).type != "cuda" or (
            torch.device(device).index not in (None, self.device.index)
        ):
            raise ValueError("device must be the CUDA device holding params")
        dev, bf = self.device, _IO_DTYPE
        S, H, HV = self.slots, self.H, self.HV
        self.hist_q = torch.zeros(S, CHUNK, H, _HEAD_DIM, dtype=bf, device=dev)
        self.hist_k = torch.zeros(S, CHUNK, H, _HEAD_DIM, dtype=bf, device=dev)
        self.hist_v = torch.zeros(S, CHUNK, HV, _HEAD_DIM, dtype=bf, device=dev)
        self.hist_g = torch.zeros(S, CHUNK, HV, _HEAD_DIM, dtype=bf, device=dev)
        self.hist_beta = torch.zeros(S, CHUNK, HV, dtype=bf, device=dev)
        self.state = torch.zeros(
            S, HV, _HEAD_DIM, _HEAD_DIM, dtype=torch.float32, device=dev
        )
        self.scratch = torch.empty_like(self.state)
        self.count = torch.zeros(S, dtype=torch.int32, device=dev)
        self.position = [0] * S
        self.out = torch.empty(S * CHUNK, HV, _HEAD_DIM, dtype=bf, device=dev)
        self._cache: dict = {}
        self._prep: Any = None
        self._gather: Any = None
        self._finish: Any = None

    # -- loading / reading slots ------------------------------------------------------------------------------
    def load(self, slot: int, st: KdaExactDecodeState) -> None:
        n = st.n_hist
        self.state[slot].copy_(st.boundary_state)
        if n:
            self.hist_q[slot, :n].copy_(st.hist_q[:n])
            self.hist_k[slot, :n].copy_(st.hist_k[:n])
            self.hist_v[slot, :n].copy_(st.hist_v[:n])
            self.hist_g[slot, :n].copy_(st.hist_g[:n])
            self.hist_beta[slot, :n].copy_(st.hist_beta[:n])
        self.count[slot] = n
        self.position[slot] = st.position

    def _check_position(self, slot: int) -> int:
        n = self.position[slot] % CHUNK
        if int(self.count[slot]) != n:
            raise RuntimeError(
                f"slot {slot}: host position {self.position[slot]} disagrees with the device count; "
                "call advance_positions after every step"
            )
        return n

    def export(self, slot: int) -> KdaExactDecodeState:
        """The per-request form of a slot (boundary state, live history, position)."""
        self._check_position(slot)
        return KdaExactDecodeState(
            boundary_state=self.state[slot].clone(),
            position=self.position[slot],
            hist_q=self.hist_q[slot].clone(),
            hist_k=self.hist_k[slot].clone(),
            hist_v=self.hist_v[slot].clone(),
            hist_g=self.hist_g[slot].clone(),
            hist_beta=self.hist_beta[slot].clone(),
            current_state=self.current_state(slot),
        )

    def current_state(self, slot: int) -> torch.Tensor:
        """fp32 state after the last consumed token (replayed on demand; bitwise the per-request form's)."""
        n = self._check_position(slot)
        if n == 0:
            return self.state[slot].clone()
        cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        _, fs = _pinned_prefill(
            self.hist_q[slot, :n],
            self.hist_k[slot, :n],
            self.hist_v[slot, :n],
            self.hist_g[slot, :n],
            self.hist_beta[slot, :n],
            cu,
            self.state[slot][None],
            self.params,
        )
        return fs[0].clone()

    # -- the step ---------------------------------------------------------------------------------------------
    def _cu_seqlens(self, B: int) -> torch.Tensor:
        cu = self._cache.get(B)
        if cu is None:
            cu = torch.arange(
                0, CHUNK * (B + 1), CHUNK, dtype=torch.int64, device=self.device
            )
            self._cache[B] = cu
        return cu

    def _packed(self, slot_ids: torch.Tensor, contiguous: bool):
        B = slot_ids.numel()
        if contiguous:
            return (
                self.hist_q[:B].view(B * CHUNK, self.H, _HEAD_DIM),
                self.hist_k[:B].view(B * CHUNK, self.H, _HEAD_DIM),
                self.hist_v[:B].view(B * CHUNK, self.HV, _HEAD_DIM),
                self.hist_g[:B].view(B * CHUNK, self.HV, _HEAD_DIM),
                self.hist_beta[:B].view(B * CHUNK, self.HV),
            )
        key = ("packed", B)
        bufs = self._cache.get(key)
        if bufs is None:
            bufs = (
                torch.empty(
                    B * CHUNK, self.H, _HEAD_DIM, dtype=_IO_DTYPE, device=self.device
                ),
                torch.empty(
                    B * CHUNK, self.H, _HEAD_DIM, dtype=_IO_DTYPE, device=self.device
                ),
                torch.empty(
                    B * CHUNK, self.HV, _HEAD_DIM, dtype=_IO_DTYPE, device=self.device
                ),
                torch.empty(
                    B * CHUNK, self.HV, _HEAD_DIM, dtype=_IO_DTYPE, device=self.device
                ),
                torch.empty(B * CHUNK, self.HV, dtype=_IO_DTYPE, device=self.device),
            )
            self._cache[key] = bufs
        if self._gather is None:
            self._gather = _gather_kernel_module()
        self._gather[(B, CHUNK)](
            slot_ids,
            self.hist_q,
            self.hist_k,
            self.hist_v,
            self.hist_g,
            self.hist_beta,
            *bufs,
            QK=self.H * _HEAD_DIM,
            VG=self.HV * _HEAD_DIM,
            HV=self.HV,
            CH=CHUNK,
            BLOCK=2048,
            num_warps=8,
        )
        return bufs

    def step(
        self,
        slot_ids: torch.Tensor,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        g_t: torch.Tensor,
        beta_t: torch.Tensor,
        contiguous: bool = True,
    ) -> torch.Tensor:
        """Decode one token for every slot in ``slot_ids`` (int32 ``[B]``); returns bf16 ``[B, HV, 128]``.

        With ``contiguous=True`` the caller guarantees ``slot_ids == arange(B)``, so the packed inputs are views
        of the pool; otherwise the slots' histories are gathered first. The returned buffer is owned by the pool
        and reused by the next step of the same ``B``. The host-side positions are not advanced here (a replayed
        graph cannot do it): call :meth:`advance_positions` after every step.
        """
        B = _check_token_inputs(q_t, k_t, v_t, g_t, beta_t)
        if B == 0:
            raise ValueError("empty batch")
        if q_t.shape[1] != self.H or v_t.shape[1] != self.HV:
            raise ValueError(
                f"expected {self.H} query heads and {self.HV} value heads, got {q_t.shape[1]} and {v_t.shape[1]}"
            )
        if (
            slot_ids.numel() != B
            or slot_ids.dtype != torch.int32
            or slot_ids.device != self.device
            or not slot_ids.is_contiguous()
        ):
            raise ValueError(
                "slot_ids must be a contiguous int32 tensor on the pool's device with one entry per token"
            )
        if q_t.device != self.device:
            raise ValueError("token inputs must live on the pool's device")
        state_idx, orow = self._prep_fused(slot_ids, q_t, k_t, v_t, g_t, beta_t)
        out = self._engine(slot_ids, contiguous)
        return self._finish_fused(slot_ids, state_idx, orow, out)

    def _finish_fused(self, slot_ids, state_idx, orow, out):
        if self._finish is None:
            self._finish = _finish_kernel_module()
        B = slot_ids.numel()
        key = ("rows", "out", B)
        rows = self._cache.get(key)
        if rows is None:
            rows = torch.empty(
                B, self.HV, _HEAD_DIM, dtype=_IO_DTYPE, device=self.device
            )
            self._cache[key] = rows
        row = self.HV * _HEAD_DIM * _HEAD_DIM
        block = 4096
        self._finish[(B, row // block)](
            state_idx,
            orow,
            self.scratch,
            self.state,
            out,
            rows,
            ROW=row,
            OUT=self.HV * _HEAD_DIM,
            BLOCK=block,
            num_warps=4,
        )
        return rows

    def _seq_order(self, B: int):
        key = ("order", B)
        o = self._cache.get(key)
        if o is None:
            o = torch.arange(B, dtype=torch.int32, device=self.device)
            self._cache[key] = o
        return o

    def _engine(self, slot_ids, contiguous):
        """Launch the pinned engine route directly: boundary states read from ``state[:slots]``, final states
        written to ``scratch``, both addressed by ``slot_ids``; the engine needs no workspace."""
        B = slot_ids.numel()
        q, k, v, g, beta = self._packed(slot_ids, contiguous)
        out = self.out[: B * CHUNK]
        compiled = kda_prefill_cute._get_compiled_cute_dsl_kda(
            mode=ROUTE,
            lower_bound=float(self.params.lower_bound),
            state_dtype=torch.float32,
            has_state_in=True,
            has_state_out=True,
            has_state_ckpt=False,
            has_state_indices=True,
            has_checkpoint_state_indices=False,
        )
        scale = (
            _HEAD_DIM**-0.5 if self.params.scale is None else float(self.params.scale)
        )
        compiled(
            q[None],
            k[None],
            v[None],
            g[None],
            self.params.A_log,
            self.params.dt_bias.reshape(self.HV, _HEAD_DIM),
            beta[None],
            self._cu_seqlens(B),
            self.state[: self.slots],
            out[None],
            self.scratch,
            None,
            torch.cuda.current_stream(self.device).cuda_stream,
            scale,
            state_indices=slot_ids,
            seq_order=self._seq_order(B),
        )
        return out

    def _idx_bufs(self, B: int):
        key = ("idx", B)
        bufs = self._cache.get(key)
        if bufs is None:
            bufs = (
                torch.empty(B, dtype=torch.int32, device=self.device),
                torch.empty(B, dtype=torch.int64, device=self.device),
            )
            self._cache[key] = bufs
        return bufs

    def _prep_fused(self, slot_ids, q_t, k_t, v_t, g_t, beta_t):
        if self._prep is None:
            self._prep = _prep_kernel_module()
        B = slot_ids.numel()
        sidx, orow = self._idx_bufs(B)
        for t in (q_t, k_t, v_t, g_t, beta_t):
            if not t.is_contiguous():
                raise ValueError("token inputs must be contiguous")
        self._prep[(B,)](
            slot_ids,
            self.count,
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            self.hist_q,
            self.hist_k,
            self.hist_v,
            self.hist_g,
            self.hist_beta,
            sidx,
            orow,
            QK=self.H * _HEAD_DIM,
            VG=self.HV * _HEAD_DIM,
            HV=self.HV,
            CH=CHUNK,
            BLOCK=2048,
            num_warps=8,
        )
        return sidx, orow

    def advance_positions(self, slot_ids_host: Sequence[int]) -> None:
        for s in slot_ids_host:
            self.position[s] += 1

    # -- CUDA graph form ----------------------------------------------------------------------------------------
    def capture_step(
        self, slot_ids, q_t, k_t, v_t, g_t, beta_t, contiguous: bool = True
    ):
        """Capture one step over fixed input buffers into a CUDA graph; replaying it is a step on those buffers.

        Returns the graph; its ``rows`` attribute is the bf16 ``[B, HV, 128]`` output buffer.
        """
        snap_count = self.count.clone()
        snap_state = self.state.clone()
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(2):
                self.step(slot_ids, q_t, k_t, v_t, g_t, beta_t, contiguous)
                self.count.copy_(snap_count)
                self.state.copy_(snap_state)
        torch.cuda.current_stream(self.device).wait_stream(s)
        torch.cuda.synchronize(self.device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            rows = self.step(slot_ids, q_t, k_t, v_t, g_t, beta_t, contiguous)
        self.count.copy_(snap_count)
        self.state.copy_(snap_state)
        torch.cuda.synchronize(self.device)
        g.rows = rows
        return g

    # -- benchmark helper ----------------------------------------------------------------------------------------
    def kernel_only(self, slot_ids: torch.Tensor, contiguous: bool = True):
        """A closure running only the packed engine call (the kernel floor of the step)."""
        return lambda: self._engine(slot_ids, contiguous)


def _prep_kernel_module():
    import triton
    import triton.language as tl

    @triton.jit
    def _kda_pool_prep(
        slot_ids_ptr,
        count_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        hq_ptr,
        hk_ptr,
        hv_ptr,
        hg_ptr,
        hb_ptr,
        sidx_ptr,
        orow_ptr,
        QK: tl.constexpr,
        VG: tl.constexpr,
        HV: tl.constexpr,
        CH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Per active slot: append the new token to its 16-row history, emit the slot to commit (``-1`` unless the
        chunk completes with this token) and its packed output row, and advance its count."""
        i = tl.program_id(0)
        slot = tl.load(slot_ids_ptr + i).to(tl.int64)
        cnt = tl.load(count_ptr + slot).to(tl.int64)
        row = slot * CH + cnt
        offs = tl.arange(0, BLOCK)
        for base in range(0, QK, BLOCK):
            idx = base + offs
            m = idx < QK
            tl.store(
                hq_ptr + row * QK + idx, tl.load(q_ptr + i * QK + idx, mask=m), mask=m
            )
            tl.store(
                hk_ptr + row * QK + idx, tl.load(k_ptr + i * QK + idx, mask=m), mask=m
            )
        for base in range(0, VG, BLOCK):
            idx = base + offs
            m = idx < VG
            tl.store(
                hv_ptr + row * VG + idx, tl.load(v_ptr + i * VG + idx, mask=m), mask=m
            )
            tl.store(
                hg_ptr + row * VG + idx, tl.load(g_ptr + i * VG + idx, mask=m), mask=m
            )
        hb = tl.arange(0, HV)
        tl.store(hb_ptr + row * HV + hb, tl.load(beta_ptr + i * HV + hb))
        last = cnt == CH - 1
        tl.store(sidx_ptr + i, tl.where(last, slot, -1).to(tl.int32))
        tl.store(orow_ptr + i, i * CH + cnt)
        tl.store(count_ptr + slot, ((cnt + 1) % CH).to(tl.int32))

    return _kda_pool_prep


def _finish_kernel_module():
    import triton
    import triton.language as tl

    @triton.jit
    def _kda_pool_finish(
        sidx_ptr,
        orow_ptr,
        src_ptr,
        dst_ptr,
        out_ptr,
        rows_ptr,
        ROW: tl.constexpr,
        OUT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Copy each completed chunk's final state from the scratch pool into its boundary slot and gather each
        slot's output row, in one launch."""
        i = tl.program_id(0)
        j = tl.program_id(1)
        if j == 0:
            o = tl.load(orow_ptr + i).to(tl.int64)
            offs = tl.arange(0, OUT)
            tl.store(rows_ptr + i * OUT + offs, tl.load(out_ptr + o * OUT + offs))
        sidx = tl.load(sidx_ptr + i)
        if sidx >= 0:
            slot = sidx.to(tl.int64)
            offs = j * BLOCK + tl.arange(0, BLOCK)
            tl.store(dst_ptr + slot * ROW + offs, tl.load(src_ptr + slot * ROW + offs))

    return _kda_pool_finish


def _gather_kernel_module():
    import triton
    import triton.language as tl

    @triton.jit
    def _kda_pool_gather(
        slot_ids_ptr,
        hq_ptr,
        hk_ptr,
        hv_ptr,
        hg_ptr,
        hb_ptr,
        pq_ptr,
        pk_ptr,
        pv_ptr,
        pg_ptr,
        pb_ptr,
        QK: tl.constexpr,
        VG: tl.constexpr,
        HV: tl.constexpr,
        CH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Copy row ``r`` of slot ``slot_ids[i]``'s 16-row history into packed row ``i * 16 + r`` (scattered slots)."""
        i = tl.program_id(0)
        r = tl.program_id(1)
        slot = tl.load(slot_ids_ptr + i).to(tl.int64)
        src = slot * CH + r
        dst = i.to(tl.int64) * CH + r
        offs = tl.arange(0, BLOCK)
        for base in range(0, QK, BLOCK):
            idx = base + offs
            m = idx < QK
            tl.store(
                pq_ptr + dst * QK + idx,
                tl.load(hq_ptr + src * QK + idx, mask=m),
                mask=m,
            )
            tl.store(
                pk_ptr + dst * QK + idx,
                tl.load(hk_ptr + src * QK + idx, mask=m),
                mask=m,
            )
        for base in range(0, VG, BLOCK):
            idx = base + offs
            m = idx < VG
            tl.store(
                pv_ptr + dst * VG + idx,
                tl.load(hv_ptr + src * VG + idx, mask=m),
                mask=m,
            )
            tl.store(
                pg_ptr + dst * VG + idx,
                tl.load(hg_ptr + src * VG + idx, mask=m),
                mask=m,
            )
        hb = tl.arange(0, HV)
        tl.store(pb_ptr + dst * HV + hb, tl.load(hb_ptr + src * HV + hb))

    return _kda_pool_gather
