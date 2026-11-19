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

SM90 W4A16 latent-MoE routed-expert block (mxfp4-g32 weights, bf16
activations)
=============================================================================

``w4a16_mxfp4_latent_routed_moe`` runs one EP rank's routed-expert block of a
latent MoE under GIVEN routing on SM90 (H100/H200): grouped W4A16 GEMMs over
the rank's local experts with the weight dequant kernel-internal (no bf16
weight expansion is ever materialized), the gated "situ" activation
(``4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)``) with a single bf16 rounding of
the activated intermediate, and the weighted combine back to latent space.
Routing is CONSUMED, not computed: ``topk_ids`` carry GLOBAL expert ids and
only slots with ``id < E_local`` contribute; the output is the rank-partial
combine (tokens with no local hit are zero rows).

This is a fixed-geometry specialist: the kernel is compiled for exactly

    E_local = 56, top_k = 16, L (latent hidden) = 3584, I (intermediate) = 3072

with mxfp4-pack-quantized weights in the compressed-tensors convention —
two e2m1 nibbles per uint8 with the LOW nibble at the even K index (the
convention of FlashInfer's own fp4 pack/unpack helpers), and ue8m0 uint8
group-32 scales (``scale = 2**(u8 - 127)``).

Backend selection (#5667 pattern)
---------------------------------
``BACKENDS = ("auto", "specialized", "reference")`` on the keyword-only
``backend`` kwarg. ``"specialized"`` selects the fused kernel explicitly; a
call it cannot serve (off the fixed geometry, off-SM90, kill switch) raises
``NotImplementedError`` with the refusal reason (also recorded in the
dispatch stats) — an explicitly selected backend never falls back (no other
fused SM90 W4A16 latent-MoE path exists in FlashInfer anyway:
``cutlass_fused_moe`` refuses bf16-activation / uint8-weight inputs, the
trtllm fp4 paths are arch-gated to SM10x/12x and the B12x W4A16 path is
SM120/121-only). ``"reference"`` selects the composable torch path — exact
mxfp4 dequant + situ + combine math, slow — explicitly and never runs the
fused kernel; unknown names raise. ``"auto"`` routes to ``"specialized"``
only on the dispatch allowlist
(``flashinfer/w4a16_mxfp4_latent_moe_sm90_dispatch.json``) and to the
reference path otherwise — only ``"auto"`` routes around a refusal.
Malformed/inconsistent arguments still raise.

Kill switch
-----------
``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (checked per call; the switch
shared by FlashInfer's specialized kernels) disables the fused kernel:
``"auto"`` routes to the composable ``"reference"`` path, explicit
``"specialized"`` raises. ``FLASHINFER_W4A16_MXFP4_LATENT_MOE=0`` is a
deprecated alias kept for one release (it now also routes to the reference
path instead of raising).

CUDA-graph rules
----------------
- The kernel takes the caller's current stream, re-queried on every call
  (TVM-FFI env stream), so capture follows the capture stream.
- The JIT module is never built under capture: an unbuilt module during
  capture raises with instructions to warm up eagerly first.
- The launch path is picked per call from the token count (``T <= 64`` runs
  the decode cooperative kernel, ``T > 64`` the tensor-core prefill path),
  so a capture at a fixed shape pins exactly one path into the graph.
- The internal workspace is cached per (device, stream) and grows only on an
  eager call.  Capture therefore requires a prior eager warm-up call on the
  capture stream with a token count >= the captured one (tracked per stream;
  violating calls raise instead of allocating under capture).
- Graphs capture the workspace POINTER of their capture stream's slot:
  graphs captured on the same (device, stream) — and replays of one graph —
  share that workspace, so their replays must be ordered with respect to
  each other (any stream may replay; it is concurrency between replays that
  is forbidden).  Graphs that must replay concurrently must be captured on
  distinct streams, whose slots hold distinct workspaces.
- Autotune tuning mode raises on ``backend="auto"`` only: the flat auto form
  has a single fixed tactic and must not be enumerated into flat-path tactic
  tables. An explicitly selected backend runs regardless — the unified
  ``MoELayer`` runner (``W4A16MxFp4LatentRunner``) dispatches
  ``backend="specialized"`` while being timed as a first-class cross-backend
  autotune candidate.
"""

import os
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.moe import w4a16_mxfp4_latent_routed_moe_trace
from ..utils import get_compute_capability

_E = 896  # global expert-id space of the measured checkpoint (documentation)
_E_LOCAL = 56
_TOP_K = 16
_L = 3584  # latent hidden
_I = 3072  # expert intermediate
_GROUP = 32
_MAX_TOKENS = 8192  # measured envelope (largest timed prefill chunk)

_ENV_VAR = "FLASHINFER_W4A16_MXFP4_LATENT_MOE"
# Kill switch shared by FlashInfer's specialized kernels (read per call).
_SPECIALIZED_KERNEL_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Backends behind the one public API (#5667 pattern).
BACKENDS = ("auto", "specialized", "reference")
_DISPATCH_FILE = "w4a16_mxfp4_latent_moe_sm90_dispatch.json"

_stats: dict = {
    "calls": 0,
    "backend_dispatches": {"specialized": 0, "reference": 0},
    "fallback_reasons": {},
}


def _fallback(reason):
    _stats["fallback_reasons"][reason] = _stats["fallback_reasons"].get(reason, 0) + 1


def _w4a16_mxfp4_latent_moe_stats() -> dict:
    return {
        "calls": _stats["calls"],
        "backend_dispatches": dict(_stats["backend_dispatches"]),
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "backends": BACKENDS,
        "dispatch_table_loaded": _dispatch_table() is not None,
    }


def _reset_w4a16_mxfp4_latent_moe_stats() -> None:
    _stats["calls"] = 0
    for k in _stats["backend_dispatches"]:
        _stats["backend_dispatches"][k] = 0
    _stats["fallback_reasons"].clear()


def _dispatch_table():
    """Load the dispatch surface (package data) once; ``None`` means
    ``backend="auto"`` never routes to "specialized"."""
    global _DISPATCH_TABLE
    try:
        return _DISPATCH_TABLE
    except NameError:
        pass
    try:
        import json
        from importlib import resources

        payload = json.loads(
            resources.files("flashinfer").joinpath(_DISPATCH_FILE).read_text()
        )
        _DISPATCH_TABLE = payload if isinstance(payload, dict) else None
    except Exception:
        _DISPATCH_TABLE = None
    return _DISPATCH_TABLE


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _dequant_mxfp4(wp: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """mxfp4 group-32 dequant (e2m1 nibbles, LOW nibble at the even K index,
    times ``2**(ue8m0 - 127)``) — the composable reference's helper."""
    n = wp.shape[0]
    low = (wp & 0x0F).to(torch.long)
    high = ((wp >> 4) & 0x0F).to(torch.long)
    idx = torch.stack((low, high), dim=-1).reshape(n, -1)
    mag = _E2M1.to(wp.device)[idx & 0x07]
    val = torch.where((idx & 0x08) != 0, -mag, mag)
    scale = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=wp.device),
        ws.to(torch.float32) - 127.0,
    ).repeat_interleave(_GROUP, dim=1)
    return val * scale


def _situ(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (
        25.0 * torch.tanh(u / 25.0)
    )


def _composable_reference(
    x, w13_packed, w13_scale, w2_packed, w2_scale, topk_ids, topk_weights, out
):
    """The composable torch path: exact mxfp4 dequant + gate/up GEMMs + situ
    (one bf16 rounding of the activated intermediate, serve-faithful) +
    weighted combine, fp32 accumulation. Slow — the kill-switch / fallback
    semantics of this op, not a performance path."""
    T, L = x.shape
    EL, N1, _ = w13_packed.shape
    inter = N1 // 2
    xf = x.to(torch.float32)
    ids = topk_ids.to(torch.int64)
    tw = topk_weights.to(torch.float32)
    acc = torch.zeros(T, L, dtype=torch.float32, device=x.device)
    for e in range(EL):
        rows, slots = torch.where(ids == e)
        if rows.numel() == 0:
            continue
        w13 = _dequant_mxfp4(w13_packed[e], w13_scale[e])
        w2 = _dequant_mxfp4(w2_packed[e], w2_scale[e])
        xe = xf[rows]
        g = xe @ w13[:inter].t()
        u = xe @ w13[inter:].t()
        h = _situ(g, u).to(torch.bfloat16).to(torch.float32)
        acc.index_add_(0, rows, (h @ w2.t()) * tw[rows, slots, None])
    out.copy_(acc.to(torch.bfloat16))
    return out


_module = None
# (device_index, raw cuda stream) -> largest token count run eagerly, i.e.
# the workspace the C++ side has already sized for that stream.
_warmed_tokens: dict = {}


def _killed() -> bool:
    if os.environ.get(_SPECIALIZED_KERNEL_DISABLE_ENV, "").strip() not in (
        "",
        "0",
    ):
        return True
    value = os.environ.get(_ENV_VAR, "").strip().lower()
    if value in ("0", "off", "false", "disable", "disabled"):
        import logging

        logging.getLogger(__name__).warning(
            "%s is deprecated; set %s=1 instead",
            _ENV_VAR,
            _SPECIALIZED_KERNEL_DISABLE_ENV,
        )
        return True
    return False


def _autotune_active() -> bool:
    try:
        from ..autotuner import AutoTuner

        return bool(AutoTuner.get().is_tuning_mode)
    except Exception:
        return False


def _get_module():
    global _module
    if _module is None:
        from ..jit.fused_moe import gen_w4a16_mxfp4_latent_moe_sm90_module

        _module = gen_w4a16_mxfp4_latent_moe_sm90_module().build_and_load()
    return _module


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"w4a16_mxfp4_latent_routed_moe: {msg}")


@flashinfer_api(trace=w4a16_mxfp4_latent_routed_moe_trace)
def w4a16_mxfp4_latent_routed_moe(
    x: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    backend: str = "auto",
) -> torch.Tensor:
    r"""One EP rank's W4A16 latent-MoE routed-expert block under given routing.

    Computes, per token ``t`` and per routed slot ``k`` with
    ``topk_ids[t, k] < E_local``::

        g, u = x[t] @ deq(w13[e])[:I].T, x[t] @ deq(w13[e])[I:].T
        h    = bf16(4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25))
        out[t] += topk_weights[t, k] * (h @ deq(w2[e]).T)

    with ``deq`` the mxfp4 group-32 dequant (e2m1 nibbles, LOW nibble at the
    even K index, times ``2**(ue8m0 - 127)``), fp32 accumulation inside each
    GEMM, and exactly one bf16 rounding of the activated intermediate.
    Slots pointing at non-local experts contribute nothing; a token with no
    local hit yields a zero row (rank-partial output).

    Parameters
    ----------
    x : torch.Tensor
        ``[T, 3584]`` bf16, contiguous.  Latent activations.
    w13_packed : torch.Tensor
        ``[56, 6144, 1792]`` uint8, contiguous.  fc1 mxfp4 codes; rows
        ``[0, 3072)`` are the gate projection, rows ``[3072, 6144)`` the up
        projection (gate|up stacking).
    w13_scale : torch.Tensor
        ``[56, 6144, 112]`` uint8, contiguous.  fc1 ue8m0 group-32 scales.
    w2_packed : torch.Tensor
        ``[56, 3584, 1536]`` uint8, contiguous.  fc2 (down) mxfp4 codes.
    w2_scale : torch.Tensor
        ``[56, 3584, 96]`` uint8, contiguous.  fc2 ue8m0 group-32 scales
        (96 groups of 32 over the 3072-wide reduction).
    topk_ids : torch.Tensor
        ``[T, 16]`` int32, contiguous.  GLOBAL expert ids; only ids in
        ``[0, 56)`` live on this rank (others, including negative sentinels,
        are skipped).
    topk_weights : torch.Tensor
        ``[T, 16]`` float32, contiguous.  Combine weights, consumed as-is.
    out : Optional[torch.Tensor]
        ``[T, 3584]`` bf16, contiguous; allocated when not given.  Fully
        overwritten (destination-passing style).
    backend : str
        One of ``BACKENDS = ("auto", "specialized", "reference")``.
        ``"specialized"`` selects the fused SM90 kernel explicitly; a call it
        cannot serve (off the fixed geometry, off-SM90, kill switch) raises
        ``NotImplementedError`` with the refusal reason (recorded in the
        dispatch stats) — an explicitly selected backend never falls back.
        ``"reference"`` selects the composable exact-math (slow) torch path
        explicitly and never runs the fused kernel; unknown names raise.
        ``"auto"`` (default) routes to ``"specialized"`` only on the
        dispatch allowlist and to the reference path otherwise — only
        ``"auto"`` routes around a refusal.
        ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` disables the fused
        kernel (auto: reference path; explicit: raise;
        ``FLASHINFER_W4A16_MXFP4_LATENT_MOE=0`` is a deprecated alias).

    Returns
    -------
    torch.Tensor
        ``out``.

    Notes
    -----
    The specialized kernel's decode path (``T <= 64``) stages bf16
    activations in fp16 at a ``2**7`` scale and saturates the staged values
    to the fp16 finite range, so activation magnitudes ``>= 512`` lose
    accuracy instead of producing inf/NaN.  The reference path has no such
    saturation.
    """
    _stats["calls"] += 1
    if backend not in BACKENDS:
        raise NotImplementedError(
            f"w4a16_mxfp4_latent_routed_moe supports backend in {BACKENDS}; "
            "an explicitly selected backend is never silently swapped"
        )
    if backend == "auto" and _autotune_active():
        # The flat "auto" entry has a single fixed tactic, so it refuses to
        # be enumerated by flat-path autotuners. An EXPLICITLY selected
        # backend runs regardless: the unified MoELayer runner selects
        # "specialized" while it is itself being timed as a first-class
        # cross-backend autotune candidate.
        raise RuntimeError(
            "w4a16_mxfp4_latent_routed_moe(backend='auto') has a single "
            "fixed tactic and refuses to run in autotune tuning mode; "
            "select backend='specialized' or 'reference' explicitly"
        )

    _check(x.is_cuda, "x must be a CUDA tensor")
    _check(x.dim() == 2, "x must be 2D [T, L]")
    tokens = x.shape[0]
    _check(tokens > 0, "token count must be positive")
    _check(x.dtype is torch.bfloat16, "x must be bf16 (W4A16 contract)")
    # generic shape CONSISTENCY (both paths; violations raise)
    _check(
        w13_packed.dim() == 3
        and w2_packed.dim() == 3
        and w13_scale.dim() == 3
        and w2_scale.dim() == 3,
        "weights/scales must be 3D [E_local, N, K-packed]",
    )
    _check(
        w13_packed.shape[2] * 2 == x.shape[1]
        and w13_scale.shape[2] * _GROUP == x.shape[1],
        "w13 packing/scales must match x's hidden size",
    )
    _check(
        w2_packed.shape[1] == x.shape[1] and w2_scale.shape[1] == x.shape[1],
        "w2 rows must match x's hidden size",
    )
    _check(
        w13_packed.shape[0]
        == w13_scale.shape[0]
        == w2_packed.shape[0]
        == w2_scale.shape[0],
        "weights/scales must agree on the local expert count",
    )
    _check(
        w13_packed.shape[1] == w13_scale.shape[1] and w13_packed.shape[1] % 2 == 0,
        "w13 packed/scale rows must match and stack gate|up (even row count)",
    )
    _check(
        w2_packed.shape[2] * 2 == w2_scale.shape[2] * _GROUP,
        "w2 packing/scales must agree on the reduction width",
    )
    _check(
        w2_packed.shape[2] * 2 * 2 == w13_packed.shape[1],
        "w2's reduction width must match w13's intermediate size (w13 stacks gate|up)",
    )
    # fixed-geometry ELIGIBILITY of the specialized kernel (a miss is a
    # fallback reason, not an error — the reference serves any consistent
    # geometry)
    _spec_reason = None
    if get_compute_capability(x.device) != (9, 0):
        _spec_reason = "not SM90"
    elif x.shape[1] != _L:
        _spec_reason = f"L={x.shape[1]} != {_L}"
    elif not 0 < tokens <= _MAX_TOKENS:
        _spec_reason = f"tokens {tokens} outside (0, {_MAX_TOKENS}]"
    elif w13_packed.shape != (_E_LOCAL, 2 * _I, _L // 2):
        _spec_reason = "w13_packed off the fixed geometry"
    elif w13_scale.shape != (_E_LOCAL, 2 * _I, _L // _GROUP):
        _spec_reason = "w13_scale off the fixed geometry"
    elif w2_packed.shape != (_E_LOCAL, _L, _I // 2):
        _spec_reason = "w2_packed off the fixed geometry"
    elif w2_scale.shape != (_E_LOCAL, _L, _I // _GROUP):
        _spec_reason = "w2_scale off the fixed geometry"
    for name, t in (
        ("w13_packed", w13_packed),
        ("w13_scale", w13_scale),
        ("w2_packed", w2_packed),
        ("w2_scale", w2_scale),
    ):
        _check(t.dtype is torch.uint8, f"{name} must be uint8 (mxfp4-g32 packing)")
    _check(
        topk_ids.dim() == 2
        and topk_ids.shape[0] == tokens
        and topk_ids.dtype is torch.int32,
        "topk_ids must be int32 [T, K]",
    )
    _check(
        topk_weights.shape == topk_ids.shape and topk_weights.dtype is torch.float32,
        "topk_weights must be float32 with topk_ids's shape",
    )
    if _spec_reason is None and topk_ids.shape[1] != _TOP_K:
        _spec_reason = f"top_k={topk_ids.shape[1]} != {_TOP_K}"
    # the specialized kernel loads the packed weights through 16-byte vector
    # types; a contiguous view with a storage offset can defeat that
    if _spec_reason is None and (
        w13_packed.data_ptr() % 16 != 0 or w2_packed.data_ptr() % 16 != 0
    ):
        _spec_reason = "packed weights not 16-byte aligned"
    if out is None:
        out = torch.empty(tokens, x.shape[1], dtype=torch.bfloat16, device=x.device)
    _check(
        out.shape == (tokens, x.shape[1]) and out.dtype is torch.bfloat16,
        "out must be bf16 [T, L]",
    )
    tensors = (
        x,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        out,
    )
    for name, t in zip(
        (
            "x",
            "w13_packed",
            "w13_scale",
            "w2_packed",
            "w2_scale",
            "topk_ids",
            "topk_weights",
            "out",
        ),
        tensors,
        strict=True,
    ):
        _check(t.is_contiguous(), f"{name} must be contiguous")
        _check(t.device == x.device, f"{name} must be on the same device as x")
    _check(
        (x.data_ptr() | out.data_ptr()) % 16 == 0,
        "x and out must be 16-byte aligned",
    )
    # Keep the specialized no-overlap contract. The reference reads every
    # input before its single write to `out`.
    if _spec_reason is None:
        out_lo = out.data_ptr()
        out_hi = out_lo + out.numel() * out.element_size()
        for t in tensors[:-1]:
            t_lo = t.data_ptr()
            t_hi = t_lo + t.numel() * t.element_size()
            if max(out_lo, t_lo) < min(out_hi, t_hi):
                _spec_reason = "out overlaps an input"
                break

    # ---- backend resolution (#5667 order) -----------------------------------
    chosen = None
    if _killed():
        _fallback(f"{backend}: kill switch")
        if backend == "specialized":
            # An explicitly selected backend never falls back.
            raise NotImplementedError(
                "w4a16_mxfp4_latent_routed_moe backend='specialized' is "
                "disabled (kill switch); an explicitly selected backend "
                "never falls back — use backend='auto' or 'reference'"
            )
        chosen = "reference"
    elif backend == "reference":
        chosen = "reference"
    elif backend == "specialized":
        if _spec_reason is None:
            chosen = "specialized"
        else:
            _fallback(f"specialized: {_spec_reason}")
            # An explicitly selected backend never falls back.
            raise NotImplementedError(
                "w4a16_mxfp4_latent_routed_moe backend='specialized' cannot "
                f"serve this call ({_spec_reason}); an explicitly selected "
                "backend never falls back — use backend='auto' or 'reference'"
            )
    else:  # "auto"
        if _spec_reason is None and _dispatch_table() is not None:
            chosen = "specialized"
        else:
            _fallback(f"auto: {_spec_reason or 'allowlist unavailable'}")
            chosen = "reference"
    if chosen == "reference":
        _composable_reference(
            x,
            w13_packed,
            w13_scale,
            w2_packed,
            w2_scale,
            topk_ids,
            topk_weights,
            out,
        )
        _stats["backend_dispatches"]["reference"] += 1
        return out

    # capture state is a property of the CURRENT device's current stream, so
    # inspect it (and run the kernel) with x's device current
    with torch.cuda.device(x.device):
        capturing = torch.cuda.is_current_stream_capturing()
        stream_key = (
            x.device.index,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if capturing:
            if _module is None:
                raise RuntimeError(
                    "w4a16_mxfp4_latent_routed_moe: the JIT module is not "
                    "built; never built under CUDA-graph capture — run one "
                    "eager call first"
                )
            if _warmed_tokens.get(stream_key, 0) < tokens:
                raise RuntimeError(
                    "w4a16_mxfp4_latent_routed_moe: workspace for this stream "
                    f"is sized for <= {_warmed_tokens.get(stream_key, 0)} "
                    f"tokens; capture at T={tokens} would allocate under "
                    "capture.  Run one eager call with at least this token "
                    "count on the capture stream first (and do not run a "
                    "larger eager call on the same stream afterwards while "
                    "the graph is in use)."
                )
        module = _get_module()
        try:
            module.w4a16_mxfp4_latent_routed_moe(
                x,
                w13_packed,
                w13_scale,
                w2_packed,
                w2_scale,
                topk_ids,
                topk_weights,
                out,
            )
        except Exception:
            # a failed eager call can leave the native slot without the
            # workspace this record promises; drop the record so the next
            # capture guard re-warms instead of trusting stale capacity
            if not capturing:
                _warmed_tokens.pop(stream_key, None)
            raise
    _stats["backend_dispatches"]["specialized"] += 1
    if not capturing and _warmed_tokens.get(stream_key, 0) < tokens:
        _warmed_tokens[stream_key] = tokens
    return out


__all__ = ["BACKENDS", "w4a16_mxfp4_latent_routed_moe"]
