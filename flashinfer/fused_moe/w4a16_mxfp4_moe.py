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
group-32 scales (``scale = 2**(u8 - 127)``).  Calls outside the contract
raise; there is no fallback because no other SM90 W4A16 latent-MoE path
exists in FlashInfer (``cutlass_fused_moe`` refuses bf16-activation /
uint8-weight inputs, the trtllm fp4 paths are arch-gated to SM10x/12x and
the B12x W4A16 path is SM120/121-only).

Kill switch
-----------
``FLASHINFER_W4A16_MXFP4_LATENT_MOE=0`` (checked per call) disables the API
cleanly: every call raises ``RuntimeError`` and nothing is JIT-built.

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
- Autotune tuning mode raises: this path has a single fixed tactic and must
  not be timed into tactic tables.
"""

import os
from typing import Optional

import torch

from ..utils import get_compute_capability

_E = 896  # global expert-id space of the measured checkpoint (documentation)
_E_LOCAL = 56
_TOP_K = 16
_L = 3584  # latent hidden
_I = 3072  # expert intermediate
_GROUP = 32
_MAX_TOKENS = 8192  # measured envelope (largest timed prefill chunk)

_ENV_VAR = "FLASHINFER_W4A16_MXFP4_LATENT_MOE"

_module = None
# (device_index, raw cuda stream) -> largest token count run eagerly, i.e.
# the workspace the C++ side has already sized for that stream.
_warmed_tokens: dict = {}


def _killed() -> bool:
    value = os.environ.get(_ENV_VAR, "").strip().lower()
    return value in ("0", "off", "false", "disable", "disabled")


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
        ``[56, 3584, 112]`` uint8, contiguous.  fc2 ue8m0 group-32 scales.
    topk_ids : torch.Tensor
        ``[T, 16]`` int32, contiguous.  GLOBAL expert ids; only ids in
        ``[0, 56)`` live on this rank (others, including negative sentinels,
        are skipped).
    topk_weights : torch.Tensor
        ``[T, 16]`` float32, contiguous.  Combine weights, consumed as-is.
    out : Optional[torch.Tensor]
        ``[T, 3584]`` bf16, contiguous; allocated when not given.  Fully
        overwritten (destination-passing style).

    Returns
    -------
    torch.Tensor
        ``out``.
    """
    if _killed():
        raise RuntimeError(
            f"w4a16_mxfp4_latent_routed_moe is disabled ({_ENV_VAR}="
            f"{os.environ.get(_ENV_VAR)!r}); unset the kill switch to use it"
        )
    if _autotune_active():
        raise RuntimeError(
            "w4a16_mxfp4_latent_routed_moe has a single fixed tactic and "
            "refuses to run in autotune tuning mode"
        )

    _check(x.is_cuda, "x must be a CUDA tensor")
    _check(
        get_compute_capability(x.device) == (9, 0),
        "this path is compiled for SM90 (H100/H200) only",
    )
    _check(x.dim() == 2 and x.shape[1] == _L, f"x must be [T, {_L}]")
    tokens = x.shape[0]
    _check(0 < tokens <= _MAX_TOKENS, f"token count must be in (0, {_MAX_TOKENS}]")
    _check(x.dtype is torch.bfloat16, "x must be bf16 (W4A16 contract)")
    _check(
        w13_packed.shape == (_E_LOCAL, 2 * _I, _L // 2),
        f"w13_packed must be [{_E_LOCAL}, {2 * _I}, {_L // 2}] "
        "(gate|up stacked, two e2m1 codes per byte along L)",
    )
    _check(
        w13_scale.shape == (_E_LOCAL, 2 * _I, _L // _GROUP),
        f"w13_scale must be [{_E_LOCAL}, {2 * _I}, {_L // _GROUP}] "
        f"(ue8m0, group-{_GROUP} along L)",
    )
    _check(
        w2_packed.shape == (_E_LOCAL, _L, _I // 2),
        f"w2_packed must be [{_E_LOCAL}, {_L}, {_I // 2}]",
    )
    _check(
        w2_scale.shape == (_E_LOCAL, _L, _I // _GROUP),
        f"w2_scale must be [{_E_LOCAL}, {_L}, {_I // _GROUP}]",
    )
    for name, t in (
        ("w13_packed", w13_packed),
        ("w13_scale", w13_scale),
        ("w2_packed", w2_packed),
        ("w2_scale", w2_scale),
    ):
        _check(t.dtype is torch.uint8, f"{name} must be uint8 (mxfp4-g32 packing)")
    _check(
        topk_ids.shape == (tokens, _TOP_K) and topk_ids.dtype is torch.int32,
        f"topk_ids must be int32 [T, {_TOP_K}] (global expert ids; pad unused "
        "slots with a non-local id to express a smaller top-k)",
    )
    _check(
        topk_weights.shape == (tokens, _TOP_K)
        and topk_weights.dtype is torch.float32,
        f"topk_weights must be float32 [T, {_TOP_K}]",
    )
    if out is None:
        out = torch.empty(tokens, _L, dtype=torch.bfloat16, device=x.device)
    _check(
        out.shape == (tokens, _L) and out.dtype is torch.bfloat16,
        f"out must be bf16 [T, {_L}]",
    )
    tensors = (x, w13_packed, w13_scale, w2_packed, w2_scale, topk_ids,
               topk_weights, out)
    for name, t in zip(
        ("x", "w13_packed", "w13_scale", "w2_packed", "w2_scale", "topk_ids",
         "topk_weights", "out"),
        tensors,
    ):
        _check(t.is_contiguous(), f"{name} must be contiguous")
        _check(t.device == x.device, f"{name} must be on the same device as x")
    _check(
        (x.data_ptr() | out.data_ptr()) % 16 == 0,
        "x and out must be 16-byte aligned",
    )

    capturing = torch.cuda.is_current_stream_capturing()
    stream_key = (x.device.index, torch.cuda.current_stream(x.device).cuda_stream)
    if capturing:
        if _module is None:
            raise RuntimeError(
                "w4a16_mxfp4_latent_routed_moe: the JIT module is not built; "
                "never built under CUDA-graph capture — run one eager call "
                "first"
            )
        if _warmed_tokens.get(stream_key, 0) < tokens:
            raise RuntimeError(
                "w4a16_mxfp4_latent_routed_moe: workspace for this stream is "
                f"sized for <= {_warmed_tokens.get(stream_key, 0)} tokens; "
                f"capture at T={tokens} would allocate under capture.  Run "
                "one eager call with at least this token count on the "
                "capture stream first (and do not run a larger eager call on "
                "the same stream afterwards while the graph is in use)."
            )
    module = _get_module()
    module.w4a16_mxfp4_latent_routed_moe(
        x, w13_packed, w13_scale, w2_packed, w2_scale, topk_ids, topk_weights,
        out,
    )
    if not capturing and _warmed_tokens.get(stream_key, 0) < tokens:
        _warmed_tokens[stream_key] = tokens
    return out


__all__ = ["w4a16_mxfp4_latent_routed_moe"]
