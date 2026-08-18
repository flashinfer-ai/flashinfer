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

Fused GDN Decode Step - API Layer
=================================

One decode step of a GDN (gated delta net) linear-attention layer, fusing the
serving chain around :func:`gated_delta_rule_decode_pretranspose`:

1. ``ba = hidden_states @ w_ba`` (bf16 GEMV, fp32 accumulation); ``b =
   ba[:, :HV]`` feeds the beta gate, ``a = ba[:, HV:]`` the decay gate.
2. Depthwise causal conv1d update (width 4, silu) over the ``mixed_qkv``
   channels; the paged bf16 conv-state pool rows at ``state_indices`` shift
   left and append the raw input. The pool is consumed as a logical
   ``[P, qkv_dim, state_len]`` view of either an SD pool (``(state_len,
   dim)`` physical rows, the vLLM default — pass the transposed view) or a
   DS-dense pool (``(dim, state_len)`` rows).
3. q/k/v head split of the activated conv output.
4. Gated delta-rule decode with qk-L2-norm on the paged fp32 state pool
   (pretranspose / V-major ``[P, HV, V, K]`` layout, padded row stride
   supported), updated in place.

The composable torch implementation below works on any CUDA arch and is the
executable specification of the op.  Specialized kernels serve a registered
set of traced workload signatures on SM120: they are selected by
:mod:`.gdn_fused_decode_specialized` (see the package README.md for the
registry schema) and implemented in :mod:`.kernel`, and this module keeps
only a thin dispatch hook.

**There is no backend option and no environment gate.**  Which
implementation runs — one of the specialized kernels or the composable path
— is decided by the library from the registry and the device, not by the
caller: this is one fused operation, not a family of interchangeable
backends, and picking among its internal kernels is not a decision a caller
has the information to make.  Consumers that need to know *whether* the
fast path applies before committing to it ask
:func:`gdn_fused_decode_step_supported`; consumers that want the operation
*not* to run make that decision on their own side, because there is no
pre-existing FlashInfer implementation of this fused step for an
environment variable to fall back to.
"""

import functools
import math
from typing import Optional, Tuple

import torch

from ...api_logging import flashinfer_api
from ...trace.templates.gdn import gdn_fused_decode_trace


@functools.cache
def _get_gdn_specialized():
    """Import the specialized backends (lazily, on first probe or call)."""
    from . import gdn_fused_decode_specialized

    return gdn_fused_decode_specialized


def gdn_fused_decode_step_supported(
    batch_size: int,
    hidden_size: int = 5120,
    n_ba: int = 96,
    qkv_dim: int = 10240,
    num_qk_heads: int = 16,
    num_v_heads: int = 48,
    head_dim: int = 128,
    conv_width: int = 4,
    conv_state_len: int = 3,
    device: Optional[torch.device] = None,
    conv_state_layout: str = "SD",
) -> bool:
    r"""Cheap routing probe for framework consumers.

    Returns ``True`` when :func:`gdn_fused_decode_step` would serve this
    geometry with a specialized kernel on this device: the geometry and
    conv-state pool layout registered in
    ``gdn_fused_decode_registry.json`` for this device's
    compute capability, and a registered impl importable and not latched off
    by an earlier kernel failure.  ``conv_state_layout`` names the physical
    conv-state pool layout: ``"SD"`` (``(state_len, dim)`` rows, the vLLM
    default) or ``"DS"`` (``(dim, state_len)`` rows).  Callers should keep
    their own optimized composition when this returns ``False``: the
    composable fallback inside :func:`gdn_fused_decode_step` is a
    correctness path, not a fast one.  Host-side only (capture-safe).

    This answers *support*, never *policy*: a framework that has decided not
    to use this operation must not call it, rather than expect this probe to
    say ``False``.

    Serving calls this once per layer per decode step, so it must be cheap
    on the answer it repeats: the geometry -> answer mapping is memoized
    (see :func:`~flashinfer.gdn_kernels.experimental.
    gdn_fused_decode_specialized.gdn_fused_decode_supported_geometry`) and
    only the first call for a given geometry touches the registry or the
    device.
    """
    return _get_gdn_specialized().gdn_fused_decode_supported_geometry(
        batch_size,
        hidden_size,
        n_ba,
        qkv_dim,
        num_qk_heads,
        num_v_heads,
        head_dim,
        conv_width,
        conv_state_len,
        conv_state_layout,
        device,
    )


def _gdn_fused_decode_step_fallback(
    hidden_states: torch.Tensor,
    w_ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    use_qk_l2norm: bool,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composable torch implementation (any CUDA arch); pools update in place.

    ``scale`` follows the public op: ``None`` or ``0.0`` means ``1/sqrt(D)``.
    """
    B = hidden_states.shape[0]
    hv = A_log.shape[0]
    qkv_dim = mixed_qkv.shape[1]
    d = ssm_state.shape[-1]
    h_q = (qkv_dim - hv * d) // (2 * d)
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(d)
    idx = state_indices.to(torch.long)

    # 1) in_proj_ba GEMV: bf16 operands, fp32 accumulation, bf16 result.
    ba = (hidden_states.float() @ w_ba.float()).to(torch.bfloat16)
    b_gate = ba[:, :hv]
    a_gate = ba[:, hv:]

    # 2) causal conv1d update (depthwise, silu), fp32 math, bf16 out; the pool
    #    keeps the last width-1 raw inputs per channel and updates in place.
    st = conv_state.index_select(0, idx)
    x_t = mixed_qkv.to(conv_state.dtype)
    window = torch.cat([st, x_t.unsqueeze(-1)], dim=-1)
    y = (window.float() * conv_weight.float().unsqueeze(0)).sum(dim=-1)
    y = y + conv_bias.float()
    y = y * torch.sigmoid(y)
    conv_out = y.to(torch.bfloat16)
    conv_state.index_copy_(0, idx, window[..., 1:])

    # 3) q/k/v head split.
    q = conv_out[:, : h_q * d].view(B, h_q, d).float()
    k = conv_out[:, h_q * d : 2 * h_q * d].view(B, h_q, d).float()
    v = conv_out[:, 2 * h_q * d :].view(B, hv, d).float()

    # 4) gated delta rule with qk-L2-norm on gathered fp32 state rows
    #    (V-major [P, HV, V, K] pool; padded row stride preserved in place).
    if use_qk_l2norm:
        q = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
    group = hv // h_q
    q = q.repeat_interleave(group, dim=1)  # (B, HV, K)
    k = k.repeat_interleave(group, dim=1)  # (B, HV, K)

    g = torch.exp(
        -torch.exp(A_log.float())
        * torch.nn.functional.softplus(a_gate.float() + dt_bias.float())
    )  # (B, HV)
    beta = torch.sigmoid(b_gate.float())  # (B, HV)

    state = ssm_state[idx]  # (B, HV, V, K) view gather -> copy
    state = state * g[:, :, None, None]
    old_v = torch.einsum("bhk,bhvk->bhv", k, state)
    delta = beta[:, :, None] * (v - old_v)
    state = state + delta[..., None] * k[:, :, None, :]
    attn_out = scale * torch.einsum("bhk,bhvk->bhv", q, state)

    ssm_state[idx] = state
    result = attn_out.unsqueeze(1).to(torch.bfloat16)
    if out is not None:
        out.copy_(result)
        return out, conv_state, ssm_state
    return result, conv_state, ssm_state


@flashinfer_api(trace=gdn_fused_decode_trace)
def gdn_fused_decode_step(
    hidden_states: torch.Tensor,
    w_ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Fused single-token GDN decode step over paged conv/ssm state pools.

    Folds the per-layer decode chain (b/a projection GEMV, causal conv1d
    update, q/k/v split, gated delta-rule decode with qk-L2-norm) into one
    operation. Both state pools are updated **in place** and returned.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Layer input of shape ``[B, hidden]``, bfloat16.
    w_ba : torch.Tensor
        Fused b/a projection weight of shape ``[hidden, 2*HV]``, bfloat16
        (columns ``[:HV]`` produce the beta-gate input, ``[HV:]`` the decay
        input).
    mixed_qkv : torch.Tensor
        Raw (pre-conv) fused q/k/v channels of shape ``[B, qkv_dim]``,
        bfloat16, with ``qkv_dim = (2*H_q + HV) * D``.
    conv_weight : torch.Tensor
        Depthwise conv weight of shape ``[qkv_dim, width]``, bfloat16.
    conv_bias : torch.Tensor
        Conv bias of shape ``[qkv_dim]``, bfloat16.
    conv_state : torch.Tensor
        Paged conv-state pool as a logical ``[P, qkv_dim, width-1]`` view
        holding the last ``width-1`` raw channel inputs, bfloat16. Updated in
        place. Two physical pool layouts are supported: an SD pool
        (``(width-1, qkv_dim)`` rows, the vLLM default — pass
        ``pool.transpose(-1, -2)``) or a DS-dense pool (``(qkv_dim,
        width-1)`` rows, contiguous); the page stride may be padded.
    A_log : torch.Tensor
        Log decay parameter of shape ``[HV]``, float32.
    dt_bias : torch.Tensor
        Decay bias of shape ``[HV]``, bfloat16.
    scale : float, optional
        Query scale. ``None`` **and** ``0.0`` both select the default
        ``1/sqrt(D)``: a zero scale would make the whole attention output
        zero, so it is treated as "unset" rather than honoured (frameworks
        that keep the scale in a config default it to 0). Pass an explicit
        non-zero value to override.
    ssm_state : torch.Tensor
        Paged fp32 recurrent-state pool of shape ``[P, HV, V, K]`` (V-major /
        K-last), row stride may be padded (``stride(0) >= HV*V*K``). Updated
        in place.
    state_indices : torch.Tensor
        Per-batch pool slot indices of shape ``[B]``, int32.
    use_qk_l2norm : bool
        Apply L2 normalization to q and k. Default ``True``.
    out : torch.Tensor, optional
        Pre-allocated attention output of shape ``[B, 1, HV, V]``, bfloat16,
        dense (contiguous). Written in place and returned when provided
        (avoids a separate copy into framework-owned output buffers).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(output, conv_state, ssm_state)`` with ``output`` of shape
        ``[B, 1, HV, V]`` (bfloat16) and both pools mutated in place.

    Notes
    -----
    - There is no backend selector and no environment gate: the library
      chooses between its specialized kernels and the composable path from
      the registry and the device. The choice is observable —
      :func:`gdn_fused_decode_step_supported` answers it before the call,
      without running anything — but not overridable per call. A framework
      that does not want this operation simply does not call it.
    - The specialized kernels serve registered traced workload signatures on
      SM120; on any other device, or for any geometry the registry does not
      list, this function is exactly the composable torch implementation.
    - A specialized-kernel failure never breaks this op: it warns once,
      latches that implementation off for the rest of the process, and the
      call is served by the composable path.
    - CUDA graphs: each specialized implementation compiles lazily on its
      first eager dispatch of a (batch, scale, conv-state layout) variant;
      during capture one is recorded only when that variant is already warm,
      otherwise the (capture-safe) composable path is baked for that shape.
    """
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(ssm_state.shape[-1])

    # Specialized kernels: the registry and all specialized dispatch logic
    # live in gdn_fused_decode_specialized, the impl modules under kernel/
    # (see the package README.md).  This hook stays deliberately thin: a lazy
    # import and one call that returns None for everything the registry does
    # not serve.
    result = _get_gdn_specialized().try_run_gdn_fused_decode_specialized(
        hidden_states,
        w_ba,
        mixed_qkv,
        conv_weight,
        conv_bias,
        conv_state,
        A_log,
        dt_bias,
        float(scale),
        ssm_state,
        state_indices,
        use_qk_l2norm,
        out=out,
    )
    if result is not None:
        return result

    return _gdn_fused_decode_step_fallback(
        hidden_states,
        w_ba,
        mixed_qkv,
        conv_weight,
        conv_bias,
        conv_state,
        A_log,
        dt_bias,
        scale,
        ssm_state,
        state_indices,
        use_qk_l2norm,
        out=out,
    )
