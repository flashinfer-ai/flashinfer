# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental: the Qwen 3.5 GDN (gated delta net) block.

The linear-attention sub-layer of a Qwen3.5-style model between its QKVZ projection and its out projection, in one
launch per step: BA projection, causal conv + SiLU, gated delta rule over the SSM state pool, gated RMSNorm and static
FP8 quantization of the output. Two forms:

* ``qwen_3_5_gdn_block_decode``: one token per request;
* ``qwen_3_5_gdn_block_verify``: an MTP verify step, T tokens per request (the per-token state layout).

Backends are pluggable (``flashinfer.experimental.qwen_3_5_gdn_block.api.register_backend``); the default, ``"frost"``,
is CuTe DSL for SM100 (cc 10.0). Every entry point is experimental (API and backend): calling one is the opt-in. Nothing
here imports the kernels or the DSL at import time; ``flashinfer/experimental/qwen_3_5_gdn_block/README.md`` is the
feature page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .api_logging import flashinfer_experimental_api

if TYPE_CHECKING:
    from .experimental.qwen_3_5_gdn_block.api import Qwen35GdnPlan
    from .experimental.qwen_3_5_gdn_block.support import (
        Qwen35GdnDecline,
        Qwen35GdnGeometry,
        Qwen35GdnSupport,
    )

__all__ = [
    "Qwen35GdnDecline",
    "Qwen35GdnGeometry",
    "Qwen35GdnPlan",
    "Qwen35GdnSupport",
    "prepare_qwen_3_5_gdn_block",
    "qwen_3_5_gdn_block_decode",
    "qwen_3_5_gdn_block_support",
    "qwen_3_5_gdn_block_verify",
]

_FEATURE = "Qwen 3.5 GDN block"
_LAZY = {
    "Qwen35GdnDecline": "support",
    "Qwen35GdnGeometry": "support",
    "Qwen35GdnSupport": "support",
    "Qwen35GdnPlan": "api",
}


def __getattr__(name: str):
    # Re-exported lazily so `import flashinfer.qwen_3_5_gdn_block` loads no backend module.
    if name in _LAZY:
        import importlib

        return getattr(
            importlib.import_module(
                f".experimental.qwen_3_5_gdn_block.{_LAZY[name]}", __package__
            ),
            name,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_5_gdn_block_support(
    device,
    geometry: "Qwen35GdnGeometry",
    form: str,
    backend: str = "frost",
    *,
    requests: Optional[int] = None,
    tokens_per_request: Optional[int] = None,
) -> "Qwen35GdnSupport":
    """Whether ``backend`` serves ``form`` of the block for this geometry on this device -- a serving stack's init-time
    dispatch predicate. Never raises.

    Parameters
    ----------
    device : torch.device or str
        The device the block would run on.
    geometry : Qwen35GdnGeometry
        The sub-layer's shape at this tensor-parallel rank.
    form : str
        ``"decode"`` or ``"verify"``.
    backend : str
        A registered backend name (``"frost"`` is built in).
    requests, tokens_per_request : Optional[int]
        The step shape, when the caller wants it judged too (``tokens_per_request`` is 1 for decode).

    Returns
    -------
    Qwen35GdnSupport
        ``served``; when declined, ``kind`` (one of ``"argument"``, ``"backend"``, ``"form"``, ``"device"``, ``"dsl"``,
        ``"geometry"``, ``"shape"``) and ``reason``; and the device's compute capability. A served answer is one
        ``prepare_qwen_3_5_gdn_block`` accepts.
    """
    from .experimental.qwen_3_5_gdn_block import support as _impl

    return _impl.qwen_3_5_gdn_block_support(
        device,
        geometry,
        form,
        backend,
        requests=requests,
        tokens_per_request=tokens_per_request,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_qwen_3_5_gdn_block(
    form: str,
    *,
    geometry: "Qwen35GdnGeometry",
    device,
    requests: int,
    tokens_per_request: int = 1,
    backend: str = "frost",
    param_dtypes=(torch.float32, torch.bfloat16, torch.bfloat16),
) -> "Qwen35GdnPlan":
    """Build a plan for one step shape, outside CUDA-graph capture (not cached; the functional entries keep their own
    LRU). Pass it as ``plan=`` to the entry points, or call ``plan.run(**tensors)``.

    Parameters
    ----------
    form : str
        ``"decode"`` or ``"verify"``.
    geometry : Qwen35GdnGeometry
    device : torch.device or str
    requests : int
        Requests per step (decode: token rows).
    tokens_per_request : int
        1 for decode; ``T = 1 + drafts`` for verify.
    backend : str
        A registered backend name.
    param_dtypes : tuple of torch.dtype
        The dtypes of ``(A_log, dt_bias, norm_weight)`` the steps will pass, each ``torch.float32`` or
        ``torch.bfloat16``.

    Returns
    -------
    Qwen35GdnPlan
        ``ValueError`` for a malformed argument, ``NotImplementedError`` naming what the backend does not serve,
        ``RuntimeError`` under capture.
    """
    from .experimental.qwen_3_5_gdn_block import api as _impl

    return _impl.prepare(
        form,
        geometry=geometry,
        device=device,
        requests=requests,
        tokens_per_request=tokens_per_request,
        backend=backend,
        param_dtypes=param_dtypes,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_5_gdn_block_decode(
    qkv: torch.Tensor,
    z: torch.Tensor,
    hidden: torch.Tensor,
    ba_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    out_scale: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    *,
    geometry: "Qwen35GdnGeometry",
    backend: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
    out_bf16: Optional[torch.Tensor] = None,
    plan: Optional["Qwen35GdnPlan"] = None,
) -> torch.Tensor:
    """One decode step (one token per request) of the GDN block; updates ``conv_state`` and ``ssm_state`` in place.

    ``Hv, Hk, Dv, Dk, K`` are ``geometry.num_v_heads, num_k_heads, head_v_dim, head_k_dim, conv_kernel``;
    ``C = geometry.conv_dim = 2 * Hk * Dk + Hv * Dv``. Row ``r`` reads and writes slot ``state_indices[r]`` of both
    pools; a slot ``<= 0`` marks a padding row (zero outputs, no state touched).

    Parameters
    ----------
    qkv : torch.Tensor
        ``[rows, C]`` bf16, the projected ``q | k | v`` before the conv (channels contiguous, any 16-byte-multiple
        row stride).
    z : torch.Tensor
        ``[rows, Hv, Dv]`` bf16, the output gate (heads contiguous, any 16-byte-multiple row stride).
    hidden : torch.Tensor
        ``[rows, d_model]`` bf16 contiguous, the normed layer input the BA projection reads.
    ba_weight : torch.Tensor
        ``[2 * Hv, d_model]`` bf16 contiguous: rows ``[0, Hv)`` give ``b``, rows ``[Hv, 2 Hv)`` give ``a``.
    A_log, dt_bias : torch.Tensor
        ``[Hv]``, float32 or bfloat16.
    ssm_state : torch.Tensor
        ``[num_slots, Hv, Dv, Dk]`` float32 pool (each slot contiguous, any 16-byte-multiple slot stride).
    state_indices : torch.Tensor
        ``[rows]`` int32.
    norm_weight : torch.Tensor
        ``[Dv]`` gated-RMSNorm weight, float32 or bfloat16.
    out_scale : torch.Tensor
        One-element float32 static FP8 scale: ``out = fp8(normed / out_scale)``.
    conv_state : torch.Tensor
        ``[num_slots, C, K - 1]`` bf16 view of the conv-state pool, oldest column first (any strides).
    conv_weight : torch.Tensor
        ``[C, K]`` bf16 contiguous (tap ``K - 1`` multiplies the newest input).
    geometry : Qwen35GdnGeometry
    backend : Optional[str]
        ``None``: the plan's backend, else ``"frost"``.
    out : Optional[torch.Tensor]
        ``[rows, Hv, Dv]`` float8_e4m3fn contiguous; default: the plan's buffer (overwritten by its next run).
    out_bf16 : Optional[torch.Tensor]
        ``[rows, Hv, Dv]`` bf16 contiguous, to also receive the normed value before quantization.
    plan : Optional[Qwen35GdnPlan]
        A prepared plan; default: the cached plan of this shape.

    Returns
    -------
    torch.Tensor
        The FP8 output ``[rows, Hv, Dv]``.
    """
    from .experimental.qwen_3_5_gdn_block import api as _impl

    return _impl.decode(
        qkv,
        z,
        hidden,
        ba_weight,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        norm_weight,
        out_scale,
        conv_state,
        conv_weight,
        geometry=geometry,
        backend=backend,
        out=out,
        out_bf16=out_bf16,
        plan=plan,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_5_gdn_block_verify(
    qkv: torch.Tensor,
    z: torch.Tensor,
    hidden: torch.Tensor,
    ba_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    norm_weight: torch.Tensor,
    out_scale: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    *,
    geometry: "Qwen35GdnGeometry",
    backend: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
    out_bf16: Optional[torch.Tensor] = None,
    plan: Optional["Qwen35GdnPlan"] = None,
) -> torch.Tensor:
    """One MTP verify step (``T`` tokens per request, ``rows = requests * T``) of the GDN block, with the per-token
    state layout; updates ``conv_state`` and ``ssm_state`` in place.

    Request ``n`` (accepted count ``a = num_accepted_tokens[n]`` from the previous verify) starts from the SSM state in
    slot ``state_indices[n, a - 1]``, writes the state after its token ``t`` to slot ``state_indices[n, t]`` (skipped
    when ``<= 0``), and convolves with the history ``conv_state[state_indices[n, 0], :, a - 1 : a + K - 2]``, then
    stores ``[old[a : a + K - 2], x_0 .. x_{T-1}]`` back into that slot. A request whose source slot is ``<= 0`` is
    padding: zero outputs, no state touched. Shapes as in :func:`qwen_3_5_gdn_block_decode`, except:

    Parameters
    ----------
    state_indices : torch.Tensor
        ``[requests, T]`` int32 (any row stride); a live request needs ``state_indices[n, 0] > 0``.
    cu_seqlens : torch.Tensor
        ``[requests + 1]`` int32 token offsets (``n * T``: every request carries ``T`` tokens).
    num_accepted_tokens : torch.Tensor
        ``[requests]`` int32, each in ``[1, T]``.
    conv_state : torch.Tensor
        ``[num_slots, C, T + K - 2]`` bf16 view (FROST: channel-contiguous, i.e. stored
        ``[num_slots, T + K - 2, C]``).

    Returns
    -------
    torch.Tensor
        The FP8 output ``[rows, Hv, Dv]``.
    """
    from .experimental.qwen_3_5_gdn_block import api as _impl

    return _impl.verify(
        qkv,
        z,
        hidden,
        ba_weight,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        norm_weight,
        out_scale,
        conv_state,
        conv_weight,
        geometry=geometry,
        backend=backend,
        out=out,
        out_bf16=out_bf16,
        plan=plan,
    )
