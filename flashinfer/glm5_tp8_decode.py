# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental GLM-5 TP8 decode on one node with eight B300 GPUs.

These collective operators specialize batch-one decoding to query lengths one
and four, hidden size 6144 and contiguous caches. Logical rank zero runs the q4
indexer or publishes a supplied selection; ranks one through seven run
attention. All eight ranks run MoE. The workspace, weight packing, plans and
JIT modules live in ``flashinfer.experimental.glm5_tp8_decode``, whose README
documents the full contract.
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch

from .api_logging import flashinfer_experimental_api

_FEATURE = "GLM-5 TP8 decode"


@flashinfer_experimental_api(feature=_FEATURE)
def create_glm5_tp8_decode_workspace(
    *,
    rank: int,
    max_seq_len: int = 202752,
    group: Any = None,
    device: Optional[torch.device] = None,
    comm_backend: Any = None,
    operators: Sequence[str] = ("attention", "moe"),
) -> Any:
    r"""Collectively create and warm an eight-rank decode workspace.

    Every rank of the eight-rank group must call it. The workspace holds the
    communication records, generation counters and operator scratch of both
    operator families. All ranks must create, prepare, execute, reset and
    destroy workspaces in matching order. Call ``destroy()`` collectively after
    all graph replays finish and before destroying the process group. Plans and
    captured graphs must not outlive the workspace. Serialize persistent grids
    on these GPUs, including grids from different workspaces, and order graph
    replays across CUDA streams explicitly.

    Parameters
    ----------
    rank : int
        Logical rank in ``[0, 8)`` within the communication group, independent
        of ``device``.
    max_seq_len : int
        Largest supported ``cur_pos + q_len``, from 1 to 1052672 (a 1M-token
        prefix plus 4096 decode tokens); default 202752. Cache and scratch
        allocation grows with this capacity; the communication arenas have a
        fixed size.
    group : torch.distributed.ProcessGroup, optional
        Eight-rank group of the default torch.distributed communication backend
        (default: the default process group). A Gloo group avoids creating an
        additional NCCL device communicator.
    device : torch.device, optional
        CUDA device of this rank (default: the current device).
    comm_backend : optional
        An explicit FlashInfer ``CommBackend`` instead of
        ``TorchDistBackend(group)``.
    operators : sequence of str
        ``"attention"`` and/or ``"moe"``; naming one family skips allocation and
        compilation of the other.

    Returns
    -------
    DecodeWorkspace
        Collective state shared by the plans of this rank.
    """
    from .experimental.glm5_tp8_decode.workspace import DecodeWorkspace

    return DecodeWorkspace(
        rank=rank,
        max_seq_len=max_seq_len,
        group=group,
        device=device,
        comm_backend=comm_backend,
        operators=operators,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def pack_glm5_tp8_moe_weights(
    weights: Mapping[str, torch.Tensor],
    *,
    rank: int,
    device: Optional[Union[torch.device, str]] = None,
) -> Any:
    r"""Pack one rank's MoE weights from a full FP8 checkpoint layer.

    ``weights`` holds one sparse decoder layer's unsharded tensors, keyed by
    the Hugging Face GLM-5 checkpoint names without the ``model.layers.{i}.``
    prefix. The packer reads only the 1545 entries listed below and ignores
    all others, so every rank can pass the same layer mapping. Each rank keeps
    intermediate channels ``[256 * rank, 256 * rank + 256)`` of the shared
    expert and of every routed expert; the norm, router and bias are
    replicated. Packing rearranges bytes and performs no quantization,
    requantization or scale conversion. Call it outside CUDA graph capture.
    The dense MLP layers (``mlp.gate_proj``, ``mlp.up_proj``,
    ``mlp.down_proj``) are not served by this operator.

    Parameters
    ----------
    weights : mapping of str to torch.Tensor
        Strided tensors, all on one CPU or CUDA device. Noncontiguous views
        and aliasing between entries are accepted; inputs are only read.
        Scales (``*.weight_scale_inv``) are FP32 scales of 128 x 128 blocks
        that multiply the decoded FP8 E4M3FN codes.

        - ``post_attention_layernorm.weight``: BF16 or FP32 ``[6144]``,
          widened exactly to FP32.
        - ``mlp.gate.weight``: BF16 ``[256, 6144]`` router; FP32 is rejected
          because narrowing would round.
        - ``mlp.gate.e_score_correction_bias``: FP32 or BF16 ``[256]``,
          widened exactly to FP32.
        - ``mlp.shared_experts.{gate_proj,up_proj}.weight``: FP8
          ``[2048, 6144]``; rows ``[256 * rank, +256)`` are used, with scale
          rows ``[2 * rank, 2 * rank + 2)`` of the FP32 ``[16, 48]``
          ``.weight_scale_inv``.
        - ``mlp.shared_experts.down_proj.weight``: FP8 ``[6144, 2048]``;
          columns ``[256 * rank, +256)`` are used, with scale columns
          ``[2 * rank, 2 * rank + 2)`` of the FP32 ``[48, 16]``
          ``.weight_scale_inv``.
        - ``mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`` and
          ``.weight_scale_inv`` for ``e`` in ``0..255``: as for the shared
          expert.

        Bank zero of the packed weights is the shared expert and bank
        ``e + 1`` is routed expert ``e``.
    rank : int
        Logical rank in ``[0, 8)``.
    device : torch.device or str, optional
        Device of the packed weights (default: the device of ``weights``).
        Rank-local slices are taken on the input device and only they are
        copied. Passing CPU tensors, for example memory-mapped safetensors,
        with ``device="cuda"`` copies about one eighth of the expert bytes;
        whole-layer CUDA inputs need about 9.7 GB per GPU.

    Returns
    -------
    MoEWeights
        Packed weights on ``device`` that own their storage and alias no
        input; keep them unchanged while plans use them.
    """
    from .experimental.glm5_tp8_decode.weights import pack_moe_weights

    return pack_moe_weights(weights, rank=rank, device=device)


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_glm5_tp8_moe(
    x: torch.Tensor,
    weights: Any,
    *,
    out: Optional[torch.Tensor] = None,
    workspace: Any,
    enable_pdl: bool = True,
) -> Any:
    r"""Prepare a callable fused RMSNorm/router/experts/TP-reduction plan.

    The plan runs the MoE block of one decode step across the eight ranks
    (RMSNorm, routing, routed and shared experts, tensor-parallel reduction)
    and adds the residual ``x`` once; ``out`` is bitwise identical on every
    rank. Preparation is collective and must run outside CUDA graph capture.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous BF16 ``[q_len, 6144]`` residual input, ``q_len`` 1 or 4; the
        same values on every rank.
    weights : MoEWeights
        This rank's weights from :func:`pack_glm5_tp8_moe_weights`.
    out : torch.Tensor, optional
        Contiguous BF16 output with the shape of ``x``; allocated when omitted.
        It must not overlap ``x``, the weights or workspace scratch.
    workspace : DecodeWorkspace
        From :func:`create_glm5_tp8_decode_workspace` with the ``"moe"`` family.
    enable_pdl : bool
        Launch with programmatic dependent launch (default ``True``): the kernel
        may start while the preceding kernel on the stream is still running and
        waits for it before reading activations or workspace state. Packed
        weights may be read before that wait, so that kernel must not write
        them. ``False`` launches with plain stream ordering.

    Returns
    -------
    plan
        Calling it launches on the current stream without host synchronization
        and returns ``out``. Capture the call in a CUDA graph and keep the plan
        alive during replay; update tensor contents in place, since changing
        tensor addresses requires a new plan. Every launch is cooperative: the
        persistent 148-CTA grid must be resident at once, and a launch that
        cannot be (for example under an MPS or green-context SM limit) fails
        with an error.
    """
    from .experimental.glm5_tp8_decode.backend import prepare_moe

    return prepare_moe(
        x,
        weights,
        out=out,
        workspace=workspace,
        enable_pdl=enable_pdl,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def pack_glm5_tp8_attention_weights(
    weights: Mapping[str, torch.Tensor],
    *,
    rank: int,
    device: Optional[Union[torch.device, str]] = None,
) -> Any:
    r"""Pack one rank's attention or indexer weights from a full FP8 checkpoint layer.

    ``weights`` holds one decoder layer's unsharded tensors, keyed by the
    Hugging Face GLM-5 checkpoint names without the ``model.layers.{i}.``
    prefix. Rank zero packs the DSA indexer (11 entries); ranks one through
    seven pack MLA (13 entries). The packer reads only the entries that
    ``rank`` needs and ignores all others, so every rank can pass the same
    layer mapping. Packing rearranges bytes and performs no quantization,
    requantization or scale conversion. Call it outside CUDA graph capture.
    On layers without indexer tensors (indexer type ``"shared"``), rank zero
    does not call this function and prepares shared mode with
    ``weights=None``.

    Parameters
    ----------
    weights : mapping of str to torch.Tensor
        Strided tensors, all on one CPU or CUDA device. Noncontiguous views
        and aliasing between entries are accepted; inputs are only read.
        Scales (``*.weight_scale_inv``) are FP32 scales of 128 x 128 blocks
        that multiply the decoded FP8 E4M3FN codes. 1-D norm weights and
        biases are BF16 or FP32 and are widened exactly to FP32.

        Every rank:

        - ``input_layernorm.weight``: ``[6144]``.
        - ``self_attn.q_a_proj.weight``: FP8 ``[2048, 6144]``; scale
          ``[16, 48]``.
        - ``self_attn.q_a_layernorm.weight``: ``[2048]``.

        Rank zero, used whole:

        - ``self_attn.indexer.wk.weight``: FP8 ``[128, 6144]``; scale
          ``[1, 48]``.
        - ``self_attn.indexer.wq_b.weight``: FP8 ``[4096, 2048]``; scale
          ``[32, 16]``.
        - ``self_attn.indexer.weights_proj.weight``: BF16 ``[32, 6144]``;
          FP32 is rejected because narrowing would round.
        - ``self_attn.indexer.k_norm.weight`` and ``.bias``: ``[128]``.

        Ranks one through seven. Rank ``r`` owns the global heads
        ``h0 = 10 * (r - 1)`` to ``h0 + hv - 1``, ``hv = min(10, 64 - h0)``,
        in ten head slots; rank seven has four heads, and the packer fills
        its six absent slots with zero codes and zero scales.

        - ``self_attn.kv_a_proj_with_mqa.weight``: FP8 ``[576, 6144]`` (512
          latent rows, then 64 RoPE key rows), used whole; scale ``[5, 48]``.
        - ``self_attn.kv_a_layernorm.weight``: ``[512]``.
        - ``self_attn.q_b_proj.weight``: FP8 ``[16384, 2048]``; rows
          ``[256 * h0, 256 * (h0 + hv))`` are used, with scale rows
          ``[2 * h0, 2 * (h0 + hv))`` of ``[128, 16]``.
        - ``self_attn.kv_b_proj.weight``: FP8 ``[28672, 512]``; rows
          ``[448 * h0, 448 * (h0 + hv))`` are used, with scale rows
          ``[3.5 * h0, 3.5 * (h0 + hv))`` of ``[224, 4]``. Per head, the
          first 192 rows (key without RoPE) are absorbed into the query and
          the last 256 rows (value) project the latent attention output.
        - ``self_attn.o_proj.weight``: FP8 ``[6144, 16384]``; columns
          ``[256 * h0, 256 * (h0 + hv))`` are used, with scale columns
          ``[2 * h0, 2 * (h0 + hv))`` of ``[48, 128]``.
    rank : int
        Logical rank in ``[0, 8)``.
    device : torch.device or str, optional
        Device of the packed weights (default: the device of ``weights``).
        Rank-local slices are taken on the input device and only they are
        copied.

    Returns
    -------
    AttentionWeights
        Packed weights on ``device`` that own their storage and alias no
        input; keep them unchanged while plans use them.
    """
    from .experimental.glm5_tp8_decode.attention_weights import pack_attention_weights

    return pack_attention_weights(weights, rank=rank, device=device)


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_glm5_tp8_sparse_attention(
    x: torch.Tensor,
    weights: Any,
    *,
    caches: Optional[Dict[str, torch.Tensor]],
    rope: Optional[torch.Tensor],
    cur_pos: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    workspace: Any,
    indexer: Optional[str] = None,
    selection: Optional[torch.Tensor] = None,
    enable_pdl: bool = True,
) -> Any:
    r"""Prepare fused sparse MLA/indexing, cache writes and TP residual reduction.

    Rank zero either runs the q4 indexer, which writes the new
    ``indexer_k_cache`` rows, or publishes a supplied selection, and sends the
    selected token IDs to the attention ranks. Ranks one through seven write their new
    ``kv_cache`` and ``pe_cache`` rows and compute MLA over the selected
    tokens. The tensor-parallel reduction adds the residual ``x`` once;
    ``out`` is bitwise identical on every rank. Preparation is collective and
    must run outside CUDA graph capture. See the backend README for numerical
    boundaries, lifecycle and concurrency requirements.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous BF16 ``[q_len, 6144]`` residual input, ``q_len`` 1 or 4; the
        same values on every rank.
    weights : AttentionWeights or None
        This rank's weights from :func:`pack_glm5_tp8_attention_weights`; may
        be ``None`` on rank zero in shared mode.
    caches : dict of str to torch.Tensor or None
        Contiguous BF16 caches with at least ``workspace.max_seq_len + 8`` rows:
        ``"indexer_k_cache"`` ``[rows, 128]`` on rank zero, ``"kv_cache"``
        ``[rows, 512]`` and ``"pe_cache"`` ``[rows, 64]`` on the other ranks.
        May be ``None`` on rank zero in shared mode.
    rope : torch.Tensor or None
        The complete FP32 ``[rows, 64]`` RoPE table with at least as many rows
        as the caches, not a slice for the current queries. May be ``None`` on
        rank zero in shared mode.
    cur_pos : torch.Tensor
        Device int32 scalar holding the cache row of the first query, updated
        in place between replays. Every replay requires ``0 <= cur_pos`` and
        ``cur_pos + q_len <= workspace.max_seq_len``; preparation checks only
        the initial value.
    out : torch.Tensor, optional
        Contiguous BF16 output with the shape of ``x``; allocated when omitted.
        It must not overlap the inputs, weights, caches or workspace scratch.
    workspace : DecodeWorkspace
        From :func:`create_glm5_tp8_decode_workspace` with the ``"attention"``
        family.
    indexer : str, optional
        ``"full"`` (the q4 default) computes the exact top-2048 selection;
        ``"shared"`` (the only q1 mode) reuses ``selection``.
    selection : torch.Tensor, optional
        Shared mode only, on rank zero: int32 ``[q_len, 2048]`` token IDs that
        are valid, unique and causal for sparse queries. Dense queries use
        their causal prefix and ignore unused entries.
    enable_pdl : bool
        Launch with programmatic dependent launch (default ``True``): the kernel
        may start while the preceding kernel on the stream is still running and
        waits for it before reading activations, positions, caches or workspace
        state. Packed weights may be read before that wait, so that kernel must
        not write them. ``False`` launches with plain stream ordering.

    Returns
    -------
    plan
        Calling it launches on the current stream without host synchronization
        and returns ``out``. Capture the call in a CUDA graph and keep the plan
        and workspace alive during replay; update tensor contents, positions
        and selections in place, since changing tensor addresses requires a new
        plan. Every launch is cooperative: the persistent 148-CTA grid must be
        resident at once, and a launch that cannot be (for example under an
        MPS or green-context SM limit) fails with an error.
    """
    from .experimental.glm5_tp8_decode.backend import prepare_attention

    return prepare_attention(
        x,
        weights,
        caches=caches,
        rope=rope,
        cur_pos=cur_pos,
        out=out,
        workspace=workspace,
        indexer=indexer,
        selection=selection,
        enable_pdl=enable_pdl,
    )


__all__ = [
    "create_glm5_tp8_decode_workspace",
    "pack_glm5_tp8_attention_weights",
    "pack_glm5_tp8_moe_weights",
    "prepare_glm5_tp8_moe",
    "prepare_glm5_tp8_sparse_attention",
]
