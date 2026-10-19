# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental: the Qwen 3.5 gated attention block (bf16) on SM107.

The whole Qwen3.5-style full-attention sub-layer -- QKV + gate projection, QK-RMSNorm with partial RoPE, paged-KV
write-through, GQA attention at head dim 256, sigmoid gate, out projection -- in three serving forms over the serving
stack's own paged KV cache:

* ``qwen_3_5_block_bf16_prefill``: a uniform ``[batch, seq]`` step with no cached prefix (K / V written through, attention
  over the step's own tokens);
* ``qwen_3_5_block_bf16_extend``: a packed step of new tokens behind cached prefixes (chunked prefill, a prefix-cache hit,
  a speculative verify; an empty prefix is the prefill case), bounded by a token capacity;
* ``qwen_3_5_block_bf16_decode``: a uniform ``[batch, q_len]`` decode step (``q_len = 1 + drafts``, 1..4 on the packed
  decode tile).

Every entry point is experimental (API and backend): calling one is the opt-in.  The kernels are CuTe DSL (Rubin, cc
10.7, ``nvidia-cutlass-dsl >= 4.8.0``) vendored from cuDNN Frontend; everything backend-specific lives under
``flashinfer.experimental.qwen_3_5_block`` (its ``README.md`` is the feature page).  Nothing here imports the kernels or
the DSL at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import torch

from .api_logging import flashinfer_experimental_api

if TYPE_CHECKING:
    from .experimental.qwen_3_5_block.api import Qwen35BlockPlan
    from .experimental.qwen_3_5_block.support import (
        Qwen35BlockFormSupport,
        Qwen35BlockGeometry,
        Qwen35BlockSource,
        Qwen35BlockSupport,
    )
    from .experimental.qwen_3_5_block.weights import Qwen35BlockWeights

__all__ = [
    "Qwen35BlockFormSupport",
    "Qwen35BlockGeometry",
    "Qwen35BlockPlan",
    "Qwen35BlockSource",
    "Qwen35BlockSupport",
    "Qwen35BlockWeights",
    "Qwen35BlockWrapper",
    "prepare_qwen_3_5_block_bf16",
    "qwen_3_5_block_bf16_decode",
    "qwen_3_5_block_bf16_extend",
    "qwen_3_5_block_bf16_prefill",
    "qwen_3_5_block_pack_weights",
    "qwen_3_5_block_rope_tables",
    "qwen_3_5_block_support",
]

_FEATURE = "Qwen 3.5 gated attention block (bf16)"
_LAZY = {
    "Qwen35BlockGeometry": "support",
    "Qwen35BlockSupport": "support",
    "Qwen35BlockFormSupport": "support",
    "Qwen35BlockSource": "support",
    "Qwen35BlockWeights": "weights",
    "Qwen35BlockPlan": "api",
}


def __getattr__(name: str):
    # The records live in the experimental package; they are re-exported here lazily so `import flashinfer.qwen_3_5_block`
    # loads no backend module.
    if name in _LAZY:
        import importlib

        return getattr(
            importlib.import_module(
                f".experimental.qwen_3_5_block.{_LAZY[name]}", __package__
            ),
            name,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_5_block_support(
    device, geometry: "Qwen35BlockGeometry", *, dtype: torch.dtype = torch.bfloat16
) -> "Qwen35BlockSupport":
    """The static answer to "can this process run the block for this geometry on this device" -- a serving stack's init-time
    dispatch predicate.  Never raises for an unsupported device (``served`` is False and ``reason`` says why).

    Parameters
    ----------
    device : torch.device or str
        The device the block would run on.
    geometry : Qwen35BlockGeometry
        The sub-layer's shape at this tensor-parallel rank.
    dtype : torch.dtype
        The activation / weight / pool dtype (``torch.bfloat16`` is the only served value of the bf16 entry points).

    Returns
    -------
    Qwen35BlockSupport
        ``served``, ``reason``, the device and CuTe DSL facts, the served ``forms``, one record per form with the typed
        declines it can raise at plan time, the validated page sizes and layouts, the decode envelope, and ``source``
        (the vendored cuDNN Frontend commit / version the kernels come from).
    """
    from .experimental.qwen_3_5_block import support as _impl

    return _impl.qwen_3_5_block_support(device, geometry, dtype=dtype)


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_5_block_pack_weights(
    qkv_proj_weight: torch.Tensor,
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    geometry: "Qwen35BlockGeometry",
    *,
    act_dtype: torch.dtype = torch.bfloat16,
) -> "Qwen35BlockWeights":
    """Pack a serving stack's merged QKV linear into the block's weights -- load time only; pure torch; CPU-safe.

    ``qkv_proj_weight`` is the rank's merged ``[2 * H_q * D + 2 * H_kv * D, d_model]`` weight whose rows are
    ``[q_proj (per-head [q_h | gate_h]) | k_proj | v_proj]`` (vLLM's ``QKVParallelLinear``; the HF ``q_proj`` carries the
    output gate per head).  The result's ``w_qkvg`` is a row PERMUTATION of it into the ``Q | GATE | K | V`` bands (same byte
    count, so a layer may replace its merged weight in place); the two zero-centered QK-norm weights come back as ``(1 + w)``
    formed in fp32 and rounded once into ``act_dtype`` (at most ``2^-8`` relative per channel in bf16).  ``o_proj.weight``
    ``[d_model, H_q * D]`` is used as ``w_o`` unchanged.

    Parameters
    ----------
    qkv_proj_weight : torch.Tensor
        The merged weight at this rank's head counts.
    q_norm_weight, k_norm_weight : Optional[torch.Tensor]
        ``q_norm.weight`` / ``k_norm.weight``, ``[head_dim]``, zero-centered; both ``None`` iff ``geometry.qk_norm`` is False.
    geometry : Qwen35BlockGeometry
        The geometry at this rank.
    act_dtype : torch.dtype
        The dtype the three results are returned in (contiguous).

    Returns
    -------
    Qwen35BlockWeights
        ``(w_qkvg, w_q_norm, w_k_norm)``.
    """
    from .experimental.qwen_3_5_block import weights as _impl

    return _impl.qwen_3_5_block_pack_weights(
        qkv_proj_weight, q_norm_weight, k_norm_weight, geometry, act_dtype=act_dtype
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_5_block_rope_tables(
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    out_cos: Optional[torch.Tensor] = None,
    out_sin: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The per-token ``cos`` / ``sin`` tables the block reads, from a serving stack's ``positions`` and rotary cache.

    The kernels consume ``[num_tokens, rope_dim]`` tables with duplicated halves (``cos[:, i] == cos[:, i + rope_dim // 2]``:
    byte-identical to what HF ``apply_rotary_pos_emb`` holds); a caller that already has them passes them directly.  This
    helper is for a stack that holds ``positions [num_tokens]`` and a ``cos_sin_cache [max_position, rope_dim]`` whose row is
    ``cos | sin`` halves of width ``rope_dim // 2`` (vLLM's ``RotaryEmbedding.cos_sin_cache``): one gather and two concats,
    into ``out_cos`` / ``out_sin`` when given (capturable: the gather's intermediate comes from the graph's pool under
    capture, nothing else is allocated).  mRoPE / YaRN / NTK differ only in how the cache is built -- the caller's business.

    Parameters
    ----------
    positions : torch.Tensor
        ``[num_tokens]`` int32 / int64 positions.
    cos_sin_cache : torch.Tensor
        ``[max_position, rope_dim]`` with the cos half then the sin half.
    out_cos, out_sin : Optional[torch.Tensor]
        Caller-owned ``[num_tokens, rope_dim]`` outputs.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)``, each ``[num_tokens, rope_dim]`` in the cache's dtype.
    """
    from .experimental.qwen_3_5_block import rope as _impl

    return _impl.qwen_3_5_block_rope_tables(
        positions, cos_sin_cache, out_cos=out_cos, out_sin=out_sin
    )


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_qwen_3_5_block_bf16(
    form: str,
    *,
    geometry: "Qwen35BlockGeometry",
    device,
    weights: "Qwen35BlockWeights",
    w_o: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    page_size: int,
    batch_size: int,
    kv_layout: str = "NHD",
    seq_len: Optional[int] = None,
    token_capacity: Optional[int] = None,
    q_len_per_req: int = 1,
    max_pages: Optional[int] = None,
    block_table: Optional[torch.Tensor] = None,
    max_seq_len: Optional[int] = None,
    max_query_len: Optional[int] = None,
    decode_split_kv: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
) -> "Qwen35BlockPlan":
    """Declare, check and compile the block for one shape bucket, outside CUDA-graph capture.

    The plan compiles in: the form, ``batch_size`` (sequences; the sequence capacity on ``extend``), ``seq_len`` (prefill) /
    ``token_capacity`` (extend) / ``q_len_per_req`` (decode), ``page_size`` and the pools' layout and page count (read from
    ``kv_cache``, the cache the steps will bind), the block table's columns and row stride (from ``block_table`` when given,
    else a contiguous ``[batch_size, max_pages]`` table), ``max_seq_len`` (the logical KV bound; ``None`` =
    ``max_pages * page_size``) and the decode split.  Typed errors: ``NotImplementedError`` for a device, DSL version, dtype,
    geometry or form the kernels do not serve (named), ``ValueError`` for a malformed argument (named), ``RuntimeError``
    under capture.

    Parameters
    ----------
    form : str
        ``"prefill"``, ``"extend"`` or ``"decode"``.
    geometry : Qwen35BlockGeometry
    device : torch.device or str
    weights : Qwen35BlockWeights
        Sample weights of the layer group (shape, dtype and strides are compiled in; the per-layer tensors are passed to ``run``).
    w_o : torch.Tensor
        ``[d_model, H_q * head_dim]``.
    kv_cache : torch.Tensor or (torch.Tensor, torch.Tensor)
        The paged cache the steps will bind (5-D ``[num_pages, 2, ...]`` or a ``(k, v)`` pair; see ``kv_layout``).
    page_size : int
        A positive multiple of 16 (16 / 32 / 64 / 128 validated).
    batch_size : int
    kv_layout : str
        ``"NHD"`` (``[num_pages, 2, page_size, H_kv, D]``) or ``"HND"`` (``[num_pages, 2, H_kv, page_size, D]``).
    seq_len : Optional[int]
        prefill: the uniform sequence length.
    token_capacity : Optional[int]
        extend: the step's token capacity ``T_max`` (the token tensors carry ``T_max`` rows, or exactly ``num_actual_tokens``).
    q_len_per_req : int
        decode: query rows per sequence (1..4 claimed on the packed decode tile; up to 32 routed and declared).
    max_pages : Optional[int]
        Table columns a step may address (default: ``block_table``'s columns).
    block_table : Optional[torch.Tensor]
        The ``[>= batch_size, >= max_pages]`` int32 table the steps will bind (its row stride is compiled in).
    max_seq_len : Optional[int]
        decode: the logical KV bound ``kv_len <= max_seq_len <= max_pages * page_size``.
    max_query_len : Optional[int]
        extend: the longest sequence of new tokens a step carries (default ``token_capacity``).
    decode_split_kv : Optional[int]
        decode: the split-KV -- ``None`` = the decode split model, ``>= 2`` the split tile with the sigmoid gate in its combine,
        ``1`` the unsplit tile with the gate as its own launch; a performance knob: the same function under every value.
    workspace_buffer : Optional[torch.Tensor]
        A caller-owned uint8 buffer of at least ``plan.workspace_size`` bytes; the plan allocates (and zeroes) its own when ``None``.

    Returns
    -------
    Qwen35BlockPlan
        ``plan.run(...)`` launches it; ``plan.workspace_size``, ``plan.declaration`` (the form, the compiled-in facts, the
        stages, the decode route, the out-projection tile and ``live_rows_multiple`` of an extend plan, the cache roots) and
        ``plan.supports(...)`` describe it.  On an extend plan of at most 128 tokens every ``num_actual_tokens`` must be a
        multiple of ``plan.live_rows_multiple`` (the widest epilogue vector that divides the capacity; 32 on a 64- or 128-token
        bf16 plan): the small-capacity out-projection tile takes the token rows as its epilogue-vector axis; a typed
        ``ValueError`` names the remedy.
    """
    from .experimental.qwen_3_5_block import api as _impl

    return _impl.prepare(
        form,
        geometry=geometry,
        device=device,
        weights=weights,
        w_o=w_o,
        kv_cache=kv_cache,
        page_size=page_size,
        batch_size=batch_size,
        kv_layout=kv_layout,
        seq_len=seq_len,
        token_capacity=token_capacity,
        q_len_per_req=q_len_per_req,
        max_pages=max_pages,
        block_table=block_table,
        max_seq_len=max_seq_len,
        max_query_len=max_query_len,
        decode_split_kv=decode_split_kv,
        workspace_buffer=workspace_buffer,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_5_block_bf16_prefill(
    hidden_states: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    weights: "Qwen35BlockWeights",
    w_o: torch.Tensor,
    *,
    geometry: "Qwen35BlockGeometry",
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    slot_mapping: torch.Tensor,
    kv_layout: str = "NHD",
    out: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    plan: Optional["Qwen35BlockPlan"] = None,
) -> torch.Tensor:
    """The dense prefill form: a uniform ``[batch, seq]`` step with no cached prefix.

    Every sequence starts at position 0; the step's K / V are written through into the pools at ``slot_mapping`` and
    attention runs over the step's own tokens (causal).  A packed, variable-length step -- or any step behind a cached
    prefix -- is :func:`qwen_3_5_block_bf16_extend`.

    Parameters
    ----------
    hidden_states : torch.Tensor
        ``[batch, seq, d_model]`` bf16 (post input-layernorm); ``[num_tokens, d_model]`` is one sequence of ``num_tokens``.
    rope_cos, rope_sin : torch.Tensor
        ``[batch, seq, rope_dim]`` (or ``[num_tokens, rope_dim]``) tables with duplicated halves (see :func:`qwen_3_5_block_rope_tables`).
    weights : Qwen35BlockWeights
        From :func:`qwen_3_5_block_pack_weights`.
    w_o : torch.Tensor
        ``[d_model, H_q * head_dim]``.
    geometry : Qwen35BlockGeometry
    kv_cache : torch.Tensor or (torch.Tensor, torch.Tensor)
        The paged cache (see ``kv_layout``); the page size is read from its shape.
    slot_mapping : torch.Tensor
        ``[batch * seq]`` int32 / int64 flat slots (``page * page_size + offset``); a negative slot writes nothing.
    kv_layout : str
        ``"NHD"`` or ``"HND"``.
    out : Optional[torch.Tensor]
        Caller-owned output of ``hidden_states``' shape (allocated when ``None``).
    workspace_buffer : Optional[torch.Tensor]
        Overrides the plan's own workspace (uint8, at least ``plan.workspace_size``).
    plan : Optional[Qwen35BlockPlan]
        A prepared plan to run (skips the plan cache).  Without one the call runs the bucket's cached plan (an LRU of 32 shape
        buckets): under CUDA-graph capture only a bucket already cached (a cold one is a typed ``RuntimeError`` -- warm it with
        one eager call first), and a plan a captured graph runs is retained when the cache evicts it, released by
        ``flashinfer.experimental.qwen_3_5_block.api.clear_plan_cache()`` only (the README's "Capture and replay").

    Returns
    -------
    torch.Tensor
        ``out``.
    """
    from .experimental.qwen_3_5_block import api as _impl

    return _impl.prefill(
        hidden_states,
        rope_cos,
        rope_sin,
        weights,
        w_o,
        geometry=geometry,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        kv_layout=kv_layout,
        out=out,
        workspace_buffer=workspace_buffer,
        plan=plan,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_5_block_bf16_extend(
    hidden_states: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    weights: "Qwen35BlockWeights",
    w_o: torch.Tensor,
    *,
    geometry: "Qwen35BlockGeometry",
    query_start_loc: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    token_capacity: Optional[int] = None,
    max_query_len: Optional[int] = None,
    num_actual_tokens: Optional[int] = None,
    kv_layout: str = "NHD",
    out: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    plan: Optional["Qwen35BlockPlan"] = None,
) -> torch.Tensor:
    """The extend form: a packed step of new tokens behind cached prefixes (chunked prefill, a prefix-cache hit, a verify).

    The step's tokens are written through into the pools; attention reads each sequence's WHOLE logical range
    (``seq_lens[b]`` keys through ``block_table[b]``, the prefix and the new tokens) on the bottom-right diagonal.  A sequence
    with an empty prefix is the prefill case.  The step is BOUNDED: the plan's ``token_capacity`` rows are a capacity, a step
    may hand ``num_actual_tokens <= token_capacity`` live rows (the token tensors then carry the capacity or exactly the live
    rows; pad rows carry finite values and slot ``-1``), and ``query_start_loc`` / ``seq_lens`` / ``block_table`` describe the
    plan's ``batch_size`` sequences with zero-length / ``seq_lens == 0`` / row-``0`` pads.  ``out`` rows past the live rows are
    untouched.

    Parameters
    ----------
    hidden_states : torch.Tensor
        ``[token_capacity, d_model]`` or ``[num_actual_tokens, d_model]`` bf16.
    rope_cos, rope_sin : torch.Tensor
        Tables of the same row count, ``[rows, rope_dim]``.
    weights : Qwen35BlockWeights
    w_o : torch.Tensor
    geometry : Qwen35BlockGeometry
    query_start_loc : torch.Tensor
        ``[batch_size + 1]`` int32 prefix sums of the new tokens per sequence.
    kv_cache : torch.Tensor or (torch.Tensor, torch.Tensor)
    slot_mapping : torch.Tensor
        ``[rows]`` int32 / int64 flat slots of the new tokens; ``-1`` on pad rows.
    block_table : torch.Tensor
        ``[batch_size, >= max_pages]`` int32 page table (row ``0`` for a padded entry).
    seq_lens : torch.Tensor
        ``[batch_size]`` int32 LOGICAL KV lengths INCLUDING the step's new tokens (``0`` = a padded entry).
    token_capacity : Optional[int]
        The plan's capacity (default: ``hidden_states``' rows).
    max_query_len : Optional[int]
        The longest sequence of new tokens the plan admits (default ``token_capacity``).
    num_actual_tokens : Optional[int]
        The live token count (a host int: one captured graph per bucket).  On a plan of at most 128 tokens it must be a multiple
        of ``plan.live_rows_multiple`` (32 in bf16; the small-capacity out-projection tile), else a typed ``ValueError``.
    kv_layout, out, workspace_buffer, plan
        As in :func:`qwen_3_5_block_bf16_prefill`.

    Returns
    -------
    torch.Tensor
        ``out`` (``[rows, d_model]``).
    """
    from .experimental.qwen_3_5_block import api as _impl

    return _impl.extend(
        hidden_states,
        rope_cos,
        rope_sin,
        weights,
        w_o,
        geometry=geometry,
        query_start_loc=query_start_loc,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        block_table=block_table,
        seq_lens=seq_lens,
        token_capacity=token_capacity,
        max_query_len=max_query_len,
        num_actual_tokens=num_actual_tokens,
        kv_layout=kv_layout,
        out=out,
        workspace_buffer=workspace_buffer,
        plan=plan,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_5_block_bf16_decode(
    hidden_states: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    weights: "Qwen35BlockWeights",
    w_o: torch.Tensor,
    *,
    geometry: "Qwen35BlockGeometry",
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len_per_req: int = 1,
    max_seq_len: Optional[int] = None,
    decode_split_kv: Optional[int] = None,
    kv_layout: str = "NHD",
    out: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    plan: Optional["Qwen35BlockPlan"] = None,
) -> torch.Tensor:
    """The decode form: ``batch`` sequences x ``q_len_per_req`` new tokens each over the paged cache.

    The step's tokens are written through, then the Rubin d256 decode tile attends per sequence over the first ``seq_lens[b]``
    keys of ``block_table[b]`` (the prefix and the step's own tokens, bottom-right causal), with the sigmoid gate applied in
    the split combine.  A padded entry is ``seq_lens == 0`` + table row ``0`` + slot ``-1`` and yields ``out`` rows exactly 0.
    ``q_len_per_req`` 1..4 is the claimed envelope (the packed tile); 5..32 is routed on the unpacked tile and declared.

    Parameters
    ----------
    hidden_states : torch.Tensor
        ``[batch * q_len_per_req, d_model]`` bf16, the sequences' rows consecutive.
    rope_cos, rope_sin : torch.Tensor
        ``[batch * q_len_per_req, rope_dim]`` tables.
    weights : Qwen35BlockWeights
    w_o : torch.Tensor
    geometry : Qwen35BlockGeometry
    kv_cache : torch.Tensor or (torch.Tensor, torch.Tensor)
    slot_mapping : torch.Tensor
        ``[batch * q_len_per_req]`` int32 / int64 flat slots; ``-1`` on a padded entry.
    block_table : torch.Tensor
        ``[batch, >= max_pages]`` int32 page table.
    seq_lens : torch.Tensor
        ``[batch]`` int32 LOGICAL KV lengths INCLUDING the step's tokens.
    q_len_per_req : int
    max_seq_len : Optional[int]
        The logical KV bound (default ``max_pages * page_size``).
    decode_split_kv : Optional[int]
        The split-KV: ``None`` = the decode split model, ``>= 2`` the split tile with the gate in its combine, ``1`` the unsplit
        tile with the gate as its own launch (a performance knob: the same function under every value).
    kv_layout, out, workspace_buffer, plan
        As in :func:`qwen_3_5_block_bf16_prefill`.

    Returns
    -------
    torch.Tensor
        ``out`` (``[batch * q_len_per_req, d_model]``).
    """
    from .experimental.qwen_3_5_block import api as _impl

    return _impl.decode(
        hidden_states,
        rope_cos,
        rope_sin,
        weights,
        w_o,
        geometry=geometry,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        block_table=block_table,
        seq_lens=seq_lens,
        q_len_per_req=q_len_per_req,
        max_seq_len=max_seq_len,
        decode_split_kv=decode_split_kv,
        kv_layout=kv_layout,
        out=out,
        workspace_buffer=workspace_buffer,
        plan=plan,
    )


class Qwen35BlockWrapper:
    """Plan-and-run wrapper for the Qwen 3.5 gated attention block: one instance per (layer group, form), FlashInfer's
    ``plan()`` / ``run()`` idiom over a caller-owned workspace.

    .. warning::
        ``Qwen35BlockWrapper`` is experimental: it provides no compatibility guarantees and may change or be removed
        without deprecation.  Calling ``plan`` or ``run`` is the opt-in and needs no environment variable; the
        implementation lives in ``flashinfer.experimental.qwen_3_5_block``.

    SM107 (Rubin, cc 10.7) only, bf16.  ``plan`` runs outside CUDA-graph capture (it compiles the kernels and sizes the
    workspace; a workspace smaller than ``plan_info.workspace_size`` is a typed ``ValueError`` naming the size); the first
    ``run`` after ``plan`` may already be captured.  One wrapper is a single-writer resource: do not ``plan`` while a
    captured or in-flight ``run`` may read its buffers.  Re-plan when the shape bucket changes (another batch / token
    capacity / page count / layout / table row stride / decode query length): a plan is exactly one bucket.
    """

    #: Mechanical identification: the decorators only mark ``plan`` and ``run``.
    is_experimental = True

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        *,
        geometry: "Qwen35BlockGeometry",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if (
            not isinstance(workspace_buffer, torch.Tensor)
            or workspace_buffer.dtype != torch.uint8
            or workspace_buffer.dim() != 1
        ):
            raise ValueError(
                "Qwen35BlockWrapper: workspace_buffer must be a 1-D uint8 CUDA tensor (the FlashInfer workspace convention)"
            )
        self._workspace = workspace_buffer
        self._kv_layout = str(kv_layout).upper()
        self._geometry = geometry
        self._dtype = dtype
        self._plan: Any = None

    @property
    def plan_info(self) -> "Qwen35BlockPlan":
        """The current plan (``None`` before ``plan``)."""
        return self._plan

    @flashinfer_experimental_api(feature=_FEATURE)
    def plan(
        self,
        form: str,
        *,
        weights: "Qwen35BlockWeights",
        w_o: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        page_size: int,
        batch_size: int,
        seq_len: Optional[int] = None,
        token_capacity: Optional[int] = None,
        q_len_per_req: int = 1,
        max_pages: Optional[int] = None,
        block_table: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        max_query_len: Optional[int] = None,
        decode_split_kv: Optional[int] = None,
    ) -> None:
        """Plan one shape bucket over this wrapper's workspace, outside CUDA-graph capture.

        Parameters
        ----------
        form : str
            ``"prefill"``, ``"extend"`` or ``"decode"``.
        weights, w_o, kv_cache, page_size, batch_size, seq_len, token_capacity, q_len_per_req, max_pages, block_table, max_seq_len, max_query_len, decode_split_kv
            As in :func:`prepare_qwen_3_5_block_bf16` (``weights`` / ``w_o`` are samples: shape and dtype).
        """
        from .experimental.qwen_3_5_block import api as _impl

        self._plan = _impl.prepare(
            form,
            geometry=self._geometry,
            device=self._workspace.device,
            weights=weights,
            w_o=w_o,
            kv_cache=kv_cache,
            page_size=page_size,
            batch_size=batch_size,
            dtype=self._dtype,
            kv_layout=self._kv_layout,
            seq_len=seq_len,
            token_capacity=token_capacity,
            q_len_per_req=q_len_per_req,
            max_pages=max_pages,
            block_table=block_table,
            max_seq_len=max_seq_len,
            max_query_len=max_query_len,
            decode_split_kv=decode_split_kv,
            workspace_buffer=self._workspace,
        )

    @flashinfer_experimental_api(feature=_FEATURE)
    def run(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        weights: "Qwen35BlockWeights",
        w_o: torch.Tensor,
        *,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        slot_mapping: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        num_actual_tokens: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the planned form on the current CUDA stream (capturable).

        Parameters
        ----------
        hidden_states, rope_cos, rope_sin, weights, w_o, kv_cache, slot_mapping, block_table, seq_lens, query_start_loc, num_actual_tokens, out
            The planned form's arguments, as in the functional entry point of that form.

        Returns
        -------
        torch.Tensor
            ``out``.
        """
        if self._plan is None:
            raise RuntimeError("Qwen35BlockWrapper.run: call plan() first")
        return self._plan.run(
            hidden_states,
            rope_cos,
            rope_sin,
            weights,
            w_o,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            block_table=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_actual_tokens=num_actual_tokens,
            out=out,
            workspace_buffer=self._workspace,
        )
