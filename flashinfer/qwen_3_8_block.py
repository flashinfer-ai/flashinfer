# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental: the Qwen 3.8 Flash-Next gated attention block (bf16) on SM107 -- the DECODE step with the in-block indexer.

The whole Qwen 3.8 Flash-Next sparse-attention sub-layer at decode, from the post-layernorm hidden states to the attention output,
in ONE call over the serving stack's own caches: the QKV + gate + indexer projection, QK-RMSNorm with partial RoPE, the paged-KV
write-through, the in-block indexer -- the raw-key ring and the paged compressed-key cache written for the block the step completes,
the paged scorer over every complete block, the per-row top-k --, block-sparse GQA attention at head dim 256 over the selected 4-token
blocks and the open tail, the sigmoid gate, the out projection.

* ``qwen_3_8_block_bf16_decode``: a uniform ``[batch, q_len]`` decode step (``q_len = 1 + drafts``, up to 16 at four indexer heads);
* ``prepare_qwen_3_8_block_bf16("decode", ...)``: the plan behind it (one per shape bucket; refused under capture);
* ``Qwen38BlockWrapper``: the same in FlashInfer's ``plan()`` / ``run()`` idiom over a caller-owned workspace.

A sibling of :mod:`flashinfer.qwen_3_5_block` (the dense Qwen 3.5 block): the same vendored cuDNN Frontend tree, the same cache roots,
the same capture rules; the geometry, the weights and the plan carry the indexer appended.  DECODE only: the prefill / extend forms of
the indexer are a later cuDNN Frontend PR's and decline by name.  Every entry point is experimental (API and backend): calling one is the
opt-in.  The kernels are CuTe DSL (Rubin, cc 10.7, ``nvidia-cutlass-dsl >= 4.8.0``) vendored from cuDNN Frontend; everything
backend-specific lives under ``flashinfer.experimental.qwen_3_8_block`` (its ``README.md`` is the feature page).  Nothing here imports
the kernels or the DSL at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import torch

from .api_logging import flashinfer_experimental_api

if TYPE_CHECKING:
    from .experimental.qwen_3_8_block.api import Qwen38BlockPlan
    from .experimental.qwen_3_8_block.support import (
        Qwen38BlockFormSupport,
        Qwen38BlockGeometry,
        Qwen38BlockSource,
        Qwen38BlockSupport,
    )
    from .experimental.qwen_3_8_block.weights import Qwen38BlockWeights

__all__ = [
    "Qwen38BlockFormSupport",
    "Qwen38BlockGeometry",
    "Qwen38BlockPlan",
    "Qwen38BlockSource",
    "Qwen38BlockSupport",
    "Qwen38BlockWeights",
    "Qwen38BlockWrapper",
    "prepare_qwen_3_8_block_bf16",
    "qwen_3_8_block_bf16_decode",
    "qwen_3_8_block_pack_weights",
    "qwen_3_8_block_rope_position_tables",
    "qwen_3_8_block_rope_tables",
    "qwen_3_8_block_support",
    "qwen_3_8_block_workspace_size",
]

_FEATURE = "Qwen 3.8 Flash-Next gated attention block (bf16)"
_LAZY = {
    "Qwen38BlockGeometry": "support",
    "Qwen38BlockSupport": "support",
    "Qwen38BlockFormSupport": "support",
    "Qwen38BlockSource": "support",
    "Qwen38BlockWeights": "weights",
    "Qwen38BlockPlan": "api",
}


def __getattr__(name: str):
    # The records live in the experimental package; they are re-exported here lazily so `import flashinfer.qwen_3_8_block`
    # loads no backend module.
    if name in _LAZY:
        import importlib

        return getattr(
            importlib.import_module(
                f".experimental.qwen_3_8_block.{_LAZY[name]}", __package__
            ),
            name,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_8_block_support(
    device,
    geometry: "Qwen38BlockGeometry",
    *,
    dtype: torch.dtype = torch.bfloat16,
    kv_cache_dtype: Optional[torch.dtype] = None,
) -> "Qwen38BlockSupport":
    """The static answer to "can this process run the Qwen 3.8 block for this geometry on this device" -- a serving stack's init-time
    dispatch predicate.  Never raises for an unsupported device (``served`` is False and ``reason`` says why).

    Parameters
    ----------
    device : torch.device or str
        The device the block would run on.
    geometry : Qwen38BlockGeometry
        The sub-layer's shape at this tensor-parallel rank, the indexer included.
    dtype : torch.dtype
        The activation / weight / pool dtype (``torch.bfloat16`` is the only served value of the bf16 entry points).
    kv_cache_dtype : Optional[torch.dtype]
        The pools' dtype a stack intends to bind (``None`` = ``dtype``); a dtype the decode form does not serve under the indexer
        (``torch.float8_e4m3fn`` today) reads as ``decode.served=False`` with the vendored decline text.

    Returns
    -------
    Qwen38BlockSupport
        ``served``, ``reason``, the device and CuTe DSL facts, the served ``forms`` (at most ``("decode",)``; the prefill / extend
        records say why they are not), one record per form with the typed declines it can raise at plan time, the validated page
        sizes and cache spellings, the decode envelope (``decode_max_q_len``, ``decode_capture_rows_max``), the served ``rope_dims``,
        the indexer facts READ off the vendored envelope (``index_sources``, ``index_list_forms``, ``index_cache_dtypes``,
        ``index_head_dims``, ``index_kv_heads``, ``index_block_sizes``, ``index_top_k_range``, ``index_query_rows_max``,
        ``index_state_rows_extra``, ``index_page_sizes``, ``kv_cache_dtypes``, ``fuse_prepare_default``, ``pdl_default``, and -- the
        sparse core's two bodies -- ``sparse_tiles`` (the bodies this part serves by ``kernel_template``: ``"sparse_d256_f16"``, the tcgen05
        core, everywhere; ``"sparse_lean_d256_f16"``, the lean decode tile, on cc 10.7) and ``sparse_tile_lean_default`` (whether
        ``sparse_tile_form="auto"`` consults the lean placement rule on this part; the per-cell reading is
        ``flashinfer.experimental.qwen_3_8_block.support.sparse_tile``)) and ``source`` (the vendored cuDNN Frontend commit / version the
        kernels come from).  The geometry verdict is the one ``prepare_qwen_3_8_block_bf16`` applies (one shared gate).
    """
    from .experimental.qwen_3_8_block import support as _impl

    return _impl.qwen_3_8_block_support(
        device, geometry, dtype=dtype, kv_cache_dtype=kv_cache_dtype
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_8_block_pack_weights(
    qkv_proj_weight: torch.Tensor,
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    geometry: "Qwen38BlockGeometry",
    *,
    act_dtype: torch.dtype = torch.bfloat16,
    index_qk_proj_weight: Optional[torch.Tensor] = None,
    index_q_norm_weight: Optional[torch.Tensor] = None,
    index_k_norm_weight: Optional[torch.Tensor] = None,
) -> "Qwen38BlockWeights":
    """Pack a serving stack's merged QKV (+ indexer) linear into the block's weights -- load time only; pure torch; CPU-safe.

    ``qkv_proj_weight`` is the rank's merged ``[2 * H_q * D + 2 * H_kv * D, d_model]`` weight whose rows are ``[q_proj (per-head
    [q_h | gate_h]) | k_proj | v_proj]`` -- or that weight with the indexer projection's ``(index_heads + index_kv_heads) x
    index_head_dim`` rows APPENDED (a stack that packs the replicated indexer projection into the same GEMM), in which case
    ``index_qk_proj_weight`` stays ``None``; otherwise ``index_qk_proj_weight`` (``[640, d_model]`` at the defaults: the indexer's query
    heads then its raw key head, the checkpoint's row order) is required.  The result's ``w_qkvg`` is a row PERMUTATION into the
    ``Q | GATE | K | V | INDEX`` bands (same byte count, so a layer may replace its merged weight in place); the four zero-centered
    RMSNorm weights (the QK norms, the indexer's query and compressed-key norms) come back as ``(1 + w)`` formed in fp32 and rounded
    once into ``act_dtype``.  ``o_proj.weight`` ``[d_model, H_q * D]`` is used as ``w_o`` unchanged.

    Parameters
    ----------
    qkv_proj_weight : torch.Tensor
        The merged weight at this rank's head counts, with or without the indexer band appended.
    q_norm_weight, k_norm_weight : Optional[torch.Tensor]
        ``q_norm.weight`` / ``k_norm.weight``, ``[head_dim]``, zero-centered; both ``None`` iff ``geometry.qk_norm`` is False.
    geometry : Qwen38BlockGeometry
        The geometry at this rank.
    act_dtype : torch.dtype
        The dtype the five results are returned in (contiguous).
    index_qk_proj_weight : Optional[torch.Tensor]
        The indexer projection when it does not ride ``qkv_proj_weight``; given twice is refused.
    index_q_norm_weight, index_k_norm_weight : Optional[torch.Tensor]
        The indexer's ``q_layernorm.weight`` / ``k_layernorm.weight``, ``[index_head_dim]``, zero-centered; REQUIRED.

    Returns
    -------
    Qwen38BlockWeights
        ``(w_qkvg, w_q_norm, w_k_norm, w_iq_norm, w_ik_norm)``.
    """
    from .experimental.qwen_3_8_block import weights as _impl

    return _impl.qwen_3_8_block_pack_weights(
        qkv_proj_weight,
        q_norm_weight,
        k_norm_weight,
        geometry,
        act_dtype=act_dtype,
        index_qk_proj_weight=index_qk_proj_weight,
        index_q_norm_weight=index_q_norm_weight,
        index_k_norm_weight=index_k_norm_weight,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_8_block_rope_position_tables(
    cos_sin_cache: torch.Tensor,
    *,
    num_positions: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The POSITION-indexed ``cos_pos`` / ``sin_pos`` tables the decode step reads, from a serving stack's rotary cache -- formed ONCE at load.

    The kernels rotate every row of the step (the Q / K heads, the indexer queries, the completing block's compressed key at its block
    START) at a device-derived position from ``[1, num_positions, rope_dim]`` tables with duplicated halves, so a stack hands the model's
    table once instead of gathering per-token rows every step.  ``cos_sin_cache`` is ``[max_position, rope_dim]`` with the ``cos | sin``
    halves (vLLM's ``RotaryEmbedding.cos_sin_cache``); ``num_positions`` (default: the cache's rows) PADS the tables by repeating the last
    row so they cover every position the block table can address (the block refuses a table shorter than ``max_pages x page_size`` by
    form; positions past the model's maximum are never reached by a token).  Bitwise the per-token helper's rows on every position.

    Parameters
    ----------
    cos_sin_cache : torch.Tensor
        ``[max_position, rope_dim]`` with the cos half then the sin half.
    num_positions : Optional[int]
        Rows of the result (``>= max_pages x page_size`` of the plans that will read it).
    dtype : Optional[torch.dtype]
        The result's dtype (default: the cache's; the plans need the activation dtype, bf16).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(cos_pos, sin_pos)``, each ``[1, num_positions, rope_dim]`` contiguous.
    """
    from .experimental.qwen_3_8_block import rope as _impl

    return _impl.qwen_3_8_block_rope_position_tables(
        cos_sin_cache, num_positions=num_positions, dtype=dtype
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_8_block_rope_tables(
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    out_cos: Optional[torch.Tensor] = None,
    out_sin: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The per-token ``cos`` / ``sin`` rows (the Qwen 3.5 helper, re-exported): what a plan with ``rope_position_tables_only=False`` (the
    unfused prepare arm) reads per step beside the position tables.  See :func:`flashinfer.qwen_3_5_block.qwen_3_5_block_rope_tables`."""
    from .experimental.qwen_3_8_block import rope as _impl

    return _impl.qwen_3_8_block_rope_tables(
        positions, cos_sin_cache, out_cos=out_cos, out_sin=out_sin
    )


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_qwen_3_8_block_bf16(
    form: str,
    *,
    geometry: "Qwen38BlockGeometry",
    device,
    weights: "Qwen38BlockWeights",
    w_o: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    page_size: int,
    batch_size: int,
    kv_layout: str = "NHD",
    q_len_per_req: int = 1,
    max_pages: Optional[int] = None,
    block_table: Optional[torch.Tensor] = None,
    max_seq_len: Optional[int] = None,
    decode_split_kv: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    index_kv_cache: Optional[torch.Tensor] = None,
    index_state_cache: Optional[torch.Tensor] = None,
    index_state_block_table: Optional[torch.Tensor] = None,
    position_tables: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    rope_position_tables_only: bool = True,
    ring_slots: bool = False,
    fuse_prepare: Optional[bool] = None,
    use_pdl: Optional[bool] = None,
    kv_cache_dtype: Optional[torch.dtype] = None,
    sparse_tile_form: str = "auto",
) -> "Qwen38BlockPlan":
    """Declare, check and compile the block for one DECODE shape bucket, outside CUDA-graph capture.

    The plan compiles in: ``batch_size``, ``q_len_per_req`` (1 .. 16 at four indexer heads), ``page_size`` (16 / 32 / 64 / 128) and the
    pools' layout, page count and strides (read from ``kv_cache``), the block table's columns and row stride (from ``block_table`` when
    given -- the sparse core binds the whole table, so ``max_pages`` must be its width), ``max_seq_len``, the split, the side caches'
    FORM (the compressed-key cache's rows per page, the ring's capacity ``cap >= q_len_per_req + 3``; never their strides), the two
    input-contract flags and the knobs.  Typed errors: ``NotImplementedError`` for a form other than ``"decode"``, a device, DSL
    version, dtype, geometry, page size, step width or pool dtype the kernels do not serve (named), ``ValueError`` for a malformed
    argument (named), ``RuntimeError`` under capture.

    Parameters
    ----------
    form : str
        ``"decode"`` (``"prefill"`` / ``"extend"`` decline by name: a later cuDNN Frontend PR vendors the indexer's prefill forms).
    geometry : Qwen38BlockGeometry
    device : torch.device or str
    weights : Qwen38BlockWeights
        Sample weights of the layer group (shape, dtype and strides are compiled in; the per-layer tensors are passed to ``run``).
    w_o : torch.Tensor
        ``[d_model, H_q * head_dim]``.
    kv_cache : torch.Tensor or (torch.Tensor, torch.Tensor)
        The paged cache the steps will bind: FlashInfer's 5-D ``[num_pages, 2, ...]`` or a ``(k, v)`` pair (``kv_layout`` ``"NHD"`` /
        ``"HND"``), or a serving stack's UNIFIED row ``[num_pages, page_size, H_kv, 2 x D]`` (``kv_layout="UNIFIED"``: K at ``[..., :D]``, V at
        ``[..., D:]``); bound by strides, never copied.
    page_size : int
        16, 32, 64 or 128 (the pages the paged scorer addresses).
    batch_size : int
    kv_layout : str
        ``"NHD"``, ``"HND"`` or ``"UNIFIED"``.
    q_len_per_req : int
        Query rows per sequence (``1 + drafts``); at most ``INDEX_QUERY_ROWS_MAX // next_pow2(index_heads)`` = 16 at four heads.
    max_pages : Optional[int]
        Table columns a step may address (default and, with ``block_table``, required: the table's width).
    block_table : Optional[torch.Tensor]
        The ``[>= batch_size, max_pages]`` int32 table the steps will bind (its row stride is compiled in); its leading rows are the sample.
    max_seq_len : Optional[int]
        The logical KV bound ``kv_len <= max_seq_len <= max_pages * page_size``.
    decode_split_kv : Optional[int]
        The split-KV of the sparse tile -- ``None`` = the fill rule, an int the explicit split; a performance knob (the same function).
    workspace_buffer : Optional[torch.Tensor]
        A caller-owned uint8 buffer of at least ``plan.workspace_size`` bytes (the serving idiom: ONE buffer for every bucket, sized at
        the MAXIMUM of :func:`qwen_3_8_block_workspace_size` over every bucket the stack will plan -- not at the largest batch: under
        ``sparse_tile_form="auto"`` a smaller batch can take the lean body, whose split-KV slab is wider than the core's at a larger
        batch); the plan allocates (and zeroes) its own when ``None``.
    index_kv_cache : torch.Tensor
        REQUIRED: the paged compressed-key cache ``[num_pages, page_size / 4, 1, 128]`` bf16 whose page axis is the MAIN block table's (one
        row per complete 4-token block; the step writes the block it completes, the scorer reads every complete block).
    index_state_cache : torch.Tensor
        REQUIRED: the raw-key ring ``[num_ring_blocks, cap, 1, 128]`` bf16, ``cap >= q_len_per_req + 3`` (block 0 the null block of a padded
        entry); a run may bind a ring of another capacity that meets the rule.
    index_state_block_table : Optional[torch.Tensor]
        A sample ``[>= batch_size, 1]`` int32 ring table (form only; a zeros sample when ``None``).
    position_tables : (torch.Tensor, torch.Tensor)
        REQUIRED: ``(cos_pos, sin_pos)`` from :func:`qwen_3_8_block_rope_position_tables`, ``[1, >= max_pages x page_size, rope_dim]`` bf16,
        held by the plan: what its ``run`` binds unless handed a pair of its own (``position_tables=``, checked against the plan).
    rope_position_tables_only : bool
        ``True`` (default): every rotation reads the position tables at the row's device-derived position and ``run`` takes
        ``rope_cos=None, rope_sin=None``; ``False``: the per-token tables are required at ``run`` (the unfused arm's form; bitwise).
    ring_slots : bool
        ``True``: ``run(index_state_slot_mapping=)`` names the ring WRITE rows (a stack's own circular-buffer slots ``block x cap + pos % cap``,
        ``-1`` = no write); ``False``: the rows derive from the ring table and ``seq_lens`` in the kernel.  Bitwise; a compile key.
    fuse_prepare : Optional[bool]
        ``None`` = the vendored MEASURED default (one prepare launch on cc 10.7); ``False`` = the four-launch twin; both bitwise.  ``False``
        with ``rope_position_tables_only=True`` is a typed decline (the unfused query norm reads per-token rows).
    use_pdl : Optional[bool]
        ``None`` = the vendored default (off on the sparse chain today); ``True`` is a typed decline until the vendored tree serves it.
    kv_cache_dtype : Optional[torch.dtype]
        ``None`` = bf16 pools; ``torch.float8_e4m3fn`` is a typed decline until the sparse core's in-kernel dequant arm lands.
    sparse_tile_form : str
        WHICH BODY of the sparse core serves the step: ``"auto"`` (default) = the vendored placement rule -- on cc 10.7 the LEAN decode
        tile (``mma.sync`` + ``cp.async``, one CTA of ``128 x N_MT`` threads, no TMEM / TMA / cluster) at ``batch_size x H_kv <= 4`` CTAs
        per chunk -- the bound's unit: the step's rows ride inside the CTA, so ``q_len_per_req`` is not in it -- at every served step width
        the body admits (under the in-block indexer the list is per row, which the lean tile serves at ``q_len_per_req == 1`` only:
        ``batch_size <= 2`` at ``q_len_per_req 1`` on two KV heads), the tcgen05 core
        everywhere else --, ``"lean"`` / ``"tcgen05"`` pin a body (an explicit body the step does not admit is a typed decline naming the
        arm).  Performance-only: the same function under either body within the vendored tree's budget, never bitwise ACROSS the bodies
        (bitwise within each); the body that compiled reads back in ``plan.sparse_tile`` (``"sparse_lean_d256_f16"`` / ``"sparse_d256_f16"``).
        A vocabulary other than these three is a ``ValueError`` before any vendored code.

    Returns
    -------
    Qwen38BlockPlan
        ``plan.run(...)`` launches it; ``plan.workspace_size``, ``plan.declaration`` (the compiled-in facts, the stages, the knob
        readbacks ``fuse_prepare`` / ``use_pdl`` / ``pdl_stages`` / ``sparse_tile_form`` / ``sparse_tile``, ``index_list_form``,
        ``n_blocks_max``, ``topk_knobs``, the launch count statement, the cache roots) and ``plan.supports(...)`` describe it.
    """
    from .experimental.qwen_3_8_block import api as _impl

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
        q_len_per_req=q_len_per_req,
        max_pages=max_pages,
        block_table=block_table,
        max_seq_len=max_seq_len,
        decode_split_kv=decode_split_kv,
        workspace_buffer=workspace_buffer,
        index_kv_cache=index_kv_cache,
        index_state_cache=index_state_cache,
        index_state_block_table=index_state_block_table,
        position_tables=position_tables,
        rope_position_tables_only=rope_position_tables_only,
        ring_slots=ring_slots,
        fuse_prepare=fuse_prepare,
        use_pdl=use_pdl,
        kv_cache_dtype=kv_cache_dtype,
        sparse_tile_form=sparse_tile_form,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_8_block_bf16_decode(
    hidden_states: torch.Tensor,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
    weights: "Qwen38BlockWeights",
    w_o: torch.Tensor,
    *,
    geometry: "Qwen38BlockGeometry",
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    index_kv_cache: torch.Tensor,
    index_state_cache: torch.Tensor,
    index_state_block_table: torch.Tensor,
    index_state_slot_mapping: Optional[torch.Tensor] = None,
    position_tables: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    q_len_per_req: int = 1,
    max_seq_len: Optional[int] = None,
    decode_split_kv: Optional[int] = None,
    kv_layout: str = "NHD",
    out: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    plan: Optional["Qwen38BlockPlan"] = None,
    block_ids_out: Optional[torch.Tensor] = None,
    block_lens_out: Optional[torch.Tensor] = None,
    sparse_tile_form: str = "auto",
) -> torch.Tensor:
    """The decode form: ``batch`` sequences x ``q_len_per_req`` new tokens each over the paged caches, the selection made in the block.

    Per step the block projects the step's rows (the Q / GATE / K / V bands and the indexer band), norms and rotates them, writes the K / V
    rows into the pools at ``slot_mapping`` and the raw indexer key into the ring, pools / norms / rotates the compressed key of the block a
    row completes into the paged compressed-key cache (through the MAIN block table), scores every complete block per row, keeps the
    ``top_k`` best, attends per row over those blocks plus the open tail of its own sequence (bottom-right causal), gates and projects out.
    A padded entry is ``seq_lens == 0`` + table row ``0`` + slot ``-1`` + ring row ``0`` (ring slot ``-1``) and yields ``out`` rows exactly 0
    with no write to any pool, ring or cache row.

    Parameters
    ----------
    hidden_states : torch.Tensor
        ``[batch * q_len_per_req, d_model]`` bf16, the sequences' rows consecutive.
    rope_cos, rope_sin : Optional[torch.Tensor]
        ``None`` (the position-table arm: every rotation reads ``position_tables`` at the row's position) or the per-token
        ``[batch * q_len_per_req, rope_dim]`` tables (the unfused arm; the position tables are still required for the block starts).
    weights : Qwen38BlockWeights
    w_o : torch.Tensor
    geometry : Qwen38BlockGeometry
    kv_cache : torch.Tensor or (torch.Tensor, torch.Tensor)
        See :func:`prepare_qwen_3_8_block_bf16`; the page size is read from its shape.
    slot_mapping : torch.Tensor
        ``[batch * q_len_per_req]`` int32 / int64 flat slots of the step's tokens; ``-1`` on a padded entry.
    block_table : torch.Tensor
        ``[batch, max_pages]`` int32 page table (the MAIN table; the compressed-key cache is addressed through it too).
    seq_lens : torch.Tensor
        ``[batch]`` int32 LOGICAL KV lengths INCLUDING the step's tokens.
    index_kv_cache : torch.Tensor
        The paged compressed-key cache ``[num_pages, page_size / 4, 1, 128]`` bf16 (read and written in place).
    index_state_cache : torch.Tensor
        The raw-key ring ``[num_ring_blocks, cap, 1, 128]`` bf16 (read and written in place).
    index_state_block_table : torch.Tensor
        ``[batch, 1]`` int32: the ring block per sequence (``0`` for a padded entry).
    index_state_slot_mapping : Optional[torch.Tensor]
        ``None`` (the ring rows derive in the kernel) or ``[batch * q_len_per_req]`` of ``slot_mapping``'s dtype: the stack's ring WRITE
        rows ``block x cap + pos % cap``, ``-1`` = no write (the ring-slots arm; bitwise).
    position_tables : (torch.Tensor, torch.Tensor)
        ``(cos_pos, sin_pos)`` of :func:`qwen_3_8_block_rope_position_tables`, bound for THIS call and checked against the bucket's plan
        (``[1 or batch, >= max_pages x page_size, rope_dim]``, its dtype and device): required without ``plan`` -- the cached plan never
        substitutes the pair another call compiled in; with ``plan``, ``None`` binds the pair the plan was prepared with.
    q_len_per_req : int
    max_seq_len : Optional[int]
    decode_split_kv : Optional[int]
    kv_layout : str
    out : Optional[torch.Tensor]
        Caller-owned output of ``hidden_states``' shape (allocated when ``None``).
    workspace_buffer : Optional[torch.Tensor]
        Overrides the plan's own workspace (uint8, at least ``plan.workspace_size``).
    plan : Optional[Qwen38BlockPlan]
        A prepared plan to run (skips the plan cache).  Without one the call runs the bucket's cached plan (an LRU of 32 shape buckets):
        under CUDA-graph capture only a bucket already cached (a cold one is a typed ``RuntimeError`` -- warm it with one eager call first),
        and a plan a captured graph runs is retained when the cache evicts it, released by
        ``flashinfer.experimental.qwen_3_8_block.api.clear_plan_cache()`` only.
    block_ids_out, block_lens_out : Optional[torch.Tensor]
        Optional ``[batch * q_len_per_req, top_k]`` / ``[batch * q_len_per_req]`` int32 buffers that receive the block's own selection
        (the ids strictly descending then ``-1``; the valid-prefix length) -- tests, A/B, a reuse stack; nothing on the step reads them.
    sparse_tile_form : str
        The sparse core's body knob of the bucket's cached plan (as in :func:`prepare_qwen_3_8_block_bf16`; a plan-cache key); ignored
        when ``plan`` is given (that plan's body runs).

    Returns
    -------
    torch.Tensor
        ``out`` (``[batch * q_len_per_req, d_model]``).
    """
    from .experimental.qwen_3_8_block import api as _impl

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
        index_kv_cache=index_kv_cache,
        index_state_cache=index_state_cache,
        index_state_block_table=index_state_block_table,
        index_state_slot_mapping=index_state_slot_mapping,
        position_tables=position_tables,
        q_len_per_req=q_len_per_req,
        max_seq_len=max_seq_len,
        decode_split_kv=decode_split_kv,
        kv_layout=kv_layout,
        out=out,
        workspace_buffer=workspace_buffer,
        plan=plan,
        block_ids_out=block_ids_out,
        block_lens_out=block_lens_out,
        sparse_tile_form=sparse_tile_form,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def qwen_3_8_block_workspace_size(
    form: str,
    *,
    geometry: "Qwen38BlockGeometry",
    device,
    page_size: int,
    batch_size: int,
    q_len_per_req: int = 1,
    max_pages: int,
    decode_split_kv: Optional[int] = None,
    fuse_prepare: Optional[bool] = None,
    ring_cap: Optional[int] = None,
    sparse_tile_form: str = "auto",
) -> int:
    """The workspace bytes a decode plan of this shape bucket needs, with no plan kept: what a serving stack sizes its ONE shared
    workspace with at init (before its memory profiling, when the caches do not exist yet) -- as the MAXIMUM of this query over
    EVERY ``(batch_size, q_len_per_req)`` bucket it will plan, with the knobs it will plan them with.  The largest batch is not the
    largest workspace: under ``sparse_tile_form="auto"`` an admissible batch 1 or 2 bucket (``top_k >= 8``) takes the lean body, whose split-KV partial slab
    (split 65 at the default k) is wider than the tcgen05 core's at batch 4 (at the 24 / 2 geometry, q_len 1, page 16, 512 pages,
    ring_cap 4: about 3.3 MB at batch 2 against about 1.8 MB at batch 4), so a buffer sized from the largest batch alone is rejected
    by the smaller bucket's plan with the typed workspace error.  The bucket is declared on stand-in samples and its kernels compiled (the carve includes
    the projection engines' scratch; the compile is served from the in-process memo or the compiled-plan cache after the first
    process).  The carve depends on the bucket, ``max_pages x page_size`` (the block table's reach) and the split -- never on the
    caches' page count or strides -- so the number equals the compiled plan's ``workspace_size``.  Typed errors as
    :func:`prepare_qwen_3_8_block_bf16`.

    Parameters
    ----------
    form : str
        ``"decode"``.
    geometry : Qwen38BlockGeometry
    device : torch.device or str
    page_size, batch_size, q_len_per_req, max_pages, decode_split_kv, fuse_prepare
        The bucket's facts, as in :func:`prepare_qwen_3_8_block_bf16` (``max_pages`` required: the table's columns).
    ring_cap : Optional[int]
        The ring's rows per block (default ``q_len_per_req + 3``, the minimum).
    sparse_tile_form : str
        The sparse core's body knob the plan will be made with (as in :func:`prepare_qwen_3_8_block_bf16`): the carve is the body's --
        the lean tile's split writes a wider partial slab than the core's -- so size the workspace with the knob the plans use.

    Returns
    -------
    int
        The bytes.
    """
    from .experimental.qwen_3_8_block import api as _impl

    return _impl.workspace_size(
        form,
        geometry=geometry,
        device=device,
        page_size=page_size,
        batch_size=batch_size,
        q_len_per_req=q_len_per_req,
        max_pages=max_pages,
        decode_split_kv=decode_split_kv,
        fuse_prepare=fuse_prepare,
        ring_cap=ring_cap,
        sparse_tile_form=sparse_tile_form,
    )


class Qwen38BlockWrapper:
    """Plan-and-run wrapper for the Qwen 3.8 Flash-Next gated attention block: one instance per (layer group, shape bucket),
    FlashInfer's ``plan()`` / ``run()`` idiom over a caller-owned workspace -- the serving idiom (one workspace sized at the maximum of
    :func:`qwen_3_8_block_workspace_size` over every bucket the stack plans serves every bucket and every layer run in turn on one
    stream; under ``"auto"`` the largest batch is not the largest workspace).

    .. warning::
        ``Qwen38BlockWrapper`` is experimental: it provides no compatibility guarantees and may change or be removed
        without deprecation.  Calling ``plan`` or ``run`` is the opt-in and needs no environment variable; the
        implementation lives in ``flashinfer.experimental.qwen_3_8_block``.

    SM107 (Rubin, cc 10.7) only, bf16, the DECODE form only.  ``plan`` runs outside CUDA-graph capture (it compiles the kernels and sizes
    the workspace; a workspace smaller than ``plan_info.workspace_size`` is a typed ``ValueError`` naming the size); the first ``run``
    after ``plan`` may already be captured.  One wrapper is a single-writer resource: do not ``plan`` while a captured or in-flight ``run``
    may read its buffers.  Re-plan when the shape bucket changes: a plan is exactly one bucket.
    """

    #: Mechanical identification: the decorators only mark ``plan`` and ``run``.
    is_experimental = True

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        *,
        geometry: "Qwen38BlockGeometry",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if (
            not isinstance(workspace_buffer, torch.Tensor)
            or workspace_buffer.dtype != torch.uint8
            or workspace_buffer.dim() != 1
        ):
            raise ValueError(
                "Qwen38BlockWrapper: workspace_buffer must be a 1-D uint8 CUDA tensor (the FlashInfer workspace convention)"
            )
        self._workspace = workspace_buffer
        self._kv_layout = str(kv_layout).upper()
        self._geometry = geometry
        self._dtype = dtype
        self._plan: Any = None

    @property
    def plan_info(self) -> "Qwen38BlockPlan":
        """The current plan (``None`` before ``plan``)."""
        return self._plan

    @flashinfer_experimental_api(feature=_FEATURE)
    def plan(
        self,
        form: str,
        *,
        weights: "Qwen38BlockWeights",
        w_o: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        page_size: int,
        batch_size: int,
        q_len_per_req: int = 1,
        max_pages: Optional[int] = None,
        block_table: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        decode_split_kv: Optional[int] = None,
        index_kv_cache: Optional[torch.Tensor] = None,
        index_state_cache: Optional[torch.Tensor] = None,
        index_state_block_table: Optional[torch.Tensor] = None,
        position_tables: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_position_tables_only: bool = True,
        ring_slots: bool = False,
        fuse_prepare: Optional[bool] = None,
        use_pdl: Optional[bool] = None,
        kv_cache_dtype: Optional[torch.dtype] = None,
        sparse_tile_form: str = "auto",
    ) -> None:
        """Plan one shape bucket over this wrapper's workspace, outside CUDA-graph capture.

        Parameters
        ----------
        form : str
            ``"decode"``.
        weights, w_o, kv_cache, page_size, batch_size, q_len_per_req, max_pages, block_table, max_seq_len, decode_split_kv, index_kv_cache, index_state_cache, index_state_block_table, position_tables, rope_position_tables_only, ring_slots, fuse_prepare, use_pdl, kv_cache_dtype, sparse_tile_form
            As in :func:`prepare_qwen_3_8_block_bf16` (``weights`` / ``w_o`` are samples: shape and dtype).
        """
        from .experimental.qwen_3_8_block import api as _impl

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
            q_len_per_req=q_len_per_req,
            max_pages=max_pages,
            block_table=block_table,
            max_seq_len=max_seq_len,
            decode_split_kv=decode_split_kv,
            workspace_buffer=self._workspace,
            index_kv_cache=index_kv_cache,
            index_state_cache=index_state_cache,
            index_state_block_table=index_state_block_table,
            position_tables=position_tables,
            rope_position_tables_only=rope_position_tables_only,
            ring_slots=ring_slots,
            fuse_prepare=fuse_prepare,
            use_pdl=use_pdl,
            kv_cache_dtype=kv_cache_dtype,
            sparse_tile_form=sparse_tile_form,
        )

    @flashinfer_experimental_api(feature=_FEATURE)
    def run(
        self,
        hidden_states: torch.Tensor,
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
        weights: "Qwen38BlockWeights",
        w_o: torch.Tensor,
        *,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        index_kv_cache: torch.Tensor,
        index_state_cache: torch.Tensor,
        index_state_block_table: torch.Tensor,
        index_state_slot_mapping: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        block_ids_out: Optional[torch.Tensor] = None,
        block_lens_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the planned decode step on the current CUDA stream (capturable from the first run).

        Parameters
        ----------
        hidden_states, rope_cos, rope_sin, weights, w_o, kv_cache, slot_mapping, block_table, seq_lens, index_kv_cache, index_state_cache, index_state_block_table, index_state_slot_mapping, out, block_ids_out, block_lens_out
            The decode form's arguments, as in :func:`qwen_3_8_block_bf16_decode`.

        Returns
        -------
        torch.Tensor
            ``out``.
        """
        if self._plan is None:
            raise RuntimeError("Qwen38BlockWrapper.run: call plan() first")
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
            out=out,
            workspace_buffer=self._workspace,
            index_kv_cache=index_kv_cache,
            index_state_cache=index_state_cache,
            index_state_block_table=index_state_block_table,
            index_state_slot_mapping=index_state_slot_mapping,
            block_ids_out=block_ids_out,
            block_lens_out=block_lens_out,
        )
