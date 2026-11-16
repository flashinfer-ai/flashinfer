# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""TraceTemplates of the SDSA (Simplified DSA) APIs in :mod:`flashinfer.sdsa`."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

indexer_scores_trace = TraceTemplate(
    op_type="sdsa_indexer_scores",
    description=(
        "Single-head paged MQA logits: FP32 dot products of query rows with their "
        "request's paged BF16 or NVFP4 keys (no head weights or ReLU). SM100/SM103."
    ),
    axes={
        "rows": Var(description="Query rows, 1 to 8 per request group."),
        "groups": Var(description="Groups of query rows, one request each."),
        "requests": Var(description="Block-table rows."),
        "pages": Var(description="Key cache pages."),
        "blocks": Var(description="64-key blocks per request."),
        "max_context": Var(description="Score columns: the context capacity."),
        "page_size": Const(description="Keys per page."),
        "dim": Const(description="Elements per key: 128 BF16, or 64 NVFP4 bytes."),
        "scales": Var(description="NVFP4's E4M3 scales per key, one per 16 values."),
    },
    inputs={
        "query": Tensor(["rows", "dim"]),
        "key_cache": Tensor(["pages", "page_size", "dim"]),
        "block_table": Tensor(["requests", "blocks"]),
        "context_lengths": Tensor(["groups"]),
        "requests": Tensor(["groups"]),
        "scores": Tensor(["rows", "max_context"]),
        "query_scales": Tensor(["rows", "scales"], optional=True),
        "key_scales": Tensor(["pages", "page_size", "scales"], optional=True),
    },
    outputs={
        "scores": Tensor(["rows", "max_context"], dtype="float32", param="scores")
    },
    tags=["arch:sm100", "arch:sm103"],
)


indexer_threshold_select_trace = TraceTemplate(
    op_type="sdsa_indexer_threshold_select",
    description=(
        "Keys of each query row's request scoring above the row's threshold, with "
        "one query head over paged BF16 or NVFP4 keys, written as (key, score) "
        "candidate and tie slots. SM100/SM103."
    ),
    axes={
        "rows": Var(description="Query rows, 128 per tile."),
        "tiles": Var(description="Query tiles, one request each."),
        "requests": Var(description="Block-table rows."),
        "pages": Var(description="Key cache pages."),
        "blocks": Var(description="64-key blocks per request."),
        "page_size": Const(description="Keys per page."),
        "dim": Const(description="Elements per key: 128 BF16, or 64 NVFP4 bytes."),
        "scales": Var(description="NVFP4's E4M3 scales per key, one per 16 values."),
        "slots": Var(description="Tie slots, then candidate slots."),
        "counters": Const(description="Candidate and tie counts."),
        "pair": Const(description="Key index and score bits."),
        "range": Var(description="A row's first and past-last Top-K slot."),
    },
    inputs={
        "query": Tensor(["rows", "dim"]),
        "key_cache": Tensor(["pages", "page_size", "dim"]),
        "row_ends": Tensor(["rows"]),
        "tile_requests": Tensor(["tiles"]),
        "block_table": Tensor(["requests", "blocks"]),
        "thresholds": Tensor(["rows"]),
        "counters": Tensor(["rows", "counters"]),
        "candidates": Tensor(["rows", "slots", "pair"]),
        "max_context": Scalar("int32"),
        "top_k": Scalar("int32"),
        "query_scales": Tensor(["rows", "scales"], optional=True),
        "key_scales": Tensor(["pages", "page_size", "scales"], optional=True),
        "arrivals": Tensor(["tiles"], optional=True),
        "ranges": Tensor(["range", "rows"], optional=True),
    },
    outputs={
        "thresholds": Tensor(["rows"], dtype="float32", param="thresholds"),
        "ranges": Tensor(
            ["range", "rows"], dtype="int32", param="ranges", optional=True
        ),
        "counters": Tensor(["rows", "counters"], dtype="int32", param="counters"),
        "candidates": Tensor(
            ["rows", "slots", "pair"], dtype="int32", param="candidates"
        ),
    },
    constraints=["rows == 128 * tiles"],
    tags=["arch:sm100", "arch:sm103"],
)

quantize_write_index_qk_nvfp4_trace = TraceTemplate(
    op_type="sdsa_quantize_write_index_qk_nvfp4",
    description=(
        "Indexer keys and queries quantized to NVFP4 (E4M3 scale per 16 values) in "
        "one launch: keys at their slots of a 64-key paged cache, queries at their "
        "rows."
    ),
    axes={
        "tokens": Var(description="Keys to write."),
        "queries": Var(description="Queries to write."),
        "dim": Const(description="Key and query dimension."),
        "slots": Var(description="Slot mapping entries, one or more per key."),
        "query_slots": Var(description="Query row entries, one or more per query."),
        "pages": Var(description="Key cache pages."),
        "rows": Var(description="Packed query rows."),
        "page_size": Const(description="Keys per page."),
        "packed_dim": Const(description="Packed NVFP4 bytes per key or query."),
        "scales": Const(description="E4M3 scales per key or query, one per 16 values."),
    },
    inputs={
        "key": Tensor(["tokens", "dim"]),
        "slots": Tensor(["slots"]),
        "query": Tensor(["queries", "dim"]),
        "query_rows": Tensor(["query_slots"]),
        "key_cache": Tensor(["pages", "page_size", "packed_dim"]),
        "key_scales": Tensor(["pages", "page_size", "scales"]),
        "packed_query": Tensor(["rows", "packed_dim"]),
        "query_scales": Tensor(["rows", "scales"]),
    },
    outputs={
        "key_cache": Tensor(
            ["pages", "page_size", "packed_dim"], dtype="uint8", param="key_cache"
        ),
        "key_scales": Tensor(
            ["pages", "page_size", "scales"],
            dtype_from="key_scales",
            dtype="float8_e4m3fn",
            param="key_scales",
        ),
        "packed_query": Tensor(
            ["rows", "packed_dim"], dtype="uint8", param="packed_query"
        ),
        "query_scales": Tensor(
            ["rows", "scales"],
            dtype_from="query_scales",
            dtype="float8_e4m3fn",
            param="query_scales",
        ),
    },
    tags=["arch:sm100", "arch:sm103"],
)


gather_key_sample_trace = TraceTemplate(
    op_type="sdsa_gather_key_sample",
    description=(
        "A strided random sample of each request's paged BF16 or NVFP4 keys, "
        "copied into dense sample rows."
    ),
    axes={
        "pages": Var(description="Key cache pages."),
        "page_size": Const(description="Keys per page."),
        "dim": Const(description="Elements per key: 128 BF16, or 64 NVFP4 bytes."),
        "scales": Var(description="NVFP4's E4M3 scales per key, one per 16 values."),
        "requests": Var(description="Block-table rows, one sample row each."),
        "blocks": Var(description="64-key blocks per request."),
        "capacity": Var(description="Sample slots per request, in 128-key tiles."),
    },
    inputs={
        "key_cache": Tensor(["pages", "page_size", "dim"]),
        "block_table": Tensor(["requests", "blocks"]),
        "offsets": Tensor(["capacity"]),
        "sample_ends": Tensor(["requests"]),
        "sample_keys": Tensor(["requests", "capacity", "dim"]),
        "samples_per_run": Scalar("int32"),
        "pages_per_run": Scalar("int32"),
        "key_scales": Tensor(["pages", "page_size", "scales"], optional=True),
        "sample_scales": Tensor(["requests", "capacity", "scales"], optional=True),
    },
    outputs={
        "sample_keys": Tensor(
            ["requests", "capacity", "dim"],
            dtype_from="key_cache",
            param="sample_keys",
        ),
        "sample_scales": Tensor(
            ["requests", "capacity", "scales"],
            dtype_from="sample_scales",
            dtype="float8_e4m3fn",
            param="sample_scales",
            optional=True,
        ),
    },
    tags=["arch:sm100", "arch:sm103"],
)


thresholds_from_sample_trace = TraceTemplate(
    op_type="sdsa_thresholds_from_sample",
    description=(
        "Each query row's threshold as the least of its sampled Top scores, with "
        "its threshold-select counters zeroed."
    ),
    axes={
        "rows": Var(description="Query rows."),
        "samples": Var(description="Sampled scores per row."),
        "rank": Var(description="Top positions per row."),
        "counters": Const(description="Candidate and tie counts."),
    },
    inputs={
        "sample_scores": Tensor(["rows", "samples"]),
        "sample_top": Tensor(["rows", "rank"]),
        "thresholds": Tensor(["rows"]),
        "counters": Tensor(["rows", "counters"]),
    },
    outputs={
        "thresholds": Tensor(["rows"], dtype="float32", param="thresholds"),
        "counters": Tensor(["rows", "counters"], dtype="int32", param="counters"),
    },
    tags=["arch:sm100", "arch:sm103"],
)


top_k_from_candidates_trace = TraceTemplate(
    op_type="sdsa_top_k_from_candidates",
    description=(
        "Each query row's exact Top-K keys among the (key, score bits) candidate "
        "and tie pairs that single-head threshold select stored in its range of "
        "slots."
    ),
    axes={
        "rows": Var(description="Query rows."),
        "slots": Var(description="Tie slots, then candidate slots."),
        "pair": Const(description="Key index and score bits."),
        "range": Const(description="A row's first and past-last Top-K slot."),
        "top_k": Var(description="Keys per row."),
    },
    inputs={
        "candidates": Tensor(["rows", "slots", "pair"]),
        "ranges": Tensor(["range", "rows"]),
        "top_keys": Tensor(["rows", "top_k"]),
    },
    outputs={
        "top_keys": Tensor(["rows", "top_k"], dtype="int32", param="top_keys"),
    },
    tags=["arch:sm100", "arch:sm103"],
)

token_sparse_kv_indices_trace = TraceTemplate(
    op_type="sdsa_token_sparse_kv_indices",
    axes={
        "tokens": Var(description="Query tokens."),
        "top_k": Const(description="Key positions per token, a multiple of 512."),
        "requests": Var(description="Block-table rows."),
        "blocks": Var(description="64-key blocks per request."),
        "rows": Var(description="Attention requests, requests_per_token per token."),
        "lists": Const(description="K slots, then V slots.", value=2),
    },
    inputs={
        "positions": Tensor(["tokens", "top_k"]),
        "requests": Tensor(["tokens"]),
        "block_table": Tensor(["requests", "blocks"]),
        "block_tables": Tensor(["rows", "lists", "top_k"]),
        "seq_lens": Tensor(["rows"]),
        "requests_per_token": Scalar("int32"),
    },
    outputs={
        "block_tables": Tensor(
            ["rows", "lists", "top_k"], dtype="int32", param="block_tables"
        ),
        "seq_lens": Tensor(["rows"], dtype="int32", param="seq_lens"),
    },
    description="Token-sparse attention's K/V slot lists from key positions.",
)

group_union_masks_trace = TraceTemplate(
    op_type="sdsa_group_union_masks",
    axes={
        "tokens": Var(
            description="Query tokens, in groups of consecutive tokens, all one size."
        ),
        "top_k": Const(description="Key positions per token, a multiple of 512."),
        "groups": Var(description="Token groups, one union each."),
        "requests": Var(description="Block-table rows."),
        "blocks": Var(description="64-key blocks per request."),
        "unions": Var(description="Union rows."),
        "columns": Var(description="Union capacity in keys."),
        "masks": Var(description="Mask rows."),
        "words": Var(description="Mask words per row, 32 union columns each."),
    },
    inputs={
        "positions": Tensor(["tokens", "top_k"]),
        "query_rows": Tensor(["tokens"]),
        "group_unions": Tensor(["groups"]),
        "requests": Tensor(["groups"]),
        "block_table": Tensor(["requests", "blocks"]),
        "union": Tensor(["unions", "columns"]),
        "union_sizes": Tensor(["unions"]),
        "mask": Tensor(["masks", "words"]),
        "num_keys": Scalar("int32"),
    },
    outputs={
        "union": Tensor(["unions", "columns"], dtype="int32", param="union"),
        "union_sizes": Tensor(["unions"], dtype="int32", param="union_sizes"),
        "mask": Tensor(["masks", "words"], dtype="int32", param="mask"),
    },
    description="Each token group's union of keys as cache slots, and each token's bitmask over it.",
)
