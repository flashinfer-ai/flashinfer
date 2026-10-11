# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""Inputs of token-sparse attention."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

token_sparse_kv_indices_trace = TraceTemplate(
    op_type="token_sparse_kv_indices",
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
        "block_tables": Tensor(["rows", "lists", "top_k"], dtype="int32"),
        "seq_lens": Tensor(["rows"], dtype="int32"),
    },
    description="Token-sparse attention's K/V slot lists from key positions.",
)

group_union_masks_trace = TraceTemplate(
    op_type="group_union_masks",
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
        "union": Tensor(["unions", "columns"], dtype="int32"),
        "union_sizes": Tensor(["unions"], dtype="int32"),
        "mask": Tensor(["masks", "words"], dtype="int32"),
    },
    description="Each token group's union of keys as cache slots, and each token's bitmask over it.",
)
