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
