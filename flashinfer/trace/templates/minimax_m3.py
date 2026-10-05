# SPDX-License-Identifier: Apache-2.0
"""Tracing schema for paged MiniMax M3 index selection."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

minimax_m3_index_decode_trace = TraceTemplate(
    op_type="minimax_m3_index_decode",
    axes={
        "batch": Var(),
        "pages": Var(),
        "heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
        "page_size": Const(abbrev="ps"),
        "table_width": Const(abbrev="bt"),
        "num_kv_heads": Const(abbrev="kv"),
        "topk": Const(abbrev="k"),
    },
    inputs={
        "idx_q": Tensor(["batch", "heads", "head_dim"]),
        "index_kv_cache": Tensor(["pages", "page_size", "head_dim"]),
        "block_table": Tensor(["batch", "table_width"]),
        "seq_lens": Tensor(["batch"]),
        "max_seq_len": Scalar("int32"),
        "topk": Scalar("int32"),
        "init_blocks": Scalar("int32"),
        "local_blocks": Scalar("int32"),
        "num_kv_heads": Scalar("int32"),
        "decode_query_len": Scalar("int32"),
        "max_decode_query_len": Scalar("int32"),
    },
    outputs={"indices": Tensor(["num_kv_heads", "batch", "topk"], dtype="int32")},
    description="Logical block IDs with a valid selected prefix and trailing -1.",
)
