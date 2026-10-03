# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TraceTemplates for the QSA ops (``flashinfer.qsa_ops``)."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

_ROWS = ["num_rows"]
_ROUTE = ["num_rows", "route_width"]
_INDEX = {
    "token_to_request": Tensor(
        _ROWS, description="Request of each row; a negative entry empties the row."
    ),
    "query_positions": Tensor(_ROWS, description="Position of each query."),
    "seq_lens": Tensor(["num_requests"], description="KV length of each request."),
}
_TABLE = Tensor(
    ["num_requests", "table_width"],
    description="Logical to physical page per request; negative is unmapped.",
)
_PAGING = {
    "page_size": Scalar("int32", description="KV entries per physical page."),
    "num_slots": Scalar("int32", description="KV entries the cache holds."),
}


qsa_pre_indexer_trace = TraceTemplate(
    op_type="qsa_pre_indexer",
    name_prefix="qsa_pre_indexer",
    description=(
        "QSA pre-indexer: RMSNorm and partial NeoX RoPE of the index queries, "
        "mean-pooled compressed keys for every completed group, and the commit "
        "of each request's raw-key suffix to its ring."
    ),
    axes={
        "num_tokens": Var(),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
        "q_width": Const(abbrev="", description="num_heads * head_dim."),
        "max_position": Var(),
        "rope_width": Const(abbrev="", description="head_dim // 2."),
        "num_ring_blocks": Var(),
        "ring_size": Const(abbrev="ring"),
        "ring_width": Const(abbrev="", description="head_dim, or head_dim + 12."),
        "one": Const(abbrev="", value=1),
        "num_requests": Var(),
        "len_indptr": Var(),
        "num_compressed_blocks": Var(),
        "compressed_page": Const(abbrev="ps"),
        "num_work": Var(),
        "two": Const(abbrev="", value=2),
    },
    inputs={
        "q": Tensor(["num_tokens", "q_width"]),
        "k": Tensor(["num_tokens", "head_dim"]),
        "positions": Tensor(
            ["num_tokens"],
            description="Rotary coordinates; [3, num_tokens] for a three-axis rope.",
        ),
        "cos_sin_cache": Tensor(["max_position", "rope_width"]),
        "q_norm_weight": Tensor(["head_dim"]),
        "k_norm_weight": Tensor(["head_dim"]),
        "eps": Scalar("float32"),
        "q_out": Tensor(["num_tokens", "num_heads", "head_dim"]),
        "state_cache": Tensor(["num_ring_blocks", "ring_size", "one", "ring_width"]),
        "state_slots": Tensor(["num_tokens"]),
        "state_block_table": Tensor(["num_requests", "one"]),
        "qo_indptr": Tensor(["len_indptr"]),
        "logical_positions": Tensor(["num_tokens"]),
        "compressed_cache": Tensor(
            ["num_compressed_blocks", "compressed_page", "one", "head_dim"]
        ),
        "compressed_slots": Tensor(["num_tokens"]),
        "work_metadata": Tensor(["num_work", "two"]),
        "compress_ratio": Scalar("int32"),
        "mrope_h": Scalar("int32", optional=True),
        "mrope_w": Scalar("int32", optional=True),
        "is_k_mrope": Scalar("int32", optional=True, description="Bool."),
        "cache_has_rope_pos": Scalar("int32", optional=True, description="Bool."),
    },
    outputs={
        "q_out": Tensor(
            ["num_tokens", "num_heads", "head_dim"], param="q_out", dtype_from="q_out"
        ),
        "state_cache": Tensor(
            ["num_ring_blocks", "ring_size", "one", "ring_width"],
            param="state_cache",
            dtype_from="state_cache",
        ),
        "compressed_cache": Tensor(
            ["num_compressed_blocks", "compressed_page", "one", "head_dim"],
            param="compressed_cache",
            dtype_from="compressed_cache",
        ),
    },
    tags=["status:verified", "sparse"],
)

qsa_paged_scores_trace = TraceTemplate(
    op_type="qsa_paged_scores",
    name_prefix="qsa_paged_scores",
    description=(
        "QSA scorer: sum over heads of ReLU(K . Q) / divisor for every compressed "
        "entry a query can see; -inf on unmapped pages, and the count scored."
    ),
    axes={
        "num_rows": Var(),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
        "num_pages": Var(),
        "page_size": Const(abbrev="ps"),
        "num_requests": Var(),
        "table_width": Var(),
        "num_columns": Var(),
    },
    inputs={
        "q": Tensor(["num_rows", "num_heads", "head_dim"]),
        "k_cache": Tensor(["num_pages", "page_size", "head_dim"]),
        "block_table": _TABLE,
        **_INDEX,
        "compress_ratio": Scalar("int32"),
        "divisor": Scalar("float32"),
        "num_columns": Scalar("int32", optional=True),
        "logits": Tensor(["num_rows", "num_columns"], optional=True),
        "visible_blocks": Tensor(_ROWS, optional=True),
    },
    outputs={
        "logits": Tensor(["num_rows", "num_columns"], dtype="float32"),
        "visible_blocks": Tensor(_ROWS, dtype_from="block_table"),
    },
    tags=["status:verified", "sparse"],
)

qsa_expand_block_route_trace = TraceTemplate(
    op_type="qsa_route",
    name_prefix="qsa_expand_block_route",
    description=(
        "QSA route expansion: every selected block becomes its compress_ratio "
        "tokens, then the seen tokens of the query's own block; -1 elsewhere."
    ),
    axes={
        "num_rows": Var(),
        "block_topk": Const(abbrev="k"),
        "num_requests": Var(),
        "route_width": Var(description="block_topk * ratio + ratio - 1."),
    },
    inputs={
        "indexer_block_ids": Tensor(["num_rows", "block_topk"]),
        **_INDEX,
        "compress_ratio": Scalar("int32"),
        "out": Tensor(_ROUTE, optional=True),
    },
    outputs={"route": Tensor(_ROUTE, dtype_from="indexer_block_ids")},
    tags=["status:verified", "sparse"],
)

qsa_route_from_blocks_trace = TraceTemplate(
    op_type="qsa_route",
    name_prefix="qsa_route_from_blocks",
    description=(
        "QSA route expansion fused with the block-table lookup: the logical route, "
        "its physical KV slots, and their packed validity mask."
    ),
    axes={
        "num_rows": Var(),
        "block_topk": Const(abbrev="k"),
        "num_requests": Var(),
        "table_width": Var(),
        "route_width": Var(description="block_topk * ratio + ratio - 1."),
        "mask_bytes": Var(description="num_rows * ceil(route_width / 8)."),
    },
    inputs={
        "indexer_block_ids": Tensor(["num_rows", "block_topk"]),
        **_INDEX,
        "block_table": _TABLE,
        "out_logical": Tensor(_ROUTE),
        "out_route": Tensor(_ROUTE),
        "out_mask": Tensor(["mask_bytes"]),
        "compress_ratio": Scalar("int32"),
        **_PAGING,
    },
    outputs={
        "out_logical": Tensor(_ROUTE, param="out_logical", dtype_from="out_logical"),
        "out_route": Tensor(_ROUTE, param="out_route", dtype_from="out_route"),
        "out_mask": Tensor(["mask_bytes"], param="out_mask", dtype="uint8"),
    },
    tags=["status:verified", "sparse"],
)

qsa_route_from_logical_trace = TraceTemplate(
    op_type="qsa_route",
    name_prefix="qsa_route_from_logical",
    description=(
        "QSA block-table lookup of a logical route kept from an earlier step: "
        "physical KV slots and their packed validity mask."
    ),
    axes={
        "num_rows": Var(),
        "num_requests": Var(),
        "table_width": Var(),
        "route_width": Var(),
        "mask_bytes": Var(description="num_rows * ceil(route_width / 8)."),
        "num_indptr": Var(description="num_rows + 1."),
    },
    inputs={
        "logical": Tensor(_ROUTE),
        "token_to_request": _INDEX["token_to_request"],
        "block_table": _TABLE,
        "out_route": Tensor(_ROUTE),
        "out_mask": Tensor(["mask_bytes"]),
        "valid_rows": Scalar("int32", description="Rows past it come out masked."),
        **_PAGING,
        "out_indptr": Tensor(["num_indptr"], optional=True),
    },
    outputs={
        "out_route": Tensor(_ROUTE, param="out_route", dtype_from="out_route"),
        "out_mask": Tensor(["mask_bytes"], param="out_mask", dtype="uint8"),
        "out_indptr": Tensor(
            ["num_indptr"],
            param="out_indptr",
            dtype="int32",
            optional=True,
            description="min(r, valid_rows) * route_width: padding rows get none.",
        ),
    },
    tags=["status:verified", "sparse"],
)

qsa_output_gate_trace = TraceTemplate(
    op_type="qsa_output_gate",
    name_prefix="qsa_output_gate",
    description="QSA output gate: attention * sigmoid(gate), formed in float.",
    axes={
        "num_rows": Var(),
        "num_attention_rows": Var(description="At least num_rows; padding is unread."),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
    },
    inputs={
        "attention": Tensor(["num_attention_rows", "num_heads", "head_dim"]),
        "gate": Tensor(["num_rows", "num_heads", "head_dim"]),
        "out": Tensor(["num_rows", "num_heads", "head_dim"], optional=True),
    },
    outputs={"out": Tensor(["num_rows", "num_heads", "head_dim"], dtype_from="gate")},
    tags=["status:verified"],
)

qsa_selection_run_trace = TraceTemplate(
    op_type="qsa_selection",
    name_prefix="qsa_selection",
    description=(
        "QSASelection.run(): score the compressed cache, keep the top blocks of "
        "each row, and expand them into the caller's logical route."
    ),
    axes={
        "num_rows": Var(),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
        "num_pages": Var(),
        "page_size": Const(abbrev="ps"),
        "num_requests": Var(),
        "table_width": Var(),
        "route_width": Const(abbrev="w"),
        "workspace_bytes": Var(),
    },
    inputs={
        "q": Tensor(["num_rows", "num_heads", "head_dim"]),
        "k_compressed": Tensor(["num_pages", "page_size", "head_dim"]),
        "block_table": _TABLE,
        **_INDEX,
        "out_route": Tensor(_ROUTE),
        "workspace": Tensor(["workspace_bytes"], optional=True),
    },
    outputs={"out_route": Tensor(_ROUTE, param="out_route", dtype="int32")},
    tags=["status:verified", "sparse"],
)

_CACHE = ["num_pages", "cache_axis_1", "cache_axis_2", "entry"]
_PLANES = ["num_pages", "cache_axis_1", "cache_axis_2", "scale_entry"]

qsa_attention_run_trace = TraceTemplate(
    op_type="qsa_attention",
    name_prefix="qsa_attention",
    description=(
        "QSAAttention.run(): map a logical token route through the block table, "
        "attend over those KV entries of a paged cache (dense, FP8 or NVFP4), "
        "and apply the output gate."
    ),
    axes={
        "num_rows": Var(),
        "num_qo_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
        "num_pages": Var(),
        "cache_axis_1": Const(abbrev="", description="page_size (NHD) or heads (HND)."),
        "cache_axis_2": Const(abbrev="", description="heads (NHD) or page_size (HND)."),
        "entry": Const(abbrev="", description="head_dim, or head_dim // 2 (NVFP4)."),
        "scale_entry": Var(description="head_dim // 16, for NVFP4 only."),
        "route_width": Const(abbrev="w"),
        "num_requests": Var(),
        "table_width": Var(),
    },
    inputs={
        "q": Tensor(["num_rows", "num_qo_heads", "head_dim"]),
        "k_data": Tensor(_CACHE),
        "v_data": Tensor(_CACHE),
        "route": Tensor(_ROUTE, description="Logical tokens, -1 where none."),
        "block_table": _TABLE,
        "token_to_request": _INDEX["token_to_request"],
        "output_gate": Tensor(
            ["num_rows", "num_qo_heads", "head_dim"],
            description="Or [num_rows, num_qo_heads * head_dim].",
        ),
        "k_sf": Tensor(_PLANES, optional=True, description="NVFP4 block scales."),
        "v_sf": Tensor(_PLANES, optional=True, description="NVFP4 block scales."),
        "k_scale": Scalar("float32", optional=True),
        "v_scale": Scalar("float32", optional=True),
        "out": Tensor(["num_rows", "num_qo_heads", "head_dim"], optional=True),
    },
    outputs={
        "out": Tensor(
            ["num_rows", "num_qo_heads", "head_dim"], dtype_from="output_gate"
        )
    },
    tags=["status:verified", "sparse"],
)
