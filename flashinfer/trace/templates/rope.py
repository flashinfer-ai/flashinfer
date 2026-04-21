# Copyright (c) 2025 by FlashInfer team.
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

"""TraceTemplates for RoPE (Rotary Position Embedding) operations."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

# ── Shared axes ───────────────────────────────────────────────────────────────

_RAGGED_AXES = {
    "nnz": Var(description="Total number of tokens across the batch."),
    "batch_size": Var(description="Number of sequences in the batch."),
    "num_q_heads": Const(abbrev="h"),
    "num_k_heads": Const(abbrev="kv"),
    "head_dim": Const(abbrev="d"),
}

_POSIDS_AXES = {
    "nnz": Var(description="Total number of tokens across the batch."),
    "num_q_heads": Const(abbrev="h"),
    "num_k_heads": Const(abbrev="kv"),
    "head_dim": Const(abbrev="d"),
}

_COSSIN_AXES = {
    "nnz": Var(description="Total number of tokens across the batch."),
    "num_q_heads_x_head_size": Const(
        description="num_q_heads * head_size (flattened query dimension).", abbrev=""
    ),
    "num_k_heads_x_head_size": Const(
        description="num_k_heads * head_size (flattened key dimension).", abbrev=""
    ),
    "head_size": Const(abbrev="d"),
    "max_seq_len": Var(description="cos_sin_cache length (max supported position)."),
    "rotary_dim": Const(
        description="Rotary dimension (cos+sin concatenated along last axis).",
        abbrev="",
    ),
}

# ── Base ragged RoPE (indptr + offsets) ──────────────────────────────────────

_RAGGED_INPUTS = {
    "q": Tensor(["nnz", "num_q_heads", "head_dim"]),
    "k": Tensor(["nnz", "num_k_heads", "head_dim"]),
    "indptr": Tensor(
        ["batch_size_plus_1"],
        dtype="int32",
        description="Ragged batch indptr, shape (batch_size + 1).",
    ),
    "offsets": Tensor(
        ["batch_size"],
        dtype="int32",
        description="Per-sequence starting position offset.",
    ),
    "rotary_dim": Scalar(
        "int32",
        optional=True,
        description="If None, uses head_dim. Rotate only the first `rotary_dim` dims.",
    ),
    "interleave": Scalar(
        "int32",
        optional=True,
        description="Bool: interleaved (True) vs half-split (False) rotation.",
    ),
    "rope_scale": Scalar("float32", optional=True, description="Scale factor."),
    "rope_theta": Scalar("float32", optional=True, description="Theta value."),
}

apply_rope_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope",
    description="Standard RoPE on ragged q/k using indptr + per-seq offsets.",
    axes={**_RAGGED_AXES, "batch_size_plus_1": Var(description="batch_size + 1.")},
    inputs=_RAGGED_INPUTS,
    outputs={
        "q_rope": Tensor(["nnz", "num_q_heads", "head_dim"], dtype_from="q"),
        "k_rope": Tensor(["nnz", "num_k_heads", "head_dim"], dtype_from="k"),
    },
    constraints=["batch_size_plus_1 == batch_size + 1"],
    tags=["status:verified"],
)

apply_rope_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_inplace",
    description="In-place standard RoPE; q and k are mutated.",
    axes={**_RAGGED_AXES, "batch_size_plus_1": Var(description="batch_size + 1.")},
    inputs=_RAGGED_INPUTS,
    outputs={
        "q": Tensor(
            ["nnz", "num_q_heads", "head_dim"],
            dtype_from="q",
            description="Updated q (in-place).",
        ),
        "k": Tensor(
            ["nnz", "num_k_heads", "head_dim"],
            dtype_from="k",
            description="Updated k (in-place).",
        ),
    },
    constraints=["batch_size_plus_1 == batch_size + 1"],
    tags=["status:verified"],
)

# ── pos_ids RoPE ──────────────────────────────────────────────────────────────

_POSIDS_INPUTS = {
    "q": Tensor(["nnz", "num_q_heads", "head_dim"]),
    "k": Tensor(["nnz", "num_k_heads", "head_dim"]),
    "pos_ids": Tensor(["nnz"], dtype="int32", description="Per-token position index."),
    "rotary_dim": Scalar("int32", optional=True),
    "interleave": Scalar("int32", optional=True),
    "rope_scale": Scalar("float32", optional=True),
    "rope_theta": Scalar("float32", optional=True),
}

apply_rope_pos_ids_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_pos_ids",
    description="Standard RoPE using explicit per-token position ids.",
    axes=_POSIDS_AXES,
    inputs=_POSIDS_INPUTS,
    outputs={
        "q_rope": Tensor(["nnz", "num_q_heads", "head_dim"], dtype_from="q"),
        "k_rope": Tensor(["nnz", "num_k_heads", "head_dim"], dtype_from="k"),
    },
    tags=["status:verified"],
)

apply_rope_pos_ids_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_pos_ids_inplace",
    description="In-place RoPE using explicit per-token position ids.",
    axes=_POSIDS_AXES,
    inputs=_POSIDS_INPUTS,
    outputs={
        "q": Tensor(
            ["nnz", "num_q_heads", "head_dim"],
            dtype_from="q",
            description="Updated q (in-place).",
        ),
        "k": Tensor(
            ["nnz", "num_k_heads", "head_dim"],
            dtype_from="k",
            description="Updated k (in-place).",
        ),
    },
    tags=["status:verified"],
)

# ── Llama 3.1 RoPE ────────────────────────────────────────────────────────────

_LLAMA31_EXTRA = {
    "low_freq_factor": Scalar(
        "float32", optional=True, description="Llama 3.1 low-frequency scaling factor."
    ),
    "high_freq_factor": Scalar(
        "float32", optional=True, description="Llama 3.1 high-frequency scaling factor."
    ),
    "old_context_len": Scalar(
        "int32", optional=True, description="Original pretraining context length."
    ),
}

_LLAMA31_RAGGED_INPUTS = {**_RAGGED_INPUTS, **_LLAMA31_EXTRA}
_LLAMA31_POSIDS_INPUTS = {**_POSIDS_INPUTS, **_LLAMA31_EXTRA}

apply_llama31_rope_trace = TraceTemplate(
    op_type="rope",
    name_prefix="llama31_rope",
    description="Llama 3.1 RoPE on ragged q/k with indptr + offsets.",
    axes={**_RAGGED_AXES, "batch_size_plus_1": Var(description="batch_size + 1.")},
    inputs=_LLAMA31_RAGGED_INPUTS,
    outputs={
        "q_rope": Tensor(["nnz", "num_q_heads", "head_dim"], dtype_from="q"),
        "k_rope": Tensor(["nnz", "num_k_heads", "head_dim"], dtype_from="k"),
    },
    constraints=["batch_size_plus_1 == batch_size + 1"],
    tags=["status:verified", "model:llama"],
)

apply_llama31_rope_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="llama31_rope_inplace",
    description="In-place Llama 3.1 RoPE with indptr + offsets.",
    axes={**_RAGGED_AXES, "batch_size_plus_1": Var(description="batch_size + 1.")},
    inputs=_LLAMA31_RAGGED_INPUTS,
    outputs={
        "q": Tensor(
            ["nnz", "num_q_heads", "head_dim"],
            dtype_from="q",
            description="Updated q (in-place).",
        ),
        "k": Tensor(
            ["nnz", "num_k_heads", "head_dim"],
            dtype_from="k",
            description="Updated k (in-place).",
        ),
    },
    constraints=["batch_size_plus_1 == batch_size + 1"],
    tags=["status:verified", "model:llama"],
)

apply_llama31_rope_pos_ids_trace = TraceTemplate(
    op_type="rope",
    name_prefix="llama31_rope_pos_ids",
    description="Llama 3.1 RoPE using per-token position ids.",
    axes=_POSIDS_AXES,
    inputs=_LLAMA31_POSIDS_INPUTS,
    outputs={
        "q_rope": Tensor(["nnz", "num_q_heads", "head_dim"], dtype_from="q"),
        "k_rope": Tensor(["nnz", "num_k_heads", "head_dim"], dtype_from="k"),
    },
    tags=["status:verified", "model:llama"],
)

apply_llama31_rope_pos_ids_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="llama31_rope_pos_ids_inplace",
    description="In-place Llama 3.1 RoPE using per-token position ids.",
    axes=_POSIDS_AXES,
    inputs=_LLAMA31_POSIDS_INPUTS,
    outputs={
        "q": Tensor(
            ["nnz", "num_q_heads", "head_dim"],
            dtype_from="q",
            description="Updated q (in-place).",
        ),
        "k": Tensor(
            ["nnz", "num_k_heads", "head_dim"],
            dtype_from="k",
            description="Updated k (in-place).",
        ),
    },
    tags=["status:verified", "model:llama"],
)

# ── cos/sin cache variant (SGL/vLLM-compatible) ───────────────────────────────

_COSSIN_INPUTS = {
    "positions": Tensor(
        ["nnz"], dtype="int32", description="Per-token position index."
    ),
    "query": Tensor(
        ["nnz", "num_q_heads_x_head_size"],
        description="Flattened query tensor (nnz, num_q_heads * head_size).",
    ),
    "key": Tensor(
        ["nnz", "num_k_heads_x_head_size"],
        description="Flattened key tensor (nnz, num_k_heads * head_size).",
    ),
    "head_size": Scalar("int32", description="Head dimension."),
    "cos_sin_cache": Tensor(
        ["max_seq_len", "rotary_dim"],
        dtype="float32",
        description="Precomputed cos+sin cache; cos first half, sin second half.",
    ),
    "is_neox": Scalar(
        "int32", optional=True, description="Bool: Neox (True) vs interleaved (False)."
    ),
}

apply_rope_with_cos_sin_cache_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_cos_sin_cache",
    description="RoPE with precomputed cos/sin cache (SGL/vLLM-compatible).",
    axes=_COSSIN_AXES,
    inputs=_COSSIN_INPUTS,
    outputs={
        "query_out": Tensor(["nnz", "num_q_heads_x_head_size"], dtype_from="query"),
        "key_out": Tensor(["nnz", "num_k_heads_x_head_size"], dtype_from="key"),
    },
    tags=["status:verified"],
)

apply_rope_with_cos_sin_cache_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_cos_sin_cache_inplace",
    description="In-place RoPE with precomputed cos/sin cache.",
    axes=_COSSIN_AXES,
    inputs=_COSSIN_INPUTS,
    outputs={
        "query": Tensor(
            ["nnz", "num_q_heads_x_head_size"],
            dtype_from="query",
            description="Updated query (in-place).",
        ),
        "key": Tensor(
            ["nnz", "num_k_heads_x_head_size"],
            dtype_from="key",
            description="Updated key (in-place).",
        ),
    },
    tags=["status:verified"],
)
