# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Trace contracts for packed QSA forward."""

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


qsa_forward_trace = TraceTemplate(
    op_type="qsa_forward",
    description="Packed BF16 QSA forward with four-token block selection.",
    axes={
        "num_tokens": Var(),
        "num_blocks": Var(),
        "num_q_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
    },
    inputs={
        "q": Tensor(["num_tokens", "num_q_heads", "head_dim"], dtype="bfloat16"),
        "k": Tensor(["num_tokens", "num_kv_heads", "head_dim"], dtype="bfloat16"),
        "v": Tensor(["num_tokens", "num_kv_heads", "head_dim"], dtype="bfloat16"),
        "sel_blocks": Tensor(["num_tokens", "num_blocks"], dtype="uint8"),
        "lo": Tensor(["num_tokens"], dtype="int32"),
        "hi": Tensor(["num_tokens"], dtype="int32"),
        "blk_base": Tensor(["num_tokens"], dtype="int32"),
        "tok_base": Tensor(["num_tokens"], dtype="int32"),
        "block_first": Tensor(["num_blocks"], dtype="int32"),
        "block_last": Tensor(["num_blocks"], dtype="int32"),
        "scale": Scalar("float32"),
        "block_size": Scalar("int32"),
    },
    outputs={
        "out": Tensor(["num_tokens", "num_q_heads", "head_dim"], dtype="bfloat16"),
        "out_fp32": Tensor(["num_tokens", "num_q_heads", "head_dim"], dtype="float32"),
        "lse": Tensor(["num_q_heads", "num_tokens"], dtype="float32"),
    },
)

qsa_indexed_forward_trace = TraceTemplate(
    op_type="qsa_indexed_forward",
    description="Packed BF16 QSA forward with caller-selected causal token IDs.",
    axes={
        "num_tokens": Var(),
        "selection_width": Var(),
        "num_q_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
    },
    inputs={
        "q": Tensor(["num_tokens", "num_q_heads", "head_dim"], dtype="bfloat16"),
        "k": Tensor(["num_tokens", "num_kv_heads", "head_dim"], dtype="bfloat16"),
        "v": Tensor(["num_tokens", "num_kv_heads", "head_dim"], dtype="bfloat16"),
        "indices": Tensor(["num_tokens", "selection_width"], dtype="int32"),
        "scale": Scalar("float32"),
    },
    outputs={
        "out": Tensor(["num_tokens", "num_q_heads", "head_dim"], dtype="bfloat16"),
        "out_fp32": Tensor(["num_tokens", "num_q_heads", "head_dim"], dtype="float32"),
        "lse": Tensor(["num_q_heads", "num_tokens"], dtype="float32"),
    },
)
