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

"""Tensor signature for the planned VDN window softmax operator."""

from ..template import Const, Tensor, TraceTemplate, Var


vdn_window_attention_run_trace = TraceTemplate(
    op_type="vdn_window_attention",
    name_prefix="vdn_window_attention",
    description=(
        "BF16 VDN window/global/anchor softmax on SM120. This tensor signature "
        "does not encode the host-side window bounds, anchor mode or scale; "
        "replay must use the original VDNWindowAttentionWrapper.plan() arguments. "
        "It covers only the softmax branch, not VDN's linear recurrence."
    ),
    axes={
        "seq_len": Var(),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
    },
    inputs={
        "query": Tensor(["seq_len", "num_heads", "head_dim"]),
        "key": Tensor(["seq_len", "num_heads", "head_dim"]),
        "value": Tensor(["seq_len", "num_heads", "head_dim"]),
    },
    outputs={
        "output": Tensor(
            ["seq_len", "num_heads", "head_dim"], dtype_from="query", param="out"
        ),
    },
    constraints=["head_dim == 128"],
    tags=["sparse:window", "precision:bf16", "arch:sm120"],
)
