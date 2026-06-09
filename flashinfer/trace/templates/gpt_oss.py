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

"""TraceTemplates for GPT-OSS-specific operations."""

from ..template import Const, Tensor, TraceTemplate, Var


gpt_oss_reshape_cache_fp8_trace = TraceTemplate(
    op_type="gpt_oss_reshape_cache_fp8",
    name_prefix="gpt_oss_reshape_cache_fp8",
    description=(
        "GPT-OSS FP8 paged-KV cache update. Converts BF16 key/value rows to "
        "FP8 E4M3 and scatters them into a [num_blocks, 16, num_heads, 64] cache."
    ),
    axes={
        "num_tokens": Var(description="Number of key/value rows to update."),
        "num_heads": Const(abbrev="h", description="Local KV heads."),
        "head_dim": Const(abbrev="d", description="Fixed GPT-OSS head dimension."),
        "num_blocks": Var(description="Number of paged-cache blocks."),
        "block_size": Const(abbrev="ps", description="Paged-cache block size."),
        "scale_elems": Var(description="Number of elements in each scale tensor."),
    },
    inputs={
        "key": Tensor(
            ["num_tokens", "num_heads", "head_dim"],
            dtype="bfloat16",
            description="BF16 key rows.",
        ),
        "value": Tensor(
            ["num_tokens", "num_heads", "head_dim"],
            dtype="bfloat16",
            description="BF16 value rows.",
        ),
        "key_cache": Tensor(
            ["num_blocks", "block_size", "num_heads", "head_dim"],
            dtype="uint8",
            description="FP8 E4M3 or uint8-backed key cache updated in-place.",
        ),
        "value_cache": Tensor(
            ["num_blocks", "block_size", "num_heads", "head_dim"],
            dtype="uint8",
            description="FP8 E4M3 or uint8-backed value cache updated in-place.",
        ),
        "slot_mapping": Tensor(
            ["num_tokens"],
            dtype="int64",
            description="Token-to-cache-slot mapping. Negative slots are skipped.",
        ),
        "k_scale": Tensor(
            ["scale_elems"],
            dtype="float32",
            description="Scalar FP32 key scale.",
        ),
        "v_scale": Tensor(
            ["scale_elems"],
            dtype="float32",
            description="Scalar FP32 value scale.",
        ),
    },
    outputs={
        "key_cache": Tensor(
            ["num_blocks", "block_size", "num_heads", "head_dim"],
            param="key_cache",
            dtype_from="key_cache",
            description="Updated key cache.",
        ),
        "value_cache": Tensor(
            ["num_blocks", "block_size", "num_heads", "head_dim"],
            param="value_cache",
            dtype_from="value_cache",
            description="Updated value cache.",
        ),
    },
    constraints=["head_dim == 64", "block_size == 16", "scale_elems == 1"],
    tags=["status:verified", "quantize:fp8", "model:gpt-oss"],
)
