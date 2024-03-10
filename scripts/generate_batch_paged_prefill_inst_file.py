"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pathlib
import sys
import re

root = pathlib.Path(__file__).parent.parent

kv_layout_literal = {
    0: "QKVLayout::kNHD",
    1: "QKVLayout::kHND",
}

pos_encoding_mode_literal = {
    0: "PosEncodingMode::kNone",
    1: "PosEncodingMode::kRoPELlama",
    2: "PosEncodingMode::kALiBi",
}

dtype_literal = {
    "f16": "half",
    "bf16": "nv_bfloat16",
    "e4m3": "__nv_fp8_e4m3",
    "e5m2": "__nv_fp8_e5m2",
}

idtype_literal = {
    "i32": "int32_t",
    "u32": "uint32_t",
    "i64": "int64_t",
    "u64": "uint64_t",
}

pattern = (
    r"batch_paged_prefill_group_([0-9]+)_head_([0-9]+)_layout_([0-9]+)_posenc_([0-9]+)_"
    r"fp16qkred_([a-z]+)_causal_([a-z]+)_dtypein_([a-z0-9]+)_dtypeout_([a-z0-9]+)_idtype_([a-z0-9]+)\.cu"
)
compiled_pattern = re.compile(pattern)
fname = sys.argv[1]
match = compiled_pattern.match(fname)
(
    group_size,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    allow_fp16_qk_reduction,
    causal,
    dtype_in,
    dtype_out,
    idtype,
) = match.groups()

content = """#include <flashinfer/attention_impl.cuh>

namespace flashinfer {{

constexpr PageStorage page_storage = PageStorage::kIndices;

template cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched<page_storage, {kv_layout}, {group_size}, {head_dim}, {pos_encoding_mode}, {allow_fp16_qk_reduction}, {causal}, {dtype_in}, {dtype_out}, {idtype}>(
  BatchPrefillHandler* handler, {dtype_in}* q, {idtype}* qo_indptr, {idtype}* q_offset,
  paged_kv_t<page_storage, {kv_layout}, {dtype_in}, {idtype}> paged_kv,
  {dtype_out}* o, float* lse,
  float sm_scale, float rope_scale,
  float rope_theta, cudaStream_t stream);

}}
""".format(
    kv_layout=kv_layout_literal[int(kv_layout)],
    group_size=group_size,
    head_dim=head_dim,
    pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
    allow_fp16_qk_reduction=allow_fp16_qk_reduction,
    causal=causal,
    dtype_in=dtype_literal[dtype_in],
    dtype_out=dtype_literal[dtype_out],
    idtype=idtype_literal[idtype],
)

path = root / "src" / "generated" / fname
with open(path, "w") as f:
    f.write(content)
