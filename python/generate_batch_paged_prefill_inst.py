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

import sys
import re
import itertools
from literal_map import (
    kv_layout_literal,
    pos_encoding_mode_literal,
    dtype_literal,
    idtype_literal,
)
from pathlib import Path


def get_cu_file_str(
    group_size,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    allow_fp16_qk_reduction,
    causal,
    dtype_in,
    dtype_out,
    idtype,
    page_size_choices=[1, 8, 16, 32],
):
    num_frags_x_choices = [1, 2]
    insts = "\n".join(
        [
            """template cudaError_t BatchPrefillWithPagedKVCacheDispatched<page_storage, {kv_layout}, {num_frags_x}, {page_size}, {group_size}, {head_dim}, {pos_encoding_mode}, {allow_fp16_qk_reduction}, {causal}, {dtype_in}, {dtype_out}, {idtype}>(
    {dtype_in}* q, {idtype}* request_indices, {idtype}* tile_indices,
    {idtype}* qo_indptr, {idtype}* q_offset,
    paged_kv_t<page_storage, {kv_layout}, {dtype_in}, {idtype}> paged_kv,
    {dtype_out}* o, float* tmp, float* lse,
    uint32_t num_qo_tiles,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);
    """.format(
                kv_layout=kv_layout_literal[int(kv_layout)],
                num_frags_x=num_frags_x,
                page_size=page_size,
                group_size=group_size,
                head_dim=head_dim,
                pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
                allow_fp16_qk_reduction=allow_fp16_qk_reduction,
                causal=causal,
                dtype_in=dtype_literal[dtype_in],
                dtype_out=dtype_literal[dtype_out],
                idtype=idtype_literal[idtype],
            )
            for num_frags_x, page_size in itertools.product(
                num_frags_x_choices,
                page_size_choices,
            )
        ]
    )

    content = f"""#include <flashinfer/attention_impl.cuh>

namespace flashinfer {{

constexpr PageStorage page_storage = PageStorage::kIndices;

{insts}

}}"""
    return content


if __name__ == "__main__":
    pattern = (
        r"batch_paged_prefill_group_([0-9]+)_head_([0-9]+)_layout_([0-9]+)_posenc_([0-9]+)_"
        r"fp16qkred_([a-z]+)_causal_([a-z]+)_dtypein_([a-z0-9]+)_dtypeout_([a-z0-9]+)_idtype_([a-z0-9]+)\.cu"
    )
    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)

    with open(path, "w") as f:
        f.write(get_cu_file_str(*match.groups()))
