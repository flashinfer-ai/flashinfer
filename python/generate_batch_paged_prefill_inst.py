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
    mask_mode_literal,
    kv_layout_literal,
    pos_encoding_mode_literal,
    dtype_literal,
    idtype_literal,
    logits_hook_literal,
)
from pathlib import Path


def get_cu_file_str(
    head_dim,
    logits_hook,
    kv_layout,
    pos_encoding_mode,
    allow_fp16_qk_reduction,
    mask_mode,
    dtype_in,
    dtype_out,
    idtype,
):
    num_frags_x_choices = [1, 2]
    insts = "\n".join(
        [
            """template cudaError_t BatchPrefillWithPagedKVCacheDispatched<page_storage, {num_frags_x}, {head_dim}, {logits_hook}, {kv_layout}, {pos_encoding_mode}, {allow_fp16_qk_reduction}, {mask_mode}, {dtype_in}, {dtype_out}, {idtype}>(
    {dtype_in}* q, {idtype}* request_indices, {idtype}* tile_indices,
    {idtype}* qo_indptr, {idtype}* q_offset,
    paged_kv_t<page_storage, {kv_layout}, {dtype_in}, {idtype}> paged_kv,
    uint8_t* custom_mask, {idtype}* qk_indptr,
    {dtype_out}* o, float* tmp, float* lse,
    uint32_t num_tiles, uint32_t num_qo_heads,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);
    """.format(
                logits_hook=logits_hook_literal[int(logits_hook)],
                kv_layout=kv_layout_literal[int(kv_layout)],
                num_frags_x=num_frags_x,
                head_dim=head_dim,
                pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
                allow_fp16_qk_reduction=allow_fp16_qk_reduction,
                mask_mode=mask_mode_literal[int(mask_mode)],
                dtype_in=dtype_literal[dtype_in],
                dtype_out=dtype_literal[dtype_out],
                idtype=idtype_literal[idtype],
            )
            for num_frags_x in num_frags_x_choices
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
        r"batch_paged_prefill_head_([0-9]+)_logitshook_([0-9]+)_layout_([0-9]+)_posenc_([0-9]+)_"
        r"fp16qkred_([a-z]+)_mask_([0-9]+)_dtypein_([a-z0-9]+)_dtypeout_([a-z0-9]+)_idtype_([a-z0-9]+)\.cu"
    )
    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)

    with open(path, "w") as f:
        f.write(get_cu_file_str(*match.groups()))
