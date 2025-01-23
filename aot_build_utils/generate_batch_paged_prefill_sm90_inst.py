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

import re
import sys
from pathlib import Path

from .literal_map import dtype_literal, idtype_literal, mask_mode_literal


def get_cu_file_str(
    head_dim,
    pos_encoding_mode,
    use_fp16_qk_reduction,
    mask_mode,
    dtype_q,
    dtype_kv,
    dtype_out,
    idtype,
):
    pos_encoding_mode = None
    use_fp16_qk_reduction = None

    def get_insts(attention_variant):
        return """
template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{head_dim},
     {mask_mode},
     /*USE_SLIDING_WINDOW=*/true,
     /*SAME_SCHEDULE_FOR_ALL_HEADS=*/true,
     {attention_variant},
     Params>
    (Params& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{head_dim},
     {mask_mode},
     /*USE_SLIDING_WINDOW=*/true,
     /*SAME_SCHEDULE_FOR_ALL_HEADS=*/false,
     {attention_variant},
     Params>
    (Params& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{head_dim},
     {mask_mode},
     /*USE_SLIDING_WINDOW=*/false,
     /*SAME_SCHEDULE_FOR_ALL_HEADS=*/true,
     {attention_variant},
     Params>
    (Params& params, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched
    <{head_dim},
     {mask_mode},
     /*USE_SLIDING_WINDOW=*/false,
     /*SAME_SCHEDULE_FOR_ALL_HEADS=*/false,
     {attention_variant},
     Params>
    (Params& params, cudaStream_t stream);
    """.format(
            head_dim=head_dim,
            mask_mode=mask_mode_literal[int(mask_mode)],
            attention_variant=attention_variant,
        )

    dtype_q = dtype_literal[dtype_q]
    dtype_kv = dtype_literal[dtype_kv]
    dtype_out = dtype_literal[dtype_out]
    idtype = idtype_literal[idtype]

    content = f""" // batch_paged_prefill_sm90 template inst
#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>


namespace flashinfer {{

using DTypeQ = cutlass_dtype_t<{dtype_q}>;
using DTypeKV = cutlass_dtype_t<{dtype_kv}>;
using DTypeO = cutlass_dtype_t<{dtype_out}>;

using Params = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, {idtype}>;

{get_insts("LogitsSoftCap")}

{get_insts("StandardAttention")}

}}"""
    return content


if __name__ == "__main__":
    pattern = (
        r"batch_paged_prefill_head_([0-9]+)_posenc_([0-9]+)_"
        r"fp16qkred_([a-z]+)_mask_([0-9]+)_dtypeq_([a-z0-9]+)_dtypekv_([a-z0-9]+)_"
        r"dtypeout_([a-z0-9]+)_idtype_([a-z0-9]+)_sm90\.cu"
    )
    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)
    path.write_text(get_cu_file_str(*match.groups()))
