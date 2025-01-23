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

from .literal_map import (
    dtype_literal,
    idtype_literal,
    mask_mode_literal,
    pos_encoding_mode_literal,
)


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
    cta_tile_q_choice = [128, 64, 16]

    def get_insts(attention_variant, dtype_out):
        return "\n".join(
            [
                """template cudaError_t BatchPrefillWithPagedKVCacheDispatched<{cta_tile_q}, {head_dim}, {pos_encoding_mode}, {use_fp16_qk_reduction}, {mask_mode}, {attention_variant}, Params>(
    Params params,
    {dtype_out}* tmp_v,
    float* tmp_s, cudaStream_t stream);
    """.format(
                    cta_tile_q=cta_tile_q,
                    head_dim=head_dim,
                    pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
                    use_fp16_qk_reduction=use_fp16_qk_reduction,
                    mask_mode=mask_mode_literal[int(mask_mode)],
                    attention_variant=attention_variant,
                    dtype_out=dtype_out,
                )
                for cta_tile_q in cta_tile_q_choice
            ]
        )

    use_custom_mask = "true" if int(mask_mode) == 2 else "false"
    dtype_q = dtype_literal[dtype_q]
    dtype_kv = dtype_literal[dtype_kv]
    dtype_out = dtype_literal[dtype_out]
    idtype = idtype_literal[idtype]

    content = f"""#include <flashinfer/attention_impl.cuh>

namespace flashinfer {{

using Params = BatchPrefillPagedParams<{dtype_q}, {dtype_kv}, {dtype_out}, {idtype}>;

using AttentionVariant1 = DefaultAttention<{use_custom_mask}, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>;

{get_insts("AttentionVariant1", dtype_out)}

using AttentionVariant2 = DefaultAttention<{use_custom_mask}, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>;

{get_insts("AttentionVariant2", dtype_out)}

using AttentionVariant3 = DefaultAttention<{use_custom_mask}, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>;

{get_insts("AttentionVariant3", dtype_out)}

using AttentionVariant4 = DefaultAttention<{use_custom_mask}, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>;

{get_insts("AttentionVariant4", dtype_out)}

}}"""
    return content


if __name__ == "__main__":
    pattern = (
        r"batch_paged_prefill_head_([0-9]+)_posenc_([0-9]+)_"
        r"fp16qkred_([a-z]+)_mask_([0-9]+)_dtypeq_([a-z0-9]+)_dtypekv_([a-z0-9]+)_dtypeout_([a-z0-9]+)_idtype_([a-z0-9]+)\.cu"
    )
    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)

    with open(path, "w") as f:
        f.write(get_cu_file_str(*match.groups()))
