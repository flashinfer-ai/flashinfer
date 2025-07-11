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
    head_dim_qk,
    head_dim_vo,
    pos_encoding_mode,
    use_fp16_qk_reduction,
    mask_mode_p,
    mask_mode_d,
    dtype_q,
    dtype_kv,
    dtype_out,
    idtype,
):
    cta_tile_q_choice = [128, 64, 16]

    def get_insts(attention_variant_p, attention_variant_d, dtype_out):
        return "\n".join(
            [
                """template cudaError_t PODWithKVCacheTensorDispatched<{head_dim_qk}, {head_dim_vo}, {pos_encoding_mode}, {use_fp16_qk_reduction}, {mask_mode_p}, {cta_tile_q_p}, {cta_tile_q_d}, {mask_mode_d}, {attention_variant_p}, {attention_variant_d}, PrefillParams, DecodeParams>(
    PrefillParams prefill_params, DecodeParams decode_params,
    {dtype_out}* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream);
    """.format(
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
                    use_fp16_qk_reduction=use_fp16_qk_reduction,
                    mask_mode_p=mask_mode_literal[int(mask_mode_p)],
                    cta_tile_q_p=cta_tile_q_p,
                    cta_tile_q_d=cta_tile_q_d,
                    mask_mode_d=mask_mode_literal[int(mask_mode_d)],
                    attention_variant_p=attention_variant_p,
                    attention_variant_d=attention_variant_d,
                    dtype_out=dtype_out,
                )
                for cta_tile_q_p in cta_tile_q_choice
                for cta_tile_q_d in cta_tile_q_choice
            ]
        )

    use_custom_mask_p = "true" if int(mask_mode_p) == 2 else "false"
    use_custom_mask_d = "true" if int(mask_mode_d) == 2 else "false"
    dtype_q = dtype_literal[dtype_q]
    dtype_kv = dtype_literal[dtype_kv]
    dtype_out = dtype_literal[dtype_out]
    idtype = idtype_literal[idtype]

    content = f"""#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/pod.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/page.cuh>

#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

using PrefillParams = BatchPrefillPagedParams<{dtype_q}, {dtype_kv}, {dtype_out}>;
using DecodeParams = BatchPrefillPagedParams<{dtype_q}, {dtype_kv}, {dtype_out}, {idtype}>;

constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;

using AttentionVariant1_P = DefaultAttention<{use_custom_mask_p}, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>;
using AttentionVariant1_D = DefaultAttention<{use_custom_mask_d}, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>;

{get_insts("AttentionVariant1_P", "AttentionVariant1_D", dtype_out)}

using AttentionVariant2_P = DefaultAttention<{use_custom_mask_p}, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>;
using AttentionVariant2_D = DefaultAttention<{use_custom_mask_d}, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>;

{get_insts("AttentionVariant2_P", "AttentionVariant2_D", dtype_out)}

using AttentionVariant3_P = DefaultAttention<{use_custom_mask_p}, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>;
using AttentionVariant3_D = DefaultAttention<{use_custom_mask_d}, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>;

{get_insts("AttentionVariant3_P", "AttentionVariant3_D", dtype_out)}

using AttentionVariant4_P = DefaultAttention<{use_custom_mask_p}, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>;
using AttentionVariant4_D = DefaultAttention<{use_custom_mask_d}, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>;

{get_insts("AttentionVariant4_P", "AttentionVariant4_D", dtype_out)}

}}"""
    return content


if __name__ == "__main__":
    pattern = (
        r"pod_head_qk_([0-9]+)_head_vo_([0-9]+)_posenc_([0-9]+)_"
        r"fp16qkred_([a-z]+)_maskp_([0-9]+)_maskd_([0-9]+)_"
        r"dtypeq_([a-z0-9]+)_dtypekv_([a-z0-9]+)_dtypeout_([a-z0-9]+)_idtype_([a-z0-9]+)\.cu"
    )
    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)

    with open(path, "w") as f:
        f.write(get_cu_file_str(*match.groups()))
