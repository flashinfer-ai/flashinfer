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
    dtype_literal_hip,
    idtype_literal,
    pos_encoding_mode_literal,
)


def get_cu_file_str(
    head_dim_qk,
    head_dim_vo,
    pos_encoding_mode,
    dtype_q,
    dtype_kv,
    dtype_out,
    idtype,
    generate_hip,
):
    if generate_hip:
        content = """#include <flashinfer/attention/generic/attention_impl.cuh>

    namespace flashinfer {{

    using Params = BatchDecodeParams<{dtype_q}, {dtype_kv}, {dtype_out}, {idtype}>;

    template hipError_t BatchDecodeWithPagedKVCacheDispatched<{head_dim_qk}, {pos_encoding_mode}, DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>, Params>(
        Params params,
        {dtype_out}* tmp_v, float* tmp_s,
        hipStream_t stream);

    template hipError_t BatchDecodeWithPagedKVCacheDispatched<{head_dim_qk}, {pos_encoding_mode}, DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>, Params>(
        Params params,
        {dtype_out}* tmp_v, float* tmp_s,
        hipStream_t stream);

    template hipError_t BatchDecodeWithPagedKVCacheDispatched<{head_dim_qk}, {pos_encoding_mode}, DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>, Params>(
        Params params,
        {dtype_out}* tmp_v, float* tmp_s,
        hipStream_t stream);

    template hipError_t BatchDecodeWithPagedKVCacheDispatched<{head_dim_qk}, {pos_encoding_mode}, DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>, Params>(
        Params params,
        {dtype_out}* tmp_v, float* tmp_s,
        hipStream_t stream);

    using ParamsMlaT = BatchDecodeParamsMLA<{dtype_q}, {dtype_kv}, {dtype_out}, {idtype}>;

    template hipError_t BatchDecodeWithPagedKVCacheDispatchedMLA<{head_dim}, {head_dim_kpe}, DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>, ParamsMlaT>(
        ParamsMlaT params,
        {dtype_out}* tmp_v, float* tmp_s,
        hipStream_t stream);

    }}
        """.format(
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
            dtype_q=dtype_literal_hip[dtype_q],
            dtype_kv=dtype_literal_hip[dtype_kv],
            dtype_out=dtype_literal_hip[dtype_out],
            idtype=idtype_literal[idtype],
            head_dim=head_dim_vo,  # NOTE(Zihao): for MLA instantiation, we should move them to a standalone file
            head_dim_kpe=head_dim_vo // 8,
        )
        return content

    else:
        content = """#include <flashinfer/attention_impl.cuh>

    namespace flashinfer {{

    using Params = BatchDecodeParams<{dtype_q}, {dtype_kv}, {dtype_out}, {idtype}>;

    template cudaError_t BatchDecodeWithPagedKVCacheDispatched<{head_dim_qk}, {pos_encoding_mode}, DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>, Params>(
        Params params,
        {dtype_out}* tmp_v, float* tmp_s,
        cudaStream_t stream);

    template cudaError_t BatchDecodeWithPagedKVCacheDispatched<{head_dim_qk}, {pos_encoding_mode}, DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>, Params>(
        Params params,
        {dtype_out}* tmp_v, float* tmp_s,
        cudaStream_t stream);

    template cudaError_t BatchDecodeWithPagedKVCacheDispatched<{head_dim_qk}, {pos_encoding_mode}, DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>, Params>(
        Params params,
        {dtype_out}* tmp_v, float* tmp_s,
        cudaStream_t stream);

    template cudaError_t BatchDecodeWithPagedKVCacheDispatched<{head_dim_qk}, {pos_encoding_mode}, DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>, Params>(
        Params params,
        {dtype_out}* tmp_v, float* tmp_s,
        cudaStream_t stream);

    using ParamsMlaT = BatchDecodeParamsMLA<{dtype_q}, {dtype_kv}, {dtype_out}, {idtype}>;

    template cudaError_t BatchDecodeWithPagedKVCacheDispatchedMLA<{head_dim}, {head_dim_kpe}, DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>, ParamsMlaT>(
        ParamsMlaT params,
        {dtype_out}* tmp_v, float* tmp_s,
        cudaStream_t stream);

    }}
        """.format(
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
            dtype_q=dtype_literal[dtype_q],
            dtype_kv=dtype_literal[dtype_kv],
            dtype_out=dtype_literal[dtype_out],
            idtype=idtype_literal[idtype],
            head_dim=head_dim_vo,  # NOTE(Zihao): for MLA instantiation, we should move them to a standalone file
            head_dim_kpe=head_dim_vo // 8,
        )
        return content


if __name__ == "__main__":
    pattern = (
        r"batch_paged_decode_head_qk_([0-9]+)_head_vo_([0-9]+)_posenc_([0-9]+)_"
        r"dtypeq_([a-z0-9]+)_dtypekv_([a-z0-9]+)_dtypeout_([a-z0-9]+)_idtype_([a-z0-9]+)\.cu"
    )

    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)
    with open(path, "w") as f:
        f.write(get_cu_file_str(*match.groups()))
