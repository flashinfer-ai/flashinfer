"""
Copyright (c) 2023 by FlashInfer team.

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

import itertools

import torch

import flashinfer
from flashinfer.jit import JitSpec
from flashinfer.utils import is_fa3_backend_supported, is_sm90a_supported


def gen_decode_attention_modules(
    q_dtypes,
    kv_dtypes,
    head_dims,
    pos_encoding_modes,
    use_sliding_window_options,
    use_logits_soft_cap_options,
) -> list[JitSpec]:
    jit_specs: list[JitSpec] = []

    for (
        q_dtype,
        kv_dtype,
        head_dim,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
    ) in itertools.product(
        q_dtypes,
        kv_dtypes,
        head_dims,
        pos_encoding_modes,
        use_sliding_window_options,
        use_logits_soft_cap_options,
    ):
        jit_specs.append(
            flashinfer.decode.gen_single_decode_module(
                q_dtype,
                kv_dtype,
                q_dtype,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                pos_encoding_mode,
                use_sliding_window,
                use_logits_soft_cap,
            )
        )
        jit_specs.append(
            flashinfer.decode.gen_batch_decode_module(
                q_dtype,
                kv_dtype,
                q_dtype,
                torch.int32,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                pos_encoding_mode,
                use_sliding_window,
                use_logits_soft_cap,
            )
        )

    return jit_specs


def gen_prefill_attention_modules(
    q_dtypes,
    kv_dtypes,
    head_dims,
    pos_encoding_modes,
    use_sliding_window_options,
    use_logits_soft_cap_options,
    use_fp16_qk_reduction_options,
) -> list[JitSpec]:
    jit_specs: list[JitSpec] = []

    for (
        q_dtype,
        kv_dtype,
        head_dim,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
    ) in itertools.product(
        q_dtypes,
        kv_dtypes,
        head_dims,
        pos_encoding_modes,
        use_sliding_window_options,
        use_logits_soft_cap_options,
        use_fp16_qk_reduction_options,
    ):
        if is_sm90a_supported(torch.device("cuda")) and is_fa3_backend_supported(
            pos_encoding_mode,
            use_fp16_qk_reduction,
            use_custom_mask=False,
            dtype_q=q_dtype,
            dtype_kv=kv_dtype,
        ):
            jit_specs.append(
                flashinfer.prefill.gen_single_prefill_module(
                    "fa3",
                    q_dtype,
                    kv_dtype,
                    q_dtype,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    pos_encoding_mode,
                    use_sliding_window,
                    use_logits_soft_cap,
                    use_fp16_qk_reduction,
                )
            )

            jit_specs.append(
                flashinfer.prefill.gen_batch_prefill_module(
                    "fa3",
                    q_dtype,
                    kv_dtype,
                    q_dtype,
                    torch.int32,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    pos_encoding_mode,
                    use_sliding_window,
                    use_logits_soft_cap,
                    use_fp16_qk_reduction,
                )
            )
        jit_specs.append(
            flashinfer.prefill.gen_single_prefill_module(
                "fa2",
                q_dtype,
                kv_dtype,
                q_dtype,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                pos_encoding_mode,
                use_sliding_window,
                use_logits_soft_cap,
                use_fp16_qk_reduction,
            )
        )
        jit_specs.append(
            flashinfer.prefill.gen_batch_prefill_module(
                "fa2",
                q_dtype,
                kv_dtype,
                q_dtype,
                torch.int32,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                pos_encoding_mode,
                use_sliding_window,
                use_logits_soft_cap,
                use_fp16_qk_reduction,
            )
        )

    # required for attention with custom mask
    jit_specs.append(flashinfer.quantization.gen_quantization_module())

    jit_specs.append(flashinfer.page.gen_page_module())

    return jit_specs
