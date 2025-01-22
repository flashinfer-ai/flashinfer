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


def jit_decode_attention_func_args(
    q_dtypes,
    kv_dtypes,
    head_dims,
    pos_encoding_modes,
    use_sliding_window_options,
    use_logits_soft_cap_options,
):
    load_module_func_args = []

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
        load_module_func_args.append(
            (
                flashinfer.decode.get_single_decode_module,
                (
                    q_dtype,
                    kv_dtype,
                    q_dtype,
                    head_dim,
                    pos_encoding_mode,
                    use_sliding_window,
                    use_logits_soft_cap,
                ),
            )
        )
        load_module_func_args.append(
            (
                flashinfer.decode.get_batch_decode_module,
                (
                    q_dtype,
                    kv_dtype,
                    q_dtype,
                    torch.int32,
                    head_dim,
                    pos_encoding_mode,
                    use_sliding_window,
                    use_logits_soft_cap,
                ),
            )
        )

    return load_module_func_args


def jit_prefill_attention_func_args(
    q_dtypes,
    kv_dtypes,
    head_dims,
    pos_encoding_modes,
    use_sliding_window_options,
    use_logits_soft_cap_options,
    use_fp16_qk_reduction_options,
):
    load_module_func_args = []

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
        load_module_func_args.append(
            (
                flashinfer.prefill.gen_single_prefill_module,
                (
                    "fa2",
                    q_dtype,
                    kv_dtype,
                    q_dtype,
                    head_dim,
                    pos_encoding_mode,
                    use_sliding_window,
                    use_logits_soft_cap,
                    use_fp16_qk_reduction,
                ),
            )
        )
        load_module_func_args.append(
            (
                flashinfer.prefill.gen_batch_prefill_module,
                (
                    "fa2",
                    q_dtype,
                    kv_dtype,
                    q_dtype,
                    torch.int32,
                    head_dim,
                    pos_encoding_mode,
                    use_sliding_window,
                    use_logits_soft_cap,
                    use_fp16_qk_reduction,
                ),
            )
        )

    load_module_func_args.append(
        (
            flashinfer.quantization.get_quantization_module,
            [],
        )  # required for attention with custom mask
    )

    return load_module_func_args
