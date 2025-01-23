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

import torch

import flashinfer
from flashinfer.jit import parallel_load_modules
from flashinfer.utils import PosEncodingMode


def test_warmpup_llama():
    parallel_load_modules(
        [
            (flashinfer.activation.get_act_and_mul_module, ["silu"]),
            (flashinfer.norm.get_norm_module, []),
            (flashinfer.sampling.get_sampling_module, []),
            (flashinfer.quantization.get_quantization_module, []),
            (flashinfer.page.get_page_module, []),
            (
                flashinfer.decode.get_batch_decode_module,
                [
                    torch.float16,
                    torch.float16,
                    torch.float16,
                    torch.int32,
                    128,
                    PosEncodingMode.NONE.value,
                    False,  # use_sliding_window
                    False,  # use_logits_soft_cap
                ],
            ),
            (
                flashinfer.prefill.gen_batch_prefill_module,
                [
                    torch.float16,
                    torch.float16,
                    torch.float16,
                    torch.int32,
                    128,
                    PosEncodingMode.NONE.value,
                    False,  # use_sliding_window
                    False,  # use_logits_soft_cap
                    False,  # use_fp16_qk_reduction
                ],
            ),
        ]
    )


def test_warmpup_llama_sm90():
    parallel_load_modules(
        [
            (flashinfer.activation.get_act_and_mul_module, ["silu"]),
            (flashinfer.norm.get_norm_module, []),
            (flashinfer.sampling.get_sampling_module, []),
            (flashinfer.quantization.get_quantization_module, []),
            (flashinfer.page.get_page_module, []),
            (
                flashinfer.decode.get_batch_decode_module,
                [
                    torch.float16,
                    torch.float16,
                    torch.float16,
                    torch.int32,
                    128,
                    PosEncodingMode.NONE.value,
                    False,  # use_sliding_window
                    False,  # use_logits_soft_cap
                ],
            ),
            (
                flashinfer.prefill.gen_batch_prefill_module,
                [
                    torch.float16,
                    torch.float16,
                    torch.float16,
                    torch.int32,
                    128,
                    PosEncodingMode.NONE.value,
                    False,  # use_sliding_window
                    False,  # use_logits_soft_cap
                    False,  # use_fp16_qk_reduction
                ],
            ),
            (
                flashinfer.prefill.gen_batch_prefill_sm90_module,
                [
                    torch.float16,
                    torch.float16,
                    torch.float16,
                    torch.int32,
                    128,
                    PosEncodingMode.NONE.value,
                    False,  # use_sliding_window
                    False,  # use_logits_soft_cap
                    False,  # use_fp16_qk_reduction
                ],
            ),
        ]
    )
