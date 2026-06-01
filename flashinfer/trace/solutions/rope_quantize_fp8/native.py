# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlashInfer flashinfer solution for rope_quantize_fp8."""

from flashinfer.rope import rope_quantize_fp8 as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "rope_quantize_fp8"
api = "flashinfer.rope.rope_quantize_fp8"
backend = "flashinfer"
inputs = (
    "q_rope",
    "k_rope",
    "q_nope",
    "k_nope",
    "cos_sin_cache",
    "pos_ids",
    "is_neox",
    "quant_scale_q",
    "quant_scale_kv",
)
outputs = ("q_rope_out", "k_rope_out", "q_nope_out", "k_nope_out")
api_kwargs = {
    "q_rope": "q_rope",
    "k_rope": "k_rope",
    "q_nope": "q_nope",
    "k_nope": "k_nope",
    "cos_sin_cache": "cos_sin_cache",
    "pos_ids": "pos_ids",
    "is_neox": "is_neox",
    "quant_scale_q": "quant_scale_q",
    "quant_scale_kv": "quant_scale_kv",
}


def run(
    q_rope,
    k_rope,
    q_nope,
    k_nope,
    cos_sin_cache,
    pos_ids,
    is_neox,
    quant_scale_q,
    quant_scale_kv,
):
    with solution_autotune(
        definition,
        backend,
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        cos_sin_cache,
        pos_ids,
        is_neox,
        quant_scale_q,
        quant_scale_kv,
    ):
        result = _api(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=is_neox,
            quant_scale_q=quant_scale_q,
            quant_scale_kv=quant_scale_kv,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "rope_quantize_fp8" + " returned None without mutating declared outputs"
        )
