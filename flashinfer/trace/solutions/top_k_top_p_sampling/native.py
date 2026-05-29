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

"""FlashInfer flashinfer solution for top_k_top_p_sampling."""

from flashinfer.sampling import top_k_top_p_sampling_from_probs as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "top_k_top_p_sampling"
api = "flashinfer.sampling.top_k_top_p_sampling_from_probs"
backend = "flashinfer"
inputs = ("probs", "top_k", "top_p")
outputs = ("samples",)
api_kwargs = {"probs": "probs", "top_k": "top_k", "top_p": "top_p"}
constants = {"vocab_size": 128256}


def run(probs, top_k, top_p):
    with solution_autotune(
        definition,
        backend,
        probs,
        top_k,
        top_p,
    ):
        result = _api(
            probs=probs,
            top_k=top_k,
            top_p=top_p,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "top_k_top_p_sampling" + " returned None without mutating declared outputs"
        )
