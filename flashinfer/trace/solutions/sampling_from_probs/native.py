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

"""FlashInfer flashinfer solution for sampling_from_probs."""

from flashinfer.sampling import sampling_from_probs as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "sampling_from_probs"
api = "flashinfer.sampling.sampling_from_probs"
backend = "flashinfer"
inputs = ("probs", "indices")
outputs = ("samples",)
api_kwargs = {"probs": "probs", "indices": "indices"}


def run(probs, indices):
    with solution_autotune(
        definition,
        backend,
        probs,
        indices,
    ):
        result = _api(
            probs=probs,
            indices=indices,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "sampling_from_probs" + " returned None without mutating declared outputs"
        )
