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

"""FlashInfer flashinfer solution for sampling_from_logits."""

from flashinfer.sampling import sampling_from_logits as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "sampling_from_logits"
api = "flashinfer.sampling.sampling_from_logits"
backend = "flashinfer"
inputs = ("logits", "indices")
outputs = ("samples",)
api_kwargs = {"logits": "logits", "indices": "indices"}


def run(logits, indices):
    with solution_autotune(
        definition,
        backend,
        logits,
        indices,
    ):
        result = _api(
            logits=logits,
            indices=indices,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "sampling_from_logits" + " returned None without mutating declared outputs"
        )
