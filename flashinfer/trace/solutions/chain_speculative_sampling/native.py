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

"""FlashInfer flashinfer solution for chain_speculative_sampling."""

from flashinfer.sampling import chain_speculative_sampling as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "chain_speculative_sampling"
api = "flashinfer.sampling.chain_speculative_sampling"
backend = "flashinfer"
inputs = ("draft_probs", "draft_token_ids", "target_probs")
outputs = ("accepted_token_ids",)
api_kwargs = {
    "draft_probs": "draft_probs",
    "draft_token_ids": "draft_token_ids",
    "target_probs": "target_probs",
}
constants = {"vocab_size": 32000}


def run(draft_probs, draft_token_ids, target_probs):
    with solution_autotune(
        definition,
        backend,
        draft_probs,
        draft_token_ids,
        target_probs,
    ):
        result = _api(
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            target_probs=target_probs,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "chain_speculative_sampling"
            + " returned None without mutating declared outputs"
        )
