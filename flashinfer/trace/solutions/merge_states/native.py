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

"""FlashInfer flashinfer solution for merge_states."""

from flashinfer.cascade import merge_states as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "merge_states"
api = "flashinfer.cascade.merge_states"
backend = "flashinfer"
inputs = ("v", "s")
outputs = ("v_merged", "s_merged")
api_kwargs = {"v": "v", "s": "s"}


def run(v, s):
    with solution_autotune(
        definition,
        backend,
        v,
        s,
    ):
        result = _api(
            v=v,
            s=s,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "merge_states" + " returned None without mutating declared outputs"
        )
