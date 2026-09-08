# Copyright (c) 2026 by FlashInfer team.
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

"""Shared recurrent-KDA prefill test inputs.

Used by the stable lane (``tests/kda/``) and the experimental lane
(``tests/experimental/``), which cannot import each other.
"""

import torch


def cpu_route_tensors(token_count=2):
    shape = (1, token_count, 1, 128)
    return {
        "q": torch.empty(shape, dtype=torch.bfloat16),
        "k": torch.empty(shape, dtype=torch.bfloat16),
        "v": torch.empty(shape, dtype=torch.bfloat16),
        "g": torch.empty(shape, dtype=torch.bfloat16),
        "beta": torch.empty((1, token_count, 1), dtype=torch.bfloat16),
        "A_log": torch.empty(1, dtype=torch.float32),
        "dt_bias": torch.empty((1, 128), dtype=torch.float32),
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "beta_is_logit": True,
    }
