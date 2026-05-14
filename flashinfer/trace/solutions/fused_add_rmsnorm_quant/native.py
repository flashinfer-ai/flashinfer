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

"""FlashInfer flashinfer solution for fused_add_rmsnorm_quant."""

import torch
from flashinfer.norm import fused_add_rmsnorm_quant as _api

definition = "fused_add_rmsnorm_quant"
api = "flashinfer.norm.fused_add_rmsnorm_quant"
backend = "flashinfer"
inputs = ("hidden_states", "residual", "weight", "scale")
outputs = ("out", "residual")
api_kwargs = {
    "input": "hidden_states",
    "residual": "residual",
    "weight": "weight",
    "scale": "scale",
}
constants = {"hidden_size": 7168}


def run(hidden_states, residual, weight, scale):
    out = torch.empty(
        (hidden_states.shape[0], 7168),
        device=hidden_states.device,
        dtype=torch.float8_e4m3fn,
    )
    result = _api(
        out=out,
        input=hidden_states,
        residual=residual,
        weight=weight,
        scale=scale,
    )
    if result is not None:
        return result
    return out, residual
