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

"""TraceTemplates for normalization operations."""

import torch

from ..template import Const, Tensor, TraceTemplate, Var

# ── RMSNorm ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def _rmsnorm_reference(hidden_states, weight):
    """Root Mean Square Normalization. Epsilon is fixed at 1e-6."""
    EPS = 1e-6
    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)


rmsnorm_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="rmsnorm",
    description="Root Mean Square Normalization. Epsilon is fixed at 1e-6.",
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "hidden_states": Tensor(["batch_size", "hidden_size"], param="input"),
        "weight": Tensor(["hidden_size"]),
    },
    outputs={
        "output": Tensor(["batch_size", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified"],
    reference=_rmsnorm_reference,
)

# ── Fused Add + RMSNorm ───────────────────────────────────────────────────────


@torch.no_grad()
def _fused_add_rmsnorm_reference(hidden_states, residual, weight):
    """Fused Add + RMSNorm. Epsilon is fixed at 1e-6."""
    EPS = 1e-6
    x = hidden_states.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)


fused_add_rmsnorm_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="fused_add_rmsnorm",
    description="Fused Add + RMSNorm. Epsilon is fixed at 1e-6.",
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "hidden_states": Tensor(["batch_size", "hidden_size"], param="input"),
        "residual": Tensor(["batch_size", "hidden_size"]),
        "weight": Tensor(["hidden_size"]),
    },
    outputs={
        "output": Tensor(["batch_size", "hidden_size"], dtype_from="input"),
        "residual": Tensor(
            ["batch_size", "hidden_size"],
            dtype_from="input",
            description="Updated residual (in-place: residual += hidden_states).",
        ),
    },
    tags=["status:verified", "fused"],
    reference=_fused_add_rmsnorm_reference,
)
