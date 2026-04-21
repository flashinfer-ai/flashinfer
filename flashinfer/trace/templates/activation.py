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

"""TraceTemplates for activation functions."""

import torch
import torch.nn.functional as F

from ..template import Const, Tensor, TraceTemplate, Var

# ── SiLU and Mul ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _silu_and_mul_reference(input):
    """Fused SiLU + Mul: silu(input[..., :H]) * input[..., H:]"""
    half = input.shape[-1] // 2
    return F.silu(input[..., :half]) * input[..., half:]


silu_and_mul_trace = TraceTemplate(
    op_type="activation",
    name_prefix="silu_and_mul",
    description="Fused SiLU + Mul: silu(x[:H]) * x[H:]. Used in LLaMA/Mistral FFN.",
    axes={
        "num_tokens": Var(description="Total number of tokens (batch_size * seq_len)."),
        "hidden_size": Const(abbrev="h", description="Output hidden size (input is 2*h)."),
    },
    inputs={
        "input": Tensor(["num_tokens", "hidden_size"], param="input",
                        description="Gated input tensor of shape [num_tokens, 2*hidden_size]."),
    },
    outputs={
        "output": Tensor(["num_tokens", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified", "fused"],
    reference=_silu_and_mul_reference,
)

# ── GeLU Tanh and Mul ────────────────────────────────────────────────────────


@torch.no_grad()
def _gelu_tanh_and_mul_reference(input):
    """Fused GeLU (tanh approx) + Mul: gelu_tanh(x[:H]) * x[H:]"""
    half = input.shape[-1] // 2
    return F.gelu(input[..., :half], approximate="tanh") * input[..., half:]


gelu_tanh_and_mul_trace = TraceTemplate(
    op_type="activation",
    name_prefix="gelu_tanh_and_mul",
    description="Fused GeLU (tanh approx) + Mul: gelu_tanh(x[:H]) * x[H:]. Used in BERT/GPT FFN.",
    axes={
        "num_tokens": Var(description="Total number of tokens."),
        "hidden_size": Const(abbrev="h", description="Output hidden size (input is 2*h)."),
    },
    inputs={
        "input": Tensor(["num_tokens", "hidden_size"], param="input",
                        description="Gated input tensor of shape [num_tokens, 2*hidden_size]."),
    },
    outputs={
        "output": Tensor(["num_tokens", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified", "fused"],
    reference=_gelu_tanh_and_mul_reference,
)

# ── GeLU and Mul ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _gelu_and_mul_reference(input):
    """Fused GeLU (exact) + Mul: gelu(x[:H]) * x[H:]"""
    half = input.shape[-1] // 2
    return F.gelu(input[..., :half]) * input[..., half:]


gelu_and_mul_trace = TraceTemplate(
    op_type="activation",
    name_prefix="gelu_and_mul",
    description="Fused GeLU (exact) + Mul: gelu(x[:H]) * x[H:].",
    axes={
        "num_tokens": Var(description="Total number of tokens."),
        "hidden_size": Const(abbrev="h", description="Output hidden size (input is 2*h)."),
    },
    inputs={
        "input": Tensor(["num_tokens", "hidden_size"], param="input",
                        description="Gated input tensor of shape [num_tokens, 2*hidden_size]."),
    },
    outputs={
        "output": Tensor(["num_tokens", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified", "fused"],
    reference=_gelu_and_mul_reference,
)
