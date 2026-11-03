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
import torch.nn.functional as F


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


def packed_prefill_inputs(device, *, seq_lens, num_heads=2, seed=0):
    """Realistic packed multi-token prefill inputs on ``device``."""

    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(shape, dtype=torch.bfloat16, scale=1.0):
        out = torch.randn(
            shape, dtype=torch.float32, device=device, generator=generator
        )
        return (scale * out).to(dtype)

    total_tokens = sum(seq_lens)
    shape = (1, total_tokens, num_heads, 128)
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    return {
        "q": randn(shape),
        "k": randn(shape),
        "v": randn(shape),
        "g": randn(shape, scale=0.1),
        "beta": randn((1, total_tokens, num_heads)),
        "A_log": randn(num_heads, dtype=torch.float32, scale=0.1),
        "dt_bias": randn((num_heads, 128), dtype=torch.float32, scale=0.1),
        "cu_seqlens": torch.tensor(offsets, dtype=torch.int64, device=device),
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "beta_is_logit": True,
    }


def _strict_prefill_kwargs(inputs, *, lower_bound=-5.0):
    return {
        **inputs,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": lower_bound,
        "beta_is_logit": True,
    }


def _make_inputs(
    *,
    seq_lens,
    num_heads: int,
    packed: bool,
    initial_state: bool = False,
    state_dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if packed:
        batch_size = 1
        seq_len = sum(seq_lens)
    else:
        if len(set(seq_lens)) != 1:
            raise ValueError("fixed test inputs require equal sequence lengths")
        batch_size = len(seq_lens)
        seq_len = seq_lens[0]
    shape = (batch_size, seq_len, num_heads, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = (0.1 * torch.randn(shape, dtype=torch.float32, device="cuda")).to(
        torch.bfloat16
    )
    beta = torch.randn(
        (batch_size, seq_len, num_heads),
        dtype=torch.bfloat16,
        device="cuda",
    )
    A_log = 0.1 * torch.randn(num_heads, dtype=torch.float32, device="cuda")
    dt_bias = 0.1 * torch.randn((num_heads, 128), dtype=torch.float32, device="cuda")
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    state = None
    if initial_state:
        state = (
            0.1
            * torch.randn(
                (len(seq_lens), num_heads, 128, 128),
                dtype=torch.float32,
                device="cuda",
            )
        ).to(state_dtype)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "initial_state": state,
        "cu_seqlens": (
            torch.tensor(offsets, dtype=torch.int64, device="cuda") if packed else None
        ),
    }


def _reference(inputs, *, lower_bound=-5.0, scale=None, checkpoint_every_n_tokens=0):
    q = inputs["q"]
    batch_size, seq_len, num_heads, head_dim = q.shape
    scale = head_dim**-0.5 if scale is None else scale
    q_flat = F.normalize(q.float(), dim=-1).reshape(-1, num_heads, head_dim)
    k_flat = F.normalize(inputs["k"].float(), dim=-1).reshape(-1, num_heads, head_dim)
    v_flat = inputs["v"].float().reshape(-1, num_heads, head_dim)
    g_flat = inputs["g"].float().reshape(-1, num_heads, head_dim)
    beta_flat = torch.sigmoid(inputs["beta"].float().reshape(-1, num_heads))
    gate_input = g_flat + inputs["dt_bias"].reshape(1, num_heads, head_dim)
    if lower_bound is None:
        gate = -torch.exp(inputs["A_log"]).reshape(1, num_heads, 1) * F.softplus(
            gate_input
        )
    else:
        gate = lower_bound * torch.sigmoid(
            torch.exp(inputs["A_log"]).reshape(1, num_heads, 1) * gate_input
        )
    decay = torch.exp(gate)
    if inputs["cu_seqlens"] is None:
        offsets = [index * seq_len for index in range(batch_size + 1)]
    else:
        offsets = [int(value) for value in inputs["cu_seqlens"].tolist()]
    if inputs["initial_state"] is None:
        state = torch.zeros(
            (len(offsets) - 1, num_heads, head_dim, head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        state = inputs["initial_state"].clone()
    out = torch.empty_like(q_flat)
    checkpoints = []
    for sequence in range(len(offsets) - 1):
        if checkpoint_every_n_tokens:
            checkpoints.append(state[sequence].clone())
        sequence_length = offsets[sequence + 1] - offsets[sequence]
        for local_token, token in enumerate(
            range(offsets[sequence], offsets[sequence + 1]), start=1
        ):
            state_f32 = state[sequence].float()
            decayed = state_f32 * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            residual = beta_flat[token].unsqueeze(-1) * (v_flat[token] - predicted)
            updated = decayed + residual.unsqueeze(-1) * k_flat[token].unsqueeze(1)
            state[sequence] = updated.to(torch.bfloat16)
            projected = torch.einsum(
                "hk,hvk->hv", q_flat[token], state[sequence].float()
            )
            out[token] = (scale * projected).to(torch.bfloat16)
            if (
                checkpoint_every_n_tokens
                and local_token % checkpoint_every_n_tokens == 0
                and local_token < sequence_length
            ):
                checkpoints.append(state[sequence].clone())
    result = (out.reshape_as(q), state)
    if checkpoint_every_n_tokens:
        return (*result, torch.stack(checkpoints))
    return result


def _h12_bf16_residual_carriers(torch, *, value, prediction, beta_logit):
    """Apply the four BF16 residual carriers selected by the public H12 ABI."""

    prediction_carrier = prediction.to(torch.bfloat16).float()
    delta_carrier = (value - prediction_carrier).to(torch.bfloat16).float()
    beta_carrier = torch.sigmoid(beta_logit).to(torch.bfloat16).float()
    update_carrier = (
        (beta_carrier.unsqueeze(-1) * delta_carrier).to(torch.bfloat16).float()
    )
    return prediction_carrier, delta_carrier, beta_carrier, update_carrier


def _chunk16_debug_reference(
    inputs, *, lower_bound=-5.0, scale=None, checkpoint_every_n_tokens=0
):
    """Clean-room H12 smoke reference for focused numerical diagnostics.

    The recurrent state carrier stays in FP32 within each 16-token chunk, but
    the state/K prediction, V-minus-prediction delta, sigmoid beta, and
    post-beta update carrier each round through BF16.  A BF16 state snapshot
    becomes the next chunk's carrier, while each output projects the unrounded
    FP32 state for its token.  The public benchmark separately compares output
    and complete final state against the pinned FlashKDA implementation.
    """

    q = inputs["q"]
    batch_size, seq_len, num_heads, head_dim = q.shape
    scale = head_dim**-0.5 if scale is None else scale
    q_flat = F.normalize(q.float(), dim=-1).reshape(-1, num_heads, head_dim)
    k_flat = F.normalize(inputs["k"].float(), dim=-1).reshape(-1, num_heads, head_dim)
    v_flat = inputs["v"].float().reshape(-1, num_heads, head_dim)
    g_flat = inputs["g"].float().reshape(-1, num_heads, head_dim)
    beta_logits_flat = inputs["beta"].float().reshape(-1, num_heads)
    gate = lower_bound * torch.sigmoid(
        torch.exp(inputs["A_log"]).reshape(1, num_heads, 1)
        * (g_flat + inputs["dt_bias"].reshape(1, num_heads, head_dim))
    )
    decay = torch.exp(gate)
    if inputs["cu_seqlens"] is None:
        offsets = [index * seq_len for index in range(batch_size + 1)]
    else:
        offsets = [int(value) for value in inputs["cu_seqlens"].tolist()]
    if inputs["initial_state"] is None:
        state = torch.zeros(
            (len(offsets) - 1, num_heads, head_dim, head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        state = inputs["initial_state"].clone()
    out = torch.empty_like(q_flat)
    checkpoints = []
    for sequence in range(len(offsets) - 1):
        if checkpoint_every_n_tokens:
            checkpoints.append(state[sequence].clone())
        carrier = state[sequence].float()
        sequence_length = offsets[sequence + 1] - offsets[sequence]
        for local_token, token in enumerate(
            range(offsets[sequence], offsets[sequence + 1]), start=1
        ):
            decayed = carrier * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            _, _, _, update_carrier = _h12_bf16_residual_carriers(
                torch,
                value=v_flat[token],
                prediction=predicted,
                beta_logit=beta_logits_flat[token],
            )
            updated = decayed + update_carrier.unsqueeze(-1) * k_flat[token].unsqueeze(
                1
            )
            state[sequence] = updated.to(torch.bfloat16)
            projected = torch.einsum("hk,hvk->hv", q_flat[token], updated)
            out[token] = (scale * projected).to(torch.bfloat16)
            carrier = state[sequence].float() if local_token % 16 == 0 else updated
            if (
                checkpoint_every_n_tokens
                and local_token % checkpoint_every_n_tokens == 0
                and local_token < sequence_length
            ):
                checkpoints.append(state[sequence].clone())
    result = (out.reshape_as(q), state)
    if checkpoint_every_n_tokens:
        return (*result, torch.stack(checkpoints))
    return result
