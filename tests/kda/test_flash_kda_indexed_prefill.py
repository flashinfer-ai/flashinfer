"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for FP32 indexed state-pool prefill through the generated Cake route.

``backend="cake"`` with an FP32 state pool and ``ssm_state_indices`` is served
by the general generated prefill portfolio for every pool capacity, including
the 257-slot pools that an exact-shape portfolio used to intercept.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import flashinfer.kda as kda_api

from tests.test_helpers.kda_prefill import _make_inputs, _strict_prefill_kwargs

_POOL_SLOTS = 257
_HEAD_DIM = 128


def _public_call_tensors() -> dict[str, torch.Tensor]:
    q = torch.empty((1, 2, 1, _HEAD_DIM), dtype=torch.bfloat16)
    return {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty_like(q),
        "beta": torch.empty((1, 2, 1), dtype=torch.bfloat16),
        "A_log": torch.empty((1,), dtype=torch.float32),
        "dt_bias": torch.empty((1, _HEAD_DIM), dtype=torch.float32),
        "initial_state": torch.empty(
            (_POOL_SLOTS, 1, _HEAD_DIM, _HEAD_DIM), dtype=torch.float32
        ),
        "ssm_state_indices": torch.zeros((1,), dtype=torch.int32),
    }


def test_explicit_cake_backend_routes_fp32_pool_to_generated_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = (object(), object())
    observed = {}
    monkeypatch.setattr(
        kda_api._kda_prefill,
        "_flash_kda_prefill_is_eligible",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        kda_api._kda_prefill,
        "_run_flash_kda_prefill",
        lambda **kwargs: observed.update(kwargs) or sentinel,
    )

    result = kda_api.recurrent_kda(
        **_public_call_tensors(),
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
        backend="cake",
    )

    assert result is sentinel
    assert observed["initial_state"].dtype == torch.float32
    assert observed["initial_state"].shape[0] == _POOL_SLOTS
    assert observed["state_indices"].dtype == torch.int32


def _fp32_state_reference(inputs, *, state, lower_bound=-5.0):
    """Token-serial KDA recurrence with the state carried in FP32."""

    q = inputs["q"]
    _, _, num_heads, head_dim = q.shape
    scale = head_dim**-0.5
    q_flat = F.normalize(q.float(), dim=-1).reshape(-1, num_heads, head_dim)
    k_flat = F.normalize(inputs["k"].float(), dim=-1).reshape(-1, num_heads, head_dim)
    v_flat = inputs["v"].float().reshape(-1, num_heads, head_dim)
    g_flat = inputs["g"].float().reshape(-1, num_heads, head_dim)
    beta_flat = torch.sigmoid(inputs["beta"].float().reshape(-1, num_heads))
    gate_input = g_flat + inputs["dt_bias"].reshape(1, num_heads, head_dim)
    decay = torch.exp(
        lower_bound
        * torch.sigmoid(
            torch.exp(inputs["A_log"]).reshape(1, num_heads, 1) * gate_input
        )
    )
    if inputs["cu_seqlens"] is None:
        offsets = [0, q.shape[1]]
    else:
        offsets = [int(value) for value in inputs["cu_seqlens"].tolist()]
    state = state.clone().float()
    out = torch.empty_like(q_flat)
    for sequence in range(len(offsets) - 1):
        for token in range(offsets[sequence], offsets[sequence + 1]):
            decayed = state[sequence] * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            residual = beta_flat[token].unsqueeze(-1) * (v_flat[token] - predicted)
            state[sequence] = decayed + residual.unsqueeze(-1) * k_flat[
                token
            ].unsqueeze(1)
            projected = torch.einsum("hk,hvk->hv", q_flat[token], state[sequence])
            out[token] = (scale * projected).to(torch.bfloat16)
    return out.reshape_as(q), state


@pytest.mark.gpu
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("num_heads", "seq_lens", "packed"),
    [
        (6, [512], False),
        (12, [63], False),
        (96, [65], False),
        (6, [17, 33, 65], True),
        (96, [128] * 8, True),
    ],
)
def test_cake_backend_fp32_257_slot_pool_takes_generated_route_and_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    num_heads: int,
    seq_lens: list[int],
    packed: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("the generated prefill portfolio targets SM100a and SM103a")
    device = torch.device("cuda")
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=packed,
        initial_state=True,
        state_dtype=torch.float32,
        seed=5837 + num_heads + len(seq_lens),
    )
    compact_state = inputs.pop("initial_state")
    num_sequences = len(seq_lens)
    state_indices = torch.tensor(
        [(31 * index + 7) % _POOL_SLOTS for index in range(num_sequences)],
        dtype=torch.int32,
        device=device,
    )
    state_pool = 0.1 * torch.randn(
        (_POOL_SLOTS, num_heads, _HEAD_DIM, _HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    state_pool.index_copy_(0, state_indices.to(torch.int64), compact_state)
    pool_before = state_pool.clone()
    expected_output, expected_state = _fp32_state_reference(inputs, state=compact_state)

    calls = []
    real_run = kda_api._kda_prefill._run_flash_kda_prefill

    def counted_run(**kwargs):
        calls.append(kwargs["initial_state"])
        return real_run(**kwargs)

    monkeypatch.setattr(kda_api._kda_prefill, "_run_flash_kda_prefill", counted_run)

    output = torch.empty_like(inputs["q"])
    actual_output, actual_state = kda_api.recurrent_kda(
        **_strict_prefill_kwargs({**inputs, "initial_state": state_pool}),
        ssm_state_indices=state_indices,
        output=output,
        output_final_state=True,
        backend="cake",
    )
    torch.cuda.synchronize()

    assert len(calls) == 1 and calls[0] is state_pool
    assert actual_output is output
    assert actual_state is state_pool
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        state_pool.index_select(0, state_indices.to(torch.int64)),
        expected_state,
        atol=1e-2,
        rtol=1e-2,
    )
    untouched = torch.ones(_POOL_SLOTS, dtype=torch.bool, device=device)
    untouched[state_indices.to(torch.int64)] = False
    assert torch.equal(state_pool[untouched], pool_before[untouched])
