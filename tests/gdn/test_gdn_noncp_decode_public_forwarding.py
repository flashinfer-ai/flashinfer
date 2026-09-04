# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Generated public-API forwarding tests for the frozen GDN non-CP GDN contract."""

import pytest
import torch

import flashinfer.gdn_decode as gdn
from flashinfer.jit import gdn_noncp as gdn_noncp


_PROMOTED_BF16_ROWS = [(4, 1, 16, 32, True, False, False, 0), (4, 2, 16, 32, False, True, True, 4), (8, 3, 16, 64, True, True, True, 3), (8, 4, 16, 64, True, True, True, 4), (8, 2, 16, 64, True, False, False, 0), (8, 4, 16, 64, True, False, True, 5), (8, 4, 16, 32, True, True, True, 4), (4, 1, 4, 8, True, False, False, 0), (1, 4, 4, 8, True, True, True, 4), (2, 4, 4, 8, True, True, True, 4), (3, 4, 4, 8, True, True, True, 4), (4, 4, 4, 8, True, True, True, 4), (5, 4, 4, 8, True, True, True, 4), (6, 4, 4, 8, True, True, True, 4), (7, 4, 4, 8, True, True, True, 4), (8, 4, 4, 8, True, True, True, 4)]


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
@pytest.mark.parametrize(
    "batch_size,seq_len,num_q_heads,num_v_heads,strided,disable,cache,cache_steps",
    _PROMOTED_BF16_ROWS,
)
def test_all_promoted_bf16_rows_resolve_on_both_architectures(
    arch, batch_size, seq_len, num_q_heads, num_v_heads,
    strided, disable, cache, cache_steps,
):
    route = gdn_noncp.select_gdn_noncp_decode_variant(
        arch=arch,
        batch_size=batch_size,
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        head_size=128,
        layout="pretranspose",
        num_k_heads=num_q_heads,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        scale=128 ** -0.5,
        seq_len=seq_len,
        use_qk_l2norm=True,
        strided_inputs=strided,
        disable_state_update=disable,
        cache_intermediate_states=cache,
        cache_steps=cache_steps,
    )
    assert route.route_id.startswith("flashinfer.gdn_decode.")
    assert route.variant_name


def _inputs(*, state_dtype, seq_len=2):
    batch_size, num_q_heads, num_v_heads, dim = 1, 1, 1, 128
    q = torch.zeros(batch_size, seq_len, num_q_heads, dim, dtype=torch.bfloat16)
    return dict(
        q=q,
        k=q.clone(),
        v=torch.zeros(batch_size, seq_len, num_v_heads, dim, dtype=torch.bfloat16),
        state=None,
        A_log=torch.zeros(num_v_heads, dtype=torch.float32),
        a=torch.zeros(batch_size, seq_len, num_v_heads, dtype=torch.bfloat16),
        dt_bias=torch.zeros(num_v_heads, dtype=torch.float32),
        b=torch.zeros(batch_size, seq_len, num_v_heads, dtype=torch.bfloat16),
        output=torch.empty(
            batch_size, seq_len, num_v_heads, dim, dtype=torch.bfloat16
        ),
        initial_state=torch.zeros(3, num_v_heads, dim, dim, dtype=state_dtype),
        initial_state_indices=torch.tensor([1], dtype=torch.int32),
        output_state_indices=torch.tensor([2], dtype=torch.int32),
    )


def test_public_bf16_verify_forwards_caller_cache_and_no_update(monkeypatch):
    tensors = _inputs(state_dtype=torch.bfloat16)
    cache = torch.empty(1, 4, 1, 128, 128, dtype=torch.bfloat16)
    captured = {}

    def fake_mtp(**kwargs):
        captured.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(gdn, "_GDN_DECODE_BF16_STATE_AVAILABLE", True)
    monkeypatch.setattr(gdn, "_gated_delta_rule_bf16_state_mtp", fake_mtp)
    output, returned_state = gdn.gated_delta_rule_decode_pretranspose(
        **tensors,
        intermediate_states_buffer=cache,
        disable_state_update=True,
    )
    assert captured["intermediate_states_buffer"] is cache
    assert captured["disable_state_update"] is True
    assert output is tensors["output"]
    assert returned_state is tensors["initial_state"]


def test_public_fp32_checkpoint_forwards_caller_cache_and_update(monkeypatch):
    tensors = _inputs(state_dtype=torch.float32)
    cache = torch.empty(1, 3, 1, 128, 128, dtype=torch.float32)
    captured = {}

    def fake_mtp(**kwargs):
        captured.update(kwargs)
        return kwargs["output"], kwargs["initial_state"]

    monkeypatch.setattr(gdn, "gated_delta_rule_mtp", fake_mtp)
    output, returned_state = gdn.gated_delta_rule_decode_pretranspose(
        **tensors,
        intermediate_states_buffer=cache,
        disable_state_update=False,
    )
    assert captured["intermediate_states_buffer"] is cache
    assert captured["disable_state_update"] is False
    assert output is tensors["output"]
    assert returned_state is tensors["initial_state"]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"intermediate_states_buffer": torch.empty(1)}, "only for T > 1"),
        ({"disable_state_update": True}, "only for T > 1"),
    ],
)
def test_public_t1_controls_fail_closed_before_launch(kwargs, match):
    tensors = _inputs(state_dtype=torch.bfloat16, seq_len=1)
    with pytest.raises(ValueError, match=match):
        gdn.gated_delta_rule_decode_pretranspose(**tensors, **kwargs)
