"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""BF16 prefill routing: one-wave grids take the M64 value split on SM100a/SM103a."""

import pytest
import torch

from flashinfer import prepare_bf16_kda_prefill

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="KDA export requires SM100a or SM103a",
)


def _prepare(lengths, heads, *, active_beta):
    torch.manual_seed(7)
    tokens = sum(lengths)
    shape = (1, tokens, heads, 128)
    q, k, v, g = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.25 for _ in range(4)
    ]
    if active_beta:
        beta = torch.rand(shape[:-1], device="cuda", dtype=torch.float32)
    else:
        beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    pool = torch.randn((len(lengths) + 2, heads, 128, 128), device="cuda") * 0.1
    indices = torch.arange(len(lengths), device="cuda", dtype=torch.int32) + 1
    offsets, cp_offsets = [0], [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
        cp_offsets.append(cp_offsets[-1] + (n + 63) // 64)
    cp = torch.empty(
        (cp_offsets[-1], heads, 128, 128), device="cuda", dtype=torch.bfloat16
    )
    out = torch.empty_like(q)
    call = prepare_bf16_kda_prefill(
        q,
        k,
        v,
        g,
        beta,
        A_log=torch.zeros(heads, device="cuda"),
        dt_bias=torch.full((heads, 128), -2.0, device="cuda"),
        out=out,
        initial_state=pool,
        final_state=pool,
        cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int64),
        sequence_lengths=tuple(lengths),
        state_indices=indices,
        state_checkpoints=cp,
        checkpoint_cu_starts=torch.tensor(cp_offsets, device="cuda", dtype=torch.int64),
        checkpoint_every_n_tokens=64,
        beta_is_logit=not active_beta,
    )
    call.launch()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert torch.isfinite(pool[indices]).all()
    assert torch.isfinite(cp).all()
    return str(call.schedule)


@pytest.mark.parametrize(
    ("lengths", "heads", "active_beta", "expected"),
    [
        # Six packed H12 sequences (72 tasks) with active FP32 beta: one wave on
        # both architectures, so the native active-beta M64 split is selected.
        (
            (17, 64, 65, 127, 128, 255),
            12,
            True,
            "fused_active_beta_checkpoint_dvsplit_m64",
        ),
        # Logit beta, FP32 state, 24 tasks: the FP32-state M64 split.
        ((33, 1025), 12, False, "fused_m64_independent_dvsplit_fp32_state"),
        # Mixed lengths with six heads (36 tasks), active beta.
        (
            (1300, 547, 2048, 963, 271, 3063),
            6,
            True,
            "fused_active_beta_checkpoint_dvsplit_m64",
        ),
    ],
)
def test_one_wave_bf16_grid_uses_m64_value_split(lengths, heads, active_beta, expected):
    assert _prepare(lengths, heads, active_beta=active_beta) == expected


def test_short_h12_logit_beta_keeps_direct_n16_tile():
    """Contiguous H12 logit beta needs a per-launch TMA refresh; the short direct tile wins."""
    assert (
        _prepare((64,), 12, active_beta=False) == "fused_checkpoint_tma_direct_m128_n16"
    )


def test_multi_wave_grid_keeps_direct_tile():
    """96 tasks exceed one M64 wave on 148 and 152 SMs; the direct family is retained."""
    schedule = _prepare((128,) * 8, 12, active_beta=True)
    assert "m64" not in schedule
    assert schedule.startswith("fused_")
