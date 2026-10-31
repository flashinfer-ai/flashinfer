# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""fp32 GDN MTP FLA-style pool scatter against real serving pools.

The public fp32 MTP entry point (``flashinfer.gdn_decode.gated_delta_rule_mtp``
with ``ssm_state_indices``) is the vLLM-compat form of speculative-decode GDN:
the initial state is gathered per request, and h after token t is scattered to
``pool[ssm_state_indices[i, t]]``. Serving frameworks hand it pools this file
exercises and the small contiguous pools of the existing tests do not:

- **page-strided pools** (per-slot stride > HV*V*K; vLLM carves the SSM state
  out of a padded page allocation) — the 4D ``use_pool_indexing`` path;
- **pools of different sizes with identical per-slot strides** across one
  process (a small profiling pool followed by the full serving pool) — the
  JIT cache must not reuse a kernel compiled for another pool shape;
- **large pools**: the flat slot index times the 16,384-element fp32 slot row
  overflows Int32 at flat index >= 131,072 (slot >= 10,923 at HV=12), so slot
  offsets must be computed in Int64. The same wrap applies to the dense
  ``intermediate_states_buffer`` cache tiles (flat index
  ``i * cache_steps * HV``, e.g. T=8, HV=32, B >= 513).

Geometry mirrors a TP-sharded Qwen3.5-class GDN layer: H=4, HV=12, K=V=128,
T=4.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from flashinfer.gdn_decode import gated_delta_rule_mtp
from flashinfer.utils import get_compute_capability

H, HV, K, V, T = 4, 12, 128, 128, 4
SLOT_PAD = 512  # extra fp32 elements per slot in the page-strided layout


def _skip_if_not_sm90_or_later():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    major, _ = get_compute_capability(torch.device("cuda"))
    if major < 9:
        pytest.skip("fp32 GDN MTP decode requires SM90 or later")


def _make_pool(slots: int, strided: bool, device) -> torch.Tensor:
    if not strided:
        pool = torch.zeros(slots, HV, V, K, dtype=torch.float32, device=device)
    else:
        slot_elems = HV * V * K
        backing = torch.zeros(
            slots * (slot_elems + SLOT_PAD), dtype=torch.float32, device=device
        )
        pool = backing.as_strided(
            (slots, HV, V, K), (slot_elems + SLOT_PAD, V * K, K, 1)
        )
    pool.normal_(0, 0.5)
    return pool


def _make_inputs(B: int, device, seed: int):
    torch.manual_seed(seed)
    return dict(
        q=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
        k=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device),
        v=torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device),
        a=(torch.randn(B, T, HV, device=device) * 0.5).to(torch.bfloat16),
        b=torch.randn(B, T, HV, device=device).to(torch.bfloat16),
        A_log=(torch.rand(HV, dtype=torch.float32, device=device) * 3.0 + 0.1).log(),
        dt_bias=(torch.randn(HV, device=device) * 0.5).to(torch.bfloat16),
        scale=1.0 / math.sqrt(K),
        use_qk_l2norm=True,
        disable_state_update=False,
    )


def _run_scatter(inputs, pool, read_idx, scat_idx):
    B = inputs["q"].shape[0]
    out = torch.zeros(B, T, HV, V, dtype=torch.bfloat16, device=pool.device)
    gated_delta_rule_mtp(
        initial_state=pool,
        initial_state_indices=read_idx,
        ssm_state_indices=scat_idx,
        output=out,
        **inputs,
    )
    torch.cuda.synchronize()
    return out


def _seed_slots(pool, read_idx, seed_states):
    for i in range(read_idx.numel()):
        pool[int(read_idx[i].item())] = seed_states[i]


def _gather_scatter_slots(pool, scat_idx):
    B, Tn = scat_idx.shape
    return torch.stack(
        [
            torch.stack([pool[int(scat_idx[i, t].item())].float() for t in range(Tn)])
            for i in range(B)
        ]
    )


# B=2 dispatches the inline MTP kernel, B=16 the warp-specialized one.
@pytest.mark.parametrize("batch_size", [2, 16])
def test_fp32_fla_scatter_strided_pool_matches_contiguous(batch_size: int):
    """Scatter into a page-strided (4D ``use_pool_indexing``) pool must
    compile and produce the same outputs and per-token states as the same
    problem laid out in a contiguous pool."""
    _skip_if_not_sm90_or_later()
    device = torch.device("cuda")
    B = batch_size
    slots = B * (T + 1) + 8
    seed = int(os.environ.get("SEED", "0"))

    inputs = _make_inputs(B, device, seed)
    read_idx = torch.arange(1, B + 1, dtype=torch.int32, device=device)
    scat_idx = torch.arange(B + 1, B + 1 + B * T, dtype=torch.int32, device=device)
    scat_idx = scat_idx.reshape(B, T).contiguous()
    seed_states = torch.randn(B, HV, V, K, dtype=torch.float32, device=device)

    results = {}
    for strided in (False, True):
        pool = _make_pool(slots, strided, device)
        _seed_slots(pool, read_idx, seed_states)
        out = _run_scatter(inputs, pool, read_idx, scat_idx)
        results[strided] = (out, _gather_scatter_slots(pool, scat_idx))

    torch.testing.assert_close(results[True][0], results[False][0], atol=0, rtol=0)
    torch.testing.assert_close(results[True][1], results[False][1], atol=0, rtol=0)


@pytest.mark.parametrize("batch_size", [16])
def test_fp32_fla_scatter_pool_size_change_same_strides(batch_size: int):
    """Two strided pools with identical per-slot strides but different slot
    counts in one process: the second call must not reuse the kernel compiled
    for the first pool's shape (and must produce the same results as the
    contiguous layout)."""
    _skip_if_not_sm90_or_later()
    device = torch.device("cuda")
    B = batch_size
    seed = int(os.environ.get("SEED", "0")) + 1

    inputs = _make_inputs(B, device, seed)
    read_idx = torch.arange(1, B + 1, dtype=torch.int32, device=device)
    scat_idx = (
        torch.arange(B + 1, B + 1 + B * T, dtype=torch.int32, device=device)
        .reshape(B, T)
        .contiguous()
    )
    seed_states = torch.randn(B, HV, V, K, dtype=torch.float32, device=device)

    reference = None
    for slots in (B * (T + 1) + 8, 4 * B * (T + 1)):
        pool_c = _make_pool(slots, False, device)
        _seed_slots(pool_c, read_idx, seed_states)
        out_c = _run_scatter(inputs, pool_c, read_idx, scat_idx)

        pool_s = _make_pool(slots, True, device)
        _seed_slots(pool_s, read_idx, seed_states)
        out_s = _run_scatter(inputs, pool_s, read_idx, scat_idx)

        torch.testing.assert_close(out_s, out_c, atol=0, rtol=0)
        torch.testing.assert_close(
            _gather_scatter_slots(pool_s, scat_idx),
            _gather_scatter_slots(pool_c, scat_idx),
            atol=0,
            rtol=0,
        )
        if reference is None:
            reference = out_c


# B=8 keeps the inline kernel in range at HV=12; B=16 exercises the
# warp-specialized kernel.
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("batch_size", [8, 16])
def test_fp32_fla_scatter_large_pool_int64_offsets(batch_size: int, strided: bool):
    """Relocation invariance on a pool past the Int32 offset-overflow
    threshold: the same problem placed at bottom-of-pool and at top-of-pool
    slots must produce identical outputs and states. The element offset wraps
    Int32 from flat index 131,072 (slot >= 10,923 at HV=12); 12,288 slots put
    the top-of-pool placement (flat up to 147,455) past it, so with Int32
    offset arithmetic it reads/writes through wrapped addresses (silent
    corruption or an illegal memory access)."""
    _skip_if_not_sm90_or_later()
    device = torch.device("cuda")
    B = batch_size
    slots = 12288
    need = slots * (HV * V * K + (SLOT_PAD if strided else 0)) * 4 * 1.1
    free, _ = torch.cuda.mem_get_info()
    if free < need:
        pytest.skip(f"needs ~{need / 2**30:.1f} GiB free device memory")
    seed = int(os.environ.get("SEED", "0")) + 2

    inputs = _make_inputs(B, device, seed)
    n = B * (T + 1)
    low = torch.arange(1, 1 + n, dtype=torch.int32, device=device)
    high = torch.arange(slots - n, slots, dtype=torch.int32, device=device)
    seed_states = torch.randn(B, HV, V, K, dtype=torch.float32, device=device)

    pool = _make_pool(slots, strided, device)
    results = []
    for idx in (low, high):
        read_idx = idx[:B].contiguous()
        scat_idx = idx[B:].reshape(B, T).contiguous()
        _seed_slots(pool, read_idx, seed_states)
        out = _run_scatter(inputs, pool, read_idx, scat_idx)
        results.append((out, _gather_scatter_slots(pool, scat_idx)))

    torch.testing.assert_close(results[1][0], results[0][0], atol=0, rtol=0)
    torch.testing.assert_close(results[1][1], results[0][1], atol=0, rtol=0)


def test_fp32_intermediate_buffer_large_batch_int64_offsets():
    """The dense ``intermediate_states_buffer`` cache tiles share the Int32
    wrap: the buffer's flat index ``i * cache_steps * HV`` exceeds 131,072
    at T=8, HV=32 from B >= 513. Scatter mode and buffer mode compute the
    same per-token states, so ``buffer[i, t]`` must equal
    ``pool[ssm_state_indices[i, t]]`` for every (i, t) and the outputs must
    match; with Int32 buffer offsets the tail of the batch writes through
    wrapped addresses."""
    _skip_if_not_sm90_or_later()
    device = torch.device("cuda")
    B, T8, HV32, H8 = 528, 8, 32, 8
    pool_slots = B * (T8 + 1) + 1
    # Peak residency: the buffer plus BOTH pools (``pool`` and its scatter
    # clone live simultaneously); the comparison below is per batch row, so
    # no gathered full-size tensor is ever materialized.
    need = (B * T8 * HV32 + 2 * pool_slots * HV32) * V * K * 4 * 1.2
    free, _ = torch.cuda.mem_get_info()
    if free < need:
        pytest.skip(f"needs ~{need / 2**30:.1f} GiB free device memory")
    seed = int(os.environ.get("SEED", "0")) + 3

    torch.manual_seed(seed)
    inputs = dict(
        q=torch.randn(B, T8, H8, K, dtype=torch.bfloat16, device=device),
        k=torch.randn(B, T8, H8, K, dtype=torch.bfloat16, device=device),
        v=torch.randn(B, T8, HV32, V, dtype=torch.bfloat16, device=device),
        a=(torch.randn(B, T8, HV32, device=device) * 0.5).to(torch.bfloat16),
        b=torch.randn(B, T8, HV32, device=device).to(torch.bfloat16),
        A_log=(torch.rand(HV32, dtype=torch.float32, device=device) * 3.0 + 0.1).log(),
        dt_bias=(torch.randn(HV32, device=device) * 0.5).to(torch.bfloat16),
        scale=1.0 / math.sqrt(K),
        use_qk_l2norm=True,
        initial_state_indices=torch.arange(1, B + 1, dtype=torch.int32, device=device),
    )
    pool = torch.zeros(pool_slots, HV32, V, K, dtype=torch.float32, device=device)
    pool.normal_(0, 0.5)
    pool_scatter = pool.clone()
    scat_idx = (
        torch.arange(B + 1, B + 1 + B * T8, dtype=torch.int32, device=device)
        .reshape(B, T8)
        .contiguous()
    )

    out_scatter = torch.zeros(B, T8, HV32, V, dtype=torch.bfloat16, device=device)
    gated_delta_rule_mtp(
        initial_state=pool_scatter,
        ssm_state_indices=scat_idx,
        output=out_scatter,
        disable_state_update=False,
        **inputs,
    )
    torch.cuda.synchronize()

    buffer = torch.zeros(B, T8, HV32, V, K, dtype=torch.float32, device=device)
    out_buffer = torch.zeros(B, T8, HV32, V, dtype=torch.bfloat16, device=device)
    gated_delta_rule_mtp(
        initial_state=pool,
        intermediate_states_buffer=buffer,
        output=out_buffer,
        disable_state_update=False,
        **inputs,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_buffer, out_scatter, atol=0, rtol=0)
    # Compare per batch row to keep the peak at buffer + two pools (a full
    # advanced-index gather would transiently add another buffer-size copy).
    for i in range(B):
        torch.testing.assert_close(
            buffer[i], pool_scatter[scat_idx[i].long()], atol=0, rtol=0
        )


if __name__ == "__main__":
    test_fp32_fla_scatter_strided_pool_matches_contiguous(2)
    test_fp32_fla_scatter_strided_pool_matches_contiguous(16)
    test_fp32_fla_scatter_pool_size_change_same_strides(16)
    for b in (8, 16):
        for st in (False, True):
            test_fp32_fla_scatter_large_pool_int64_offsets(b, st)
    test_fp32_intermediate_buffer_large_batch_int64_offsets()
    print("all pool-scatter tests passed")
