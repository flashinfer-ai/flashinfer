# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Direct public-API tests for the frozen GDN non-CP non-CP GDN prefill routes."""

import gc
import itertools
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flashinfer import gdn_prefill as prefill_mod
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.jit import cake_gdn as cake_gdn
from tests.test_helpers import cake_profiler


@pytest.fixture(autouse=True)
def _require_cake_gdn_arch():
    if not torch.cuda.is_available():
        pytest.skip("Cake GDN requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    try:
        cake_gdn.arch_for_compute_capability(major, minor)
    except cake_gdn.CakeGDNUnsupportedError as error:
        pytest.skip(str(error))


def _expand_heads(q, k, v):
    if q.shape[1] >= v.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    else:
        q = q.repeat_interleave(v.shape[1] // q.shape[1], dim=1)
        k = k.repeat_interleave(v.shape[1] // k.shape[1], dim=1)
    return q, k, v


def _reference(case, initial_state):
    q, k, v = _expand_heads(case["q"], case["k"], case["v"])
    alpha, beta = case["g"], case["beta"]
    output = torch.empty_like(case["output"], dtype=torch.float32)
    final_state = case["output_state"].clone()
    checkpoints = case["state_checkpoints"].clone()
    token_start = 0
    checkpoint_start = 0
    for seq_idx, length in enumerate(case["seq_lens"]):
        state_row = (
            int(case["state_indices"][seq_idx])
            if case["state_indices"] is not None
            else seq_idx
        )
        if initial_state is None:
            state = torch.zeros(
                (q.shape[1], 128, 128),
                dtype=case["state_dtype"],
                device=q.device,
            )
        else:
            state = initial_state[state_row].transpose(-1, -2).contiguous()
        for local_idx in range(length):
            token = token_start + local_idx
            old_state = alpha[token].reshape(-1, 1, 1) * state.float()
            old_v = torch.einsum("hd,hdv->hv", k[token].float(), old_state)
            new_v = beta[token].reshape(-1, 1) * v[token].float()
            new_v += (1.0 - beta[token].reshape(-1, 1)) * old_v
            updated = old_state - k[token].float().unsqueeze(-1) * old_v.unsqueeze(-2)
            updated += k[token].float().unsqueeze(-1) * new_v.unsqueeze(-2)
            state = updated.to(case["state_dtype"])
            output[token] = case["scale"] * torch.einsum(
                "hd,hdv->hv", q[token].float(), state.float()
            )
            interval = case["checkpoint_every"]
            if interval and (local_idx + 1) % interval == 0:
                checkpoint_idx = checkpoint_start + (local_idx + 1) // interval - 1
                checkpoints[checkpoint_idx] = state.transpose(-1, -2)
        final_state[state_row] = state.transpose(-1, -2)
        token_start += length
        if case["checkpoint_every"]:
            checkpoint_start += length // case["checkpoint_every"]
    return output.to(case["q"].dtype), final_state, checkpoints


def _caller_qk_l2norm(value):
    value_f32 = value.float()
    denominator = torch.sqrt(
        torch.sum(value_f32 * value_f32, dim=-1, keepdim=True) + 1e-6
    )
    return (value_f32 / denominator).to(value.dtype)


def _expected_noncp_route_suffix(*, seq_lens, num_q_heads=4, num_v_heads=8, **_):
    """Mirror select_cake_gdn_prefill_variant's physical-schedule rule for the live arch:
    DV-split when 2 * num_seqs * num_o_heads fits the arch's active-cluster count."""
    major, minor = torch.cuda.get_device_capability()
    arch = cake_gdn.arch_for_compute_capability(major, minor)
    tiles = 2 * len(seq_lens) * max(num_q_heads, num_v_heads)
    return ".dvsplit" if tiles <= cake_gdn._ARCH_ACTIVE_CLUSTERS[arch] else ".full_dv"


def _make_case(
    *,
    seq_lens,
    io_dtype=torch.bfloat16,
    state_dtype=torch.float32,
    indexed=False,
    checkpoint_every=0,
    num_q_heads=4,
    num_k_heads=None,
    num_v_heads=8,
    state_pool_padding=257,
    checkpoint_cu_dtype=torch.int32,
    gates=True,
):
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    hq, hv = num_q_heads, num_v_heads
    hk = min(hq, hv) if num_k_heads is None else num_k_heads
    heads = max(hq, hv)
    total = sum(seq_lens)
    q = torch.randn((total, hq, 128), generator=generator, device="cuda").to(io_dtype)
    q = _caller_qk_l2norm(q)
    k = torch.randn((total, hk, 128), generator=generator, device="cuda").to(io_dtype)
    k = _caller_qk_l2norm(k)
    v = (torch.randn((total, hv, 128), generator=generator, device="cuda") * 0.1).to(
        io_dtype
    )
    alpha = torch.rand((total, heads), generator=generator, device="cuda")
    beta = torch.rand((total, heads), generator=generator, device="cuda")
    cu = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    state_indices = None
    backing = None
    initial_state = None
    if indexed:
        pool_size = len(seq_lens) + 3
        row_elems = heads * 128 * 128
        slot_stride = row_elems + state_pool_padding
        backing = torch.full(
            (pool_size * slot_stride,), 7.0, dtype=state_dtype, device="cuda"
        )
        initial_state = torch.as_strided(
            backing,
            (pool_size, heads, 128, 128),
            (slot_stride, 16384, 128, 1),
        )
        initial_state.copy_(
            (
                torch.randn(initial_state.shape, generator=generator, device="cuda")
                * 0.05
            ).to(state_dtype)
        )
        state_indices = torch.tensor(
            list(range(2, 2 + len(seq_lens))), dtype=torch.int32, device="cuda"
        )
        output_state = initial_state
    elif checkpoint_every == 0:
        initial_state = (
            torch.randn(
                (len(seq_lens), heads, 128, 128), generator=generator, device="cuda"
            )
            * 0.05
        ).to(state_dtype)
        output_state = initial_state.clone()
    else:
        output_state = torch.zeros(
            (len(seq_lens), heads, 128, 128), dtype=state_dtype, device="cuda"
        )
    checkpoint_counts = (
        [length // checkpoint_every for length in seq_lens]
        if checkpoint_every
        else [0] * len(seq_lens)
    )
    checkpoint_cu = torch.tensor(
        [0, *torch.tensor(checkpoint_counts).cumsum(0).tolist()],
        dtype=checkpoint_cu_dtype,
        device="cuda",
    )
    state_checkpoints = torch.empty(
        (sum(checkpoint_counts), heads, 128, 128), dtype=state_dtype, device="cuda"
    )
    if state_checkpoints.numel():
        state_checkpoints.fill_(float("nan"))
    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "g": alpha.contiguous() if gates else None,
        "beta": beta.contiguous() if gates else None,
        "scale": 1.0 / math.sqrt(128),
        "initial_state": initial_state,
        "output": torch.empty((total, heads, 128), dtype=io_dtype, device="cuda"),
        "output_state": output_state,
        "state_indices": state_indices,
        "state_checkpoints": state_checkpoints,
        "checkpoint_cu_starts": checkpoint_cu,
        "checkpoint_every": checkpoint_every,
        "cu_seqlens": cu,
        "seq_lens": tuple(seq_lens),
        "state_dtype": state_dtype,
        "num_q_heads": hq,
        "num_k_heads": hk,
        "num_v_heads": hv,
        "backing": backing,
    }


def _host_cu_seqlens(seq_lens):
    """The host copy an engine already holds (prefix sums of the lengths)."""

    return [0, *itertools.accumulate(seq_lens)]


def _launch(case, **public_kwargs):
    return chunk_gated_delta_rule(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        g=case["g"],
        beta=case["beta"],
        scale=case["scale"],
        initial_state=case["initial_state"],
        output_final_state=True,
        cu_seqlens=case["cu_seqlens"],
        use_qk_l2norm_in_kernel=False,
        output=case["output"],
        output_state=case["output_state"],
        state_checkpoints=(
            case["state_checkpoints"] if case["checkpoint_every"] else None
        ),
        checkpoint_cu_starts=(
            case["checkpoint_cu_starts"] if case["checkpoint_every"] else None
        ),
        checkpoint_every_n_tokens=case["checkpoint_every"],
        use_cp=False,
        state_indices=case["state_indices"],
        backend="cake_gdn",
        **public_kwargs,
    )


def _public_route(case):
    """The frozen route the public adapter resolves for ``case`` on the live arch."""

    major, minor = torch.cuda.get_device_capability(case["q"].device)
    return cake_gdn.select_cake_gdn_prefill_variant(
        arch=cake_gdn.arch_for_compute_capability(major, minor),
        io_dtype={torch.bfloat16: "bfloat16", torch.float16: "float16"}[
            case["q"].dtype
        ],
        state_dtype={torch.bfloat16: "bfloat16", torch.float32: "float32"}[
            case["state_dtype"]
        ],
        num_seqs=len(case["seq_lens"]),
        total_seq_len=sum(case["seq_lens"]),
        max_seq_len=max(case["seq_lens"]),
        num_q_heads=case["num_q_heads"],
        num_k_heads=case["num_k_heads"],
        num_v_heads=case["num_v_heads"],
        use_initial_state=case["initial_state"] is not None,
        store_final_state=True,
        checkpoint_every_n_tokens=case["checkpoint_every"],
        use_state_indices=case["state_indices"] is not None,
        gates_present=True,
    )


def _launch_raw_indexed_prefill(case):
    """Invoke the exported TVM FFI entry without public-adapter validation."""

    major, minor = torch.cuda.get_device_capability(case["q"].device)
    arch = cake_gdn.arch_for_compute_capability(major, minor)
    route = _public_route(case)
    entry = cake_gdn.load_cake_gdn_kernel(route.variant_name, arch)
    num_o_heads = max(case["num_q_heads"], case["num_v_heads"])
    dvsplit = route.route_id.endswith(".dvsplit")
    total_tiles = len(case["seq_lens"]) * num_o_heads * (2 if dvsplit else 1)
    active_clusters = int(
        torch.cuda.get_device_properties(case["q"].device).multi_processor_count
    )
    # The public route's single host-known grid rule, with the exact bound.
    grid_x = prefill_mod._cake_gdn_prefill_grid_x(
        total_tiles, active_clusters, dvsplit, max(case["seq_lens"])
    )
    empty_i32 = torch.empty(1, dtype=torch.int32, device=case["q"].device)
    empty_state = torch.empty(1, dtype=case["state_dtype"], device=case["q"].device)
    workspace = torch.empty(grid_x * 512, dtype=torch.uint8, device=case["q"].device)
    entry(
        case["q"],
        case["k"],
        case["v"],
        case["output"],
        case["g"],
        case["beta"],
        case["cu_seqlens"],
        case["state_indices"],
        case["initial_state"],
        case["output_state"],
        case["state_checkpoints"] if case["checkpoint_every"] else empty_state,
        case["checkpoint_cu_starts"] if case["checkpoint_every"] else empty_i32,
        workspace,
        int(case["initial_state"].stride(0)),
        int(case["output_state"].stride(0)),
        case["checkpoint_every"],
        case["scale"],
        len(case["seq_lens"]),
        # Launch ABI v2: total_tokens bounds every device-side cu_seqlens read.
        sum(case["seq_lens"]),
        case["num_q_heads"],
        case["num_v_heads"],
        total_tiles,
        grid_x,
        1,
        1,
    )
    return route


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "seq_lens": (128,),
            "indexed": True,
            "state_dtype": torch.bfloat16,
            "num_q_heads": 16,
            "num_k_heads": 16,
            "num_v_heads": 16,
            "state_pool_padding": 0,
        },
        {
            "seq_lens": (1,),
            "state_dtype": torch.bfloat16,
            "num_q_heads": 16,
            "num_k_heads": 16,
            "num_v_heads": 16,
        },
        {
            "seq_lens": (128,),
            "io_dtype": torch.float16,
            "checkpoint_every": 64,
            "num_q_heads": 2,
            "num_k_heads": 2,
            "num_v_heads": 4,
        },
        {
            "seq_lens": (128,),
            "io_dtype": torch.float16,
            "state_dtype": torch.float32,
            "num_q_heads": 2,
            "num_k_heads": 2,
            "num_v_heads": 4,
        },
        # Qwen3.5 per-rank head splits with gathered BF16 state (the SGLang
        # extend route): DV-split and full-DV physical schedules.
        {
            "seq_lens": (512, 300),
            "state_dtype": torch.bfloat16,
            "num_q_heads": 8,
            "num_k_heads": 8,
            "num_v_heads": 16,
        },
        {
            "seq_lens": (1024,),
            "state_dtype": torch.bfloat16,
            "num_q_heads": 8,
            "num_k_heads": 8,
            "num_v_heads": 32,
        },
        {
            "seq_lens": (257, 64, 1000),
            "state_dtype": torch.bfloat16,
            "num_q_heads": 4,
            "num_k_heads": 4,
            "num_v_heads": 16,
        },
        {
            "seq_lens": (512,) * 8,
            "state_dtype": torch.bfloat16,
            "num_q_heads": 4,
            "num_k_heads": 4,
            "num_v_heads": 16,
        },
        {
            "seq_lens": (96,) * 16,
            "state_dtype": torch.bfloat16,
            "num_q_heads": 2,
            "num_k_heads": 2,
            "num_v_heads": 8,
        },
        {
            "seq_lens": (2048,) * 4,
            "state_dtype": torch.bfloat16,
            "num_q_heads": 8,
            "num_k_heads": 8,
            "num_v_heads": 32,
        },
        # Qwen3-Next / Qwen3.5 per-rank head splits with packed FP32 initial
        # and final state (the vLLM TP1/TP2/TP4/TP8 route), each geometry on
        # both physical schedules: DV-split while 2 * num_seqs * num_o_heads
        # fits the arch's active clusters (148/160/212), full-DV at 256 tiles
        # on every arch.
        {
            "seq_lens": (300, 512),
            "state_dtype": torch.float32,
            "num_q_heads": 8,
            "num_k_heads": 8,
            "num_v_heads": 16,
        },
        {
            "seq_lens": (512,) * 8,
            "state_dtype": torch.float32,
            "num_q_heads": 8,
            "num_k_heads": 8,
            "num_v_heads": 16,
        },
        {
            "seq_lens": (1024,),
            "state_dtype": torch.float32,
            "num_q_heads": 4,
            "num_k_heads": 4,
            "num_v_heads": 8,
        },
        {
            "seq_lens": (256,) * 16,
            "state_dtype": torch.float32,
            "num_q_heads": 4,
            "num_k_heads": 4,
            "num_v_heads": 8,
        },
        {
            "seq_lens": (640,),
            "state_dtype": torch.float32,
            "num_q_heads": 2,
            "num_k_heads": 2,
            "num_v_heads": 4,
        },
        {
            "seq_lens": (96,) * 32,
            "state_dtype": torch.float32,
            "num_q_heads": 2,
            "num_k_heads": 2,
            "num_v_heads": 4,
        },
        {
            "seq_lens": (2048,),
            "state_dtype": torch.float32,
            "num_q_heads": 8,
            "num_k_heads": 8,
            "num_v_heads": 32,
        },
        {
            "seq_lens": (512,) * 4,
            "state_dtype": torch.float32,
            "num_q_heads": 8,
            "num_k_heads": 8,
            "num_v_heads": 32,
        },
        {
            "seq_lens": (300, 1000),
            "state_dtype": torch.float32,
            "num_q_heads": 4,
            "num_k_heads": 4,
            "num_v_heads": 16,
        },
        {
            "seq_lens": (512,) * 8,
            "state_dtype": torch.float32,
            "num_q_heads": 4,
            "num_k_heads": 4,
            "num_v_heads": 16,
        },
        {
            "seq_lens": (257,),
            "state_dtype": torch.float32,
            "num_q_heads": 2,
            "num_k_heads": 2,
            "num_v_heads": 8,
        },
        {
            "seq_lens": (96,) * 16,
            "state_dtype": torch.float32,
            "num_q_heads": 2,
            "num_k_heads": 2,
            "num_v_heads": 8,
        },
    ),
)
def test_public_cake_gdn_prefill_matches_independent_recurrence(kwargs):
    case = _make_case(**kwargs)
    initial_state = (
        None if case["initial_state"] is None else case["initial_state"].clone()
    )
    backing_before = None if case["backing"] is None else case["backing"].clone()
    expected_out, expected_state, expected_checkpoints = _reference(case, initial_state)
    route = _public_route(case)
    assert route.route_id.endswith(_expected_noncp_route_suffix(**kwargs)), (
        route.route_id
    )
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out, state = _launch(case)
    stream.synchronize()
    assert out is case["output"]
    assert state is case["output_state"]
    torch.testing.assert_close(out.float(), expected_out.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )
    if case["checkpoint_every"]:
        torch.testing.assert_close(
            case["state_checkpoints"].float(),
            expected_checkpoints.float(),
            atol=1e-2,
            rtol=1e-2,
        )
    if backing_before is not None:
        row_elems = state.shape[1] * 128 * 128
        slot_stride = state.stride(0)
        for slot in range(state.shape[0]):
            torch.testing.assert_close(
                case["backing"][
                    slot * slot_stride + row_elems : (slot + 1) * slot_stride
                ],
                backing_before[
                    slot * slot_stride + row_elems : (slot + 1) * slot_stride
                ],
                atol=0,
                rtol=0,
            )


def _carry_segment(case, start, stop, initial_state):
    """``case`` cut to tokens ``[start, stop)`` of its one sequence, started from ``initial_state``."""

    length = stop - start
    segment = dict(case)
    for key in ("q", "k", "v", "g", "beta"):
        segment[key] = case[key][start:stop].contiguous()
    segment["cu_seqlens"] = torch.tensor(
        [0, length], dtype=torch.int32, device=case["q"].device
    )
    segment["seq_lens"] = (length,)
    segment["initial_state"] = initial_state
    segment["output"] = torch.empty(
        (length, *case["output"].shape[1:]),
        dtype=case["output"].dtype,
        device=case["output"].device,
    )
    segment["output_state"] = initial_state.clone()
    return segment


@pytest.mark.parametrize(
    "heads",
    ((8, 16), (4, 8), (2, 4), (8, 32), (4, 16), (2, 8)),
    ids=lambda heads: f"h{heads[0]}_hv{heads[1]}",
)
def test_public_cake_gdn_prefill_fp32_state_carry_matches_single_pass(heads):
    # Chunked prefill over the vLLM FP32-state contract: one sequence as two
    # consecutive public calls (six 64-token chunks, then four), the first
    # call's final state feeding the second as its initial state.  Output and
    # final state must match the single-pass call and the reference recurrence.
    num_q_heads, num_v_heads = heads
    split, total = 384, 640
    case = _make_case(
        seq_lens=(total,),
        num_q_heads=num_q_heads,
        num_k_heads=num_q_heads,
        num_v_heads=num_v_heads,
    )
    initial_state = case["initial_state"].clone()
    expected_out, expected_state, _ = _reference(case, initial_state)
    single_out, single_state = _launch(case)
    torch.cuda.synchronize()
    first = _carry_segment(case, 0, split, initial_state.clone())
    first_out, carried_state = _launch(first)
    second = _carry_segment(case, split, total, carried_state)
    second_out, final_state = _launch(second)
    torch.cuda.synchronize()
    assert carried_state is first["output_state"]
    assert final_state is second["output_state"]
    carried_out = torch.cat((first_out, second_out))
    for actual, expected in (
        (carried_out, single_out),
        (final_state, single_state),
        (carried_out, expected_out),
        (final_state, expected_state),
    ):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=1e-2, rtol=1e-2
        )


def test_public_cake_gdn_prefill_raw_ffi_bf16_indexed_checkpoint_b7_t421_noncompact_matches_independent_recurrence():
    case = _make_case(
        seq_lens=(52, 93, 15, 107, 72, 61, 21),
        indexed=True,
        state_dtype=torch.bfloat16,
        checkpoint_every=64,
        state_pool_padding=257,
    )
    initial_state = case["initial_state"].clone()
    backing_before = case["backing"].clone()
    expected_out, expected_state, expected_checkpoints = _reference(case, initial_state)
    original_device = torch.cuda.current_device()
    target_device = int(case["q"].device.index or 0)
    if torch.cuda.device_count() < 2:
        route = _launch_raw_indexed_prefill(case)
    else:
        other_device = (target_device + 1) % torch.cuda.device_count()
        torch.cuda.set_device(other_device)
        try:
            route = _launch_raw_indexed_prefill(case)
            assert torch.cuda.current_device() == other_device
        finally:
            torch.cuda.set_device(original_device)
    torch.cuda.synchronize(target_device)
    assert route.route_id.endswith(".dvsplit"), route.route_id
    torch.testing.assert_close(
        case["output"].float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        case["output_state"].float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        case["state_checkpoints"].float(),
        expected_checkpoints.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    row_elems = case["output_state"].shape[1] * 128 * 128
    slot_stride = case["output_state"].stride(0)
    for slot in range(case["output_state"].shape[0]):
        torch.testing.assert_close(
            case["backing"][slot * slot_stride + row_elems : (slot + 1) * slot_stride],
            backing_before[slot * slot_stride + row_elems : (slot + 1) * slot_stride],
            atol=0,
            rtol=0,
        )


def test_public_cake_gdn_prefill_api_bf16_indexed_checkpoint_b5_t4296_live_matches_independent_recurrence():
    case = _make_case(
        seq_lens=(849, 835, 862, 897, 853),
        indexed=True,
        state_dtype=torch.bfloat16,
        checkpoint_every=64,
        state_pool_padding=0,
        checkpoint_cu_dtype=torch.int64,
    )
    initial_state = case["initial_state"].clone()
    backing_before = case["backing"].clone()
    expected_out, expected_state, expected_checkpoints = _reference(case, initial_state)
    out, state = _launch(case)
    torch.cuda.synchronize(int(case["q"].device.index or 0))
    assert out is case["output"]
    assert state is case["output_state"]
    torch.testing.assert_close(
        case["output"].float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        case["output_state"].float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        case["state_checkpoints"].float(),
        expected_checkpoints.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    row_elems = case["output_state"].shape[1] * 128 * 128
    slot_stride = case["output_state"].stride(0)
    for slot in range(case["output_state"].shape[0]):
        torch.testing.assert_close(
            case["backing"][slot * slot_stride + row_elems : (slot + 1) * slot_stride],
            backing_before[slot * slot_stride + row_elems : (slot + 1) * slot_stride],
            atol=0,
            rtol=0,
        )


def test_public_cake_gdn_prefill_rejects_invalid_checkpoint_geometry_before_launch():
    """The checkpoint count is checked exactly from the host copy of cu_seqlens.

    Without ``cu_seqlens_cpu`` the route knows only host shapes (it reads nothing
    back from the device), so ``state_checkpoints.shape[0]`` is taken as given and
    the contents of ``checkpoint_cu_starts`` stay the documented caller
    precondition the kernel guards on the device.
    """

    case = _make_case(
        seq_lens=(849, 835, 862, 897, 853),
        indexed=True,
        state_dtype=torch.bfloat16,
        checkpoint_every=64,
        state_pool_padding=0,
        checkpoint_cu_dtype=torch.int64,
    )
    checkpoints = case["state_checkpoints"]
    case["state_checkpoints"] = checkpoints[:-1]
    case["output"].fill_(13.0)
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="state_checkpoints must have shape",
    ):
        _launch(case, cu_seqlens_cpu=_host_cu_seqlens(case["seq_lens"]))
    torch.cuda.synchronize()
    assert torch.all(case["output"] == 13.0), "rejected before any launch"

    # A host copy that disagrees with the device metadata is rejected on the
    # host as well: one token short of the packed length.
    case["state_checkpoints"] = checkpoints
    short = _host_cu_seqlens(case["seq_lens"])
    short[-1] -= 1
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="monotonic cu_seqlens spanning all input tokens",
    ):
        _launch(case, cu_seqlens_cpu=short)
    torch.cuda.synchronize()
    assert torch.all(case["output"] == 13.0), "rejected before any launch"


@pytest.mark.parametrize(
    "host_copy,public_kwargs,match",
    (
        pytest.param([0, 128, 191], {}, "monotonic cu_seqlens spanning", id="short"),
        pytest.param([0, 150, 100], {}, "monotonic cu_seqlens spanning", id="order"),
        pytest.param([0, 192], {}, "num_seqs \\+ 1 = 3 entries", id="length"),
        pytest.param("device", {}, "int32/int64 CPU tensor", id="cuda-tensor"),
        pytest.param(
            [0, 128, 192],
            {"max_seqlen": 100},
            "max_seqlen is smaller than the longest sequence",
            id="hint-below-exact-max",
        ),
    ),
)
def test_public_cake_gdn_prefill_rejects_malformed_cu_seqlens_cpu_before_launch(
    host_copy, public_kwargs, match
):
    """``cu_seqlens_cpu`` is validated on the host before any launch.

    A CUDA tensor is refused rather than copied: the copy would be the
    device-to-host synchronization the parameter exists to avoid.
    """

    case = _make_case(
        seq_lens=(128, 64),
        state_dtype=torch.bfloat16,
        num_q_heads=8,
        num_k_heads=8,
        num_v_heads=16,
    )
    if host_copy == "device":
        host_copy = case["cu_seqlens"]
    case["output"].fill_(13.0)
    with pytest.raises(cake_gdn.CakeGDNUnsupportedError, match=match):
        _launch(case, cu_seqlens_cpu=host_copy, **public_kwargs)
    torch.cuda.synchronize()
    assert torch.all(case["output"] == 13.0), "rejected before any launch"


@pytest.mark.parametrize(
    "host_copy_kind", ("list", "int32_tensor", "int64_tensor", "max_seqlen_hint")
)
def test_public_cake_gdn_prefill_host_copy_forms_match_device_metadata(host_copy_kind):
    """Every accepted host form and the bare hint give the same kernel result."""

    case = _make_case(
        seq_lens=(300, 212),
        state_dtype=torch.bfloat16,
        num_q_heads=8,
        num_k_heads=8,
        num_v_heads=16,
    )
    initial_state = case["initial_state"].clone()
    _launch(case)
    torch.cuda.synchronize()
    expected_out, expected_state = case["output"].clone(), case["output_state"].clone()
    offsets = _host_cu_seqlens(case["seq_lens"])
    public_kwargs = {
        "list": {"cu_seqlens_cpu": offsets},
        "int32_tensor": {"cu_seqlens_cpu": torch.tensor(offsets, dtype=torch.int32)},
        "int64_tensor": {"cu_seqlens_cpu": torch.tensor(offsets, dtype=torch.int64)},
        "max_seqlen_hint": {"max_seqlen": max(case["seq_lens"])},
    }[host_copy_kind]
    case["initial_state"].copy_(initial_state)
    case["output"].zero_()
    out, state = _launch(case, **public_kwargs)
    torch.cuda.synchronize()
    assert out is case["output"] and state is case["output_state"]
    assert torch.equal(case["output"], expected_out)
    assert torch.equal(case["output_state"], expected_state)


def _forbidden_sync_names(names):
    """Device-to-host copies and host synchronizations among profiler event names."""

    return [
        name
        for name in names
        if "Memcpy DtoH" in name or "Synchronize" in name or name == "cudaMemcpy"
    ]


def _profile_window(fn, attempts=cake_profiler.ATTEMPTS):
    """Names of the CUDA runtime/device events of ``fn`` (CPU events inside its window).

    Sentinel kernels open and close the window (``tests.test_helpers.cake_profiler``): a
    session whose device records miss a leading or the trailing one lost CUDA activity
    (some torch / CUPTI builds of the nightly images do), so the window is profiled again;
    after ``attempts`` such sessions the test is skipped.  The sentinels' own records are
    not among the names (their ``cudaLaunchKernel`` runtime events are, harmlessly), and the
    synchronization that drains the trailing one runs outside the window.
    """

    for attempt in range(1, attempts + 1):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            with torch.profiler.record_function("cake_gdn_prefill_window"):
                cake_profiler.launch_sentinel()
                cake_profiler.launch_sentinel()
                fn()
                cake_profiler.launch_sentinel()
            torch.cuda.synchronize()
        events = profile.events()
        # The annotation is reported once per activity source (the CPU record and,
        # on newer profilers, a device-side user annotation of the same name); the
        # window is the CPU record's span, and every copy of the marker is skipped.
        windows = [
            event
            for event in events
            if event.name == "cake_gdn_prefill_window"
            and event.device_type == torch.autograd.DeviceType.CPU
        ]
        assert windows, [
            event.name for event in events if event.name == "cake_gdn_prefill_window"
        ]
        start = min(event.time_range.start for event in windows)
        end = max(event.time_range.end for event in windows)
        names = []
        device = []
        for event in events:
            if event.name == "cake_gdn_prefill_window":
                continue
            if event.device_type == torch.autograd.DeviceType.CPU:
                if start <= event.time_range.start and event.time_range.end <= end:
                    names.append(event.name)
            else:
                device.append(event)
        device.sort(key=lambda event: event.time_range.start)
        device_names = [event.name for event in device]
        inner = cake_profiler.between_sentinels(device_names)
        if inner is not None:
            return names + inner
        cake_profiler.warn_lost_records(attempt, attempts, device_names)
    cake_profiler.skip_lost_records(attempts)


@pytest.mark.parametrize(
    "with_host_copy", (False, True), ids=("device_metadata", "cu_seqlens_cpu")
)
@pytest.mark.parametrize("cu_dtype", (torch.int32, torch.int64), ids=("i32", "i64"))
def test_public_cake_gdn_prefill_performs_no_device_to_host_sync(
    with_host_copy, cu_dtype
):
    """Zero-sync proof: 36 calls under sync_debug_mode="error" and a profile.

    The first call per kernel variant compiles and loads it (nvcc + tvm_ffi); that
    one-time step is eager by design and runs before the guard.  Every later call
    -- with or without the host copy, int32 or int64 metadata -- performs no
    device-to-host read: no ``.tolist()``, no cached host ints, no stream or
    device synchronization.
    """

    case = _make_case(
        seq_lens=(300, 212),
        state_dtype=torch.bfloat16,
        num_q_heads=8,
        num_k_heads=8,
        num_v_heads=16,
    )
    case["cu_seqlens"] = case["cu_seqlens"].to(cu_dtype)
    public_kwargs = (
        {"cu_seqlens_cpu": _host_cu_seqlens(case["seq_lens"])} if with_host_copy else {}
    )
    initial_state = case["initial_state"].clone()
    _launch(case, **public_kwargs)  # one-time kernel compile/load, eager by design
    torch.cuda.synchronize()
    expected_out, expected_state = case["output"].clone(), case["output_state"].clone()

    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(36):
            case["initial_state"].copy_(initial_state)
            _launch(case, **public_kwargs)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    torch.cuda.synchronize()
    assert torch.equal(case["output"], expected_out)
    assert torch.equal(case["output_state"], expected_state)

    # The detector sees a real device-to-host read (positive control) ...
    control = _profile_window(lambda: case["cu_seqlens"].cpu())
    assert _forbidden_sync_names(control), control
    # ... and none in the route.
    names = _profile_window(lambda: _launch(case, **public_kwargs))
    assert not _forbidden_sync_names(names), names
    torch.cuda.synchronize()


def test_public_cake_gdn_prefill_captures_a_cuda_graph_without_an_eager_call():
    """Capture needs no eager metadata resolution.

    The only eager step is the one-time compile/load of the kernel variant,
    done here through the loader for the variant the public route resolves (no
    launch, no metadata read) before capturing a batch the route has never seen.
    Replay then reads the live ``cu_seqlens``.
    """

    case = _make_case(
        seq_lens=(200, 56),
        state_dtype=torch.bfloat16,
        num_q_heads=8,
        num_k_heads=8,
        num_v_heads=16,
    )
    initial_state = case["initial_state"].clone()
    expected_out, expected_state, _ = _reference(case, initial_state)
    major, minor = torch.cuda.get_device_capability()
    arch = cake_gdn.arch_for_compute_capability(major, minor)
    route = cake_gdn.select_cake_gdn_prefill_variant(
        arch=arch,
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        num_seqs=len(case["seq_lens"]),
        total_seq_len=sum(case["seq_lens"]),
        max_seq_len=sum(case["seq_lens"]),  # no hint: the packed length is the bound
        num_q_heads=8,
        num_k_heads=8,
        num_v_heads=16,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=0,
        use_state_indices=False,
        gates_present=True,
    )
    cake_gdn.load_cake_gdn_kernel(route.variant_name, arch)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out, state = _launch(case)
    graph.replay()
    torch.cuda.synchronize()
    assert out is case["output"]
    assert state is case["output_state"]
    torch.testing.assert_close(
        case["output"].float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        case["output_state"].float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


def test_public_cake_gdn_prefill_graph_replay_reads_rewritten_buffers_in_place():
    """Replay follows the live inputs: no identity-keyed metadata cache.

    Capture with one split, rewrite q/k/v/gates/cu_seqlens/initial_state in place
    with a second batch of the same shapes (a different split and different
    values), replay, and compare with the reference and with an eager call on
    the rewritten buffers.
    """

    heads = {"num_q_heads": 8, "num_k_heads": 8, "num_v_heads": 16}
    live = _make_case(seq_lens=(128, 128), state_dtype=torch.bfloat16, **heads)
    second = _make_case(seq_lens=(200, 56), state_dtype=torch.bfloat16, **heads)
    second["v"] = (-second["v"]).contiguous()
    second["g"] = second["g"].flip(0).contiguous()
    second["initial_state"] = (second["initial_state"] * 0.5).contiguous()
    expected_out, expected_state, _ = _reference(second, second["initial_state"])

    initial_state = live["initial_state"].clone()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch(live)
    stream.synchronize()
    with torch.cuda.stream(stream):
        live["initial_state"].copy_(initial_state)
        live["output"].zero_()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _launch(live)

    with torch.cuda.stream(stream):
        for key in ("q", "k", "v", "g", "beta", "cu_seqlens", "initial_state"):
            live[key].copy_(second[key])
        live["output"].zero_()
    stream.synchronize()
    graph.replay()
    stream.synchronize()
    torch.testing.assert_close(
        live["output"].float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        live["output_state"].float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )
    replayed_out, replayed_state = live["output"].clone(), live["output_state"].clone()

    with torch.cuda.stream(stream):
        live["initial_state"].copy_(second["initial_state"])
        live["output"].zero_()
        _launch(live)
    stream.synchronize()
    assert torch.equal(live["output"], replayed_out)
    assert torch.equal(live["output_state"], replayed_state)


@pytest.mark.parametrize(
    "state_dtype", (torch.float32, torch.bfloat16), ids=("fp32_state", "bf16_state")
)
@pytest.mark.parametrize("chunks", (2, 4, 8))
@pytest.mark.parametrize(
    "seq_lens", ((1024,), (768, 256)), ids=("one_sequence", "unequal_batch")
)
def test_public_cake_gdn_prefill_chunked_state_carry_matches_one_call(
    state_dtype, chunks, seq_lens
):
    """k chunked calls carrying output_state -> initial_state match one call.

    Every chunk call passes its own host ``cu_seqlens_cpu``; the carried state
    is the kernel's final state of the previous chunk in the state dtype.
    """

    # H16/HV32 (Qwen3.5 TP1): the head split shipped for both the fp32-state
    # and the bf16-state initial-state kernels.
    case = _make_case(
        seq_lens=seq_lens,
        state_dtype=state_dtype,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
    )
    initial_state = case["initial_state"].clone()
    expected_out, expected_state, _ = _reference(case, initial_state)
    single_out, single_state = _launch(case)
    torch.cuda.synchronize()
    single_out, single_state = single_out.clone(), single_state.clone()

    starts = _host_cu_seqlens(seq_lens)
    heads = max(case["num_q_heads"], case["num_v_heads"])
    carried = initial_state.clone()
    output = torch.empty_like(case["output"])
    for step in range(chunks):
        ranges = []
        for start, length in zip(starts, seq_lens, strict=False):
            assert length % chunks == 0
            piece = length // chunks
            ranges.append((start + step * piece, start + (step + 1) * piece))
        index = torch.cat([torch.arange(lo, hi, device="cuda") for lo, hi in ranges])
        chunk_lens = [hi - lo for lo, hi in ranges]
        chunk_out = torch.empty(
            (len(index), heads, 128), dtype=case["q"].dtype, device="cuda"
        )
        next_state = torch.empty_like(carried)
        out, state = chunk_gated_delta_rule(
            q=case["q"][index].contiguous(),
            k=case["k"][index].contiguous(),
            v=case["v"][index].contiguous(),
            g=case["g"][index].contiguous(),
            beta=case["beta"][index].contiguous(),
            scale=case["scale"],
            initial_state=carried,
            output_final_state=True,
            cu_seqlens=torch.tensor(
                _host_cu_seqlens(chunk_lens), dtype=torch.int32, device="cuda"
            ),
            output=chunk_out,
            output_state=next_state,
            use_cp=False,
            backend="cake_gdn",
            cu_seqlens_cpu=_host_cu_seqlens(chunk_lens),
        )
        assert out is chunk_out and state is next_state
        output[index] = chunk_out
        carried = next_state
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        carried.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(output.float(), single_out.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        carried.float(), single_state.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize(
    "malformed,effective",
    (
        pytest.param((0, 128, 500), (128, 64), id="end-past-total-is-clamped"),
        pytest.param((0, 128, 64), (128, 0), id="non-monotonic-pair-is-empty"),
        pytest.param((0, 250, 192), (192, 0), id="start-past-total-is-clamped"),
    ),
)
def test_public_cake_gdn_prefill_malformed_device_metadata_degrades_to_a_no_op(
    malformed, effective
):
    """Without a host copy the host trusts ``cu_seqlens``; the kernel clamps (ABI v2).

    Every offset is clamped into ``[0, total_tokens]`` and a non-monotonic pair
    becomes an empty sequence: the result equals the batch with the effective
    lengths, rows of an emptied sequence stay untouched and its state slot keeps
    its initial state -- never an out-of-bounds access.
    """

    heads = {"num_q_heads": 8, "num_k_heads": 8, "num_v_heads": 16}
    case = _make_case(seq_lens=(128, 64), state_dtype=torch.bfloat16, **heads)
    # The same tensors with the effective sequence structure: the reference
    # leaves the rows and the state slot of a zero-length sequence alone.
    reference_case = dict(case, seq_lens=tuple(effective))
    initial_state = case["initial_state"].clone()
    expected_out, expected_state, _ = _reference(reference_case, initial_state)
    case["cu_seqlens"].copy_(torch.tensor(malformed, dtype=torch.int32, device="cuda"))
    case["output"].fill_(13.0)
    _launch(case)
    torch.cuda.synchronize()
    covered = sum(effective)
    torch.testing.assert_close(
        case["output"][:covered].float(),
        expected_out[:covered].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.all(case["output"][covered:] == 13.0), "emptied rows stay untouched"
    for slot, length in enumerate(effective):
        if length:
            torch.testing.assert_close(
                case["output_state"][slot].float(),
                expected_state[slot].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        else:
            assert torch.equal(case["output_state"][slot], initial_state[slot])


def test_public_cake_gdn_prefill_checkpoint_is_cuda_graph_safe():
    case = _make_case(
        seq_lens=(64,),
        indexed=True,
        state_dtype=torch.bfloat16,
        checkpoint_every=64,
        state_pool_padding=0,
        checkpoint_cu_dtype=torch.int64,
    )
    initial_state = case["initial_state"].clone()
    expected_out, expected_state, expected_checkpoints = _reference(case, initial_state)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch(case)
    stream.synchronize()
    with torch.cuda.stream(stream):
        case["initial_state"].copy_(initial_state)
        case["output"].zero_()
        case["state_checkpoints"].fill_(float("nan"))
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out, state = _launch(case)
    graph.replay()
    stream.synchronize()
    assert out is case["output"]
    assert state is case["output_state"]
    torch.testing.assert_close(
        case["output"].float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        case["output_state"].float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        case["state_checkpoints"].float(),
        expected_checkpoints.float(),
        atol=1e-2,
        rtol=1e-2,
    )


@torch.inference_mode()
def test_public_cake_gdn_prefill_int64_metadata_matches_int32_and_follows_rewrites():
    """Int64 cu_seqlens / slots: bitwise-identical to int32, never cached.

    A ``metadata_i64`` record (every sm_107a record) reads the caller's int32 or
    int64 buffer directly; every other record takes int32 and an int64 buffer is
    cast on the stream at every call.  Either way nothing is cached, so an int64
    buffer the caller rewrites in place under inference mode (no version bump)
    drives the kernel with the new offsets, exactly like passing them as int32.
    """

    case = _make_case(
        seq_lens=(128, 64),
        indexed=True,
        state_dtype=torch.bfloat16,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
        state_pool_padding=0,
    )
    initial_state = case["initial_state"].clone()
    _launch(case)
    torch.cuda.synchronize()
    out_i32 = case["output"].clone()
    state_i32 = case["output_state"].clone()

    cu64 = case["cu_seqlens"].to(torch.int64)
    slots64 = case["state_indices"].to(torch.int64)
    case["cu_seqlens"], case["state_indices"] = cu64, slots64
    case["initial_state"].copy_(initial_state)
    case["output"].zero_()
    _launch(case)
    torch.cuda.synchronize()
    assert torch.equal(case["output"], out_i32)
    assert torch.equal(case["output_state"], state_i32)

    # A call on another stream is just another call (no copy to hand over).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        case["initial_state"].copy_(initial_state)
        case["output"].zero_()
        _launch(case)
    side.synchronize()
    assert torch.equal(case["output"], out_i32)
    assert torch.equal(case["output_state"], state_i32)

    # The other split as int32 ...
    other = [0, 64, 192]
    case["cu_seqlens"] = torch.tensor(other, dtype=torch.int32, device="cuda")
    case["initial_state"].copy_(initial_state)
    case["output"].zero_()
    _launch(case)
    torch.cuda.synchronize()
    out_other = case["output"].clone()
    state_other = case["output_state"].clone()
    assert not torch.equal(out_other, out_i32), "the split must matter"

    # ... equals the int64 buffer rewritten in place to that split.
    cu64.copy_(torch.tensor(other, dtype=torch.int64, device="cuda"))
    case["cu_seqlens"] = cu64
    case["initial_state"].copy_(initial_state)
    case["output"].zero_()
    _launch(case)
    torch.cuda.synchronize()
    assert torch.equal(case["output"], out_other)
    assert torch.equal(case["output_state"], state_other)


@torch.inference_mode()
def test_public_cake_gdn_prefill_int64_state_indices_follow_in_place_updates():
    """Slots rewritten in place (no version bump under inference mode) reach the kernel.

    state_indices are validated on-device every call and were never under the
    host-int immutability contract, so an int64 slot buffer that the caller refreshes
    with ``copy_`` must drive the kernel with the new slots, bitwise-identical to
    passing those slots as int32 (read in place on a ``metadata_i64`` record, cast
    per call on every other record).
    """

    case = _make_case(
        seq_lens=(128, 64),
        indexed=True,
        state_dtype=torch.bfloat16,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
        state_pool_padding=0,
    )
    pristine_pool = case["initial_state"].clone()
    new_slots = [0, 4]  # pool has len(seq_lens) + 3 rows; default slots are [2, 3]

    case["state_indices"] = torch.tensor(new_slots, dtype=torch.int32, device="cuda")
    _launch(case)
    torch.cuda.synchronize()
    out_i32 = case["output"].clone()
    pool_i32 = case["initial_state"].clone()

    slots64 = torch.tensor([2, 3], dtype=torch.int64, device="cuda")
    case["state_indices"] = slots64
    case["initial_state"].copy_(pristine_pool)
    case["output"].zero_()
    _launch(case)
    torch.cuda.synchronize()
    assert not torch.equal(case["initial_state"], pool_i32), "slots [2, 3] must differ"

    slots64.copy_(torch.tensor(new_slots, dtype=torch.int64, device="cuda"))
    case["initial_state"].copy_(pristine_pool)
    case["output"].zero_()
    _launch(case)
    torch.cuda.synchronize()
    assert torch.equal(case["output"], out_i32)
    assert torch.equal(case["initial_state"], pool_i32)


def test_public_cake_gdn_prefill_int64_graph_replay_follows_in_place_rewrite(
    monkeypatch,
):
    """A graph captured with int64 cu_seqlens reads the live buffer on replay.

    On a ``metadata_i64`` record the captured kernel reads the caller's int64
    buffer itself (no cast is recorded); on every other record the cast is
    recorded into the graph (its int32 copy lives in the graph's private pool).
    Rewriting the caller's int64 buffer in place between replays changes what
    the kernel reads, bitwise-identical to the eager int32 call on the same split.
    """

    case = _make_case(
        seq_lens=(128, 64),
        indexed=True,
        state_dtype=torch.bfloat16,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
        state_pool_padding=0,
    )
    cu64 = case["cu_seqlens"].to(torch.int64)
    case["cu_seqlens"] = cu64
    initial_state = case["initial_state"].clone()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch(case)
    stream.synchronize()
    with torch.cuda.stream(stream):
        case["initial_state"].copy_(initial_state)
        case["output"].zero_()
    stream.synchronize()
    casts = _record_metadata_casts(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _launch(case)
    if _public_route(case).metadata_i64:
        assert casts == [], casts
    else:
        assert torch.int64 in casts, casts

    other = [0, 64, 192]
    with torch.cuda.stream(stream):
        case["cu_seqlens"] = torch.tensor(other, dtype=torch.int32, device="cuda")
        case["initial_state"].copy_(initial_state)
        case["output"].zero_()
        _launch(case)
    stream.synchronize()
    out_other = case["output"].clone()
    state_other = case["output_state"].clone()

    with torch.cuda.stream(stream):
        cu64.copy_(torch.tensor(other, dtype=torch.int64, device="cuda"))
        case["initial_state"].copy_(initial_state)
        case["output"].zero_()
    stream.synchronize()
    graph.replay()
    stream.synchronize()
    assert torch.equal(case["output"], out_other)
    assert torch.equal(case["output_state"], state_other)


def _live_arch(case):
    major, minor = torch.cuda.get_device_capability(case["q"].device)
    return cake_gdn.arch_for_compute_capability(major, minor)


def _device_kernel_names(fn):
    """Names of the device kernels ``fn`` launches (profiler CUDA activities; copies excluded).

    ``tests.test_helpers.cake_profiler`` brackets the call with sentinel kernels and retries a
    session that lost its CUDA records; ``fn`` runs once more as the profiler's warm-up call.
    """

    return cake_profiler.cuda_kernel_names(fn)


def _record_metadata_casts(monkeypatch):
    """Dtypes handed to ``_cake_gdn_prefill_i32`` (the per-call int64 -> int32 cast path)."""

    seen = []
    original = prefill_mod._cake_gdn_prefill_i32

    def recording(values):
        seen.append(values.dtype)
        return original(values)

    monkeypatch.setattr(prefill_mod, "_cake_gdn_prefill_i32", recording)
    return seen


def _reset_outputs(case, initial_state):
    case["initial_state"].copy_(initial_state)
    case["output"].zero_()
    if case["checkpoint_every"]:
        case["state_checkpoints"].fill_(float("nan"))
    torch.cuda.synchronize()


def _outputs_of(case):
    return (
        case["output"].clone(),
        case["output_state"].clone(),
        case["state_checkpoints"].clone() if case["checkpoint_every"] else None,
    )


def _assert_same_outputs(case, expected):
    out, state, checkpoints = expected
    assert torch.equal(case["output"], out)
    assert torch.equal(case["output_state"], state)
    if checkpoints is not None:
        assert torch.equal(case["state_checkpoints"], checkpoints)


_METADATA_WIDTHS = (
    pytest.param((torch.int64, torch.int64, torch.int64), id="all-i64"),
    pytest.param((torch.int64, torch.int32, torch.int64), id="i64-cu-i32-slots"),
    pytest.param((torch.int32, torch.int64, torch.int32), id="i64-slots-only"),
    pytest.param((torch.int32, torch.int32, torch.int64), id="i64-checkpoints-only"),
)


@pytest.mark.parametrize("widths", _METADATA_WIDTHS)
def test_public_cake_gdn_prefill_metadata_i64_record_reads_mixed_widths_without_a_cast(
    monkeypatch, widths
):
    """A ``metadata_i64`` record (every sm_107a record) reads int32 or int64 metadata in place.

    ``cu_seqlens`` / ``state_indices`` / ``checkpoint_cu_starts`` each go to the
    kernel in the width the caller holds; the kernel decodes the widths from its
    launch mask and every output is bitwise identical to the all-int32 call.  On
    such a record the host never calls the per-call cast and a call is exactly
    one device kernel.  On an int32-only record the same inputs take the cast
    path (extra kernels per call) and reach the same outputs.
    """

    cu_dtype, slots_dtype, checkpoints_dtype = widths
    case = _make_case(
        seq_lens=(52, 93, 15, 107),
        indexed=True,
        state_dtype=torch.bfloat16,
        checkpoint_every=64,
        state_pool_padding=0,
    )
    initial_state = case["initial_state"].clone()
    zero_copy = _live_arch(case) == "sm_107a"
    route = _public_route(case)
    assert route.metadata_i64 is zero_copy, route
    _launch(case)  # all-int32 reference (also the one-time kernel compile/load)
    torch.cuda.synchronize()
    expected = _outputs_of(case)

    case["cu_seqlens"] = case["cu_seqlens"].to(cu_dtype)
    case["state_indices"] = case["state_indices"].to(slots_dtype)
    case["checkpoint_cu_starts"] = case["checkpoint_cu_starts"].to(checkpoints_dtype)
    casts = _record_metadata_casts(monkeypatch)
    _reset_outputs(case, initial_state)
    _launch(case)
    torch.cuda.synchronize()
    _assert_same_outputs(case, expected)
    # Validate casts even when the profiler cannot capture CUDA activity.
    if zero_copy:
        assert casts == [], casts
    else:
        assert torch.int64 in casts, casts

    _reset_outputs(case, initial_state)
    kernels = _device_kernel_names(lambda: _launch(case))
    _assert_same_outputs(case, expected)
    if zero_copy:
        assert len(kernels) == 1, kernels
    else:
        assert len(kernels) > 1, kernels


@pytest.mark.parametrize(
    "slots_dtype", (torch.int32, torch.int64), ids=("i32-slots", "i64-slots")
)
def test_public_cake_gdn_prefill_metadata_i64_single_sequence_with_sentinel_checkpoints(
    monkeypatch, slots_dtype
):
    """One sequence: int64 cu_seqlens, a one-element slot buffer, no checkpoints.

    ``checkpoint_cu_starts`` is absent, so the kernel receives the one-element
    int32 sentinel for it while ``cu_seqlens`` is int64 and ``state_indices`` is
    the one-element buffer in either width; the outputs match the all-int32 call
    bitwise and a ``metadata_i64`` record never takes the cast path.
    """

    case = _make_case(
        seq_lens=(421,),
        indexed=True,
        state_dtype=torch.bfloat16,
        state_pool_padding=0,
    )
    initial_state = case["initial_state"].clone()
    _launch(case)
    torch.cuda.synchronize()
    expected = _outputs_of(case)

    case["cu_seqlens"] = case["cu_seqlens"].to(torch.int64)
    case["state_indices"] = case["state_indices"].to(slots_dtype)
    assert case["state_indices"].numel() == 1
    casts = _record_metadata_casts(monkeypatch)
    _reset_outputs(case, initial_state)
    _launch(case)
    torch.cuda.synchronize()
    _assert_same_outputs(case, expected)
    if _public_route(case).metadata_i64:
        assert casts == [], casts
        _reset_outputs(case, initial_state)
        kernels = _device_kernel_names(lambda: _launch(case))
        _assert_same_outputs(case, expected)
        assert len(kernels) == 1, kernels
    else:
        assert torch.int64 in casts, casts


def test_public_cake_gdn_prefill_metadata_i64_non_indexed_int64_cu_seqlens(monkeypatch):
    """Without state slots the kernel gets the slot sentinel and int64 ``cu_seqlens``."""

    case = _make_case(seq_lens=(300, 212), state_dtype=torch.bfloat16)
    initial_state = case["initial_state"].clone()
    _launch(case)
    torch.cuda.synchronize()
    expected = _outputs_of(case)
    case["cu_seqlens"] = case["cu_seqlens"].to(torch.int64)
    casts = _record_metadata_casts(monkeypatch)
    _reset_outputs(case, initial_state)
    _launch(case)
    torch.cuda.synchronize()
    _assert_same_outputs(case, expected)
    if _public_route(case).metadata_i64:
        assert casts == [], casts
    else:
        assert casts == [torch.int64], casts


@torch.inference_mode()
def test_public_cake_gdn_prefill_absent_gates_reuse_one_cached_unit_tensor():
    """Calls without g and beta share one cached read-only all-ones gate tensor.

    The compiled kernels read both gates from memory (``UNIT_GATES`` stays off),
    so an absent gate used to be filled fresh on every call: one or two fill
    kernels ahead of the main kernel.  The adapter now keeps one FP32 ones tensor
    per device and ``[tokens, heads]`` shape and passes it for both gates.  The
    output must match explicit unit gates bitwise, stay bitwise-identical across
    launches, and the kernel must never write the shared tensor.

    The rows are stateless 64-token sequences; the exact host bound
    (``cu_seqlens_cpu``) lets the resolver reach the shipped single-chunk
    kernel (``_dvsplit_single_chunk``), the only stateless (3, 1) key the
    manifest ships.
    """
    from flashinfer import gdn_prefill as prefill_mod

    case = _make_case(
        seq_lens=(64,) * 4,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        gates=False,
    )
    assert case["g"] is None and case["beta"] is None
    total_tokens = sum(case["seq_lens"])
    num_o_heads = max(case["num_q_heads"], case["num_v_heads"])
    device = case["q"].device

    def launch(g, beta):
        return chunk_gated_delta_rule(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            g=g,
            beta=beta,
            scale=case["scale"],
            initial_state=None,
            output_final_state=False,
            cu_seqlens=case["cu_seqlens"],
            cu_seqlens_cpu=_host_cu_seqlens(case["seq_lens"]),
            use_qk_l2norm_in_kernel=False,
            output=case["output"],
            use_cp=False,
            backend="cake_gdn",
        )

    explicit_ones = torch.ones((total_tokens, num_o_heads), device=device)
    launch(explicit_ones, explicit_ones.clone())
    torch.cuda.synchronize()
    expected = case["output"].clone()

    prefill_mod._CAKE_GDN_UNIT_GATES.clear()
    case["output"].zero_()
    launch(None, None)
    torch.cuda.synchronize()
    first = case["output"].clone()
    assert torch.equal(first, expected), "absent gates must act as explicit ones"

    assert list(prefill_mod._CAKE_GDN_UNIT_GATES) == [
        (device.index, total_tokens, num_o_heads)
    ]
    gates = next(iter(prefill_mod._CAKE_GDN_UNIT_GATES.values()))
    assert gates.dtype == torch.float32
    assert gates.is_contiguous()
    assert tuple(gates.shape) == (total_tokens, num_o_heads)
    assert gates.device == device
    assert torch.equal(gates, torch.ones_like(gates))
    assert (
        prefill_mod._cake_gdn_unit_gates(device.index, total_tokens, num_o_heads)
        is gates
    )

    for _ in range(2):
        case["output"].zero_()
        launch(None, None)
        torch.cuda.synchronize()
        assert torch.equal(case["output"], first)
    assert len(prefill_mod._CAKE_GDN_UNIT_GATES) == 1
    assert torch.equal(gates, torch.ones_like(gates)), (
        "the kernel must not write the shared unit gates"
    )


def test_public_cake_gdn_prefill_is_cuda_graph_safe():
    case = _make_case(
        seq_lens=(128,),
        indexed=True,
        state_dtype=torch.bfloat16,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
        state_pool_padding=0,
    )
    initial_state = case["initial_state"].clone()
    expected_out, expected_state, _ = _reference(case, initial_state)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch(case)
    stream.synchronize()
    with torch.cuda.stream(stream):
        case["initial_state"].copy_(initial_state)
        case["output"].zero_()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _launch(case)
    graph.replay()
    stream.synchronize()
    torch.testing.assert_close(
        case["output"].float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        case["output_state"].float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize(
    "kwargs,invalid_value,expected_route_suffix",
    (
        (
            {
                "seq_lens": (128,),
                "indexed": True,
                "state_dtype": torch.bfloat16,
                "num_q_heads": 16,
                "num_k_heads": 16,
                "num_v_heads": 16,
                "state_pool_padding": 0,
            },
            -1,
            ".dvsplit",
        ),
        (
            # 2 * 3 seqs * 32 heads = 192 tiles: full-DV on SM100a/SM103a (148/160
            # active clusters), DV-split on SM107 (212) -- the suffix follows the arch.
            {
                "seq_lens": (128, 192, 64),
                "indexed": True,
                "state_dtype": torch.bfloat16,
                "num_q_heads": 32,
                "num_k_heads": 32,
                "num_v_heads": 32,
            },
            "upper",
            "auto",
        ),
        (
            # 2 * 4 seqs * 32 heads = 256 tiles: full-DV on every supported arch.
            {
                "seq_lens": (128, 192, 64, 96),
                "indexed": True,
                "state_dtype": torch.bfloat16,
                "num_q_heads": 32,
                "num_k_heads": 32,
                "num_v_heads": 32,
            },
            "upper",
            ".full_dv",
        ),
        (
            {
                "seq_lens": (52, 93, 15, 107, 72, 61, 21),
                "indexed": True,
                "state_dtype": torch.bfloat16,
                "checkpoint_every": 64,
                "state_pool_padding": 0,
            },
            "upper",
            ".dvsplit",
        ),
    ),
)
def test_public_cake_gdn_prefill_raw_abi_invalid_slot_is_noop(
    kwargs, invalid_value, expected_route_suffix
):
    if expected_route_suffix == "auto":
        expected_route_suffix = _expected_noncp_route_suffix(**kwargs)
    case = _make_case(**kwargs)
    case["output"].fill_(13.0)
    if case["state_checkpoints"].numel():
        case["state_checkpoints"].fill_(11.0)
    output_before = case["output"].clone()
    state_before = case["output_state"].clone()
    backing_before = case["backing"].clone()
    checkpoints_before = case["state_checkpoints"].clone()
    invalid_value = (
        int(case["initial_state"].shape[0])
        if invalid_value == "upper"
        else invalid_value
    )
    case["state_indices"].fill_(invalid_value)
    route = _launch_raw_indexed_prefill(case)
    torch.cuda.synchronize()
    assert route.route_id.endswith(expected_route_suffix), route.route_id
    torch.testing.assert_close(case["output"], output_before, atol=0, rtol=0)
    torch.testing.assert_close(case["output_state"], state_before, atol=0, rtol=0)
    torch.testing.assert_close(
        case["state_checkpoints"], checkpoints_before, atol=0, rtol=0
    )
    torch.testing.assert_close(case["backing"], backing_before, atol=0, rtol=0)


def test_public_cake_gdn_prefill_rejects_cpu_state_indices_before_launch():
    case = _make_case(
        seq_lens=(128,),
        indexed=True,
        state_dtype=torch.bfloat16,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
        state_pool_padding=0,
    )
    case["state_indices"] = case["state_indices"].cpu()
    stream = torch.cuda.Stream()
    with (
        pytest.raises(
            cake_gdn.CakeGDNUnsupportedError, match="all tensors on one CUDA device"
        ),
        torch.cuda.stream(stream),
    ):
        _launch(case)


def test_public_cake_gdn_prefill_rejects_mismatched_cuda_device_before_launch():
    if torch.cuda.device_count() < 2:
        pytest.skip("mixed-device rejection requires two visible CUDA devices")
    case = _make_case(
        seq_lens=(128,),
        indexed=True,
        state_dtype=torch.bfloat16,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
        state_pool_padding=0,
    )
    current = case["q"].device.index or 0
    other = (current + 1) % torch.cuda.device_count()
    case["state_indices"] = case["state_indices"].to(torch.device("cuda", other))
    stream = torch.cuda.Stream(device=case["q"].device)
    with (
        pytest.raises(
            cake_gdn.CakeGDNUnsupportedError, match="all tensors on one CUDA device"
        ),
        torch.cuda.stream(stream),
    ):
        _launch(case)


def _run_public_prefill_invalid_slot_child(case_name):
    case = _make_case(
        seq_lens=(128,),
        indexed=True,
        state_dtype=torch.bfloat16,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
        state_pool_padding=0,
    )
    if case_name.startswith("graph_"):
        initial_state = case["initial_state"].clone()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            _launch(case)
        stream.synchronize()
        with torch.cuda.stream(stream):
            case["initial_state"].copy_(initial_state)
            case["output"].zero_()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _launch(case)
        case["state_indices"][0] = (
            -1 if case_name == "graph_negative" else int(case["initial_state"].shape[0])
        )
        torch.cuda.synchronize()
        graph.replay()
        stream.synchronize()
        return
    case["state_indices"][0] = (
        -1 if case_name == "eager_negative" else int(case["initial_state"].shape[0])
    )
    _launch(case)
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "case_name", ("eager_negative", "eager_upper", "graph_negative", "graph_upper")
)
def test_public_cake_gdn_prefill_invalid_cuda_slots_fail_in_isolated_process(
    case_name,
):
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cake-gdn-invalid-slot",
            case_name,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "FLASHINFER_CAKE_GDN_VALIDATE_SLOTS": "1"},
    )
    combined = completed.stdout + completed.stderr
    device_asserted = any(
        marker in combined
        for marker in ("device-side assert", "CUDA error", "_assert_async_cuda_kernel")
    )
    exact_async_assertion = (
        "_assert_async_cuda_kernel" in combined
        and "GDN non-CP prefill state_indices must contain slots in" in combined
    )
    assert completed.returncode != 0 or exact_async_assertion, combined
    assert device_asserted, combined


if (
    __name__ == "__main__"
    and len(sys.argv) == 3
    and sys.argv[1] == "--cake-gdn-invalid-slot"
):
    _run_public_prefill_invalid_slot_child(sys.argv[2])


@pytest.mark.parametrize(
    "kwargs,gated",
    (
        (
            {
                "seq_lens": (64, 32),
                "io_dtype": torch.float16,
                "num_q_heads": 6,
                "num_v_heads": 2,
            },
            False,
        ),
        (
            {
                "seq_lens": (64, 32),
                "io_dtype": torch.float16,
                "num_q_heads": 6,
                "num_v_heads": 2,
            },
            True,
        ),
        (
            {
                "seq_lens": (128, 64),
                "indexed": True,
                "state_dtype": torch.bfloat16,
                "checkpoint_every": 64,
                "state_pool_padding": 0,
            },
            True,
        ),
    ),
    ids=(
        "ungated_f16_initial_final",
        "gated_f16_initial_final",
        "gated_indexed_checkpoints_i32",
    ),
)
def test_public_cake_gdn_prefill_repeated_calls_allocate_no_device_memory(
    kwargs, gated
):
    """A warm public call allocates and frees no device memory.

    The export engine rejects a delivered route whose second call allocates or
    frees device memory: one allocation per call is a launch-path cost the kernel
    does not pay and defeats CUDA Graph buffer reuse.  The route without ``g`` and
    ``beta`` reads FP32 unit gates; the host reuses a bounded cache of them instead
    of filling a new tensor per call.  Int32 checkpoint starts are passed through.
    """

    case = _make_case(**kwargs)
    reference_case = case
    if not gated:
        ones = torch.ones_like(case["g"])
        case["g"] = None
        case["beta"] = None
        reference_case = dict(case, g=ones, beta=ones)
    initial_state = (
        None if case["initial_state"] is None else case["initial_state"].clone()
    )
    expected_out, expected_state, expected_checkpoints = _reference(
        reference_case, initial_state
    )
    # The first call compiles/loads the variant and warms the host-side caches.
    _launch(case)
    torch.cuda.synchronize()
    if initial_state is not None:
        case["initial_state"].copy_(initial_state)
    case["output"].zero_()
    if case["state_checkpoints"].numel():
        case["state_checkpoints"].fill_(float("nan"))
    torch.cuda.synchronize()
    # Garbage left by earlier tests of this process must not be collected inside the measured call: a cyclic
    # GC pass there frees unrelated device blocks (84 frees with 0 allocations in one nightly shard).  Collect
    # it first, and keep the automatic collector off while the call runs.
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        before = torch.cuda.memory_stats()
        _launch(case)
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
    finally:
        if gc_was_enabled:
            gc.enable()
    allocated = after["allocation.all.allocated"] - before["allocation.all.allocated"]
    freed = after["allocation.all.freed"] - before["allocation.all.freed"]
    assert (allocated, freed) == (0, 0)
    torch.testing.assert_close(
        case["output"].float(), expected_out.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        case["output_state"].float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )
    if case["checkpoint_every"]:
        torch.testing.assert_close(
            case["state_checkpoints"].float(),
            expected_checkpoints.float(),
            atol=1e-2,
            rtol=1e-2,
        )
