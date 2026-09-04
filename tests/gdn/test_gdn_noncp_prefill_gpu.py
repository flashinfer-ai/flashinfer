# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Direct public-API tests for the frozen GDN non-CP non-CP GDN prefill routes."""

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.jit import gdn_noncp as gdn_noncp


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
    v = (torch.randn((total, hv, 128), generator=generator, device="cuda") * 0.1).to(io_dtype)
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
            (torch.randn(initial_state.shape, generator=generator, device="cuda") * 0.05).to(
                state_dtype
            )
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
    checkpoint_counts = [length // checkpoint_every for length in seq_lens] if checkpoint_every else [0] * len(seq_lens)
    checkpoint_cu = torch.tensor(
        [0, *torch.tensor(checkpoint_counts).cumsum(0).tolist()],
        dtype=torch.int32,
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
        "g": alpha.contiguous(),
        "beta": beta.contiguous(),
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


def _launch(case):
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
        backend="gdn_noncp",
    )


def _launch_raw_indexed_prefill(case):
    """Invoke the exported TVM FFI entry without public-adapter validation."""

    major, minor = torch.cuda.get_device_capability(case["q"].device)
    arch = gdn_noncp.arch_for_compute_capability(major, minor)
    route = gdn_noncp.select_gdn_noncp_prefill_variant(
        arch=arch,
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
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=case["checkpoint_every"],
        use_state_indices=True,
        gates_present=True,
        seq_lens=case["seq_lens"],
    )
    entry = gdn_noncp.load_gdn_noncp_kernel(route.variant_name, arch)
    num_o_heads = max(case["num_q_heads"], case["num_v_heads"])
    total_tiles = len(case["seq_lens"]) * num_o_heads * (
        2 if route.route_id.endswith(".dvsplit") else 1
    )
    active_clusters = int(
        torch.cuda.get_device_properties(case["q"].device).multi_processor_count
    )
    if route.route_id.endswith(".dvsplit") or total_tiles <= 128:
        grid_x = min(active_clusters, total_tiles)
    else:
        max_chunks = max((length + 63) // 64 for length in case["seq_lens"])
        if max_chunks <= 8:
            grid_x = min(128, total_tiles)
        elif active_clusters in (148, 160) and total_tiles == 256:
            grid_x = 128
        else:
            grid_x = min(active_clusters, total_tiles)
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
    ),
)
def test_public_gdn_noncp_prefill_matches_independent_recurrence(kwargs):
    case = _make_case(**kwargs)
    initial_state = None if case["initial_state"] is None else case["initial_state"].clone()
    backing_before = None if case["backing"] is None else case["backing"].clone()
    expected_out, expected_state, expected_checkpoints = _reference(case, initial_state)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out, state = _launch(case)
    stream.synchronize()
    assert out is case["output"]
    assert state is case["output_state"]
    torch.testing.assert_close(out.float(), expected_out.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(state.float(), expected_state.float(), atol=1e-2, rtol=1e-2)
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
                case["backing"][slot * slot_stride + row_elems : (slot + 1) * slot_stride],
                backing_before[slot * slot_stride + row_elems : (slot + 1) * slot_stride],
                atol=0,
                rtol=0,
            )


def test_public_gdn_noncp_prefill_raw_ffi_bf16_indexed_checkpoint_b7_t421_noncompact_matches_independent_recurrence():
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


def test_public_gdn_noncp_prefill_is_cuda_graph_safe():
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
            {
                "seq_lens": (128, 192, 64),
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
def test_public_gdn_noncp_prefill_raw_abi_invalid_slot_is_noop(
    kwargs, invalid_value, expected_route_suffix
):
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


def test_public_gdn_noncp_prefill_rejects_cpu_state_indices_before_launch():
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
    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError, match="all tensors on one CUDA device"
    ), torch.cuda.stream(stream):
        _launch(case)


def test_public_gdn_noncp_prefill_rejects_mismatched_cuda_device_before_launch():
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
    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError, match="all tensors on one CUDA device"
    ), torch.cuda.stream(stream):
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
def test_public_gdn_noncp_prefill_invalid_cuda_slots_fail_in_isolated_process(case_name):
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--gdn-noncp-invalid-slot", case_name],
        capture_output=True,
        text=True,
        timeout=120,
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


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--gdn-noncp-invalid-slot":
    _run_public_prefill_invalid_slot_child(sys.argv[2])
