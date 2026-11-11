# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team

import importlib
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashinfer.comm.mnnvl_cutedsl.runtime import collective_support
from flashinfer.comm import (
    init_tp_moe_finalize_rmsnorm,
    supports_tp_moe_finalize_rmsnorm,
    tp_moe_finalize_rmsnorm,
)

TOPK = 8
HIDDEN_SIZE = 4096
HIDDEN_SIZE_PADDED = HIDDEN_SIZE + 32
EXTRA_ROWS = 17


def _assert_model_close(
    actual: torch.Tensor, expected: torch.Tensor, *, context: str = ""
) -> None:
    delta = actual.float() - expected.float()
    rel_l2 = delta.norm() / expected.float().norm().clamp_min(1e-12)
    max_abs = delta.abs().max()
    prefix = f"{context}: " if context else ""
    assert rel_l2 < 1e-3, f"{prefix}relative L2 error {rel_l2.item():.6g}"
    # One BF16 ULP for output magnitudes in [8, 16); relative L2 limits aggregate error.
    assert max_abs <= 0.0625, f"{prefix}maximum absolute error {max_abs.item():.6g}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _reference(
    gemm2_permuted: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    gain_center: float,
    weight_before_norm: bool,
) -> torch.Tensor:
    num_tokens = expanded_idx_to_permuted_idx.shape[0]
    indices = expanded_idx_to_permuted_idx.reshape(-1).long()
    valid = (indices >= 0) & (indices < gemm2_permuted.shape[0])
    rows = gemm2_permuted.index_select(
        0, indices.clamp(0, gemm2_permuted.shape[0] - 1)
    )[:, :HIDDEN_SIZE]
    rows = rows.view(num_tokens, TOPK, HIDDEN_SIZE)
    rows.masked_fill_(~valid.view(num_tokens, TOPK, 1), 0)
    # Bound oracle memory while enforcing rank order rather than NCCL's tree.
    reduced = torch.empty_like(rows)
    for start in range(0, num_tokens, 32):
        local = rows[start : start + 32].contiguous()
        peers = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(peers, local)
        total = torch.zeros_like(local, dtype=torch.float32)
        for peer in peers:
            total += peer.float()
        reduced[start : start + 32] = total.bfloat16()
    rows = reduced.float()
    if weight_before_norm:
        rows *= norm_weight.float() + gain_center
    inv_rms = torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + eps)
    normalized = rows * inv_rms
    if not weight_before_norm:
        normalized *= norm_weight.float() + gain_center
    result = torch.zeros_like(normalized[:, 0])
    for expert in range(TOPK):
        result += normalized[:, expert] * expert_weights[:, expert].float().unsqueeze(
            -1
        )
    return result.bfloat16()


def _rounded_reference(
    gemm2_permuted: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    gain_center: float,
    weight_before_norm: bool,
) -> torch.Tensor:
    """The obsolete contract with an expert-local BF16 normalization store."""
    num_tokens = expanded_idx_to_permuted_idx.shape[0]
    indices = expanded_idx_to_permuted_idx.reshape(-1).long()
    valid = (indices >= 0) & (indices < gemm2_permuted.shape[0])
    rows = gemm2_permuted.index_select(
        0, indices.clamp(0, gemm2_permuted.shape[0] - 1)
    )[:, :HIDDEN_SIZE]
    rows = rows.view(num_tokens, TOPK, HIDDEN_SIZE)
    rows.masked_fill_(~valid.view(num_tokens, TOPK, 1), 0)
    rows = rows.float()
    dist.all_reduce(rows)
    rows = rows.to(torch.bfloat16).float()
    if weight_before_norm:
        rows *= norm_weight.float() + gain_center
    inv_rms = torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (rows * inv_rms).to(torch.bfloat16)
    if not weight_before_norm:
        normalized = (normalized.float() * (norm_weight.float() + gain_center)).to(
            torch.bfloat16
        )
    return (
        normalized.float()
        .mul(expert_weights.float().unsqueeze(-1))
        .sum(dim=1)
        .to(torch.bfloat16)
    )


def _make_permuted(
    partial_rows: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter logical rows into a padded physical GEMM2 buffer.

    Row zero and all unused storage stay NaN so an invalid map entry catches
    any implementation that reads a fallback row and tries to mask it later.
    """
    num_tokens = partial_rows.shape[0]
    num_logical_rows = num_tokens * TOPK
    num_rows = num_logical_rows + EXTRA_ROWS
    gemm2_permuted = torch.full(
        (num_rows, HIDDEN_SIZE_PADDED),
        float("nan"),
        dtype=partial_rows.dtype,
        device=partial_rows.device,
    )
    generator = torch.Generator(device=partial_rows.device).manual_seed(seed)
    physical_rows = (
        torch.randperm(num_rows - 1, device=partial_rows.device, generator=generator)[
            :num_logical_rows
        ]
        + 1
    )
    gemm2_permuted[physical_rows, :HIDDEN_SIZE] = partial_rows.reshape(
        num_logical_rows, HIDDEN_SIZE
    )
    return gemm2_permuted, physical_rows.to(torch.int32).view(num_tokens, TOPK)


def _init_group(local_rank: int, master_port: int | None, world_size: int) -> None:
    torch.cuda.set_device(local_rank)
    if master_port is None:
        store = dist.TCPStore(
            os.environ["MASTER_ADDR"],
            int(os.environ["MASTER_PORT"]),
            world_size,
            False,
            timeout=timedelta(seconds=180),
        )
        store = dist.PrefixStore(os.environ["FLASHINFER_TP_TEST_CASE"], store)
        dist.init_process_group(
            "nccl", store=store, rank=int(os.environ["RANK"]), world_size=world_size
        )
    else:
        dist.init_process_group(
            "nccl",
            rank=local_rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{master_port}",
        )


def _run_env_worker(worker, args) -> None:
    # Isolate collectives so a stalled kernel becomes a failing test ID and
    # cannot strand the rest of the suite in the same CUDA context.
    os.environ["FLASHINFER_TP_TEST_CASE"] = repr((worker.__name__, args[2:]))
    process = mp.get_context("spawn").Process(target=worker, args=args)
    process.start()
    process.join(180)
    stalled = process.is_alive()
    if stalled:
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
    assert not stalled, "collective worker timed out after 180 seconds"
    assert process.exitcode == 0, f"collective worker exited with {process.exitcode}"


def _worker(local_rank: int, master_port: int | None, world_size: int) -> None:
    _init_group(local_rank, master_port, world_size)
    device = torch.device("cuda", local_rank)
    finalize_module = importlib.import_module("flashinfer.comm.tp_moe_finalize_rmsnorm")
    assert not collective_support(local_rank != 0, dist.group.WORLD, device), (
        "one rank's failed initialization verdict must disable every rank"
    )
    assert init_tp_moe_finalize_rmsnorm(backend="cute")
    assert torch.cuda.get_device_properties(device).multi_processor_count <= (
        finalize_module.NSLOT
    )
    execution_stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(execution_stream)
    contract_rows = torch.empty(
        1, TOPK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    contract_gemm2, contract_indices = _make_permuted(
        contract_rows, seed=100 + local_rank
    )
    contract_weights = torch.empty(1, TOPK, dtype=torch.bfloat16, device=device)
    contract_norm = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    contract_output = torch.empty(1, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    assert supports_tp_moe_finalize_rmsnorm(
        contract_gemm2,
        contract_weights,
        contract_indices,
        contract_norm,
        contract_output,
    )
    contract_weights_fp32 = contract_weights.float()
    assert supports_tp_moe_finalize_rmsnorm(
        contract_gemm2,
        contract_weights_fp32,
        contract_indices,
        contract_norm,
        contract_output,
    )
    assert not supports_tp_moe_finalize_rmsnorm(
        contract_gemm2,
        contract_weights.to(torch.float16),
        contract_indices,
        contract_norm,
        contract_output,
    )
    misaligned_rows = torch.empty(
        contract_gemm2.numel() + 1, dtype=torch.bfloat16, device=device
    )[1:].view_as(contract_gemm2)
    assert misaligned_rows.is_contiguous() and misaligned_rows.data_ptr() % 16 != 0
    assert not supports_tp_moe_finalize_rmsnorm(
        misaligned_rows,
        contract_weights,
        contract_indices,
        contract_norm,
        contract_output,
    )

    initialized_device = finalize_module._WS.dev_ord
    finalize_module._WS.dev_ord = (initialized_device + 1) % torch.cuda.device_count()
    try:
        assert not supports_tp_moe_finalize_rmsnorm(
            contract_gemm2,
            contract_weights,
            contract_indices,
            contract_norm,
            contract_output,
        )
    finally:
        finalize_module._WS.dev_ord = initialized_device

    for expert_weights_dtype in (torch.bfloat16, torch.float32):
        for weight_before_norm in (False, True):
            for num_tokens in (1, 17, 65):
                generator = torch.Generator(device=device).manual_seed(
                    1000 + local_rank
                )
                partial_rows = torch.randn(
                    num_tokens,
                    TOPK,
                    HIDDEN_SIZE,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                gemm2_permuted, expanded_idx_to_permuted_idx = _make_permuted(
                    partial_rows,
                    seed=3000 + 100 * num_tokens + local_rank,
                )
                shared = torch.Generator(device=device).manual_seed(2000)
                expert_weights = torch.randn(
                    num_tokens,
                    TOPK,
                    dtype=expert_weights_dtype,
                    device=device,
                    generator=shared,
                )
                norm_weight = torch.randn(
                    HIDDEN_SIZE,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=shared,
                )
                expected = _reference(
                    gemm2_permuted,
                    expanded_idx_to_permuted_idx,
                    expert_weights,
                    norm_weight,
                    1e-8,
                    1.0,
                    weight_before_norm,
                )
                output = torch.empty_like(expected)
                tp_moe_finalize_rmsnorm(
                    gemm2_permuted,
                    expert_weights,
                    expanded_idx_to_permuted_idx,
                    norm_weight,
                    output,
                    eps=1e-8,
                    gain_center=1.0,
                    weight_before_norm=weight_before_norm,
                )
                if num_tokens == 65 and not weight_before_norm:
                    identity_gemm2 = torch.full_like(gemm2_permuted, float("nan"))
                    identity_indices = torch.arange(
                        1,
                        num_tokens * TOPK + 1,
                        dtype=torch.int32,
                        device=device,
                    ).view(num_tokens, TOPK)
                    identity_gemm2[1 : num_tokens * TOPK + 1, :HIDDEN_SIZE] = (
                        partial_rows.view(-1, HIDDEN_SIZE)
                    )
                    identity_output = torch.empty_like(output)
                    tp_moe_finalize_rmsnorm(
                        identity_gemm2,
                        expert_weights,
                        identity_indices,
                        norm_weight,
                        identity_output,
                        eps=1e-8,
                        gain_center=1.0,
                        weight_before_norm=weight_before_norm,
                    )
                    torch.testing.assert_close(output, identity_output, atol=0, rtol=0)
                _assert_model_close(
                    output,
                    expected,
                    context=(
                        f"tokens={num_tokens}, weight_before_norm="
                        f"{weight_before_norm}, expert_weights_dtype="
                        f"{expert_weights_dtype}"
                    ),
                )

                gathered = [torch.empty_like(output) for _ in range(world_size)]
                dist.all_gather(gathered, output)
                assert all(
                    torch.equal(gathered[0], rank_output) for rank_output in gathered
                )

    other_stream = torch.cuda.Stream(device=device)
    # Ordered eager calls may reuse the workspace from a different stream.
    other_stream.wait_stream(execution_stream)
    with torch.cuda.stream(other_stream):
        output.fill_(float("nan"))
        tp_moe_finalize_rmsnorm(
            gemm2_permuted,
            expert_weights,
            expanded_idx_to_permuted_idx,
            norm_weight,
            output,
            eps=1e-8,
            gain_center=1.0,
            weight_before_norm=True,
        )
    execution_stream.wait_stream(other_stream)
    _assert_model_close(output, expected)

    # Invalid entries must contribute zero without touching row zero or another
    # padded row, both of which intentionally contain NaNs.
    invalid_generator = torch.Generator(device=device).manual_seed(6000 + local_rank)
    invalid_rows = torch.randn(
        17,
        TOPK,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=invalid_generator,
    )
    invalid_gemm2, invalid_indices = _make_permuted(
        invalid_rows, seed=6100 + local_rank
    )
    invalid_indices[0, 0] = -1
    invalid_indices[0, 1] = invalid_gemm2.shape[0]
    metadata_generator = torch.Generator(device=device).manual_seed(6200)
    invalid_weights = torch.randn(
        17,
        TOPK,
        dtype=torch.bfloat16,
        device=device,
        generator=metadata_generator,
    )
    invalid_norm = torch.randn(
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=metadata_generator,
    )
    for weight_before_norm in (False, True):
        expected = _reference(
            invalid_gemm2,
            invalid_indices,
            invalid_weights,
            invalid_norm,
            1e-8,
            1.0,
            weight_before_norm,
        )
        output = torch.empty_like(expected)
        tp_moe_finalize_rmsnorm(
            invalid_gemm2,
            invalid_weights,
            invalid_indices,
            invalid_norm,
            output,
            eps=1e-8,
            gain_center=1.0,
            weight_before_norm=weight_before_norm,
        )
        assert torch.isfinite(output).all()
        _assert_model_close(output, expected)

    # Isolate post-collective arithmetic by making the other three ranks
    # contribute exact zeros. This must prefer the FP32-until-final contract
    # over the obsolete expert-local BF16 round.
    rounding_tokens = 17
    shared = torch.Generator(device=device).manual_seed(7000)
    rank0_rows = torch.randn(
        rounding_tokens,
        TOPK,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=shared,
    )
    rounding_rows = rank0_rows if local_rank == 0 else torch.zeros_like(rank0_rows)
    rounding_gemm2, rounding_indices = _make_permuted(
        rounding_rows, seed=7300 + local_rank
    )
    rounding_weights = torch.randn(
        rounding_tokens,
        TOPK,
        dtype=torch.bfloat16,
        device=device,
        generator=shared,
    )
    rounding_norm = torch.randn(
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=shared,
    )
    for weight_before_norm in (False, True):
        expected = _reference(
            rounding_gemm2,
            rounding_indices,
            rounding_weights,
            rounding_norm,
            1e-8,
            1.0,
            weight_before_norm,
        )
        rounded = _rounded_reference(
            rounding_gemm2,
            rounding_indices,
            rounding_weights,
            rounding_norm,
            1e-8,
            1.0,
            weight_before_norm,
        )
        assert torch.count_nonzero(expected != rounded) > 0
        output = torch.empty_like(expected)
        tp_moe_finalize_rmsnorm(
            rounding_gemm2,
            rounding_weights,
            rounding_indices,
            rounding_norm,
            output,
            eps=1e-8,
            gain_center=1.0,
            weight_before_norm=weight_before_norm,
        )
        fp32_error = (output.float() - expected.float()).abs().mean()
        rounded_error = (output.float() - rounded.float()).abs().mean()
        assert fp32_error < rounded_error

    fp32_weights = torch.randn(
        rounding_tokens,
        TOPK,
        dtype=torch.float32,
        device=device,
        generator=shared,
    )
    expected_fp32_weights = _reference(
        rounding_gemm2,
        rounding_indices,
        fp32_weights,
        rounding_norm,
        1e-8,
        1.0,
        True,
    )
    expected_bf16_weights = _reference(
        rounding_gemm2,
        rounding_indices,
        fp32_weights.to(torch.bfloat16),
        rounding_norm,
        1e-8,
        1.0,
        True,
    )
    assert torch.count_nonzero(expected_fp32_weights != expected_bf16_weights) > 0
    fp32_weights_output = torch.empty_like(expected_fp32_weights)
    tp_moe_finalize_rmsnorm(
        rounding_gemm2,
        fp32_weights,
        rounding_indices,
        rounding_norm,
        fp32_weights_output,
        eps=1e-8,
        gain_center=1.0,
        weight_before_norm=True,
    )
    _assert_model_close(fp32_weights_output, expected_fp32_weights)
    fp32_error = (fp32_weights_output.float() - expected_fp32_weights.float()).norm()
    bf16_error = (fp32_weights_output.float() - expected_bf16_weights.float()).norm()
    assert fp32_error < bf16_error

    graph_rows_generator = torch.Generator(device=device).manual_seed(7000 + local_rank)
    graph_logical_rows = torch.randn(
        17,
        TOPK,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=graph_rows_generator,
    )
    graph_gemm2, graph_indices = _make_permuted(
        graph_logical_rows, seed=7100 + local_rank
    )
    graph_metadata_generator = torch.Generator(device=device).manual_seed(7200)
    graph_weights = torch.randn(
        17,
        TOPK,
        dtype=torch.bfloat16,
        device=device,
        generator=graph_metadata_generator,
    )
    graph_norm = torch.randn(
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=graph_metadata_generator,
    )
    graph_output = torch.empty(17, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    tp_moe_finalize_rmsnorm(
        graph_gemm2,
        graph_weights,
        graph_indices,
        graph_norm,
        graph_output,
        eps=1e-4,
        gain_center=1.0,
        weight_before_norm=True,
    )
    execution_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert torch.cuda.is_current_stream_capturing()
        capture_stream_handle = int(torch._C._cuda_getCurrentRawStream(local_rank))
        assert capture_stream_handle != execution_stream.cuda_stream
        tp_moe_finalize_rmsnorm(
            graph_gemm2,
            graph_weights,
            graph_indices,
            graph_norm,
            graph_output,
            eps=1e-4,
            gain_center=1.0,
            weight_before_norm=True,
        )
    for iteration in range(4):
        replay_stream = (other_stream, execution_stream)[iteration % 2]
        replay_stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(replay_stream)
        replay_generator = torch.Generator(device=device).manual_seed(
            8000 + 10 * iteration + local_rank
        )
        replay_rows = torch.randn(
            graph_logical_rows.shape,
            dtype=torch.bfloat16,
            device=device,
            generator=replay_generator,
        )
        replay_gemm2, replay_indices = _make_permuted(
            replay_rows,
            seed=9000 + 10 * iteration + local_rank,
        )
        graph_gemm2.copy_(replay_gemm2)
        graph_indices.copy_(replay_indices)
        expected = _reference(
            graph_gemm2,
            graph_indices,
            graph_weights,
            graph_norm,
            1e-4,
            1.0,
            True,
        )
        graph.replay()
        _assert_model_close(graph_output, expected)

    execution_stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(execution_stream)
    # Eager execution can resume after ordered graph replays on either stream.
    tp_moe_finalize_rmsnorm(
        graph_gemm2,
        graph_weights,
        graph_indices,
        graph_norm,
        graph_output,
        eps=1e-4,
        gain_center=1.0,
        weight_before_norm=True,
    )
    execution_stream.synchronize()
    dist.destroy_process_group()


def _allocation_failure_worker(
    local_rank: int, master_port: int | None, world_size: int, site: str
) -> None:
    _init_group(local_rank, master_port, world_size)
    import torch.distributed._symmetric_memory as symm_mem

    # Rank 0 fails either the symmetric workspace or the local FP32 scratch
    # (the only FP32 allocation during initialization).
    target = symm_mem if site == "workspace" else torch
    original_empty = target.empty
    if dist.get_rank() == 0:

        def fail(*args, **kwargs):
            if site == "workspace" or kwargs.get("dtype") == torch.float32:
                raise torch.OutOfMemoryError(f"injected {site} allocation failure")
            return original_empty(*args, **kwargs)

        target.empty = fail
    try:
        assert not init_tp_moe_finalize_rmsnorm(backend="cute")
    finally:
        target.empty = original_empty
        dist.destroy_process_group()


def _path_worker(
    local_rank: int,
    master_port: int | None,
    world_size: int,
    path: str,
    routing_fp32: bool,
    weight_before_norm: bool,
) -> None:
    _init_group(local_rank, master_port, world_size)
    module = importlib.import_module("flashinfer.comm.tp_moe_finalize_rmsnorm")
    selector = module._select_path
    crossover_fn = module._token_crossover
    try:
        if path == "token1":
            # the real selector, with the token-owner path from the first token
            module._token_crossover = lambda nsm: 1
        elif path != "auto":
            module._select_path = lambda *_: path
        # The CuTe paths are under test here (the selector and crossover are patched above); the Cake
        # backend has its own tests below.
        assert init_tp_moe_finalize_rmsnorm(backend="cute")
        crossover = module._token_crossover(module._WS.nsm)
        tokens = dict.fromkeys(
            n
            for n in (
                1,
                8,
                32,
                33,
                crossover - 1,
                crossover,
                crossover + 1,
                module._WS.nsm,
                module._WS.nsm + 1,
                512,
                1025,
                8192,
                8193,
            )
            if n > 0
        )
        # Revisit small counts after the multi-launch sizes: launches of other
        # sizes advance shared slots, which must not leave stale Lamport data.
        tokens = [*tokens, 33, 2, module._WS.nsm]
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        for num_tokens in tokens:
            shared = torch.Generator(device=device).manual_seed(11000 + num_tokens)
            local = torch.Generator(device=device).manual_seed(12000 + rank)
            partial = torch.randn(
                num_tokens,
                TOPK,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=device,
                generator=local,
            )
            # Include epsilon-sensitive rows and invalid slots on every shape.
            partial[0] *= 1e-4
            rows, indices = _make_permuted(partial, seed=13000 + rank)
            indices[0, 0] = -1
            indices[-1, -1] = rows.shape[0]
            weights = torch.rand(
                num_tokens,
                TOPK,
                device=device,
                generator=shared,
                dtype=torch.float32 if routing_fp32 else torch.bfloat16,
            )
            weights /= weights.sum(-1, keepdim=True)
            norm = torch.randn(
                HIDDEN_SIZE, device=device, dtype=torch.bfloat16, generator=shared
            )
            norm *= 0.1
            out = torch.empty(
                num_tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
            )
            gain = float(num_tokens % 2)

            def call():
                tp_moe_finalize_rmsnorm(
                    rows,
                    weights,
                    indices,
                    norm,
                    out,
                    eps=1e-8,
                    gain_center=gain,
                    weight_before_norm=weight_before_norm,
                )

            def check():
                expected = _reference(
                    rows, indices, weights, norm, 1e-8, gain, weight_before_norm
                )
                _assert_model_close(
                    out, expected, context=f"{path=} {num_tokens=} {rank=}"
                )
                rank0 = out.clone()
                dist.broadcast(rank0, src=0)
                assert torch.equal(out, rank0), (
                    f"rank disagreement: {path=} {num_tokens=}"
                )

            call()
            check()
            # The build falls back to the staged path when a fast path cannot
            # run; require the selected path so a silent fallback fails here.
            key = module._program_key(
                num_tokens, rows, weights, weight_before_norm, 1e-8, gain
            )
            built = module._PROGRAMS[key].path
            selected = module._select_path(num_tokens, module._WS.nsm)
            assert built == selected, f"{path=} {num_tokens=}: built {built}"
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for replay in range(3):
                partial.normal_(generator=local)
                partial[0] *= 1e-4
                new_rows, new_indices = _make_permuted(
                    partial, seed=14000 + rank + replay
                )
                new_indices[0, 0] = -1
                new_indices[-1, -1] = rows.shape[0]
                rows.copy_(new_rows)
                indices.copy_(new_indices)
                weights.uniform_(generator=shared)
                weights /= weights.sum(-1, keepdim=True)
                norm.normal_(generator=shared)
                norm *= 0.1
                graph.replay()
                check()
            if num_tokens == 33:
                norm.zero_()
                weights.fill_(1.0 / TOPK)
                if world_size >= 4:
                    # A tree reduction retains the unit that ordered FP32
                    # accumulation loses between the cancelling large rows.
                    partial.fill_((2**24, 1, -(2**24))[rank] if rank < 3 else 0)
                    new_rows, new_indices = _make_permuted(partial, seed=15000 + rank)
                    rows.copy_(new_rows)
                    indices.copy_(new_indices)
                    graph.replay()
                    check()
                partial.fill_(1 if rank == 0 else 0)
                new_rows, new_indices = _make_permuted(partial, seed=16000 + rank)
                rows.copy_(new_rows)
                indices.copy_(new_indices)
                weights.zero_()
                weights[:, 0] = 2**24
                weights[:, 1] = 1
                weights[:, 2] = -(2**24)
                graph.replay()
                check()
            del graph
    finally:
        module._select_path = selector
        module._token_crossover = crossover_fn
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("path", ["auto", "expert", "token", "token1"])
@pytest.mark.parametrize("routing_fp32", [False, True])
@pytest.mark.parametrize("weight_before_norm", [False, True])
def test_tp_moe_finalize_rmsnorm_paths(
    world_size: int, path: str, routing_fp32: bool, weight_before_norm: bool
) -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("TP MoE finalize requires SM100, SM103, or SM107")
    if "LOCAL_RANK" in os.environ:
        if int(os.environ["WORLD_SIZE"]) != world_size:
            pytest.skip("different torchrun world size")
        _run_env_worker(
            _path_worker,
            (
                int(os.environ["LOCAL_RANK"]),
                None,
                world_size,
                path,
                routing_fp32,
                weight_before_norm,
            ),
        )
    else:
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"{world_size} GPUs are required")
        mp.spawn(
            _path_worker,
            args=(_free_port(), world_size, path, routing_fp32, weight_before_norm),
            nprocs=world_size,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_tp_moe_finalize_rmsnorm_eager_and_cudagraph(world_size: int) -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("TP MoE finalize requires SM100, SM103, or SM107")
    if "LOCAL_RANK" in os.environ:
        if int(os.environ["WORLD_SIZE"]) != world_size:
            pytest.skip("different torchrun world size")
        _run_env_worker(_worker, (int(os.environ["LOCAL_RANK"]), None, world_size))
        return
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} GPUs are required")
    mp.spawn(_worker, args=(_free_port(), world_size), nprocs=world_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("site", ["workspace", "scratch"])
def test_tp_moe_finalize_rmsnorm_allocation_failure_is_rank_consistent(
    world_size: int, site: str
) -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("TP MoE finalize requires SM100, SM103, or SM107")
    if "LOCAL_RANK" in os.environ:
        if int(os.environ["WORLD_SIZE"]) != world_size:
            pytest.skip("different torchrun world size")
        _run_env_worker(
            _allocation_failure_worker,
            (int(os.environ["LOCAL_RANK"]), None, world_size, site),
        )
        return
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} GPUs are required")
    mp.spawn(
        _allocation_failure_worker,
        args=(_free_port(), world_size, site),
        nprocs=world_size,
    )


# --------------------------------------------------------------------------- the Cake backend


def _cake_routed(device: torch.device, world_size: int) -> bool:
    """Whether the Cake package routes this device's architecture at ``world_size``."""
    cake_jit = importlib.import_module(
        "flashinfer.cake_tp_moe_finalize_rmsnorm_backend.cake_jit"
    )
    arch = cake_jit.arch_for_capability(torch.cuda.get_device_capability(device))
    return arch is not None and cake_jit.route_available(
        arch, cake_jit.route_key(world_size)
    )


def _warm_cake_libraries() -> None:
    """Build the Cake libraries of this device's architecture once, outside the
    collective workers (whose first call would otherwise compile them under the
    worker timeout)."""
    cake_jit = importlib.import_module(
        "flashinfer.cake_tp_moe_finalize_rmsnorm_backend.cake_jit"
    )
    arch = cake_jit.arch_for_capability(torch.cuda.get_device_capability())
    if arch is None:
        return
    for spec in cake_jit.specs_for_arch(arch):
        spec.build_and_load()


def _cake_worker(
    local_rank: int,
    master_port: int | None,
    world_size: int,
    backend: str,
    routing_fp32: bool,
    weight_before_norm: bool,
) -> None:
    _init_group(local_rank, master_port, world_size)
    module = importlib.import_module("flashinfer.comm.tp_moe_finalize_rmsnorm")
    cake_jit = importlib.import_module(
        "flashinfer.cake_tp_moe_finalize_rmsnorm_backend.cake_jit"
    )
    try:
        device = torch.device("cuda", local_rank)
        routed = _cake_routed(device, world_size)
        available = init_tp_moe_finalize_rmsnorm(backend=backend)
        cake = module._WS.cake
        assert (cake is not None) == routed, (world_size, backend, routed)
        if backend == "cake":
            assert available == routed
            assert not module._WS.available, "backend='cake' prepares no CuTe workspace"
        else:
            assert available
        with pytest.raises(RuntimeError, match="already initialized with backend"):
            init_tp_moe_finalize_rmsnorm(backend="cute")
        rank = dist.get_rank()
        nsm = torch.cuda.get_device_properties(device).multi_processor_count
        tokens = [1, 2, 8, 17, 24, 31, 32, 33, 36, 37, 64, 65, 76, 128]
        tokens += [144, 145, 148, 149, 152, 153, 256, 1025, 8193, 33, 2, nsm]
        arch = cake_jit.arch_for_capability(torch.cuda.get_device_capability(device))
        for num_tokens in tokens:
            shared = torch.Generator(device=device).manual_seed(21000 + num_tokens)
            local = torch.Generator(device=device).manual_seed(22000 + rank)
            partial = torch.randn(
                num_tokens,
                TOPK,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=device,
                generator=local,
            )
            partial[0] *= 1e-4
            rows, indices = _make_permuted(partial, seed=23000 + rank)
            indices[0, 0] = -1
            indices[-1, -1] = rows.shape[0]
            weights = torch.rand(
                num_tokens,
                TOPK,
                device=device,
                generator=shared,
                dtype=torch.float32 if routing_fp32 else torch.bfloat16,
            )
            weights /= weights.sum(-1, keepdim=True)
            norm = torch.randn(
                HIDDEN_SIZE, device=device, dtype=torch.bfloat16, generator=shared
            )
            norm *= 0.1
            out = torch.empty(
                num_tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
            )
            gain = float(num_tokens % 2)

            def call():
                tp_moe_finalize_rmsnorm(
                    rows,
                    weights,
                    indices,
                    norm,
                    out,
                    eps=1e-8,
                    gain_center=gain,
                    weight_before_norm=weight_before_norm,
                    backend=backend,
                )

            def check():
                expected = _reference(
                    rows, indices, weights, norm, 1e-8, gain, weight_before_norm
                )
                _assert_model_close(
                    out, expected, context=f"{backend=} {num_tokens=} {rank=}"
                )
                rank0 = out.clone()
                dist.broadcast(rank0, src=0)
                assert torch.equal(out, rank0), (
                    f"rank disagreement: {backend=} {num_tokens=}"
                )

            if backend == "cake" and not routed:
                with pytest.raises(RuntimeError, match="Cake TP MoE finalize"):
                    call()
                break
            call()
            check()
            if cake is not None:
                # the routed Cake program ran: the record whose window admits the
                # count, at the grid the agreed capacity allows
                selected = cake.select(num_tokens)
                assert selected is not None, (num_tokens, arch, world_size)
                assert cake.last == selected, (cake.last, selected)
                record = cake_jit.MODULES[selected[0]]
                assert cake_jit.admits(record["select_rule"], num_tokens, arch)
                cake.last = None
            else:
                # backend="auto" on an (architecture, world size) without Cake
                # programs: the CuTe path ran and compiled its specialization
                key = module._program_key(
                    num_tokens, rows, weights, weight_before_norm, 1e-8, gain
                )
                assert key in module._PROGRAMS, num_tokens
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for replay in range(3):
                partial.normal_(generator=local)
                partial[0] *= 1e-4
                new_rows, new_indices = _make_permuted(
                    partial, seed=24000 + rank + replay
                )
                new_indices[0, 0] = -1
                new_indices[-1, -1] = rows.shape[0]
                rows.copy_(new_rows)
                indices.copy_(new_indices)
                weights.uniform_(generator=shared)
                weights /= weights.sum(-1, keepdim=True)
                norm.normal_(generator=shared)
                norm *= 0.1
                graph.replay()
                check()
            if num_tokens == 33:
                norm.zero_()
                weights.fill_(1.0 / TOPK)
                partial.fill_(1 if rank == 0 else 0)
                new_rows, new_indices = _make_permuted(partial, seed=26000 + rank)
                rows.copy_(new_rows)
                indices.copy_(new_indices)
                weights.zero_()
                weights[:, 0] = 2**24
                weights[:, 1] = 1
                weights[:, 2] = -(2**24)
                graph.replay()
                check()
            del graph
        if cake is not None and backend == "cake":
            assert not module._PROGRAMS, "backend='cake' compiled a CuTe specialization"
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("backend", ["cake", "auto"])
@pytest.mark.parametrize("routing_fp32", [False, True])
@pytest.mark.parametrize("weight_before_norm", [False, True])
def test_tp_moe_finalize_rmsnorm_cake_backend(
    world_size: int, backend: str, routing_fp32: bool, weight_before_norm: bool
) -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("TP MoE finalize requires SM100, SM103, or SM107")
    if "LOCAL_RANK" in os.environ:
        if int(os.environ["WORLD_SIZE"]) != world_size:
            pytest.skip("different torchrun world size")
        _warm_cake_libraries()
        _run_env_worker(
            _cake_worker,
            (
                int(os.environ["LOCAL_RANK"]),
                None,
                world_size,
                backend,
                routing_fp32,
                weight_before_norm,
            ),
        )
    else:
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"{world_size} GPUs are required")
        _warm_cake_libraries()
        mp.spawn(
            _cake_worker,
            args=(_free_port(), world_size, backend, routing_fp32, weight_before_norm),
            nprocs=world_size,
        )


def test_tp_moe_finalize_rmsnorm_backend_keyword() -> None:
    module = importlib.import_module("flashinfer.comm.tp_moe_finalize_rmsnorm")
    rows = torch.zeros(1, HIDDEN_SIZE, dtype=torch.bfloat16)
    weights = torch.zeros(1, TOPK, dtype=torch.bfloat16)
    indices = torch.zeros(1, TOPK, dtype=torch.int32)
    norm = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16)
    out = torch.zeros(1, HIDDEN_SIZE, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="backend must be one of"):
        init_tp_moe_finalize_rmsnorm(backend="fast")
    with pytest.raises(ValueError, match="backend must be one of"):
        supports_tp_moe_finalize_rmsnorm(
            rows, weights, indices, norm, out, backend="fast"
        )
    with pytest.raises(ValueError, match="backend must be one of"):
        tp_moe_finalize_rmsnorm(
            rows,
            weights,
            indices,
            norm,
            out,
            eps=1e-8,
            gain_center=0.0,
            weight_before_norm=False,
            backend="fast",
        )
    if module._WS.initialized:
        pytest.skip("the workspace is initialized in this process")
    for backend in ("auto", "cake", "cute"):
        assert not supports_tp_moe_finalize_rmsnorm(
            rows, weights, indices, norm, out, backend=backend
        )
    with pytest.raises(RuntimeError, match="not prepared for this group"):
        tp_moe_finalize_rmsnorm(
            rows,
            weights,
            indices,
            norm,
            out,
            eps=1e-8,
            gain_center=0.0,
            weight_before_norm=False,
            backend="cake",
        )
