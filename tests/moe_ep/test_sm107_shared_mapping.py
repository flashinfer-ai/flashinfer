# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Local shared tasks must not index the routed expert-count array."""

import math

import pytest
import torch


@pytest.mark.arch_rubin
@pytest.mark.parametrize("experts", [31, 32, 33])
@pytest.mark.parametrize("shared_tokens", [None, 0, 65])
def test_shared_task_mapping_replay(experts, shared_tokens):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg

    pkg.require_sm107_dsl()
    pkg.bootstrap_paths()
    from sources.kernel_src.schedulers.fc12_mapping import (
        create_fc12_task_mapping_state,
        map_fc12_linear_work_id,
    )

    @cute.kernel
    def map_tasks(sizes: cute.Tensor, output: cute.Tensor):
        worker, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        state = create_fc12_task_mapping_state(
            expert_count=experts,
            intermediate_gateup_size=256,
            hidden_size=128,
            mapping_cta_tile_shape_mnk=(64, 128, 128),
            mapping_cluster_shape_mn=(1, 1),
            group_hint=4,
            token_padding_block=64,
            sf_padding_block=128,
            is_swap_ab=True,
            expert_token_sizes=sizes,
            expert_token_prefix_sum=None,
            shared_token_count=shared_tokens,
        )
        for work_id in range(worker, output.shape[0], 3):
            tile, state = map_fc12_linear_work_id(cutlass.Int32(work_id), (0, 0), state)
            fields = tile.to_rmem()
            if lane == 0:
                for field in cutlass.range_constexpr(8):
                    output[work_id, field] = fields[field]

    @cute.jit
    def launch(sizes: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        map_tasks(sizes, output).launch(grid=(3, 1, 1), block=(32, 1, 1), stream=stream)

    sizes = torch.zeros(experts, device="cuda", dtype=torch.int32)
    # 131 rows need three token tiles, each with two FC1 tiles and one FC2 tile.
    output = torch.empty(((experts + 1) * 9 + 3, 8), device="cuda", dtype=torch.int32)
    args = (from_dlpack(sizes), from_dlpack(output))
    executable = cute.compile(
        launch, *args, cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )
    executable(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            executable(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        for counts in (
            [131] * experts,
            [i % 3 * 65 for i in range(experts)],
            [0] * experts,
        ):
            sizes.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
            output.fill_(-99)
            graph.replay()
            torch.cuda.synchronize()
            rows = output.cpu().tolist()
            counts = counts + ([] if shared_tokens is None else [shared_tokens])
            expected = set()
            data_start = scale_start = token_start = 0
            for expert, count in enumerate(counts):
                for phase, columns in ((1, 2), (2, 1)):
                    for token in range(math.ceil(count / 64)):
                        for column in range(columns):
                            expected.add(
                                (
                                    expert,
                                    column,
                                    token,
                                    data_start,
                                    scale_start,
                                    token_start,
                                    min(64, count - token * 64),
                                    phase,
                                )
                            )
                data_start += math.ceil(count / 64) * 64
                scale_start += math.ceil(count / 128) * 128
                token_start += math.ceil(count / 64)
            live = [tuple(row) for row in rows if row[0] >= 0]
            assert len(live) == len(expected) and set(live) == expected
            assert all(row[0] == -1 for row in rows[len(expected) :])
            for expert in range(len(counts)):
                fc1 = [
                    index
                    for index, row in enumerate(rows)
                    if row[0] == expert and row[7] == 1
                ]
                fc2 = [
                    index
                    for index, row in enumerate(rows)
                    if row[0] == expert and row[7] == 2
                ]
                if fc1:
                    assert max(fc1) < min(fc2)
    finally:
        graph.reset()


@pytest.mark.arch_rubin
@pytest.mark.parametrize("experts", [31, 32, 33])
@pytest.mark.parametrize("has_shared", [False, True])
def test_shared_phase_streams_replay(experts, has_shared):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg

    pkg.require_sm107_dsl()
    pkg.bootstrap_paths()
    from sources.kernel_src.schedulers.fc12_mapping import (
        create_phase_interleaved_fc12_mapping_state,
        map_phase_interleaved_fc12_work_id,
        resolve_phase_interleaved_fc1_claim_target,
    )

    @cute.kernel
    def map_tasks(
        sizes: cute.Tensor,
        shared: cute.Tensor,
        output: cute.Tensor,
        target: cute.Tensor,
    ):
        worker, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        state = create_phase_interleaved_fc12_mapping_state(
            expert_count=experts,
            intermediate_gateup_size=256,
            hidden_size=128,
            mapping_cta_tile_shape_mnk=(64, 128, 128),
            mapping_cluster_shape_mn=(1, 1),
            token_padding_block=64,
            sf_padding_block=128,
            is_swap_ab=True,
            expert_token_sizes=sizes,
            expert_token_prefix_sum=None,
            shared_token_count=shared[0] if has_shared else None,
        )
        claim_target, exhausted = resolve_phase_interleaved_fc1_claim_target(
            cutlass.Int32(7), state
        )
        if worker == 0 and lane == 0:
            target[0] = claim_target
            target[1] = cutlass.Int32(exhausted)
        for stream in cutlass.range_constexpr(2):
            for work_id in range(worker, output.shape[1], 3):
                tile, _, state = map_phase_interleaved_fc12_work_id(
                    cutlass.Int32(work_id), cutlass.Int32(stream + 1), (0, 0), state
                )
                fields = tile.to_rmem()
                if lane == 0:
                    for field in cutlass.range_constexpr(8):
                        output[stream, work_id, field] = fields[field]

    @cute.jit
    def launch(
        sizes: cute.Tensor,
        shared: cute.Tensor,
        output: cute.Tensor,
        target: cute.Tensor,
        stream: cuda.CUstream,
    ):
        map_tasks(sizes, shared, output, target).launch(
            grid=(3, 1, 1), block=(32, 1, 1), stream=stream
        )

    sizes = torch.zeros(experts, device="cuda", dtype=torch.int32)
    shared = torch.zeros(1, device="cuda", dtype=torch.int32)
    output = torch.empty(
        (2, (experts + 1) * 6 + 3, 8), device="cuda", dtype=torch.int32
    )
    target = torch.empty(2, device="cuda", dtype=torch.int32)
    args = tuple(from_dlpack(t) for t in (sizes, shared, output, target))
    executable = cute.compile(
        launch, *args, cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )
    executable(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            executable(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        for counts, shared_count in (
            ([131] * experts, 129),
            ([0] * experts, 65),
            ([0] * experts, 0),
            ([i % 3 * 65 for i in range(experts)], 64),
            ([0] * experts, 1),
            ([131] * experts, 129),
        ):
            sizes.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
            shared.fill_(shared_count)
            output.fill_(-99)
            graph.replay()
            torch.cuda.synchronize()
            counts = counts + ([shared_count] if has_shared else [])
            for stream, columns in enumerate((2, 1)):
                expected = []
                data_start = scale_start = token_start = 0
                for expert, count in enumerate(counts):
                    for token in range(math.ceil(count / 64)):
                        for column in range(columns):
                            expected.append(
                                [
                                    expert,
                                    column,
                                    token,
                                    data_start,
                                    scale_start,
                                    token_start,
                                    min(64, count - token * 64),
                                    stream + 1,
                                ]
                            )
                    data_start += math.ceil(count / 64) * 64
                    scale_start += math.ceil(count / 128) * 128
                    token_start += math.ceil(count / 64)
                rows = output[stream].cpu().tolist()
                assert rows[: len(expected)] == expected
                assert all(row[0] == -1 for row in rows[len(expected) :])
            fc1_tiles = sum(math.ceil(count / 64) * 2 for count in counts)
            assert target.cpu().tolist() == [min(7, fc1_tiles), int(fc1_tiles < 7)]
    finally:
        graph.reset()
