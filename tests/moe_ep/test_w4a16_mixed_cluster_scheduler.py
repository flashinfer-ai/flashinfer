# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Audit real W4A16 scheduler work coverage without GEMM or EP communication."""

import pytest
import torch


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("load_balance_mode", ["static", "atomic_counter"])
@pytest.mark.parametrize("launch_shape", ["preferred", "fallback", "mixed"])
@pytest.mark.parametrize("group_hint", [1, 5])
def test_mixed_cluster_scheduler_coverage(load_balance_mode, launch_shape, group_hint):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Requires Blackwell preferred-cluster support")

    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.cute_dsl.megamoe.bf16_nvfp4.custom_ext import (
        W4A16Fc12WorkTileInfo,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.bf16_nvfp4.fc1_fc2_fuse_sched import (
        MoEFusedFc12SchedulerParams,
    )

    launch_clusters = 4
    if launch_shape == "mixed":
        # Use the production occupancy recipe, rather than a tiny grid that
        # normally fits entirely as preferred clusters. Reserve >half the SM's
        # shared memory below to keep this audit at one resident CTA per SM.
        hardware = utils.HardwareInfo()
        preferred_count = int(hardware.get_max_active_clusters(4))
        fallback_capacity = int(hardware.get_max_active_clusters(2)) * 2
        fallback_count = ((fallback_capacity - preferred_count * 4) // 4) * 2
        launch_clusters = preferred_count + fallback_count // 2

    class IdentityExtension:
        WorkTileInfo = W4A16Fc12WorkTileInfo

        @cute.jit
        def enrich_work_tile_info(self, work):
            return work

    class SchedulerAudit:
        @cute.jit
        def __call__(self, sizes, counters, tokens, records, active_shapes, stream):
            ext = IdentityExtension()
            params = MoEFusedFc12SchedulerParams(
                scenario="2Dx3D",
                expert_shape=(8, 768, 384),
                cta_tile_shape_mnk=(128, 64, 256),
                cluster_shape_mn=(4, 1),
                group_hint=group_hint,
                token_padding_block=128,
                expert_token_sizes=sizes,
                load_balance_mode=load_balance_mode,
                load_balance_counter_ptr=counters.iterator,
                fallback_cluster_shape_mn=(2, 1),
                fallback_cluster_count=2 * launch_clusters,
                fallback_registration_counter_ptr=counters.iterator + 1,
                fallback_group_token_ptr=tokens.iterator,
            )
            self.storage_cls = params.get_scheduler_type().make_storage_struct(
                params, ext=ext
            )
            kernel = self.kernel(params, records, active_shapes)
            if cutlass.const_expr(launch_shape == "mixed"):
                kernel.launch(
                    grid=(4, 1, launch_clusters),
                    block=(32, 1, 1),
                    cluster=(4, 1, 1),
                    fallback_cluster=(2, 1, 1),
                    stream=stream,
                )
            else:
                cluster_m = 4 if launch_shape == "preferred" else 2
                kernel.launch(
                    grid=(4, 1, launch_clusters),
                    block=(32, 1, 1),
                    cluster=(cluster_m, 1, 1),
                    stream=stream,
                )

        @cute.kernel
        def kernel(self, params, records: cute.Tensor, active_shapes: cute.Tensor):
            smem = utils.SmemAllocator()
            storage = smem.allocate(self.storage_cls)
            residency = smem.allocate_tensor(
                cutlass.Int32, cute.make_layout((32768,)), byte_alignment=128
            )
            tidx = cute.arch.thread_idx()[0]
            residency[32736 + tidx] = tidx
            active_m = cute.arch.block_in_cluster_dim()[0]
            scheduler = params.get_scheduler_type().create(
                params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                sched_storage=storage,
                num_consumer_threads=32,
                ext=IdentityExtension(),
                active_cluster_shape_mn=(active_m, cutlass.Int32(1)),
            )
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            scheduler.internal_init(warp_idx=cutlass.Int32(0), sched_warp_id=0)
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
                work = scheduler.current_work
                if cute.arch.thread_idx()[0] == 0:
                    coord = (
                        work.phase - 1,
                        work.expert_idx,
                        work.tile_n_idx,
                        work.tile_m_idx,
                        0,
                    )
                    offset = cute.crd2idx(coord, records.layout)
                    cute.arch.atomic_add(records.iterator + offset, cutlass.Int32(1))
                    records[(*coord[:4], 1)] = work.valid_tokens_in_cta_tile
                    records[(*coord[:4], 2)] = work.cumulative_data_physical_row
                    records[(*coord[:4], 3)] = work.cumulative_token_block_count
                scheduler.gen_next_work()
            if cute.arch.thread_idx()[0] == 0:
                bx, _, bz = cute.arch.block_idx()
                active_shapes[bx + 4 * bz] = active_m + residency[32737] - 1
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    # Both phases contain a partial preferred channel group. Zero-token experts,
    # partial token tiles, and group boundaries exercise the persistent decoder.
    patterns = (
        [0, 1, 63, 64, 65, 129, 0, 193],
        [0] * 8,
        [193, 0, 129, 65, 64, 63, 1, 0],
    )
    sizes = torch.empty(8, device="cuda", dtype=torch.int32)
    counters = torch.zeros(2, device="cuda", dtype=torch.int32)
    tokens = torch.zeros(2 * launch_clusters, device="cuda", dtype=torch.int64)
    records = torch.zeros((2, 8, 4, 8, 4), device="cuda", dtype=torch.int32)
    active_shapes = torch.zeros(4 * launch_clusters, device="cuda", dtype=torch.int32)
    args = tuple(
        from_dlpack(t, assumed_align=16)
        for t in (sizes, counters, tokens, records, active_shapes)
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    executable = cute.compile(SchedulerAudit(), *args, stream)
    for pattern in patterns:
        sizes.copy_(torch.tensor(pattern, device="cuda", dtype=torch.int32))
        counters.zero_()
        tokens.zero_()
        records.zero_()
        executable(*args, stream)
        torch.cuda.synchronize()
        expected = torch.zeros_like(records, device="cpu")
        cumulative_rows = 0
        cumulative_token_tiles = 0
        for expert, count in enumerate(pattern):
            token_tiles = (count + 63) // 64
            for phase, channels in enumerate((768, 384)):
                # Padded channel CTAs must still consume work to satisfy the
                # active-cluster TMA pipeline. GEMM epilogues mask their output.
                channel_tiles = ((channels + 511) // 512) * 4
                for token_tile in range(token_tiles):
                    expected[phase, expert, token_tile, :channel_tiles] = torch.tensor(
                        [
                            1,
                            min(64, count - token_tile * 64),
                            cumulative_rows,
                            cumulative_token_tiles,
                        ],
                        dtype=torch.int32,
                    )
            cumulative_rows += ((count + 127) // 128) * 128
            cumulative_token_tiles += token_tiles
        torch.testing.assert_close(records.cpu(), expected, rtol=0, atol=0)
        shapes = set(active_shapes.cpu().tolist())
        assert shapes <= {2, 4}
        if launch_shape != "mixed":
            assert shapes == ({4} if launch_shape == "preferred" else {2})
    shape_counts = {
        size: int((active_shapes == size).sum().item()) // size for size in (2, 4)
    }
    print(
        f"W4A16 scheduler {load_balance_mode=} {launch_shape=} {group_hint=}: "
        f"physical cluster counts={shape_counts}"
    )
