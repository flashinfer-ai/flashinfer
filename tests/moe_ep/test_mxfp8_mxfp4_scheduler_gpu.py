# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device permutation, pool prefixes and decoded records against the host model."""

import random

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.device_scheduler import (
    DeviceSchedulerParams,
    PersistentDeviceScheduler,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.scheduler import (
    PersistentFc1Fc2Scheduler,
    SchedulerConfig,
)


class _SchedulerProbe:
    def __init__(self, experts, order, group_hint):
        self.experts = experts
        self.order = order
        self.group_hint = group_hint

    @cute.jit
    def __call__(self, counts, counters, tables, records, stream):
        params = DeviceSchedulerParams(
            expert_token_sizes=counts,
            expert_count=self.experts,
            gate_up_size=512,
            hidden_size=256,
            cta_tile_tokens=64,
            cta_tile_features=128,
            cluster_shape_token_feature=(1, 1),
            group_hint=self.group_hint,
            token_padding_block=64,
            sf_padding_block=128,
            fc1_ready_counter_ptr=None,
            fc1_done_counter_ptr=counters.iterator,
            fc2_ready_threshold=4,
            claim_order=self.order,
        )
        self.kernel(params, tables, records).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, params, tables, records):
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(
            PersistentDeviceScheduler.make_storage_struct(params)
        )
        scheduler = PersistentDeviceScheduler.create(
            params, cute.arch.block_idx(), cute.arch.grid_dim(), storage, 32
        )
        scheduler.internal_init(cutlass.Int32(0), 0)
        lane = cute.arch.lane_idx()
        for expert in cutlass.range(lane, self.experts, 32):
            tables[0, expert] = scheduler._order[expert]
            tables[1, expert] = scheduler._pool_data_row[expert]
            tables[2, expert] = scheduler._pool_sf_row[expert]
            tables[3, expert] = scheduler._pool_token_block_row[expert]
        scheduler.gen_next_work()  # consume internal_init's pending first claim
        for index in cutlass.range(cute.size(records, mode=[0])):
            words = scheduler.current_work.to_rmem()
            if lane == 0:
                for word in cutlass.range_constexpr(8):
                    records[index, word] = words[word]
            scheduler.gen_next_work()


@pytest.mark.parametrize(
    "experts,empty", [(7, False), (32, False), (33, False), (64, False), (33, True)]
)
@pytest.mark.parametrize("order", ["expert", "longest_first"])
@pytest.mark.parametrize("group_hint", [5, 4096])
def test_device_scheduler_tables_and_records(experts, empty, order, group_hint):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell")
    rng = random.Random(811 + experts)
    counts = tuple(rng.choice([0, 0, 17, 33, 65, 257]) for _ in range(experts))
    if empty:
        counts = (0,) * experts
    host = PersistentFc1Fc2Scheduler(
        SchedulerConfig(
            expert_token_counts=counts,
            gate_up_size=512,
            hidden_size=256,
            cta_tile_m=64,
            cta_tile_n=128,
            cluster_shape_mn=(1, 1),
            group_hint=group_hint,
            token_padding_block=64,
            sf_padding_block=128,
            num_persistent_clusters=1,
            fc2_ready_threshold=4,
            wait_for_dispatch=False,
            claim_order=order,
        )
    )
    device_counts = torch.tensor(counts, device="cuda", dtype=torch.int32)
    counters = torch.full(
        (sum((n + 63) // 64 for n in counts) + 1,), 4, device="cuda", dtype=torch.int32
    )
    tables = torch.empty((4, experts), device="cuda", dtype=torch.int32)
    records = torch.empty(
        (host.cluster_tile_count + 1, 8), device="cuda", dtype=torch.int32
    )
    views = [from_dlpack(t) for t in (device_counts, counters, tables, records)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    run = cute.compile(_SchedulerProbe(experts, order, group_hint), *views, stream)
    run(*views, stream)
    expected_tables = [list(host.order)] + [
        [getattr(e, field) for e in host._experts]
        for field in ("data_row", "sf_row", "token_block_row")
    ]
    assert tables.cpu().tolist() == expected_tables
    expected = [
        list(
            host.enrich_readiness_peek(
                host.record_for(i), fc1_done_counters=[4] * counters.numel()
            ).to_words()
        )
        for i in range(host.cluster_tile_count + 1)
    ]
    assert records.cpu().tolist() == expected
