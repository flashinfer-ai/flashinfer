# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in NT4 static scheduler with the original grouped FC1/FC2 work order.

A scheduler warp builds four 128-entry prefix/count arrays after routing counts
are ready. Direct searches replace per-work traversal through expert groups.
Work IDs, worker assignment, producer/consumer pipelines, and readiness peeks
are unchanged. This is not CLC and does not separate FC1 and FC2 launches.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, extract_mlir_values, new_from_mlir_values
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import (
    MoEFusedFc12PersistentTileScheduler as Base,
)


class GroupedLookupScheduler:
    @staticmethod
    def make_storage_struct(params, ext, **kwargs):
        base_storage = Base.make_storage_struct(params, ext)

        @cute.struct
        class Storage:
            base: base_storage
            metadata: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 512], 16]

        return Storage

    @staticmethod
    def create(
        params,
        block_idx,
        grid_dim,
        sched_storage,
        num_consumer_threads,
        ext,
        single_group=False,
    ):
        result = GroupedLookupScheduler()
        result.single_group = single_group
        result.base = Base.create(
            params, block_idx, grid_dim, sched_storage.base, num_consumer_threads, ext
        )
        result.metadata = cute.make_tensor(
            sched_storage.metadata.data_ptr(),
            cute.make_layout((128, 4), stride=(1, 128)),
        )
        result.next_id = Int32(block_idx[2])
        result.stride = Int32(grid_dim[2])
        return result

    @property
    def current_work(self):
        return self.base.current_work

    def make_consumer(self):
        return self.base.make_consumer()

    @cute.jit
    def internal_init(self, warp_idx, sched_warp_id):
        if warp_idx == sched_warp_id:
            lane = cute.arch.thread_idx()[0] % 32
            params = self.base.params
            m = params.cta_tile_shape_mnk[0]
            data_pad = params.token_padding_block
            sf_pad = params.sf_padding_block
            block_carry = Int32(0)
            data_carry = Int32(0)
            sf_carry = Int32(0)
            for chunk in cutlass.range_constexpr(4):
                expert = Int32(chunk * 32) + lane
                tokens = Int32(0)
                if cutlass.const_expr(params.expert_token_sizes is not None):
                    tokens = params.expert_token_sizes[expert]
                else:
                    previous = Int32(0)
                    if expert > Int32(0):
                        previous = params.expert_token_prefix_sum[expert - Int32(1)]
                    tokens = params.expert_token_prefix_sum[expert] - previous
                blocks = (tokens + Int32(m - 1)) // Int32(m)
                data = ((tokens + Int32(data_pad - 1)) // Int32(data_pad)) * Int32(
                    data_pad
                )
                sf = ((tokens + Int32(sf_pad - 1)) // Int32(sf_pad)) * Int32(sf_pad)
                for step in cutlass.range_constexpr(5):
                    offset = 1 << step
                    source = cutlass.max(lane - Int32(offset), Int32(0))
                    b = cute.arch.shuffle_sync(
                        blocks, offset=source, mask=0xFFFFFFFF, mask_and_clamp=31
                    )
                    d = cute.arch.shuffle_sync(
                        data, offset=source, mask=0xFFFFFFFF, mask_and_clamp=31
                    )
                    s = cute.arch.shuffle_sync(
                        sf, offset=source, mask=0xFFFFFFFF, mask_and_clamp=31
                    )
                    if lane >= Int32(offset):
                        blocks = blocks + b
                        data = data + d
                        sf = sf + s
                blocks = blocks + block_carry
                data = data + data_carry
                sf = sf + sf_carry
                self.metadata[expert, 0] = blocks
                self.metadata[expert, 1] = data
                self.metadata[expert, 2] = sf
                self.metadata[expert, 3] = tokens
                block_carry = cute.arch.shuffle_sync(
                    blocks, offset=31, mask=0xFFFFFFFF, mask_and_clamp=31
                )
                data_carry = cute.arch.shuffle_sync(
                    data, offset=31, mask=0xFFFFFFFF, mask_and_clamp=31
                )
                sf_carry = cute.arch.shuffle_sync(
                    sf, offset=31, mask=0xFFFFFFFF, mask_and_clamp=31
                )
            cute.arch.sync_warp()

    @cute.jit
    def upper_bound(self, value):
        lo = Int32(0)
        hi = Int32(128)
        while lo < hi:
            mid = (lo + hi) // Int32(2)
            if self.metadata[mid, 0] <= value:
                lo = mid + Int32(1)
            else:
                hi = mid
        return lo

    @cute.jit
    def gen_next_work(self):
        tile = self.next_id
        self.next_id = tile + self.stride
        params = self.base.params
        n1 = self.base._num_fc1_intermediate_blocks
        n2 = self.base._num_fc2_hidden_blocks
        total_blocks = self.metadata[127, 0]
        work = self.base._ext.WorkTileInfo(
            expert_idx=Int32(-1),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            cumulative_data_physical_row=Int32(0),
            cumulative_sf_physical_row=Int32(0),
            cumulative_token_block_count=Int32(0),
            valid_tokens_in_cta_cluster_tile=Int32(0),
            phase_and_peek=Int32(0),
            fc1_counter_index=Int32(0),
        )
        if tile < total_blocks * (n1 + n2):
            group_start = Int32(0)
            group_end = total_blocks
            if cutlass.const_expr(not self.single_group):
                target_blocks = (Int32(params.group_hint) + n1 - Int32(1)) // n1
                end_expert = cutlass.min(
                    self.upper_bound(target_blocks - Int32(1)), Int32(127)
                )
                group_end = self.metadata[end_expert, 0]
                while tile >= group_end * (n1 + n2):
                    group_start = group_end
                    end_expert = cutlass.min(
                        self.upper_bound(group_start + target_blocks - Int32(1)),
                        Int32(127),
                    )
                    group_end = self.metadata[end_expert, 0]
            relative = tile - group_start * (n1 + n2)
            fc1_tiles = (group_end - group_start) * n1
            phase = Int32(1)
            columns = n1
            if relative >= fc1_tiles:
                relative = relative - fc1_tiles
                columns = n2
                phase = Int32(2)
            absolute_block = group_start + relative // columns
            expert = self.upper_bound(absolute_block)
            first_block = Int32(0)
            data_row = Int32(0)
            sf_row = Int32(0)
            if expert > Int32(0):
                first_block = self.metadata[expert - Int32(1), 0]
                data_row = self.metadata[expert - Int32(1), 1]
                sf_row = self.metadata[expert - Int32(1), 2]
            tile_m = absolute_block - first_block
            tile_n = relative % columns
            m = params.cta_tile_shape_mnk[0]
            count = cutlass.min(
                cutlass.max(self.metadata[expert, 3] - tile_m * Int32(m), Int32(0)),
                Int32(m),
            )
            work = self.base._ext.WorkTileInfo(
                expert_idx=expert,
                tile_m_idx=tile_m,
                tile_n_idx=tile_n,
                cumulative_data_physical_row=data_row,
                cumulative_sf_physical_row=sf_row,
                cumulative_token_block_count=first_block,
                valid_tokens_in_cta_cluster_tile=(count << Int32(16)) | count,
                phase_and_peek=phase,
                fc1_counter_index=tile_m,
            )
        self.base.current_work = self.base._ext.enrich_work_tile_info(work)

    @cute.jit
    def publish_work(self):
        self.base.publish_work()

    @cute.jit
    def produce_tail(self):
        self.base.produce_tail()

    def __extract_mlir_values__(self):
        values = []
        for name in ("base", "metadata", "next_id", "stride"):
            values.extend(extract_mlir_values(getattr(self, name)))
        return values

    def __new_from_mlir_values__(self, values):
        result = GroupedLookupScheduler()
        result.single_group = self.single_group
        index = 0
        for name in ("base", "metadata", "next_id", "stride"):
            original = getattr(self, name)
            count = len(extract_mlir_values(original))
            setattr(
                result,
                name,
                new_from_mlir_values(original, values[index : index + count]),
            )
            index += count
        return result
