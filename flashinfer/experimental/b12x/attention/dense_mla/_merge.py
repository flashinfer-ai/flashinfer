"""Base-2 split reduction for standalone dense MLA."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from b12x._lib.intrinsics import (
    bfloat2_to_float2_scaled,
    get_ptr_as_int64,
    ld_global_v2_u32,
    pack_f32x2_to_bfloat2,
    st_global_v2_u32,
)

from ._math import exp2_approx, log2_approx


class DenseMlaMergeKernel:
    """Merge normalized BF16 split partials and emit natural-log LSE."""

    def __init__(self, num_splits: int, value_dim: int):
        self.num_splits = int(num_splits)
        self.value_dim = int(value_dim)

    def _get_shared_storage_cls(self):
        """Return the per-CTA split-weight storage.

        Dense K3 MLA has a fixed 512-wide value head.  A single four-warp CTA
        owns that whole head, so the split normalization only needs to be
        formed once and broadcast to all 128 output lanes.
        """

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "weight": cute.struct.Align[
                cute.struct.MemRange[Float32, self.num_splits],
                16,
            ],
            "inverse": cute.struct.Align[
                cute.struct.MemRange[Float32, 1],
                16,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Tensor,
        partial_lse: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        active_splits: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            partial_output,
            partial_lse,
            output,
            final_lse,
            active_splits,
        ).launch(
            grid=(output.shape[0], output.shape[1], 1),
            block=[128, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial_output: cute.Tensor,
        partial_lse: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        active_splits: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        query, head, _ = cute.arch.block_idx()
        thread = Int32(thread)
        query = Int32(query)
        head = Int32(head)
        warp = thread >> Int32(5)
        dimension = thread * Int32(4)

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        split_weight = storage.weight.get_tensor(
            cute.make_layout((self.num_splits,), stride=(1,))
        )
        inverse_storage = storage.inverse.get_tensor(
            cute.make_layout((1,), stride=(1,))
        )

        # Only one warp reads and normalizes the split LSEs.  The previous
        # four-CTA implementation repeated this serial fold in every output
        # lane (128 copies per head).  Here each lane of warp 0 owns a strided
        # subset, then the warp reductions form the common maximum and sum.
        if warp == Int32(0):
            merged_max = Float32(-Float32.inf)
            split = lane
            while split < active_splits:
                part_lse = Float32(partial_lse[query, head, split])
                if part_lse > merged_max:
                    merged_max = part_lse
                split += Int32(32)

            for offset in (16, 8, 4, 2, 1):
                other_max = cute.arch.shuffle_sync_bfly(
                    merged_max,
                    offset=offset,
                )
                if other_max > merged_max:
                    merged_max = other_max

            merged_sum = Float32(0.0)
            split = lane
            while split < active_splits:
                part_lse = Float32(partial_lse[query, head, split])
                weight = Float32(0.0)
                if part_lse != Float32(-Float32.inf):
                    weight = exp2_approx(part_lse - merged_max)
                split_weight[split] = weight
                merged_sum += weight
                split += Int32(32)

            for offset in (16, 8, 4, 2, 1):
                merged_sum += cute.arch.shuffle_sync_bfly(
                    merged_sum,
                    offset=offset,
                )

            if lane == Int32(0):
                inverse = Float32(0.0)
                lse = Float32(-Float32.inf)
                if merged_sum > Float32(0.0):
                    inverse = Float32(1.0) / merged_sum
                    lse = (merged_max + log2_approx(merged_sum)) * Float32(
                        0.6931471805599453
                    )
                inverse_storage[0] = inverse
                final_lse[query, head] = lse

        cute.arch.barrier()

        while dimension < Int32(self.value_dim):
            accumulator = cute.make_rmem_tensor(4, Float32)
            for idx in cutlass.range_constexpr(4):
                accumulator[idx] = Float32(0.0)

            split = Int32(0)
            while split < active_splits:
                weight = Float32(split_weight[split])
                # Capture-static tail splits deliberately leave their partial
                # vectors undefined and publish -inf LSE.  Do not load those
                # vectors: NaN * 0 would otherwise poison the merged output.
                if weight > Float32(0.0):
                    partial_offset = (
                        (Int64(query) * Int64(partial_output.shape[1]) + Int64(head))
                        * Int64(self.num_splits)
                        + Int64(split)
                    ) * Int64(self.value_dim) + Int64(dimension)
                    packed0, packed1 = ld_global_v2_u32(
                        get_ptr_as_int64(partial_output, partial_offset)
                    )
                    value0, value1 = bfloat2_to_float2_scaled(
                        packed0,
                        weight,
                    )
                    value2, value3 = bfloat2_to_float2_scaled(
                        packed1,
                        weight,
                    )
                    accumulator[0] += value0
                    accumulator[1] += value1
                    accumulator[2] += value2
                    accumulator[3] += value3
                split += Int32(1)

            inverse = Float32(inverse_storage[0])
            output_offset = cute.crd2idx(
                (query, head, dimension),
                output.layout,
            )
            st_global_v2_u32(
                get_ptr_as_int64(output, output_offset),
                pack_f32x2_to_bfloat2(
                    accumulator[0] * inverse,
                    accumulator[1] * inverse,
                ),
                pack_f32x2_to_bfloat2(
                    accumulator[2] * inverse,
                    accumulator[3] * inverse,
                ),
            )
            dimension += Int32(128 * 4)


__all__ = ["DenseMlaMergeKernel"]
