# SPDX-License-Identifier: Apache-2.0
import torch
import cutlass
import cutlass.cute as cute
from flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache import KeyedCompileMixin
from .preprocess import launch


class Normalize128(KeyedCompileMixin):
    def __init__(self, dtype, rows):
        self.dtype, self.rows = dtype, rows
        self.manual_cache_key("dtype", "rows")

    @cute.jit
    def __call__(self, x, y, stream):
        @cute.struct
        class Storage:
            partial: cute.struct.MemRange[cutlass.Float32, 4]

        self.storage = Storage
        self.kernel(x, y).launch(
            grid=(self.rows, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, x, y):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        warp = tid // 32
        sm = cutlass.utils.SmemAllocator().allocate(self.storage)
        part = sm.partial.get_tensor(cute.make_layout(4))
        value = cutlass.Float32(x[row * 128 + tid])
        total = value * value
        for shift in cutlass.range_constexpr(5):
            offset = 1 << shift
            prev = cute.arch.shuffle_sync_up(total, offset, mask_and_clamp=0)
            if lane >= offset:
                total += prev
        if lane == 31:
            part[warp] = total
        cute.arch.sync_threads()
        total = cutlass.Float32(1e-6)
        for i in cutlass.range_constexpr(4):
            total += part[i]
        y[row * 128 + tid] = self.dtype(value * cute.math.rsqrt(total, fastmath=True))


def normalize(x):
    y = torch.empty_like(x)
    dtype = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[x.dtype]
    launch(Normalize128(dtype, x.numel() // 128), (x, y), x.device)
    return y
