# SPDX-License-Identifier: Apache-2.0
# Algorithm adapted from FlashQLA (c)2026 Qwen team, MIT license.
import torch
import cutlass
import cutlass.cute as cute
from flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache import (
    KeyedCompileMixin,
    cached_compile,
)
from .primitives import _mma, _acc, _coords, _gemm, _vector_load_tile


def wrap(a):
    return cute.runtime.from_dlpack(
        a.reshape(-1),
        assumed_align=min(16, (a.data_ptr() & -a.data_ptr()) or a.element_size()),
        enable_tvm_ffi=True,
    ).mark_layout_dynamic()


def launch(kernel, args, device):
    import cuda.bindings.driver as driver

    args = tuple(map(wrap, args)) + (
        driver.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    fn = cached_compile(kernel, *args, compile_options=(cute.GPUArch("sm_120a"),))
    fn(*args)


class Prefix32(KeyedCompileMixin):
    def __init__(self, heads, slots, index_dtype, g_dtype):
        self.heads, self.slots, self.index_dtype, self.g_dtype = (
            heads,
            slots,
            index_dtype,
            g_dtype,
        )
        self.manual_cache_key("heads", "slots", "index_dtype", "g_dtype")

    @cute.jit
    def __call__(self, g, indices, prefix, stream):
        self.kernel(g, indices, prefix).launch(
            grid=(self.slots, self.heads, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, g, indices, prefix):
        slot, head, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        token = cutlass.Int32(indices[slot * 2])
        end = cutlass.Int32(indices[slot * 2 + 1])
        value = cutlass.Float32(0.0)
        if token + lane < end:
            value = cutlass.Float32(g[(token + lane) * self.heads + head])
        for shift in cutlass.range_constexpr(5):
            offset = 1 << shift
            prev = cute.arch.shuffle_sync_up(value, offset, mask_and_clamp=0)
            if lane >= offset:
                value += prev
        if token + lane < end:
            prefix[(token + lane) * self.heads + head] = prefix.element_type(value)


class Inverse32(KeyedCompileMixin):
    def __init__(self, dtype, hk, heads, slots, index_dtype, b_dtype):
        self.dtype, self.hk, self.heads, self.slots, self.index_dtype, self.b_dtype = (
            dtype,
            hk,
            heads,
            slots,
            index_dtype,
            b_dtype,
        )
        self.manual_cache_key("dtype", "hk", "heads", "slots", "index_dtype", "b_dtype")

    @cute.jit
    def __call__(self, k, b, indices, a, stream):
        dtype = self.dtype

        @cute.struct
        class Storage:
            k: cute.struct.Align[cute.struct.MemRange[dtype, 32 * 136], 128]
            l: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 32 * 33], 16]
            tmp: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 16 * 17], 16]
            b: cute.struct.MemRange[cutlass.Float32, 32]

        self.storage = Storage
        self.kernel(k, b, indices, a).launch(
            grid=(self.slots, self.heads, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, k, b, indices, a):
        slot, head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        token = cutlass.Int32(indices[slot * 2])
        end = cutlass.Int32(indices[slot * 2 + 1])
        kh = head * self.hk // self.heads
        sm = cutlass.utils.SmemAllocator().allocate(self.storage)
        sk = sm.k.get_tensor(cute.make_layout((32, 128), stride=(136, 1)))
        sl = sm.l.get_tensor(cute.make_layout((32, 32), stride=(33, 1)))
        temp = sm.tmp.get_tensor(cute.make_layout((16, 16), stride=(17, 1)))
        sb = sm.b.get_tensor(cute.make_layout(32))
        if token + 32 <= end:
            _vector_load_tile(
                k, sk, (token * self.hk + kh) * 128, self.hk * 128, 128, tid
            )
        else:
            for row in cutlass.range_constexpr(32):
                value = self.dtype(0.0)
                if token + row < end:
                    value = k[((token + row) * self.hk + kh) * 128 + tid]
                sk[row, tid] = value
        if tid < 32:
            value = cutlass.Float32(0.0)
            if token + tid < end:
                value = cutlass.Float32(b[(token + tid) * self.heads + head])
            sb[tid] = value
        cute.arch.sync_threads()
        kk = _acc(self.dtype, 32, 32)
        xy = _coords(self.dtype, 32, 32)
        _gemm(sk, sk, kk, 32, 32, 128, True)
        for i in cutlass.range_constexpr(cute.size(kk)):
            row, col = xy[i]
            value = cutlass.Float32(0.0)
            if row > col:
                value = kk[i] * sb[row]
            elif row == col:
                value = cutlass.Float32(1.0)
            sl[row, col] = value
        cute.arch.sync_threads()
        # Two independent16x16 unit-lower solves, retaining FP32 between rows.
        block = tid // 32
        col = tid % 32
        for row in cutlass.range_constexpr(1, 16):
            value = cutlass.Float32(0.0)
            if tid < 64 and col < row:
                for j in cutlass.range_constexpr(row):
                    value -= (
                        sl[block * 16 + row, block * 16 + j]
                        * sl[block * 16 + j, block * 16 + col]
                    )
            cute.arch.sync_threads()
            if tid < 64 and col < row:
                sl[block * 16 + row, block * 16 + col] = value
            cute.arch.sync_threads()
        # Coupling=-L11^-1 L10 L00^-1, same twoFP32 matrix products as reference.
        for repeat in cutlass.range_constexpr(2):
            index = tid + repeat * 128
            row, col = index // 16, index % 16
            value = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(16):
                value -= sl[16 + row, 16 + j] * sl[16 + j, col]
            temp[row, col] = value
        cute.arch.sync_threads()
        vals = cute.make_rmem_tensor(2, cutlass.Float32)
        for repeat in cutlass.range_constexpr(2):
            index = tid + repeat * 128
            row, col = index // 16, index % 16
            value = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(16):
                value += temp[row, j] * sl[j, col]
            vals[repeat] = value
        cute.arch.sync_threads()
        for repeat in cutlass.range_constexpr(2):
            index = tid + repeat * 128
            sl[16 + index // 16, index % 16] = vals[repeat]
        cute.arch.sync_threads()
        for repeat in cutlass.range_constexpr(8):
            index = tid + repeat * 128
            row, col = index // 32, index % 32
            if token + row < end:
                a[((token + row) * self.heads + head) * 32 + col] = self.dtype(
                    sl[row, col]
                )


def precompute(k, b, g, indices):
    length, hk, _ = k.shape
    heads = b.shape[1]
    slots = indices.shape[0]
    a = torch.empty((length, heads, 32), device=k.device, dtype=k.dtype)
    prefix = torch.empty_like(g)
    dtype = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[k.dtype]
    launch(
        Prefix32(heads, slots, indices.dtype, g.dtype), (g, indices, prefix), k.device
    )
    launch(
        Inverse32(dtype, hk, heads, slots, indices.dtype, b.dtype),
        (k, b, indices, a),
        k.device,
    )
    return a, prefix


@cute.jit
def write_tile(
    acc,
    dst,
    offset,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    row_stride: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    mma = _mma(dtype, m, n)
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        dst.element_type,
        num_bits_per_copy=dst.element_type.width * 2,
    )
    copy = cute.make_tiled_copy_C(atom, mma)
    thread = copy.get_slice(tid % 128)
    target = cute.make_tensor(
        dst.iterator + cute.assume(offset, divby=2),
        cute.make_layout((m, n), stride=(row_stride, 1)),
    )
    target = thread.partition_D(target)
    target = cute.make_tensor(
        target.iterator.align(dst.element_type.width // 4), target.layout
    )
    values = cute.make_fragment_like(acc, dst.element_type)
    for i in cutlass.range_constexpr(cute.size(acc)):
        values[i] = dst.element_type(acc[i])
    cute.copy(copy, thread.retile(values), target)


@cute.jit
def load_tile(
    src,
    dst,
    offset,
    row_stride: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    width: cutlass.Constexpr,
    lane,
):
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), src.element_type, num_bits_per_copy=128
    )
    threads_per_row = width // 8
    copy = cute.make_tiled_copy_tv(
        atom,
        cute.make_layout(
            (128 // threads_per_row, threads_per_row), stride=(threads_per_row, 1)
        ),
        cute.make_layout((1, 8), stride=(8, 1)),
    )
    source = cute.make_tensor(
        src.iterator + cute.assume(offset, divby=8),
        cute.make_layout((rows, width), stride=(row_stride, 1)),
    )
    thr = copy.get_slice(lane)
    cute.copy(copy, thr.partition_S(source), thr.partition_D(dst))


@cute.jit
def async_load32(
    src, dst, offset, row_stride: cutlass.Constexpr, width: cutlass.Constexpr, lane
):
    atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        src.element_type,
        num_bits_per_copy=128,
    )
    threads_per_row = width // 8
    copy = cute.make_tiled_copy_tv(
        atom,
        cute.make_layout(
            (128 // threads_per_row, threads_per_row), stride=(threads_per_row, 1)
        ),
        cute.make_layout((1, 8), stride=(8, 1)),
    )
    source = cute.make_tensor(
        src.iterator + cute.assume(offset, divby=8),
        cute.make_layout((32, width), stride=(row_stride, 1)),
    )
    thr = copy.get_slice(lane)
    cute.copy(copy, thr.partition_S(source), thr.partition_D(dst))
