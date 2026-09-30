"""Registered device staging and byte gathering for immutable disk row planes."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from dataclasses import dataclass


@triton.jit
def _gather(arena, descriptors, weights, scales, BLOCK: tl.constexpr):
    fragment = tl.program_id(0).to(tl.int64)
    source = tl.load(descriptors + fragment * 4)
    dest = tl.load(descriptors + fragment * 4 + 1)
    length = tl.load(descriptors + fragment * 4 + 2)
    plane = tl.load(descriptors + fragment * 4 + 3)
    for start in range(0, length, BLOCK):
        col = start + tl.arange(0, BLOCK).to(tl.int64)
        data = tl.load(arena + source + col, col < length, 0)
        if plane:
            tl.store(scales + dest + col, data, col < length)
        else:
            tl.store(weights + dest + col, data, col < length)


@dataclass(frozen=True)
class _Pointer:
    dtype: torch.dtype

    def data_ptr(self):
        return 16


def compile_gather(ordinal):
    with torch.cuda.device(ordinal):
        byte, descriptor = _Pointer(torch.uint8), _Pointer(torch.int64)
        return _gather.warmup(byte, descriptor, byte, byte, BLOCK=256, num_warps=4, grid=(1,))


class GdsRows:
    def __init__(self, cache, queue_depth):
        from b12x.loader._gds_native import load
        from .disk_table import _tensor_from_pointer

        self.native = load()
        self.reader = None
        self.device = cache.device
        self.closed = False
        self.used = False
        self.done = torch.cuda.Event()
        with torch.cuda.device(cache.device):
            try:
                self.reader = self.native.create(
                    cache.shard_rows, cache.table_rows, cache.shard_start, cache.shard_end,
                    cache.weight_row_bytes, cache.scale_row_bytes, cache.max_lookups,
                    queue_depth, cache.device.index,
                )
                arena, descriptors, nbytes, capacity, self.batch_count = self.native.info(self.reader)
                self.arena = _tensor_from_pointer(arena, shape=(nbytes,), dtype=torch.uint8,
                                                  device=cache.device, nbytes=nbytes)
                self.descriptors = _tensor_from_pointer(descriptors, shape=(capacity, 4), dtype=torch.int64,
                                                        device=cache.device, nbytes=capacity * 32)
                self.weight = torch.empty((cache.max_lookups, cache.weight_row_bytes), dtype=torch.uint8, device=cache.device)
                self.scale = torch.empty((cache.max_lookups, cache.scale_row_bytes), dtype=torch.uint8, device=cache.device) if cache.scale_row_bytes else None
                self.program = compile_gather(cache.device.index)
                self.program._init_handles()
                metadata = self.program.metadata
                if (metadata.global_scratch_size or metadata.profile_scratch_size or
                        metadata.num_ctas != 1 or metadata.launch_cooperative_grid or metadata.launch_pdl):
                    raise RuntimeError("GDS byte gather requires a scratch-free ordinary CUDA launch")
                self.native.configure_gather(self.reader, self.program.function,
                    metadata.num_warps * metadata.warp_size, metadata.shared,
                    self.weight.data_ptr(), (self.scale if self.scale is not None else self.weight).data_ptr())
            except BaseException:
                self.close()
                raise

    def read(self, ids_buffer, count, stream):
        if self.used:
            self.done.synchronize()
        try:
            self.native.begin(self.reader, ids_buffer, count, stream.cuda_stream)
            self.weight[:count].zero_()
            if self.scale is not None:
                self.scale[:count].zero_()
            self.native.execute(self.reader, stream.cuda_stream)
            self.done.record(stream)
            self.used = True
        except BaseException:
            # Include any enqueued descriptor copy or gather before teardown/retry.
            self.done.record(stream)
            self.used = True
            try:
                self.close()
            except Exception:
                pass
            raise

    def close(self):
        if self.closed:
            return
        if self.used:
            self.done.synchronize()
        if self.reader is not None:
            self.native.close(self.reader)
        self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
