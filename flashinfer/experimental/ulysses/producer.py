# SPDX-License-Identifier: Apache-2.0
"""Coarse producer and reduction-geometry primitives, with no distributed state."""

import torch

from ...comm.ulysses_head_chunk import _positive_int, _storage_ranges_overlap


def validate_schedule(world_size, heads, head_dim, local_seq, schedule):
    for name, number in (
        ("world_size", world_size),
        ("heads", heads),
        ("head_dim", head_dim),
        ("local_seq", local_seq),
    ):
        _positive_int(number, name)
    if heads % world_size:
        raise ValueError("heads must be divisible by world_size")
    schedule = tuple(schedule)
    if not schedule:
        raise ValueError("schedule must not be empty")
    for count in schedule:
        _positive_int(count, "head count")
    if sum(schedule) != heads // world_size:
        raise ValueError("schedule must partition the heads of EACH destination")
    return schedule


class _StreamBound:
    def _bind(self):
        stream = torch.cuda.current_stream(self.device)
        key = stream.cuda_stream
        if self._stream_id is not None and self._stream_id != key:
            raise RuntimeError("workspace is bound to its first calling stream")
        self._stream_id = key


class GroupedQKVProducer(_StreamBound):
    def __init__(self, weight, *, world_size, heads, head_dim, local_seq, schedule):
        self.schedule = validate_schedule(
            world_size, heads, head_dim, local_seq, schedule
        )
        if not isinstance(weight, torch.Tensor) or not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if (
            weight.dtype != torch.bfloat16
            or weight.ndim != 2
            or not weight.is_contiguous()
        ):
            raise ValueError("weight must be contiguous BF16 [3*H*D,K]")
        if weight.shape[0] != 3 * heads * head_dim or weight.shape[1] <= 0:
            raise ValueError("invalid QKV weight geometry")
        if weight.requires_grad:
            raise ValueError("producer is inference-only; detach frozen weights")
        self.world, self.heads, self.dim = world_size, heads, head_dim
        self.rows, self.hidden, self.device = local_seq, weight.shape[1], weight.device
        self._stream_id = None
        self.weights, self.raw = [], []
        view = weight.view(3, world_size, heads // world_size, head_dim, self.hidden)
        offset = 0
        for count in self.schedule:
            # Every coarse group contains a head band for ALL destinations.
            # Weight packing is setup work; N changes can change GEMM rounding.
            self.weights.append(
                view[:, :, offset : offset + count].reshape(-1, self.hidden).clone()
            )
            self.raw.append(
                torch.empty(
                    local_seq,
                    3 * world_size * count * head_dim,
                    dtype=weight.dtype,
                    device=weight.device,
                )
            )
            offset += count
        self._setup = torch.cuda.Event()
        self._setup.record(torch.cuda.current_stream(self.device))

    def produce(self, x, group_index):
        if type(group_index) is not int or not 0 <= group_index < len(self.schedule):
            raise ValueError("invalid producer group_index")
        if not isinstance(x, torch.Tensor) or x.shape != (self.rows, self.hidden):
            raise ValueError("x must match the prepared [S_local,K] shape")
        if (
            x.device != self.device
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
        ):
            raise ValueError("x must be contiguous BF16 on the prepared device")
        if x.requires_grad:
            raise ValueError("producer is inference-only")
        if any(_storage_ranges_overlap(x, buf) for buf in (*self.raw, *self.weights)):
            raise ValueError("input must not alias producer storage")
        self._bind()
        torch.cuda.current_stream(self.device).wait_event(self._setup)
        raw = self.raw[group_index]
        torch.mm(x, self.weights[group_index].t(), out=raw)
        qkv = raw.view(self.rows, 3, self.world * self.schedule[group_index], self.dim)
        return tuple(t.unsqueeze(0) for t in qkv.unbind(1))


class ShapeStableKMean(_StreamBound):
    def __init__(self, sequence, heads, head_dim, *, device, dtype):
        for name, value in (
            ("sequence", sequence),
            ("heads", heads),
            ("head_dim", head_dim),
        ):
            _positive_int(value, name)
        device = torch.device(device)
        if device.type != "cuda" or dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("requires CUDA BF16/FP16 storage")
        self.scratch = torch.zeros(
            (1, sequence, heads, head_dim), device=device, dtype=dtype
        )
        self.mean = torch.empty((1, 1, heads, head_dim), device=device, dtype=dtype)
        self.device, self._stream_id = self.scratch.device, None
        self._setup = torch.cuda.Event()
        self._setup.record(torch.cuda.current_stream(self.device))

    def update(self, chunk, *, head_offset):
        if type(head_offset) is not int or head_offset < 0:
            raise ValueError("head_offset must be a nonnegative integer")
        b, s, h, d = self.scratch.shape
        if not isinstance(chunk, torch.Tensor) or chunk.ndim != 4:
            raise ValueError("chunk must be [1,S,HC,D]")
        count = chunk.shape[2]
        if chunk.shape != (b, s, count, d) or count <= 0 or head_offset + count > h:
            raise ValueError("invalid K chunk geometry")
        if (
            chunk.device != self.device
            or chunk.dtype != self.scratch.dtype
            or chunk.requires_grad
        ):
            raise ValueError("chunk device/dtype must match inference workspace")
        if _storage_ranges_overlap(chunk, self.scratch) or _storage_ranges_overlap(
            chunk, self.mean
        ):
            raise ValueError("chunk must not alias K-mean workspace")
        self._bind()
        torch.cuda.current_stream(self.device).wait_event(self._setup)
        self.scratch[:, :, head_offset : head_offset + count].copy_(chunk)
        # Preserve the whole-H mean launch geometry; only consume this band's
        # result. This does not promise backend-independent reduction order.
        torch.mean(self.scratch, dim=1, keepdim=True, out=self.mean)
        return self.mean[:, :, head_offset : head_offset + count]
