# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workspace-owned tensor maps, schedules and launches for TIRx KDA."""

import ctypes

import torch
import tvm
import tvm_ffi

from .cache import get_kernel
from .schedule import _host_item_table

D = 128
_MIXED = (1300, 547, 2048, 963, 271, 3063)


class _TensorMap:
    def __init__(self, tensor, dims, strides, box, swizzle, promotion):
        self.storage = ctypes.create_string_buffer(128 + 64)
        self.ptr = ctypes.c_void_p((ctypes.addressof(self.storage) + 63) & ~63)
        tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
            self.ptr,
            "bfloat16",
            len(dims),
            ctypes.c_void_p(tensor.data_ptr()),
            *dims,
            *strides,
            *box,
            *([1] * len(dims)),
            0,
            swizzle,
            promotion,
            0,
        )


def _convert(args):
    return tuple(
        tvm_ffi.from_dlpack(arg) if isinstance(arg, torch.Tensor) else arg
        for arg in args
    )


def make_plan(data, offsets, fixed):
    q = data["q"]
    B, T, H, _ = q.shape
    T *= B
    arch = "sm_%d%da" % torch.cuda.get_device_capability(q.device)
    if fixed and T % 32 == 0:
        return _split_plan(data, T, H, arch)
    return _fused_plan(data, offsets, T, H, arch)


def _fused_plan(data, offsets, T, H, arch):
    q = data["q"]
    dev = q.device
    lengths = tuple(
        end - start for start, end in zip(offsets, offsets[1:], strict=False)
    )
    nseq = len(lengths)
    sm_count = torch.cuda.get_device_properties(dev).multi_processor_count
    num_ctas = min(sm_count, H * nseq)
    # Keep the imported INT21 choices; other shapes use whole chains and FP32 handoffs.
    official = sm_count == 148 and H in (64, 96) and lengths in (_MIXED, (1024,) * 8)
    force_lpt = not official or (H == 96 and lengths == _MIXED)
    bf16_handoff = official and H == 64
    lists = _host_item_table(offsets, H, num_ctas, force_lpt)
    max_items = max(160, ((max(map(len, lists)) + 31) // 32) * 32)
    items = torch.zeros(num_ctas, max_items, 2, dtype=torch.int64)
    for cta, entries in enumerate(lists):
        if entries:
            items[cta, : len(entries)] = torch.tensor(entries, dtype=torch.int64)
    # Conversion preserves the packed uint32 words, including the first/last bits.
    items_g = items.reshape(-1).to(torch.int32).view(torch.uint32).to(dev)
    counts = torch.tensor(list(map(len, lists)), dtype=torch.int32, device=dev)
    hand = torch.empty(
        (num_ctas + 1) * D * D,
        dtype=torch.bfloat16 if bf16_handoff else torch.float32,
        device=dev,
    )
    flags = torch.zeros(num_ctas + 1, dtype=torch.int32, device=dev)

    def activation_map(tensor, rows=64, slabs=2):
        return _TensorMap(
            tensor, (64, T, 2 * H), (H * 256, 128), (64, rows, slabs), 3, 2
        )

    maps = [activation_map(data[n], rows=32, slabs=1) for n in ("q", "k")]
    maps += [activation_map(data[n]) for n in ("v", "g")]
    maps += [
        _TensorMap(data["beta"], (H, T), (H * 2,), (8, 64), 0, 2),
        activation_map(data["output"], rows=32),
    ]
    executable = get_kernel(
        "fused", arch, H, (H == 64 and nseq == 1, bf16_handoff, max_items)
    )
    args = _convert(
        (
            *(m.ptr for m in maps),
            data["output"].view(-1),
            data["A_log"],
            data["dt_bias"].view(-1),
            data["initial_state"].view(-1),
            data["final_state"].view(-1),
            hand,
            flags,
            items_g,
            counts,
            num_ctas,
            data["scale"],
        )
    )

    def launch():
        flags.zero_()
        with tvm_ffi.use_torch_stream():
            executable(*args)

    return {
        "launch": launch,
        "keep": (maps, hand, flags, items_g, counts, args, executable),
        "route": "fused",
        "max_items": max_items,
    }


def _split_plan(data, T, H, arch):
    dev = data["q"].device
    sm_count = torch.cuda.get_device_properties(dev).multi_processor_count
    hpc = 1 if sm_count // 2 >= H else 2
    front = get_kernel("front", arch, H)
    chain = get_kernel("chain", arch, H, (hpc,))
    C = 32
    NC = T // C
    NI = NC * H
    kbar, qt, w1 = (
        torch.empty(NI, C * D, dtype=torch.bfloat16, device=dev) for _ in range(3)
    )
    t1, aqk = (
        torch.empty(NI, C * C, dtype=torch.bfloat16, device=dev) for _ in range(2)
    )
    vec = torch.empty(NI, 160, dtype=torch.float32, device=dev)
    flags = torch.zeros(NI, dtype=torch.int32, device=dev)
    maps = {
        n: _TensorMap(data[n], (D, T, H), (H * D * 2, D * 2), (64, C, 1), 3, 3)
        for n in ("q", "k", "v", "g", "output")
    }

    def tile_map(tensor, rows, cols):
        return _TensorMap(
            tensor,
            (cols, rows, NI),
            (cols * 2, rows * cols * 2),
            (min(cols, 64), rows, 1),
            3 if cols == D else 2,
            3,
        )

    tile_maps = [
        tile_map(kbar, C, D),
        tile_map(qt, C, D),
        tile_map(t1, C, C),
        tile_map(aqk, C, C),
        tile_map(w1, D, C),
    ]
    concurrent = sm_count > H // hpc + 8
    ctas_front = max(1, min(sm_count - H // hpc if concurrent else sm_count, NI))
    a1 = _convert(
        (
            *(data[n].view(-1) for n in ("q", "k", "g", "beta", "A_log", "dt_bias")),
            vec.view(-1),
            kbar.view(-1),
            qt.view(-1),
            t1.view(-1),
            aqk.view(-1),
            w1.view(-1),
            flags,
            maps["q"].ptr,
            maps["k"].ptr,
            maps["g"].ptr,
            0,
            NI,
            ctas_front,
            (NI + ctas_front - 1) // ctas_front,
            int(concurrent),
        )
    )
    a2 = _convert(
        (
            data["v"].view(-1),
            data["initial_state"].view(-1),
            data["final_state"].view(-1),
            data["output"].view(-1),
            vec.view(-1),
            kbar.view(-1),
            qt.view(-1),
            t1.view(-1),
            aqk.view(-1),
            w1.view(-1),
            maps["v"].ptr,
            *(m.ptr for m in tile_maps),
            maps["output"].ptr,
            flags,
            data["scale"],
            NC,
            0 if concurrent else NI,
            1,
        )
    )
    # The front-end budget reserves SMs for the state chain. Join the private
    # chain stream to the caller's stream on every invocation, including capture.
    chain_stream = torch.cuda.Stream(device=dev, priority=-1) if concurrent else None
    fork, join = torch.cuda.Event(), torch.cuda.Event()
    sequential_args = a2[:-2] + (NI, 1)
    initialized = False

    def launch():
        nonlocal initialized
        flags.zero_()
        if not concurrent or not initialized:
            # CUDA's lazy module loading may synchronize the context. Load both
            # kernels before starting a consumer that polls for its producer.
            # This first invocation still updates the caller's state only once.
            with tvm_ffi.use_torch_stream():
                front(*a1)
                chain(*sequential_args)
            initialized = True
            return
        current = torch.cuda.current_stream(dev)
        fork.record(current)
        chain_stream.wait_event(fork)
        if H == 64:
            with tvm_ffi.use_torch_stream(torch.cuda.stream(chain_stream)):
                chain(*a2)
            with tvm_ffi.use_torch_stream():
                front(*a1)
        else:
            with tvm_ffi.use_torch_stream():
                front(*a1)
            with tvm_ffi.use_torch_stream(torch.cuda.stream(chain_stream)):
                chain(*a2)
        join.record(chain_stream)
        current.wait_event(join)

    return {
        "launch": launch,
        "keep": (
            maps,
            tile_maps,
            a1,
            a2,
            front,
            chain,
            chain_stream,
            fork,
            join,
            flags,
        ),
        "route": "split",
    }
