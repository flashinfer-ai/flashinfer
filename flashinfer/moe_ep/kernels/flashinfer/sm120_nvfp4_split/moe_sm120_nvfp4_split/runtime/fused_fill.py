"""One-kernel replacement for a list of constant region fills.

The graph-owned replay reset clears about a dozen small counter / flag regions.
Issued as one fill node per region they form a serial chain of kernel nodes in
front of K1 (each node adds launch latency). ``launch_fused_fill`` issues the
same stores from a single kernel, one CTA per chunk of at most 16 KB.

Region addresses are baked into the compiled kernel (the workspace is allocated
once per tester); a different region list compiles a new kernel.
"""

import os
from typing import Dict, List, Sequence, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.cutlass_dsl import Int32, Int64

_Region = Tuple[int, int, int]  # (address, bytes, 32-bit fill pattern)
_cache: Dict[Tuple[int, Tuple[_Region, ...]], object] = {}
_CHUNK = 16384


def enabled() -> bool:
    """Default on (MEGA_SPLIT_FUSED_RESET=0 restores the fill-node chain): the
    graph-owned reset is one kernel node in front of K1 instead of a chain of
    fill nodes (port of the W4A8 decode front-path set)."""
    return os.environ.get("MEGA_SPLIT_FUSED_RESET", "1") == "1"


def _pattern(t: torch.Tensor, value: int):
    """32-bit pattern that fills ``t`` with ``value``, or None if none exists."""
    size = t.element_size()
    if value == 0:
        return 0
    if value == -1 and not t.dtype.is_floating_point:
        return 0xFFFFFFFF
    if size == 4 and not t.dtype.is_floating_point:
        return value & 0xFFFFFFFF
    return None


def plan_regions(ops: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[List[_Region], List[Tuple[torch.Tensor, int]]]:
    """Split fills into fusable regions and the rest (non-contiguous / odd sizes)."""
    regions, rest = [], []
    for t, value in ops:
        nbytes = t.numel() * t.element_size()
        pat = _pattern(t, value)
        addr = int(t.data_ptr())
        if nbytes == 0:
            continue
        if pat is None or not t.is_contiguous() or addr % 4 or nbytes % 4:
            rest.append((t, value))
        else:
            regions.append((addr, nbytes, pat))
    return regions, rest


def _compile(regions: Tuple[_Region, ...], stream: torch.cuda.Stream):
    # One CTA per chunk of at most _CHUNK bytes (16-byte aligned chunks).
    plan = []
    for addr, nbytes, pat in regions:
        word = pat if pat < (1 << 31) else pat - (1 << 32)
        off = 0
        while off < nbytes:
            size = min(_CHUNK, nbytes - off)
            a = addr + off
            vec = a % 16 == 0 and size >= 16
            n16 = size // 16 if vec else 0
            plan.append((a, word, vec, n16, (size - n16 * 16) // 4))
            off += size
    plan = tuple(plan)
    n = len(plan)

    def ptr(dtype, addr, align):
        return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)

    @cute.kernel
    def fill_kernel():
        b, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        for r in cutlass.range_constexpr(n):
            addr, word, vec, n16, rem = plan[r]
            if Int32(b) == Int32(r):
                if cutlass.const_expr(vec):
                    v = cute.make_tensor(ptr(cutlass.Int32, Int64(addr), 16), cute.make_layout((4, n16)))
                    i = tid
                    while i < Int32(n16):
                        for k in cutlass.range_constexpr(4):
                            v[k, i] = Int32(word)
                        i = i + Int32(256)
                if cutlass.const_expr(rem > 0):
                    w = cute.make_tensor(ptr(cutlass.Int32, Int64(addr + n16 * 16), 4), cute.make_layout(rem))
                    j = tid
                    while j < Int32(rem):
                        w[j] = Int32(word)
                        j = j + Int32(256)

    @cute.jit
    def launch(s: cuda.CUstream):
        fill_kernel().launch(grid=[n, 1, 1], block=[256, 1, 1], stream=s)

    return cute.compile(launch, cuda.CUstream(stream.cuda_stream))


def launch_fused_fill(ops: Sequence[Tuple[torch.Tensor, int]], stream: torch.cuda.Stream) -> None:
    regions, rest = plan_regions(ops)
    for t, value in rest:
        t.fill_(value)
    if not regions:
        return
    key = (torch.cuda.current_device(), tuple(regions))
    if key not in _cache:
        _cache[key] = _compile(key[1], stream)
    _cache[key](cuda.CUstream(stream.cuda_stream))


__all__ = ["launch_fused_fill", "plan_regions"]
