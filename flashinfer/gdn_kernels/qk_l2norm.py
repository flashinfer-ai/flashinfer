"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Q/K normalization for chunked GDN prefill.
"""

import functools
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64
from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from .cute_dsl_cache_naming import make_kernel_name
from .device_target import gdn_compile_options, gdn_device_target, target_arch


@cute.jit
def _rows_norm(
    x: cute.Tensor,
    y: cute.Tensor,
    first_row: Int64,
    lane,
    d: cutlass.Constexpr[int],
    lanes: cutlass.Constexpr[int],
    vector_elems: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    partials,
):
    chunks = cute.ceil_div(d, lanes * vector_elems)
    values = chunks * vector_elems
    row_groups = 256 // lanes
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        x.element_type,
        num_bits_per_copy=vector_elems * x.element_type.width,
    )
    rx = cute.make_rmem_tensor(values * rows_per_group, x.element_type)
    r = cute.make_rmem_tensor((values, rows_per_group), Float32)
    ry = cute.make_rmem_tensor(values * rows_per_group, y.element_type)
    inv = cute.make_rmem_tensor(rows_per_group, Float32)
    warp_lanes = min(lanes, 32)
    warp_idx = cute.arch.thread_idx()[0] // 32
    # Each mask names only lanes belonging to one row, including in tail CTAs.
    mask = cutlass.Uint32((1 << warp_lanes) - 1) << (
        cute.arch.lane_idx() // warp_lanes * warp_lanes
    )

    for group in cutlass.range_constexpr(rows_per_group):
        row = first_row + group * row_groups
        if row < x.shape[0]:
            for chunk in cutlass.range_constexpr(chunks):
                col = (chunk * lanes + lane) * vector_elems
                offset = cute.assume(row * d + col, divby=vector_elems)
                gx = cute.make_tensor(
                    x.iterator + offset, cute.make_layout(vector_elems)
                )
                part = cute.make_tensor(
                    rx.iterator + group * values + chunk * vector_elems,
                    cute.make_layout(vector_elems),
                )
                if cutlass.const_expr(d % (lanes * vector_elems) == 0):
                    cute.copy(atom, gx, part)
                else:
                    part.fill(x.element_type(0))
                    if col < d:
                        cute.copy(atom, gx, part)

    for group in cutlass.range_constexpr(rows_per_group):
        row = first_row + group * row_groups
        ss = Float32(0)
        if row < x.shape[0]:
            for i in cutlass.range_constexpr(values):
                r[i, group] = Float32(rx[group * values + i])
                ss = ss + r[i, group] * r[i, group]
            for offset in [16, 8, 4, 2, 1]:
                if cutlass.const_expr(offset < warp_lanes):
                    ss = ss + cute.arch.shuffle_sync_bfly(
                        ss, offset=offset, mask=mask, mask_and_clamp=31
                    )
            if cutlass.const_expr(lanes <= 32):
                inv[group] = cute.rsqrt(ss + Float32(1e-6), fastmath=True)
        if cutlass.const_expr(lanes > 32):
            if lane % 32 == 0:
                partials[warp_idx, group] = ss

    if cutlass.const_expr(lanes > 32):
        # All threads participate, even when their row lies beyond the tile.
        cute.arch.sync_threads()
        first_warp = warp_idx // (lanes // 32) * (lanes // 32)
        for group in cutlass.range_constexpr(rows_per_group):
            ss = Float32(0)
            for warp in cutlass.range_constexpr(lanes // 32):
                ss = ss + partials[first_warp + warp, group]
            inv[group] = cute.rsqrt(ss + Float32(1e-6), fastmath=True)

    for group in cutlass.range_constexpr(rows_per_group):
        row = first_row + group * row_groups
        if row < x.shape[0]:
            for i in cutlass.range_constexpr(values):
                ry[group * values + i] = y.element_type(r[i, group] * inv[group])
            for chunk in cutlass.range_constexpr(chunks):
                col = (chunk * lanes + lane) * vector_elems
                offset = cute.assume(row * d + col, divby=vector_elems)
                gy = cute.make_tensor(
                    y.iterator + offset, cute.make_layout(vector_elems)
                )
                part = cute.make_tensor(
                    ry.iterator + group * values + chunk * vector_elems,
                    cute.make_layout(vector_elems),
                )
                if cutlass.const_expr(d % (lanes * vector_elems) == 0):
                    cute.copy(atom, part, gy)
                else:
                    if col < d:
                        cute.copy(atom, part, gy)


@cute.kernel
def _kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    oq: cute.Tensor,
    ok: cute.Tensor,
    d: cutlass.Constexpr[int],
    lanes: cutlass.Constexpr[int],
    vector_elems: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
):
    bx, by, _ = cute.arch.block_idx()
    tx, _, _ = cute.arch.thread_idx()
    first_row = Int64(bx) * (256 // lanes) * rows_per_group + tx // lanes
    lane = tx % lanes
    partials = None
    if cutlass.const_expr(lanes > 32):
        partials = cutlass.utils.SmemAllocator().allocate_tensor(
            Float32, cute.make_layout((8, rows_per_group)), byte_alignment=4
        )
    if by == 0:
        _rows_norm(
            q, oq, first_row, lane, d, lanes, vector_elems, rows_per_group, partials
        )
    else:
        _rows_norm(
            k, ok, first_row, lane, d, lanes, vector_elems, rows_per_group, partials
        )


@cute.jit
def _launch(
    q: cute.Tensor,
    k: cute.Tensor,
    oq: cute.Tensor,
    ok: cute.Tensor,
    d: cutlass.Constexpr[int],
    lanes: cutlass.Constexpr[int],
    vector_elems: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    stream,
):
    rows = cutlass.max(q.shape[0], k.shape[0])
    rows_per_block = (256 // lanes) * rows_per_group
    _kernel(q, k, oq, ok, d, lanes, vector_elems, rows_per_group).launch(
        grid=(cute.ceil_div(rows, rows_per_block), 2, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@functools.cache
def _compiled(dtype: torch.dtype, d: int, target_key: tuple[int, str]):
    device = torch.device("cuda", target_key[0])
    dt = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }[dtype]
    # Every row must preserve the copy's alignment. Odd D uses scalar copies.
    vector_elems = min(16 // dtype.itemsize, d & -d)
    lanes = min(256, 1 << (((d + 7) // 8) - 1).bit_length())
    values = ((d + lanes * vector_elems - 1) // (lanes * vector_elems)) * vector_elems
    # Keep several short rows in flight without multiplying large-row storage.
    rows_per_group = 4 if values <= 8 else 2 if values <= 16 else 1
    qm, km = cute.sym_int(64), cute.sym_int(64)

    def fake(m):
        return cute.runtime.make_fake_compact_tensor(
            dt, (m, d), stride_order=(1, 0), assumed_align=16
        )

    q, k, oq, ok = fake(qm), fake(km), fake(qm), fake(km)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return build_and_load_cute_dsl_kernel(
        "gdn_qk_l2norm",
        make_kernel_name("qk_l2norm", dtype, d, target_arch(target_key)),
        lambda: cute.compile[gdn_compile_options(device, cute.EnableTVMFFI(True))](
            _launch, q, k, oq, ok, d, lanes, vector_elems, rows_per_group, stream
        ),
        extra_key_files=(__file__,),
    )


def normalize_qk(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize BF16/FP16/FP32 rows of up to 64 KiB using FP32 accumulation.

    Chunked GDN consumes normalized BF16/FP16 operands. Keeping the reduction
    in FP32 and adding epsilon before rsqrt matches the decode convention.
    Q and K may have different head counts.
    """
    if q.ndim == 0 or k.ndim == 0 or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K normalization requires matching last dimensions")
    if (
        q.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or k.dtype != q.dtype
    ):
        raise ValueError("Q/K normalization requires matching BF16/FP16/FP32 inputs")
    d = q.shape[-1]
    if not 0 < d <= 65536 // q.element_size():
        raise ValueError("Q/K normalization requires a positive row size up to 64 KiB")
    if not q.is_cuda or q.device != k.device:
        raise ValueError("Q/K normalization requires inputs on the same CUDA device")
    q = q.contiguous()
    k = k.contiguous()
    # Contiguous views can still have an unaligned storage offset.
    if q.data_ptr() % 16:
        q = q.clone()
    if k.data_ptr() % 16:
        k = k.clone()
    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    if q.numel() == 0 and k.numel() == 0:
        return oq, ok
    with torch.cuda.device(q.device):
        _compiled(q.dtype, d, gdn_device_target(q.device).compile_key)(
            q.reshape(-1, d), k.reshape(-1, d), oq.reshape(-1, d), ok.reshape(-1, d)
        )
    return oq, ok
