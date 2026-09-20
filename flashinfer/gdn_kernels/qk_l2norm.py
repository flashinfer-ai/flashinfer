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
def _rows_norm(x: cute.Tensor, y: cute.Tensor, first_row: Int64, lane):
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), x.element_type, num_bits_per_copy=128
    )
    rx = cute.make_rmem_tensor(32, x.element_type)
    r = cute.make_rmem_tensor((8, 4), Float32)
    ry = cute.make_rmem_tensor(32, y.element_type)
    inv = cute.make_rmem_tensor(4, Float32)
    # A partial tile may have only one valid half-warp. Each reduction names
    # only its own 16 lanes, all of which take the same row predicate.
    mask = cutlass.Uint32(0xFFFF) << (cute.arch.lane_idx() & 16)

    # Eight adjacent 16-bit elements per lane give 128-bit copies. Keeping
    # four rows per thread increases each CTA's work to 64 independent rows.
    for group in cutlass.range_constexpr(4):
        row = first_row + group * 16
        if row < x.shape[0]:
            offset = cute.assume(row * 128 + lane * 8, divby=8)
            gx = cute.make_tensor(x.iterator + offset, cute.make_layout(8))
            part = cute.make_tensor(rx.iterator + group * 8, cute.make_layout(8))
            cute.copy(atom, gx, part)

    for group in cutlass.range_constexpr(4):
        row = first_row + group * 16
        if row < x.shape[0]:
            ss = Float32(0)
            for i in cutlass.range_constexpr(8):
                r[i, group] = Float32(rx[group * 8 + i])
                ss = ss + r[i, group] * r[i, group]
            for offset in [8, 4, 2, 1]:
                ss = ss + cute.arch.shuffle_sync_bfly(
                    ss, offset=offset, mask=mask, mask_and_clamp=31
                )
            inv[group] = cute.rsqrt(ss + Float32(1e-6), fastmath=True)

    for group in cutlass.range_constexpr(4):
        row = first_row + group * 16
        if row < x.shape[0]:
            for i in cutlass.range_constexpr(8):
                ry[group * 8 + i] = y.element_type(r[i, group] * inv[group])
            offset = cute.assume(row * 128 + lane * 8, divby=8)
            gy = cute.make_tensor(y.iterator + offset, cute.make_layout(8))
            part = cute.make_tensor(ry.iterator + group * 8, cute.make_layout(8))
            cute.copy(atom, part, gy)


@cute.kernel
def _kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    oq: cute.Tensor,
    ok: cute.Tensor,
):
    bx, by, _ = cute.arch.block_idx()
    tx, _, _ = cute.arch.thread_idx()
    first_row = Int64(bx) * 64 + tx // 16
    lane = tx % 16
    if by == 0:
        _rows_norm(q, oq, first_row, lane)
    else:
        _rows_norm(k, ok, first_row, lane)


@cute.jit
def _launch(
    q: cute.Tensor,
    k: cute.Tensor,
    oq: cute.Tensor,
    ok: cute.Tensor,
    stream,
):
    rows = cutlass.max(q.shape[0], k.shape[0])
    _kernel(q, k, oq, ok).launch(
        grid=(cute.ceil_div(rows, 64), 2, 1), block=(256, 1, 1), stream=stream
    )


@functools.cache
def _compiled(dtype: torch.dtype, d: int, target_key: tuple[int, str]):
    device = torch.device("cuda", target_key[0])
    dt = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
    }[dtype]
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
            _launch, q, k, oq, ok, stream
        ),
        extra_key_files=(__file__,),
    )


def normalize_qk(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize 128-element BF16/FP16 heads using FP32 accumulation.

    Chunked GDN consumes normalized BF16/FP16 operands. Keeping the reduction
    in FP32 and adding epsilon before rsqrt matches the decode convention.
    Q and K may have different head counts.
    """
    d = q.shape[-1]
    if d != 128 or k.shape[-1] != 128:
        raise ValueError("Q/K normalization requires head_size=128")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        raise ValueError("Q/K normalization requires matching BF16/FP16 inputs")
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
