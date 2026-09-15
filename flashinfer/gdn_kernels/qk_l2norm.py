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


@cute.jit
def _row_norm(x: cute.Tensor, y: cute.Tensor, row: Int64, D: cutlass.Constexpr[int]):
    lane = cute.arch.lane_idx()
    if row < x.shape[0]:
        r = cute.make_rmem_tensor((D + 31) // 32, Float32)
        ss = Float32(0)
        for i in cutlass.range_constexpr((D + 31) // 32):
            col = lane + 32 * i
            r[i] = Float32(0)
            if col < D:
                r[i] = Float32(x[row, col])
            ss = ss + r[i] * r[i]
        for offset in [16, 8, 4, 2, 1]:
            ss = ss + cute.arch.shuffle_sync_bfly(
                ss, offset=offset, mask=-1, mask_and_clamp=31
            )
        inv = cute.rsqrt(ss + Float32(1e-6), fastmath=True)
        for i in cutlass.range_constexpr((D + 31) // 32):
            col = lane + 32 * i
            if col < D:
                y[row, col] = y.element_type(r[i] * inv)


@cute.kernel
def _kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    oq: cute.Tensor,
    ok: cute.Tensor,
    D: cutlass.Constexpr[int],
):
    bx, by, _ = cute.arch.block_idx()
    tx, _, _ = cute.arch.thread_idx()
    row = Int64(bx) * 8 + tx // 32
    if by == 0:
        _row_norm(q, oq, row, D)
    else:
        _row_norm(k, ok, row, D)


@cute.jit
def _launch(
    q: cute.Tensor,
    k: cute.Tensor,
    oq: cute.Tensor,
    ok: cute.Tensor,
    D: cutlass.Constexpr[int],
    stream,
):
    rows = cutlass.max(q.shape[0], k.shape[0])
    _kernel(q, k, oq, ok, D).launch(
        grid=(cute.ceil_div(rows, 8), 2, 1), block=(256, 1, 1), stream=stream
    )


@functools.cache
def _compiled(dtype: torch.dtype, d: int, capability: tuple[int, int]):
    dt = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
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
        make_kernel_name("qk_l2norm", dtype, d, capability),
        lambda: cute.compile(
            _launch, q, k, oq, ok, d, stream, options="--enable-tvm-ffi"
        ),
        extra_key_files=(__file__,),
    )


def normalize_qk(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each head in FP32, then round back to the input dtype.

    Chunked GDN consumes normalized BF16/FP16 operands. Keeping the reduction
    in FP32 and adding epsilon before rsqrt matches the decode convention.
    Q and K may have different head counts.
    """
    d = q.shape[-1]
    q = q.contiguous()
    k = k.contiguous()
    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    with torch.cuda.device(q.device):
        _compiled(q.dtype, d, torch.cuda.get_device_capability(q.device))(
            q.reshape(-1, d), k.reshape(-1, d), oq.reshape(-1, d), ok.reshape(-1, d)
        )
    return oq, ok
