# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The two opaque primitives of ``lse.row`` evaluated on the device for the exact CPU model: ``ex2.approx.ftz.f32`` and
``lg2.approx.ftz.f32`` are the only operations of the class the CPU cannot reproduce, so the model runs the kernels' own
inline-PTX instructions (``cute_ext.ex2_approx_ftz`` / ``lg2_approx_ftz``) on a batch of binary32 arguments and reads
the bits back."""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from .. import jit
from .cute_ext import CuteF32

THREADS = 256
MUFU_MODULE = "batch_invariant_sampling_mufu"


class _TableKernel:
    """``y[i] = f(x[i])`` for one of the two MUFU primitives."""

    def __init__(self, which: str):
        self.which = which

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor, n: Int32, stream):
        self.kernel(mX, mY, n).launch(
            grid=[cute.ceil_div(n, THREADS), 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mY: cute.Tensor, n: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * THREADS + tidx
        if i < n:
            if cutlass.const_expr(self.which == "ex2"):
                mY[i] = CuteF32.ex2(mX[i])
            else:
                mY[i] = CuteF32.lg2(mX[i])


@functools.cache
def _table_kernel(which: str):
    def compile_fn():
        f32 = cutlass.Float32
        x_fake = cute.runtime.make_fake_tensor(
            f32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        y_fake = cute.runtime.make_fake_tensor(
            f32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            _TableKernel(which),
            x_fake,
            y_fake,
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MUFU_MODULE, f"{which}_approx_ftz_table", compile_fn)


def _apply(which: str, x):
    import torch

    xc = x.detach().to(torch.float32).reshape(-1).contiguous().cuda()
    y = torch.empty_like(xc)
    if xc.numel():
        _table_kernel(which)(xc, y, xc.numel())
        torch.cuda.synchronize()
    return y.cpu().view(x.shape)


def ex2_device(z):
    """``ex2.approx.ftz.f32`` of every element of a tensor, on the device (returned on the CPU as fp32)."""
    return _apply("ex2", z)


def lg2_device(s):
    """``lg2.approx.ftz.f32`` of every element of a tensor, on the device (returned on the CPU as fp32)."""
    return _apply("lg2", s)
