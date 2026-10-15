"""GPU regressions for the DSA indexer's cooperative grid barriers.

Each CTA's last warp publishes a global write after a delay that grows with the
CTA index; every CTA then crosses the barrier and checks that it sees every
other CTA's write. The barrier must not release a CTA until all of its warps'
prior writes are ordered before thread zero's release arrival.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.utils import current_cuda_stream
from b12x.attention.dsa_indexer.fused_indexer import (
    _COOP_STATE_WORDS,
    _fused_group_barrier,
)

from ..conftest import require_b12x

_CTAS = 4
_THREADS = 1024
_DELAY_CYCLES = 200_000


def test_barrier_orders_delayed_warp_writes():
    device = require_b12x()

    class DelayedPublish:
        @cute.jit
        def __call__(
            self,
            values: cute.Tensor,
            state: cute.Tensor,
            errors: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.kernel(values, state, errors).launch(
                grid=(_CTAS, 1, 1),
                block=(_THREADS, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(self, values: cute.Tensor, state: cute.Tensor, errors: cute.Tensor):
            tid, _, _ = cute.arch.thread_idx()
            cta, _, _ = cute.arch.block_idx()
            if tid >= _THREADS - 32:
                start = cute.arch.clock64()
                while cute.arch.clock64() - start < Int64(cta + 1) * _DELAY_CYCLES:
                    pass
                if tid == _THREADS - 1:
                    values[cta] = Int32(1)
            _fused_group_barrier(state, Int32(0), Int32(0), Int32(_CTAS), Int32(tid))
            if tid == 0:
                missing = Int32(0)
                for other in cutlass.range_constexpr(_CTAS):
                    missing += Int32(values[other] != Int32(1))
                errors[cta] = missing

    values = torch.zeros(_CTAS, device=device, dtype=torch.int32)
    state = torch.zeros(_COOP_STATE_WORDS, device=device, dtype=torch.int32)
    errors = torch.empty(_CTAS, device=device, dtype=torch.int32)
    tensors = tuple(from_dlpack(t, assumed_align=4) for t in (values, state, errors))
    kernel = b12x_compile(
        DelayedPublish(),
        *tensors,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "test.dsa.fused_indexer_barrier", 1, (_CTAS, _THREADS)
        ),
    )
    for _ in range(8):
        values.zero_()
        state.zero_()
        errors.fill_(-1)
        kernel(*tensors, current_cuda_stream())
        torch.cuda.synchronize(device)
        assert torch.equal(errors, torch.zeros_like(errors)), errors.tolist()
