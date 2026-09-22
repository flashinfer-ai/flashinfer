# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import itertools

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from flashinfer.experimental.prims_ts_sparse_mla.views import (
    SparsePrefixMask,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


class PrefixBits:
    @cute.jit
    def __call__(self, cases, out, stream):
        self.kernel(cases, out).launch(
            grid=((cases.shape[0] + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, cases, out):
        i = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if i < cases.shape[0]:
            mask = SparsePrefixMask((Int32(cases[i, 0]), Int32(cases[i, 1])))
            out[i, 0] = mask.prefix_mask32(Int32(cases[i, 2]), Int32(cases[i, 3]))
            out[i, 1] = Int32(mask.tile_is_full(Int32(cases[i, 2]), Int32(cases[i, 3])))


def test_prefix_mask_word_boundaries():
    sizes = (0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257)
    cases = torch.tensor(
        [
            (origin, size, origin + offset, origin + max(0, size - crop))
            for origin, size, offset, crop in itertools.product(
                (0, 128, 256), sizes, (0, 32, 64, 96, 128, 256), (0, 7)
            )
        ],
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.empty((cases.shape[0], 2), device="cuda", dtype=torch.int32)
    tensors = [from_dlpack(t, assumed_align=16) for t in (cases, out)]
    compiled = cute.compile[cute.FrontendNext](
        PrefixBits(),
        *tensors,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 2",
    )
    compiled(cases, out)
    end = torch.minimum(cases[:, 0] + cases[:, 1], cases[:, 3]).long()
    positions = cases[:, 2].long()[:, None] + torch.arange(32, device="cuda")
    expected_bits = (
        (positions >= end[:, None]).long()
        * (1 << torch.arange(32, device="cuda", dtype=torch.int64))
    ).sum(-1)
    torch.testing.assert_close(out[:, 0], expected_bits.to(torch.int32), atol=0, rtol=0)
    torch.testing.assert_close(
        out[:, 1], (cases[:, 2].long() + 128 <= end).int(), atol=0, rtol=0
    )
