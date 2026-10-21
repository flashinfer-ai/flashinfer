# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Dense GEMM split-K reduction kernels."""

import triton.language as tl
import torch
import triton


@triton.jit(do_not_specialize=["total"], do_not_specialize_on_alignment=["total"])
def _reduce_split_k2_bf16_kernel(partials, out, total, BLOCK: tl.constexpr) -> None:
    pid = tl.program_id(0).to(tl.int64)
    total = total.to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    accum = tl.load(partials + offs, mask=mask).to(tl.float32)
    accum += tl.load(partials + total + offs, mask=mask).to(tl.float32)
    tl.store(out + offs, accum, mask=mask)


def _reduce_split_k2_bf16(
    partials: torch.Tensor, out: torch.Tensor, *, m: int, n: int
) -> None:
    """Fused 2-way split-K FP32-partials reduction (exact); faster than torch.add.

    Falls back to torch.add when the scratch/output layout is not the expected
    [m, n, 2] / [m, n, 1] contiguous-row form.
    """
    total = int(m) * int(n)
    if (
        partials.shape == (m, n, 2)
        and partials.stride() == (n, 1, total)
        and out.shape == (m, n, 1)
        and out.stride()[0] == n
        and out.stride()[1] == 1
    ):
        block = 1024
        grid = (triton.cdiv(total, block),)
        _reduce_split_k2_bf16_kernel[grid](partials, out, total, BLOCK=block)
    else:
        torch.add(partials[:, :, 0], partials[:, :, 1], out=out[:, :, 0])
