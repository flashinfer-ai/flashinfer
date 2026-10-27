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

"""Reusable dense GEMM buffers."""

import torch


_ALPHA_ONE_CACHE: dict = {}


def _cached_alpha_one(device: torch.device | str) -> torch.Tensor:
    # Per-device cached scalar-one alpha, to avoid a per-call torch.ones((1,))
    # host/device alloc on the generic FP8 dense-GEMM path. Mirrors
    # wo_projection._cached_alpha_one (not imported -- wo_projection imports
    # dense, so importing back would be circular).
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    key = (resolved.type, resolved.index)
    alpha = _ALPHA_ONE_CACHE.get(key)
    if alpha is None or alpha.device != resolved:
        alpha = torch.ones((1,), dtype=torch.float32, device=resolved)
        _ALPHA_ONE_CACHE[key] = alpha
    return alpha


def _empty_dense_gemm_output(
    m: int,
    n: int,
    l: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Allocate an `[M,N,L]` dense-GEMM output in the layout the kernel writes.

    The CuTe dense GEMM hardcodes ``c_major='n'`` and builds the C tensor from
    the data pointer with order ``(1,0,2)`` -- i.e. it writes the grouped output
    as physical ``[L,M,N]`` (an ``[M,N,L]`` view with strides ``(N,1,M*N)``) and
    ignores the runtime tensor's actual strides. A plain contiguous ``(M,N,L)``
    buffer (strides ``(N*L,L,1)``) would scatter the ``L`` groups to the wrong
    offsets, so back ``L>1`` with ``[L,M,N]`` physical storage. ``L==1`` is the
    same either way. Mirrors ``empty_dense_gemm_mnl_view`` in wo_projection.
    """
    if l > 1:
        return torch.empty((l, m, n), dtype=dtype, device=device).as_strided(
            (m, n, l), (n, 1, m * n)
        )
    return torch.empty((m, n, l), dtype=dtype, device=device)
