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

Import-light ``@backend_requirement`` checker of ``mm_fp4(backend="cake")``.

Only dtype / shape / layout / compute-capability logic lives here so that
``flashinfer.gemm`` can import it at module level; the kernels are reached
through a deferred import of :mod:`.cake_backend`.
"""

from __future__ import annotations

from typing import Optional

import torch

from ...utils import experimental_backend, supported_compute_capability

CAKE_K_TILE = 256  # smallest mainloop K tile of the generated GEMM programs


@experimental_backend
@supported_compute_capability([100, 103])
def cake_mm_fp4_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,
    backend: str = "auto",
    use_nvfp4: bool = True,
    enable_pdl: bool = True,
) -> bool:
    """Requirement check of the cake per-token NVFP4 GEMM.

    The backend serves the per-token-alpha NVFP4 case only: ``alpha`` holds one
    FP32 scale per row of ``a``, both operands carry 128x4 swizzled E4M3 block
    scales (``block_size`` 16), ``a`` is a contiguous ``[M, K/2]`` tensor,
    ``b`` the column-major ``[K/2, N]`` view of a contiguous ``[N, K/2]``
    weight, ``N % 8 == 0``, ``K % 256 == 0`` and the output is bf16 / fp16.
    """
    explicit = backend == "cake"

    def reject(message: str) -> bool:
        if explicit:
            raise ValueError(message)
        return False

    if alpha is None or alpha.numel() <= 1:
        return reject(
            "the cake mm_fp4 backend implements the per-token alpha path only "
            "(alpha of shape [M]); use another backend for a scalar alpha"
        )
    if not use_nvfp4 or block_size != 16:
        return reject("the cake mm_fp4 backend serves NVFP4 (block_size 16) only")
    if use_8x4_sf_layout:
        return reject("the cake mm_fp4 backend requires 128x4 scale factors")
    if out_dtype not in (torch.bfloat16, torch.float16):
        return reject("the cake mm_fp4 backend writes bf16 or fp16 outputs")
    if not a.is_contiguous():
        return reject("the cake mm_fp4 backend requires a contiguous [M, K/2] a")
    if b.dim() != 2 or b.stride() != (1, b.shape[0]):
        return reject(
            "the cake mm_fp4 backend requires b as the column-major [K/2, N] view "
            "of a contiguous [N, K/2] weight (pass b_fp4.T)"
        )
    if b.shape[1] % 8:
        return reject(
            f"the cake mm_fp4 backend requires N % 8 == 0, got n={b.shape[1]}"
        )
    if (2 * a.shape[1]) % CAKE_K_TILE:
        return reject(
            f"the cake mm_fp4 backend requires K % {CAKE_K_TILE} == 0, "
            f"got k={2 * a.shape[1]}"
        )
    if out is not None and not out.is_contiguous():
        return reject("the cake mm_fp4 backend requires a contiguous [M, N] out")
    if not enable_pdl:
        return reject(
            "the cake mm_fp4 backend builds its programs with programmatic "
            "dependent launch; enable_pdl=False is not available"
        )
    return True
