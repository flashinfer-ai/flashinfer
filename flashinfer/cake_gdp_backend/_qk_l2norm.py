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

The q / k L2 normalization of the Cake GDP prefill (``use_qk_l2norm_in_kernel=True``) on SM100 / SM103: one launch
of ``cake_gdp_qk_l2norm.cu`` over q and k, each 128-element row ``x * rsqrt(sum(x^2) + 1e-6)`` in FP32, rounded
once to BF16 (the operand dtype of the prefill kernels).  The library is built through FlashInfer's JIT, named by
the digest of its source.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any, Optional

import torch

_SOURCE = Path(__file__).with_name("cake_gdp_qk_l2norm.cu")
# Architectures of the Blackwell prefill body, as ``device_arch`` names them.
_GENCODE = {
    "sm_100a": "-gencode=arch=compute_100a,code=sm_100a",
    "sm_103a": "-gencode=arch=compute_103a,code=sm_103a",
}
_VECTOR_BYTES = 16


@functools.cache
def _module(arch: str) -> Any:
    """The loaded library of ``arch``, or None without FlashInfer's JIT (a package vendored without FlashInfer)."""

    try:
        from flashinfer.jit.core import gen_jit_spec
    except ImportError:
        return None

    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    spec = gen_jit_spec(
        f"cake_gdp_qk_l2norm_{arch.replace('_', '')}_{digest}",
        [_SOURCE],
        extra_cuda_cflags=[_GENCODE[arch]],
        use_fast_math=False,
    )
    return spec.build_and_load()


def _readable(x: torch.Tensor) -> torch.Tensor:
    """``x`` as the kernel reads it: a dense innermost dimension and 16-byte aligned rows (else one copy)."""

    element = x.element_size()
    if (
        x.stride(-1) == 1
        and x.data_ptr() % _VECTOR_BYTES == 0
        and all((s * element) % _VECTOR_BYTES == 0 for s in x.stride()[:-1])
    ):
        return x
    x = x.contiguous()
    return x if x.data_ptr() % _VECTOR_BYTES == 0 else x.clone()


def qk_l2norm(
    q: torch.Tensor, k: torch.Tensor, arch: str
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Dense BF16 ``(l2norm(q), l2norm(k))`` of q ``[T, H, 128]`` and k ``[T n, H, 128]`` (BF16 or FP16, one
    dtype) in one launch on the current stream, capturable in a CUDA graph; None when the library cannot be built
    here (no FlashInfer JIT)."""

    import tvm_ffi

    module = _module(arch)
    if module is None:
        return None
    q, k = _readable(q), _readable(k)
    q_out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    k_out = torch.empty(k.shape, dtype=torch.bfloat16, device=k.device)
    with tvm_ffi.use_torch_stream():
        module.run(q, k, q_out, k_out)
    return q_out, k_out
