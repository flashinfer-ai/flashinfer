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

Helpers shared by the MSA op wrappers.
"""

import torch

# MSA KV-block size in tokens; also the KV-cache page size on the paged path.
_BLK_KV = 128

# Process-wide cache of cute.compile results, shared by all MSA ops; each op
# keys it with a tuple starting with its own tag (e.g. "sparse_prefill", ...).
_compile_cache: dict = {}


def _q_offset_tensor(
    q_offset,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    device,
) -> torch.Tensor:
    """Build the per-batch causal offset tensor the kernels consume.

    A query's global position is ``q_offset[b] + local_index`` (MSA q_offset
    semantics). When ``q_offset`` is None, queries are right-aligned to the end
    of the KV sequence: ``q_offset[b] = seqlen_k[b] - seqlen_q[b]``."""
    if q_offset is None:
        return (
            (cu_seqlens_k[1:] - cu_seqlens_k[:-1])
            - (cu_seqlens_q[1:] - cu_seqlens_q[:-1])
        ).to(torch.int32)
    return _q_offset_explicit(q_offset, cu_seqlens_q.numel() - 1, device)


def _q_offset_explicit(q_offset, batch_size: int, device) -> torch.Tensor:
    """Normalize a caller-provided ``q_offset`` (int or int32 tensor) to a
    ``(batch_size,)`` int32 tensor on ``device``."""
    if isinstance(q_offset, int):
        return torch.full((batch_size,), q_offset, dtype=torch.int32, device=device)
    if q_offset.dtype != torch.int32:
        raise ValueError("q_offset must be int32")
    return q_offset.to(device)


def _cutlass_dtype(torch_dtype: torch.dtype):
    """Map a torch dtype to the corresponding cutlass dtype."""
    import cutlass

    return {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.uint8: cutlass.Uint8,
        torch.int32: cutlass.Int32,
    }[torch_dtype]


def _fake(dtype, shape, align=16):
    """Make a fake compact row-major tensor for ``cute.compile``.

    Compiling against symbolic shapes (``cute.sym_int()``) yields one kernel
    that accepts torch tensors of any matching-rank shape at runtime."""
    import cutlass.cute as cute

    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )
