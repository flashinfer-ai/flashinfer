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
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..utils import is_sm12x_supported


_topk_compile_cache: dict = {}


def _get_compiled_topk(topk: int, small: bool):
    """``small`` picks the O(N^2) count-rank kernel, else the radix kernel; the
    two give identical selections only on distinct-score inputs (ties may differ)."""
    import cutlass
    import cutlass.cute as cute

    key = (topk, small)
    compiled = _topk_compile_cache.get(key)
    if compiled is not None:
        return compiled

    if small:
        from .cute_dsl.topk_select_countrank_sm12x import (
            TopKSelectCountRankSm12x as _TopKKernel,
        )
    else:
        from .cute_dsl.topk_select_radix_sm12x import (  # type: ignore[assignment,no-redef]
            TopKSelectRadixSm12x as _TopKKernel,
        )
    kernel_obj = _TopKKernel(topk=topk)

    def fk(dtype, ndim, align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            tuple(cute.sym_int() for _ in range(ndim)),
            stride_order=tuple(reversed(range(ndim))),
            assumed_align=align,
        )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel_obj,
        fk(cutlass.Float32, 3, 4),  # max_score (H, P, S)
        fk(cutlass.Int32, 3, 4),  # out (S, H, topk)
        cutlass.Int32(1),  # num_valid_pages
        cutlass.Int32(0),  # force_begin
        cutlass.Int32(0),  # force_end
        cutlass.Int32(1),  # total_qo_len
        cutlass.Int32(1),  # num_qo_heads
        stream_fake,
        options="--enable-tvm-ffi",
    )
    _topk_compile_cache[key] = compiled
    return compiled


@flashinfer_api
def msa_topk_select(
    max_score: torch.Tensor,
    topk: int,
    num_valid_pages: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
) -> torch.Tensor:
    """Select the top-K KV blocks per query token based on attention scores.

    Implements the block-scoring pass of Minimax Sparse Attention: given the
    per-block maximum attention scores from a cheap proxy prefill, selects the
    ``topk`` most important KV blocks for each (query token, head) pair and
    returns their sorted indices.

    Parameters
    ----------
    max_score : torch.Tensor
        Shape ``(num_qo_heads, max_k_tiles, total_qo_len)``, dtype float32.
        Per-KV-block maximum attention scores produced by the proxy prefill
        pass.  Entries for invalid tiles (beyond the actual KV length) must be
        set to ``-inf`` by the caller.
    topk : int
        Number of KV blocks to select per (query token, head).  Must be 16.
    num_valid_pages : int, optional
        Actual number of valid KV pages (``<= max_k_tiles``).  Indices
        ``>= num_valid_pages`` are replaced with -1 and sorted to the tail.
        Defaults to ``max_k_tiles`` (disables clamping).
    output : torch.Tensor, optional
        Pre-allocated output tensor of shape
        ``(total_qo_len, num_qo_heads, topk)``, dtype int32.  Allocated
        internally if not provided.
    force_begin_blocks : int
        Number of KV blocks at the beginning (sink tokens) to always include.
    force_end_blocks : int
        Number of KV blocks at the end (local window) to always include.

    Returns
    -------
    torch.Tensor
        Shape ``(total_qo_len, num_qo_heads, topk)``, dtype int32.
        Ascending KV-block indices; ``-1`` entries are tail-padded invalid
        slots.
    """
    if not is_sm12x_supported(max_score.device):
        raise RuntimeError(
            "msa_topk_select requires SM120 or SM121 (Blackwell) and CUDA >= 12.8"
        )

    if max_score.dtype != torch.float32:
        raise ValueError(f"max_score must be float32, got {max_score.dtype}")
    if not max_score.is_contiguous():
        raise ValueError("max_score must be contiguous")
    if max_score.ndim != 3:
        raise ValueError(
            f"max_score must be 3D (num_qo_heads, max_k_tiles, total_qo_len), got {max_score.ndim}D"
        )
    if topk != 16:
        raise ValueError(f"topk must be 16, got {topk}")

    num_qo_heads, max_k_tiles, total_qo_len = max_score.shape

    if num_valid_pages is None:
        num_valid_pages = max_k_tiles

    # Input guards (the radix kernel does not clamp internally): an out-of-range
    # num_valid_pages reads max_score out of bounds, and oversized forced regions
    # overrun the kernel's fixed forced-index buffer or underflow to negative blocks.
    if not 0 < num_valid_pages <= max_k_tiles:
        raise ValueError(
            f"num_valid_pages must be in (0, max_k_tiles={max_k_tiles}], "
            f"got {num_valid_pages}"
        )
    if force_begin_blocks < 0 or force_end_blocks < 0:
        raise ValueError("force_begin_blocks / force_end_blocks must be >= 0")
    if force_begin_blocks + force_end_blocks > topk:
        raise ValueError(
            f"force_begin_blocks + force_end_blocks ({force_begin_blocks} + "
            f"{force_end_blocks}) must be <= topk ({topk})"
        )
    if force_begin_blocks + force_end_blocks > num_valid_pages:
        raise ValueError(
            f"force_begin_blocks + force_end_blocks ({force_begin_blocks} + "
            f"{force_end_blocks}) must be <= num_valid_pages ({num_valid_pages})"
        )

    if output is None:
        output = torch.empty(
            (total_qo_len, num_qo_heads, topk),
            dtype=torch.int32,
            device=max_score.device,
        )
    else:
        if output.shape != (total_qo_len, num_qo_heads, topk):
            raise ValueError(
                f"output shape must be ({total_qo_len}, {num_qo_heads}, {topk}), "
                f"got {tuple(output.shape)}"
            )
        if output.dtype != torch.int32:
            raise ValueError(f"output must be int32, got {output.dtype}")

    from .cute_dsl.topk_select_countrank_sm12x import _MAX_BLOCKS

    # Dispatch on the runtime valid-page count: the count-rank kernel only ever
    # touches blocks below num_valid_pages, regardless of the allocated score
    # dimension.
    small = int(num_valid_pages) <= _MAX_BLOCKS
    _get_compiled_topk(topk, small)(
        max_score,
        output,
        int(num_valid_pages),
        int(force_begin_blocks),
        int(force_end_blocks),
        int(total_qo_len),
        int(num_qo_heads),
    )

    return output
