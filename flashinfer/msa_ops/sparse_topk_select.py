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


def _compile_topk(kernel_obj, num_score_tensors: int, num_scalars: int):
    """Shared compile plumbing so every top-k variant builds with identical
    fake-tensor setup and options. All tensor arguments are 3D and the scalar
    values are placeholders (the compiled kernels take them dynamically)."""
    import cutlass
    import cutlass.cute as cute

    def fk(dtype):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            tuple(cute.sym_int() for _ in range(3)),
            stride_order=(2, 1, 0),
            assumed_align=4,
        )

    args = [fk(cutlass.Float32)]  # max_score (H, P, S)
    args += [fk(cutlass.Int32) for _ in range(num_score_tensors - 1)]
    args += [cutlass.Int32(1)] * num_scalars
    args.append(cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True))
    return cute.compile(kernel_obj, *args, options="--enable-tvm-ffi")


def _get_compiled_topk(topk: int, small: bool):
    """``small`` picks the O(N^2) count-rank kernel, else the radix kernel; the
    two give identical selections only on distinct-score inputs (ties may differ)."""
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
    # Tensors: max_score, out. Scalars: nvp, force_begin/end, total_q, heads.
    compiled = _compile_topk(_TopKKernel(topk=topk), 2, 5)
    _topk_compile_cache[key] = compiled
    return compiled


def _get_compiled_topk_chunked(topk: int):
    """Two-kernel (per-chunk rank + merge) variant; selections match the
    count-rank kernel exactly (same bit-key and tie order)."""
    key = (topk, "chunked")
    compiled = _topk_compile_cache.get(key)
    if compiled is not None:
        return compiled

    from .cute_dsl.topk_select_chunked_sm12x import TopKSelectChunkedSm12x

    # Tensors: max_score, candidate keys, candidate indices, out. Scalars: nvp,
    # force_begin/end, num_chunks, chunk_len, total_q, heads.
    compiled = _compile_topk(TopKSelectChunkedSm12x(topk=topk), 4, 7)
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

    Notes
    -----
    Several kernels implement this selection, chosen by problem size. They
    return identical indices unless two blocks have nearly identical scores
    competing for the last slots; either block may then be selected, so exact
    output indices are only reproducible for a fixed problem size and library
    version. The selected score values match in every case.
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

    from .cute_dsl.topk_select_chunked_sm12x import (
        _CHUNK_BLOCKS,
        _MAX_CHUNK_BLOCKS,
        _MAX_CHUNKED_ROWS,
        _MAX_CHUNKS,
        _MIN_BLOCKS,
        _MIN_CHUNKS,
    )
    from .cute_dsl.topk_select_countrank_sm12x import _MAX_BLOCKS

    # Dispatch on the runtime valid-page count: the kernels only ever touch
    # blocks below num_valid_pages. The chunked path serves grids too small to
    # fill the GPU, and still wins at full ones by reading the scores once
    # where the radix kernel makes a pass per stage; prefill-sized grids keep
    # the single-kernel paths. Everything here is shape-constant, so the
    # choice is CUDA-graph safe.
    small = int(num_valid_pages) <= _MAX_BLOCKS
    n_mid = int(num_valid_pages) - force_begin_blocks - force_end_blocks
    chunked = (
        total_qo_len * num_qo_heads <= _MAX_CHUNKED_ROWS
        and _MIN_BLOCKS < n_mid <= _MAX_CHUNKS * _MAX_CHUNK_BLOCKS
    )
    if chunked:
        num_chunks = max(_MIN_CHUNKS, min(_MAX_CHUNKS, -(-n_mid // _CHUNK_BLOCKS)))
        chunk_len = -(-n_mid // num_chunks)
        # One allocation for both candidate buffers keeps this hot path at a
        # single allocator call.
        cand = torch.empty(
            (2, total_qo_len, num_qo_heads, num_chunks * topk),
            dtype=torch.int32,
            device=max_score.device,
        )
        cand_key, cand_idx = cand[0], cand[1]
        _get_compiled_topk_chunked(topk)(
            max_score,
            cand_key,
            cand_idx,
            output,
            int(num_valid_pages),
            int(force_begin_blocks),
            int(force_end_blocks),
            num_chunks,
            chunk_len,
            int(total_qo_len),
            int(num_qo_heads),
        )
        return output

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
