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

import functools
import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch

from ...cute_dsl.utils import get_num_sm
from ...utils import _get_cache_buf, get_compute_capability
from .single_pass_multi_cta_radix_topk import (
    STATE_SIZE,
    SinglePassMultiCTARadixTopKKernel,
)

_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


# ---------------------------------------------------------------------------
# Kernel compilation cache (follows MXFP8 pattern)
# ---------------------------------------------------------------------------
@functools.cache
def _get_compiled_kernel(
    cutlass_dtype,
    chunk_size: int,
    top_k: int,
    next_n: int,
    num_copy_bits: int,
    ctas_per_group: int,
    num_sms: int,
    return_val: bool,
):
    """Compile and cache a single-pass multi-CTA radix top-k kernel."""
    n_rows = cute.sym_int()
    n_cols = cute.sym_int()
    n_batch = cute.sym_int()
    n_groups = cute.sym_int()

    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (n_rows, n_cols),
        stride_order=(1, 0),
        assumed_align=32,
    )
    row_states_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_groups, STATE_SIZE),
        stride_order=(1, 0),
        assumed_align=32,
    )
    seqlen_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_batch,),
        stride_order=(0,),
    )
    output_indices_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_rows, top_k),
        stride_order=(1, 0),
    )
    if return_val:
        output_values_fake = cute.runtime.make_fake_compact_tensor(
            cutlass_dtype,
            (n_rows, top_k),
            stride_order=(1, 0),
        )
    else:
        output_values_fake = None
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel_obj = SinglePassMultiCTARadixTopKKernel(
        dtype=cutlass_dtype,
        chunk_size=chunk_size,
        top_k=top_k,
        next_n=next_n,
        num_copy_bits=num_copy_bits,
        ctas_per_group=ctas_per_group,
        num_sms=num_sms,
    )
    compiled_kernel = cute.compile(
        kernel_obj,
        input_fake,
        row_states_fake,
        seqlen_fake,
        output_indices_fake,
        output_values_fake,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )
    return compiled_kernel


# ---------------------------------------------------------------------------
# Chunk-size / CTA-count heuristics (ported 1:1 from TRT-LLM)
# ---------------------------------------------------------------------------
def _compute_max_chunk(cutlass_dtype, num_copy_bits: int = 256):
    """Compute the maximum chunk_size a single CTA can handle."""
    max_smem = cutlass.utils.get_smem_capacity_in_bytes()
    overhead = 256 * 4 * 2 + 4 * 4 + 8 * 4
    if cutlass_dtype == cutlass.Float32:
        ordered_elem_size = 4
    else:
        ordered_elem_size = 2
    vec_size = num_copy_bits // cutlass_dtype.width
    max_chunk = (max_smem - overhead) // ordered_elem_size
    max_chunk = (max_chunk // vec_size) * vec_size
    return max_chunk, vec_size


def _get_chunk_config(
    cutlass_dtype,
    num_cols: int,
    chunk_size: Optional[int] = None,
    num_copy_bits: int = 256,
    num_rows: int = 1,
    num_sms: int = 148,
):
    """Resolve chunk_size and ctas_per_group.

    If chunk_size is provided, use it (clamped and aligned).
    Otherwise use an SM-aware heuristic that targets
    total_ctas ≈ num_sms by balancing parallelism against
    per-CTA reduce overhead.

    Returns (chunk_size, ctas_per_group, vec_size).
    """
    max_chunk, vec_size = _compute_max_chunk(cutlass_dtype, num_copy_bits)

    if chunk_size is not None:
        chunk_size = min(chunk_size, max_chunk)
        chunk_size = (chunk_size // vec_size) * vec_size
        if chunk_size < vec_size:
            chunk_size = vec_size
    else:
        ideal_ctas_per_group = max(1, num_sms // max(num_rows, 1))

        if ideal_ctas_per_group <= 1:
            ctas_per_group = math.ceil(num_cols / max_chunk)
            if ctas_per_group < 1:
                ctas_per_group = 1
            chunk_size = math.ceil(num_cols / ctas_per_group)
            chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
            if chunk_size > max_chunk:
                chunk_size = max_chunk
        else:
            chunk_size = math.ceil(num_cols / ideal_ctas_per_group)
            chunk_size = max(chunk_size, 8192)

            ctas_per_group = math.ceil(num_cols / chunk_size)
            if ctas_per_group == 2 and chunk_size < 32768:
                chunk_size = num_cols

            snap_up = 1 << math.ceil(math.log2(max(chunk_size, 1)))
            if snap_up > max_chunk:
                snap_up = 1 << int(math.log2(max_chunk))
            chunk_size = snap_up

    ctas_per_group = math.ceil(num_cols / chunk_size)
    return chunk_size, ctas_per_group, vec_size


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def top_k_cute_dsl(
    input: torch.Tensor,
    k: int,
    sorted: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CuTe-DSL Top-K backend using single-pass multi-CTA radix top-k.

    Requires Blackwell (SM100+) and nvidia-cutlass-dsl.
    Supports k <= 2048, dtypes float16/bfloat16/float32.
    """
    if input.dim() != 2:
        raise ValueError(
            f"Expected 2D input [batch_size, vocab_size], got shape {tuple(input.shape)}"
        )
    if input.dtype not in _TORCH_TO_CUTLASS_DTYPE:
        raise ValueError(
            f"Unsupported dtype {input.dtype}. "
            "Supported: float16, bfloat16, float32."
        )
    if k <= 0 or k > 2048:
        raise ValueError(
            f"k must be in [1, 2048] for cute-dsl backend, got {k}."
        )

    major, minor = get_compute_capability(input.device)
    sm = major * 10 + minor
    if sm < 100:
        raise RuntimeError(
            f"backend='cute-dsl' top-k requires SM100+, got SM{sm}."
        )

    torch_dtype = input.dtype
    cutlass_dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
    num_rows, num_cols = input.shape
    device = input.device
    num_sms = get_num_sm(device)
    num_copy_bits = 256
    next_n = 1
    return_val = False

    # --- Dispatch heuristic (ported 1:1 from TRT-LLM) ---
    is_fp32 = (torch_dtype == torch.float32)

    if is_fp32 and num_cols < 65536:
        use_single_pass_multi_cta = False
    else:
        chunk_size, ctas_per_group, _ = _get_chunk_config(
            cutlass_dtype, num_cols,
            num_copy_bits=num_copy_bits, num_rows=num_rows,
            num_sms=num_sms,
        )

        if ctas_per_group >= 2:
            use_single_pass_multi_cta = (num_rows * ctas_per_group <= num_sms)
            if is_fp32:
                use_single_pass_multi_cta = (
                    use_single_pass_multi_cta and num_cols >= 65536
                )
        else:
            use_single_pass_multi_cta = (not is_fp32 and num_rows <= num_sms)

    if not use_single_pass_multi_cta:
        chunk_size, ctas_per_group, _ = _get_chunk_config(
            cutlass_dtype, num_cols,
            num_copy_bits=num_copy_bits, num_rows=num_rows,
            num_sms=num_sms,
        )

    # --- Compile kernel ---
    key = (cutlass_dtype, chunk_size, k, next_n, num_copy_bits,
           ctas_per_group, num_sms, return_val)
    compiled_kernel = _get_compiled_kernel(*key)

    # --- Allocate buffers ---
    num_groups = min(num_sms // ctas_per_group, num_rows)
    if num_groups < 1:
        num_groups = 1

    row_states = _get_cache_buf(
        f"cute_dsl_topk_row_states_{device}",
        num_sms * STATE_SIZE * 4,
        device,
        zero_init=True,
    )
    row_states_2d = torch.zeros(
        (num_sms, STATE_SIZE), dtype=torch.int32, device=device
    )

    seq_lens = torch.full((num_rows,), num_cols, dtype=torch.int32, device=device)
    output_indices_i32 = torch.empty(
        (num_rows, k), dtype=torch.int32, device=device
    )

    # --- Execute kernel ---
    compiled_kernel(
        input,
        row_states_2d,
        seq_lens,
        output_indices_i32,
        None,  # output_values (return_val=False)
    )

    # --- Gather values and convert indices ---
    indices = output_indices_i32.long()
    values = torch.gather(input, dim=-1, index=indices)

    if sorted:
        values, sort_perm = torch.sort(values, dim=-1, descending=True)
        indices = torch.gather(indices, dim=-1, index=sort_perm)

    return values, indices
