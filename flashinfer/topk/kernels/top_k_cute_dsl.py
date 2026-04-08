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
import logging
import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch

from ...cute_dsl.utils import get_num_sm
from ...utils import get_compute_capability
from .single_pass_multi_cta_radix_topk import (
    STATE_SIZE as DISTRIBUTED_STATE_SIZE,
    SinglePassMultiCTARadixTopKKernel,
)
from .single_pass_multi_cta_radix_topk_cluster import (
    STATE_SIZE as CLUSTER_STATE_SIZE,
    SinglePassMultiCTARadixTopKClusterKernel,
    _query_max_cluster_size,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached buffer management (avoids per-call GPU malloc + memset/fill kernels)
# ---------------------------------------------------------------------------
_row_states_cache: dict = {}
_row_states_initialized: dict = {}


def _get_cached_row_states(key: str, num_sms: int, state_size: int, device: torch.device):
    """Get or create cached row_states buffer. Zeroed once on first use."""
    if key not in _row_states_cache or _row_states_cache[key].shape != (num_sms, state_size):
        _row_states_cache[key] = torch.zeros(
            (num_sms, state_size), dtype=torch.int32, device=device
        )
        _row_states_initialized[key] = True
    return _row_states_cache[key]


_seq_lens_cache: dict = {}


def _get_cached_seq_lens(num_rows: int, num_cols: int, device: torch.device):
    """Get or create cached seq_lens tensor. Reused when shape matches."""
    key = (num_rows, num_cols, device)
    if key not in _seq_lens_cache:
        _seq_lens_cache[key] = torch.full(
            (num_rows,), num_cols, dtype=torch.int32, device=device
        )
    return _seq_lens_cache[key]


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
    kernel_variant: str,
    cutlass_dtype,
    chunk_size: int,
    top_k: int,
    next_n: int,
    num_copy_bits: int,
    ctas_per_group: int,
    num_sms: int,
    return_val: bool,
):
    """Compile and cache a single-pass multi-CTA radix top-k kernel.

    kernel_variant: "cluster" or "distributed"
    """
    state_size = CLUSTER_STATE_SIZE if kernel_variant == "cluster" else DISTRIBUTED_STATE_SIZE

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
        (n_groups, state_size),
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

    if kernel_variant == "cluster":
        kernel_cls = SinglePassMultiCTARadixTopKClusterKernel
    else:
        kernel_cls = SinglePassMultiCTARadixTopKKernel

    kernel_obj = kernel_cls(
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


def _get_cluster_chunk_config(
    cutlass_dtype,
    num_cols: int,
    chunk_size: Optional[int] = None,
    num_copy_bits: int = 256,
    num_rows: int = 1,
    num_sms: int = 148,
):
    """Resolve chunk_size and ctas_per_group, clamped to hw max cluster size.

    Returns (chunk_size, ctas_per_group, vec_size) or (None, None, None)
    if the problem cannot be handled by the cluster kernel.
    """
    chunk_size, ctas_per_group, vec_size = _get_chunk_config(
        cutlass_dtype, num_cols, chunk_size, num_copy_bits, num_rows, num_sms
    )

    hw_max_cluster = _query_max_cluster_size()
    if ctas_per_group > hw_max_cluster:
        max_chunk, vec_size = _compute_max_chunk(cutlass_dtype, num_copy_bits)
        chunk_size = math.ceil(num_cols / hw_max_cluster)
        chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
        if chunk_size > max_chunk:
            logger.warning(
                f"Cluster top-k: num_cols={num_cols} requires "
                f"chunk_size={chunk_size} which exceeds max shared "
                f"memory capacity ({max_chunk}). Falling back to "
                f"non-cluster variant."
            )
            return None, None, None
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

    Uses the cluster-accelerated variant (DSMEM + cluster barriers) as
    the primary path, falling back to the non-cluster variant (global
    memory atomics) when the problem size exceeds cluster capacity.

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
        use_multi_cta = False
    else:
        chunk_size, ctas_per_group, _ = _get_chunk_config(
            cutlass_dtype, num_cols,
            num_copy_bits=num_copy_bits, num_rows=num_rows,
            num_sms=num_sms,
        )

        if ctas_per_group >= 2:
            use_multi_cta = (num_rows * ctas_per_group <= num_sms)
            if is_fp32:
                use_multi_cta = use_multi_cta and num_cols >= 65536
        else:
            use_multi_cta = (not is_fp32 and num_rows <= num_sms)

    # --- Select kernel variant ---
    if use_multi_cta:
        # Try cluster first (primary path for DSv3.2)
        cluster_config = _get_cluster_chunk_config(
            cutlass_dtype, num_cols,
            num_copy_bits=num_copy_bits, num_rows=num_rows,
            num_sms=num_sms,
        )
        if cluster_config[0] is not None:
            chunk_size, ctas_per_group, _ = cluster_config
            # Check max supported cols for cluster
            max_chunk, _ = _compute_max_chunk(cutlass_dtype, num_copy_bits)
            hw_max_cluster = _query_max_cluster_size()
            if num_cols <= max_chunk * hw_max_cluster:
                kernel_variant = "cluster"
                state_size = CLUSTER_STATE_SIZE
            else:
                kernel_variant = "distributed"
                state_size = DISTRIBUTED_STATE_SIZE
                chunk_size, ctas_per_group, _ = _get_chunk_config(
                    cutlass_dtype, num_cols,
                    num_copy_bits=num_copy_bits, num_rows=num_rows,
                    num_sms=num_sms,
                )
        else:
            # Cluster can't handle this size, fall back to distributed
            kernel_variant = "distributed"
            state_size = DISTRIBUTED_STATE_SIZE
            chunk_size, ctas_per_group, _ = _get_chunk_config(
                cutlass_dtype, num_cols,
                num_copy_bits=num_copy_bits, num_rows=num_rows,
                num_sms=num_sms,
            )
    else:
        # Single-CTA path (still uses the kernel class with ctas_per_group=1)
        kernel_variant = "cluster"
        state_size = CLUSTER_STATE_SIZE
        chunk_size, ctas_per_group, _ = _get_chunk_config(
            cutlass_dtype, num_cols,
            num_copy_bits=num_copy_bits, num_rows=num_rows,
            num_sms=num_sms,
        )

    # --- Compile kernel ---
    compiled_kernel = _get_compiled_kernel(
        kernel_variant, cutlass_dtype, chunk_size, k, next_n,
        num_copy_bits, ctas_per_group, num_sms, return_val,
    )

    # --- Allocate buffers ---
    # row_states: kernel resets its slots at end-of-kernel, so we only zero once.
    row_states_2d = _get_cached_row_states(
        f"{kernel_variant}_{device}", num_sms, state_size, device
    )
    # seq_lens: constant per (num_rows, num_cols) — cache to avoid GPU fill kernel.
    seq_lens = _get_cached_seq_lens(num_rows, num_cols, device)
    # output: must be fresh each call (kernel writes into it).
    output_indices_i32 = torch.empty(
        (num_rows, k), dtype=torch.int32, device=device
    )
    output_values = torch.empty(
        (num_rows, k), dtype=torch_dtype, device=device
    )

    # --- Execute kernel ---
    compiled_kernel(
        input,
        row_states_2d,
        seq_lens,
        output_indices_i32,
        None,  # return_val=False
    )

    # --- Gather values and convert indices ---
    indices = output_indices_i32.to(torch.int64)
    if k <= num_cols:
        values = torch.gather(input, dim=-1, index=indices)
    else:
        gather_indices = indices.clamp(min=0)
        values = torch.gather(input, dim=-1, index=gather_indices)
        values[indices < 0] = float("-inf")

    if sorted:
        values, sort_perm = torch.sort(values, dim=-1, descending=True)
        indices = torch.gather(indices, dim=-1, index=sort_perm)

    return values, indices
