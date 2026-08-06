# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Paged MQA logits: FP8 and FP4 attention-score kernels for Blackwell (SM100).

These kernels compute, for each batch element b and KV position pos:

    output[b*next_n + t, pos] = relu(Σ_h w[b*next_n+t, h] · (Q[b,t,h,:] @ K[pos,:]ᵀ)) · scale[pos]

where K is paged via block_table, used for sparse attention indexing in DeepSeek MLA.

Two variants:
  fp8_paged_mqa_logits — FP8 Q/K with per-token FP32 KV scales
  fp4_paged_mqa_logits — MXFP4 Q/K with per-(token, K-group) UE8M0 block scales

Both are SM100 (B200-class Blackwell) only.
"""

import functools
import importlib.util
from typing import Tuple

import numpy as np
import torch

from ..api_logging import flashinfer_api
from ..utils import (
    get_device_index,
    get_device_sm_count,
    supported_compute_capability,
)

# FP8 kernel epilogue supports fp32/fp16 only (matches TRT-LLM's validated
# surface); bf16 epi has no kernel branch and would silently miscompute.
_FP8_DTYPES = (torch.float32, torch.float16)
# FP4 kernel supports fp32/fp16/bf16 for epilogue and output.
_FP4_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@functools.cache
def _cached_num_sms(device_index: int) -> int:
    """Cache SM count per device — get_device_sm_count has non-trivial overhead."""
    return get_device_sm_count(torch.device("cuda", device_index))


def _validate_paged_inputs(
    context_lens: torch.Tensor, block_table: torch.Tensor
) -> None:
    """context_lens / block_table must be int32 CUDA tensors (the kernels are
    compiled against on-device Int32 fakes; a CPU or int64 tensor would make the
    kernel dereference a bad pointer or misread storage)."""
    if not (context_lens.is_cuda and context_lens.dtype == torch.int32):
        raise ValueError(
            f"context_lens must be an int32 CUDA tensor, got "
            f"dtype={context_lens.dtype}, device={context_lens.device}"
        )
    if not (block_table.is_cuda and block_table.dtype == torch.int32):
        raise ValueError(
            f"block_table must be an int32 CUDA tensor, got "
            f"dtype={block_table.dtype}, device={block_table.device}"
        )


def _validate_out(
    out: torch.Tensor,
    rows: int,
    aligned_ctx: int,
    device: torch.device,
    out_dtype: torch.dtype,
) -> None:
    """Validate a caller-provided ``out=`` buffer.

    The kernel writes UNCONDITIONALLY into the SPLIT_KV-aligned trailing region,
    so ``out`` must have at least ``aligned_ctx`` columns (use
    :func:`aligned_context_len`) and ``rows`` rows — otherwise the store spills
    past each row / past the buffer (silent corruption or illegal address)."""
    if out.device != device:
        raise ValueError(f"out.device ({out.device}) must match q.device ({device})")
    if out.dtype != out_dtype:
        raise ValueError(
            f"out.dtype ({out.dtype}) must match output_dtype ({out_dtype})"
        )
    if out.dim() != 2 or out.shape[0] < rows or out.shape[1] < aligned_ctx:
        raise ValueError(
            f"out must be at least ({rows}, {aligned_ctx}); the kernel writes into "
            f"the SPLIT_KV=256-aligned trailing region. Use "
            f"aligned_context_len(max_context_len) for the column count. "
            f"Got shape {tuple(out.shape)}."
        )


_CUTE_DSL_AVAILABLE = (
    importlib.util.find_spec("cutlass") is not None
    and importlib.util.find_spec("cutlass.cute") is not None
)

# SM100 / SM103 only (B200-class Blackwell)
_SM100_CCS = [100, 103]

# ──────────────────────────────────────────────────────────────────────────────
# Schedule-metadata computation (pure Python, mirrors DeepGEMM scheduler)
# ──────────────────────────────────────────────────────────────────────────────

_COMPUTE_BLOCK_KV = 128  # kernel's fixed compute tile (immutable)
_NUM_MATH_WG = 2  # warp groups per CTA; kernel multiplies col-1 by this


def _compute_schedule_metadata(
    context_lens_cpu: torch.Tensor,
    num_ctas: int,
) -> torch.Tensor:
    """Return [num_ctas+1, 2] int32 on CPU.

    Each row (q_idx, kv_split_half) marks a CTA boundary.  The kernel
    multiplies col-1 by NUM_MATH_WG=2 internally to get block-granularity
    kv_idx.  Algorithm mirrors DeepGEMM's PagedMQALogitsScheduler.

    Implemented with vectorized numpy to avoid Python-loop overhead (~150
    torch.tensor() allocations per call that otherwise cost ~600µs).
    """
    ctx_np = context_lens_cpu.numpy().astype(np.int64)
    num_kv = (ctx_np + _COMPUTE_BLOCK_KV - 1) // _COMPUTE_BLOCK_KV
    splits = (num_kv + 1) // 2  # ceil_div(num_kv, NUM_MATH_WG)

    total = int(splits.sum())
    q_div, r_mod = divmod(total, num_ctas)
    batch_size = len(splits)

    # Cumulative splits: cum[j] = sum(splits[0..j-1]), cum[0]=0
    cum = np.concatenate([[0], np.cumsum(splits)])  # [B+1]

    # For each CTA boundary i, compute the target = total splits before CTA i
    i_vals = np.arange(num_ctas + 1, dtype=np.int64)
    targets = i_vals * q_div + np.minimum(i_vals, r_mod)  # [num_ctas+1]

    # seq_idx[i] = number of fully-assigned sequences before CTA i
    # searchsorted(cum[1:], target, 'right') → first j where cum[j+1] > target
    seq_idx = np.searchsorted(cum[1:], targets, side="right")  # [num_ctas+1]

    # Clamp and build sentinel mask
    out_of_range = seq_idx >= batch_size
    seq_idx_clamped = np.minimum(seq_idx, batch_size - 1) if batch_size > 0 else seq_idx

    # local[i] = target[i] - cum[seq_idx[i]]  (offset within current sequence)
    local = targets - cum[seq_idx_clamped]

    # Apply sentinel: when all sequences done, row = (batch_size, 0)
    seq_out = np.where(out_of_range, batch_size, seq_idx).astype(np.int32)
    loc_out = np.where(out_of_range, 0, local).astype(np.int32)

    schedule = torch.empty((num_ctas + 1, 2), dtype=torch.int32)
    schedule[:, 0] = torch.from_numpy(seq_out)
    schedule[:, 1] = torch.from_numpy(loc_out)
    return schedule


# ──────────────────────────────────────────────────────────────────────────────
# Torch ↔ CuTe DSL dtype helpers
# ──────────────────────────────────────────────────────────────────────────────

if _CUTE_DSL_AVAILABLE:
    import cutlass
    import cutlass.cute as cute

    _TORCH_TO_CUTLASS = {
        torch.float32: cutlass.Float32,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }

    def _to_cutlass(dtype: torch.dtype):
        try:
            return _TORCH_TO_CUTLASS[dtype]
        except KeyError:
            raise ValueError(
                f"Unsupported dtype for paged_mqa_logits: {dtype}"
            ) from None


# ──────────────────────────────────────────────────────────────────────────────
# FP8 kernel: compile cache + source-file tracker
# ──────────────────────────────────────────────────────────────────────────────


@functools.cache
def _fp8_source_files() -> Tuple[str, ...]:
    from .kernels import fp8_paged_mqa_logits as _m

    return (__file__, _m.__file__)


@functools.cache
def _compile_fp8_kernel(
    phys_block_kv: int,
    num_heads: int,
    head_dim: int,
    next_n: int,
    num_sms: int,
    epi_dtype,  # cutlass dtype object
    acc_dtype,  # cutlass dtype object
    output_dtype,  # cutlass dtype object
    num_epi_subtiles: int,
):
    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from .kernels import FP8MQALogitsKernel

    N = next_n * num_heads
    block_bytes = phys_block_kv * (head_dim + 4)

    sym_npb = cute.sym_int()
    sym_B = cute.sym_int()
    max_ctx = cute.sym_int()
    max_blocks = cute.sym_int()
    num_ctas_sym = cute.sym_int()

    kv_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_npb, block_bytes), stride_order=(1, 0)
    )
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (N, head_dim, sym_B), stride_order=(1, 0, 2)
    )
    w_dtype = cutlass.Float16 if epi_dtype == cutlass.Float16 else epi_dtype
    w_fake = cute.runtime.make_fake_compact_tensor(
        w_dtype, (N, sym_B), stride_order=(0, 1)
    )
    logits_fake = cute.runtime.make_fake_tensor(
        output_dtype,
        (cute.sym_int(), max_ctx),
        stride=(cute.sym_int64(), 1),
    )
    bt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_B, max_blocks), stride_order=(1, 0)
    )
    cl_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_B,), stride_order=(0,)
    )
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_ctas_sym, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = FP8MQALogitsKernel(
        block_kv=_COMPUTE_BLOCK_KV,
        phys_block_kv=phys_block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        next_n=next_n,
        num_sms=num_sms,
        num_epi_subtiles=num_epi_subtiles,
        epi_dtype=epi_dtype,
        acc_dtype=acc_dtype,
        output_dtype=output_dtype,
    )

    def _compile_fn():
        return cute.compile(
            kernel,
            kv_fake,
            q_fake,
            w_fake,
            logits_fake,
            bt_fake,
            cl_fake,
            sm_fake,
            cutlass.Int32(1),
            cutlass.Int32(1),
            fake_stream,
            options="--enable-tvm-ffi",
        )

    tag = (
        f"fp8_pbk{phys_block_kv}_H{num_heads}_D{head_dim}_nn{next_n}"
        f"_sms{num_sms}_epi{epi_dtype}_acc{acc_dtype}_out{output_dtype}"
        f"_sub{num_epi_subtiles}"
    )
    return build_and_load_cute_dsl_kernel(
        "attn_scores_fp8",
        tag,
        _compile_fn,
        extra_key_files=_fp8_source_files(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# FP4 kernel: compile cache + source-file tracker
# ──────────────────────────────────────────────────────────────────────────────


@functools.cache
def _fp4_source_files() -> Tuple[str, ...]:
    from .kernels import fp4_paged_mqa_logits as _m

    return (__file__, _m.__file__)


@functools.cache
def _compile_fp4_kernel(
    phys_block_kv: int,
    num_heads: int,
    head_dim: int,
    next_n: int,
    num_sms: int,
    epi_dtype,  # cutlass dtype object
    output_dtype,  # cutlass dtype object
    num_epi_subtiles: int,
    remove_online_sf_transpose: bool,
):
    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from .kernels import FP4MQALogitsKernel

    N = next_n * num_heads
    half_D = head_dim // 2
    block_bytes = phys_block_kv * (half_D + 4)

    sym_npb = cute.sym_int()
    sym_B = cute.sym_int()
    max_ctx = cute.sym_int()
    max_blocks = cute.sym_int()
    num_ctas_sym = cute.sym_int()

    kv_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_npb, block_bytes), stride_order=(1, 0)
    )
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (N, half_D, sym_B), stride_order=(1, 0, 2)
    )
    sf_q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (N, sym_B), stride_order=(0, 1)
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        epi_dtype, (N, sym_B), stride_order=(0, 1)
    )
    logits_fake = cute.runtime.make_fake_tensor(
        output_dtype,
        (cute.sym_int(), max_ctx),
        stride=(cute.sym_int64(), 1),
    )
    bt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_B, max_blocks), stride_order=(1, 0)
    )
    cl_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_B,), stride_order=(0,)
    )
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_ctas_sym, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = FP4MQALogitsKernel(
        block_kv=_COMPUTE_BLOCK_KV,
        phys_block_kv=phys_block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        next_n=next_n,
        num_sms=num_sms,
        num_epi_subtiles=num_epi_subtiles,
        epi_dtype=epi_dtype,
        output_dtype=output_dtype,
        remove_online_sf_transpose=remove_online_sf_transpose,
    )

    def _compile_fn():
        return cute.compile(
            kernel,
            kv_fake,
            q_fake,
            sf_q_fake,
            w_fake,
            logits_fake,
            bt_fake,
            cl_fake,
            sm_fake,
            cutlass.Int32(1),
            cutlass.Int32(1),
            fake_stream,
            options="--enable-tvm-ffi",
        )

    tag = (
        f"fp4_pbk{phys_block_kv}_H{num_heads}_D{head_dim}_nn{next_n}"
        f"_sms{num_sms}_epi{epi_dtype}_out{output_dtype}"
        f"_sub{num_epi_subtiles}_sfT{int(remove_online_sf_transpose)}"
    )
    return build_and_load_cute_dsl_kernel(
        "attn_scores_fp4",
        tag,
        _compile_fn,
        extra_key_files=_fp4_source_files(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

_SPLIT_KV = _COMPUTE_BLOCK_KV * _NUM_MATH_WG  # 256 — output alignment granularity


def _gpu_schedule(
    context_lens: torch.Tensor,
    schedule_meta: torch.Tensor,
    num_sms: int,
) -> None:
    """Run GPU schedule kernel in-place: fills schedule_meta from context_lens.

    Both tensors must already be on the same CUDA device.
    schedule_meta must be [num_sms+1, 2] int32.
    """
    from .kernels.schedule_kernel import _compile_schedule_kernel

    batch_size = int(context_lens.shape[0])
    aligned_b = max(((batch_size + 31) // 32) * 32, 32)
    compiled = _compile_schedule_kernel(aligned_b, _SPLIT_KV, num_sms)
    compiled(context_lens, schedule_meta, batch_size)


def aligned_context_len(max_context_len: int) -> int:
    """Return the minimum allocated context dimension for paged MQA logits output.

    The kernel may write unconditionally into SPLIT_KV-aligned trailing positions,
    so the output tensor must be allocated with at least this many columns.

    Use this to pre-allocate the ``out`` parameter:
        out = torch.empty((B * next_n, aligned_context_len(max_ctx)), dtype=..., device="cuda")
        logits = fp8_paged_mqa_logits(..., out=out)
    """
    return ((max_context_len + _SPLIT_KV - 1) // _SPLIT_KV) * _SPLIT_KV


def compute_paged_mqa_logits_schedule(
    context_lens: torch.Tensor,
    device: torch.device = None,
    *,
    use_gpu_kernel: bool = True,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Compute the CTA schedule tensor for paged MQA logits kernels.

    Returns [num_sms+1, 2] int32 on CUDA, ready to pass as ``schedule_meta``
    to :func:`fp8_paged_mqa_logits` or :func:`fp4_paged_mqa_logits`.

    Args:
        context_lens:   [B] int32, on CPU or CUDA.
        device:         target CUDA device. Defaults to context_lens.device
                        (or cuda:0 if CPU).
        use_gpu_kernel: if True (default), compute entirely on-GPU via
                        :class:`PagedMQALogitsScheduleKernel` — no D2H copy,
                        CUDA-graph-capturable.  Falls back to CPU numpy if
                        CuTe DSL is unavailable.
        out:            optional pre-allocated [num_sms+1, 2] int32 on CUDA.
                        Required for CUDA-graph capture (static buffer).
                        If None, a new tensor is allocated each call.

    Returns:
        schedule_meta: [num_sms+1, 2] int32 on CUDA (``out`` if provided).
    """
    if device is None:
        device = context_lens.device if context_lens.is_cuda else torch.device("cuda")
    device = torch.device(device)
    num_sms = _cached_num_sms(get_device_index(device))

    if use_gpu_kernel and _CUTE_DSL_AVAILABLE and context_lens.is_cuda:
        if out is None:
            out = torch.empty((num_sms + 1, 2), dtype=torch.int32, device=device)
        _gpu_schedule(context_lens, out, num_sms)
        return out

    # CPU fallback: D2H copy + numpy schedule + H2D copy
    result = _compute_schedule_metadata(context_lens.cpu(), num_sms).to(device)
    if out is not None:
        out.copy_(result)
        return out
    return result


@supported_compute_capability(_SM100_CCS)
@flashinfer_api
def fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
    *,
    output_dtype: torch.dtype = torch.float32,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    num_epi_subtiles: int = 1,
    schedule_meta: torch.Tensor = None,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """FP8 paged MQA logits for Blackwell (SM100).

    Args:
        q:               [B, next_n, H, D]  float8_e4m3fn
        kv_fused:        [num_blocks, phys_block_kv, 1, D+4]  uint8
                         Layout per block: [KV data (phys_block_kv*D bytes)]
                                           [scales (phys_block_kv*4 bytes, float32)]
        weights:         [B*next_n, H]  float32  per-head mixing weights
        context_lens:    [B]  int32  (CUDA)
        block_table:     [B, max_blocks]  int32  (CUDA)
        max_context_len: int  maximum KV sequence length
        output_dtype:    output tensor dtype (float32 or float16)
        epi_dtype:       epilogue accumulation dtype (float32 or float16)
        acc_dtype:       MMA accumulator dtype (float32 or float16)
        num_epi_subtiles: epilogue subtile count (perf knob, default 1)
        schedule_meta:   optional pre-computed [num_sms+1, 2] int32 CTA schedule
                         on CUDA.  If None, computed from context_lens each call.
                         Pass a pre-computed tensor to avoid the CPU overhead when
                         the schedule is stable across calls (e.g. fixed batch).
                         Use compute_paged_mqa_logits_schedule() to generate it.
        out:             optional pre-allocated output tensor [B*next_n, aligned_ctx]
                         where aligned_ctx >= max_context_len and is a multiple of
                         SPLIT_KV=256.  If None, allocated each call.
                         Use flashinfer.attn_scores.attn_scores.aligned_context_len()
                         to compute the required allocation size.
                         Required for CUDA graph capture.

    Returns:
        logits: [B*next_n, max_context_len]  output_dtype  (a view of ``out`` if provided)
    """
    if not _CUTE_DSL_AVAILABLE:
        raise RuntimeError("fp8_paged_mqa_logits requires nvidia-cutlass-dsl")

    B, next_n, H, D = q.shape
    N = next_n * H
    phys_block_kv = kv_fused.shape[1]
    num_phys_blocks = kv_fused.shape[0]
    num_sms = _cached_num_sms(get_device_index(q.device))

    if (
        output_dtype not in _FP8_DTYPES
        or epi_dtype not in _FP8_DTYPES
        or acc_dtype not in _FP8_DTYPES
    ):
        raise ValueError(
            "fp8_paged_mqa_logits supports output/epi/acc dtype in {float32, "
            f"float16}}; got output_dtype={output_dtype}, epi_dtype={epi_dtype}, "
            f"acc_dtype={acc_dtype}."
        )
    _validate_paged_inputs(context_lens, block_table)

    cutlass_epi = _to_cutlass(epi_dtype)
    cutlass_acc = _to_cutlass(acc_dtype)
    cutlass_out = _to_cutlass(output_dtype)

    # Reshape inputs to kernel convention (no .contiguous() — strides must stay
    # B-independent so the compile cache stays hot across different batch sizes)
    q_3d = q.reshape(B, N, D).permute(1, 2, 0)  # [N, D, B]
    if epi_dtype == torch.float16:
        w_2d = weights.reshape(B, N).half().t()  # [N, B]
    else:
        w_2d = weights.reshape(B, N).t()  # [N, B]
    kv_flat = kv_fused.reshape(num_phys_blocks, -1)  # [num_blocks, block_bytes]

    # Output: aligned to SPLIT_KV so the kernel can store unconditionally.
    # Use pre-allocated buffer if provided (avoids allocation, enables CUDA graphs).
    aligned_ctx = ((max_context_len + _SPLIT_KV - 1) // _SPLIT_KV) * _SPLIT_KV
    if out is not None:
        _validate_out(out, B * next_n, aligned_ctx, q.device, output_dtype)
        logits = out[:, :max_context_len]
    else:
        logits_full = torch.empty(
            (B * next_n, aligned_ctx), device=q.device, dtype=output_dtype
        )
        logits = logits_full[:, :max_context_len]

    # Schedule metadata — use caller's precomputed tensor or compute it now.
    # GPU kernel is used by default when context_lens is on CUDA (no D2H copy).
    if schedule_meta is None:
        schedule_meta = compute_paged_mqa_logits_schedule(context_lens, device=q.device)

    compiled = _compile_fp8_kernel(
        phys_block_kv,
        H,
        D,
        next_n,
        num_sms,
        cutlass_epi,
        cutlass_acc,
        cutlass_out,
        num_epi_subtiles,
    )

    # FP8 tensor passed as uint8 view (DLPack lacks float8 support)
    q_for_ffi = (
        q_3d.view(torch.uint8)
        if q_3d.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        else q_3d
    )
    compiled(
        kv_flat,
        q_for_ffi,
        w_2d,
        logits,
        block_table,
        context_lens,
        schedule_meta,
        num_phys_blocks,
        B,
    )
    return logits


@supported_compute_capability(_SM100_CCS)
@flashinfer_api
def fp4_paged_mqa_logits(
    q: torch.Tensor,
    sf_q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    epi_dtype: torch.dtype = torch.float32,
    num_epi_subtiles: int = 1,
    remove_online_sf_transpose: bool = False,
    schedule_meta: torch.Tensor = None,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """FP4 (MXFP4) paged MQA logits for Blackwell (SM100).

    Args:
        q:               [B, next_n, H, D//2]  uint8 (two FP4 per byte, E2M1)
        sf_q:            [B, next_n, H]  int32 (4 UE8M0 scale factors packed per token)
        kv_fused:        [num_blocks, phys_block_kv, 1, D//2+4]  uint8
                         Layout per block: [KV data (phys_block_kv*D//2 bytes)]
                                           [KV SF   (phys_block_kv*4 bytes, int32)]
        weights:         [B*next_n, H]  float32  per-head mixing weights
        context_lens:    [B]  int32  (CUDA)
        block_table:     [B, max_blocks]  int32  (CUDA)
        max_context_len: int  maximum KV sequence length
        output_dtype:    output tensor dtype (float32, float16, or bfloat16)
        epi_dtype:       epilogue dtype (float32, float16, or bfloat16)
        num_epi_subtiles: epilogue subtile count (perf knob, default 1)
        remove_online_sf_transpose: if True, skip in-kernel SF SMEM transpose
                         (requires KV SF pre-arranged in UTCCP chunk layout,
                         phys_block_kv=128 only)
        schedule_meta:   optional pre-computed [num_sms+1, 2] int32 CTA schedule
                         on CUDA.  If None, computed from context_lens each call.
                         Pass a pre-computed tensor to avoid the CPU overhead when
                         the schedule is stable across calls.
                         Use compute_paged_mqa_logits_schedule() to generate it.
        out:             optional pre-allocated output tensor [B*next_n, aligned_ctx].
                         Required for CUDA graph capture.

    Returns:
        logits: [B*next_n, max_context_len]  output_dtype

    Note:
        next_n=4 is handled internally via atom-split (2B × next_n=2) so callers
        can pass next_n=4 directly. next_n ∈ {1,2,3} are natively supported.
    """
    if not _CUTE_DSL_AVAILABLE:
        raise RuntimeError("fp4_paged_mqa_logits requires nvidia-cutlass-dsl")

    B, next_n, H, half_D = q.shape
    D = half_D * 2
    phys_block_kv = kv_fused.shape[1]
    num_phys_blocks = kv_fused.shape[0]
    num_sms = _cached_num_sms(get_device_index(q.device))

    if output_dtype not in _FP4_DTYPES or epi_dtype not in _FP4_DTYPES:
        raise ValueError(
            "fp4_paged_mqa_logits supports output/epi dtype in {float32, "
            f"float16, bfloat16}}; got output_dtype={output_dtype}, "
            f"epi_dtype={epi_dtype}."
        )
    _validate_paged_inputs(context_lens, block_table)

    cutlass_epi = _to_cutlass(epi_dtype)
    cutlass_out = _to_cutlass(output_dtype)

    # next_n=4 is not natively supported (TMEM cap). Decompose as caller-side
    # atom-split: [B,4,H,half_D] → [2B,2,H,half_D], duplicate block_table,
    # split context_lens into (ctx-2, ctx) pairs to preserve causal mask.
    if next_n == 4:
        exp_B = B * 2
        kernel_q = q.reshape(exp_B, 2, H, half_D)
        kernel_sf_q = sf_q.reshape(exp_B, 2, H)
        ctx_pair = torch.stack([context_lens - 2, context_lens], dim=1)  # [B, 2]
        kernel_ctx_lens = ctx_pair.reshape(exp_B).contiguous()
        kernel_block_table = block_table.repeat_interleave(2, dim=0)
        kernel_next_n = 2
        kernel_B = exp_B
    else:
        kernel_q = q
        kernel_sf_q = sf_q
        kernel_ctx_lens = context_lens
        kernel_block_table = block_table
        kernel_next_n = next_n
        kernel_B = B

    kernel_N = kernel_next_n * H

    # Reshape to kernel convention
    q_3d = kernel_q.reshape(kernel_B, kernel_N, half_D).permute(
        1, 2, 0
    )  # [kernel_N, D//2, kernel_B]
    sf_q_2d = kernel_sf_q.reshape(kernel_B, kernel_N).t()  # [kernel_N, kernel_B]
    # weights [B*next_n, H] → [kernel_B, kernel_N] → [kernel_N, kernel_B]
    # For next_n=4: [4B, H].reshape(2B, 2H) groups the original (b,t) pairs correctly.
    if epi_dtype == torch.float16:
        w_2d = weights.reshape(kernel_B, kernel_N).half().t()
    elif epi_dtype == torch.bfloat16:
        w_2d = weights.reshape(kernel_B, kernel_N).bfloat16().t()
    else:
        w_2d = weights.reshape(kernel_B, kernel_N).t()
    kv_flat = kv_fused.reshape(num_phys_blocks, -1)

    aligned_ctx = ((max_context_len + _SPLIT_KV - 1) // _SPLIT_KV) * _SPLIT_KV
    if out is not None:
        _validate_out(out, B * next_n, aligned_ctx, q.device, output_dtype)
        logits = out[:, :max_context_len]
    else:
        logits_full = torch.empty(
            (B * next_n, aligned_ctx), device=q.device, dtype=output_dtype
        )
        logits = logits_full[:, :max_context_len]

    # Schedule metadata — route through the shared helper so the on-GPU schedule
    # kernel is used when kernel_ctx_lens is on CUDA (no D2H copy; CUDA-graph
    # capturable), matching fp8_paged_mqa_logits.
    if schedule_meta is None:
        schedule_meta = compute_paged_mqa_logits_schedule(
            kernel_ctx_lens, device=q.device
        )

    compiled = _compile_fp4_kernel(
        phys_block_kv,
        H,
        D,
        kernel_next_n,
        num_sms,
        cutlass_epi,
        cutlass_out,
        num_epi_subtiles,
        remove_online_sf_transpose,
    )
    compiled(
        kv_flat,
        q_3d,
        sf_q_2d,
        w_2d,
        logits,
        kernel_block_table,
        kernel_ctx_lens,
        schedule_meta,
        num_phys_blocks,
        kernel_B,
    )
    return logits


# ──────────────────────────────────────────────────────────────────────────────
# Pre-compilation helper (Item 5: AOT warm-up for common configs)
# ──────────────────────────────────────────────────────────────────────────────


def precompile_paged_mqa_logits(device: torch.device = None) -> None:
    """Pre-compile paged MQA logits kernels for common static configs.

    Populates the on-disk CuTe-DSL kernel cache so subsequent calls to
    :func:`fp8_paged_mqa_logits` and :func:`fp4_paged_mqa_logits` skip
    compilation on first use.  Call once during deployment setup or as part
    of a package-build step.

    Args:
        device: CUDA device to target.  Defaults to cuda:0.
    """
    if not _CUTE_DSL_AVAILABLE:
        return
    if device is None:
        device = torch.device("cuda", 0)
    num_sms = _cached_num_sms(get_device_index(torch.device(device)))
    num_heads, head_dim = 64, 128

    # FP8 common configs: phys_block_kv × next_n, fp32 acc/epi/out
    fp8_cfgs = [(pbk, nn) for pbk in (64, 128) for nn in (1, 2, 3, 4)]
    for pbk, nn in fp8_cfgs:
        _compile_fp8_kernel(
            pbk,
            num_heads,
            head_dim,
            nn,
            num_sms,
            _to_cutlass(torch.float32),
            _to_cutlass(torch.float32),
            _to_cutlass(torch.float32),
            1,
        )

    # FP4 common configs: phys_block_kv × next_n, fp32 epi, bf16 out
    fp4_cfgs = [(pbk, nn) for pbk in (32, 64, 128) for nn in (1, 2, 3)]
    for pbk, nn in fp4_cfgs:
        _compile_fp4_kernel(
            pbk,
            num_heads,
            head_dim,
            nn,
            num_sms,
            _to_cutlass(torch.float32),
            _to_cutlass(torch.bfloat16),
            1,
            False,
        )
