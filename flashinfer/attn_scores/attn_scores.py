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
from ..trace.templates.attn_scores import (
    fp4_paged_mqa_logits_trace,
    fp8_paged_mqa_logits_trace,
)
from ..utils import (
    backend_requirement,
    get_device_index,
    get_device_sm_count,
    supported_compute_capability,
)

# FP8 kernel epilogue supports fp32/fp16 only (matches TRT-LLM's validated
# surface); bf16 epi has no kernel branch and would silently miscompute.
_FP8_DTYPES = (torch.float32, torch.float16)
# FP4 kernel supports fp32/fp16/bf16 for epilogue and output.
_FP4_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
# FP8 UMMA instruction K. The kernel derives mma_inst_tile_k = head_dim // this
# via integer division, so head_dim must be an exact multiple (see the guard in
# fp8_paged_mqa_logits).
_FP8_MMA_INST_K = 32
# FP4 hardcodes these in FP4MQALogitsKernel.__init__ (asserts on head_dim and
# num_heads); mirrored at the API boundary for a clearer error.
_FP4_REQUIRED_HEAD_DIM = 128
_FP4_REQUIRED_NUM_HEADS = 64
# Max TMA sub-copies per compute tile (kernel: `num_blocks_per_mma <= 4`).
_MAX_BLOCKS_PER_MMA = 4
# UMMA N-mode limits, reported by the DSL as
# "expects the N-mode to satisfy 8 <= N <= 256 and N % 8 == 0".
# N here is next_n * num_heads.
_MMA_N_MIN, _MMA_N_MAX, _MMA_N_MULTIPLE = 8, 256, 8
# FP4 natively supports next_n in {1,2,3}; next_n=4 is handled by this wrapper
# via caller-side atom-split (2B x next_n=2), so 4 is supported at the API level.
_FP4_MAX_NEXT_N = 4


@functools.cache
def _cached_num_sms(device_index: int) -> int:
    """Cache SM count per device — get_device_sm_count has non-trivial overhead."""
    return get_device_sm_count(torch.device("cuda", device_index))


def _validate_paged_inputs(
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    batch_size: int,
) -> None:
    """Validate context_lens / block_table (cheap host-side checks; no device sync).

    They must be int32 CUDA tensors (the kernels are compiled against on-device
    Int32 fakes; a CPU or int64 tensor would make the kernel dereference a bad
    pointer or misread storage) and have exactly ``batch_size`` rows.

    NOTE (caller invariant, not checked here — would need a device sync): the
    kernel reads ceil(context_lens[b]/128) compute tiles * (128 // block_size)
    physical blocks per row, so ``block_table`` must have at least
    ``max_b ceil(context_lens[b]/128) * (128 // block_size)`` columns. A
    narrower block_table causes an out-of-bounds read. Verifying this cheaply is
    impossible without a D2H copy of context_lens (which would break CUDA-graph
    capture), so it is the caller's responsibility."""
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
    if context_lens.shape[0] != batch_size:
        raise ValueError(
            f"context_lens.shape[0] ({context_lens.shape[0]}) must equal "
            f"batch_size ({batch_size}) inferred from q.shape[0]"
        )
    if block_table.dim() != 2 or block_table.shape[0] != batch_size:
        raise ValueError(
            f"block_table must be 2-D with shape[0] == batch_size ({batch_size}) "
            f"inferred from q.shape[0]; got shape {tuple(block_table.shape)}"
        )


def _validate_schedule_meta(schedule_meta: torch.Tensor, num_sms: int, device) -> None:
    """A caller-supplied schedule_meta must be an int32 CUDA [num_sms+1, 2] tensor;
    a smaller one causes an out-of-bounds schedule read in the kernel."""
    if not (schedule_meta.is_cuda and schedule_meta.dtype == torch.int32):
        raise ValueError(
            f"schedule_meta must be an int32 CUDA tensor, got "
            f"dtype={schedule_meta.dtype}, device={schedule_meta.device}"
        )
    if tuple(schedule_meta.shape) != (num_sms + 1, 2):
        raise ValueError(
            f"schedule_meta must have shape ({num_sms + 1}, 2) for this device; "
            f"got {tuple(schedule_meta.shape)}. Use compute_paged_mqa_logits_schedule()."
        )


def _validate_out(
    out: torch.Tensor,
    rows: int,
    padded_ctx_len: int,
    device: torch.device,
    out_dtype: torch.dtype,
) -> None:
    """Validate a caller-provided ``out=`` buffer.

    The kernel writes UNCONDITIONALLY into the SPLIT_KV-padded trailing region,
    so ``out`` must have at least ``padded_ctx_len`` columns (use
    :func:`padded_context_len`) and ``rows`` rows — otherwise the store spills
    past each row / past the buffer (silent corruption or illegal address)."""
    if out.device != device:
        raise ValueError(f"out.device ({out.device}) must match q.device ({device})")
    if out.dtype != out_dtype:
        raise ValueError(
            f"out.dtype ({out.dtype}) must match output_dtype ({out_dtype})"
        )
    if out.dim() != 2 or out.shape[0] < rows or out.shape[1] < padded_ctx_len:
        raise ValueError(
            f"out must be at least ({rows}, {padded_ctx_len}); the kernel writes into "
            f"the SPLIT_KV=256-padded trailing region. Use "
            f"padded_context_len(max_context_len) for the column count. "
            f"Got shape {tuple(out.shape)}."
        )


def _validate_phys_block_kv(block_size: int, fn_name: str) -> None:
    """Validate the physical KV page size, taken from ``kv_fused.shape[1]``.

    Both kernels tile KV in a fixed ``_COMPUTE_BLOCK_KV``-token compute tile and
    issue ``_COMPUTE_BLOCK_KV // block_size`` TMA sub-copies to fill it,
    capped at ``_MAX_BLOCKS_PER_MMA``. So the page size must divide the compute
    tile and not be too small. Measured on sm_100a: {32, 64, 128} work, while
    16 (too many sub-copies) and 48 / 96 / 256 (not divisors) each trip a bare
    assertion from inside kernel construction.
    """
    if (
        block_size <= 0
        or _COMPUTE_BLOCK_KV % block_size != 0
        or _COMPUTE_BLOCK_KV // block_size > _MAX_BLOCKS_PER_MMA
    ):
        supported = [
            b
            for b in range(1, _COMPUTE_BLOCK_KV + 1)
            if _COMPUTE_BLOCK_KV % b == 0
            and _COMPUTE_BLOCK_KV // b <= _MAX_BLOCKS_PER_MMA
        ]
        raise ValueError(
            f"{fn_name}: block_size (kv_fused.shape[1]) must divide the "
            f"{_COMPUTE_BLOCK_KV}-token compute tile into at most "
            f"{_MAX_BLOCKS_PER_MMA} sub-blocks; supported values are {supported}, "
            f"got {block_size}."
        )


def _fp8_smem_bytes(block_kv: int, head_dim: int, n: int, epi_bytes: int) -> int:
    """Predict the FP8 kernel's per-CTA shared-memory usage, in bytes.

    Mirrors ``FP8MQALogitsKernel.__init__`` for the configuration this wrapper
    always builds (``max_kv_pipeline=False`` -> 3 KV stages, 3 Q stages, both
    math groups resident).  Verified against the driver: head_dim=256 with
    block_kv=128, n=64, fp32 epilogue predicts 249856 B and the launch reports
    "Allocated: 249856 bytes".
    """
    num_kv_stages = 3
    num_q_stages = 3
    # KV + per-token fp32 scales, ×2 math groups.
    kv_scale_per_stage = 2 * (block_kv * head_dim + block_kv * 4)
    # Weights are padded to a 128 B stage stride for TMA alignment.
    w_stage_stride = ((n * epi_bytes + 127) // 128 * 128) // epi_bytes
    qw_per_stage = n * head_dim + w_stage_stride * epi_bytes
    barriers = 256
    return barriers + kv_scale_per_stage * num_kv_stages + qw_per_stage * num_q_stages


@functools.cache
def _cached_max_smem_per_block(device_index: int) -> int:
    """Opt-in per-CTA SMEM cap for the device (232448 B on sm_100a)."""
    props = torch.cuda.get_device_properties(torch.device("cuda", device_index))
    for attr in ("shared_memory_per_block_optin", "sharedMemPerBlockOptin"):
        val = getattr(props, attr, None)
        if val:
            return int(val)
    return 232448  # sm_100a fallback


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
def _cached_fp8_source_files() -> Tuple[str, ...]:
    from .kernels import fp8_paged_mqa_logits as _m

    return (__file__, _m.__file__)


@functools.cache
def _cached_compile_fp8_kernel(
    block_size: int,
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
    block_bytes = block_size * (head_dim + 4)

    sym_npb = cute.sym_int()
    sym_B = cute.sym_int()
    max_ctx = cute.sym_int()
    max_blocks = cute.sym_int()
    num_ctas_sym = cute.sym_int()

    # KV may come from a K-cache pool view that is strided in dim 0 (pool
    # layouts interleave layers, e.g. [num_blocks, num_layers, kvFactor,
    # block_bytes]). Declare the outer stride as a symbol so the actual
    # per-block stride is read at runtime; the innermost stride is fixed to 1
    # (bytes are contiguous within one logical block). A compact-tensor
    # declaration would bake block_bytes in as the outer stride and reject
    # such a view at the FFI boundary. Matches TensorRT-LLM's production
    # CuteDSLPagedMQALogitsRunner._compile.
    kv_fake = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (sym_npb, block_bytes),
        stride=(cute.sym_int64(), 1),
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
        phys_block_kv=block_size,  # kernel kwarg name is fixed (verbatim TRT-LLM port)
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
        f"fp8_bs{block_size}_H{num_heads}_D{head_dim}_nn{next_n}"
        f"_sms{num_sms}_epi{epi_dtype}_acc{acc_dtype}_out{output_dtype}"
        f"_sub{num_epi_subtiles}"
    )
    return build_and_load_cute_dsl_kernel(
        "attn_scores_fp8",
        tag,
        _compile_fn,
        extra_key_files=_cached_fp8_source_files(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# FP4 kernel: compile cache + source-file tracker
# ──────────────────────────────────────────────────────────────────────────────


@functools.cache
def _cached_fp4_source_files() -> Tuple[str, ...]:
    from .kernels import fp4_paged_mqa_logits as _m

    return (__file__, _m.__file__)


@functools.cache
def _cached_compile_fp4_kernel(
    block_size: int,
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
    block_bytes = block_size * (half_D + 4)

    sym_npb = cute.sym_int()
    sym_B = cute.sym_int()
    max_ctx = cute.sym_int()
    max_blocks = cute.sym_int()
    num_ctas_sym = cute.sym_int()

    # Symbolic outer stride so a strided K-cache pool view works zero-copy;
    # see the matching comment in _cached_compile_fp8_kernel.
    kv_fake = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (sym_npb, block_bytes),
        stride=(cute.sym_int64(), 1),
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
        phys_block_kv=block_size,  # kernel kwarg name is fixed (verbatim TRT-LLM port)
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
        f"fp4_bs{block_size}_H{num_heads}_D{head_dim}_nn{next_n}"
        f"_sms{num_sms}_epi{epi_dtype}_out{output_dtype}"
        f"_sub{num_epi_subtiles}_sfT{int(remove_online_sf_transpose)}"
    )
    return build_and_load_cute_dsl_kernel(
        "attn_scores_fp4",
        tag,
        _compile_fn,
        extra_key_files=_cached_fp4_source_files(),
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


def padded_context_len(max_context_len: int) -> int:
    """Return the minimum allocated context dimension for paged MQA logits output.

    The kernel may write unconditionally into SPLIT_KV-padded trailing positions,
    so the output tensor must be allocated with at least this many columns.

    Use this to pre-allocate the ``out`` parameter:
        out = torch.empty((B * next_n, padded_context_len(max_ctx)), dtype=..., device="cuda")
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
def _check_fp8_paged_mqa_logits_supported(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
    output_dtype: torch.dtype = torch.float32,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    num_epi_subtiles: int = 1,
    schedule_meta: torch.Tensor = None,
    out: torch.Tensor = None,
) -> bool:
    """Return True when the FP8 kernel supports this problem, else raise ``ValueError``.

    Signature mirrors :func:`fp8_paged_mqa_logits` (``backend_requirement`` binds
    the public arguments and forwards them all by keyword); ``schedule_meta`` and
    ``out`` are accepted but validated in the API body instead — they describe
    caller-supplied output storage rather than problem supportedness, and must
    stay enforced even under ``skip_check=True`` because a too-small buffer is a
    silent out-of-bounds write.
    """
    B, next_n, H, D = q.shape
    N = next_n * H
    block_size = kv_fused.shape[1]

    if q.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"fp8_paged_mqa_logits requires q.dtype == float8_e4m3fn; got {q.dtype}. "
            "(e5m2 has a different byte layout and would be silently misread as e4m3.)"
        )
    if num_epi_subtiles < 1:
        raise ValueError(f"num_epi_subtiles must be >= 1, got {num_epi_subtiles}")
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
    # --- head_dim supportedness -------------------------------------------
    # The FP8 kernel is parametric over head_dim (unlike FP4, which hardcodes
    # 128), but two conditions bound it. Both are checked here so the caller
    # gets an actionable error instead of a silently-wrong result or a bare
    # cudaErrorInvalidValue from the driver at launch.
    #
    # (1) Multiple of the MMA instruction K. The kernel computes
    #     mma_inst_tile_k = head_dim // 32 with integer division, so a
    #     non-multiple silently truncates the QK contraction (head_dim=100
    #     would reduce over only 96 elements) and returns wrong logits.
    if D % _FP8_MMA_INST_K != 0:
        raise ValueError(
            f"head_dim must be a multiple of {_FP8_MMA_INST_K} (FP8 MMA instruction K); "
            f"got head_dim={D} from q.shape. A non-multiple would silently truncate the "
            f"QK contraction to {D // _FP8_MMA_INST_K * _FP8_MMA_INST_K} elements."
        )
    # (2) SMEM budget. Tile sizing scales linearly with head_dim; an oversized
    #     config fails at launch with an opaque driver error. Measured on
    #     sm_100a: head_dim <= 192 fits, 256 does not (249856 B > 232448 B).
    _smem_needed = _fp8_smem_bytes(
        _COMPUTE_BLOCK_KV, D, N, 2 if epi_dtype == torch.float16 else 4
    )
    _smem_limit = _cached_max_smem_per_block(get_device_index(q.device))
    if _smem_needed > _smem_limit:
        raise ValueError(
            f"head_dim={D} with num_heads={H}, next_n={next_n} needs {_smem_needed} B "
            f"of shared memory per CTA but this device allows {_smem_limit} B. "
            f"Reduce head_dim (<=192 fits at num_heads=64, next_n=1), num_heads, "
            f"or next_n."
        )
    # UMMA N-mode: N = next_n * num_heads must be a multiple of 8 in [8, 256].
    # Exceeding it fails inside the DSL with an opaque OpError (measured:
    # next_n=5 at num_heads=64 gives N=320 -> "expects the N-mode to satisfy
    # 8 <= N <= 256 and N % 8 == 0, but got 320").
    if N < _MMA_N_MIN or N > _MMA_N_MAX or N % _MMA_N_MULTIPLE != 0:
        raise ValueError(
            f"next_n * num_heads must be a multiple of {_MMA_N_MULTIPLE} in "
            f"[{_MMA_N_MIN}, {_MMA_N_MAX}] (UMMA N-mode); got next_n={next_n} * "
            f"num_heads={H} = {N}."
        )
    _validate_phys_block_kv(block_size, "fp8_paged_mqa_logits")
    if kv_fused.dim() != 4 or kv_fused.shape[2] != 1 or kv_fused.shape[-1] != D + 4:
        raise ValueError(
            f"kv_fused must be [num_blocks, block_size, 1, head_dim+4={D + 4}] "
            f"(head_dim={D} from q); got shape {tuple(kv_fused.shape)}"
        )
    _validate_paged_inputs(context_lens, block_table, B)
    return True


@backend_requirement({}, common_check=_check_fp8_paged_mqa_logits_supported)
@flashinfer_api(trace=fp8_paged_mqa_logits_trace)
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
        q:               [batch_size, next_n, num_heads, head_dim]  float8_e4m3fn
        kv_fused:        [num_blocks, block_size, 1, head_dim+4]  uint8
                         Per block: [KV data (block_size*head_dim bytes)]
                                    [scales (block_size*4 bytes, float32)]
        weights:         [batch_size*next_n, num_heads]  float32  per-head weights
        context_lens:    [batch_size]  int32  (CUDA)
        block_table:     [batch_size, max_blocks_per_seq]  int32  (CUDA)
                         Values are physical block indices into kv_fused's dim 0.
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
        out:             optional pre-allocated output
                         [batch_size*next_n, padded_ctx_len], where padded_ctx_len
                         >= max_context_len and is a multiple of SPLIT_KV=256.
                         If None, allocated each call.  Use padded_context_len()
                         to size it.  Required for CUDA graph capture.

    Returns:
        logits: [batch_size*next_n, max_context_len]  output_dtype
                (a view of ``out`` when provided)
    """
    if not _CUTE_DSL_AVAILABLE:
        raise RuntimeError("fp8_paged_mqa_logits requires nvidia-cutlass-dsl")

    B, next_n, H, D = q.shape
    block_size = kv_fused.shape[1]
    num_blocks = kv_fused.shape[0]
    num_sms = _cached_num_sms(get_device_index(q.device))

    cutlass_epi = _to_cutlass(epi_dtype)
    cutlass_acc = _to_cutlass(acc_dtype)
    cutlass_out = _to_cutlass(output_dtype)

    q_3d = q.reshape(B, next_n * H, D).permute(1, 2, 0)  # [next_n*H, D, B]
    if epi_dtype == torch.float16:
        w_2d = weights.reshape(B, next_n * H).half().t()  # [next_n*H, B]
    else:
        w_2d = weights.reshape(B, next_n * H).t()  # [next_n*H, B]
    kv_flat = kv_fused.flatten(1)  # [num_blocks, block_bytes]

    padded_ctx_len = ((max_context_len + _SPLIT_KV - 1) // _SPLIT_KV) * _SPLIT_KV
    if out is not None:
        _validate_out(out, B * next_n, padded_ctx_len, q.device, output_dtype)
        logits = out[:, :max_context_len]
    else:
        logits_full = torch.empty(
            (B * next_n, padded_ctx_len), device=q.device, dtype=output_dtype
        )
        logits = logits_full[:, :max_context_len]

    if schedule_meta is None:
        schedule_meta = compute_paged_mqa_logits_schedule(context_lens, device=q.device)
    else:
        _validate_schedule_meta(schedule_meta, num_sms, q.device)

    compiled = _cached_compile_fp8_kernel(
        block_size,
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
        num_blocks,
        B,
    )
    return logits


@supported_compute_capability(_SM100_CCS)
def _check_fp4_paged_mqa_logits_supported(
    q: torch.Tensor,
    sf_q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
    output_dtype: torch.dtype = torch.bfloat16,
    epi_dtype: torch.dtype = torch.float32,
    num_epi_subtiles: int = 1,
    remove_online_sf_transpose: bool = False,
    schedule_meta: torch.Tensor = None,
    out: torch.Tensor = None,
) -> bool:
    """Return True when the FP4 kernel supports this problem, else raise ``ValueError``.

    Mirrors :func:`fp4_paged_mqa_logits`'s signature; see
    :func:`_check_fp8_paged_mqa_logits_supported` for why ``schedule_meta`` and
    ``out`` are validated in the API body rather than here.
    """
    B, next_n, H, half_D = q.shape
    D = half_D * 2
    block_size = kv_fused.shape[1]

    if q.dtype != torch.uint8:
        raise ValueError(
            f"fp4_paged_mqa_logits requires q.dtype == uint8 (packed FP4 e2m1, two "
            f"per byte); got {q.dtype}"
        )
    if sf_q.dtype != torch.int32 or tuple(sf_q.shape) != (B, next_n, H):
        raise ValueError(
            f"sf_q must be an int32 [B={B}, next_n={next_n}, H={H}] tensor; "
            f"got dtype={sf_q.dtype}, shape={tuple(sf_q.shape)}"
        )
    if num_epi_subtiles < 1:
        raise ValueError(f"num_epi_subtiles must be >= 1, got {num_epi_subtiles}")
    if output_dtype not in _FP4_DTYPES or epi_dtype not in _FP4_DTYPES:
        raise ValueError(
            "fp4_paged_mqa_logits supports output/epi dtype in {float32, "
            f"float16, bfloat16}}; got output_dtype={output_dtype}, "
            f"epi_dtype={epi_dtype}."
        )
    # --- head_dim / num_heads supportedness --------------------------------
    # Unlike FP8, the FP4 kernel hardcodes both (see FP4MQALogitsKernel.__init__:
    # `assert head_dim == 128` / `assert num_heads == 64`). The scale-factor
    # buffer-offset math bakes in head_dim // sf_vec_size == 128 // 32 == 4
    # packed UE8M0 values per token, and the TMEM/SMEM budget is sized for
    # num_heads=64. Check here so callers get a clear error at the API boundary
    # rather than an assertion from inside JIT compilation.
    if D != _FP4_REQUIRED_HEAD_DIM:
        raise ValueError(
            f"fp4_paged_mqa_logits requires head_dim == {_FP4_REQUIRED_HEAD_DIM}; got "
            f"head_dim={D} (from q.shape[-1]*2). The FP4 kernel hardcodes this: its "
            f"scale-factor layout assumes exactly {_FP4_REQUIRED_HEAD_DIM // 32} UE8M0 "
            f"groups per token."
        )
    if H != _FP4_REQUIRED_NUM_HEADS:
        raise ValueError(
            f"fp4_paged_mqa_logits requires num_heads == {_FP4_REQUIRED_NUM_HEADS}; got "
            f"num_heads={H}. The FP4 kernel hardcodes this for its TMEM/SMEM budget."
        )
    # The kernel natively supports next_n in {1,2,3} (TMEM cap); next_n=4 is
    # decomposed in the API body into 2 atoms of 2. Anything larger has no
    # decomposition and would trip a kernel assertion during JIT compilation.
    if next_n < 1 or next_n > _FP4_MAX_NEXT_N:
        raise ValueError(
            f"fp4_paged_mqa_logits supports next_n in 1..{_FP4_MAX_NEXT_N} "
            f"(1-3 natively, 4 via atom-split); got next_n={next_n}."
        )
    _validate_phys_block_kv(block_size, "fp4_paged_mqa_logits")
    if (
        kv_fused.dim() != 4
        or kv_fused.shape[2] != 1
        or kv_fused.shape[-1] != half_D + 4
    ):
        raise ValueError(
            f"kv_fused must be [num_blocks, block_size, 1, head_dim//2+4="
            f"{half_D + 4}] (head_dim={D} from q); got shape {tuple(kv_fused.shape)}"
        )
    _validate_paged_inputs(context_lens, block_table, B)
    return True


@backend_requirement({}, common_check=_check_fp4_paged_mqa_logits_supported)
@flashinfer_api(trace=fp4_paged_mqa_logits_trace)
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
        q:               [batch_size, next_n, num_heads, head_dim//2]  uint8
                         (two FP4 per byte, E2M1)
        sf_q:            [batch_size, next_n, num_heads]  int32
                         (4 UE8M0 scale factors packed per token)
        kv_fused:        [num_blocks, block_size, 1, head_dim//2+4]  uint8
                         Per block: [KV data (block_size*head_dim//2 bytes)]
                                    [KV SF   (block_size*4 bytes, int32)]
        weights:         [batch_size*next_n, num_heads]  float32  per-head weights
        context_lens:    [batch_size]  int32  (CUDA)
        block_table:     [batch_size, max_blocks_per_seq]  int32  (CUDA)
                         Values are physical block indices into kv_fused's dim 0.
        max_context_len: int  maximum KV sequence length
        output_dtype:    output tensor dtype (float32, float16, or bfloat16)
        epi_dtype:       epilogue dtype (float32, float16, or bfloat16)
        num_epi_subtiles: epilogue subtile count (perf knob, default 1)
        remove_online_sf_transpose: if True, skip in-kernel SF SMEM transpose
                         (requires KV SF pre-arranged in UTCCP chunk layout,
                         block_size=128 only)
        schedule_meta:   optional pre-computed [num_sms+1, 2] int32 CTA schedule
                         on CUDA.  If None, computed from context_lens each call.
                         Pass a pre-computed tensor to avoid the CPU overhead when
                         the schedule is stable across calls.
                         Use compute_paged_mqa_logits_schedule() to generate it.
        out:             optional pre-allocated output
                         [batch_size*next_n, padded_ctx_len].  Use
                         padded_context_len() to size it.  Required for CUDA
                         graph capture.

    Returns:
        logits: [batch_size*next_n, max_context_len]  output_dtype

    Note:
        next_n=4 is handled internally via atom-split (2*batch_size rows of
        next_n=2) so callers can pass next_n=4 directly. next_n in {1,2,3}
        are natively supported.
    """
    if not _CUTE_DSL_AVAILABLE:
        raise RuntimeError("fp4_paged_mqa_logits requires nvidia-cutlass-dsl")

    B, next_n, H, half_D = q.shape
    D = half_D * 2
    block_size = kv_fused.shape[1]
    num_blocks = kv_fused.shape[0]
    num_sms = _cached_num_sms(get_device_index(q.device))

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
    kv_flat = kv_fused.flatten(1)

    padded_ctx_len = ((max_context_len + _SPLIT_KV - 1) // _SPLIT_KV) * _SPLIT_KV
    if out is not None:
        _validate_out(out, B * next_n, padded_ctx_len, q.device, output_dtype)
        logits = out[:, :max_context_len]
    else:
        logits_full = torch.empty(
            (B * next_n, padded_ctx_len), device=q.device, dtype=output_dtype
        )
        logits = logits_full[:, :max_context_len]

    # Built from kernel_ctx_lens, so this must follow the next_n=4 atom-split.
    if schedule_meta is None:
        schedule_meta = compute_paged_mqa_logits_schedule(
            kernel_ctx_lens, device=q.device
        )
    else:
        _validate_schedule_meta(schedule_meta, num_sms, q.device)

    compiled = _cached_compile_fp4_kernel(
        block_size,
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
        num_blocks,
        kernel_B,
    )
    return logits


# ──────────────────────────────────────────────────────────────────────────────
# Pre-compilation helper (Item 5: AOT warm-up for common configs)
# ──────────────────────────────────────────────────────────────────────────────


def precompile_paged_mqa_logits(
    device: torch.device = None,
    variants: Tuple[str, ...] = ("fp8", "fp4"),
) -> None:
    """Pre-compile paged MQA logits kernels for common static configs.

    Populates the on-disk CuTe-DSL kernel cache so subsequent calls to
    :func:`fp8_paged_mqa_logits` and :func:`fp4_paged_mqa_logits` skip
    compilation on first use.  Call once during deployment setup or as part
    of a package-build step.

    Only the configs listed below are covered; anything else (fp16 epilogue,
    other head_dim / num_heads, num_epi_subtiles != 1) still compiles on first
    use.  Measured on sm_100a: ~9s for the 8 fp8 kernels, ~3s for the 9 fp4.

    Args:
        device:   CUDA device to target.  Defaults to cuda:0.
        variants: Which precisions to build.  A deployment normally runs one
                  indexer precision, so pass e.g. ``("fp8",)`` to avoid
                  compiling kernels that will never be called.
    """
    if not _CUTE_DSL_AVAILABLE:
        return
    unknown = set(variants) - {"fp8", "fp4"}
    if unknown:
        raise ValueError(
            f"precompile_paged_mqa_logits: unknown variants {sorted(unknown)}; "
            f"supported values are 'fp8' and 'fp4'."
        )
    if device is None:
        device = torch.device("cuda", 0)
    num_sms = _cached_num_sms(get_device_index(torch.device(device)))
    num_heads, head_dim = 64, 128

    if "fp8" in variants:
        # block_size × next_n, fp32 acc/epi/out
        for block_size in (64, 128):
            for nn in (1, 2, 3, 4):
                _cached_compile_fp8_kernel(
                    block_size,
                    num_heads,
                    head_dim,
                    nn,
                    num_sms,
                    _to_cutlass(torch.float32),
                    _to_cutlass(torch.float32),
                    _to_cutlass(torch.float32),
                    1,
                )

    if "fp4" in variants:
        # block_size × next_n, fp32 epi, bf16 out
        for block_size in (32, 64, 128):
            for nn in (1, 2, 3):
                _cached_compile_fp4_kernel(
                    block_size,
                    num_heads,
                    head_dim,
                    nn,
                    num_sms,
                    _to_cutlass(torch.float32),
                    _to_cutlass(torch.bfloat16),
                    1,
                    False,
                )
