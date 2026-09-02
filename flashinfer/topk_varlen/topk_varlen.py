"""
Copyright (c) 2025 by FlashInfer team.

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

"""Top-K decode: GVR (Blackwell), CuTe DSL radix, and masked-radix backends.

Public API
----------
:func:`top_k_varlen` — selects top-K per row of decode-step logits.

Backend choices
---------------
``"radix"``          — CuTe DSL single-pass multi-CTA radix top-K; native
                       varlen support (no logit masking). Requires Blackwell
                       (sm_100+, incl. Rubin sm_107) and nvidia-cutlass-dsl.
                       No ``pre_idx`` needed.
``"gvr"``            — GVR (Guess-Verify-Refine) load-balance kernel.
                       Requires a datacentre Blackwell-class GPU (sm_100/103,
                       or Rubin sm_107), nvidia-cutlass-dsl, and a
                       ``pre_idx`` hint from the previous decode step.
``"radix_cutlass"``  — masked-radix fallback; masks logits to ``seq_lens`` then
                       calls the FlashInfer CUTLASS radix top-K.  Runs on any GPU.
``"radix_filter"``   — filtered-radix (coarse histogram → filter → on-chip
                       refine); hint-free like ``"radix"``, large-N specialist.
                       Datacentre Blackwell-class (sm_100/103/107) only;
                       requires nvidia-cutlass-dsl >= 4.8 (see below);
                       opt-in — never chosen by ``"auto"``.
``"auto"``           — GVR (if pre_idx provided) > radix (Blackwell) >
                       radix_cutlass (default).
"""

import functools
import math
from typing import Literal, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.topk import top_k_varlen_trace
from ..topk import get_topk_module

from ..cute_dsl.availability import is_cute_dsl_available

_CUTE_DSL_AVAILABLE = is_cute_dsl_available()

from ..utils import (
    _get_cache_buf,
    backend_requirement,
    get_device_sm_count,
    get_shared_bytes_per_block_optin,
    supported_compute_capability,
)

# ---------------------------------------------------------------------------
# Supported compute-capability sets (major * 10 + minor)
# ---------------------------------------------------------------------------

# All SM tiers FlashInfer ships kernels for.
_ALL_CCS = [75, 80, 86, 89, 90, 100, 103, 107, 110, 120, 121]

# CuTe DSL radix backend: all Blackwell-plus tiers.
#
# Rubin (SM107) runs the Blackwell CuTe-DSL kernels as-is: they use only
# family-portable ops (block/warp scans, ``cute.arch.barrier``,
# ``shuffle_sync_up``, ``warp_redux_sync``, griddepcontrol) with no
# arch-specific ``tcgen05``/block-scaled MMA, so the DSL compiles them for
# ``sm_107a`` natively. ``_cute_dsl_supports_arch`` below keeps a DSL that
# predates the device from being selected.
_BLACKWELL_PLUS_CCS = [100, 103, 107, 110, 120, 121]

# GVR is B200-class only (SM100/103/107). The non-LB (cluster_size=1) GVR
# CuTe-DSL kernel fails to build on sm_120a: libNVVM rejects the generated
# device IR (verified on an RTX 5080), so consumer Blackwell (SM120/121) can't
# use GVR even without load balancing. The LB path additionally needs
# cluster_size=4 programmatic multicast, which SM120/121's 1×1×1 cluster shape
# lacks. Rubin keeps both (datacentre cluster shape, same PTX surface). radix
# serves SM110+.
_GVR_CCS = [100, 103, 107]

# DKG filtered-radix backend (vendored CuTe-DSL kernel, see
# kernels/filtered_topk_util.py). Upstream's own arch table
# ``_LARGE_OCCUPANCY_MIN_BLOCKS_PER_MP`` covers sm_100/103/107/109. 109 is left
# out here because the shipped CuTe DSL has no ``sm_109`` entry in
# ``SMEM_CAPACITY_MAP``, so sizing raises before the kernel ever compiles.
_RADIX_FILTER_CCS = [100, 103, 107]

# ---------------------------------------------------------------------------
# Backend requirement checkers
# ---------------------------------------------------------------------------


@functools.cache
def _cute_dsl_supports_arch(major: int, minor: int) -> bool:
    """Whether the installed CuTe DSL can target this compute capability.

    The compute-capability lists above say which tiers FlashInfer *has* kernels
    for; this says whether the *installed* DSL can emit code for the device. The
    two differ on new silicon: a DSL release predating Rubin resolves ``sm_107a``
    to a ``KeyError`` deep inside ``cute.compile``. Consulting this here makes
    ``backend="auto"`` fall back to ``radix_cutlass`` instead of crashing, and
    an explicit ``backend=`` request fail at dispatch with a clear error.

    Note the introspection helper ``top_k_varlen.is_backend_supported`` does
    NOT consult this (or any) runtime probe: it answers the *static* question
    -- is the backend registered and does FlashInfer ship a kernel for this
    compute capability -- from the registration dict and the CC lists alone.
    Environment gates (installed-DSL arch support, the ``cutlass.memory``
    requirement of ``radix_filter``) apply only at call time, through the
    per-backend checkers. A True from ``is_backend_supported`` therefore does
    not guarantee a call will be accepted in this environment. Mirrors
    ``flashinfer.norm._cute_dsl_supports_arch``.
    """
    try:
        from ..cute_dsl.utils import is_cute_dsl_arch_supported

        return is_cute_dsl_arch_supported(major, minor)
    except Exception:
        # Never let the capability probe itself break top-k dispatch.
        return True


def _cute_dsl_ready(device: torch.device) -> bool:
    """``True`` when the CuTe-DSL backends are usable on ``device``."""
    if not _CUTE_DSL_AVAILABLE:
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return _cute_dsl_supports_arch(major, minor)


@supported_compute_capability(_ALL_CCS)
def _radix_cutlass_top_k_varlen_check(
    logits,
    seq_lens,
    top_k,
    pre_idx=None,
    compress_ratio=1,
    next_n=1,
    return_values=False,
    out_indices=None,
    out_values=None,
    backend="auto",
    load_balance=True,
    workspace=None,
):  # extra kwargs mirror the public signature; unused by the check
    """Radix masked-fallback: runs on all supported SM tiers."""
    return True


@supported_compute_capability(_GVR_CCS)
def _gvr_top_k_varlen_check(
    logits,
    seq_lens,
    top_k,
    pre_idx=None,
    compress_ratio=1,
    next_n=1,
    return_values=False,
    out_indices=None,
    out_values=None,
    backend="auto",
    load_balance=True,
    workspace=None,
):
    """Return True only when GVR can run on this exact configuration.

    Used by backend="auto" routing: returning False here causes the heuristic to
    fall back to radix or radix_cutlass rather than reaching GVR and crashing.
    """
    if not (_cute_dsl_ready(logits.device) and pre_idx is not None):
        return False
    # GvrParams only has entries for top_k in {512, 1024, 2048}; other values
    # raise inside the kernel's __init__ during compilation.
    if top_k not in (512, 1024, 2048):
        return False
    # GvrTopKLBPrepareKernel asserts compress_ratio in (1, 4).
    if compress_ratio not in (1, 4):
        return False
    # GVR uses 128-bit vectorized loads — each row must be 16-byte aligned.
    N = logits.shape[1]
    elem_align = 16 // logits.element_size()
    if N % elem_align != 0:
        return False
    # LB prepare kernel caps batch_size at 1024 (max_batch_size must be a power
    # of 2 in [64, 1024]); the single-CTA path (load_balance=False) has no cap.
    if load_balance and seq_lens.shape[0] > 1024:
        return False
    return True


def _top_k_varlen_heuristic(
    suitable_backends,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    pre_idx=None,
    compress_ratio: int = 1,
    next_n: int = 1,
    return_values: bool = False,
    out_indices=None,
    out_values=None,
    backend: str = "auto",
    load_balance: bool = True,
    workspace=None,
):
    """GVR (needs pre_idx) > radix (CuTe DSL, Blackwell) > radix_cutlass (all GPUs).

    The full signature must be spelled out (not **kwargs) so that the decorator can
    call this function with positional args on the skip_check=True path without
    raising TypeError.  Mirrors the pattern used by _heuristic_func_mm_fp4.
    """
    # "radix_filter" is deliberately absent: its checker accepts a strict
    # subset of radix's configurations and its CC list is a subset of
    # radix's, so listed after radix it could never be chosen (a dead
    # entry), and listed before radix it would regress the small-row and
    # small-batch regions where radix wins. Making it auto-selectable needs
    # a shape/architecture-aware crossover rule (it wins at large N --
    # roughly N >= 64K with enough rows on SM100, wider on SM107); until
    # that rule exists it is explicit-only, as documented in the module
    # docstring.
    return [b for b in ("gvr", "radix", "radix_cutlass") if b in suitable_backends]


# ---------------------------------------------------------------------------
# Internal: compiled-kernel cache
# ---------------------------------------------------------------------------

if _CUTE_DSL_AVAILABLE:
    import cutlass
    import cutlass.cute as cute
    from .kernels.config import GvrTopKConfig, GvrTopKLBConfig
    from ..cute_dsl.utils import torch_to_cutlass_dtype
    from .kernels import (
        GvrTopKKernel,
        GvrTopKLBKernel,
        GvrTopKLBPrepareKernel,
        SinglePassMultiCTARadixTopKKernel,
    )
    from .kernels.radix_topk import STATE_SIZE as _RADIX_STATE_SIZE


@functools.cache
def _gvr_kernel_source_files() -> Tuple[str, ...]:
    # All source files whose content the compiled GVR kernels depend on, so a
    # change to any of them invalidates the persistent CuTe-DSL cache. The three
    # _compile_gvr* helpers share the "gvr_topk" module directory, so they MUST
    # pass an identical (union) key here — the key is module-level, and a
    # per-function subset would let one helper's build serve another's stale
    # artifact. GvrTopKLBKernel/GvrTopKLBPrepareKernel live in
    # gvr_topk_decode_lb.py, which imports GvrTopKKernel from gvr_topk_decode.py;
    # both decode modules pull warp_scan/block_prefix_sum from block_scan.py.
    from .kernels import gvr_topk_decode, gvr_topk_decode_lb, block_scan

    return (
        __file__,
        gvr_topk_decode.__file__,
        gvr_topk_decode_lb.__file__,
        block_scan.__file__,
    )


@functools.cache
def _radix_kernel_source_files() -> Tuple[str, ...]:
    # radix_topk.py defines SinglePassMultiCTARadixTopKKernel and pulls
    # block_prefix_sum_kernel from block_scan.py; both invalidate the cache.
    from .kernels import radix_topk, block_scan

    return (__file__, radix_topk.__file__, block_scan.__file__)


@functools.cache
def _compile_gvr_lb_prepare(num_threads: int, long_threshold: int, compress_ratio: int):
    # batch_size (the seq_lens length) is DYNAMIC: it is passed as a runtime scalar
    # (cutlass.Int32(batch_size)) and only bounds the classifier's per-thread guard
    # (tidx < batch_size); it is never a const_expr / SMEM size / static unroll, and
    # the grid is fixed (1,1,1). So seq_lens is compiled with a symbolic length and
    # one kernel serves every batch size. num_threads stays static (it sizes the
    # order_row buffer and the block-prefix-sum SMEM); counters is constant (2,).
    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    prep = GvrTopKLBPrepareKernel(
        long_threshold=long_threshold,
        compress_ratio=compress_ratio,
        num_threads=num_threads,
    )
    sym_batch = cute.sym_int()

    def _compile_fn():
        return cute.compile(
            prep,
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_batch,), stride_order=(0,)
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (num_threads,), stride_order=(0,)
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (2,), stride_order=(0,)
            ),
            cutlass.Int32(0),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return build_and_load_cute_dsl_kernel(
        "gvr_topk",
        f"lb_prepare_nt{num_threads}_lt{long_threshold}_cr{compress_ratio}",
        _compile_fn,
        extra_key_files=_gvr_kernel_source_files(),
    )


@functools.cache
def _compile_gvr_lb(
    cute_dtype,
    top_k,
    next_n,
    compress_ratio,
    max_batch_size,
    num_threads,
    cluster_size,
    return_output_values,
    # Launch-config knobs chosen by GvrTopKConfig.auto() per shape. Defaults
    # preserve the pre-auto() LB behavior (GvrTopKLBKernel's defaults, which
    # DIFFER from _compile_gvr: 256-bit loads ON, phase3-unroll OFF).
    min_blocks_per_mp=3,
    use_256bit_load=True,
    enable_unroll_4=True,
    enable_phase3_unroll=False,
    enable_warp_parallel_reduce=False,
):
    # num_rows and N are DYNAMIC (symbolic) — see _compile_gvr. The LB kernel's
    # grid is fixed at max_batch_size * next_n * cluster_size (surplus clusters
    # early-exit via counters), so num_rows never touches the launch config, and
    # N is a per-row runtime bound. Only max_batch_size stays static (it sizes the
    # grid and the order_row buffer). The cache keys on those specializations plus
    # the launch-config knobs; shapes stay symbolic within each.
    kernel = GvrTopKLBKernel(
        dtype=cute_dtype,
        top_k=top_k,
        next_n=next_n,
        num_threads=num_threads,
        compress_ratio=compress_ratio,
        return_output_values=return_output_values,
        cluster_size=cluster_size,
        max_batch_size=max_batch_size,
        min_blocks_per_mp=min_blocks_per_mp,
        use_256bit_load=use_256bit_load,
        enable_unroll_4=enable_unroll_4,
        enable_phase3_unroll=enable_phase3_unroll,
        enable_warp_parallel_reduce=enable_warp_parallel_reduce,
    )
    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    sym_groups = cute.sym_int()  # request count (= num_rows // next_n)
    sym_n = cute.sym_int()  # per-row logits width
    sym_rows = sym_groups * next_n

    dtype_name = str(cute_dtype).split(".")[-1]
    kernel_name = (
        f"lb_{dtype_name}_topk{top_k}_nextn{next_n}_cr{compress_ratio}"
        f"_bs{max_batch_size}_nt{num_threads}_cl{cluster_size}_rv{int(return_output_values)}"
        f"_mbpm{min_blocks_per_mp}_256b{int(use_256bit_load)}_u4{int(enable_unroll_4)}"
        f"_p3u{int(enable_phase3_unroll)}_wpr{int(enable_warp_parallel_reduce)}"
    )

    def _compile_fn():
        return cute.compile(
            kernel,
            cute.runtime.make_fake_compact_tensor(
                cute_dtype, (sym_rows, sym_n), stride_order=(1, 0), assumed_align=16
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (sym_groups, top_k),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_groups,), stride_order=(0,)
            ),
            cute.runtime.make_fake_compact_tensor(
                cute_dtype, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16
            )
            if return_output_values
            else None,
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (max_batch_size,), stride_order=(0,)
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (2,), stride_order=(0,)
            ),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return build_and_load_cute_dsl_kernel(
        "gvr_topk",
        kernel_name,
        _compile_fn,
        extra_key_files=_gvr_kernel_source_files(),
    )


@functools.cache
def _compile_gvr(
    cute_dtype,
    top_k,
    next_n,
    compress_ratio,
    num_threads,
    return_output_values,
    # Launch-config knobs chosen by GvrTopKConfig.auto() per shape. Each is a
    # compile-time specialization keying the JIT cache; defaults preserve the
    # pre-auto() behavior (GvrTopKKernel's defaults at num_threads=512).
    min_blocks_per_mp=3,
    use_256bit_load=False,
    enable_unroll_4=True,
    enable_phase3_unroll=True,
    enable_warp_parallel_reduce=False,
):
    # num_rows and N (per-row width) are DYNAMIC dimensions: the kernel reads
    # them from the tensor shapes at runtime (num_rows = input.shape[0] drives the
    # grid in __call__; N is derived per-row from seq_lens) and never uses them in
    # a const_expr / SMEM sizing / static unroll. So they are compiled symbolically
    # via cute.sym_int() — one compiled kernel serves every batch size and
    # sequence length, matching the FlashInfer CuTe-DSL convention (cf.
    # rmsnorm_fp4quant._get_compiled_kernel). The cache keys on the true
    # specializations (dtype, top_k, next_n, compress_ratio, return_output_values)
    # plus the launch-config knobs; shapes stay symbolic within each.
    # sym_groups is the request axis; num_rows = sym_groups * next_n.
    kernel = GvrTopKKernel(
        dtype=cute_dtype,
        top_k=top_k,
        next_n=next_n,
        num_threads=num_threads,
        compress_ratio=compress_ratio,
        return_output_values=return_output_values,
        cluster_size=1,
        min_blocks_per_mp=min_blocks_per_mp,
        use_256bit_load=use_256bit_load,
        enable_unroll_4=enable_unroll_4,
        enable_phase3_unroll=enable_phase3_unroll,
        enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        use_constant_hint=False,
        seqlen_sorted=True,
    )
    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    sym_groups = cute.sym_int()  # request count (= num_rows // next_n)
    sym_n = cute.sym_int()  # per-row logits width
    sym_rows = sym_groups * next_n

    dtype_name = str(cute_dtype).split(".")[-1]
    kernel_name = (
        f"{dtype_name}_topk{top_k}_nextn{next_n}_cr{compress_ratio}"
        f"_nt{num_threads}_rv{int(return_output_values)}"
        f"_mbpm{min_blocks_per_mp}_256b{int(use_256bit_load)}_u4{int(enable_unroll_4)}"
        f"_p3u{int(enable_phase3_unroll)}_wpr{int(enable_warp_parallel_reduce)}"
    )

    def _compile_fn():
        return cute.compile(
            kernel,
            cute.runtime.make_fake_compact_tensor(
                cute_dtype, (sym_rows, sym_n), stride_order=(1, 0), assumed_align=16
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (sym_groups, top_k),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_groups,), stride_order=(0,)
            ),
            cute.runtime.make_fake_compact_tensor(
                cute_dtype, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16
            )
            if return_output_values
            else None,
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16
            ),
            cute.runtime.make_fake_compact_tensor(  # order_row: request-level LJF order
                cutlass.Int32, (sym_groups,), stride_order=(0,)
            ),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return build_and_load_cute_dsl_kernel(
        "gvr_topk",
        kernel_name,
        _compile_fn,
        extra_key_files=_gvr_kernel_source_files(),
    )


# ---------------------------------------------------------------------------
# Radix CuTe DSL: SM-aware chunk / multi-CTA config
# ---------------------------------------------------------------------------
# The single-pass multi-CTA radix kernel stages one chunk of the row in shared
# memory per CTA (``shared_ordered[chunk_size]``, ordered_type). A single CTA
# can only cover a row that fits in SMEM; longer rows must be split across a
# CTA *group* (``ctas_per_group`` CTAs cooperating via a global histogram merge
# in ``row_states``). Historically ``_run_radix`` hardcoded
# ``ctas_per_group=1``, which (a) FAULTED on rows too large for SMEM
# (``chunk_size = N`` → ``2304 + itemsize*N`` bytes > the 232448-byte sm_100a
# cap at N≈115K bf16 / 57K fp32) and (b) left the kernel's multi-CTA path dead,
# serializing very long rows on one CTA. These helpers mirror TRT-LLM's
# ``_compute_max_chunk`` / ``_get_chunk_config`` (cute_dsl_custom_ops.py) so
# FlashInfer picks the same SMEM-bounded chunk_size and CTA count.

_RADIX_NUM_COPY_BITS = 256  # 256-bit vectorized SMEM loads (matches the kernel)
# Fixed per-CTA SMEM before shared_ordered, 128-byte-padded to match the
# kernel's SmemAllocator layout (radix_topk.py):
#   local_histogram[256] int32  = 1024
#   prefix_buf[256]      int32  = 1024
#   s_scalars[4]         int32  =   16 → padded to 128 (next alloc is 128-aligned)
#   s_warp_sums[8]       int32  =   32 → padded to 128
# Total = 2304. (Verified: 2304 + 2*131072 = 264448 == the observed overflow.)
_RADIX_SMEM_OVERHEAD = 2304


def _radix_ordered_elem_size(torch_dtype) -> int:
    """Bytes per staged ordered element: Uint32 (4) for fp32, Uint16 (2) for half."""
    return 4 if torch_dtype == torch.float32 else 2


def _radix_compute_max_chunk(
    torch_dtype, smem_capacity: int, num_copy_bits: int = _RADIX_NUM_COPY_BITS
) -> int:
    """Largest chunk_size (elements) a single CTA can stage in SMEM, vec-aligned.

    ``smem_capacity`` is the opt-in dynamic shared memory per block for the
    target device, obtained from ``get_shared_bytes_per_block_optin``.
    """
    ordered_elem_size = _radix_ordered_elem_size(torch_dtype)
    dtype_width_bits = 32 if torch_dtype == torch.float32 else 16
    vec_size = num_copy_bits // dtype_width_bits  # 8 fp32, 16 half
    max_chunk = (smem_capacity - _RADIX_SMEM_OVERHEAD) // ordered_elem_size
    max_chunk -= max_chunk % vec_size  # floor to a whole vector
    return max_chunk


def _radix_get_chunk_config(
    N: int,
    torch_dtype,
    num_rows: int,
    num_sms: int,
    smem_capacity: int,
    num_copy_bits: int = _RADIX_NUM_COPY_BITS,
) -> Tuple[int, int]:
    """Return ``(ctas_per_group, chunk_size)`` — SM-aware, mirrors TRT-LLM.

    Two regimes (TRT-LLM ``CuteDSLTopKDecodeSinglePassMultiCTARunner._get_chunk_config``):

    * **Large batch** (``num_sms // num_rows <= 1``): there is already one CTA
      per SM's worth of rows, so minimize ``ctas_per_group`` — split a row only
      as far as the SMEM cap forces (``ceil(N / max_chunk)``). Rows that fit in
      one CTA's SMEM stay single-CTA.
    * **Small batch, spare SMs** (``num_sms // num_rows > 1``): split each row
      across ``num_sms // num_rows`` CTAs to fill the machine (more parallelism
      over one long row), with a ``>= 8192`` min chunk to amortize the per-CTA
      reduce and a power-of-2 snap for JIT-cache friendliness. This is the
      regime that closes the 1.3-1.8x uniform large-N / small-batch gap — the
      old code left it single-CTA and serialized the whole row.
    """
    dtype_width_bits = 32 if torch_dtype == torch.float32 else 16
    vec_size = num_copy_bits // dtype_width_bits
    max_chunk = _radix_compute_max_chunk(torch_dtype, smem_capacity, num_copy_bits)

    ideal_ctas_per_group = max(1, num_sms // max(num_rows, 1))
    if ideal_ctas_per_group <= 1:
        # Large batch: minimize ctas_per_group, bounded by SMEM capacity.
        ctas_per_group = max(1, math.ceil(N / max_chunk))
        chunk_size = math.ceil(N / ctas_per_group)
        chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
        chunk_size = min(chunk_size, max_chunk)
    else:
        # Small batch, spare SMs: split each row across ideal_ctas_per_group CTAs.
        chunk_size = math.ceil(N / ideal_ctas_per_group)
        chunk_size = max(chunk_size, 8192)  # min chunk: amortize per-CTA reduce
        ctas_per_group = math.ceil(N / chunk_size)
        if ctas_per_group == 2 and chunk_size < 32768:
            # A 2-way split with a small chunk costs more in sync than it saves.
            chunk_size = N
        # Snap up to a power of 2 (fewer distinct chunk_size specializations),
        # clamped to the SMEM cap.
        snap_up = 1 << math.ceil(math.log2(max(chunk_size, 1)))
        if snap_up > max_chunk:
            snap_up = 1 << int(math.log2(max_chunk))
        chunk_size = snap_up

    ctas_per_group = max(1, math.ceil(N / chunk_size))
    return ctas_per_group, chunk_size


@functools.cache
def _compile_radix(
    cute_dtype,
    top_k,
    next_n,
    compress_ratio,
    N,
    return_output_values,
    ctas_per_group,
    chunk_size,
    num_sms,
):
    # N (vocab size) is static: it feeds the fake input tensor's column extent so
    # the kernel is specialized per vocab width. chunk_size (the per-CTA SMEM
    # span) is passed explicitly — computed once by _radix_get_chunk_config so
    # the compiled SMEM constant matches the runtime budget — instead of being
    # re-derived here (the old N/ctas division could exceed the SMEM cap).
    # num_rows (batch) and sym_groups are dynamic — one compiled kernel serves all
    # batch sizes at this (dtype, top_k, N, ctas_per_group, chunk_size) specialization.
    # compress_ratio is a const_expr in the kernel so it is a cache key here too.
    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    kernel = SinglePassMultiCTARadixTopKKernel(
        dtype=cute_dtype,
        chunk_size=chunk_size,
        top_k=top_k,
        next_n=next_n,
        compress_ratio=compress_ratio,
        ctas_per_group=ctas_per_group,
        num_sms=num_sms,
    )
    sym_groups = cute.sym_int()  # number of requests (= num_rows // next_n)
    sym_n = N  # static vocab width
    sym_rows = sym_groups * next_n
    max_num_groups = max(1, num_sms // ctas_per_group)

    dtype_name = str(cute_dtype).split(".")[-1]
    kernel_name = (
        f"{dtype_name}_topk{top_k}_nextn{next_n}_cr{compress_ratio}"
        f"_N{N}_rv{int(return_output_values)}_cta{ctas_per_group}_chunk{chunk_size}_sms{num_sms}"
    )

    def _compile_fn():
        return cute.compile(
            kernel,
            cute.runtime.make_fake_compact_tensor(
                cute_dtype, (sym_rows, sym_n), stride_order=(1, 0), assumed_align=16
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (max_num_groups, _RADIX_STATE_SIZE), stride_order=(1, 0)
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_groups,), stride_order=(0,)
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16
            ),
            cute.runtime.make_fake_compact_tensor(
                cute_dtype, (sym_rows, top_k), stride_order=(1, 0), assumed_align=16
            )
            if return_output_values
            else None,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return build_and_load_cute_dsl_kernel(
        "radix_topk",
        kernel_name,
        _compile_fn,
        extra_key_files=_radix_kernel_source_files(),
    )


# ---------------------------------------------------------------------------
# Internal: GVR backend implementation
# ---------------------------------------------------------------------------


def _lb_max_batch_size(batch_size: int) -> int:
    for cap in (64, 128, 256, 512, 1024):
        if batch_size <= cap:
            return cap
    raise ValueError(f"batch_size {batch_size} exceeds maximum supported 1024")


def _n_is_256bit_aligned(torch_dtype, N: int) -> bool:
    """256-bit vectorized loads need 32-byte-aligned rows: N * itemsize % 32 == 0.
    (The up-front N check only guarantees 16-byte / 128-bit alignment.)
    """
    itemsize = torch.tensor([], dtype=torch_dtype).element_size()
    return (N % (32 // itemsize)) == 0


def _auto_gvr_knobs(
    logits: torch.Tensor, is_lb: bool, cluster_size: int = 1
) -> Tuple[int, dict]:
    """Launch-config knobs from the shape-aware analytical heuristic
    GvrTopKConfig.auto(). These knobs (num_threads / min_blocks_per_mp /
    vec-width / unrolls) are memory-pipeline/occupancy parameters whose optimum
    is determined by the tensor shape, so an analytical rule picks them well —
    no runtime profiling needed. Returns (num_threads, knob_kwargs).

    ``cluster_size`` (LB long-row branch): each of the ``cluster_size`` CTAs in a
    cluster scans only ``N / cluster_size`` columns, so the knobs must be tuned
    on that *per-CTA* width, not the full row. Feeding the full N picks a
    much-too-large num_threads (1024 vs 512) / warp_parallel_reduce / wrong
    min_blocks_per_mp and regresses LB by up to 2x at N=131072. This mirrors
    TRT-LLM's ``N_per_cta = N_row // cluster_size`` fed to its ``_pick_tuning``.

    256-bit is force-disabled unless the FULL row is 32-byte aligned: the kernel
    issues a 32B-aligned LDG when use_256bit_load=True, which faults on a
    16B-but-not-32B-aligned row. auto() only enables 256-bit for fp32/large-N,
    but the LB path's historical default is 256-bit-on for any dtype (it measures
    faster for the clustered long-row branch on B200), so enable it there when
    the row is 32B-aligned.
    """
    N = logits.shape[1]
    num_rows = logits.shape[0]
    # Per-CTA scan width for the clustered (long-row) LB branch.
    n_eff = N // cluster_size if cluster_size > 1 else N
    cfg = GvrTopKConfig.auto(
        logits.dtype, n_eff, num_rows, get_device_sm_count(logits.device)
    )
    use_256 = cfg.use_256bit_load
    if is_lb:
        # LB historically prefers 256-bit; enable it when the row is 32B-aligned.
        use_256 = True
    if not _n_is_256bit_aligned(logits.dtype, N):  # alignment on the FULL row
        use_256 = False
    return cfg.num_threads_per_block, dict(
        min_blocks_per_mp=cfg.min_blocks_per_mp,
        use_256bit_load=use_256,
        enable_unroll_4=cfg.enable_unroll_4,
        enable_phase3_unroll=cfg.enable_phase3_unroll,
        enable_warp_parallel_reduce=cfg.enable_warp_parallel_reduce,
    )


def _run_gvr_lb(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    return_output_values: bool,
    out_indices: Optional[torch.Tensor],
    out_values: Optional[torch.Tensor],
    order_row: Optional[torch.Tensor] = None,
    counters: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """LB prepare (sort rows by length) + GVR decode."""
    cute_dtype = torch_to_cutlass_dtype(logits.dtype)
    batch_size = seq_lens.shape[0]
    max_batch_size = _lb_max_batch_size(batch_size)
    lb_cfg = GvrTopKLBConfig(max_batch_size=max_batch_size)

    # The prepare kernel overwrites both tensors before they are read, so
    # torch.empty is correct (no zero-initialization needed).
    if order_row is None:
        order_row = torch.empty(max_batch_size, dtype=torch.int32, device=logits.device)
    if counters is None:
        counters = torch.empty(2, dtype=torch.int32, device=logits.device)
    _compile_gvr_lb_prepare(max_batch_size, lb_cfg.long_threshold, compress_ratio)(
        seq_lens, order_row, counters, cutlass.Int32(batch_size)
    )

    # Shape-aware launch config from GvrTopKConfig.auto(), tuned on the per-CTA
    # scan width N/cluster_size (the long-row cluster branch splits the row).
    num_threads, knobs = _auto_gvr_knobs(
        logits, is_lb=True, cluster_size=lb_cfg.cluster_size
    )
    _compile_gvr_lb(
        cute_dtype,
        top_k,
        next_n,
        compress_ratio,
        max_batch_size,
        num_threads,
        lb_cfg.cluster_size,
        return_output_values,
        **knobs,
    )(
        logits,
        pre_idx,
        seq_lens,
        out_values if return_output_values else None,
        out_indices,
        order_row,
        counters,
    )
    return out_indices, (out_values if return_output_values else None)


def _run_gvr(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    return_output_values: bool,
    out_indices: Optional[torch.Tensor],
    out_values: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run GVR without load-balancing: one CTA per row, rows sorted longest-first."""
    cute_dtype = torch_to_cutlass_dtype(logits.dtype)

    # Sort requests longest-first (LJF) so early waves process the heaviest rows;
    # short rows fill the tail, eliminating the load-imbalance tail penalty.
    # seq_lens is already request-level with shape (num_rows // next_n,); slicing
    # with [::next_n] would produce a tensor that's next_n times too short.
    order_row = torch.argsort(seq_lens, descending=True).to(torch.int32)

    # Shape-aware launch config from GvrTopKConfig.auto().
    num_threads, knobs = _auto_gvr_knobs(logits, is_lb=False)
    _compile_gvr(
        cute_dtype,
        top_k,
        next_n,
        compress_ratio,
        num_threads,
        return_output_values,
        **knobs,
    )(
        logits,
        pre_idx,
        seq_lens,
        out_values if return_output_values else None,
        out_indices,
        order_row,
    )
    return out_indices, (out_values if return_output_values else None)


# ---------------------------------------------------------------------------
# Internal: radix_cutlass (masked-radix CUTLASS) backend implementation
# ---------------------------------------------------------------------------


def _run_radix_cutlass(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    return_output_values: bool,
    out_indices: Optional[torch.Tensor],
    out_values: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Masked-radix fallback: uses radix_topk_ragged_transform to pass per-row
    lengths directly into the kernel, avoiding a full (num_rows, N) masked copy
    and a double-allocation for indices.
    """
    num_rows, N = logits.shape
    if next_n > 1:
        row_seq_lens = seq_lens.repeat_interleave(next_n)
        row_offsets = torch.arange(
            next_n, device=logits.device, dtype=torch.int32
        ).repeat(seq_lens.shape[0])
        row_seq_lens = (row_seq_lens - next_n + row_offsets + 1) // compress_ratio
    else:
        row_seq_lens = seq_lens // compress_ratio
    lengths = row_seq_lens.clamp(max=N).to(torch.int32)

    # offsets=zeros: output indices are local column indices (0..N-1), no shift.
    offsets = torch.zeros(num_rows, dtype=torch.int32, device=logits.device)

    row_states_buffer = _get_cache_buf(
        f"radix_topk_row_states_{logits.device}",
        1024 * 1024,
        logits.device,
        zero_init=True,
    )
    # Write directly into out_indices — no second allocation or copy_ needed.
    get_topk_module().radix_topk_ragged_transform(
        logits,
        out_indices,
        offsets,
        lengths,
        row_states_buffer,
        top_k,
        False,  # deterministic
        0,  # tie_break = TopKTieBreak.NONE
        False,  # dsa_graph_safe
    )

    if return_output_values:
        # Gather values at selected indices — O(num_rows*top_k) vs O(num_rows*N).
        # radix_topk_ragged_transform writes the -1 sentinel into surplus slots
        # when a row's length <= top_k (topk.cuh Ragged branch), so clamp the
        # index before gathering — a raw -1 trips a device-side bounds assert —
        # then zero those sentinel slots so they don't carry column-0 values.
        # No-op for rows with seq_len >= top_k (no sentinels).
        sentinel = out_indices < 0
        gather_idx = out_indices.long().clamp_(min=0)
        out_values.copy_(torch.gather(logits, 1, gather_idx))
        out_values.masked_fill_(sentinel, 0)

    return out_indices, (out_values if return_output_values else None)


# ---------------------------------------------------------------------------
# Internal: CuTe DSL radix backend implementation
# ---------------------------------------------------------------------------


@supported_compute_capability(_BLACKWELL_PLUS_CCS)
def _radix_top_k_varlen_check(
    logits,
    seq_lens,
    top_k,
    pre_idx=None,
    compress_ratio=1,
    next_n=1,
    return_values=False,
    out_indices=None,
    out_values=None,
    backend="auto",
    load_balance=True,
    workspace=None,
):
    """CuTe DSL multi-CTA radix: Blackwell-plus only, no pre_idx required."""
    return _cute_dsl_ready(logits.device)


def _run_radix(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    return_output_values: bool,
    out_indices: torch.Tensor,
    out_values: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """CuTe DSL multi-CTA radix top-k: native varlen, no logit masking needed."""
    cute_dtype = torch_to_cutlass_dtype(logits.dtype)
    num_rows, N = logits.shape
    # The kernel handles both the next_n per-row adjustment and compress_ratio
    # division internally (length = (seq_len - next_n + nn + 1) // compress_ratio),
    # so seq_lens is passed through unmodified in uncompressed (token) units.

    num_sms = get_device_sm_count(logits.device)
    # Opt-in dynamic SMEM capacity for this device (cached after the first
    # query per device). Used to bound chunk_size so each CTA's shared_ordered
    # buffer fits in SMEM; the correct value varies across SM100/SM120/SM121.
    smem_capacity = get_shared_bytes_per_block_optin(logits.device)
    # SM-aware chunk/CTA config: rows that fit in SMEM stay single-CTA
    # (ctas_per_group=1); longer rows split across a CTA group so each CTA's
    # chunk fits in shared memory. This both avoids the SMEM-overflow fault at
    # large N and activates the kernel's multi-CTA path (parallelism over one
    # long row) instead of serializing it on one CTA.
    ctas_per_group, chunk_size = _radix_get_chunk_config(
        N, logits.dtype, num_rows, num_sms, smem_capacity
    )

    compiled = _compile_radix(
        cute_dtype,
        top_k,
        next_n,
        compress_ratio,
        N,
        return_output_values,
        ctas_per_group,
        chunk_size,
        num_sms,
    )

    # row_states holds the global histograms + inter-CTA barrier counters used
    # when ctas_per_group > 1. max_num_groups must equal the compile-time value
    # (both = num_sms // ctas_per_group) so the buffer matches the grid.
    #
    # The buffer is allocated for the worst case (ctas_per_group=1 → num_sms
    # groups) and sliced to the current max_num_groups. Fixing the allocation at
    # num_sms means the shared, device-keyed buffer never has to grow between
    # calls with different N/ctas_per_group (a grow would re-zero and churn), and
    # matches TRT-LLM's `[num_sms, state_size]` allocation.
    #
    # zero_init=True: the multi-CTA path spins on _ARRIVAL_COUNTER, so the buffer
    # MUST be zero before the first launch. The kernel self-resets the slots it
    # touches at end-of-kernel, so steady state stays clean without re-zeroing —
    # but a prior single-CTA call (which never writes row_states) could otherwise
    # leave garbage a later multi-CTA call reads. Zeroing on the one-time
    # allocation (outside any CUDA-graph capture) keeps both paths correct.
    max_num_groups = max(1, num_sms // ctas_per_group)
    nbytes = max_num_groups * _RADIX_STATE_SIZE * 4  # int32 → 4 bytes each
    row_states = (
        _get_cache_buf(
            f"radix_row_states_{logits.device}",
            num_sms * _RADIX_STATE_SIZE * 4,  # worst case: ctas_per_group=1
            logits.device,
            zero_init=True,
        )[:nbytes]
        .view(torch.int32)
        .view(max_num_groups, _RADIX_STATE_SIZE)
    )

    compiled(
        logits,
        row_states,
        seq_lens,
        out_indices,
        out_values if return_output_values else None,
    )
    return out_indices, (out_values if return_output_values else None)


# ---------------------------------------------------------------------------
# Internal: DKG filtered-radix (`radix_filter`) backend implementation
# ---------------------------------------------------------------------------

_RADIX_FILTER_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@functools.cache
def _radix_filter_kernel_dsl_ok() -> bool:
    """Whether the installed CuTe DSL has the APIs the vendored kernel uses.

    The vendored kernels require **nvidia-cutlass-dsl >= 4.8**. They use the
    ``cutlass.memory`` namespace (``SmemAllocator``,
    ``get_smem_capacity_in_bytes``); 4.7.x exposes those APIs under
    ``cutlass.utils`` instead, lacks the sm_107 architecture/capacity
    metadata the Rubin sizing needs, and predates the ``cutlass.block``
    namespace the TMA path uses -- so a 4.7 "compatibility alias" would not
    actually run these kernels, and this repo's minimum DSL pin is older
    still. On such environments the kernel would raise ``AttributeError``
    from inside ``cute.compile``. Unlike the arch probe above,
    this one fails CLOSED: without the module the kernel definitely cannot run,
    so reporting unsupported (clean fallback / clean dispatch error) is strictly
    better than the deferred crash.
    """
    try:
        import cutlass.memory  # noqa: F401, PLC0415

        return hasattr(cutlass.memory, "SmemAllocator") and hasattr(
            cutlass.memory, "get_smem_capacity_in_bytes"
        )
    except Exception:
        return False


@supported_compute_capability(_RADIX_FILTER_CCS)
def _radix_filter_top_k_varlen_check(
    logits,
    seq_lens,
    top_k,
    pre_idx=None,
    compress_ratio=1,
    next_n=1,
    return_values=False,
    out_indices=None,
    out_values=None,
    backend="auto",
    load_balance=True,
    workspace=None,
):
    """Return True only when the vendored DKG kernel covers this configuration.

    Upstream's decode wrapper has no ``compress_ratio`` and no ``pre_idx``
    parameter at all, so those are hard exclusions rather than tuning knobs --
    returning False here makes an explicit ``backend="radix_filter"`` call fail
    at backend validation instead of silently ignoring the argument or failing
    deep inside the kernel constructor.
    """
    if not _cute_dsl_ready(logits.device):
        return False
    if not _radix_filter_kernel_dsl_ok():
        return False
    if pre_idx is not None:
        return False
    # Vendored kernel bound: FilteredTopKKernelVarlen rejects top_k outside
    # [1, 16384]; enforce it here so the failure is a backend-validation error.
    if not 1 <= top_k <= 16384:
        return False
    if compress_ratio != 1:
        return False
    if logits.dtype not in _RADIX_FILTER_DTYPES:
        return False
    return True


def _run_radix_filter(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    return_output_values: bool,
    out_indices: torch.Tensor,
    out_values: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """DKG filtered-radix (coarse histogram -> filter -> refine) decode top-k.

    The upstream wrapper's contract already matches this API: it takes the
    ``(batch * next_n, N)`` logits and the ``(batch,)`` int32 ``seq_lens``, and
    returns row-relative indices padded with ``-1``.

    Integration boundary (vs. the fused sparse-attention interface): this
    backend produces ROW-RELATIVE TOP-K INDICES ONLY. It does not fuse the
    page-table translation, does not take ``row_starts`` /
    ``page_table_row_starts`` / ``row_to_batch``, offers no deterministic
    tie-breaking, and has no ``dsa_graph_safe`` mode -- all of which
    :func:`flashinfer.top_k_page_table_transform` provides in one fused
    launch. A consumer of that interface can substitute this backend for
    ordinary decode only by adding a separate index-transform launch, and
    cannot substitute it at all where ``row_starts``-based ragged/extend
    semantics or deterministic ties are required. It allocates its own
    outputs, so ``out_indices``/``out_values`` are filled by copy when the
    caller supplied them.
    """
    from .kernels.filtered_topk_decode import (  # noqa: PLC0415
        cute_dsl_radix_filter_topk_wrapper,
    )

    # The kernel ABI declares a symbolic leading stride (padded row views are
    # zero-copy) but requires a unit inner stride and a 32-byte-aligned base
    # for its vectorized loads; anything else previously failed late with an
    # opaque FFI alignment error. Materialize only the genuinely unsupported
    # layouts (rare: transposed/gathered views, or a base sliced off
    # alignment -- torch allocations themselves are 256-byte aligned).
    if logits.stride(-1) != 1 or (logits.data_ptr() % 32) != 0:
        logits = logits.contiguous()
        if (logits.data_ptr() % 32) != 0:
            logits = logits.clone()

    # Compile and launch under the input tensor's device: the persistent JIT
    # cache tags artifacts by the CURRENT device's architecture, and the
    # kernel launches on the current stream, so both must agree with where
    # the data lives on a multi-GPU host.
    with torch.cuda.device(logits.device):
        # Caller-supplied buffers are threaded through to the kernel launch, so
        # the kernel writes them directly: no per-call output allocation, no
        # num_rows x top_k device copy, and CUDA-graph users keep stable
        # destinations across replays.
        idx, val = cute_dsl_radix_filter_topk_wrapper(
            logits,
            seq_lens,
            top_k,
            next_n,
            return_val=return_output_values,
            out_indices=out_indices,
            out_values=out_values if return_output_values else None,
        )

    if out_indices is not None:
        idx = out_indices  # same storage; preserve the caller's shape/object
    if return_output_values and out_values is not None:
        val = out_values
    return idx, (val if return_output_values else None)


# ---------------------------------------------------------------------------
# Public API: top_k_varlen
# ---------------------------------------------------------------------------


@backend_requirement(
    {
        "radix": _radix_top_k_varlen_check,
        "gvr": _gvr_top_k_varlen_check,
        "radix_cutlass": _radix_cutlass_top_k_varlen_check,
        "radix_filter": _radix_filter_top_k_varlen_check,
    },
    heuristic_func=_top_k_varlen_heuristic,
)
@flashinfer_api(trace=top_k_varlen_trace)
def top_k_varlen(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    pre_idx: Optional[torch.Tensor] = None,
    compress_ratio: int = 1,
    next_n: int = 1,
    return_values: bool = False,
    out_indices: Optional[torch.Tensor] = None,
    out_values: Optional[torch.Tensor] = None,
    backend: Literal["radix", "gvr", "radix_cutlass", "radix_filter", "auto"] = "auto",
    load_balance: bool = True,
    workspace: Optional[dict] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Top-K selection over batched decode-step logits.

    Selects the top-``top_k`` elements from each row of ``logits``,
    respecting per-request KV-cache lengths given by ``seq_lens``.

    Backend selection
    -----------------
    ``backend="auto"`` (default) chooses GVR when available (Blackwell +
    ``pre_idx`` supplied), else the CuTe DSL ``radix`` backend on Blackwell,
    else the ``radix_cutlass`` masked fallback.  Force a specific backend with
    ``backend="radix"``, ``backend="gvr"``, or ``backend="radix_cutlass"``.

    Parameters
    ----------
    logits : torch.Tensor
        2-D float tensor of shape ``(num_rows, max_seq_len)``.
        Supported dtypes: ``float32``, ``bfloat16``, ``float16``.
        For the ``"gvr"`` backend the row width ``max_seq_len`` must be a
        multiple of ``16 // itemsize`` (8 for fp16/bf16, 4 for fp32) so each
        row is 16-byte aligned for GVR's 128-bit vectorized loads; a
        ``ValueError`` is raised otherwise.  The ``"radix_cutlass"`` backend has no
        such constraint.
    seq_lens : torch.Tensor
        1-D ``int32`` tensor of shape ``(num_rows // next_n,)`` with the
        effective KV-cache length per request.  Logits at or beyond
        ``seq_lens[i]`` are excluded from the search.
    top_k : int
        Number of top elements per row.  GVR backend supports
        ``{512, 1024, 2048}``; radix backend has no restriction.
    pre_idx : torch.Tensor, optional
        ``int32[num_rows // next_n, top_k]`` — top-K KV-cache indices
        selected by **this same layer** at the **previous token's decode
        step**.  GVR exploits the strong correlation between a layer's
        attention pattern at step ``t`` and step ``t+1``; the kernel
        internally applies a ``+1`` offset (DSv3.2) so the previous step's
        indices land correctly in the current step's grown KV-cache space.
        ``pre_idx[:, 0]`` must be the argmax index.
        Required by the ``"gvr"`` backend; ignored by ``"radix_cutlass"``.
    compress_ratio : int, optional
        KV-index compression factor (``1`` for DSv3.2, ``4`` for DSv4).
        Default ``1``.
    next_n : int, optional
        Speculative-decode temporal stride.  Default ``1``.
    return_values : bool, optional
        When ``True`` also return the selected logit values.
        Default ``False``.
    out_indices : torch.Tensor, optional
        Pre-allocated ``int32[num_rows, top_k]`` output buffer.
    out_values : torch.Tensor, optional
        Pre-allocated values buffer (same dtype as ``logits``).
        Only used when ``return_values=True``.
    backend : {"radix", "gvr", "radix_cutlass", "auto"}, optional
        Backend to use.  Default ``"auto"``.

        ``"radix"``         — CuTe DSL single-pass multi-CTA radix top-K
                              (Blackwell sm_100+ incl. Rubin sm_107; native
                              varlen, no ``pre_idx``,
                              no logit masking).
        ``"gvr"``           — GVR kernel (Blackwell sm_100+ only; requires
                              ``pre_idx``). ``load_balance`` selects the LB vs
                              single-CTA path.
        ``"radix_cutlass"`` — Masked CUTLASS radix top-K (all GPUs, no
                              ``pre_idx`` needed).
        ``"auto"``          — GVR (if ``pre_idx`` supplied) > radix (Blackwell)
                              > radix_cutlass.
    load_balance : bool, optional
        Selects the GVR kernel path (ignored by the radix backend).  Default
        ``True``.

        ``True`` (default) — two-kernel LB path (``GvrTopKLBPrepareKernel`` +
                     ``GvrTopKLBKernel``): a prepare kernel classifies requests
                     into long/short buckets, then the main kernel splits each
                     long row across a CTA cluster and packs short rows.  Best
                     for the ragged decode batches GVR targets.
        ``False``  — single-kernel path (``GvrTopKKernel``): one CTA per row,
                     no prepare step.  Faster when the batch has no length
                     variance (all rows short, or all long).

        Both settings are CUDA-graph safe (no host branch on device data).

    workspace : dict, optional
        Reusable workspace buffers for the GVR ``load_balance=True`` path.
        When provided, the LB prepare and decode kernels read/write these
        tensors instead of allocating fresh ones per call — useful in decode
        loops where the same batch size is reused at every step.

        Required keys (both ``int32`` on the same device as ``logits``):

        * ``"gvr_order_row"``: shape ``(M,)`` where ``M`` is the smallest
          power of 2 in ``[64, 1024]`` that is ``>= seq_lens.shape[0]``.
        * ``"gvr_counters"``: shape ``(2,)``.

        .. warning::
            Do **not** share the same workspace dict across concurrent CUDA
            streams — each stream must have its own workspace to avoid races
            on the device tensors.  When ``workspace`` is ``None`` (default)
            buffers are allocated locally and are safe for any concurrency.

    Returns
    -------
    (indices, values) : Tuple[torch.Tensor, Optional[torch.Tensor]]
        Always a 2-tuple. ``indices`` is ``int32[num_rows, top_k]``.
        ``values`` holds the selected logits (same dtype as ``logits``) when
        ``return_values=True``, otherwise ``None``.

    Raises
    ------
    BackendSupportedError
        If the requested backend is not supported on the current device or
        the required inputs (e.g. ``pre_idx``) are missing.

    Examples
    --------
    >>> import torch, flashinfer
    >>> torch.manual_seed(42)
    >>> B, N_max, top_k = 32, 8192, 1024
    >>>
    >>> # Step t: no prior indices; use radix to get the first top-K.
    >>> # Each request has a different KV-cache length in [top_k+1, N_max-1].
    >>> logits = torch.randn(B, N_max, dtype=torch.bfloat16, device="cuda")
    >>> seq_lens_t = torch.randint(top_k + 1, N_max, (B,), dtype=torch.int32, device="cuda")
    >>> indices_t, _ = flashinfer.top_k_varlen(logits, seq_lens_t, top_k, backend="radix_cutlass")
    >>> # Reference check: every selected value must be >= the K-th largest.
    >>> for i in range(B):
    ...     s = seq_lens_t[i].item()
    ...     kth = torch.topk(logits[i, :s].float(), top_k).values[-1]
    ...     assert (logits[i, :s].float()[indices_t[i].long()] < kth - 1e-5).sum() == 0
    >>>
    >>> # Step t+1: one new token appended per request; seq_lens grows by 1.
    >>> logits_t1 = torch.randn(B, N_max, dtype=torch.bfloat16, device="cuda")
    >>> seq_lens_t1 = seq_lens_t + 1
    >>> # Pass indices_t as pre_idx; GVR uses it to warm-start the threshold search.
    >>> indices_t1, _ = flashinfer.top_k_varlen(logits_t1, seq_lens_t1, top_k, pre_idx=indices_t)
    >>> for i in range(B):
    ...     s = seq_lens_t1[i].item()
    ...     kth = torch.topk(logits_t1[i, :s].float(), top_k).values[-1]
    ...     assert (logits_t1[i, :s].float()[indices_t1[i].long()] < kth - 1e-5).sum() == 0

    See Also
    --------
    flashinfer.top_k : General-purpose radix/clusters top-K (uniform lengths).
    """
    assert logits.is_cuda and logits.dim() == 2, "logits must be a 2-D CUDA tensor"
    assert seq_lens.is_cuda and seq_lens.dim() == 1 and seq_lens.dtype == torch.int32
    # Grouped-row ABI, shared by every backend: row r belongs to sequence
    # r // next_n, so seq_lens must hold exactly one entry per group. Validated
    # here (not in the per-backend checkers) because a violation is a silent
    # device-side OOB read of seq_lens or a wrong grouping in every backend,
    # and the API body still runs under skip_check=True. Real exceptions, not
    # asserts: this must hold under `python -O` too.
    if next_n < 1:
        raise ValueError(f"next_n must be >= 1, got {next_n}")
    if logits.shape[0] != seq_lens.shape[0] * next_n:
        raise ValueError(
            f"logits has {logits.shape[0]} rows but seq_lens has "
            f"{seq_lens.shape[0]} entries with next_n={next_n}: expected "
            f"seq_lens.shape[0] * next_n == logits.shape[0] "
            f"(= {seq_lens.shape[0] * next_n}). Rows are grouped as "
            f"row // next_n -> sequence; for per-row lengths pass next_n=1 "
            f"with one seq_lens entry per row."
        )

    if backend == "auto":
        backend = top_k_varlen.suitable_auto_backends[0]

    num_rows = logits.shape[0]
    if out_indices is None:
        out_indices = torch.empty(
            (num_rows, top_k), dtype=torch.int32, device=logits.device
        )
    if return_values and out_values is None:
        out_values = torch.empty(
            (num_rows, top_k), dtype=logits.dtype, device=logits.device
        )

    if backend == "radix":
        out_i, out_v = _run_radix(
            logits,
            seq_lens,
            top_k,
            next_n,
            compress_ratio,
            return_values,
            out_indices,
            out_values,
        )
    elif backend == "gvr":
        use_lb = bool(load_balance)
        if use_lb:
            out_i, out_v = _run_gvr_lb(
                logits,
                pre_idx,
                seq_lens,
                top_k,
                next_n,
                compress_ratio,
                return_values,
                out_indices,
                out_values,
                order_row=workspace.get("gvr_order_row") if workspace else None,
                counters=workspace.get("gvr_counters") if workspace else None,
            )
        else:
            out_i, out_v = _run_gvr(
                logits,
                pre_idx,
                seq_lens,
                top_k,
                next_n,
                compress_ratio,
                return_values,
                out_indices,
                out_values,
            )
    elif backend == "radix_cutlass":
        out_i, out_v = _run_radix_cutlass(
            logits,
            seq_lens,
            top_k,
            next_n,
            compress_ratio,
            return_values,
            out_indices,
            out_values,
        )
    elif backend == "radix_filter":
        out_i, out_v = _run_radix_filter(
            logits,
            seq_lens,
            top_k,
            next_n,
            compress_ratio,
            return_values,
            out_indices,
            out_values,
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Expected 'radix', 'gvr', 'radix_cutlass', or 'radix_filter'."
        )

    return out_i, out_v
