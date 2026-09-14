# Copyright (c) 2026 by FlashInfer team.
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

"""Task-scheduled DSV4 CSA FMHA task with the TRT-LLM launch ABI.

This module intentionally starts at contract-F. Cache append/compression,
candidate scoring, slot composition, and local-to-physical page-table lowering
are external operations. The launch inputs therefore match the ``trtllm-gen``
generated FMHA task rather than providing a second routing API.
"""

from __future__ import annotations

import ctypes
import functools
import math
import os
from typing import Optional

import torch


_HEADS = 128
_HEAD_DIM = 512
_SWA_TOPK = 128
_TILE_K = 128
_COMPILE_OPTIONS_PREFIX = "--enable-tvm-ffi"
_RAW_GATHER4_DESCRIPTOR_BYTES = 128
_RAW_GATHER4_DESCRIPTOR_PAIR_CACHE: dict[
    tuple[tuple[int, int, int], tuple[int, int, int]], torch.Tensor
] = {}


def _dsv4_persistent_enabled() -> bool:
    """Select the scheduler mode for a diagnostic DSV4 CSA build.

    Persistent is the production/source contract.  The opt-out exists solely
    to distinguish a work-id-pipeline failure from a dataflow failure while
    porting; it is deliberately part of the compilation-cache key below.
    """
    return os.environ.get("FLASHINFER_DSV4_CSA_STATIC_GRID", "0") not in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_raw_k_gather_enabled() -> bool:
    """Keep the raw W9-fed K issuer on except during a porting diagnosis."""
    return os.environ.get("FLASHINFER_DSV4_CSA_HIGHLEVEL_K", "0") not in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_pv_k128_enabled() -> bool:
    """Use the source BMM2 K128/raw-V/whole-O path by default.

    ``0`` is retained only as a diagnostic opt-out for the older K64/per-N-O
    fallback.  The K128 path has the source BMM2 operand shape and is the
    faster path qualified by the DSV4 CSA regression; callers must not need
    an environment setting to obtain it.
    """
    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_PV_K128", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_pair_ring_enabled() -> bool:
    """Use the qualified source W9/W12--W15 pair FSM by default.

    It is the source-shape page-index producer/consumer cardinality: W9
    advances one six-stage entry for two K128 tiles, while W12--W15 keep K one
    tile ahead of V.  The legacy one-entry-per-tile form is retained only as
    an explicit diagnostic opt-out; it has the same Contract-F result but is
    materially slower on the shared 8K workload.
    """
    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_PAIR_RING", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_workid_enabled() -> bool:
    """Use the qualified source CLC WorkId scheduler by default.

    This changes only the WorkId response path: one stage, 960 consumers, W10
    scheduler and W11 padding. Packed/multi-work correctness and BF16/Rope
    performance gates qualify the CLC path as production; explicit ``0``
    retains the static persistent grid-stride scheduler for diagnostics.
    """
    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_WORKID", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_throttle_enabled() -> bool:
    """Use source's W12--W15 to W10 CLC throttle by default.

    The throttle has no data payload and is deliberately independent from the
    WorkId response pipeline. It is canonicalized off when WorkId is disabled;
    explicit ``0`` retains an isolated no-throttle CLC diagnostic.
    """
    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_THROTTLE", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_s_release_after_p_enabled() -> bool:
    """Delay DSV4 S release until softmax has stored/fenced P by default.

    This is source's S/P happens-before edge and the readiness signal used by
    the direct-P path.  ``0`` remains available only together with disabling
    the source S handoff/direct-P diagnostics.
    """
    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_S_RELEASE_AFTER_P", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_s_handoff_enabled() -> bool:
    """Use the complete source W8 S acquire/commit timeline by default.

    Both the faster static grid-stride scheduler and the source CLC WorkId
    scheduler carry the qualified split acquire/commit cursors across work
    tiles.  Explicit ``0`` is retained for schedule A/B diagnostics.
    """
    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_S_HANDOFF", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_direct_p_enabled() -> bool:
    """Select source's direct P handoff after the source S cursor gate.

    The implementation forces the source S handoff when this switch is set;
    otherwise a direct-P launch could silently reintroduce F-034's missing
    P-ready dependency. K-boundary, partial-tile, packed persistent, and 8K
    performance gates qualify this as the default; ``0`` remains diagnostic.
    """
    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_DIRECT_P", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_early_corr_enabled() -> bool:
    """Select TRTLLM-gen's early-max/terminal-stat correction pipeline.

    The source sends one ``(old_max, new_max)`` token per sparse K tile before
    P materialization, followed by one terminal ``(row_sum, row_max)`` token.
    Boundary, packed-persistent, and shared-8K gates qualify it as the default;
    explicit ``0`` retains the former full-stat handoff for diagnosis.
    """
    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_EARLY_CORR", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_softmax_pair_reduction_enabled() -> bool:
    """Select source's staged 2x2-warp row-max exchange by default.

    ``0`` retains the prior one-buffer, full-warpgroup, two-barrier reduction
    solely for isolated correctness/performance comparison.
    """
    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_SOFTMAX_PAIR_REDUCTION", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_softmax_pipeline_enabled() -> bool:
    """Use the qualified generated TmemS/TmemP software schedule by default.

    The source schedule aliases P with the consumed S stage, uses the S empty
    cursor as P-ready, and preserves the generated row-reduction/conversion
    order.  ``0`` retains the former standalone-SmemP graph only for an
    isolated porting A/B.
    """

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_SOFTMAX_PIPELINE", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_contractible_ffma_enabled() -> bool:
    """Emit the qualified source-shaped contractible softmax arithmetic.

    Packed MUL+ADD pairs and the final FFMA expose the generated source's
    contraction/scheduling freedom. Correctness and three-variant performance
    gates qualify this as production; explicit ``0`` remains an isolated C0
    lowering diagnostic.
    """

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_CONTRACTIBLE_FFMA", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_epilogue_pair_barrier_enabled() -> bool:
    """Use source's two independent 64-thread correction reductions."""

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_EPILOGUE_PAIR_BARRIER", "0") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_page_offsets_cache_policy_enabled() -> bool:
    """Use source W9's qualified ``.cg.L2::128B`` cache policy by default."""

    return os.environ.get(
        "FLASHINFER_DSV4_CSA_SOURCE_PAGE_OFFSETS_CACHE_POLICY", "1"
    ) in ("1", "true", "TRUE")


def _dsv4_source_page_offsets_transcnt_enabled() -> bool:
    """Use source's two-instruction W9 barrier publication protocol.

    The default ``0`` keeps CUTLASS DSL's semantically equivalent single
    ``cp.async.mbarrier.arrive.noinc`` lowering.  This switch exists for an
    isolated source-SASS/performance experiment rather than as a production
    requirement.
    """

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_PAGE_OFFSETS_TRANSCNT", "0") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_tmem_lifecycle_enabled() -> bool:
    """Use source W8-signal/W4-deallocate TMEM exit ownership.

    This remains an opt-in source-protocol experiment until fresh lowering,
    persistent phase-wrap correctness, and paired timing all qualify it.
    """

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_TMEM_LIFECYCLE", "0") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_loader_no_tail_enabled() -> bool:
    """Skip source-absent terminal drains for W9 and W12--W15 loaders."""

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_LOADER_NO_TAIL", "0") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_register_protocol_enabled() -> bool:
    """Use one source-shaped setmaxnreg action per DSV4 warp role."""

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_REGISTER_PROTOCOL", "0") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_active_tma_prefetch_enabled() -> bool:
    """Prefetch only Q and the two raw Gather4 tensor maps by default."""

    return os.environ.get("FLASHINFER_DSV4_CSA_ACTIVE_TMA_PREFETCH", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_mbarrier_init_enabled() -> bool:
    """Batch active pipeline mbarrier initialization in one W0 region."""

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_MBAR_INIT", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_ptx_knobs_enabled() -> bool:
    """Emit the PTXAS scheduling pragmas used by the generated DSV4 kernel."""

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_PTX_KNOBS", "0") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_source_local_ptx_knobs_enabled() -> bool:
    """Emit qualified function-local source scheduling hints by default."""

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_LOCAL_PTX_KNOBS", "1") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_compiler_opt_level() -> int:
    """Return the isolated CuTe compiler optimization-level diagnostic.

    TRTLLM-gen compiles the generated source with ``-O3``.  The first TS
    prototype explicitly selected O2, so keep the choice in the compilation
    key while measuring whether O3 is safe and useful for this kernel.
    """

    raw = os.environ.get("FLASHINFER_DSV4_CSA_OPT_LEVEL", "2")
    try:
        level = int(raw)
    except ValueError as exc:
        raise ValueError("FLASHINFER_DSV4_CSA_OPT_LEVEL must be an integer") from exc
    if level < 0 or level > 3:
        raise ValueError("FLASHINFER_DSV4_CSA_OPT_LEVEL must be in [0, 3]")
    return level


def _dsv4_source_ptxas_options_enabled() -> bool:
    """Enable the native-compatible subset of TRTLLM-gen's ptxas options."""

    return os.environ.get("FLASHINFER_DSV4_CSA_SOURCE_PTXAS_OPTIONS", "0") in (
        "1",
        "true",
        "TRUE",
    )


def _dsv4_persistent_cluster_cap() -> int:
    """Return an optional positive cap for the DSV4 persistent grid.

    ``0`` keeps the hardware-reported active-cluster count.  A smaller cap is
    a porting diagnostic for the static grid-stride scheduler, and is included
    in the JIT key below so a benchmark cannot accidentally reuse another
    launch shape.
    """
    raw = os.environ.get("FLASHINFER_DSV4_CSA_MAX_ACTIVE_CLUSTERS", "0")
    try:
        cap = int(raw)
    except ValueError as exc:
        raise ValueError(
            "FLASHINFER_DSV4_CSA_MAX_ACTIVE_CLUSTERS must be an integer"
        ) from exc
    if cap < 0:
        raise ValueError("FLASHINFER_DSV4_CSA_MAX_ACTIVE_CLUSTERS must be >= 0")
    return cap


def _encode_raw_gather4_descriptor_host(pool: torch.Tensor) -> torch.Tensor:
    """Encode the source DSV4 ``[H128, page-slot=1]`` map on the host.

    The public CuTe Gather4 atom API only creates a dense-MMA-width map.
    TRTLLM-gen instead encodes a token-sparse 2-D map over ``[512, INT_MAX]``
    with one page slot per transaction.  Keep the encoded bytes host-resident
    until the complete descriptor object (or pair) is assembled: mutating a
    device-resident tensor-map object would require explicit tensor-map proxy
    fencing before the first TMA use.
    """

    from cuda.bindings import driver as cuda_drv

    err, tensor_map = cuda_drv.cuTensorMapEncodeTiled(
        cuda_drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        pool.data_ptr(),
        [
            cuda_drv.cuuint64_t(_HEAD_DIM),
            cuda_drv.cuuint64_t((1 << 31) - 1),
        ],
        [cuda_drv.cuuint64_t(_HEAD_DIM)],
        [cuda_drv.cuuint32_t(128), cuda_drv.cuuint32_t(1)],
        [cuda_drv.cuuint32_t(1), cuda_drv.cuuint32_t(1)],
        cuda_drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda_drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda_drv.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        cuda_drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled for DSV4 Gather4 failed: {err}")

    return torch.tensor(
        list(ctypes.string_at(tensor_map.getPtr(), _RAW_GATHER4_DESCRIPTOR_BYTES)),
        dtype=torch.uint8,
    )


def _raw_gather4_descriptor_pair(
    sliding_window_kv_pool: torch.Tensor, compressed_kv_pool: torch.Tensor
) -> torch.Tensor:
    """Pack raw Gather4 maps in the existing static-split workspace ABI."""

    swa_key = (
        sliding_window_kv_pool.device.index or 0,
        sliding_window_kv_pool.data_ptr(),
        sliding_window_kv_pool.numel(),
    )
    compressed_key = (
        compressed_kv_pool.device.index or 0,
        compressed_kv_pool.data_ptr(),
        compressed_kv_pool.numel(),
    )
    key = (swa_key, compressed_key)
    cached = _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE.get(key)
    if cached is not None:
        return cached

    # TRTLLM-gen constructs both TMA maps on the host.  Concatenate there and
    # publish the immutable 256-byte pair with one ordered H2D transfer.  The
    # former device ``empty`` + two D2D ``copy_`` sequence was shape-correct,
    # but it did not emit ``fence.proxy.tensormap`` between descriptor mutation
    # and TMA consumption and therefore relied on an undocumented proxy-order
    # interaction.
    host_descriptor_pair = torch.cat(
        (
            _encode_raw_gather4_descriptor_host(sliding_window_kv_pool),
            _encode_raw_gather4_descriptor_host(compressed_kv_pool),
        )
    )
    descriptor_pair = host_descriptor_pair.to(
        device=sliding_window_kv_pool.device, non_blocking=False
    )
    if descriptor_pair.data_ptr() % 64:
        raise RuntimeError("DSV4 Gather4 descriptor pair must be 64-byte aligned")
    _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE[key] = descriptor_pair
    return descriptor_pair


def _require_cuda_tensor(
    tensor: torch.Tensor, name: str, *, dtype: torch.dtype, ndim: int
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got rank {tensor.ndim}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


def _require_int32_cuda_vector(tensor: torch.Tensor, name: str) -> None:
    _require_cuda_tensor(tensor, name, dtype=torch.int32, ndim=1)


@functools.lru_cache(maxsize=None)
def _get_compiled_dsv4_csa(
    device_index: int,
    is_persistent: bool,
    use_raw_k_gather: bool,
    use_source_pv_k128: bool,
    use_source_pair_ring: bool,
    use_source_workid: bool,
    use_source_throttle: bool,
    release_s_after_p: bool,
    use_source_s_handoff: bool,
    use_source_direct_p: bool,
    use_source_early_corr: bool,
    use_source_softmax_pair_reduction: bool,
    use_source_softmax_pipeline: bool,
    use_source_contractible_ffma: bool,
    use_source_epilogue_pair_barrier: bool,
    use_source_page_offsets_cache_policy: bool,
    use_source_page_offsets_transcnt: bool,
    use_source_tmem_lifecycle: bool,
    use_source_loader_no_tail: bool,
    use_source_register_protocol: bool,
    use_active_tma_prefetch: bool,
    use_source_mbarrier_init: bool,
    use_source_ptx_knobs: bool,
    use_source_local_ptx_knobs: bool,
    compiler_opt_level: int,
    use_source_ptxas_options: bool,
    enable_skip_correction: bool,
    fuses_inv_rope_fp8_quant: bool,
    stores_lse: bool,
    persistent_cluster_cap: int,
):
    """Compile one DSV4 CSA launch specialization.

    Packed ``B/T/maxQ/Kmax`` are runtime launch fields, matching TRTLLM-gen's
    parameter ABI.  They must not enter this cache key: normal decode-serving
    shape changes do not control any static CuTe layout or resource allocation
    in this DSV4 specialization.
    """

    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    from cuda.bindings import driver as cuda_drv

    from .kernels.mla_decode.throughput_2cta.kernel import MlaDecodeTs

    with torch.cuda.device(device_index):
        stream = cuda_drv.CUstream(torch.cuda.current_stream(device_index).cuda_stream)
        max_active_clusters = cutlass_utils.HardwareInfo(
            device_index
        ).get_max_active_clusters(2, stream)
    if persistent_cluster_cap:
        max_active_clusters = min(max_active_clusters, persistent_cluster_cap)

    kernel = MlaDecodeTs(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        max_active_clusters=max_active_clusters,
        page_size=1,
        is_persistent=is_persistent,
        is_var_seq=True,
        is_var_split_kv=False,
        static_split_kv=1,
        static_seq_len_k=None,
        qkv_dtype="e4m3",
        out_dtype="e4m3" if fuses_inv_rope_fp8_quant else "bf16",
        rope_dim=0,
        num_heads=_HEADS,
        # DSV4's group ratio is fixed at one by Hq=M=128. Runtime maxQ/B
        # arrive through the packed metadata/scalar ABI and affect only the
        # scheduler; these representative values retain constructor checks.
        seq_len_q=1,
        batch_size=1,
        mask_type="causal",
        is_dynamic_token_sparse=True,
        sparse_swa_topk=_SWA_TOPK,
        dsv4_use_raw_k_gather=use_raw_k_gather,
        dsv4_use_source_pv_k128=use_source_pv_k128,
        dsv4_use_source_pair_ring=use_source_pair_ring,
        dsv4_use_source_workid=use_source_workid,
        dsv4_use_source_throttle=use_source_throttle,
        dsv4_release_s_after_p=release_s_after_p,
        dsv4_use_source_s_handoff=use_source_s_handoff,
        dsv4_use_source_direct_p=use_source_direct_p,
        dsv4_use_source_early_corr=use_source_early_corr,
        dsv4_use_source_softmax_pair_reduction=(use_source_softmax_pair_reduction),
        dsv4_use_source_softmax_pipeline=use_source_softmax_pipeline,
        dsv4_use_source_contractible_ffma=use_source_contractible_ffma,
        dsv4_use_source_epilogue_pair_barrier=(use_source_epilogue_pair_barrier),
        dsv4_use_source_page_offsets_cache_policy=(
            use_source_page_offsets_cache_policy
        ),
        dsv4_use_source_page_offsets_transcnt=(use_source_page_offsets_transcnt),
        dsv4_use_source_tmem_lifecycle=use_source_tmem_lifecycle,
        dsv4_use_source_loader_no_tail=use_source_loader_no_tail,
        dsv4_use_source_register_protocol=use_source_register_protocol,
        dsv4_use_active_tma_prefetch=use_active_tma_prefetch,
        dsv4_use_source_mbarrier_init=use_source_mbarrier_init,
        dsv4_use_source_ptx_knobs=use_source_ptx_knobs,
        dsv4_use_source_local_ptx_knobs=use_source_local_ptx_knobs,
        dsv4_enable_skip_correction=enable_skip_correction,
        dsv4_fuses_inv_rope_fp8_quant=fuses_inv_rope_fp8_quant,
        stores_lse=stores_lse,
    )
    compressed_rows = cute.sym_int()
    swa_rows = cute.sym_int()
    runtime_batch = cute.sym_int()
    runtime_num_q_offsets = cute.sym_int()
    runtime_total_q = cute.sym_int()
    runtime_sparse_capacity = cute.sym_int()
    q_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (_HEADS, _HEAD_DIM, runtime_total_q),
        stride=(_HEAD_DIM, 1, _HEADS * _HEAD_DIM),
        assumed_align=16,
    )
    compressed_cache_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (1, _HEAD_DIM, compressed_rows),
        stride=(_HEAD_DIM, 1, _HEAD_DIM),
        assumed_align=16,
    )
    swa_cache_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (1, _HEAD_DIM, swa_rows),
        stride=(_HEAD_DIM, 1, _HEAD_DIM),
        assumed_align=16,
    )
    physical_indices_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (runtime_sparse_capacity, runtime_total_q),
        stride=(1, runtime_sparse_capacity),
        assumed_align=16,
    )
    if fuses_inv_rope_fp8_quant:
        # The source physical value tensor is [H/8,T,8,D].  The device
        # epilogue owns its non-affine logical-head mapping, so expose the
        # storage as one compact element span at the shared kernel boundary.
        out_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (runtime_total_q * _HEADS * _HEAD_DIM,),
            stride_order=(0,),
            assumed_align=16,
        )
        runtime_cos_sin_elts = cute.sym_int()
        runtime_scale_elts = cute.sym_int()
        inv_rope_cos_sin_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (runtime_cos_sin_elts,),
            stride_order=(0,),
            assumed_align=16,
        )
        dsv4_o_scale_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (runtime_scale_elts,),
            stride_order=(0,),
            assumed_align=16,
        )
    else:
        out_fake = cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (_HEADS, _HEAD_DIM, runtime_total_q),
            stride=(_HEAD_DIM, 1, _HEADS * _HEAD_DIM),
            assumed_align=16,
        )
        inv_rope_cos_sin_fake = None
        dsv4_o_scale_fake = None
    lse_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (_HEADS, runtime_total_q),
        stride=(1, _HEADS),
        assumed_align=16,
    )
    raw_seq_lens_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (runtime_batch,), stride_order=(0,), assumed_align=16
    )
    sparse_lens_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (runtime_total_q,), stride_order=(0,), assumed_align=16
    )
    cu_seqlens_q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (runtime_num_q_offsets,),
        stride_order=(0,),
        assumed_align=4,
    )
    raw_tma_descriptor_pair_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (2 * _RAW_GATHER4_DESCRIPTOR_BYTES,),
        stride_order=(0,),
        assumed_align=64,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_specializations: tuple[object, ...] = (cute.FrontendNext,)
    compile_options = f"{_COMPILE_OPTIONS_PREFIX} --opt-level {compiler_opt_level}"
    if use_source_ptxas_options:
        # ``options=`` is parsed into a fresh CompileOptions object by CuTe DSL.
        # Keep the ptxas flags in that same channel: a PtxasOptions specialization
        # passed through ``cute.compile[...]`` would otherwise be overwritten.
        # The current in-process backend accepts ``-uumn`` and uses the emitted
        # user-machine pragmas.  It rejects ``-knob AntiDepWeight=25``; that
        # source option remains external/manual-ptxas-only until the DSL backend
        # exposes a compatible pass-through path.
        compile_options += " --ptxas-options=-uumn"
    with torch.cuda.device(device_index):
        return cute.compile[compile_specializations](
            kernel,
            q_fake,
            q_fake,
            compressed_cache_fake,
            compressed_cache_fake,
            physical_indices_fake,
            out_fake,
            lse_fake,
            None,
            raw_tma_descriptor_pair_fake,
            cutlass.Int32(1),
            raw_seq_lens_fake,
            sparse_lens_fake,
            cu_seqlens_q_fake,
            None,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            swa_cache_fake,
            cutlass.Int32(1),
            cutlass.Float32(8.0 if enable_skip_correction else 0.0),
            inv_rope_cos_sin_fake,
            dsv4_o_scale_fake,
            stream_fake,
            options=compile_options,
        )


def _validate_optional_values(
    *,
    sparse_topk_lens: torch.Tensor,
    sparse_capacity: int,
    seq_lens_kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> None:
    """Perform device-synchronizing value checks only when requested."""

    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") in ("", "0"):
        return
    if int(cu_seqlens_q[0]) != 0 or int(cu_seqlens_q[-1]) != sparse_topk_lens.numel():
        raise ValueError("ptr_cum_seq_lens_q must start at zero and end at T")
    query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    if torch.any(query_lens <= 0):
        raise ValueError("every packed request must contain at least one Q row")
    if torch.any(seq_lens_kv < query_lens):
        raise ValueError("ptr_seq_lens_kv must be raw lengths after this forward")
    if torch.any(sparse_topk_lens < _SWA_TOPK) or torch.any(
        sparse_topk_lens > sparse_capacity
    ):
        raise ValueError("ptr_sparse_mla_topk_lens values must be in [128, Kmax]")


def _compiled_dsv4_for_launch(
    *,
    device_index: int,
    enable_skip_correction: bool,
    fuses_inv_rope_fp8_quant: bool,
    stores_lse: bool,
):
    """Resolve canonical source-shaped switches and return one JIT variant."""

    use_source_workid = _dsv4_source_workid_enabled()
    # Throttle has no valid standalone meaning without WorkId.
    use_source_throttle = use_source_workid and _dsv4_source_throttle_enabled()
    use_source_softmax_pipeline = _dsv4_source_softmax_pipeline_enabled()
    # Source's P store/fence, S release, and W8 acquire form one happens-before
    # chain. Canonicalize it before constructing the compilation key.
    use_source_direct_p = _dsv4_source_direct_p_enabled() or use_source_softmax_pipeline
    use_source_early_corr = (
        _dsv4_source_early_corr_enabled() or use_source_softmax_pipeline
    )
    use_source_s_handoff = _dsv4_source_s_handoff_enabled() or use_source_direct_p
    release_s_after_p = _dsv4_source_s_release_after_p_enabled() or use_source_s_handoff
    return _get_compiled_dsv4_csa(
        device_index,
        _dsv4_persistent_enabled(),
        _dsv4_raw_k_gather_enabled(),
        _dsv4_source_pv_k128_enabled(),
        _dsv4_source_pair_ring_enabled(),
        use_source_workid,
        use_source_throttle,
        release_s_after_p,
        use_source_s_handoff,
        use_source_direct_p,
        use_source_early_corr,
        _dsv4_source_softmax_pair_reduction_enabled(),
        use_source_softmax_pipeline,
        _dsv4_source_contractible_ffma_enabled(),
        _dsv4_source_epilogue_pair_barrier_enabled(),
        _dsv4_source_page_offsets_cache_policy_enabled(),
        _dsv4_source_page_offsets_transcnt_enabled(),
        _dsv4_source_tmem_lifecycle_enabled(),
        _dsv4_source_loader_no_tail_enabled(),
        _dsv4_source_register_protocol_enabled(),
        _dsv4_active_tma_prefetch_enabled(),
        _dsv4_source_mbarrier_init_enabled(),
        _dsv4_source_ptx_knobs_enabled(),
        _dsv4_source_local_ptx_knobs_enabled(),
        _dsv4_compiler_opt_level(),
        _dsv4_source_ptxas_options_enabled(),
        enable_skip_correction,
        fuses_inv_rope_fp8_quant,
        stores_lse,
        _dsv4_persistent_cluster_cap(),
    )


def prims_ts_dsv4_csa(
    query: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    ptr_page_idx_kv: torch.Tensor,
    ptr_sparse_mla_topk_lens: torch.Tensor,
    ptr_seq_lens_kv: torch.Tensor,
    ptr_cum_seq_lens_q: torch.Tensor,
    *,
    max_seq_len_q: int,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    skip_corr_threshold: float = 8.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Run one DSV4 CSA FMHA task over TRT-LLM-compatible physical metadata.

    ``query`` is packed E4M3 ``[T, 128, 512]``. Both KV pools are independent
    physical E4M3 row pools ``[N, 512]``. ``ptr_page_idx_kv[T, Kmax]`` is
    physical routing: slots ``[0,128)`` address SWA and the remainder address
    the compressed pool. To match generated TRTLLM dynamic routing, inactive
    SWA slots may be ``-1``; the raw Gather4/TMA path and subsequent mask must
    preserve the source behavior. Other reachable slots must follow the source
    selector allocation contract rather than being rewritten by this API.
    ``ptr_sparse_mla_topk_lens[T]`` is Lq, the active sparse *scan width*. Raw
    post-append lengths and packed Q offsets are
    ``ptr_seq_lens_kv[B]`` and ``ptr_cum_seq_lens_q[B+1]``.

    ``skip_corr_threshold`` follows TRTLLM-gen's E4M3 enable contract. A
    positive value automatically selects the skip-correction kernel
    specialization; the recommended and default value is ``8.0``. Passing
    ``0.0`` disables it for diagnostics. The runtime threshold controls row-max
    freezing, the E4M3 P scale (1.75 versus 448), and the warp-wide correction
    skip as one indivisible numerical protocol.

    The routing tensors are contract-E outputs. This function does not append
    or compress KV, select CSA candidates, or lower page tables. Those stages
    must happen before the timed FMHA task. ``max_seq_len_q`` is runtime launch
    geometry (the maximum request-local Q length), not a tensor extent baked
    into the compiled specialization.
    """

    _require_cuda_tensor(query, "query", dtype=torch.float8_e4m3fn, ndim=3)
    if tuple(query.shape[1:]) != (_HEADS, _HEAD_DIM):
        raise ValueError(
            f"query must have shape [T, {_HEADS}, {_HEAD_DIM}], got {tuple(query.shape)}"
        )
    total_q = int(query.shape[0])
    if total_q <= 0:
        raise ValueError("query must contain at least one packed Q row")
    if not isinstance(max_seq_len_q, int) or max_seq_len_q <= 0:
        raise ValueError("max_seq_len_q must be a positive Python int")
    device = query.device
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 3)):
        raise NotImplementedError(
            f"DSV4 CSA TS requires SM100 or SM103, got SM{capability[0]}{capability[1]}"
        )

    for pool, name in (
        (compressed_kv_pool, "compressed_kv_pool"),
        (sliding_window_kv_pool, "sliding_window_kv_pool"),
    ):
        _require_cuda_tensor(pool, name, dtype=torch.float8_e4m3fn, ndim=2)
        if pool.shape[0] <= 0 or pool.shape[1] != _HEAD_DIM:
            raise ValueError(f"{name} must have shape [N>0, {_HEAD_DIM}]")
        if pool.device != device:
            raise ValueError(f"{name} must be on {device}, got {pool.device}")

    _require_cuda_tensor(ptr_page_idx_kv, "ptr_page_idx_kv", dtype=torch.int32, ndim=2)
    if ptr_page_idx_kv.device != device or ptr_page_idx_kv.shape[0] != total_q:
        raise ValueError("ptr_page_idx_kv must have shape [T, Kmax] on query.device")
    sparse_capacity = int(ptr_page_idx_kv.shape[1])
    # The source W9 loads four int32 selector entries per lane with a 16-byte
    # cp.async and clamps the pair tail to ``((Lq - 1) >> 2) << 2``.  Hence
    # Kmax must be int32x4-addressable, but it need not be a full K128 tile:
    # the generated HCA smoke uses Kmax=192 (128 SWA + 64 compressed slots).
    if sparse_capacity < _SWA_TOPK or sparse_capacity % 4:
        raise ValueError("Kmax must be at least 128 and divisible by 4")

    _require_int32_cuda_vector(ptr_sparse_mla_topk_lens, "ptr_sparse_mla_topk_lens")
    if ptr_sparse_mla_topk_lens.device != device or ptr_sparse_mla_topk_lens.shape != (
        total_q,
    ):
        raise ValueError("ptr_sparse_mla_topk_lens must have shape [T] on query.device")
    _require_int32_cuda_vector(ptr_seq_lens_kv, "ptr_seq_lens_kv")
    _require_int32_cuda_vector(ptr_cum_seq_lens_q, "ptr_cum_seq_lens_q")
    if ptr_seq_lens_kv.device != device or ptr_cum_seq_lens_q.device != device:
        raise ValueError("sequence metadata must be on query.device")
    batch_size = int(ptr_seq_lens_kv.numel())
    if batch_size <= 0 or ptr_cum_seq_lens_q.shape != (batch_size + 1,):
        raise ValueError("sequence metadata must have shapes [B] and [B + 1]")
    _validate_optional_values(
        sparse_topk_lens=ptr_sparse_mla_topk_lens,
        sparse_capacity=sparse_capacity,
        seq_lens_kv=ptr_seq_lens_kv,
        cu_seqlens_q=ptr_cum_seq_lens_q,
    )
    for scale, name in ((bmm1_scale, "bmm1_scale"), (bmm2_scale, "bmm2_scale")):
        if not isinstance(scale, (float, int)) or not math.isfinite(float(scale)):
            raise ValueError(f"{name} must be a finite Python number")
    if (
        isinstance(skip_corr_threshold, bool)
        or not isinstance(skip_corr_threshold, (float, int))
        or not math.isfinite(float(skip_corr_threshold))
        or not 0.0 <= float(skip_corr_threshold) <= 8.0
    ):
        raise ValueError("skip_corr_threshold must be a finite number in [0, 8]")
    enable_skip_correction = float(skip_corr_threshold) > 0.0
    if enable_skip_correction and float(bmm1_scale) <= 0.0:
        raise ValueError("positive skip_corr_threshold requires a positive bmm1_scale")

    expected_out_shape = (total_q, _HEADS, _HEAD_DIM)
    if out is None:
        out = torch.empty(expected_out_shape, device=device, dtype=torch.bfloat16)
    else:
        _require_cuda_tensor(out, "out", dtype=torch.bfloat16, ndim=3)
        if out.device != device or tuple(out.shape) != expected_out_shape:
            raise ValueError(f"out must have shape {expected_out_shape}")
    expected_lse_shape = (total_q, _HEADS)
    stores_lse = lse is not None
    if lse is None:
        # Keep the fixed kernel ABI (and its compile-time shape validation)
        # while specializing the epilogue away from the otherwise invisible
        # LSE log/store path.  TRTLLM's source benchmark selects the analogous
        # ``storesSoftmaxStats=false`` variant.
        lse_kernel = torch.empty(expected_lse_shape, device=device, dtype=torch.float32)
    else:
        _require_cuda_tensor(lse, "lse", dtype=torch.float32, ndim=2)
        if lse.device != device or tuple(lse.shape) != expected_lse_shape:
            raise ValueError(f"lse must have shape {expected_lse_shape}")
        lse_kernel = lse

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    compiled = _compiled_dsv4_for_launch(
        device_index=device_index,
        enable_skip_correction=enable_skip_correction,
        fuses_inv_rope_fp8_quant=False,
        stores_lse=stores_lse,
    )
    q_kernel = query.permute(1, 2, 0)
    compressed_kernel = compressed_kv_pool.view(-1, 1, _HEAD_DIM).permute(1, 2, 0)
    swa_kernel = sliding_window_kv_pool.view(-1, 1, _HEAD_DIM).permute(1, 2, 0)
    raw_tma_descriptor_pair = _raw_gather4_descriptor_pair(
        sliding_window_kv_pool, compressed_kv_pool
    )
    compiled(
        q_kernel,
        q_kernel,
        compressed_kernel,
        compressed_kernel,
        ptr_page_idx_kv.transpose(0, 1),
        out.permute(1, 2, 0),
        lse_kernel.transpose(0, 1),
        None,
        raw_tma_descriptor_pair,
        1,
        ptr_seq_lens_kv,
        ptr_sparse_mla_topk_lens,
        ptr_cum_seq_lens_q,
        None,
        float(bmm1_scale),
        float(bmm2_scale),
        swa_kernel,
        max_seq_len_q,
        float(skip_corr_threshold),
        None,
        None,
    )
    return out


def prims_ts_dsv4_sparse_mla_rope_quant(
    query: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    ptr_page_idx_kv: torch.Tensor,
    ptr_sparse_mla_topk_lens: torch.Tensor,
    ptr_seq_lens_kv: torch.Tensor,
    ptr_cum_seq_lens_q: torch.Tensor,
    inv_rope_cos_sin_cache: torch.Tensor,
    *,
    max_seq_len_q: int,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    skip_corr_threshold: float = 8.0,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Run source-compatible DSV4 sparse MLA with fused inverse-RoPE/FP8 quant.

    Attention and routing follow :func:`prims_ts_dsv4_csa`; the same compiled
    kernel handles CSA (normally R=4) and HCA (normally R=128).  The fused
    epilogue matches TRTLLM-gen's physical ABI rather than returning logical
    ``[T,H,D]`` storage:

    * ``out`` is E4M3 ``[16, T, 8, 512]``;
    * ``out_scale`` is FP32 ``[16, 32, pad4(T)]``;
    * every contiguous D128 block has one dequant scale ``amax / 448``;
    * inverse non-NeoX RoPE is applied only to logical D ``[448:512]`` before
      quantizing the last block, using cache rows ``[cos(32), sin(32)]``.

    For packed request ``b`` and local query ``q``, the cache position is
    ``ptr_seq_lens_kv[b] - query_len[b] + q``.  Therefore
    ``inv_rope_cos_sin_cache`` must be FP32 ``[max_position, 64]`` and cover
    every such position.  As in TRTLLM-gen, skip correction defaults to 8.
    """

    _require_cuda_tensor(query, "query", dtype=torch.float8_e4m3fn, ndim=3)
    if tuple(query.shape[1:]) != (_HEADS, _HEAD_DIM):
        raise ValueError(
            f"query must have shape [T, {_HEADS}, {_HEAD_DIM}], got {tuple(query.shape)}"
        )
    total_q = int(query.shape[0])
    if total_q <= 0:
        raise ValueError("query must contain at least one packed Q row")
    if not isinstance(max_seq_len_q, int) or max_seq_len_q <= 0:
        raise ValueError("max_seq_len_q must be a positive Python int")
    device = query.device
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 3)):
        raise NotImplementedError(
            f"DSV4 sparse MLA TS requires SM100 or SM103, got "
            f"SM{capability[0]}{capability[1]}"
        )

    for pool, name in (
        (compressed_kv_pool, "compressed_kv_pool"),
        (sliding_window_kv_pool, "sliding_window_kv_pool"),
    ):
        _require_cuda_tensor(pool, name, dtype=torch.float8_e4m3fn, ndim=2)
        if pool.shape[0] <= 0 or pool.shape[1] != _HEAD_DIM:
            raise ValueError(f"{name} must have shape [N>0, {_HEAD_DIM}]")
        if pool.device != device:
            raise ValueError(f"{name} must be on {device}, got {pool.device}")

    _require_cuda_tensor(ptr_page_idx_kv, "ptr_page_idx_kv", dtype=torch.int32, ndim=2)
    if ptr_page_idx_kv.device != device or ptr_page_idx_kv.shape[0] != total_q:
        raise ValueError("ptr_page_idx_kv must have shape [T, Kmax] on query.device")
    sparse_capacity = int(ptr_page_idx_kv.shape[1])
    if sparse_capacity < _SWA_TOPK or sparse_capacity % 4:
        raise ValueError("Kmax must be at least 128 and divisible by 4")

    _require_int32_cuda_vector(ptr_sparse_mla_topk_lens, "ptr_sparse_mla_topk_lens")
    if ptr_sparse_mla_topk_lens.device != device or ptr_sparse_mla_topk_lens.shape != (
        total_q,
    ):
        raise ValueError("ptr_sparse_mla_topk_lens must have shape [T] on query.device")
    _require_int32_cuda_vector(ptr_seq_lens_kv, "ptr_seq_lens_kv")
    _require_int32_cuda_vector(ptr_cum_seq_lens_q, "ptr_cum_seq_lens_q")
    if ptr_seq_lens_kv.device != device or ptr_cum_seq_lens_q.device != device:
        raise ValueError("sequence metadata must be on query.device")
    batch_size = int(ptr_seq_lens_kv.numel())
    if batch_size <= 0 or ptr_cum_seq_lens_q.shape != (batch_size + 1,):
        raise ValueError("sequence metadata must have shapes [B] and [B + 1]")
    _validate_optional_values(
        sparse_topk_lens=ptr_sparse_mla_topk_lens,
        sparse_capacity=sparse_capacity,
        seq_lens_kv=ptr_seq_lens_kv,
        cu_seqlens_q=ptr_cum_seq_lens_q,
    )

    for scale, name in ((bmm1_scale, "bmm1_scale"), (bmm2_scale, "bmm2_scale")):
        if not isinstance(scale, (float, int)) or not math.isfinite(float(scale)):
            raise ValueError(f"{name} must be a finite Python number")
    if (
        isinstance(skip_corr_threshold, bool)
        or not isinstance(skip_corr_threshold, (float, int))
        or not math.isfinite(float(skip_corr_threshold))
        or not 0.0 <= float(skip_corr_threshold) <= 8.0
    ):
        raise ValueError("skip_corr_threshold must be a finite number in [0, 8]")
    enable_skip_correction = float(skip_corr_threshold) > 0.0
    if enable_skip_correction and float(bmm1_scale) <= 0.0:
        raise ValueError("positive skip_corr_threshold requires a positive bmm1_scale")

    _require_cuda_tensor(
        inv_rope_cos_sin_cache,
        "inv_rope_cos_sin_cache",
        dtype=torch.float32,
        ndim=2,
    )
    if (
        inv_rope_cos_sin_cache.device != device
        or inv_rope_cos_sin_cache.shape[0] <= 0
        or inv_rope_cos_sin_cache.shape[1] != 64
    ):
        raise ValueError(
            "inv_rope_cos_sin_cache must have shape [max_position>0, 64] "
            "on query.device"
        )
    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("", "0"):
        if int(ptr_seq_lens_kv.max()) > inv_rope_cos_sin_cache.shape[0]:
            raise ValueError("inv_rope_cos_sin_cache does not cover raw KV positions")

    expected_out_shape = (_HEADS // 8, total_q, 8, _HEAD_DIM)
    if out is None:
        out = torch.empty(expected_out_shape, device=device, dtype=torch.float8_e4m3fn)
    else:
        _require_cuda_tensor(out, "out", dtype=torch.float8_e4m3fn, ndim=4)
        if out.device != device or tuple(out.shape) != expected_out_shape:
            raise ValueError(f"out must have shape {expected_out_shape}")
    scale_buf_m = (total_q + 3) // 4 * 4
    expected_scale_shape = (_HEADS // 8, 8 * 4, scale_buf_m)
    if out_scale is None:
        out_scale = torch.empty(
            expected_scale_shape, device=device, dtype=torch.float32
        )
    else:
        _require_cuda_tensor(out_scale, "out_scale", dtype=torch.float32, ndim=3)
        if out_scale.device != device or tuple(out_scale.shape) != expected_scale_shape:
            raise ValueError(f"out_scale must have shape {expected_scale_shape}")

    expected_lse_shape = (total_q, _HEADS)
    stores_lse = lse is not None
    if lse is None:
        lse_kernel = torch.empty(expected_lse_shape, device=device, dtype=torch.float32)
    else:
        _require_cuda_tensor(lse, "lse", dtype=torch.float32, ndim=2)
        if lse.device != device or tuple(lse.shape) != expected_lse_shape:
            raise ValueError(f"lse must have shape {expected_lse_shape}")
        lse_kernel = lse

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    compiled = _compiled_dsv4_for_launch(
        device_index=device_index,
        enable_skip_correction=enable_skip_correction,
        fuses_inv_rope_fp8_quant=True,
        stores_lse=stores_lse,
    )
    q_kernel = query.permute(1, 2, 0)
    compressed_kernel = compressed_kv_pool.view(-1, 1, _HEAD_DIM).permute(1, 2, 0)
    swa_kernel = sliding_window_kv_pool.view(-1, 1, _HEAD_DIM).permute(1, 2, 0)
    raw_tma_descriptor_pair = _raw_gather4_descriptor_pair(
        sliding_window_kv_pool, compressed_kv_pool
    )
    compiled(
        q_kernel,
        q_kernel,
        compressed_kernel,
        compressed_kernel,
        ptr_page_idx_kv.transpose(0, 1),
        out.view(-1),
        lse_kernel.transpose(0, 1),
        None,
        raw_tma_descriptor_pair,
        1,
        ptr_seq_lens_kv,
        ptr_sparse_mla_topk_lens,
        ptr_cum_seq_lens_q,
        None,
        float(bmm1_scale),
        float(bmm2_scale),
        swa_kernel,
        max_seq_len_q,
        float(skip_corr_threshold),
        inv_rope_cos_sin_cache.view(-1),
        out_scale.view(-1),
    )
    return out, out_scale


# CSA and HCA differ only in runtime routing/compression metadata, not in the
# compiled Contract-F kernel. Keep explicit public spellings for callers while
# retaining one implementation and one JIT cache.
prims_ts_dsv4_csa_rope_quant = prims_ts_dsv4_sparse_mla_rope_quant
prims_ts_dsv4_hca_rope_quant = prims_ts_dsv4_sparse_mla_rope_quant


__all__ = [
    "prims_ts_dsv4_csa",
    "prims_ts_dsv4_sparse_mla_rope_quant",
    "prims_ts_dsv4_csa_rope_quant",
    "prims_ts_dsv4_hca_rope_quant",
]
