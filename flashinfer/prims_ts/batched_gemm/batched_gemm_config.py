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

"""Configuration for the BatchedGemm TS example.

All kernel-wide constants, feature-flag enums, and per-variant knobs.
Only this file may use `from __future__ import annotations`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, field, replace, asdict
from functools import cached_property
from cutlass.experimental import primitives as prims

from flashinfer.tllm_enums import WeightLayout

from .epilogue_schedule import normalize_epilogue_steps, resolve_preload_epilogue


# `SmemSfLdgstsResource._tail_cp_async_wait_group` lowers the runtime
# `cp.async.wait_group` drain for short K depths by hand-rolling one branch
# per static `cp_async_wait_group_depth` value (depth 0/1/2).  Supporting
# deeper prefetch windows would require either an exponentially larger
# switch or a generic runtime helper, so cap the SFB LDGSTS
# producer-commit prefetch window at this depth and reject anything
# larger at resource construction time.
MAX_PRODUCER_COMMIT_PREFETCH_DEPTH: int = 3


from .config_types import (
    BatchMode,
    RouteImpl,
    TileScheduler,
    CtaRasterOrder,
    ActKind,
    BiasType,
    DType,
    SfLayout,
    SfSmemToTmemCopy,
)


# Hardware geometry shared by config derivation and epilogue resources.
WARP_SIZE = 32
WARPS_PER_WARPGROUP = 4
TMA_MAX_SWIZZLE_BYTES = 128
TMA_SWIZZLE_ATOM_BYTES = 16
MAX_NON_SWAP_TMEM_FRAGMENTS = 2
MAX_NON_SWAP_GATED_OUTPUT_COLS_PER_CALL = 32
MAX_NON_SWAP_NVFP4_GATED_OUTPUT_COLS_PER_CALL = 64

# shuffleMatrix transposes an 8-row group against the remaining rows in each
# block.  This formula is the source of truth for both host preprocessing and
# the batch-M epilogue; keeping literal lookup tables in both places invites
# drift as new valid epilogue shapes are added.
SHUFFLED_MATRIX_ROW_GROUP = 8


def shuffled_matrix_block_size(epilogue_tile_m: int) -> int:
    """Return TRTLLM-gen's shuffle block size for an epilogue M tile."""
    return 32 if epilogue_tile_m % 128 == 0 else 16


def shuffled_matrix_row_index(epilogue_tile_m: int, logical_row: int) -> int:
    """Map one logical row to its physical row in shuffleMatrix storage."""
    block_size = shuffled_matrix_block_size(epilogue_tile_m)
    block_base = logical_row // block_size * block_size
    row_in_block = logical_row % block_size
    num_row_groups = block_size // SHUFFLED_MATRIX_ROW_GROUP
    return (
        block_base
        + (row_in_block % num_row_groups) * SHUFFLED_MATRIX_ROW_GROUP
        + row_in_block // num_row_groups
    )


def shuffled_matrix_row_indices(epilogue_tile_m: int) -> tuple[int, ...]:
    """Return TRTLLM-gen's source-row to shuffled-row map."""
    block_size = shuffled_matrix_block_size(epilogue_tile_m)
    return tuple(
        shuffled_matrix_row_index(epilogue_tile_m, row) for row in range(block_size)
    )


MX_MMA_FORMAT_E4M3 = 0
MX_MMA_FORMAT_E2M1 = 5

# A 32-bit TMEM SF word packs four 1-byte scale factors.  MMA selects one byte
# from the word with the 2-bit SF id, so one SF K-group spans four SF vectors.
TMEM_SF_PACK_SIZE_BYTES = 4

# S2T_32x128b_WARPX4 UTCCP copies 128 bits at a time.  Each SF TMEM column is
# one 32-bit packed-SF word, so one copy covers 128 / (8 * 4) = 4 columns.
TMEM_SF_UTCCP_COLS_PER_COPY = 4


DTYPE_BITS = {
    int(DType.FP32): 32,
    int(DType.FP16): 16,
    int(DType.BF16): 16,
    int(DType.E2M1): 4,
    int(DType.MXE2M1): 4,
    int(DType.MXE4M3): 8,
    int(DType.E4M3): 8,
}


def dtype_bits(dtype_kind: object) -> int:
    """Return the logical bit width for a concrete DType value."""
    kind = int(dtype_kind)
    if kind not in DTYPE_BITS:
        raise ValueError(f"Unsupported dtype kind: {kind}")
    return DTYPE_BITS[kind]


def tmem_d_cols_per_stage(tile_n: int) -> int:
    """Return accumulator TMEM columns consumed by one D stage."""
    return tile_n


def tmem_sf_vec4_groups_per_mma(mma_k: int, sf_vec_size: int) -> int:
    """Return how many packed vec4 SF words share one MMA scale base."""
    if mma_k % sf_vec_size != 0:
        raise ValueError(
            f"mma_k must be divisible by sf_vec_size={sf_vec_size}, got {mma_k}"
        )
    scale_vec_size = mma_k // sf_vec_size
    return max(1, scale_vec_size // TMEM_SF_PACK_SIZE_BYTES)


def tmem_sf_k_group_col_stride(
    tile_mn: int, mma_k: int, vec4_groups_per_mma: int
) -> int:
    """Return the effective TMEM column stride for one vec4 SF K-group."""
    col_stride = 2 * ((tile_mn + 63) // 64)
    if vec4_groups_per_mma > 1 or mma_k == 96:
        col_stride = max(4, col_stride)
    return col_stride


def tmem_sf_col_stride(tile_mn: int, mma_k: int, sf_vec_size: int) -> int:
    """Return the TMEM column stride between consecutive MMA scale bases."""
    vec4_groups_per_mma = tmem_sf_vec4_groups_per_mma(mma_k, sf_vec_size)
    return (
        tmem_sf_k_group_col_stride(tile_mn, mma_k, vec4_groups_per_mma)
        * vec4_groups_per_mma
    )


def tmem_sf_k_groups(tile_k: int, sf_vec_size: int) -> int:
    """Return the number of physical vec4 SF K-groups in one K tile."""
    if tile_k % sf_vec_size != 0:
        raise ValueError(
            f"tile_k must be divisible by sf_vec_size={sf_vec_size}, got {tile_k}"
        )
    sf_group_k = TMEM_SF_PACK_SIZE_BYTES * sf_vec_size
    return (tile_k + sf_group_k - 1) // sf_group_k


def tmem_sf_cols(tile_mn: int, tile_k: int, sf_vec_size: int, mma_k: int) -> int:
    """Return physical TMEM columns reserved for one SF operand stage."""
    k_groups = tmem_sf_k_groups(tile_k, sf_vec_size)
    vec4_groups_per_mma = tmem_sf_vec4_groups_per_mma(mma_k, sf_vec_size)
    mma_scale_groups = (k_groups + vec4_groups_per_mma - 1) // vec4_groups_per_mma
    return mma_scale_groups * tmem_sf_col_stride(tile_mn, mma_k, sf_vec_size)


def sfb_r128c4_utccp_offsets(cfg: BatchedGemmConfig, copy_idx: int) -> tuple[int, int]:
    """Return the SMEM and TMEM offsets for one R128c4 SFB UTCCP copy.

    R128c4 SMEM stores complete 128-row blocks in N-major order, while the
    TMEM scale-factor layout interleaves those blocks within each MMA K group.
    The returned pair is ``(SMEM descriptor increment, TMEM column delta)``.
    """
    num_utccp_k = cfg.tile_k // cfg.input_sf_block_size_b // TMEM_SF_PACK_SIZE_BYTES
    i_mn, utccp_idx_k = divmod(copy_idx, num_utccp_k)
    vec4_groups_per_mma = cfg.tmem_sf_vec4_groups_per_mma("b")
    ki, vi = divmod(utccp_idx_k, vec4_groups_per_mma)
    inner_group_stride = 8 if cfg.tile_n > 128 else 4
    tmem_delta = (
        ki * cfg.tmem_sf_col_stride("b")
        + i_mn * TMEM_SF_UTCCP_COLS_PER_COPY
        + vi * inner_group_stride
    )
    return 32 * copy_idx, tmem_delta


def uniform_pipeline_stage_overrides(
    load_stages: int,
    *,
    sf_stages: int | None = None,
    tmem_acc_stages: int | None = None,
) -> dict[str, int]:
    """Build explicit stage overrides for generated variants with one stage count."""
    sf_stages = load_stages if sf_stages is None else sf_stages
    result = {
        "num_stages_a": int(load_stages),
        "num_stages_b": int(load_stages),
        "num_stages_smem_sfa": int(sf_stages),
        "num_stages_smem_sfb": int(sf_stages),
        "num_stages_tmem_sfa": int(sf_stages),
        "num_stages_tmem_sfb": int(sf_stages),
    }
    if tmem_acc_stages is not None:
        result["num_stages_tmem_acc"] = int(tmem_acc_stages)
    return result


@dataclass(frozen=True)
class WarpLayout:
    """Resolved warp assignments; never accepted as tuning inputs."""

    cast_a_warp_idx_first: int = 0
    copy_sfa_warp_idx: int = 12
    copy_sfb_warp_idx: int = 4
    epilogue_warp_idx: int = 0
    load_a_warp_idx: int = 10
    load_b_warp_idx: int = 8
    load_sfa_warp_idx: int = 11
    load_sfb_warp_idx: int = 9
    mma_warp_idx: int = 13
    num_cast_a_warps: int = 0
    num_copy_sfa_warps: int = 1
    num_copy_sfb_warps: int = 4
    num_epilogue_warps: int = 4
    num_load_a_warps: int = 1
    num_load_b_warps: int = 1
    num_load_sfa_warps: int = 1
    num_load_sfb_warps: int = 1
    num_mma_warps: int = 1
    num_padding_warps: int = 2
    num_workid_warps: int = 1
    padding_warp_idx: int = 14
    workid_warp_idx: int = 14
    threads_per_cta: int = 512
    load_sfab_warp_idx: int = 12
    num_load_sfab_warps: int = 0
    num_sync_warps: int = 0
    sync_warp_idx: int = 10
    epilogue_regs: int = 160
    load_a_regs: int = 48
    load_b_regs: int = 48
    load_sfa_regs: int = 48
    load_sfb_regs: int = 48
    copy_sfa_regs: int = 48
    copy_sfb_regs: int = 48


@dataclass(kw_only=True, frozen=True)
class BatchedGemmConfig:
    """Immutable tuning requests and operation semantics for batched GEMM.

    Loader counts use zero for automatic selection. Explicit counts are
    honored or rejected. ``warp_layout`` holds the resolved task assignments
    and register budgets; resolving it never changes these requests.
    ``validate_config`` checks the supported combinations without mutation.

    **Meaning of "dense" in tuning options:** ``dense_*`` names the ordinary
    GEMM fast path. Its shared TMA pipeline operates on regular matrix tiles
    without MoE activation or scale-factor routing. The implementation targets
    Rubin NVFP4, non-swapped ``BATCH_M``, tile N=K=256, R128c4 scale factors,
    and physical 2xY CGAs, with tile256 and tile512 variants.

    Routed tile512 also reuses the M-fragment MMA and plain-output epilogue.
    Its optional split weight pipeline fuses B/SFB while retaining separate
    routed A/SFA loaders; see :attr:`uses_routed_weight_pipeline`.

    This is an optimization-family name, not a claim that other GEMM kernels
    use sparse matrices, nor a general promise of support for every epilogue.
    Each option has its own constraints, enforced by ``validate_config``.
    In particular, ``dense_`` does not itself mean "one expert":
    :attr:`dense_single_expert` separately requests that specialization and
    requires ``num_experts=top_k=1`` at launch. Individual variants, such as
    the eight-warp epilogue, can require that option explicitly.
    """

    # --- Architecture / shape ---
    sm_arch: int = 100
    """Target streaming-multiprocessor architecture as ``major * 10 + minor``.

    ``100`` and ``103`` select the Blackwell limits and descriptor formats.
    ``107`` selects Rubin-only features: 2x-MMAK FP8/MX, NVFP4 ``mma_k=128``,
    the extended 15-bit SMEM-address descriptor, the 327 KiB dynamic-SMEM
    budget, and the 576-column exclusive TMEM allocation. Runtime helpers fill
    this from the active CUDA device unless the caller supplies it explicitly.
    """

    umma_cta_count: int = 1
    """Number of CTAs in one logical UMMA instruction atom.

    ``1`` selects CTA_1; ``2`` selects a 2-CTA UMMA atom that multicasts the
    shared operand over cluster-scoped mbarriers. Drives :attr:`uses_two_cta_mma`,
    :attr:`cta_group`, and per-CTA B sizing (:attr:`split_b_across_ctas`).
    Physical CGA dimensions are configured separately.
    """

    cluster_dim_x: int = 0
    """Preferred physical CGA X dimension.

    ``0`` preserves the legacy behavior and uses :attr:`umma_cta_count`. Wider-CGA
    kernels keep ``umma_cta_count=2`` as the UMMA instruction atom while setting
    this field to ``4``. This separation prevents physical CGA size from
    changing atom-local byte counts, descriptors, or ``CTA_2`` instructions.
    """

    cluster_dim_y: int = 1
    """Preferred physical CGA Y dimension (``1``, ``2``, or ``4``)."""

    fallback_cluster_dim_x: int = 0
    """Fallback physical CGA X dimension; ``0`` disables flexible launch.

    Preferred 2x1 must use a fixed launch and rejects an enabled fallback.
    """

    fallback_cluster_dim_y: int = 1
    """Fallback physical CGA Y dimension when flexible launch is enabled."""

    epi_tile_m: int = 128
    """Epilogue tile M (rows processed per epilogue stage), in elements.

    Typically equals :attr:`tile_m` (``128``). Sets the C SMEM staging size
    (:attr:`num_bytes_c_per_stage`) and the epilogue reference block shuffle.
    Batch-M requires this to equal :attr:`tile_m`; Batch-N split epilogues may
    use a smaller value.
    """

    epi_tile_n: int = 8
    """Epilogue TMA-store tile N, in elements.

    The TMEM-to-register fragment is the largest supported divisor of this
    tile that does not exceed :attr:`tmem_ldst_max_num_regs`.
    """

    tmem_ldst_max_num_regs: int = 32
    """Maximum register count for one TMEM load; either ``32`` or ``64``.

    Matches TRTLLM-gen's ``tmemLdstMaxNumRegs`` control. Both the transposed
    ``16x256b`` and non-transposed ``32x32b`` epilogues split a logical
    fragment into the largest legal loads that do not exceed this cap. The
    DeepSeek FP8 epilogues are exempt and always issue one full-width load.
    """

    mma_k: int = 64
    """MMA instruction K depth, in elements; one of ``{16, 32, 64, 128}``.

    Must match the MMA dtype (:func:`validate_config`): ``16`` for BF16/CastA;
    ``32`` for MX and FP8 on Blackwell; ``32`` or ``64`` for MX and FP8 on
    Rubin; ``64`` for NVFP4 on Blackwell; and ``64`` or ``128`` for NVFP4 on
    Rubin. :attr:`tile_k` must be a multiple of it.
    """

    mma_m: int = 128
    """MMA instruction M dimension, in elements (passed to ``Tcgen05InstrDesc``).

    Generated TS kernels use ``64`` for DeepSeek FP8 split-epilogue shapes,
    ``128`` for 1-CTA shapes, and ``256`` for 2-CTA high-throughput shapes.
    ``validate_config`` range-checks this broad generated-kernel domain; exact
    tcgen05 descriptor legality is still enforced by ``Tcgen05InstrDesc.build``.
    """

    mma_n: int = 8
    """MMA instruction N dimension, in elements (passed to ``Tcgen05InstrDesc``).

    Generated TS kernels use ``{8, 16, 32, 64, 128, 256}``.
    ``validate_config`` range-checks this broad generated-kernel domain; exact
    tcgen05 descriptor legality is still enforced by ``Tcgen05InstrDesc.build``.
    """

    tile_k: int = 512
    """GEMM tile K depth (reduction length per K-tile), in elements.

    Must be a multiple of :attr:`mma_k` for the active dtype
    (:func:`validate_config`).

    .. note::
        Routed-SF TMA gather4 needs ``sf_k >= 32`` for the routed operand, where
        ``sf_k = tile_k // input_sf_block_size_{a,b}``. The producer-commit
        prefetch guard also requires ``3 * tile_k <= K``. CastA currently
        requires ``tile_k = 256`` because the cast stage produces a fixed
        128x256 BF16 tile in TMEM.
    """

    tile_n: int = 8
    """GEMM tile N dimension (token/output columns per CTA), in elements.

    One of ``{8, 16, 32, 64, 128, 256}`` (:func:`validate_config`). Drives the
    SFB layout (:attr:`uses_sfb_8x4_load`), the tile256 overlap path
    (:attr:`use_tile256_tmem_overlap`), and the epilogue warp count.
    """

    # --- Semantic dtypes ---
    dtype_a: int = int(DType.E2M1)
    """Operand A element type, as a :class:`DType` integer value.

    With :attr:`dtype_b` it selects the MMA kind: NVFP4 (``E2M1``x``E2M1``), MX
    (``MXE2M1``/``MXE4M3`` on both sides), FP8 (``E4M3``x``E4M3``), BF16
    (``BF16``x``BF16``), or CastA (``MXE2M1`` A x ``BF16`` B). See
    :attr:`dtype_a_smem_bits` / :attr:`dtype_a_tma_bits` for the staged widths.
    """

    dtype_b: int = int(DType.E2M1)
    """Operand B element type, as a :class:`DType` integer value.

    Paired with :attr:`dtype_a` to select the MMA kind (see :attr:`dtype_a`).
    """

    dtype_c: int = int(DType.BF16)
    """Output (epilogue) element type, as a :class:`DType` integer value.

    ``BF16``/``FP16`` store plain; ``E4M3`` is plain FP8 output only for FP8 MMA
    (:attr:`uses_fp8_output`); ``E2M1``, ``MXE2M1``, and ``MXE4M3`` enable output
    quantization with block scales (:attr:`uses_fp4_output_quant` /
    :attr:`uses_mx_output_quant`). The accumulator stays FP32 -- this is the
    final cast. SwapAB output quantization supports gated epilogues and
    non-gated ``RELU2``/``SILU`` epilogues when ``epi_tile_m == tile_m``,
    ``epi_tile_n * num_epilogue_warpgroups`` divides ``tile_n``, and the output
    is ``E2M1`` or ``MXE4M3`` (:func:`validate_config`).
    """

    weight_layout: int = int(WeightLayout.MajorK)
    """Physical layout of the per-expert weight operand.

    ``MajorK`` keeps the existing storage, logically ``[expert, Mn, K]``.
    ``BlockMajorK`` stores the weight matrix as
    ``[expert, K_bytes / block_bytes, Mn, block_bytes]`` and is exposed to TMA
    as ``(blockK_elems, Mn, K / blockK_elems, expert)``. ``block_bytes`` is
    128, or 64 for padded formats (MXFP4 staged as 8-bit), whose TMA boxes are
    exactly 64B wide (:attr:`block_major_k_bytes`). The layout always applies
    to the weight operand: A when :attr:`is_swap_ab`, B otherwise.
    """

    gated_weight_group_size: int = 1
    """Number of consecutive up/gate weight rows in each gated storage group.

    The default stores U0,G0,U1,G1. The non-swap NVFP4/SwiGLU epilogue may
    use 64 to store U0..U63,G0..G63, matching its two 64-register fragments.
    Under a low initial register limit, each fragment uses two x32 loads.
    The host runner permutes weights and their scale factors together.
    Optional shuffleMatrix permutation follows this grouping; MajorK or
    BlockMajorK packing then defines the physical weight storage.
    """

    # --- Pipeline stages ---
    num_stages_a: int = 5
    """Depth of the operand-A SMEM load pipeline ring (TMA prefetch).

    Sizes ``num_bytes_a_per_stage * num_stages_a`` of SMEM.
    """

    num_stages_b: int = 5
    """Depth of the operand-B SMEM load pipeline ring.

    Two-CTA UMMA kernels stage one N slice in each CTA; see
    :attr:`split_b_across_ctas` and :attr:`num_bytes_b_smem_per_stage`.
    """

    num_stages_c_smem: int = 2
    """Depth of the epilogue C SMEM staging pipeline ring.

    Used only with :attr:`use_tma_store`; see :attr:`num_bytes_c_smem_scratch`.
    """

    num_stages_smem_sfa: int = 5
    """Depth of the scale-factor-A SMEM pipeline ring (block-scaled kernels only).

    Distinct from the TMEM-side depth :attr:`num_stages_tmem_sfa`.
    """

    num_stages_smem_sfb: int = 5
    """Depth of the scale-factor-B SMEM pipeline ring (block-scaled kernels only)."""

    num_stages_tmem_acc: int = 2
    """Depth of the accumulator (C) TMEM pipeline ring.

    Limited by the architecture's TMEM capacity. Must be ``1`` when
    :attr:`use_max_tmem_overlap` is set.
    """

    num_stages_tmem_sfa: int = 5
    """Depth of the scale-factor-A TMEM pipeline ring after the SMEM->TMEM copy.

    Allocated only when SF copies use separate tasks.
    """

    num_stages_tmem_sfb: int = 5
    """Depth of the scale-factor-B TMEM pipeline ring (unfused copy path only)."""

    num_stages_workid: int = 3
    """Depth of the persistent CLC work-id fetch pipeline ring.

    Used only when :attr:`is_persistent`. Queue depth does not serialize the
    operand loaders with the epilogue of the preceding work tile.
    """

    # Zero requests automatic selection; these fields never hold resolved counts.
    num_load_a_warps: int = 0
    num_load_b_warps: int = 0
    num_load_sfa_warps: int = 0
    num_load_sfb_warps: int = 0

    # --- Register budgets (per-thread setmaxnreg) ---
    cast_a_regs: int = 160
    """Per-thread register budget for the CastA warp group."""

    copy_sf_regs: int = 48
    """Per-thread register budget for the CopySf tasks.

    The compact R8c4-STTM SFB path overrides this to ``72``
    (:attr:`copy_sfb_task_regs`).
    """

    epilogue_regs: int = 160
    """Per-thread register budget for the epilogue warp group."""

    load_a_regs: int = 0
    """Operand A loader budget for any transport; 0 selects the schedule default."""

    load_b_regs: int = 0
    """Operand B loader budget for any transport; 0 selects the schedule default."""

    load_regs: int = 48
    """Default register budget for the A/B load tasks when no per-task
    override (:attr:`load_a_regs` / :attr:`load_b_regs`) is set."""

    load_sf_regs: int = 48
    """Default register budget for the scale-factor load tasks when no per-task
    override is set."""

    load_sfa_regs: int = 0
    """SFA loader register budget; ``0`` falls back to
    :attr:`load_sf_regs`."""

    load_sfb_regs: int = 0
    """SFB loader register budget; ``0`` falls back to
    :attr:`load_sf_regs`."""

    mma_regs: int = 48
    """Per-thread register budget for the MMA task."""

    padding_regs: int = 48
    """Per-thread register budget for the padding warps."""

    workid_regs: int = 48
    """Per-thread register budget for the CLC work-id task."""

    # --- Feature flags (int 0/1 or enum-as-int; never bool/str) ---
    act_kind: int = int(ActKind.NONE)
    """Epilogue activation, as an :class:`ActKind` integer value.

    Allowed: ``NONE``, ``SWIGLU``, ``GEGLU``, ``RELU2``, ``SILU``, ``SITU``.
    ``RELU2`` and ``SILU`` are elementwise. Gated kinds
    (:attr:`has_gated_epilogue`) consume ``(gate, up)`` pairs and halve the
    paired output dimension. In ``BATCH_N`` / swap-A/B mode the paired dimension
    is output M (hidden rows); in ``BATCH_M`` / non-swap mode it is output N
    (columns). Both layouts have epilogue handling.
    """

    batch_mode: int = int(BatchMode.BATCH_N)
    """MoE expert-batching axis, as a :class:`BatchMode` integer value.

    ``BATCH_N`` batches routed activation/token tiles along N, selects the
    swapped-A/B path (A=weights, B=activations), and stores C in the M-major
    (column-major) layout. ``BATCH_M`` batches along M, selects the non-swapped
    path, and stores C row-major. Also selects the expert-coordinate lookup.
    """

    per_token_sf_dtype: int = int(DType.FP32)
    """Element type of per-token FP8 scale factors, as a :class:`DType` value.

    One of ``FP32``/``FP16``/``BF16`` (:func:`validate_config`). Relevant only
    when :attr:`use_per_token_sf_b` is set.
    """

    route_act: int = int(RouteImpl.NONE)
    """MoE activation gather transport, as a :class:`RouteImpl` value.

    ``NONE`` (no gather), ``TMA`` (gather4), or ``LDGSTS`` (cp.async).
    See :attr:`has_gather`,
    :attr:`has_tma_route`.
    """

    route_sfs_act: int = int(RouteImpl.AUTO)
    """Requested activation SF transport, as a :class:`RouteImpl` value.

    AUTO selects LDGSTS for TMA-routed block-scaled activations and NONE
    otherwise. Explicit NONE uses pre-expanded SF. TMA requires at least 32
    SF elements per K tile. :attr:`sf_route` is the effective transport.
    """

    sf_layout_a: int = int(SfLayout.R128c4)
    """SMEM layout of the A scale factors, as a :class:`SfLayout` value.

    Default ``R128c4``. ``R8c4`` is rejected for input SFA; CastA requires
    ``R128c4``. ``LINEAR`` requires routed SFA (:func:`validate_config`).
    The effective layout is :attr:`smem_sfa_layout` (routing may override it).
    """

    sf_layout_b: int = int(SfLayout.R8c4)
    """SMEM layout of the B scale factors, as a :class:`SfLayout` value.

    Default ``R8c4``, valid only for non-gather B with ``tile_n <= 64`` or routed
    low-N SFB. ``LINEAR`` requires routed SFB (:func:`validate_config`).
    Effective layout is :attr:`smem_sfb_layout`.
    """

    sf_layout_c: int = int(SfLayout.R8c4)
    """SMEM layout of the output scale factors, as a :class:`SfLayout` value.

    Used only when the output is quantized (:attr:`has_output_block_scaling`), where it
    must be ``R8c4`` or ``R128c4`` (:func:`validate_config`).
    """

    tile_scheduler: int = int(TileScheduler.STATIC)
    """CTA scheduling mode, as a :class:`TileScheduler` value.

    ``STATIC`` (fixed hardware grid) or ``PERSISTENT`` (dynamic CLC work queue).
    See :attr:`is_persistent`.
    """

    cta_raster_order: int = int(CtaRasterOrder.ALONG_M)
    """Persistent CTA rasterization order, as a :class:`CtaRasterOrder` value.

    ``ALONG_M`` visits M tiles first; ``ALONG_N`` visits N tiles first. TRTLLM-gen
    exposes these as ``ctaSwizzleType=rasterizeAlongM/rasterizeAlongN``.
    :attr:`cta_raster_group_m` optionally interleaves M clusters within N order.
    """

    cta_raster_group_m: int = 1
    """Number of neighboring preferred M clusters to interleave in N order.

    One preserves ordinary rasterization. Larger groups visit the same N
    cluster for each M cluster before advancing N, improving weight locality
    when neighboring M clusters belong to the same expert. An incomplete final
    M group retains ordinary N order. Requires the dense fused pipeline and
    persistent N rasterization; physical cluster offsets remain unchanged.
    """

    transpose_mma_output: int = 1
    """Whether MMA output is transposed before the epilogue. ``0``/``1``.

    The generated ABI exposes this separately from :attr:`batch_mode`. The
    current kernel supports only ``1`` for the ``BATCH_N`` swapped-A/B path and
    ``0`` for the ``BATCH_M`` non-swapped path. FP8 per-token SFB requires ``1``.
    """

    use_shuffled_matrix: int = 0
    """Use TRTLLM-gen ``shuffleMatrix`` storage for the weight rows. ``0``/``1``.

    ``BATCH_N`` always uses this storage because its transposed-output epilogue
    requires it.  Setting this option to ``1`` additionally enables the same
    weight and weight-scale-factor storage for ``BATCH_M``.  Its epilogue then
    reads the shuffled accumulator columns in logical order, allowing one
    preprocessed weight representation to be shared by both batch modes.

    ``0`` preserves the historical unshuffled ``BATCH_M`` path.
    """

    use_clc_fast_drain: int = 0
    """Enable batched CLC cancellation for persistent early-exit overlaunch.

    When a CUDA graph launches more token CTAs than the active batch needs,
    this lets the work queue cancel queued tail CTAs in batches. Cancellation
    drops every cluster that has not launched yet, so it runs only when an
    early-exit skip is a suffix of the CLC raster order.
    """

    use_early_exit: int = 0
    """Let over-launched token CTAs skip work past the active batch. ``0``/``1``.

    Needs a runtime ``num_non_exiting_ctas`` value; pairs with :attr:`use_pdl` and
    the required PDL wait before reading the active CTA count.
    """

    use_full_tmem_dealloc_barrier: int = 0
    """Full-CTA (vs epilogue-warpgroup-only) barrier before TMEM dealloc. ``0``/``1``.

    ``1`` syncs all warps before ``tcgen05.dealloc``; ``0`` syncs only the
    epilogue warpgroup (lower overhead).
    """

    use_global_scales: int = 1
    """Enable per-tensor FP32 global scales in the epilogue. ``0``/``1``.

    Affects only NVFP4 and plain-FP8 paths (ignored for MX, which carries block
    scales). See :attr:`uses_global_scales` for the effective value.
    """

    # --- Dense GEMM tuning; "dense" is defined in the class docstring above. ---
    dense_rolled_mma: int = 0
    """Acquire the accumulator once, then use one K loop including the first tile."""

    dense_async_sf: int = 0
    """Keep fused UTCCP and MMA in their ordered async stream, as DKG does."""

    dense_m_repeats: int = 1
    """Independent 128-row fragments per CTA (two for the dense tile512 path).

    Tile512 supports multiple experts with generic expert lookup and a problem
    N divisible by tile_n, preserving the weight scale-factor expert stride.
    Non-swap routed A (LDGSTS/TMA) and LDGSTS SFA retain separate loader tasks.
    """

    dense_b_reuse: int = 0
    """Retain B in the SM107 collector across the two M-fragment MMAs."""

    dense_local_barrier: int = 0
    """Use group-2 local completion addresses, matching DKG's TMA lowering.

    Requires CGA specialization. Native TMA-A routing also supports this
    for separate B/SFB pipelines; gather4 A keeps its leader mapping.
    """

    dense_dkg_store_order: int = 0
    """Wait for C-store ring reuse after commit, matching DKG's epilogue.

    Requires :attr:`dense_rolled_epilogue`; applies to its non-swapped,
    non-gated TMA-store path. Other epilogue paths retain their own waits.
    """

    dense_rolled_epilogue: int = 0
    """Keep the dense epilogue's subtile loop in generated code.

    Supports the non-gated dense path and non-swapped NVFP4/SwiGLU output
    on dense or native routed tile256/tile512. Requires TMA stores without
    a TMEM overlap window. Preferred/fallback CGAs retain separate bodies;
    only the subtile calls within each epilogue become a runtime loop.
    Disabled by default; :attr:`dense_preload_epilogue` takes precedence
    on its supported plain 16-bit output paths.
    """

    dense_preload_epilogue: int = 0
    """Retain packed 16-bit output in registers before releasing the accumulator.

    With four warps, tile512 stores its first N64 fragment, retains the other
    seven in 224 packed registers per thread, and releases TMEM before their
    stores. With eight warps, each group loads half of every N64 fragment;
    all eight fragments occupy 128 packed registers per thread. Set
    :attr:`dense_preload_store_prefix_fragments` to 2 to store two first and
    retain six in 96 packed registers. By default TMEM is released before the
    retained fragments' stores; :attr:`epilogue_tmem_release` can delay it.
    Both groups cooperate through one or two shared C
    scratch stages.
    Requires 248/256 registers for four warps or 144/160/168/176 for eight, N64
    fragments, and plain FP16/BF16 output. Optional per-expert global scaling
    is applied in FP32 before conversion and register packing.
    Four-warp fused MoE finalization instead stages the complete output tile
    in shared memory before releasing TMEM; see :attr:`uses_full_tile_finalize`.
    """

    dense_preload_store_prefix_fragments: int = 0
    """N64 fragments stored before TMEM release by the eight-warp preload.

    Only 0 and 2 are supported. Default 0 disables prefix stores: retain all
    eight fragments and store them after TMEM release. With 2, store the
    first two, retain six in 96 packed registers per thread, release TMEM,
    then store the remaining six. This reduces register pressure, especially
    with FP32 global output scaling; it does not require scaling to be enabled.
    A nonzero value requires dense_preload_epilogue=1 and
    dense_epilogue_warps=8, with its plain FP16/BF16 output restrictions.
    This option does not control the four-warp preload schedules.
    The preload, retained-vector size and store tasks share this compile-time
    count so every fragment is written exactly once. With fuse_moe_finalize,
    prefix fragments are staged in shared memory before TMEM release; bulk
    reduction waits until the remaining fragments complete each N256 row.
    """

    epilogue_steps: tuple = ()
    """Explicit compile-time EpilogueStep sequence; empty uses legacy presets.

    Counts describe ordered N64 fragments. The plan derives SMEM slots, bulk
    source-read waits and writer barriers. Supports plain-output preload and
    interleaved group1 NVFP4 FC1. FC1 counts are per four-warp group (four
    fragments with eight warps, eight with four); two fragments fill a C slot.
    Cannot be combined with legacy prefix/read/staging/release controls.
    """

    epilogue_tmem_load_width: int = 0
    """Preload read width: 0 resolves the existing mapping, or explicitly 32/64.

    Eight cooperative warps require 32, four-warp full-tile finalization
    requires 64, and plain four-warp preload retains its mixed-width schedule.
    See ``epilogue_schedule.py`` for supported combinations and named presets.
    """

    epilogue_tmem_load_ahead: int = -1
    """Pending preload reads: -1 preserves the existing schedule; 0/1 is explicit.

    Eight-warp prefix-zero preload supports serial or one-read-ahead loads.
    Other schedules currently require their original setting. The final
    register-heavy reads remain serial even with one-read-ahead enabled.
    """

    epilogue_smem_staging: int = 0
    """0: resolve existing schedule; 1: fragment ring; 2: complete output tile.

    Fragment-ring preload stages the prefix and retains the tail in registers.
    Full-tile staging requires four-warp fused finalization and two C stages.
    Both stage converted output. Ownership determines layout, fences and waits.
    """

    epilogue_tmem_release: int = 0
    """0/1: release after preload; 2: release after the output-store task.

    All accumulator reads finish before either release point. Setting 2 delays
    MMA reuse until output issue/source-read waits finish; it does not promise
    global completion of asynchronous stores. Tail draining remains required.
    """

    fuse_moe_finalize: int = 0
    """Scatter weighted BF16 FC2 rows directly into the token output.

    Requires native Rubin NVFP4 BatchM with four/eight-warp preload and global
    expert scales. The caller supplies original router coefficients, routing
    index maps and a zeroed token output. Router weights multiply FP32
    accumulators before conversion; bulk BF16 atomic reductions combine
    experts in unspecified order. Full
    N256 rows use padded shared-memory scratch; operand stage counts must
    leave room for it. Use the separate finalizer for deterministic FP32
    accumulation.
    The legacy four-warp profile uses two C scratch stages: stage both 128-row halves before
    releasing TMEM, then reduce both halves in one bulk group per thread.
    Explicit epilogue_steps can instead emit and reuse a single C slot.
    Eight warps retain the configurable register preload/prefix schedule.
    """

    dense_streaming_c: int = 0
    """Use an L2 evict-first hint for packed-epilogue output TMA stores.

    Requires dense_preload_epilogue. Default 0 preserves normal output cache
    retention; enable only after considering reuse by the following kernel.
    """

    dense_a_l2_promotion: int = 0
    """L2 promotion bytes for dense activation TMA descriptors: 0, 128, or 256.

    Applies to full and partitioned A descriptors; B and scale-factor maps
    keep their default policy. Independent of the A/SFA evict-last hint.
    Supports dense shared pipelines and routed tile512 TMA A. LDGSTS has no
    TensorMap promotion policy. Default 0 disables promotion.
    """

    dense_retain_a: int = 0
    """Prefer retaining dense activation operands and their scales in L2.

    Adds an L2 evict-last hint to A/SFA loads, leaving B/SFB unchanged.
    Supports dense TMA and routed tile512 TMA/LDGSTS. Default 0 keeps normal
    input caching; this does not retain activations in shared memory.
    """

    dense_epilogue_warps: int = 4
    """Four or eight warps for the dense/native-routed rolled epilogue.

    NVFP4/SwiGLU uses two four-warp groups to partition N, each with its own
    persistent C scratch ring and TMA store queue. Both groups release TMEM
    through the shared accumulator pipeline after completing their reads.
    For plain FP16/BF16 output without preloading, the independent groups have
    their own C scratch rings and store queues. This requires single-expert
    output, at most 128 registers, and complete C rings per M fragment.
    With dense_preload_epilogue, both groups share one scratch ring and store
    queue; multiple experts are supported. See its register requirements.
    All eight-warp paths require output without bias or per-token scaling.
    """

    dense_split_tma_pipeline: int = 0
    """Use independent A/SFA and B/SFB rings and loader warps.

    Each pair shares a stage cursor and completion barrier. Reuse waits cover
    only the CTAs sharing that pair, allowing weights and activations to
    advance independently. Operand and SF depths must match within each pair.
    Requires dense_fuse_tma_loads, persistent scheduling, and no work throttle.
    Routed tile512 fuses only B/SFB, keeping A/SFA loaders and their relay.
    Its weight cursor is independent of A, permitting five A/six B stages.
    """

    dense_fuse_tma_loads: int = 0
    """Issue dense A/B/SFA/SFB loads from one task with one shared cursor.

    dense_split_tma_pipeline instead creates one loader and cursor per pair.

    Requires :attr:`dense_partition_tma_loads` and either the non-routed
    shared pipeline or a routed weight pipeline or fused native routed loaders.
    """

    dense_specialize_tma_cga: int = 0
    """Compile preferred/fallback dense CGA bodies within one kernel.

    Supports the dense shared pipeline, routed tile512, or native-SFA TMA-A
    routing, with a 2x1x1 fallback. A cluster-wide runtime branch selects the
    body for the actual shape; other GEMM configurations remain excluded.
    """

    _active_cga_shape: tuple[int, int, int] | None = None

    dense_single_expert: int = 0
    """Specialize expert-map coordinates to zero for a validated dense launch.

    Requires :attr:`dense_fuse_tma_pipeline` and ``num_experts=top_k=1`` at launch.
    This assumption is explicit; the ``dense_`` naming prefix alone does not
    select it for other options.
    """

    dense_partition_tma_loads: int = 0
    """Distribute shared-pipeline A/SFA slices across Y and SFB slices across X.

    Requires the dense shared pipeline or routed tile512 weight pipeline,
    tile M=128/256 per CTA, preferred Y=1/2/4, a 2x1 fallback, and no TMA
    OOB optimization.
    Each issuer multicasts a disjoint slice. A complete stage in each receiving
    CTA is unchanged, so full-barrier bytes and empty-barrier consumer counts
    stay unchanged. Loader participation (and hence CLC arrivals) increases.
    With routed tile512, partitions TMA A gather4 rows across Y and SFB
    slices across X. LDGSTS A/SFA remain CTA-local. Flexible routed TMA A
    requires static CGA specialization so each body has a fixed slice size.
    """

    dense_fuse_tma_pipeline: int = 0
    """Share dense NVFP4 A/B/SFA/SFB pipeline barriers on Rubin. ``0``/``1``.

    "Dense" refers to the GEMM fast path defined in the class docstring.
    Supports non-routed batch-M GEMM with tile N=K=256, matched operand/SF
    stage counts, and physical 2xY CGAs, including the tile512 M extension.
    Fused loaders also support 4xY: every atom arms its local full barrier,
    while the first X atom supplies multicast B and partitioned SFB payloads.
    MMA waits and releases once per K tile.
    Multi-atom CGAs arm one local full barrier for the total operand bytes;
    single-atom CGAs retain four local arrivals. The shared empty barrier uses
    the union of operand release destinations for the active CGA. SFB is
    multicast across each row, including a single 2x1 atom.
    """

    fuse_sf_copy_to_mma: int = 0
    """Fuse the scale-factor SMEM-to-TMEM copy into the MMA task. ``0``/``1``.

    This matches TRT-LLM Gen's ``fuseUtccpWithUtcmma`` path and removes the
    otherwise separate CopySf tasks.  Unlike :attr:`use_max_tmem_overlap`, it
    does not require a tile-256 accumulator layout.
    """

    routed_native_sfa: int = 0
    """Stage routed SFA in row order with LDGSTS.128 and 128x128b UTCCP.

    Opt-in Rubin NVFP4 tile256/tile512 path; the MMA descriptor and TMEM
    scale stride must use the matching native layout for each M fragment.
    Supports fused LDGSTS A/SFA or separate TMA A and native SFA loaders.
    TMA-A with CLC also supports a 2x1 fallback; partitioned gather4
    requires CGA specialization. The static tile queue remains fixed-CGA.
    """

    routed_low_smem: int = 0
    """Use the measured tile256 payload layout with controls below byte1024.

    Restricted to eight operand stages, two accumulator stages and one
    BF16 scratch slot. Other configurations keep the standard allocator.
    """

    routed_sfa_first: int = 0
    """Place native routed SFA before the other SMEM payloads. ``0``/``1``.

    Preserves alignment, pipeline registration and relative order of the
    remaining allocations. This changes addresses, not SF layout or copy
    scheduling. Requires routed_native_sfa and excludes routed_low_smem,
    which selects a separate fixed tile256 layout.
    """

    static_persistent_clusters: int = 0
    """Resident cluster budget for STATIC_PERSISTENT; supplied by the host."""

    fused_a_single_commit: int = 0
    """Commit the shared LDGSTS A/SFA completion ring once per fused loader."""
    fuse_operand_sf_loads: int = 0
    """Fuse each operand load with its scale-factor load. ``0``/``1``.

    The clustered high-throughput LDGSTS path uses the same four warps for the
    routed activation and activation-SF copies, and the same warp for the
    weight and weight-SF TMA loads.  This matches the generated kernel's
    12-warp schedule and lets both LDGSTS streams share one delayed commit
    window.
    """

    use_max_tmem_overlap: int = 0
    """Enable the tile256 maximum-TMEM-overlap path. ``0``/``1``.

    Valid for block-scaled kernels and unscaled Batch-M BF16/plain-FP8 kernels
    with ``tile_n == 256``, ``umma_cta_count == 2``, persistent scheduling,
    and ``num_stages_tmem_acc == 1`` (:func:`validate_config`). See
    :attr:`use_tile256_tmem_overlap`.
    """

    weight_l2_evict_first: int = 0
    """Load the swap-AB weight operand and its scale factors with the
    evict-first L2 policy. ``0``/``1``.

    The runtime routing map selects evict-first only for experts with one
    token CTA. Experts with multiple token CTAs use evict-normal to preserve
    weight reuse. CTAs along M read different weight slices. Requires
    :attr:`is_swap_ab` and token-cluster width one (:func:`validate_config`).
    Weight and weight-scale loaders construct the policy once per work tile
    with PTX ``createpolicy``.
    """

    fused_finalize_top_k: int = 0
    """Top-k of a MoE finalize fused into this swap-AB BF16 epilogue, or ``0``
    for a plain C store.

    With ``k > 0`` every output column (permuted token row) whose token has
    exactly one expert in this launch is scaled by that expert's routing
    weight and stored straight into the token's row of the MoE output; the
    columns of tokens with several experts go to C unscaled, for the finalize
    kernel to reduce. The launch then also takes the MoE output, the
    expanded-to-permuted index map and the expert weights
    (:func:`validate_config`).
    """

    use_pdl: int = 0
    """Enable programmatic dependent launch (PDL). ``0``/``1``.

    Overlaps this kernel's prologue with the prior kernel; pairs with
    the required PDL wait before reading the active CTA count.
    """

    skip_early_exit_pdl: int = 0
    """Skip the kernel-entry PDL wait that early exit otherwise requires. ``0``/``1``.

    Requires :attr:`use_pdl` and :attr:`use_early_exit`. By default that
    combination issues ``griddepcontrol.wait`` at kernel entry so that
    ``num_non_exiting_ctas`` is visible. With ``1`` the entry wait is skipped
    and the PDL wait comes only from the ``PdlWaitBarrier`` resource on the
    GMEM A/B operand sourced from the predecessor, so this kernel's CTAs can
    become resident while the predecessor is still running. The count is then
    read without waiting for the predecessor, so it must have been produced by a
    kernel that completed before the predecessor passed its own grid wait and
    triggered ``launch_dependents`` (e.g. routing, with the predecessor waiting
    on it before triggering).
    """

    # Per-token scale factors. SF-A is consumed by the epilogue for plain FP8;
    # SF-B is consumed by the B/output-channel epilogue path for plain FP8 and
    # NVFP4/E2M1.
    use_per_token_sf_a: int = 0
    """Plain-FP8 per-token scale factors on A. ``0``/``1``.

    In swap-A/B MoE configs this is the A-side channel row scale.
    """

    use_per_token_sf_b: int = 0
    """Per-token scale factors on B. ``0``/``1``.

    Valid for plain FP8 or NVFP4 MMA. FP8 with :attr:`bias_type` = ``M`` is
    parsed but not implemented. Dtype is
    :attr:`per_token_sf_dtype`.
    """

    has_gemm1_alpha: int = 0
    """Load a per-expert float32 SwiGLU-OAI alpha pointer in the epilogue."""

    has_gemm1_beta: int = 0
    """Load a per-expert float32 SwiGLU-OAI beta pointer in the epilogue."""

    has_gemm1_clamp_limit: int = 0
    """Load a per-expert float32 SwiGLU-OAI clamp-limit pointer in the epilogue."""

    # DeepSeek FP8: per-K-tile FP32 dequant scales loaded into SMEM via
    # cp.async and applied as ffma2 in the epilogue, BEFORE the FP32 output cast.
    # Requires E4M3 MMA with tile_k=128 (the scale block, 128 or 32, divides
    # it) and is mutually exclusive with MX/NVFP4 block scaling and per-token SF.
    use_deepseek_fp8: int = 0
    deepseek_fp8_block_size: int = 128
    """Side of the square DeepSeek FP8 scale block, in elements. ``128`` or ``32``.

    One FP32 weight scale covers ``block`` rows x ``block`` K elements, and one
    activation scale covers ``block`` K elements of a token row. The K tile
    stays 128 wide (the FP8 TMA box) and holds ``128 / block`` sub-blocks: the
    MMA writes every sub-block to its own accumulator stage and the epilogue
    drains and scales each one. ``32`` requires the swapAB split-M epilogue
    (``mma_m=64``), which also keeps ``mma_k`` at 32.
    """
    load_sfab_regs: int = 48
    use_tma_oob_opt: int = 1
    """Use TMA out-of-bounds predication for partial tiles. ``0``/``1``.

    Pads descriptor coordinates so the hardware predicates OOB accesses, cutting
    epilogue predicate math. Applies to A/B loads and C stores; see
    :attr:`use_tma_oob_opt_a` / :attr:`use_tma_oob_opt_b`.
    """

    use_tma_store: int = 0
    """Epilogue store kind: ``0`` = direct STG, ``1`` = SMEM-staged TMA store.

    TMA store improves locality for gated / quantized outputs; it needs the C SMEM
    scratch (:attr:`num_bytes_c_smem_scratch`).
    """

    use_unroll_loop_2x_for_mma: int = 0
    """Unroll the MMA K loop by two stages. ``0``/``1``.

    The captured loop keeps a scalar logical step and asks the DSL compiler to
    unroll it twice, so each K tile retains its own pipeline-stage identity.
    Odd K-tile counts automatically use the scalar loop.
    """

    use_work_throttle: int = 1
    """Throttle CLC work-id fetch in persistent kernels. ``0``/``1``.

    Effective only when :attr:`is_persistent` (see
    :attr:`use_work_throttle_barrier`); paces clustered loads to avoid SMEM
    pipeline ring overrun.
    """

    record_runtime_cga_shapes: int = 0
    """Record preferred/fallback physical CGA counts for GPU regressions.

    This test hook is compile-time disabled by default. When enabled, one
    elected thread per physical CGA atomically increments counters stored next
    to ``num_non_exiting_ctas``.
    """

    # --- Gather (LDGSTS route) ---

    sync_regs: int = 48
    """Per-thread register budget for the sync warp."""

    # --- Host-derived schedule constants ---
    # Set by ``validate_config(..., problem_mnk=...)`` on the Python launch
    # path.  The JIT kernel receives ``cfg`` as a constexpr, so resources and
    # task factories can consume this as a static delayed-commit prefetch cap
    # instead of guarding head/tail entries against runtime K.

    # --- Activation and epilogue knobs ---
    bias_type: int = int(BiasType.NONE)
    """Epilogue bias, as a :class:`BiasType` value.

    ``NONE`` or ``M`` (per-channel float32 ``[experts, M]`` bias added before
    activation). Plain FP8 per-token SF-B with ``M`` is parsed but not
    implemented; NVFP4 per-token SF-B with ``M`` is supported. See
    :attr:`has_bias_m`.
    """

    # --- Derived sizes (properties) ---
    _problem_k: int | None = field(default=None, init=False, repr=False)

    @property
    def ldgsts_sfb_producer_commit_prefetch_depth(self) -> int:
        depth = _ldgsts_sfb_base_producer_commit_prefetch_depth(self)
        if self._problem_k is not None:
            depth = min(depth, (self._problem_k + self.tile_k - 1) // self.tile_k)
        return depth

    @cached_property
    def warp_layout(self) -> WarpLayout:
        return _resolve_warp_layout(self)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epilogue_steps", normalize_epilogue_steps(self.epilogue_steps)
        )
        # Normalize effective format flags once. Invalid values remain visible
        # to validation rather than being silently converted to booleans.
        if self.use_shuffled_matrix in (0, 1) and self.is_swap_ab:
            object.__setattr__(self, "use_shuffled_matrix", 1)
        if self.use_global_scales in (0, 1) and not (
            self.is_nvfp4_mma or self.is_fp8_mma
        ):
            object.__setattr__(self, "use_global_scales", 0)

    def compile_signature(self) -> dict:
        """Effective constexpr inputs, independent of defaults and resolution history."""
        result = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        plan = self.preload_epilogue_plan
        if self.epilogue_steps:
            result["epilogue_steps"] = plan.signature()
        elif plan is not None:
            # AUTO and an equivalent explicit preset name the same kernel.
            result.update(
                epilogue_tmem_load_width=plan.load_width,
                epilogue_tmem_load_ahead=plan.load_ahead,
                epilogue_smem_staging=int(plan.staging),
                epilogue_tmem_release=int(plan.release),
            )
        result.update(asdict(self.warp_layout))
        result["route_sfs_act"] = self.sf_route
        for name in ("load_regs", "load_sf_regs", "copy_sf_regs"):
            result.pop(name)
        result["cluster_dim_x"] = self.preferred_cluster_shape[0]
        result["ldgsts_sfb_producer_commit_prefetch_depth"] = (
            self.ldgsts_sfb_producer_commit_prefetch_depth
        )
        if not self.is_persistent:
            for name in ("num_stages_workid", "use_work_throttle", "workid_regs"):
                result[name] = 0
        if not self.use_tma_store:
            result["num_stages_c_smem"] = 0
        if not self.has_scale_factor_a and not self.use_deepseek_fp8:
            result["num_stages_smem_sfa"] = 0
        if not self.has_input_block_scaling:
            result["num_stages_smem_sfb"] = 0
        if not self.has_input_block_scaling or self.fuse_sf_copy_to_mma:
            result["num_stages_tmem_sfa"] = result["num_stages_tmem_sfb"] = 0
        return result

    @property
    def tile_m(self) -> int:
        """Rows staged per CTA by the implemented operand layouts."""
        return 128 * self.dense_m_repeats

    @property
    def uses_routed_tile512(self) -> bool:
        """Tile512 with routed A and LDGSTS SFA loaders.

        The two M fragments share MMA/epilogue code with dense tile512.
        Routed loaders retain completion relays; B/SFB may use a dense pair.
        """
        return (
            self.dense_m_repeats == 2
            and not self.is_swap_ab
            and self.has_routed_act
            and self.uses_ldgsts_routed_sfs
        )

    @property
    def uses_routed_weight_pipeline(self) -> bool:
        """Keep the dense B/SFB pair independent of routed activation rings."""
        return bool(
            (
                self.uses_routed_tile512
                or (
                    self.routed_native_sfa
                    and self.has_tma_route
                    and not self.is_swap_ab
                )
                or (self.fuse_operand_sf_loads and not self.is_swap_ab)
            )
            and self.dense_split_tma_pipeline
        )

    @property
    def partitions_routed_tma_a(self) -> bool:
        return bool(
            self.uses_routed_weight_pipeline
            and self.has_tma_route
            and self.dense_partition_tma_loads
        )

    @property
    def routed_a_y_partitions(self) -> int:
        """Static issuer count for the specialized gather4 body."""
        return (
            (self._active_cga_shape or self.preferred_cluster_shape)[1]
            if self.partitions_routed_tma_a
            else 1
        )

    @property
    def is_rubin(self) -> bool:
        """Whether this configuration targets SM107 Rubin."""
        return self.sm_arch == 107

    @property
    def arch_name(self) -> str:
        """CuTe DSL architecture spelling used by architecture helpers."""
        return f"sm_{self.sm_arch}"

    @property
    def smem_capacity_bytes(self) -> int:
        """Usable dynamic shared memory per CTA for the target architecture."""
        return (328 - 1) * 1024 if self.is_rubin else (228 - 1) * 1024

    @property
    def tmem_capacity_columns(self) -> int:
        """Maximum TMEM allocation columns for the target architecture."""
        return 576 if self.is_rubin else 512

    @property
    def dtype_a_bits(self) -> int:
        """Logical bit width of operand A (4/8/16/32), via :func:`dtype_bits`."""
        return dtype_bits(self.dtype_a)

    @property
    def dtype_b_bits(self) -> int:
        """Logical bit width of operand B (4/8/16/32)."""
        return dtype_bits(self.dtype_b)

    @property
    def dtype_c_bits(self) -> int:
        """Logical bit width of the output (4/8/16/32)."""
        return dtype_bits(self.dtype_c)

    @property
    def is_mx_mma(self) -> bool:
        """True when both operands are MX types (``MXE2M1``/``MXE4M3``);
        block-scaled MX MMA with ``mma_k = 32`` or Rubin ``mma_k = 64``.
        """
        mx_dtypes = (int(DType.MXE2M1), int(DType.MXE4M3))
        return self.dtype_a in mx_dtypes and self.dtype_b in mx_dtypes

    @property
    def is_fp8_mma(self) -> bool:
        """True for E4M3xE4M3 plain FP8 MMA (K32, or K64 on Rubin)."""
        return self.dtype_a == int(DType.E4M3) and self.dtype_b == int(DType.E4M3)

    @property
    def uses_native_mx_fp4_operands(self) -> bool:
        """Whether Rubin 2x-MMAK consumes MXFP4 directly from 4-bit SMEM.

        The legacy K32 MX instruction requires TMA to expand MXE2M1 operands
        into padded 8-bit SMEM. Rubin's K64 instruction consumes their packed
        4-bit representation directly.
        """
        return self.is_mx_mma and self.mma_k >= 64

    @property
    def has_cast_a(self) -> bool:
        """True for the MXFP4-weight / BF16-activation MMA path.

        A is loaded as packed ``MXE2M1`` + UE8M0 scales, cast to BF16 in a four-warp
        CastA task, staged in TMEM, then fed to a BF16 MMA. Requires
        ``dtype_a=MXE2M1`` and ``dtype_b=BF16``; the C output may be BF16 or
        FP16 after the FP32 accumulator cast.
        """
        return self.dtype_a == int(DType.MXE2M1) and self.dtype_b == int(DType.BF16)

    @property
    def is_nvfp4_mma(self) -> bool:
        """True when both operands are ``E2M1`` -- NVFP4 MMA.

        NVFP4 uses ``mma_k = 64`` on Blackwell and may use ``mma_k = 128``
        on Rubin.
        """
        return self.dtype_a == int(DType.E2M1) and self.dtype_b == int(DType.E2M1)

    @property
    def has_input_block_scaling(self) -> bool:
        """True when the inputs carry block scale factors
        (:attr:`is_mx_mma` or :attr:`is_nvfp4_mma`).
        """
        return self.is_mx_mma or self.is_nvfp4_mma

    @property
    def has_scale_factor_a(self) -> bool:
        """True when operand A has scale factors (block scaling or
        :attr:`has_cast_a`).
        """
        return self.has_input_block_scaling or self.has_cast_a

    @property
    def has_scale_factor_b(self) -> bool:
        """True when operand B has scale factors (block scaling)."""
        return self.has_input_block_scaling

    @property
    def uses_f16_mma(self) -> bool:
        """True when the MMA runs in 16-bit (BF16, or the CastA BF16 path)."""
        return self.is_bf16_mma or self.has_cast_a

    @property
    def uses_plain_mma(self) -> bool:
        """True for non-block-scaled MMA (BF16, CastA, or plain FP8)."""
        return self.uses_f16_mma or self.is_fp8_mma

    @property
    def uses_mx_scale_factors(self) -> bool:
        """True when any UE8M0 MX scale factors are present
        (MX input, CastA, or MXFP8 output).
        """
        return self.is_mx_mma or self.has_cast_a or self.uses_mxfp8_output_quant

    @property
    def uses_global_scales(self) -> bool:
        """True when per-tensor FP32 global scales participate in the epilogue.

        Applies to NVFP4 and plain-FP8 paths and requires :attr:`use_global_scales`
        != 0. MX inputs already carry UE8M0 block scales, so the flag is ignored there.
        """
        return (self.is_nvfp4_mma or self.is_fp8_mma) and self.use_global_scales != 0

    @property
    def uses_fp8_output(self) -> bool:
        """True for plain FP8 MMA writing an ``E4M3`` output
        (no output block scaling).
        """
        return self.is_fp8_mma and self.dtype_c == int(DType.E4M3)

    @property
    def has_deepseek_fp8_c_scale(self) -> bool:
        """DeepSeek FP8 E4M3 output uses the C-scale pointer for FP32 dq scales."""
        return self.use_deepseek_fp8 and self.uses_fp8_output

    @property
    def has_per_token_sf_a(self) -> bool:
        """True when plain-FP8 per-token A scales are active."""
        return self.is_fp8_mma and self.use_per_token_sf_a != 0

    @property
    def has_per_token_sf_b(self) -> bool:
        """True when plain-FP8 or NVFP4 per-token B scales are active
        (:attr:`use_per_token_sf_b`).
        """
        return (self.is_fp8_mma or self.is_nvfp4_mma) and self.use_per_token_sf_b != 0

    @property
    def has_swiglu_oai_params(self) -> bool:
        """True when any runtime per-expert SwiGLU-OAI parameter is active."""
        return self.act_kind == int(ActKind.SWIGLU) and self.has_gemm1_activation_params

    @property
    def has_gemm1_activation_params(self) -> bool:
        """True when any runtime per-expert gated-activation parameter is active."""
        return bool(
            self.has_gemm1_alpha or self.has_gemm1_beta or self.has_gemm1_clamp_limit
        )

    @property
    def has_deepseek_fp8_two_epilogue(self) -> bool:
        """DeepSeek FP8 FC2 path splits M across two epilogues."""
        return (
            self.use_deepseek_fp8
            and self.is_swap_ab
            and self.tile_m == 128
            and self.mma_m == 64
        )

    @property
    def uses_deepseek_fp8_atomic_absmax(self) -> bool:
        """Whether split DeepSeek epilogues share the atomic absmax scratch."""
        return self.has_deepseek_fp8_two_epilogue and self.epi_tile_n == 64

    @property
    def deepseek_fp8_epi_subtile_cnt(self) -> int:
        """Number of output-N epilogue subtiles required by DeepSeek FP8."""
        if self.is_swap_ab:
            warpgroup_count = self.num_epilogue_warpgroups
            if self.has_deepseek_fp8_two_epilogue:
                warpgroup_count = 1
            epi_cols_per_call = self.epi_tile_n * warpgroup_count
        else:
            epi_cols_per_call = self.epi_tile_n // 4
        return max(1, self.tile_n // max(1, epi_cols_per_call))

    @property
    def dsfp8_k_subblocks_per_tile(self) -> int:
        """DeepSeek scale blocks along K inside one ``tile_k`` tile (``1`` at block 128)."""
        return self.tile_k // self.deepseek_fp8_block_size

    @property
    def num_dsfp8_act_sfs_per_subblock(self) -> int:
        """DeepSeek FP8 activation scales staged per K sub-block."""
        token_tile_dim = self.tile_n if self.is_swap_ab else self.tile_m
        return max(token_tile_dim, 32)

    @property
    def num_dsfp8_act_sfs_per_stage(self) -> int:
        """DeepSeek FP8 activation scales staged per pipeline stage, ``[sub-block][token]``."""
        return self.dsfp8_k_subblocks_per_tile * self.num_dsfp8_act_sfs_per_subblock

    @property
    def num_bytes_dsfp8_sfab_per_stage(self) -> int:
        """SMEM bytes per DeepSeek SF stage, rounded for cp.async alignment."""
        raw = (
            self.num_dsfp8_act_sfs_per_stage + self.num_dsfp8_weight_sfs_per_stage
        ) * 4
        return ((raw + 15) // 16) * 16

    @property
    def num_dsfp8_weight_sfs_per_subblock(self) -> int:
        """DeepSeek FP8 weight scales staged per K sub-block.

        SwapAB stages one scale per ``deepseek_fp8_block_size`` rows of the
        ``tile_m`` weight tile. The non-swap path keeps one scale per
        ``tile_n`` weight tile.
        """
        if self.is_swap_ab:
            return self.dsfp8_row_groups_per_tile
        return 1

    @property
    def num_dsfp8_weight_sfs_per_stage(self) -> int:
        """DeepSeek FP8 weight scales staged per pipeline stage, ``[sub-block][row group]``."""
        return self.dsfp8_k_subblocks_per_tile * self.num_dsfp8_weight_sfs_per_subblock

    @property
    def dsfp8_row_groups_per_tile(self) -> int:
        """DeepSeek scale groups along the swapAB weight rows of one ``tile_m`` tile.

        Weight dequant scales and the FP32 output scales are both one value
        per group. ``1`` at block 128.
        """
        return self.tile_m // self.deepseek_fp8_block_size

    @property
    def uses_fp4_output_quant(self) -> bool:
        """True when C uses packed E2M1 storage (NVFP4 or MXFP4)."""
        return self.has_output_block_scaling and self.dtype_c in (
            int(DType.E2M1),
            int(DType.MXE2M1),
        )

    @property
    def uses_mxfp4_output_quant(self) -> bool:
        """True when the output is quantized to MXFP4 (``MXE2M1``)."""
        return self.has_output_block_scaling and self.dtype_c == int(DType.MXE2M1)

    @property
    def uses_mxfp8_output_quant(self) -> bool:
        """True when the output is quantized to MXFP8 (``MXE4M3``)."""
        return self.has_output_block_scaling and self.dtype_c == int(DType.MXE4M3)

    @property
    def uses_mx_output_quant(self) -> bool:
        """True when quantized C uses UE8M0 MX scale factors."""
        return self.uses_mxfp4_output_quant or self.uses_mxfp8_output_quant

    @property
    def input_sf_block_size_a(self) -> int:
        """Resolved A scale-factor block length in K elements
        (``32`` MX/CastA, ``16`` NVFP4).
        """
        return 32 if (self.is_mx_mma or self.has_cast_a) else 16

    @property
    def input_sf_block_size_b(self) -> int:
        """Resolved B scale-factor block length in K elements
        (``32`` MX, ``16`` NVFP4).
        """
        return 32 if self.is_mx_mma else 16

    @property
    def output_sf_block_size_c(self) -> int:
        """Resolved output scale-factor block length
        (``32`` MXFP4/MXFP8, ``16`` NVFP4).
        """
        return 32 if self.uses_mx_output_quant else 16

    @property
    def has_fused_act(self) -> bool:
        """True when a fused activation is selected (:attr:`act_kind` != NONE)."""
        return self.act_kind != int(ActKind.NONE)

    @property
    def has_activation_epilogue(self) -> bool:
        """True when the epilogue runs an activation, routed or fused."""
        return self.has_routed_act or self.has_fused_act

    @property
    def has_activation_side_effects(self) -> bool:
        """True when the epilogue does activation-side work.

        Covers routed or fused activation, output block scaling, or plain-FP8 output
        (:attr:`has_routed_act`, :attr:`has_fused_act`,
        :attr:`has_output_block_scaling`, :attr:`uses_fp8_output`).
        """
        return (
            self.has_routed_act
            or self.has_fused_act
            or self.has_output_block_scaling
            or self.uses_fp8_output
        )

    @property
    def has_output_block_scaling(self) -> bool:
        """True when the output is quantized and needs scale factors
        (``E2M1``, ``MXE2M1``, or ``MXE4M3``).
        """
        return self.dtype_c in (
            int(DType.E2M1),
            int(DType.MXE2M1),
            int(DType.MXE4M3),
        )

    @property
    def num_stages_cast_a(self) -> int:
        """CastA TMEM pipeline ring depth; mirrors :attr:`num_stages_a`."""
        return self.num_stages_a

    def _smem_bits_for_dtype(self, dtype_kind: int, dtype_bits: int) -> int:
        """Return the SMEM staging bit width for ``dtype_kind``
        (MX/FP8 stage at 8-bit except native Rubin 2x-MMAK MXFP4).
        """
        if dtype_kind == int(DType.MXE2M1) and self.uses_native_mx_fp4_operands:
            return 4
        if dtype_kind in (int(DType.MXE2M1), int(DType.MXE4M3), int(DType.E4M3)):
            return 8
        return dtype_bits

    def _tma_bits_for_dtype(self, dtype_kind: int, dtype_bits: int) -> int:
        """Return the TMA transfer bit width for ``dtype_kind``
        (``MXE2M1`` packs to 4-bit).
        """
        if dtype_kind == int(DType.MXE2M1):
            return 4
        if dtype_kind in (int(DType.MXE4M3), int(DType.E4M3)):
            return 8
        return dtype_bits

    @property
    def dtype_a_smem_bits(self) -> int:
        """SMEM staging width of A in bits.

        MXFP4 is 4-bit for CastA and Rubin K64 MX MMA, otherwise MX/FP8 uses
        8-bit staging.
        """
        if self.has_cast_a:
            return 4
        return self._smem_bits_for_dtype(self.dtype_a, self.dtype_a_bits)

    @property
    def dtype_b_smem_bits(self) -> int:
        """SMEM staging width of B in bits (native MXFP4 may remain 4-bit)."""
        return self._smem_bits_for_dtype(self.dtype_b, self.dtype_b_bits)

    @property
    def dtype_a_tma_bits(self) -> int:
        """TMA transfer width of A in bits (``4`` for ``MXE2M1``,
        ``8`` for MX/FP8, else logical).
        """
        return self._tma_bits_for_dtype(self.dtype_a, self.dtype_a_bits)

    @property
    def dtype_b_tma_bits(self) -> int:
        """TMA transfer width of B in bits."""
        return self._tma_bits_for_dtype(self.dtype_b, self.dtype_b_bits)

    @property
    def weight_layout_kind(self) -> int:
        """Integer :class:`~flashinfer.tllm_enums.WeightLayout` value."""
        return int(self.weight_layout)

    @property
    def uses_block_major_k_weight(self) -> bool:
        """True when the physical weight operand uses BlockMajorK storage."""
        return self.weight_layout_kind == int(WeightLayout.BlockMajorK)

    @property
    def uses_block_major_k_weight_a(self) -> bool:
        """True when operand A is the BlockMajorK weight tensor."""
        return self.uses_block_major_k_weight and self.is_swap_ab

    @property
    def uses_block_major_k_weight_b(self) -> bool:
        """True when operand B is the BlockMajorK weight tensor."""
        return self.uses_block_major_k_weight and not self.is_swap_ab

    @property
    def weight_dtype_tma_bits(self) -> int:
        """TMA transfer bit width of the physical weight operand."""
        return self.dtype_a_tma_bits if self.is_swap_ab else self.dtype_b_tma_bits

    @property
    def weight_dtype_smem_bits(self) -> int:
        """SMEM staging bit width of the physical weight operand."""
        return self.dtype_a_smem_bits if self.is_swap_ab else self.dtype_b_smem_bits

    @property
    def block_major_k_pad_multiplier(self) -> int:
        """TRT-LLM Gen BlockMajorK padding multiplier for the weight operand."""
        tma_bits = self.weight_dtype_tma_bits
        smem_bits = self.weight_dtype_smem_bits
        if smem_bits % tma_bits != 0:
            raise ValueError(
                "BlockMajorK weight SMEM bit width must be a multiple of TMA "
                f"bit width, got smem_bits={smem_bits}, tma_bits={tma_bits}"
            )
        return max(1, smem_bits // tma_bits)

    @property
    def block_major_k_elems_in_128b(self) -> int:
        """Logical K elements in one unpadded 128B BlockMajorK slice."""
        return 128 * 8 // self.weight_dtype_tma_bits

    @property
    def block_major_k_elems(self) -> int:
        """Logical K elements covered by one BlockMajorK K block.

        The elements of a 128B K block, divided by the padding multiplier of
        padded formats (MXFP4 staged as 8-bit), whose TMA boxes are exactly
        64B wide. One TMA box then loads the ``tile_k / blockK_elems`` K blocks
        of a K tile for any ``tile_k``.
        """
        return self.block_major_k_elems_in_128b // self.block_major_k_pad_multiplier

    @property
    def block_major_k_bytes(self) -> int:
        """BlockMajorK K-block bytes of the weight TensorMap."""
        return self.block_major_k_elems * self.weight_dtype_tma_bits // 8

    @property
    def block_major_k_tile_block_elems(self) -> int:
        """Fastest TMA dimension used for one BlockMajorK tile load."""
        return min(self.block_major_k_elems, self.tile_k)

    @property
    def block_major_k_tile_slices(self) -> int:
        """Number of BlockMajorK K-block slices covered by one tile load."""
        return self.tile_k // self.block_major_k_tile_block_elems

    @property
    def mx_a_format(self) -> int:
        """MX MMA format code for A (:data:`MX_MMA_FORMAT_E4M3` /
        :data:`MX_MMA_FORMAT_E2M1`); raises if A is not an MX dtype.
        """
        if self.dtype_a == int(DType.MXE4M3):
            return MX_MMA_FORMAT_E4M3
        if self.dtype_a == int(DType.MXE2M1):
            return MX_MMA_FORMAT_E2M1
        raise ValueError(f"dtype_a is not an MX MMA dtype: {self.dtype_a}")

    @property
    def mx_b_format(self) -> int:
        """MX MMA format code for B; raises if B is not an MX dtype."""
        if self.dtype_b == int(DType.MXE4M3):
            return MX_MMA_FORMAT_E4M3
        if self.dtype_b == int(DType.MXE2M1):
            return MX_MMA_FORMAT_E2M1
        raise ValueError(f"dtype_b is not an MX MMA dtype: {self.dtype_b}")

    @property
    def tmem_sfa_cols(self) -> int:
        """TMEM columns per stage for the A scale factors
        (``0`` without block scaling).
        """
        return self.tmem_sf_cols("a")

    @property
    def tmem_sfb_cols(self) -> int:
        """TMEM columns per stage for the B scale factors
        (``0`` without block scaling).
        """
        return self.tmem_sf_cols("b")

    def _tmem_sf_tile_mn(self, operand: str) -> int:
        """Rows/columns addressed by one MMA's scale base, excluding M repeats."""
        if operand == "a":
            return 128
        if operand == "b":
            return self.tile_n
        raise ValueError(f"Unsupported scale-factor operand: {operand}")

    def _tmem_sf_vec_size(self, operand: str) -> int:
        """Return the SF block length for ``operand`` (:attr:`input_sf_block_size_a` / ``_b``)."""
        if operand == "a":
            return self.input_sf_block_size_a
        if operand == "b":
            return self.input_sf_block_size_b
        raise ValueError(f"Unsupported scale-factor operand: {operand}")

    def tmem_sf_cols(self, operand: str) -> int:
        """Return TMEM columns for one stage of scale-factor ``operand``
        (``"a"``/``"b"``); ``0`` without block scaling.
        """
        if not self.has_input_block_scaling:
            return 0
        if self.routed_native_sfa and operand == "a":
            return 4 * self.dense_m_repeats
        return (self.dense_m_repeats if operand == "a" else 1) * tmem_sf_cols(
            self._tmem_sf_tile_mn(operand),
            self.tile_k,
            self._tmem_sf_vec_size(operand),
            self.mma_k,
        )

    def tmem_sf_col_stride(self, operand: str) -> int:
        """Return the TMEM column stride for one four-vector SF group
        of ``operand`` (``"a"``/``"b"``).
        """
        if self.routed_native_sfa and operand == "a":
            return 2
        return tmem_sf_col_stride(
            self._tmem_sf_tile_mn(operand),
            self.mma_k,
            self._tmem_sf_vec_size(operand),
        )

    def tmem_sf_vec4_groups_per_mma(self, operand: str) -> int:
        """Packed vec4 SF words consumed from one MMA scale base."""
        return tmem_sf_vec4_groups_per_mma(self.mma_k, self._tmem_sf_vec_size(operand))

    @property
    def tmem_cast_a_cols(self) -> int:
        """TMEM columns per CastA stage (``128`` when :attr:`has_cast_a`,
        else ``0``).
        """
        return 128 if self.has_cast_a else 0

    @property
    def tmem_c_cols_per_stage(self) -> int:
        """TMEM columns for all M fragments in one accumulator stage."""
        return self.dense_m_repeats * tmem_d_cols_per_stage(self.tile_n)

    @property
    def num_bytes_a_per_stage(self) -> int:
        """SMEM bytes for one A tile stage
        (``tile_m * tile_k * dtype_a_smem_bits / 8``).
        """
        return self.tile_m * self.tile_k * self.dtype_a_smem_bits // 8

    @property
    def num_bytes_b_per_stage(self) -> int:
        """SMEM bytes for one full (cluster-wide) B tile stage."""
        return self.tile_n * self.tile_k * self.dtype_b_smem_bits // 8

    @property
    def num_bytes_a_tma_per_stage(self) -> int:
        """GMEM bytes moved by one A TMA load
        (uses the packed :attr:`dtype_a_tma_bits`).
        """
        return self.tile_m * self.tile_k * self.dtype_a_tma_bits // 8

    @property
    def num_bytes_b_tma_per_stage(self) -> int:
        """GMEM bytes moved for one B stage per CTA.

        Two-CTA UMMA partitions B along N, so each CTA transfers one
        ``1 / umma_cta_count`` slice.
        """
        bytes_per_stage = self.tile_n * self.tile_k * self.dtype_b_tma_bits // 8
        if self.split_b_across_ctas:
            return bytes_per_stage // self.umma_cta_count
        return bytes_per_stage

    @property
    def num_bytes_b_smem_per_stage(self) -> int:
        """Per-CTA SMEM bytes for one B tile stage.

        Two-CTA UMMA partitions B along N, so each CTA stages one
        ``1 / umma_cta_count`` slice. Single-CTA kernels stage the full tile.
        """
        if self.split_b_across_ctas:
            return self.num_bytes_b_per_stage // self.umma_cta_count
        return self.num_bytes_b_per_stage

    @property
    def num_bytes_sfa_per_stage(self) -> int:
        """SMEM bytes for one A scale-factor stage."""
        return self.tile_m * (self.tile_k // self.input_sf_block_size_a)

    @property
    def smem_sfa_stage_stride_bytes(self) -> int:
        """128-byte-aligned SMEM stride between A scale-factor stages."""
        return ((self.num_bytes_sfa_per_stage + 127) // 128) * 128

    @property
    def num_bytes_sfb_per_stage(self) -> int:
        """SMEM bytes for one B scale-factor stage.

        The physical ``R128c4`` layout stores complete 128-row groups.  Its
        TMA descriptor transfers those padded groups, so both the allocation
        and the pipeline transaction byte count must use the same rounded
        extent for non-power-of-two token tiles such as 96 and 192.
        """
        if self.smem_sfb_layout == int(SfLayout.R128c4):
            mn = ((self.tile_n + 127) // 128) * 128
        else:
            mn = max(16, self.tile_n) if self.is_mx_mma else self.tile_n
        return mn * (self.tile_k // self.input_sf_block_size_b)

    @property
    def smem_sfb_stage_stride_bytes(self) -> int:
        """128-byte-aligned SMEM stride between B scale-factor stages."""
        return ((self.num_bytes_sfb_per_stage + 127) // 128) * 128

    @property
    def num_bytes_sfb_tma_per_stage(self) -> int:
        """GMEM bytes moved by one SFB TMA load.

        Smaller than the SMEM allocation for the compact 8x4 SFB layout
        (:attr:`uses_sfb_8x4_load`); otherwise equals :attr:`num_bytes_sfb_per_stage`.
        """
        if self.uses_sfb_8x4_load:
            k_groups = (
                self.tile_k // self.input_sf_block_size_b // TMEM_SF_PACK_SIZE_BYTES
            )
            outer_groups = (self.tile_n + 7) // 8
            return 32 * k_groups * outer_groups
        return self.num_bytes_sfb_per_stage

    @property
    def num_bytes_c_per_stage(self) -> int:
        """SMEM bytes for one epilogue C tile
        (``epi_tile_m * epi_tile_n * dtype_c_bits / 8``).
        """
        if self.fuse_moe_finalize:
            return (
                self.epi_tile_m * self.finalize_smem_row_stride * self.dtype_c_bits // 8
            )
        return self.epi_tile_m * self.epi_tile_n * self.dtype_c_bits // 8

    @property
    def finalize_smem_row_stride(self) -> int:
        # Bulk reductions need contiguous rows rather than TMA swizzling.
        # A 16-byte pad keeps each row aligned and separates neighboring
        # lanes' vector stores across banks instead of a 128-byte stride.
        return self.tile_n + 8

    @property
    def uses_full_tile_finalize(self) -> bool:
        """Stage both M128 halves before TMEM release, then scatter owned rows.

        Four warps own complete rows, so no cross-warp N-fragment assembly is
        needed. Validation requires two C scratch stages for this schedule.
        """
        plan = self.preload_epilogue_plan
        return plan is not None and plan.full_tile

    @cached_property
    def preload_epilogue_plan(self):
        """Compile-time data-lifetime plan shared by TMEM, SMEM and store tasks."""
        return resolve_preload_epilogue(self)

    @property
    def tma_store_cols(self) -> int:
        """N columns emitted by one C TMA store transaction.

        Non-swap C is row-major, so N is the descriptor's contiguous dimension.
        ``s128b`` is the largest hardware TensorMap swizzle mode; cap that
        contiguous dimension at 128 bytes and emit a wider ``epi_tile_n`` in
        :attr:`tma_store_passes` slices.

        SwapAB C is M-major, so N is not the contiguous dimension and does not
        select or constrain the swizzle. Its descriptor covers the complete N
        stripe; :attr:`swap_tma_store_rows` handles the 128-byte cap along M.
        """
        if self.is_swap_ab:
            return self.epi_tile_n
        return self.non_swap_tma_store_cols

    @property
    def non_swap_tma_store_cols(self) -> int:
        """N columns in one non-swap TMA transaction.

        N is contiguous for row-major non-swap C. Convert the hardware's
        maximum 128-byte swizzle width into output elements and slice a wider
        ``epi_tile_n`` across multiple TMA stores.
        """
        epilogue_output_cols = self.epi_tile_n
        if self.has_gated_epilogue:
            epilogue_output_cols //= 2
        max_swizzled_cols = TMA_MAX_SWIZZLE_BYTES * 8 // self.dtype_c_bits
        return min(epilogue_output_cols, max_swizzled_cols)

    @property
    def tma_store_passes(self) -> int:
        """Number of descriptor-width N stores per epilogue stripe."""
        epilogue_output_cols = self.epi_tile_n
        if not self.is_swap_ab and self.has_gated_epilogue:
            epilogue_output_cols //= 2
        return epilogue_output_cols // self.tma_store_cols

    @property
    def c_tma_swizzle_bytes(self) -> int:
        """C descriptor's capped contiguous bytes used to select its swizzle.

        The contiguous epilogue dimension is M for swapAB and N otherwise.
        TensorMap swizzle modes stop at ``s128b``, so wider tiles use the
        128-byte mode and are emitted in multiple slices along that dimension.
        """
        contiguous_elems = (
            self.swap_tma_store_tile_rows if self.is_swap_ab else self.tma_store_cols
        )
        return min(
            contiguous_elems * self.dtype_c_bits // 8,
            TMA_MAX_SWIZZLE_BYTES,
        )

    @property
    def c_tma_smem_atom_elements(self) -> int:
        """C elements in one independently swizzled 16-byte SMEM atom."""
        return TMA_SWIZZLE_ATOM_BYTES * 8 // self.dtype_c_bits

    @property
    def non_swap_tma_accumulator_cols(self) -> int:
        """Accumulator columns consumed by one non-swap TMA output stripe."""
        cols = self.tma_store_cols
        if self.has_gated_epilogue:
            cols *= 2
        return cols

    @property
    def swap_tma_store_tile_rows(self) -> int:
        """Output M rows staged by one swapAB epilogue store tile.

        This is ``epi_tile_m`` for an ordinary swapAB epilogue. Gated
        activation combines each gate/up pair into one output row, so its
        physical TMA store tile holds half as many contiguous rows.

        The two-epilogue DeepSeek path splits ``tile_m`` between its two
        epilogue warpgroups, so one store tile spans that warpgroup's share,
        ``tile_m // 2``. Halving ``epi_tile_m`` instead under-counts whenever
        ``epi_tile_m`` is itself already the per-warpgroup share: the store
        loop then emits only the lower half of the rows the epilogue staged,
        while the SMEM group stride and the per-warpgroup global M base in
        ``gmem_c_resources`` both step a full ``tile_m // 2`` rows, leaving the
        upper half staged but never stored.
        """
        if self.has_gated_epilogue:
            return self.epi_tile_m // 2
        if self.has_deepseek_fp8_two_epilogue:
            return min(self.epi_tile_m, self.tile_m // 2)
        return self.epi_tile_m

    @property
    def swap_tma_store_rows(self) -> int:
        """Contiguous swapAB M rows covered by one C TensorMap transaction.

        SwapAB C is contiguous in M, so the swizzle mode and descriptor width
        come from :attr:`swap_tma_store_tile_rows`, capped at the hardware's
        maximum 128-byte swizzle. A wider physical tile is emitted in multiple
        transactions.
        """
        if self.c_tma_swizzle_bytes < 32:
            return self.swap_tma_store_tile_rows
        max_rows = self.c_tma_swizzle_bytes * 8 // self.dtype_c_bits
        return min(self.swap_tma_store_tile_rows, max_rows)

    @property
    def swap_tma_store_col_group_step_elems(self) -> int:
        """SMEM elements between adjacent eight-column swapAB groups."""
        return 8 * self.swap_tma_store_rows

    @property
    def swap_tma_store_offsets_are_col_linear(self) -> bool:
        """Whether eight-column steps preserve the swapAB C swizzle mask.

        The cached-offset store is for 16-bit output. At s32b and wider, its
        eight-column step spans one complete swizzle period; below s32b there
        is no XOR mask.
        """
        if not self.is_swap_ab or self.dtype_c_bits != 16:
            return False
        if self.c_tma_swizzle_bytes < 32:
            return True
        element_bytes = self.dtype_c_bits // 8
        step_bytes = self.swap_tma_store_col_group_step_elems * element_bytes
        swizzle_period_bytes = self.c_tma_swizzle_bytes * 8
        return step_bytes % swizzle_period_bytes == 0

    @property
    def swap_tma_store_row_passes(self) -> int:
        """Number of contiguous-M transactions per swapAB store tile."""
        return self.swap_tma_store_tile_rows // self.swap_tma_store_rows

    @property
    def non_swap_tmem_load_num_regs(self) -> int:
        """Largest legal non-swap ``32x32b`` TMEM load under the configured cap."""
        # The DeepSeek epilogue carries a same-sized FP32 dequant accumulator
        # through the K loop and retains its existing conservative fragment.
        if self.use_deepseek_fp8:
            return max(1, self.epi_tile_n // 4)

        width_limit = min(self.epi_tile_n, self.tmem_ldst_max_num_regs)
        if self.use_tma_store:
            width_limit = min(width_limit, self.non_swap_tma_accumulator_cols)
        num_regs = 1 << (width_limit.bit_length() - 1)
        while self.epi_tile_n % num_regs != 0:
            num_regs //= 2
        return num_regs

    @property
    def non_swap_tmem_overlap_loads(self) -> int:
        """T2R loads needed to preserve the ping-pong window intersection."""
        return self.epi_tile_n // self.non_swap_tmem_load_num_regs

    @property
    def non_swap_epilogue_warpgroup_count(self) -> int:
        """Four-warp row groups that partition N in a wide gated epilogue."""
        if self.is_swap_ab or not self.has_gated_epilogue:
            return 1
        return self.num_epilogue_warpgroups

    @property
    def non_swap_epilogue_fragments_per_group(self) -> int:
        """Resident ``32x32b`` loads consumed by one warpgroup per step."""
        if not self.has_gated_epilogue:
            return 1
        total_fragments = self.epi_tile_n // self.non_swap_tmem_load_num_regs
        return min(MAX_NON_SWAP_TMEM_FRAGMENTS, total_fragments)

    @property
    def non_swap_gated_accumulator_cols_per_step(self) -> int:
        """Pre-gating accumulator columns resident in one warpgroup step."""
        return (
            self.non_swap_tmem_load_num_regs
            * self.non_swap_epilogue_fragments_per_group
        )

    @property
    def non_swap_gated_steps_per_epi_tile(self) -> int:
        """Register-sized steps used to drain one gated epilogue tile."""
        if self.is_swap_ab or not self.has_gated_epilogue:
            return 1
        return self.epi_tile_n // self.non_swap_gated_accumulator_cols_per_step

    @property
    def non_swap_epilogue_cols_per_call(self) -> int:
        """Pre-gated accumulator columns covered by one schedule call."""
        if not self.has_gated_epilogue:
            return self.non_swap_tmem_load_num_regs
        return (
            self.non_swap_gated_accumulator_cols_per_step
            * self.non_swap_epilogue_warpgroup_count
        )

    @property
    def non_swap_gated_output_cols_per_call(self) -> int:
        """Post-activation columns produced by one non-swap epilogue group."""
        return self.non_swap_gated_accumulator_cols_per_step // 2

    @property
    def non_swap_output_sf_blocks_per_call(self) -> int:
        """Output quantization blocks produced by one non-swap epilogue call."""
        return self.non_swap_gated_output_cols_per_call // self.output_sf_block_size_c

    @property
    def non_swap_epilogue_subtile_cnt(self) -> int:
        """Schedule calls needed to drain one non-swap accumulator tile."""
        # Gated cols_per_call already includes all epilogue warpgroups.
        # Applying the plain-output division as well would skip half the tile.
        plain_warpgroups = (
            1 if self.has_gated_epilogue else self.dense_epilogue_warps // 4
        )
        return (
            self.dense_m_repeats
            * max(1, self.tile_n // self.non_swap_epilogue_cols_per_call)
            // plain_warpgroups
        )

    @property
    def uses_tmem_overlap_window(self) -> bool:
        """Whether accumulator sets share an epilogue-preloaded TMEM window."""
        return self.use_tile256_tmem_overlap and (
            self.warp_layout.num_epilogue_warps == 4
            or (
                not self.is_swap_ab
                and self.has_gated_epilogue
                and self.warp_layout.num_epilogue_warps == 8
            )
        )

    @property
    def tmem_overlap_window_cols(self) -> int:
        """TMEM columns occupied by accumulator stages sharing a window."""
        if self.uses_tmem_overlap_window:
            return 2 * self.tile_n - self.epi_tile_n
        return self.tmem_c_cols_per_stage

    @property
    def swap_tmem_load_num_regs(self) -> int:
        """Register count for one swapAB ``16x256b`` TMEM load.

        One xN repetition returns four registers and covers eight output
        columns. A logical ``epi_tile_n`` fragment may therefore require more
        than one instruction under the configured register cap.
        """
        total_regs_per_slice = max(1, self.epi_tile_n // 2)
        # The DeepSeek epilogue re-reads the accumulator every K block, so one
        # wide load per slice (as before the cap) keeps the epilogue warps from
        # falling behind the MMA at epi_tile_n=128; two half loads cost ~25%.
        if self.use_deepseek_fp8:
            return max(4, total_regs_per_slice)

        width_limit = min(total_regs_per_slice, self.tmem_ldst_max_num_regs)
        num_regs = 1 << (width_limit.bit_length() - 1)
        while total_regs_per_slice % num_regs != 0:
            num_regs //= 2
        return max(4, num_regs)

    @property
    def swap_tmem_load_reps(self) -> int:
        """xN repetition count for one swapAB ``16x256b`` TMEM load."""
        return max(1, self.swap_tmem_load_num_regs // 4)

    @property
    def swap_tmem_load_cols(self) -> int:
        """Output columns covered by one swapAB TMEM load instruction."""
        return self.swap_tmem_load_reps * 8

    @property
    def swap_tmem_overlap_loads(self) -> int:
        """T2R loads needed for one logical swapAB epilogue fragment."""
        return self.epi_tile_n // self.swap_tmem_load_cols

    @property
    def tmem_load_num_regs(self) -> int:
        """Register count issued by one layout-appropriate TMEM load."""
        if self.is_swap_ab:
            return self.swap_tmem_load_num_regs
        return self.non_swap_tmem_load_num_regs

    @property
    def tmem_overlap_loads(self) -> int:
        """T2R instructions needed to preserve the ping-pong intersection."""
        if self.is_swap_ab:
            return self.swap_tmem_overlap_loads
        return self.non_swap_tmem_overlap_loads

    @property
    def num_epilogue_warpgroups(self) -> int:
        """Number of independently synchronized epilogue warpgroups."""
        return max(1, self.warp_layout.num_epilogue_warps // WARPS_PER_WARPGROUP)

    @property
    def epilogue_threads_per_warpgroup(self) -> int:
        """Threads participating in one epilogue scratch barrier."""
        return WARP_SIZE * WARPS_PER_WARPGROUP

    @property
    def num_bytes_c_tma_store_per_group(self) -> int:
        """Bytes staged by one TMA-store epilogue warpgroup (halved for a gated epilogue)."""
        stage_bytes = self.num_bytes_c_per_stage
        if self.has_gated_epilogue:
            stage_bytes //= 2
        return stage_bytes

    @property
    def uses_non_swap_c_tma_ring(self) -> bool:
        """Whether BatchM C stores carry a multi-stage ring across work tiles."""
        return (
            not self.is_swap_ab
            and bool(self.use_tma_store)
            and self.num_stages_c_smem > 1
            and not self.use_deepseek_fp8
            and (
                self.uses_non_swap_fp4_gated_tma_c_scratch
                or (
                    not self.has_output_block_scaling
                    and (
                        not self.has_gated_epilogue
                        or self.uses_16bit_gated_tma_c_scratch
                    )
                )
            )
        )

    @property
    def uses_non_swap_fp4_gated_tma_c_scratch(self) -> bool:
        """Whether each BatchM epilogue group stages a gated NVFP4 TMA box."""
        return (
            not self.is_swap_ab
            and bool(self.use_tma_store)
            and self.uses_fp4_output_quant
            and not self.uses_mx_output_quant
            and self.has_gated_epilogue
        )

    @property
    def non_swap_gated_tma_box_bytes(self) -> int:
        """Bytes in one physical gated BatchM TMA box."""
        return self.epi_tile_m * self.tma_store_cols * self.dtype_c_bits // 8

    @property
    def num_swap_ab_nongated_quant_tma_stages(self) -> int:
        """Active C-scratch stages for non-gated swapAB block-scaled output.

        Multi-stage reuse is safe only when a tile contains more than one
        epilogue call and the calls divide evenly across the configured ring.
        The latter makes the first call of the next tile return to stage zero
        instead of reusing the previous tile's last in-flight stage.
        """
        if not (
            self.is_swap_ab
            and self.use_tma_store
            and self.has_output_block_scaling
            and not self.has_gated_epilogue
        ):
            return 1
        if self.num_stages_c_smem <= 1:
            return 1
        output_cols_per_call = self.epi_tile_n * self.num_epilogue_warpgroups
        calls_per_tile = self.tile_n // output_cols_per_call
        if (
            self.tile_n > output_cols_per_call
            and calls_per_tile % self.num_stages_c_smem == 0
        ):
            return self.num_stages_c_smem
        return 1

    @property
    def uses_16bit_gated_tma_c_scratch(self) -> bool:
        """True when a 16-bit gated epilogue stages only gated output rows in SMEM.

        This is an output property, independent of the MMA input type.  Reserve
        one half-N scratch tile per epilogue warpgroup for either BF16 or FP16
        output, including when the MMA inputs are NVFP4.
        """
        return (
            bool(self.use_tma_store)
            and self.has_gated_epilogue
            and not self.has_output_block_scaling
            and not self.uses_fp8_output
            and self.dtype_c in (int(DType.BF16), int(DType.FP16))
        )

    @property
    def uses_deepseek_fp8_split_epilogue_tma_c_scratch(self) -> bool:
        """Whether DeepSeek FP8 stages one split-M BF16/FP16 output tile.

        This path reserves one TMA scratch tile per epilogue warpgroup. It
        requires num_stages_c_smem=1 when this property is true.
        """
        return (
            self.has_deepseek_fp8_two_epilogue
            and bool(self.use_tma_store)
            and self.dtype_c in (int(DType.BF16), int(DType.FP16))
        )

    @property
    def num_bytes_c_smem_scratch(self) -> int:
        """Total SMEM bytes reserved for the C epilogue staging buffer.

        Usually :attr:`num_bytes_c_per_stage` times
        :attr:`num_stages_c_smem`. The 16-bit gated and DeepSeek split-M paths
        use reduced per-warpgroup forms. SwapAB block-scaled output additionally
        reserves packed per-warpgroup storage for every active TMA stage, plus
        the gated-MX FP32 absmax exchange scratch.
        """
        if self.uses_non_swap_fp4_gated_tma_c_scratch:
            return (
                self.num_bytes_c_tma_store_per_group
                * self.num_stages_c_smem
                * self.num_epilogue_warpgroups
            )
        if self.uses_deepseek_fp8_split_epilogue_tma_c_scratch:
            return (
                (self.tile_m // 2)
                * self.epi_tile_n
                * self.dtype_c_bits
                // 8
                * self.num_epilogue_warpgroups
            )
        if self.uses_16bit_gated_tma_c_scratch:
            bytes_per_group = self.num_bytes_c_tma_store_per_group
            if not self.is_swap_ab:
                bytes_per_group = self.non_swap_gated_tma_box_bytes
            return (
                bytes_per_group * self.num_stages_c_smem * self.num_epilogue_warpgroups
            )
        stage_bytes = (
            self.num_bytes_c_per_stage
            * self.num_stages_c_smem
            * (1 if self.dense_preload_epilogue else self.dense_epilogue_warps // 4)
        )
        if self.use_tma_store and self.has_output_block_scaling and self.is_swap_ab:
            packed_store_bytes = (
                self.num_bytes_c_tma_store_per_group
                * self.num_epilogue_warpgroups
                * self.num_swap_ab_nongated_quant_tma_stages
            )
            reduction_scratch_bytes = 0
            if self.has_gated_epilogue and self.uses_mx_output_quant:
                reduction_scratch_bytes = (
                    max(1, self.epi_tile_n // 8) * 32 * 4 * self.num_epilogue_warpgroups
                )
            stage_bytes = max(stage_bytes, packed_store_bytes + reduction_scratch_bytes)
        return stage_bytes

    @property
    def aliases_c_scratch_with_ab(self) -> bool:
        """True when the C epilogue scratch may reuse the A/B SMEM buffers.

        Persistent loaders can start the next tile before its predecessor's
        epilogue completes, even with one work-id stage. Preserve the separate
        staging required by :attr:`uses_16bit_gated_tma_c_scratch` and CastA
        paths as well.
        """
        if self.uses_16bit_gated_tma_c_scratch:
            return False
        if self.dense_fuse_tma_pipeline:
            return False
        return not self.is_persistent and not self.has_cast_a

    @property
    def tmem_alloc_blocks_second_cta(self) -> bool:
        """True when the TMEM allocation leaves no room for a second CTA on the SM.

        ``tcgen05.alloc`` blocks until the requested columns are free, so a
        second resident CTA could only wait for the first one to finish.
        """
        return 2 * self.tmem_alloc_cols > self.tmem_capacity_columns

    @property
    def smem_single_cta_floor_bytes(self) -> int:
        """Dynamic-SMEM floor that limits residency to one CTA per SM (0 = none).

        Applied only when :attr:`tmem_alloc_blocks_second_cta` holds.  Requesting
        more than half of the SM's shared memory makes the hardware block
        scheduler see the one-CTA limit that TMEM already imposes, instead of
        placing a second CTA that spins in ``tcgen05.alloc`` next to the running
        one.  Smaller footprints (16-bit gated FC1 epilogue scratch, narrow
        tiles) otherwise drop below that threshold on SM100.
        """
        if not self.tmem_alloc_blocks_second_cta:
            return 0
        return self.smem_capacity_bytes // 2 + 1

    @property
    def tmem_alloc_cols(self) -> int:
        """Hardware TMEM allocation: :attr:`tmem_required_cols` rounded up
        to a power of two (minimum ``32``), except that Rubin footprints above
        512 columns use exclusive allocation rounded to a 32-column multiple.
        """
        if self.is_rubin and self.tmem_required_cols > 512:
            return ((self.tmem_required_cols + 31) // 32) * 32
        return max(32, 1 << (self.tmem_required_cols - 1).bit_length())

    @property
    def tmem_required_cols(self) -> int:
        """TMEM column footprint before hardware power-of-two rounding.

        Sums accumulator, CastA, and scale-factor staging across stages; must stay
        <= :attr:`tmem_capacity_columns` (:func:`validate_config`). See
        :attr:`tmem_alloc_cols` for the rounded allocation.
        """
        total = self.tmem_overlap_window_cols * self.num_stages_tmem_acc
        if self.has_cast_a:
            total += self.tmem_cast_a_cols * self.num_stages_cast_a
        if not self.has_input_block_scaling:
            return total

        if self.use_combined_sfab_copy:
            total += (
                self.tmem_sfa_cols * self.num_stages_tmem_sfa
                + self.tmem_sfb_cols * self.num_stages_tmem_sfb
            )
        elif self.has_input_block_scaling and not self.fuse_sf_copy_to_mma:
            total += self.tmem_sfa_cols * self.num_stages_tmem_sfa
            total += self.tmem_sfb_cols * self.num_stages_tmem_sfb
        else:
            total += self.tmem_sfa_cols + self.tmem_sfb_cols
        return total

    @property
    def smem_sfa_layout(self) -> int:
        """Effective SMEM layout of the A scale factors, as a :class:`SfLayout` value.

        Overrides :attr:`sf_layout_a` for routed SFA: ``LINEAR`` for TMA-routed,
        ``R128c4`` for LDGSTS-routed; otherwise the configured layout.
        """
        if self.has_routed_sfs and not self.is_swap_ab:
            if self.uses_tma_routed_sfs:
                return int(SfLayout.LINEAR)
            return int(SfLayout.R128c4)
        return self.sf_layout_a

    @property
    def smem_sfb_layout(self) -> int:
        """Effective SMEM layout of the B scale factors, as a :class:`SfLayout` value.

        Overrides :attr:`sf_layout_b` for routed :attr:`is_swap_ab` SFB
        (``LINEAR`` for TMA-routed; ``R8c4`` for ``tile_n < 128`` else
        ``R128c4`` for LDGSTS-routed) and forces ``R8c4`` when
        :attr:`uses_sfb_8x4_load`.
        """
        if self.has_routed_sfs and self.is_swap_ab:
            if self.uses_tma_routed_sfs:
                return int(SfLayout.LINEAR)
            if self.tile_n < 128:
                return int(SfLayout.R8c4)
            return int(SfLayout.R128c4)
        if self.uses_sfb_8x4_load:
            return int(SfLayout.R8c4)
        return self.sf_layout_b

    def _sf_smem_to_tmem_copy(self, smem_layout: int) -> int:
        """Map an SMEM SF layout to its :class:`SfSmemToTmemCopy` strategy."""
        if not self.has_input_block_scaling:
            return int(SfSmemToTmemCopy.NONE)
        if smem_layout == int(SfLayout.R128c4):
            return int(SfSmemToTmemCopy.UTCCP)
        return int(SfSmemToTmemCopy.LDS_STTM)

    @property
    def sfa_smem_to_tmem_copy(self) -> int:
        """:class:`SfSmemToTmemCopy` strategy for A SF, from
        :attr:`smem_sfa_layout`.
        """
        return self._sf_smem_to_tmem_copy(self.smem_sfa_layout)

    @property
    def sfb_smem_to_tmem_copy(self) -> int:
        """:class:`SfSmemToTmemCopy` strategy for B SF, from
        :attr:`smem_sfb_layout`.
        """
        if (
            (self.has_input_block_scaling and not self.fuse_sf_copy_to_mma)
            and self.has_routed_sfs
            and self.is_swap_ab
            and self.uses_ldgsts_routed_sfs
            and self.tile_n % 128 != 0
        ):
            # R128c4 staging pads N=192 to two 128-row blocks. UTCCP copies
            # that padded representation verbatim, but the N=192 MMA consumes
            # six contiguous scale columns. Use LDS+STTM to compact 8 -> 6.
            return int(SfSmemToTmemCopy.LDS_STTM)
        return self._sf_smem_to_tmem_copy(self.smem_sfb_layout)

    @property
    def uses_n_major_r128c4_sfb_utccp(self) -> bool:
        """Whether SFB UTCCP consumes N-major complete R128c4 row blocks."""
        return (
            self.smem_sfb_layout == int(SfLayout.R128c4)
            and self.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.UTCCP)
            and self.tile_n % 128 == 0
        )

    @property
    def uses_split_r128c4_sfb_tma_desc(self) -> bool:
        """Whether non-routed SFB uses the rank-4 split R128c4 TensorMap."""
        return (
            uses_sfb_tma_desc(self)
            and not uses_routed_sfb_tma_desc(self)
            and (
                self.use_tile256_tmem_overlap
                or (
                    (self.is_mx_mma or self.has_cast_a or self.is_nvfp4_mma)
                    and self.smem_sfb_layout == int(SfLayout.R128c4)
                )
            )
        )

    @property
    def use_combined_sfab_copy(self) -> bool:
        """True for the single-pipeline combined SFA+SFB SMEM->TMEM copy.

        A high-throughput :attr:`is_swap_ab` path (unfused copy, persistent, clustered,
        ``tile_n >= 128``) that issues one CopySfAb warp instead of two TmemSf pipelines.
        """
        if not (
            (self.has_input_block_scaling and not self.fuse_sf_copy_to_mma)
            and self.is_swap_ab
            and self.is_persistent
            and self.uses_two_cta_mma
            and self.sfb_smem_to_tmem_copy != int(SfSmemToTmemCopy.LDS_STTM)
            and self.tile_n >= 128
        ):
            return False

        if not self.has_activation_side_effects and not self.has_routed_sfs:
            return True

        return self.has_routed_sfs and self.uses_ldgsts_routed_sfs

    @property
    def split_b_across_ctas(self) -> bool:
        """True when a clustered kernel stages only a B slice per CTA.

        Two-CTA UMMA partitions B along N. Both routed B resources (LDGSTS and
        TMA gather4) therefore assign ``tile_n / umma_cta_count`` rows to each
        CTA, as do the regular plain and block-scaled paths. Per-stage SMEM,
        TMA transaction bytes, and K-group descriptor strides must use the same
        split. See :attr:`num_bytes_b_smem_per_stage`.
        """
        return self.uses_two_cta_mma and (
            self.uses_plain_mma
            or self.has_input_block_scaling
            or (self.is_swap_ab and self.has_routed_act)
        )

    @property
    def use_tile256_tmem_overlap(self) -> bool:
        """True for a supported ``tile_n == 256`` max-TMEM-overlap path.

        Block-scaled kernels use fused scale-factor copies. Unscaled Batch-M
        BF16 and plain-FP8 kernels share only accumulator columns.
        """
        return (
            (self.has_input_block_scaling or self.has_unscaled_batch_m)
            and self.use_max_tmem_overlap != 0
            and self.tile_n == 256
        )

    @property
    def has_unscaled_batch_m(self) -> bool:
        """Whether plain BF16/FP8 Batch-M can use accumulator-only overlap."""
        return (
            not self.is_swap_ab
            and not self.has_input_block_scaling
            and (self.is_bf16_mma or self.is_fp8_mma)
        )

    @property
    def uses_two_cta_mma(self) -> bool:
        """Whether the logical UMMA instruction atom has multiple CTAs.

        This retains the legacy ``umma_cta_count > 1`` meaning. Physical preferred
        and fallback CGA shapes do not change atom-local layouts, byte counts,
        descriptor partitioning, or CTA_1/CTA_2 instruction selection.
        """
        return self.umma_cta_count > 1

    @property
    def preferred_cluster_shape(self) -> tuple[int, int, int]:
        """Preferred physical CUDA cluster shape."""
        dim_x = self.cluster_dim_x if self.cluster_dim_x > 0 else self.umma_cta_count
        return (dim_x, self.cluster_dim_y, 1)

    @property
    def fallback_cluster_shape(self) -> tuple[int, int, int] | None:
        """Fallback physical CUDA cluster shape, or ``None`` for fixed CGA."""
        if self.fallback_cluster_dim_x <= 0:
            return None
        return (self.fallback_cluster_dim_x, self.fallback_cluster_dim_y, 1)

    @property
    def is_persistent(self) -> bool:
        """True for CLC or queued static persistent scheduling."""
        return self.tile_scheduler in (
            int(TileScheduler.PERSISTENT),
            int(TileScheduler.STATIC_PERSISTENT),
        )

    @property
    def is_static_persistent(self) -> bool:
        """Static cluster-stride scheduling through a local two-stage queue."""
        return self.tile_scheduler == int(TileScheduler.STATIC_PERSISTENT)

    @property
    def raster_along_m(self) -> bool:
        """Whether the persistent CLC scheduler rasterizes along M."""
        return self.cta_raster_order == int(CtaRasterOrder.ALONG_M)

    @property
    def clc_fast_drain_skip_is_suffix(self) -> bool:
        """Whether one early-exit skip implies the rest of the CLC grid is past the limit."""
        # Grouped M rasterization can revisit active M clusters after a skip.
        return self.cta_raster_group_m == 1 and self.is_swap_ab == self.raster_along_m

    @property
    def use_work_throttle_barrier(self) -> bool:
        """True when the persistent work-throttle barrier is active.

        Equals :attr:`is_persistent` and :attr:`use_work_throttle`; paces CLC work-id
        fetch so loads cannot outrun the mainloop.
        """
        return self.is_persistent and bool(self.use_work_throttle)

    @property
    def num_work_throttle_producer_warps(self) -> int:
        """Number of load-task warps that signal the work-throttle barrier (``0`` when inactive)."""
        if not self.use_work_throttle_barrier:
            return 0
        if self.has_gather and not self.is_swap_ab:
            return self.warp_layout.num_load_b_warps
        return self.warp_layout.num_load_a_warps

    @property
    def load_a_task_regs(self) -> int:
        """Resolved register budget for this task."""
        return self.warp_layout.load_a_regs

    @property
    def load_b_task_regs(self) -> int:
        """Resolved register budget for this task."""
        return self.warp_layout.load_b_regs

    @property
    def load_sfa_task_regs(self) -> int:
        """Resolved register budget for this task."""
        return self.warp_layout.load_sfa_regs

    @property
    def load_sfb_task_regs(self) -> int:
        """Resolved register budget for this task."""
        return self.warp_layout.load_sfb_regs

    @property
    def copy_sfa_task_regs(self) -> int:
        """Resolved register budget for this task."""
        return self.warp_layout.copy_sfa_regs

    @property
    def copy_sfb_task_regs(self) -> int:
        """Resolved register budget for this task."""
        return self.warp_layout.copy_sfb_regs

    @property
    def is_swap_ab(self) -> bool:
        """True when A=weights and B=activations."""
        return self.batch_mode == int(BatchMode.BATCH_N)

    @property
    def uses_shuffled_matrix(self) -> bool:
        """Whether the physical weight rows use TRTLLM-gen's shuffle.

        The swap-A/B path has always required shuffled rows; the explicit knob
        controls whether the non-swap path consumes that same representation.
        """
        return self.is_swap_ab or self.use_shuffled_matrix != 0

    @property
    def shuffled_matrix_block_size(self) -> int:
        """Rows in one independent TRTLLM-gen weight-shuffle block."""
        return len(shuffled_matrix_row_indices(self.epi_tile_m))

    @property
    def shuffled_matrix_row_indices(self) -> tuple[int, ...]:
        """Source-row to physical-row permutation for this epilogue tile."""
        return shuffled_matrix_row_indices(self.epi_tile_m)

    @property
    def has_gated_epilogue(self) -> bool:
        """True for gated activations (``SWIGLU``/``GEGLU``/``SITU``);
        the paired output dimension is halved.
        """
        return self.act_kind in (
            int(ActKind.SWIGLU),
            int(ActKind.GEGLU),
            int(ActKind.SITU),
        )

    @property
    def has_eltwise_act(self) -> bool:
        """True for non-gated elementwise activations (``RELU2``/``SILU``)."""
        return self.act_kind in (
            int(ActKind.RELU2),
            int(ActKind.SILU),
        )

    @property
    def cta_group(self):
        """tcgen05 CTA-group kind for MMA/TMEM ops.

        Returns ``CTA_2`` for a 2-CTA cluster (``umma_cta_count >= 2``), else
        ``CTA_1``. Passed to mbarrier descriptor construction; it must agree with
        :attr:`umma_cta_count` or the barrier lowering fails. See :attr:`uses_two_cta_mma`.
        """

        return (
            prims.CTAGroup.CTA_2 if self.umma_cta_count >= 2 else prims.CTAGroup.CTA_1
        )

    @property
    def is_batch_m(self) -> bool:
        """True when experts are batched along M (:attr:`batch_mode` == ``BATCH_M``).

        BatchM is the non-swapped-A/B layout and uses ``tile_coord_m`` for expert
        lookup; BatchN/swapped-A/B uses ``tile_coord_n``.
        """
        return self.batch_mode == int(BatchMode.BATCH_M)

    @property
    def token_tile_size(self) -> int:
        """Token rows covered by one CTA tile."""
        return self.tile_m if self.is_batch_m else self.tile_n

    @property
    def token_cluster_dim(self) -> int:
        """Preferred physical CGA width along the batched token dimension."""
        preferred_x, preferred_y, _ = self.preferred_cluster_shape
        return preferred_x if self.is_batch_m else preferred_y

    @property
    def tile_token(self) -> int:
        """Token rows covered by one CGA.

        Batch-M uses ``tileM * cga_dim.x`` and Batch-N uses
        ``tileN * cga_dim.y``, matching TRTLLM-gen's CGA-granularity routing
        metadata.
        """
        return self.token_tile_size * self.token_cluster_dim

    @property
    def has_gather(self) -> bool:
        """True when activations are gathered via LDGSTS (:attr:`route_act` == ``LDGSTS``)."""
        return self.route_act == int(RouteImpl.LDGSTS)

    @property
    def has_tma_route(self) -> bool:
        """True when activations are gathered via TMA gather4 (:attr:`route_act` == ``TMA``)."""
        return self.route_act == int(RouteImpl.TMA)

    @property
    def has_routed_act(self) -> bool:
        """True when activations are routed by any transport (:attr:`route_act` != ``NONE``)."""
        return self.route_act != int(RouteImpl.NONE)

    @property
    def sf_route(self) -> int:
        """Effective SF transport; NONE always denotes pre-expanded SF storage."""
        if self.route_sfs_act == int(RouteImpl.AUTO):
            return int(
                RouteImpl.LDGSTS
                if self.has_input_block_scaling and self.has_tma_route
                else RouteImpl.NONE
            )
        return self.route_sfs_act

    @property
    def uses_auto_routed_sf_schedule(self) -> bool:
        """Automatic TMA-activation/LDGSTS-SF route uses compact loader widths."""
        return self.route_sfs_act == int(RouteImpl.AUTO) and self.sf_route == int(
            RouteImpl.LDGSTS
        )

    @property
    def use_tma_oob_opt_a(self) -> bool:
        """True when TMA out-of-bounds predication applies to operand A.

        Requires :attr:`use_tma_oob_opt`, no activation routing, and
        :attr:`is_swap_ab` false.
        """
        return (
            self.use_tma_oob_opt == 1
            and not self.has_routed_act
            and not self.is_swap_ab
        )

    @property
    def use_tma_oob_opt_b(self) -> bool:
        """True when TMA out-of-bounds predication applies to operand B.

        Requires :attr:`use_tma_oob_opt`, no activation routing, :attr:`is_swap_ab`,
        and not BF16 MMA.
        """
        return (
            self.use_tma_oob_opt == 1
            and not self.has_routed_act
            and self.is_swap_ab
            and not self.is_bf16_mma
        )

    @property
    def has_routed_sfs(self) -> bool:
        """Whether activation scale factors are gathered instead of pre-expanded."""
        return not self.has_cast_a and self.sf_route != int(RouteImpl.NONE)

    @property
    def uses_ldgsts_routed_sfs(self) -> bool:
        """Whether routed SF use cp.async / LDGSTS."""
        return self.sf_route == int(RouteImpl.LDGSTS)

    @property
    def uses_ldgsts_sf_relay(self) -> bool:
        """Forward CTA-local SF copy completion to the two-CTA consumer."""
        return bool(
            self.uses_two_cta_mma
            and self.has_input_block_scaling
            and self.has_routed_sfs
            and self.uses_ldgsts_routed_sfs
        )

    @property
    def uses_ldgsts_relay(self) -> bool:
        return bool(
            self.uses_two_cta_mma and (self.has_gather or self.uses_ldgsts_sf_relay)
        )

    @property
    def shares_ldgsts_a_sfa_pipeline(self) -> bool:
        """A/SFA can share barriers when MMA owns both matching ring slots."""
        return bool(
            self.uses_ldgsts_sf_relay
            and self.has_gather
            and not self.is_swap_ab
            and self.fuse_sf_copy_to_mma
            and self.num_stages_a == self.num_stages_smem_sfa
        )

    @property
    def uses_tma_routed_sfs(self) -> bool:
        """Whether routed SF use TMA gather4."""
        return self.sf_route == int(RouteImpl.TMA)

    @property
    def uses_sfb_8x4_load(self) -> bool:
        """True when the non-gather B scale-factor tensor uses the compact ``R8c4`` layout.

        Requires ``sf_layout_b == R8c4``, an 8-row aligned low-N tile or
        ``tile_n == 192``, no routed
        :attr:`is_swap_ab` SFB, and no tile256 overlap. Forces
        :attr:`smem_sfb_layout` to ``R8c4``.
        """
        return (
            self.sf_layout_b == int(SfLayout.R8c4)
            and (self.tile_n <= 64 or self.tile_n == 192)
            and self.tile_n % 8 == 0
            and not (self.has_routed_sfs and self.is_swap_ab)
            and not self.use_tile256_tmem_overlap
        )

    @property
    def is_bf16_mma(self) -> bool:
        """True when both operands are BF16 -- standard ``tcgen05`` MMA, not block-scaled."""
        return self.dtype_a == int(DType.BF16) and self.dtype_b == int(DType.BF16)

    @property
    def use_bf16_kbox_tma_a(self) -> bool:
        """True when operand A uses the BF16 K-box TMA descriptor shape.

        For BF16 A, except TMA-routed :attr:`is_swap_ab` false,
        :attr:`use_tma_oob_opt_a`, or
        :attr:`has_cast_a`.
        """
        return (
            self.dtype_a == int(DType.BF16)
            and not (self.has_tma_route and not self.is_swap_ab)
            and not self.use_tma_oob_opt_a
            and not self.has_cast_a
        )

    @property
    def has_bias_m(self) -> bool:
        """True when a per-channel bias is added (:attr:`bias_type` == ``M``)."""
        return self.bias_type == int(BiasType.M)


def uses_routed_sfa_tma_desc(cfg: BatchedGemmConfig) -> bool:
    """Return whether routed SFA needs a TMA gather4 TensorMap descriptor."""
    return cfg.has_routed_sfs and cfg.uses_tma_routed_sfs and not cfg.is_swap_ab


def uses_routed_sfb_tma_desc(cfg: BatchedGemmConfig) -> bool:
    """Return whether routed SFB needs a TMA gather4 TensorMap descriptor."""
    return cfg.has_routed_sfs and cfg.uses_tma_routed_sfs and cfg.is_swap_ab


def uses_sfa_tma_desc(cfg: BatchedGemmConfig) -> bool:
    """Return whether any SFA load reads a TensorMap descriptor."""
    return cfg.has_scale_factor_a and not (
        cfg.has_routed_sfs and cfg.uses_ldgsts_routed_sfs and not cfg.is_swap_ab
    )


def uses_sfb_tma_desc(cfg: BatchedGemmConfig) -> bool:
    """Return whether any SFB load reads a TensorMap descriptor."""
    return cfg.has_scale_factor_b and not (
        cfg.has_routed_sfs and cfg.uses_ldgsts_routed_sfs and cfg.is_swap_ab
    )


def _validate_binary_config_option(name: str, value: int) -> None:
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value}")


def _validate_route_config_option(name: str, value: int) -> None:
    if value in (
        int(RouteImpl.NONE),
        int(RouteImpl.TMA),
        int(RouteImpl.LDGSTS),
    ):
        return
    raise ValueError(f"Unsupported {name}={value}")


@dataclass(frozen=True)
class TaskShape:
    name: str
    count_field: str
    index_field: str


@dataclass(frozen=True)
class _WarpOrder:
    task_order: tuple[str, ...]


_WARP_TASKS = {
    task.name: task
    for task in (
        TaskShape("epilogue", "num_epilogue_warps", "epilogue_warp_idx"),
        TaskShape("cast_a", "num_cast_a_warps", "cast_a_warp_idx_first"),
        TaskShape("load_a", "num_load_a_warps", "load_a_warp_idx"),
        TaskShape("load_b", "num_load_b_warps", "load_b_warp_idx"),
        TaskShape("load_sfa", "num_load_sfa_warps", "load_sfa_warp_idx"),
        TaskShape("load_sfb", "num_load_sfb_warps", "load_sfb_warp_idx"),
        TaskShape("load_sfab", "num_load_sfab_warps", "load_sfab_warp_idx"),
        TaskShape("copy_sfa", "num_copy_sfa_warps", "copy_sfa_warp_idx"),
        TaskShape("copy_sfb", "num_copy_sfb_warps", "copy_sfb_warp_idx"),
        TaskShape("sync", "num_sync_warps", "sync_warp_idx"),
        TaskShape("mma", "num_mma_warps", "mma_warp_idx"),
        TaskShape("workid", "num_workid_warps", "workid_warp_idx"),
    )
}


def _pack_warp_layout(cfg: BatchedGemmConfig, layout: _WarpOrder) -> None:
    for name, requested in cfg._requested_load_warps.items():
        if requested < 0 or requested not in (0, 1, 2, 4, 8):
            raise ValueError(f"{name} must be automatic (0) or 1/2/4/8")
        if requested:
            if getattr(cfg, name) == 0:
                raise ValueError(f"{name} specifies an inactive loader")
            operand = name.removeprefix("num_load_").removesuffix("_warps")
            routed_operand = "b" if cfg.is_swap_ab else "a"
            is_routed = (
                cfg.has_routed_sfs and operand == f"sf{routed_operand}"
                if operand.startswith("sf")
                else cfg.has_routed_act and operand == routed_operand
            )
            if requested != 1 and not is_routed:
                raise ValueError(
                    f"{name} supports one warp for a non-routed TMA loader; "
                    "only routed loaders distribute copies across warps"
                )
            setattr(cfg, name, requested)
    if cfg.fuse_operand_sf_loads:
        for operand in ("a", "b"):
            sf_name = f"num_load_sf{operand}_warps"
            operand_count = getattr(cfg, f"num_load_{operand}_warps")
            requested = cfg._requested_load_warps[sf_name]
            if requested and requested != operand_count:
                raise ValueError(
                    f"Fused {operand.upper()}/SF loader requires equal warp counts"
                )
            setattr(cfg, sf_name, operand_count)
    if cfg.dense_fuse_tma_loads:
        layout = (
            _WarpOrder(
                (
                    "epilogue",
                    "gather",
                    "load_a",
                    "load_sfa",
                    "load_b",
                    "mma",
                    "sync",
                    "workid",
                )
            )
            if cfg.uses_routed_weight_pipeline
            else (
                _WarpOrder(("epilogue", "load_a", "load_b", "mma", "workid"))
                if cfg.dense_split_tma_pipeline
                else _WarpOrder(("epilogue", "load_a", "mma", "workid"))
            )
        )
    warp_idx = 0
    for task_name in layout.task_order:
        routed_task = "load_b" if cfg.is_swap_ab else "load_a"
        if cfg.has_gather and task_name == routed_task:
            continue
        if task_name == "gather":
            if not cfg.has_gather:
                continue
            task_name = routed_task
        task = _WARP_TASKS[task_name]
        setattr(cfg, task.index_field, warp_idx)
        warp_idx += int(getattr(cfg, task.count_field))

    # when using deepseek fp8, add another warp to load the scales
    if cfg.use_deepseek_fp8 and "load_sfab" not in layout.task_order:
        task = _WARP_TASKS["load_sfab"]
        setattr(cfg, task.index_field, warp_idx)
        warp_idx += int(getattr(cfg, task.count_field))

    if cfg.dense_fuse_tma_loads:
        if not cfg.dense_split_tma_pipeline:
            cfg.load_b_warp_idx = cfg.load_a_warp_idx
        if not cfg.uses_routed_weight_pipeline:
            cfg.load_sfa_warp_idx = cfg.load_a_warp_idx
        cfg.load_sfb_warp_idx = cfg.load_b_warp_idx
    # Task-level setmaxnreg requires complete four-warp groups.
    total_warps = ((warp_idx + 3) // 4) * 4
    cfg.padding_warp_idx = warp_idx
    cfg.num_padding_warps = total_warps - warp_idx
    cfg.threads_per_cta = total_warps * WARP_SIZE


def _compute_warp_layout(cfg: BatchedGemmConfig) -> None:
    """Pack private scratch state; the public config stays immutable."""
    requested_load_a_warps = cfg.num_load_a_warps
    requested_load_b_warps = cfg.num_load_b_warps
    requested_load_sfa_warps = cfg.num_load_sfa_warps
    requested_load_sfb_warps = cfg.num_load_sfb_warps

    if cfg.has_deepseek_fp8_two_epilogue:
        cfg.num_epilogue_warps = 8
        cfg.epilogue_warp_idx = 0
        cfg.num_mma_warps = 1
        cfg.num_load_a_warps = 1
        cfg.num_load_b_warps = max(1, requested_load_b_warps)
        cfg.num_load_sfa_warps = 0
        cfg.num_load_sfb_warps = 0
        cfg.num_copy_sfa_warps = 0
        cfg.num_copy_sfb_warps = 0
        cfg.num_cast_a_warps = 0

        cfg.num_sync_warps = 0
        cfg.num_workid_warps = 1 if cfg.is_persistent else 0
        _pack_warp_layout(
            cfg,
            _WarpOrder(
                (
                    "epilogue",
                    "mma",
                    "load_a",
                    "load_b",
                    "load_sfab",
                    "workid",
                )
            ),
        )
        return

    # LDGSTS A needs the general gather-task layout below. TMA A with LDGSTS
    # scales can use this overlap layout, but must reserve its SF relay warp.
    if cfg.use_tile256_tmem_overlap and not cfg.has_gather:
        cfg.num_mma_warps = 1
        cfg.num_copy_sfa_warps = 0
        cfg.num_copy_sfb_warps = 0

        cfg.num_sync_warps = int(cfg.uses_ldgsts_sf_relay)
        cfg.num_workid_warps = 1 if cfg.is_persistent else 0

        cfg.epilogue_warp_idx = 0
        if cfg.has_activation_side_effects:
            if cfg.uses_mxfp8_output_quant:
                cfg.num_epilogue_warps = 4
                cfg.num_load_a_warps = 1
                cfg.num_load_b_warps = 8
                cfg.num_load_sfa_warps = 1
                cfg.num_load_sfb_warps = 4

                _pack_warp_layout(
                    cfg,
                    _WarpOrder(
                        (
                            "epilogue",
                            "load_b",
                            "load_sfb",
                            "load_a",
                            "load_sfa",
                            "sync",
                            "mma",
                            "workid",
                        )
                    ),
                )
                return

            cfg.num_epilogue_warps = 8
            cfg.num_load_a_warps = (
                1
                if cfg.is_swap_ab or not cfg.has_routed_act
                else max(1, requested_load_a_warps)
            )
            cfg.num_load_b_warps = (
                max(1, requested_load_b_warps) if cfg.is_swap_ab else 1
            )
            if cfg.has_input_block_scaling:
                cfg.num_load_sfa_warps = (
                    1 if cfg.is_swap_ab or not cfg.has_routed_sfs else 4
                )
                cfg.num_load_sfb_warps = 4 if cfg.is_swap_ab else 1
            else:
                cfg.num_load_sfa_warps = 0
                cfg.num_load_sfb_warps = 0

            _pack_warp_layout(
                cfg,
                _WarpOrder(
                    (
                        "epilogue",
                        "load_b",
                        "load_sfb",
                        "load_a",
                        "load_sfa",
                        "sync",
                        "mma",
                        "workid",
                    )
                    if cfg.is_swap_ab
                    else (
                        "epilogue",
                        "load_a",
                        "load_sfa",
                        "load_b",
                        "load_sfb",
                        "sync",
                        "mma",
                        "workid",
                    )
                ),
            )
            return

        cfg.num_epilogue_warps = 4
        cfg.num_load_a_warps = 1
        cfg.num_load_b_warps = 1
        cfg.num_load_sfa_warps = int(cfg.has_input_block_scaling)
        cfg.num_load_sfb_warps = int(cfg.has_input_block_scaling)

        _pack_warp_layout(
            cfg,
            _WarpOrder(
                (
                    "epilogue",
                    "load_b",
                    "load_sfb",
                    "load_a",
                    "load_sfa",
                    "sync",
                    "mma",
                    "workid",
                )
            ),
        )
        return

    if cfg.has_cast_a:
        cfg.num_epilogue_warps = 4
        cfg.epilogue_warp_idx = 0
        cfg.num_cast_a_warps = 4
        cfg.cast_a_warp_idx_first = 4
        cfg.num_mma_warps = 1
        cfg.num_load_a_warps = 1
        cfg.num_load_b_warps = 1
        cfg.num_load_sfa_warps = 1
        cfg.num_load_sfb_warps = 0
        cfg.num_copy_sfa_warps = 0
        cfg.num_copy_sfb_warps = 0
        cfg.num_sync_warps = 0
        cfg.num_workid_warps = 1 if cfg.is_persistent else 0

        if cfg.has_gather:
            if cfg.is_swap_ab:
                cfg.num_load_b_warps = 4
            else:
                cfg.num_load_a_warps = 4
        _pack_warp_layout(
            cfg,
            _WarpOrder(
                (
                    "epilogue",
                    "cast_a",
                    "gather",
                    "load_b",
                    "load_a",
                    "load_sfa",
                    "mma",
                    "workid",
                )
            ),
        )
        return

    cfg.num_epilogue_warps = 8 if (cfg.is_fp8_mma and cfg.tile_n == 256) else 4
    if cfg.dense_epilogue_warps == 8:
        cfg.num_epilogue_warps = 8
    cfg.epilogue_warp_idx = 0
    cfg.num_cast_a_warps = 0
    cfg.num_mma_warps = 1

    # Load-warps: gather replaces one TMA load operand.
    cfg.num_load_a_warps = 1
    cfg.num_load_b_warps = 1
    if cfg.has_tma_route:
        if cfg.is_swap_ab:
            if requested_load_b_warps > 1:
                cfg.num_load_b_warps = requested_load_b_warps
            elif cfg.is_fp8_mma and cfg.uses_two_cta_mma:
                cfg.num_load_b_warps = 8
            elif cfg.tile_n == 8 or (cfg.uses_two_cta_mma and cfg.tile_n == 16):
                cfg.num_load_b_warps = 2
            elif cfg.tile_n == 256:
                cfg.num_load_b_warps = 8
            else:
                cfg.num_load_b_warps = 4
        else:
            cfg.num_load_a_warps = 8 if cfg.tile_m == 256 else 4

    if cfg.has_input_block_scaling:
        cfg.num_load_sfa_warps = 1
        cfg.num_load_sfb_warps = 1
        if requested_load_sfa_warps > 1:
            cfg.num_load_sfa_warps = requested_load_sfa_warps
        elif cfg.sfa_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM):
            cfg.num_load_sfa_warps = min(4, max(1, (cfg.tile_m + 3) // 4))
        if requested_load_sfb_warps > 1:
            cfg.num_load_sfb_warps = requested_load_sfb_warps
        elif (
            cfg.has_routed_sfs
            and (cfg.uses_tma_routed_sfs or cfg.uses_auto_routed_sf_schedule)
            and cfg.is_swap_ab
            and cfg.tile_n < 128
        ):
            # TMA-routed compact SFB kernels, and FC1 TMA-route kernels with
            # implicit routed SF, distribute activation-SF rows across the
            # generated LoadSfB warpgroup: tile8=>2, tile16=>4, and
            # tile32/tile64=>8 warps.
            cfg.num_load_sfb_warps = min(8, max(1, cfg.tile_n // 4))
    else:
        cfg.num_load_sfa_warps = 0
        cfg.num_load_sfb_warps = 0

    if cfg.has_gather:
        routed_rows = cfg.tile_n if cfg.is_swap_ab else cfg.tile_m
        # Packed FP4 has twice as many 16-byte copies per gather warp as FP8,
        # so two warps saturate the routed operand while avoiding a larger CTA.
        max_gather_warps = (
            2
            if (
                cfg.is_nvfp4_mma
                and cfg.is_swap_ab
                and cfg.dtype_b_smem_bits == 4
                and not cfg.fuse_operand_sf_loads
            )
            else 4
        )
        routed_warps = min(max_gather_warps, max(1, (routed_rows + 3) // 4))
        if cfg.is_swap_ab:
            cfg.num_load_b_warps = routed_warps
        else:
            cfg.num_load_a_warps = routed_warps

    cfg.num_sync_warps = 1 if cfg.uses_ldgsts_relay else 0
    cfg.num_workid_warps = 1 if cfg.is_persistent else 0
    # CopySf warps copy scale factors to TMEM before the MMA. The compact low-N
    # SFB path needs the 4-warp STTM warpgroup; all bulk S2T paths use one warp
    # to stay inside the launch envelope.
    if cfg.has_input_block_scaling and not cfg.fuse_sf_copy_to_mma:
        cfg.num_copy_sfa_warps = (
            4 if cfg.sfa_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM) else 1
        )
        if cfg.use_combined_sfab_copy:
            cfg.num_copy_sfb_warps = 0
        elif cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM):
            cfg.num_copy_sfb_warps = 4
        else:
            cfg.num_copy_sfb_warps = 1
    else:
        cfg.num_copy_sfa_warps = 0
        cfg.num_copy_sfb_warps = 0

    if cfg.fuse_operand_sf_loads:
        # Gen's high-throughput FC1 layout:
        #   epilogue 0-3, routed B/SFB 4-7, A/SFA 8, MMA 9, WorkId 10,
        #   padding 11.
        _pack_warp_layout(
            cfg,
            _WarpOrder(
                (
                    "epilogue",
                    "gather",
                    "load_a" if cfg.is_swap_ab else "load_b",
                    "mma",
                    "sync",
                    "workid",
                )
            ),
        )
        cfg.load_sfb_warp_idx = cfg.load_b_warp_idx
        cfg.load_sfa_warp_idx = cfg.load_a_warp_idx
        return

    sfb_uses_r8c4_sttm = cfg.sfb_smem_to_tmem_copy == int(
        SfSmemToTmemCopy.LDS_STTM
    ) and cfg.smem_sfb_layout == int(SfLayout.R8c4)
    if sfb_uses_r8c4_sttm:
        # The compact R8c4 SFB path dedicates the first post-epilogue warpgroup
        # to STTM. Keeping CopySfB warpgroup-aligned avoids issuing tcgen05_st
        # from a split warpgroup for low-N block-scaled variants.
        _pack_warp_layout(
            cfg,
            _WarpOrder(
                (
                    "epilogue",
                    "copy_sfb",
                    "gather",
                    "load_b",
                    "load_sfb",
                    "load_a",
                    "load_sfa",
                    "copy_sfa",
                    "sync",
                    "mma",
                    "workid",
                )
            ),
        )
        if cfg.has_activation_side_effects and cfg.tile_n == 64:
            cfg.epilogue_regs = max(cfg.epilogue_regs, 160)
        return

    if cfg.use_combined_sfab_copy:
        # The combined-copy path issues B/SFB first, then A/SFA, then a single
        # CopySfAb warp before MMA. Keeping this order avoids the lowering hang
        # observed with two independent TmemSf pipelines.
        _pack_warp_layout(
            cfg,
            _WarpOrder(
                (
                    "epilogue",
                    "load_b",
                    "load_sfb",
                    "load_a",
                    "load_sfa",
                    "copy_sfa",
                    "copy_sfb",
                    "gather",
                    "sync",
                    "mma",
                    "workid",
                )
            ),
        )
        return

    if (
        cfg.has_tma_route
        and cfg.is_swap_ab
        and cfg.uses_two_cta_mma
        and cfg.tile_n >= 128
        and cfg.num_load_b_warps == 4
        and cfg.num_load_sfb_warps == 4
    ):
        # Keep the two four-warp routed load tasks warpgroup-aligned.  Besides
        # matching the generated high-throughput schedule, this lets LoadB/SFB
        # use their larger register budgets without sharing a warpgroup with
        # the low-register MMA, LoadA, LoadSfA, and WorkId tasks.
        _pack_warp_layout(
            cfg,
            _WarpOrder(
                (
                    "epilogue",
                    "load_b",
                    "load_sfb",
                    "load_a",
                    "load_sfa",
                    "mma",
                    "sync",
                    "workid",
                )
            ),
        )
        return

    if (
        cfg.has_gather
        and cfg.is_swap_ab
        and cfg.umma_cta_count == 1
        and cfg.num_load_b_warps < 4
    ):
        # Low-N LDGSTS order: MMA, TMA-A, LDGSTS gather, then CLC work-id. The
        # condition keys on the task shape (is_swap_ab, 1-CTA, < 4 gather warps),
        # not on a dtype- or tile8-specific special case.
        _pack_warp_layout(
            cfg,
            _WarpOrder(
                (
                    "epilogue",
                    "mma",
                    "load_a",
                    "load_b",
                    "gather",
                    "load_sfa",
                    "load_sfb",
                    "copy_sfa",
                    "copy_sfb",
                    "sync",
                    "workid",
                )
            ),
        )
        return

    # --- pack tasks sequentially after epilogue ---
    if cfg.has_gather:
        default_order = (
            "epilogue",
            "gather",
            "mma",
            "load_a",
            "load_b",
            "load_sfa",
            "load_sfb",
            "copy_sfa",
            "copy_sfb",
            "sync",
            "workid",
        )
    else:
        default_order = (
            "epilogue",
            "mma",
            "load_a",
            "load_b",
            "load_sfa",
            "load_sfb",
            "copy_sfa",
            "copy_sfb",
            "gather",
            "sync",
            "workid",
        )
    _pack_warp_layout(cfg, _WarpOrder(default_order))


class _WarpLayoutBuilder(BatchedGemmConfig):
    """Private scratch state for packing; only WarpLayout escapes resolution."""

    __setattr__ = object.__setattr__

    @property
    def warp_layout(self):
        return self


def compute_warp_layout(cfg: BatchedGemmConfig) -> WarpLayout:
    """Resolve task assignments without modifying the requested configuration."""
    return cfg.warp_layout


def _resolve_warp_layout(cfg: BatchedGemmConfig) -> WarpLayout:
    builder = object.__new__(_WarpLayoutBuilder)
    builder.__dict__.update({f.name: getattr(cfg, f.name) for f in fields(cfg)})
    for f in fields(WarpLayout):
        if f.name not in builder.__dict__:
            setattr(builder, f.name, f.default)
    builder._requested_load_warps = {
        name: getattr(cfg, name)
        for name in (
            "num_load_a_warps",
            "num_load_b_warps",
            "num_load_sfa_warps",
            "num_load_sfb_warps",
        )
    }
    builder.num_load_sfab_warps = int(bool(cfg.use_deepseek_fp8))
    _compute_warp_layout(builder)
    overlap_load_regs = (
        cfg.mma_regs
        if cfg.use_tile256_tmem_overlap and cfg.has_activation_side_effects
        else cfg.load_regs
    )
    overlap_sf_regs = (
        cfg.mma_regs
        if cfg.use_tile256_tmem_overlap and cfg.has_activation_side_effects
        else cfg.load_sf_regs
    )
    builder.load_a_regs = cfg.load_a_regs or overlap_load_regs
    builder.load_b_regs = cfg.load_b_regs or cfg.load_regs
    builder.load_sfa_regs = cfg.load_sfa_regs or overlap_sf_regs
    builder.load_sfb_regs = cfg.load_sfb_regs or cfg.load_sf_regs
    builder.copy_sfa_regs = cfg.copy_sf_regs
    builder.copy_sfb_regs = (
        72
        if cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
        and cfg.smem_sfb_layout == int(SfLayout.R8c4)
        else cfg.copy_sf_regs
    )
    if cfg.fuse_operand_sf_loads:
        builder.load_a_regs = builder.load_sfa_regs = max(
            builder.load_a_regs, builder.load_sfa_regs
        )
        builder.load_b_regs = builder.load_sfb_regs = max(
            builder.load_b_regs, builder.load_sfb_regs
        )
    return WarpLayout(**{f.name: getattr(builder, f.name) for f in fields(WarpLayout)})


def _ldgsts_sfb_base_producer_commit_prefetch_depth(cfg: BatchedGemmConfig) -> int:
    """Return maximum delayed SFB commit depth for this config."""
    if (
        cfg.uses_two_cta_mma
        and cfg.is_swap_ab
        and cfg.tile_n >= 128
        and cfg.sfb_smem_to_tmem_copy != int(SfSmemToTmemCopy.LDS_STTM)
    ):
        return min(
            MAX_PRODUCER_COMMIT_PREFETCH_DEPTH,
            cfg.num_stages_smem_sfb - 1,
        )
    return 0


def _ldgsts_sfb_producer_commit_prefetch_depth(cfg: BatchedGemmConfig) -> int:
    """Return the host-derived delayed SFB commit depth consumed by resources."""
    if cfg.uses_ldgsts_sf_relay:
        # Hardware completion arrivals let the loader fill the ring without
        # delaying commits or draining cp.async groups in the issuing warp.
        return 0
    return min(
        _ldgsts_sfb_base_producer_commit_prefetch_depth(cfg),
        cfg.ldgsts_sfb_producer_commit_prefetch_depth,
    )


def _validate_pipeline_stage_count(cfg: BatchedGemmConfig, field_name: str) -> None:
    """Reject zero-depth pipeline rings before deriving schedule constants."""
    value = getattr(cfg, field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


def _validate_pipeline_stage_counts(cfg: BatchedGemmConfig) -> None:
    """Validate stage counts for pipelines/resources this config instantiates."""
    for field_name in ("num_stages_a", "num_stages_b", "num_stages_tmem_acc"):
        _validate_pipeline_stage_count(cfg, field_name)

    if cfg.use_tma_store != 0:
        _validate_pipeline_stage_count(cfg, "num_stages_c_smem")

    if cfg.has_scale_factor_a or cfg.use_deepseek_fp8:
        _validate_pipeline_stage_count(cfg, "num_stages_smem_sfa")

    if cfg.has_input_block_scaling:
        _validate_pipeline_stage_count(cfg, "num_stages_smem_sfb")

    if cfg.has_input_block_scaling and not cfg.fuse_sf_copy_to_mma:
        _validate_pipeline_stage_count(cfg, "num_stages_tmem_sfa")
        _validate_pipeline_stage_count(cfg, "num_stages_tmem_sfb")

    if (
        cfg.use_combined_sfab_copy
        and cfg.num_stages_tmem_sfa != cfg.num_stages_tmem_sfb
    ):
        raise ValueError("Combined SF copy requires equal TMEM SFA/SFB stage counts")

    if cfg.is_persistent:
        _validate_pipeline_stage_count(cfg, "num_stages_workid")

    if (
        cfg.has_gather
        and cfg.uses_two_cta_mma
        and not cfg.uses_routed_weight_pipeline
        and cfg.num_stages_a != cfg.num_stages_b
    ):
        raise ValueError(
            "2-CTA gather requires num_stages_a == num_stages_b, got "
            f"{cfg.num_stages_a} and {cfg.num_stages_b}"
        )


def _validate_output_store_dtype(cfg: BatchedGemmConfig) -> None:
    """Validate dtype_c against the output store formats implemented by GmemC."""
    if cfg.dtype_c in (int(DType.BF16), int(DType.FP16)):
        return

    if cfg.uses_fp8_output:
        return

    if cfg.uses_fp4_output_quant or cfg.uses_mxfp8_output_quant:
        return

    if cfg.dtype_c == int(DType.E4M3):
        raise ValueError("dtype_c=e4m3 plain FP8 output requires dtype_a=dtype_b=e4m3")

    raise ValueError(
        "Unsupported dtype_c output store: "
        f"{cfg.dtype_c}. Supported dtype_c values are bf16/fp16 plain "
        "stores, e4m3 for FP8 MMA output, and e2m1/mxe2m1/mxe4m3 for output "
        "block scaling."
    )


def resolve_problem_config(cfg: BatchedGemmConfig, problem_k: int) -> BatchedGemmConfig:
    """Return an immutable specialization for host-known K."""
    result = replace(cfg)
    object.__setattr__(result, "_problem_k", problem_k)
    num_k_tiles = (problem_k + cfg.tile_k - 1) // cfg.tile_k
    if cfg.use_unroll_loop_2x_for_mma and (num_k_tiles < 2 or num_k_tiles % 2):
        object.__setattr__(result, "use_unroll_loop_2x_for_mma", 0)
    return result


def validate_config(
    cfg: BatchedGemmConfig,
    problem_mnk: tuple[int, int, int] | None = None,
) -> None:
    """Validate config before kernel launch.  Raises ValueError for invalid
    or unsupported combinations, RuntimeError for untested-but-plausible ones.

    Called at the top of kernel assembly (both validation and GPU paths).
    """
    # ── Hard errors: combinations that are known to fail or are nonsensical ──

    if cfg.batch_mode not in (int(BatchMode.BATCH_N), int(BatchMode.BATCH_M)):
        raise ValueError(f"batch_mode must be BATCH_N or BATCH_M, got {cfg.batch_mode}")

    if cfg.weight_layout_kind not in (
        int(WeightLayout.MajorK),
        int(WeightLayout.BlockMajorK),
    ):
        if cfg.weight_layout_kind == int(WeightLayout.MajorMn):
            raise ValueError("weight_layout=MajorMn is not supported by Prims-TS")
        raise ValueError(f"Unsupported weight_layout={cfg.weight_layout}")

    _validate_binary_config_option(
        "transpose_mma_output",
        cfg.transpose_mma_output,
    )
    _validate_binary_config_option(
        "use_shuffled_matrix",
        cfg.use_shuffled_matrix,
    )
    if cfg.transpose_mma_output != int(cfg.is_swap_ab):
        raise ValueError(
            "transpose_mma_output must match the batch_mode-selected swap_ab "
            f"path, got transpose_mma_output={cfg.transpose_mma_output}, "
            f"batch_mode={BatchMode(cfg.batch_mode).name}"
        )

    if cfg.tile_scheduler not in (
        int(TileScheduler.STATIC),
        int(TileScheduler.PERSISTENT),
        int(TileScheduler.STATIC_PERSISTENT),
    ):
        raise ValueError(
            f"tile_scheduler must be STATIC, PERSISTENT or STATIC_PERSISTENT, got {cfg.tile_scheduler}"
        )

    if cfg.cta_raster_order not in (
        int(CtaRasterOrder.ALONG_M),
        int(CtaRasterOrder.ALONG_N),
    ):
        raise ValueError(
            f"cta_raster_order must be ALONG_M or ALONG_N, got {cfg.cta_raster_order}"
        )
    if cfg.cta_raster_order == int(CtaRasterOrder.ALONG_N) and not cfg.is_persistent:
        raise ValueError("cta_raster_order=ALONG_N requires persistent scheduling")

    if cfg.cta_raster_group_m not in (1, 2, 4, 8):
        raise ValueError("cta_raster_group_m must be 1, 2, 4, or 8")
    if cfg.cta_raster_group_m > 1 and not (
        cfg.is_persistent and not cfg.raster_along_m and cfg.dense_fuse_tma_pipeline
    ):
        raise ValueError(
            "cta_raster_group_m > 1 requires persistent N rasterization "
            "and dense_fuse_tma_pipeline"
        )

    if cfg.umma_cta_count not in (1, 2):
        raise ValueError(
            "umma_cta_count must be 1 or 2; the kernel only implements CTA_1/CTA_2 "
            f"cluster paths, got {cfg.umma_cta_count}"
        )

    if cfg.cluster_dim_x < 0 or cfg.fallback_cluster_dim_x < 0:
        raise ValueError("CGA X dimensions cannot be negative")
    if cfg.fallback_cluster_dim_x == 0 and cfg.fallback_cluster_dim_y != 1:
        raise ValueError("fallback_cluster_dim_y requires an enabled fallback CGA")

    preferred_x, preferred_y, preferred_z = cfg.preferred_cluster_shape
    if preferred_x not in (1, 2, 4) or preferred_y not in (1, 2, 4):
        raise ValueError(
            "physical CGA X/Y dimensions must each be 1, 2, or 4, got "
            f"{cfg.preferred_cluster_shape}"
        )
    if preferred_z != 1 or preferred_x * preferred_y > 16:
        raise ValueError(
            "physical CGA must have Z=1 and at most 16 CTAs, got "
            f"{cfg.preferred_cluster_shape}"
        )
    if cfg.preferred_cluster_shape == (2, 1, 1) and cfg.fallback_cluster_shape:
        raise ValueError(
            "preferred CGA 2x1 requires a fixed launch; disable fallback "
            "with fallback_cluster_dim_x=0"
        )
    from .batched_gemm_cga import has_multiple_umma_atoms, is_flexible_cga

    if has_multiple_umma_atoms(cfg) or is_flexible_cga(cfg):
        if cfg.umma_cta_count != 2 or preferred_x % 2 != 0:
            raise ValueError(
                "wider/flexible CGAs require umma_cta_count=2 and an even physical "
                f"X dimension, got umma_cta_count={cfg.umma_cta_count}, "
                f"shape={cfg.preferred_cluster_shape}"
            )
        if is_flexible_cga(cfg):
            fallback = cfg.fallback_cluster_shape
            assert fallback is not None
            fallback_x, fallback_y, fallback_z = fallback
            if fallback_x not in (2, 4) or fallback_y not in (1, 2, 4):
                raise ValueError(
                    "fallback CGA X must be 2 or 4 and Y must be 1, 2, or 4, "
                    f"got {fallback}"
                )
            if fallback_z != 1 or fallback_x * fallback_y > 16:
                raise ValueError(
                    f"fallback CGA must have Z=1 and at most 16 CTAs, got {fallback}"
                )
            if preferred_x % fallback_x != 0 or preferred_y % fallback_y != 0:
                raise ValueError(
                    "fallback CGA dimensions must divide the preferred CGA "
                    f"dimensions, got preferred={cfg.preferred_cluster_shape}, "
                    f"fallback={fallback}"
                )

    _validate_pipeline_stage_counts(cfg)
    _validate_output_store_dtype(cfg)

    nvfp4_swiglu_output = cfg.dtype_c == int(DType.E2M1) and cfg.act_kind == int(
        ActKind.SWIGLU
    )
    eight_warp_nvfp4_swiglu = (
        nvfp4_swiglu_output
        and cfg.is_rubin
        and cfg.is_nvfp4_mma
        and cfg.uses_non_swap_fp4_gated_tma_c_scratch
    )
    if cfg.dense_epilogue_warps not in (4, 8):
        raise ValueError("dense_epilogue_warps must be 4 or 8")
    if cfg.dense_preload_store_prefix_fragments not in (0, 2):
        raise ValueError("dense_preload_store_prefix_fragments must be 0 or 2")
    if cfg.dense_preload_store_prefix_fragments and not (
        cfg.dense_preload_epilogue and cfg.dense_epilogue_warps == 8
    ):
        raise ValueError(
            "dense_preload_store_prefix_fragments requires "
            "dense_preload_epilogue=1 and dense_epilogue_warps=8"
        )
    if (
        cfg.dense_epilogue_warps == 8
        and cfg.epilogue_regs > 128
        and not cfg.dense_preload_epilogue
        and not eight_warp_nvfp4_swiglu
    ):
        raise ValueError(
            "eight dense epilogue warps currently require epilogue_regs <= 128"
        )
    if cfg.dense_epilogue_warps == 8 and not (
        cfg.dense_rolled_epilogue
        and not cfg.is_swap_ab
        and not cfg.use_max_tmem_overlap
        and (
            eight_warp_nvfp4_swiglu
            or (
                (cfg.dense_single_expert or cfg.dense_preload_epilogue)
                and cfg.dtype_c in (int(DType.FP16), int(DType.BF16))
                and cfg.act_kind == int(ActKind.NONE)
                and cfg.non_swap_tmem_load_num_regs == cfg.tma_store_cols
            )
        )
        and cfg.bias_type == int(BiasType.NONE)
        and not cfg.has_per_token_sf_a
        and not cfg.has_per_token_sf_b
        and not cfg.use_tma_oob_opt
        and cfg.epi_tile_m == 128
    ):
        raise ValueError(
            "eight dense epilogue warps require plain dense or Rubin "
            "NVFP4/SwiGLU rolled TMA output"
        )
    if (
        cfg.dense_epilogue_warps == 8
        and not eight_warp_nvfp4_swiglu
        and (cfg.tile_n // (2 * cfg.non_swap_tmem_load_num_regs))
        % cfg.num_stages_c_smem
    ):
        raise ValueError("eight dense epilogue warps require complete C scratch rings")

    if cfg.has_cast_a and cfg.dtype_c not in (
        int(DType.BF16),
        int(DType.FP16),
    ):
        raise ValueError(
            "CastA MXFP4 input supports dtype_c=bf16/fp16 plain stores only, "
            f"got {cfg.dtype_c}"
        )

    if cfg.dense_m_repeats not in (1, 2):
        raise ValueError("dense_m_repeats must be 1 or 2")
    _validate_binary_config_option("dense_b_reuse", cfg.dense_b_reuse)
    if cfg.dense_b_reuse and cfg.dense_m_repeats != 2:
        raise ValueError("dense_b_reuse requires dense_m_repeats=2")
    if cfg.dense_m_repeats == 2 and not (
        (cfg.dense_fuse_tma_pipeline or cfg.uses_routed_tile512)
        and cfg.is_rubin
        and cfg.is_nvfp4_mma
        and not cfg.is_swap_ab
        and cfg.uses_two_cta_mma
        and cfg.tile_n == cfg.tile_k == 256
        and cfg.fuse_sf_copy_to_mma
        and cfg.smem_sfa_layout == int(SfLayout.R128c4)
        and cfg.smem_sfb_layout == int(SfLayout.R128c4)
        and (
            (
                cfg.dense_rolled_epilogue
                and cfg.dtype_c in (int(DType.FP16), int(DType.BF16))
                and cfg.act_kind == int(ActKind.NONE)
            )
            or nvfp4_swiglu_output
        )
        and (cfg.mma_m, cfg.mma_n, cfg.mma_k) == (256, 256, 128)
        and not cfg.use_max_tmem_overlap
        and cfg.num_stages_tmem_acc == 1
        and cfg.use_tma_store
        and not cfg.use_tma_oob_opt
        and not cfg.has_per_token_sf_a
        and not cfg.has_per_token_sf_b
        and cfg.bias_type == int(BiasType.NONE)
        and cfg.epi_tile_m == 128
        and cfg.dense_async_sf
    ):
        raise ValueError(
            "dense_m_repeats=2 requires Rubin non-swap two-CTA NVFP4 with "
            "a dense shared pipeline or separate routed A/LDGSTS SFA pipelines, "
            "tile_n=tile_k=256, R128c4 scales, fused SF copy/MMA, "
            "MMA256x256x128, one non-overlapping accumulator stage, "
            "async scale copies, and rolled plain FP16/BF16 or NVFP4/SwiGLU TMA output without "
            "bias/per-token scaling (epilogue M128)"
        )

    if cfg.dense_a_l2_promotion not in (0, 128, 256):
        raise ValueError("dense_a_l2_promotion must be 0, 128, or 256")
    if cfg.dense_a_l2_promotion and not (
        cfg.dense_fuse_tma_pipeline or (cfg.uses_routed_tile512 and cfg.has_tma_route)
    ):
        raise ValueError(
            "dense_a_l2_promotion requires dense_fuse_tma_pipeline or routed tile512 TMA A"
        )

    _validate_binary_config_option("dense_retain_a", cfg.dense_retain_a)
    if cfg.dense_retain_a and not (
        cfg.dense_fuse_tma_pipeline or cfg.uses_routed_tile512
    ):
        raise ValueError(
            "dense_retain_a requires dense_fuse_tma_pipeline or routed tile512"
        )

    _validate_binary_config_option("fuse_moe_finalize", cfg.fuse_moe_finalize)
    if cfg.fuse_moe_finalize and not (
        cfg.is_rubin
        and cfg.is_nvfp4_mma
        and not cfg.is_swap_ab
        and not cfg.has_routed_act
        and not cfg.dense_single_expert
        and cfg.dense_preload_epilogue
        and cfg.dense_epilogue_warps in (4, 8)
        and cfg.dtype_c == int(DType.BF16)
        and cfg.uses_global_scales
    ):
        raise ValueError(
            "fuse_moe_finalize requires native Rubin NVFP4 BatchM FC2, "
            "four/eight-warp preload, BF16 output and global scales"
        )

    _validate_binary_config_option("dense_streaming_c", cfg.dense_streaming_c)
    if cfg.dense_streaming_c and not cfg.dense_preload_epilogue:
        raise ValueError("dense_streaming_c requires dense_preload_epilogue")

    _validate_binary_config_option("dense_preload_epilogue", cfg.dense_preload_epilogue)
    # Resolve explicit schedule controls even when preload is disabled, so
    # unsupported knobs cannot be silently ignored by a different epilogue.
    resolve_preload_epilogue(cfg)
    if cfg.dense_preload_epilogue and not (
        cfg.sm_arch == 107
        and not cfg.is_swap_ab
        and cfg.tile_n == 256
        and cfg.dense_m_repeats == 2
        and cfg.dense_epilogue_warps in (4, 8)
        and (
            cfg.dense_dkg_store_order or (cfg.epilogue_steps and cfg.has_gated_epilogue)
        )
        and (
            (cfg.dtype_c in (int(DType.FP16), int(DType.BF16)) and cfg.epi_tile_n == 64)
            or (
                cfg.epilogue_steps
                and cfg.has_gated_epilogue
                and cfg.uses_non_swap_fp4_gated_tma_c_scratch
                and cfg.epi_tile_n == 128
            )
        )
        and cfg.tmem_ldst_max_num_regs == 64
        and cfg.num_stages_c_smem
        in ((1, 2) if cfg.epilogue_steps or cfg.dense_epilogue_warps == 8 else (2,))
        and cfg.epilogue_regs
        in ((144, 160, 168, 176) if cfg.dense_epilogue_warps == 8 else (248, 256))
    ):
        raise ValueError(
            "dense_preload_epilogue requires SM107 tile512, four or eight epilogue warps, "
            "FP16/BF16 output (or explicit group1 NVFP4 FC1 steps), N64 fragments, DKG store order, two C stages for four "
            "warps or one/two for eight, "
            "248/256 registers for four warps or 144/160/168/176 for eight"
        )

    if cfg.dense_m_repeats == 2:
        # Generic expert lookup expresses the SFB expert stride in N tiles.
        # A partial final tile would select the wrong expert's scale rows.
        if not cfg.dense_single_expert and problem_mnk is not None:
            _, problem_n, _ = problem_mnk
            if isinstance(problem_n, int) and problem_n % cfg.tile_n != 0:
                raise ValueError(
                    "tile512 with generic expert lookup requires problem N to be "
                    f"a multiple of tile_n={cfg.tile_n} for weight scale indexing, "
                    f"got {problem_n}"
                )
        operand_bytes = (
            cfg.num_bytes_a_per_stage * cfg.num_stages_a
            + cfg.num_bytes_b_smem_per_stage * cfg.num_stages_b
            + cfg.num_bytes_sfa_per_stage * cfg.num_stages_smem_sfa
            + cfg.num_bytes_sfb_per_stage * cfg.num_stages_smem_sfb
        )
        if operand_bytes + cfg.num_bytes_c_smem_scratch > cfg.smem_capacity_bytes:
            raise ValueError("tile512 operand and epilogue stages exceed SMEM capacity")

    if cfg.sm_arch not in (100, 103, 107):
        raise ValueError(f"sm_arch must be 100, 103, or 107, got {cfg.sm_arch}")

    if cfg.mma_k not in (16, 32, 64, 128):
        raise ValueError(
            "mma_k must be 16 (BF16), 32/64 (MX/FP8), or 64/128 "
            f"(NVFP4), got {cfg.mma_k}"
        )

    if cfg.is_bf16_mma and cfg.mma_k != 16:
        raise ValueError(f"BF16 MMA requires mma_k=16, got {cfg.mma_k}")

    if cfg.has_cast_a and cfg.mma_k != 16:
        raise ValueError(f"CastA BF16 MMA requires mma_k=16, got {cfg.mma_k}")

    if cfg.is_nvfp4_mma and cfg.mma_k not in (64, 128):
        raise ValueError(f"NVFP4 MMA requires mma_k=64 or 128, got {cfg.mma_k}")

    if cfg.is_nvfp4_mma and cfg.mma_k == 128 and not cfg.is_rubin:
        raise ValueError("NVFP4 mma_k=128 requires sm_arch=107")

    if cfg.is_nvfp4_mma and cfg.mma_k == 128 and cfg.tile_n != cfg.mma_n:
        raise ValueError(
            "NVFP4 mma_k=128 requires tile_n == mma_n so the Rubin scale-factor "
            f"layout remains contiguous, got tile_n={cfg.tile_n}, mma_n={cfg.mma_n}"
        )

    if cfg.is_mx_mma and cfg.mma_k not in (32, 64):
        raise ValueError(f"MX MMA requires mma_k=32 or 64, got {cfg.mma_k}")

    if cfg.is_mx_mma and cfg.mma_k == 64 and not cfg.is_rubin:
        raise ValueError("MX mma_k=64 requires sm_arch=107")

    if (
        cfg.is_mx_mma
        and cfg.mma_k == 64
        and cfg.dtype_a == int(DType.MXE2M1)
        and cfg.dtype_b == int(DType.MXE2M1)
    ):
        raise ValueError(
            "MXE2M1xMXE2M1 does not use the Rubin MX mma_k=64 instruction; "
            "its MxFp4NvFp4 mma_k=128 path is not implemented"
        )

    if cfg.is_fp8_mma and cfg.mma_k not in (32, 64):
        raise ValueError(f"FP8 MMA requires mma_k=32 or 64, got {cfg.mma_k}")

    if cfg.is_fp8_mma and cfg.mma_k == 64:
        if not cfg.is_rubin:
            raise ValueError("FP8 mma_k=64 requires sm_arch=107")
        if cfg.mma_m % 128 != 0:
            raise ValueError(
                "FP8 mma_k=64 requires mma_m to be a multiple of 128, "
                f"got mma_m={cfg.mma_m}"
            )

    if cfg.tile_k % cfg.mma_k != 0:
        raise ValueError(
            f"tile_k must be a multiple of mma_k, got tile_k={cfg.tile_k}, "
            f"mma_k={cfg.mma_k}"
        )

    if cfg.is_nvfp4_mma and cfg.tile_k % 256 != 0:
        raise ValueError(
            f"NVFP4 requires tile_k to be a multiple of 256, got tile_k={cfg.tile_k}"
        )

    if cfg.uses_block_major_k_weight:
        block_k = cfg.block_major_k_elems
        tile_block_k = cfg.block_major_k_tile_block_elems
        if cfg.tile_k % tile_block_k != 0:
            raise ValueError(
                "BlockMajorK requires tile_k to divide into whole TMA K-block "
                f"slices, got tile_k={cfg.tile_k}, tile_block_k={tile_block_k}"
            )
        if cfg.tile_k < block_k and block_k % cfg.tile_k != 0:
            raise ValueError(
                "BlockMajorK tile_k must either cover whole K blocks or "
                f"divide one block evenly, got tile_k={cfg.tile_k}, "
                f"block_k={block_k}, block_bytes={cfg.block_major_k_bytes}"
            )
        if problem_mnk is not None:
            _, _, problem_k = problem_mnk
            if isinstance(problem_k, int) and problem_k % block_k != 0:
                raise ValueError(
                    "BlockMajorK requires problem K to be a multiple of the "
                    f"K block, got problem_k={problem_k}, block_k={block_k}, "
                    f"block_bytes={cfg.block_major_k_bytes}"
                )

    if cfg.has_cast_a and cfg.tile_k != 256:
        raise ValueError(f"CastA MXFP4 input requires tile_k=256, got {cfg.tile_k}")

    if cfg.use_bf16_kbox_tma_a and cfg.tile_k % 64 != 0:
        raise ValueError(
            "BF16 k-box TMA A path requires tile_k to be a multiple of 64, "
            f"got tile_k={cfg.tile_k}"
        )

    if cfg.is_nvfp4_mma and (
        cfg.input_sf_block_size_a != 16 or cfg.input_sf_block_size_b != 16
    ):
        raise ValueError("NVFP4 input block scaling requires sf_block_size_a/b=16")

    if cfg.is_mx_mma and (
        cfg.input_sf_block_size_a != 32 or cfg.input_sf_block_size_b != 32
    ):
        raise ValueError(
            "MX input block scaling requires sf_block_size_a/b=32, got "
            f"{cfg.input_sf_block_size_a}/{cfg.input_sf_block_size_b}"
        )

    # The MX SFA/SFB TMA descriptors move whole 4-atom scale-factor K blocks;
    # a partial block truncates the box K extent (to 0 for tile_k=64), which
    # cuTensorMapEncodeTiled rejects at launch.
    if cfg.is_mx_mma:
        sf_pack_k = TMEM_SF_PACK_SIZE_BYTES * cfg.input_sf_block_size_a
        if cfg.tile_k % sf_pack_k != 0:
            raise ValueError(
                f"MX block scaling requires tile_k to be a multiple of {sf_pack_k}, "
                f"got tile_k={cfg.tile_k}"
            )

    problem_k = cfg._problem_k
    if problem_mnk is not None and isinstance(problem_mnk[2], int):
        problem_k = problem_mnk[2]
    if problem_k is not None and not cfg.use_deepseek_fp8:
        for operand, has_scale_factor, sf_block_size in (
            ("A", cfg.has_scale_factor_a, cfg.input_sf_block_size_a),
            ("B", cfg.has_scale_factor_b, cfg.input_sf_block_size_b),
        ):
            sf_pack_k = TMEM_SF_PACK_SIZE_BYTES * sf_block_size
            if has_scale_factor and problem_k % sf_pack_k != 0:
                num_scale_factors = (problem_k + sf_block_size - 1) // sf_block_size
                raise ValueError(
                    f"K dimension of scaling factors for {operand} "
                    f"({num_scale_factors}) must be a multiple of "
                    f"{TMEM_SF_PACK_SIZE_BYTES} (problem_k={problem_k}, "
                    f"sf_block_size={sf_block_size}); pad K to a multiple "
                    f"of {sf_pack_k}"
                )

    if cfg.uses_mx_output_quant and cfg.output_sf_block_size_c != 32:
        raise ValueError(
            f"MX output block scaling requires sf_block_size_c=32, got {cfg.output_sf_block_size_c}"
        )

    if (
        cfg.uses_fp4_output_quant
        and not cfg.uses_mxfp4_output_quant
        and cfg.output_sf_block_size_c != 16
    ):
        raise ValueError(
            f"NVFP4 output block scaling requires sf_block_size_c=16, got {cfg.output_sf_block_size_c}"
        )

    if cfg.has_cast_a:
        if cfg.input_sf_block_size_a != 32:
            raise ValueError(
                "CastA MXFP4 input requires sf_block_size_a=32, got "
                f"{cfg.input_sf_block_size_a}"
            )
        if cfg.sf_layout_a != int(SfLayout.R128c4):
            raise ValueError(
                f"CastA MXFP4 input currently requires sf_layout_a=128x4, got {cfg.sf_layout_a}"
            )
        if cfg.tmem_cast_a_cols != 128:
            raise ValueError(
                f"CastA TMEM staging expects 128 columns per stage, got {cfg.tmem_cast_a_cols}"
            )

    if cfg.mma_m not in (64, 128, 256):
        raise ValueError(f"mma_m must be 64/128/256, got {cfg.mma_m}")

    if cfg.mma_n not in (8, 16, 32, 64, 128, 192, 256):
        raise ValueError(f"mma_n must be 8/16/32/64/128/192/256, got {cfg.mma_n}")

    if cfg.tile_n % cfg.mma_n != 0:
        raise ValueError(
            f"tile_n must be a multiple of mma_n, got tile_n={cfg.tile_n}, "
            f"mma_n={cfg.mma_n}"
        )

    cluster_tile_m = cfg.tile_m * cfg.umma_cta_count
    if cluster_tile_m % cfg.mma_m != 0:
        raise ValueError(
            "cluster-wide tile_m must be a multiple of mma_m, got "
            f"tile_m={cfg.tile_m}, umma_cta_count={cfg.umma_cta_count}, mma_m={cfg.mma_m}"
        )

    if cfg.umma_cta_count > 1 and cfg.mma_n < 16:
        raise ValueError(
            f"umma_cta_count={cfg.umma_cta_count} requires mma_n >= 16, got mma_n={cfg.mma_n}"
        )

    if cfg.umma_cta_count == 2 and cfg.mma_m != 256:
        raise ValueError(f"umma_cta_count=2 requires mma_m=256, got mma_m={cfg.mma_m}")

    if cfg.is_nvfp4_mma and cfg.mma_m == 64:
        raise ValueError("NVFP4 MMA does not support mma_m=64")

    if cfg.tile_n not in (8, 16, 32, 64, 128, 192, 256):
        raise ValueError(f"tile_n must be 8/16/32/64/128/192/256, got {cfg.tile_n}")

    if cfg.epi_tile_n <= 0:
        raise ValueError(f"epi_tile_n must be positive, got {cfg.epi_tile_n}")

    if cfg.epi_tile_n % 8 != 0:
        raise ValueError(f"epi_tile_n must be a multiple of 8, got {cfg.epi_tile_n}")

    # Dense tile512 drains one 128-row epilogue fragment per M repeat.
    # All other Batch-M paths still require a full tile in one fragment.
    if not cfg.is_swap_ab and cfg.epi_tile_m * cfg.dense_m_repeats != cfg.tile_m:
        raise ValueError(
            "Batch-M epilogues store whole tile_m-row tiles; epi_tile_m must "
            "equal tile_m / dense_m_repeats, got "
            f"epi_tile_m={cfg.epi_tile_m}, tile_m={cfg.tile_m}, "
            f"dense_m_repeats={cfg.dense_m_repeats}"
        )

    if cfg.use_shuffled_matrix and not cfg.is_swap_ab:
        shuffle_block = cfg.shuffled_matrix_block_size
        if cfg.tile_n % shuffle_block != 0:
            raise ValueError(
                "Batch-M use_shuffled_matrix requires tile_n to contain whole "
                f"shuffle blocks of {shuffle_block}, got tile_n={cfg.tile_n}"
            )
        if cfg.non_swap_tmem_load_num_regs % shuffle_block != 0:
            raise ValueError(
                "Batch-M use_shuffled_matrix requires each TMEM fragment to "
                f"contain whole shuffle blocks of {shuffle_block}, got "
                f"load_regs={cfg.non_swap_tmem_load_num_regs}"
            )
        if problem_mnk is not None:
            _, problem_n, _ = problem_mnk
            if isinstance(problem_n, int) and problem_n % shuffle_block != 0:
                raise ValueError(
                    "Batch-M use_shuffled_matrix requires problem N to be a "
                    f"multiple of {shuffle_block}, got {problem_n}"
                )

    if not cfg.is_swap_ab and problem_mnk is not None:
        _, problem_n, _ = problem_mnk
        if isinstance(problem_n, int):
            output_n = problem_n // 2 if cfg.has_gated_epilogue else problem_n
            required_output_multiple = 128 // cfg.dtype_c_bits
            if output_n % required_output_multiple != 0:
                raise ValueError(
                    "Batch-M output rows must be 16-byte aligned; "
                    f"output width {output_n} with {cfg.dtype_c_bits}-bit elements "
                    f"must be a multiple of {required_output_multiple}. Pad problem N."
                )

    if (
        not cfg.is_swap_ab
        and cfg.has_gated_epilogue
        and cfg.epi_tile_n % cfg.non_swap_gated_accumulator_cols_per_step != 0
    ):
        raise ValueError(
            "Non-swap gated epilogue requires the epilogue tile width to be "
            "divisible by the accumulator columns drained per register step, got "
            f"epi_tile_n={cfg.epi_tile_n}, "
            f"step_cols={cfg.non_swap_gated_accumulator_cols_per_step}"
        )
    if cfg.gated_weight_group_size not in (1, 64):
        raise ValueError("gated_weight_group_size must be 1 or 64")
    if cfg.gated_weight_group_size == 64 and not (
        not cfg.is_swap_ab
        and cfg.dtype_c == int(DType.E2M1)
        and cfg.act_kind == int(ActKind.SWIGLU)
        and cfg.epi_tile_n == 128
        and cfg.non_swap_tmem_load_num_regs == 64
        and cfg.non_swap_gated_output_cols_per_call == 64
        and (
            cfg.num_epilogue_warpgroups == 1
            or (cfg.dense_epilogue_warps == 8 and cfg.dense_rolled_epilogue)
        )
    ):
        raise ValueError(
            "gated_weight_group_size=64 requires non-swap NVFP4/SwiGLU, "
            "epi_tile_n=128, two 64-register TMEM fragments, four epilogue "
            "warps (or eight with the rolled epilogue)"
        )
    gated_output_cols_limit = (
        MAX_NON_SWAP_NVFP4_GATED_OUTPUT_COLS_PER_CALL
        if cfg.uses_fp4_output_quant and not cfg.uses_mx_output_quant
        else MAX_NON_SWAP_GATED_OUTPUT_COLS_PER_CALL
    )
    if (
        not cfg.is_swap_ab
        and cfg.has_gated_epilogue
        and cfg.non_swap_gated_output_cols_per_call > gated_output_cols_limit
    ):
        raise ValueError(
            "Non-swap gated epilogue supports at most "
            f"{gated_output_cols_limit} output columns per "
            f"call, got {cfg.non_swap_gated_output_cols_per_call} "
            f"(epi_tile_n={cfg.epi_tile_n}, "
            f"load_regs={cfg.non_swap_tmem_load_num_regs})"
        )
    if (
        not cfg.is_swap_ab
        and cfg.has_gated_epilogue
        and cfg.use_tma_store
        and (cfg.epi_tile_n // 2) % cfg.tma_store_cols != 0
    ):
        raise ValueError(
            "Non-swap gated TMA stores require the gated epilogue tile width "
            "to be divisible by the TMA box width, got "
            f"output_cols={cfg.epi_tile_n // 2}, "
            f"tma_store_cols={cfg.tma_store_cols}"
        )
    if (
        not cfg.is_swap_ab
        and cfg.has_gated_epilogue
        and cfg.use_tma_store
        and cfg.tma_store_cols % cfg.non_swap_gated_output_cols_per_call != 0
    ):
        raise ValueError(
            "Non-swap gated TMA stores require the per-step output width to "
            "divide the TMA box, got "
            f"step_output_cols={cfg.non_swap_gated_output_cols_per_call}, "
            f"tma_store_cols={cfg.tma_store_cols}"
        )
    if (
        not cfg.is_swap_ab
        and cfg.has_gated_epilogue
        and cfg.use_max_tmem_overlap
        and cfg.non_swap_gated_steps_per_epi_tile > 1
    ):
        raise ValueError(
            "Non-swap gated max-TMEM-overlap requires one register step per "
            "epilogue tile, got "
            f"{cfg.non_swap_gated_steps_per_epi_tile} steps "
            f"(epi_tile_n={cfg.epi_tile_n}, "
            f"step_cols={cfg.non_swap_gated_accumulator_cols_per_step})"
        )
    if (
        not cfg.is_swap_ab
        and cfg.has_gated_epilogue
        and cfg.has_output_block_scaling
        and cfg.use_tma_store
        and cfg.non_swap_gated_steps_per_epi_tile > 1
    ):
        raise ValueError(
            "Non-swap block-scaled-output (NVFP4/MX) TMA does not support a "
            "streamed gated epilogue tile; use direct GMEM stores or one "
            "register step, got "
            f"{cfg.non_swap_gated_steps_per_epi_tile} steps"
        )

    if cfg.tile_n % cfg.epi_tile_n != 0:
        raise ValueError(
            f"epi_tile_n must divide tile_n, got tile_n={cfg.tile_n}, "
            f"epi_tile_n={cfg.epi_tile_n}"
        )
    if cfg.is_swap_ab and cfg.has_output_block_scaling and not cfg.has_gated_epilogue:
        output_cols_per_call = cfg.epi_tile_n * cfg.num_epilogue_warpgroups
        if cfg.tile_n % output_cols_per_call != 0:
            raise ValueError(
                "Non-gated swapAB output block scaling requires "
                "epi_tile_n * num_epilogue_warpgroups to divide tile_n, got "
                f"tile_n={cfg.tile_n}, epi_tile_n={cfg.epi_tile_n}, "
                f"epilogue_warpgroups={cfg.num_epilogue_warpgroups}"
            )

    # batch_mode selects swap_ab and the expert-coordinate axis. The generated
    # transpose flag is retained but validated above against that kernel path.

    _validate_binary_config_option("weight_l2_evict_first", cfg.weight_l2_evict_first)
    if cfg.weight_l2_evict_first:
        if not cfg.is_swap_ab:
            raise ValueError("weight_l2_evict_first requires swap-AB (BatchN)")
        if cfg.token_cluster_dim != 1:
            raise ValueError("weight_l2_evict_first requires token-cluster width one")

    if not 0 <= cfg.fused_finalize_top_k <= 32:
        raise ValueError(
            f"fused_finalize_top_k must be in [0, 32], got {cfg.fused_finalize_top_k}"
        )
    if cfg.fused_finalize_top_k and not (
        cfg.is_swap_ab
        and cfg.use_tma_store
        and cfg.dtype_c == int(DType.BF16)
        and cfg.tile_m == 128
        and cfg.token_cluster_dim == 1
        and not cfg.has_activation_epilogue
        and not cfg.use_deepseek_fp8
        and not cfg.has_output_block_scaling
        and not cfg.uses_fp8_output
        and not cfg.has_per_token_sf_a
        and not cfg.has_per_token_sf_b
    ):
        raise ValueError(
            "fused_finalize_top_k needs a swap-AB BF16 TMA-store epilogue with "
            "tile_m 128, token-cluster width one and no activation, per-token "
            "scales, output quantization or DeepSeek FP8 scaling"
        )

    if cfg.use_max_tmem_overlap not in (0, 1):
        raise ValueError(
            f"use_max_tmem_overlap must be 0 or 1, got {cfg.use_max_tmem_overlap}"
        )
    if cfg.fuse_sf_copy_to_mma not in (0, 1):
        raise ValueError(
            f"fuse_sf_copy_to_mma must be 0 or 1, got {cfg.fuse_sf_copy_to_mma}"
        )
    if cfg.fuse_sf_copy_to_mma != 0 and not cfg.has_input_block_scaling:
        raise ValueError("fuse_sf_copy_to_mma requires block-scaled MMA operands")
    for name in (
        "routed_native_sfa",
        "routed_low_smem",
        "routed_sfa_first",
        "fused_a_single_commit",
    ):
        _validate_binary_config_option(name, getattr(cfg, name))
    if cfg.routed_sfa_first and (not cfg.routed_native_sfa or cfg.routed_low_smem):
        raise ValueError(
            "routed_sfa_first requires routed_native_sfa=1 and routed_low_smem=0"
        )
    if cfg.fused_a_single_commit and not (
        cfg.fuse_operand_sf_loads
        and not cfg.is_swap_ab
        and cfg.shares_ldgsts_a_sfa_pipeline
    ):
        raise ValueError(
            "fused_a_single_commit requires fused non-swap LDGSTS A/SFA with equal stage depths"
        )
    if cfg.fuse_operand_sf_loads and not cfg.is_swap_ab and not cfg.routed_native_sfa:
        raise ValueError("non-swap fused operand/SF loaders require routed_native_sfa")
    if cfg.routed_native_sfa and not (
        cfg.is_rubin
        and cfg.is_nvfp4_mma
        and not cfg.is_swap_ab
        and cfg.has_routed_act
        and cfg.uses_ldgsts_routed_sfs
        and (
            (cfg.fuse_operand_sf_loads and cfg.fused_a_single_commit)
            or (
                cfg.route_act == int(RouteImpl.TMA)
                and not cfg.fuse_operand_sf_loads
                and not cfg.fused_a_single_commit
            )
        )
        and cfg.fuse_sf_copy_to_mma
        and cfg.dense_m_repeats in (1, 2)
        and (cfg.tile_n, cfg.tile_k, cfg.mma_k) == (256, 256, 128)
        and (cfg.mma_m, cfg.mma_n) == (256, 256)
        and cfg.sf_layout_a == cfg.sf_layout_b == int(SfLayout.R128c4)
        and cfg.preferred_cluster_shape[0] in (2, 4)
        and cfg.preferred_cluster_shape[1] in (1, 2, 4)
        and cfg.preferred_cluster_shape[2] == 1
        and (
            cfg.fallback_cluster_shape is None
            or (
                cfg.has_tma_route
                and not cfg.is_static_persistent
                and cfg.fallback_cluster_shape == (2, 1, 1)
            )
        )
        and cfg.warp_layout.num_load_a_warps
        in ((4, 8) if cfg.route_act == int(RouteImpl.TMA) else (4,))
        and cfg.warp_layout.num_load_sfa_warps == 4
        and (
            (cfg.dtype_c == int(DType.BF16) and cfg.act_kind == int(ActKind.NONE))
            or nvfp4_swiglu_output
        )
        and not cfg.use_max_tmem_overlap
        and cfg.is_persistent
        # CLC MoE launches reserve the worst-case routed-token extent. Skip
        # unused routing records before native loaders read their expert IDs.
        and (not cfg.use_early_exit or not cfg.is_static_persistent)
        and not cfg.use_pdl
    ):
        raise ValueError(
            "routed_native_sfa requires Rubin NVFP4 tile256/tile512 with a 2xY or 4xY CGA (Y=1/2/4; optional 2x1 fallback for TMA-A with CLC), four fused LDGSTS A/SFA warps or four/eight TMA A warps with four native SFA warps, BF16/no activation or NVFP4/SwiGLU output, and persistent scheduling without PDL (early exit requires CLC)"
        )
    if cfg.is_static_persistent and not (
        cfg.routed_native_sfa
        and not cfg.use_work_throttle
        and cfg.num_stages_workid == 2
        and isinstance(cfg.static_persistent_clusters, int)
        and cfg.static_persistent_clusters > 0
    ):
        raise ValueError(
            "STATIC_PERSISTENT requires routed_native_sfa, two work stages, no work throttle and a positive static_persistent_clusters budget"
        )
    if cfg.routed_native_sfa and problem_mnk is not None:
        _, problem_n, problem_k = problem_mnk
        if isinstance(problem_n, int) and problem_n % cfg.tile_n:
            raise ValueError(
                "routed_native_sfa requires problem_n to be a multiple of "
                "tile_n=256 for weight scale indexing"
            )
        if isinstance(problem_k, int) and problem_k % cfg.tile_k:
            raise ValueError(
                "routed_native_sfa requires problem_k to be a multiple of "
                "tile_k=256 for aligned 16-byte scale rows"
            )
    if cfg.routed_low_smem and not (
        cfg.routed_native_sfa
        and (
            cfg.fuse_operand_sf_loads
            or (cfg.has_tma_route and cfg.uses_routed_weight_pipeline)
        )
        and cfg.dense_m_repeats == 1
        and cfg.num_stages_a
        == cfg.num_stages_b
        == cfg.num_stages_smem_sfa
        == cfg.num_stages_smem_sfb
        == 8
        and cfg.num_stages_tmem_acc == 2
        and (
            cfg.num_stages_c_smem == 1
            or (
                cfg.uses_non_swap_fp4_gated_tma_c_scratch
                and 1 <= cfg.num_stages_c_smem <= 4
            )
        )
        and cfg.num_bytes_c_smem_scratch <= 16384
        and cfg.num_stages_workid == (2 if cfg.is_static_persistent else 3)
    ):
        raise ValueError(
            "routed_low_smem requires native tile256 with fused LDGSTS A/SFA or TMA-A/shared weights, eight operand stages, two accumulator stages, one C scratch stage (up to four for gated NVFP4, within 16 KiB) and two static/three CLC work stages"
        )
    _validate_binary_config_option("dense_rolled_mma", cfg.dense_rolled_mma)
    if cfg.dense_rolled_mma and not (
        (
            cfg.dense_fuse_tma_loads
            or cfg.uses_routed_tile512
            or cfg.fuse_operand_sf_loads
        )
        and cfg.fuse_sf_copy_to_mma
        and not cfg.use_max_tmem_overlap
    ):
        raise ValueError(
            "dense_rolled_mma requires fused dense loaders or routed tile512 "
            "without max overlap"
        )
    _validate_binary_config_option("dense_async_sf", cfg.dense_async_sf)
    if cfg.dense_async_sf and not (
        cfg.fuse_sf_copy_to_mma
        and (cfg.dense_fuse_tma_loads or cfg.uses_routed_tile512)
    ):
        raise ValueError(
            "dense_async_sf requires fused SF copy/MMA and dense loaders or routed tile512"
        )
    _validate_binary_config_option("dense_local_barrier", cfg.dense_local_barrier)
    if cfg.dense_local_barrier and not (
        (cfg.dense_fuse_tma_loads or (cfg.routed_native_sfa and cfg.has_tma_route))
        and cfg.dense_specialize_tma_cga
    ):
        raise ValueError(
            "dense_local_barrier requires fused loaders or native-SFA TMA-A, "
            "with static CGA specialization"
        )
    _validate_binary_config_option("dense_dkg_store_order", cfg.dense_dkg_store_order)
    if cfg.dense_dkg_store_order and not cfg.dense_rolled_epilogue:
        raise ValueError("dense_dkg_store_order requires the rolled dense epilogue")
    if cfg.dense_dkg_store_order and cfg.has_gated_epilogue:
        raise ValueError("dense_dkg_store_order requires non-gated output")
    _validate_binary_config_option("dense_rolled_epilogue", cfg.dense_rolled_epilogue)
    if cfg.dense_rolled_epilogue and not (
        (
            cfg.dense_fuse_tma_pipeline
            or cfg.uses_routed_tile512
            or (cfg.routed_native_sfa and nvfp4_swiglu_output)
        )
        and not cfg.uses_tmem_overlap_window
        and (
            not cfg.has_gated_epilogue
            or (nvfp4_swiglu_output and cfg.uses_non_swap_fp4_gated_tma_c_scratch)
        )
        and cfg.use_tma_store
    ):
        raise ValueError(
            "dense_rolled_epilogue requires non-overlap TMA output: a non-gated "
            "dense epilogue or non-swap dense/native-routed NVFP4/SwiGLU"
        )
    _validate_binary_config_option("dense_fuse_tma_loads", cfg.dense_fuse_tma_loads)
    if cfg.dense_fuse_tma_loads and cfg.fuse_operand_sf_loads:
        raise ValueError(
            "dense_fuse_tma_loads and fuse_operand_sf_loads select competing B/SFB producers"
        )
    if cfg.dense_fuse_tma_loads and not (
        (cfg.dense_fuse_tma_pipeline or cfg.uses_routed_weight_pipeline)
        and cfg.dense_partition_tma_loads
    ):
        raise ValueError(
            "dense_fuse_tma_loads requires partitioned loads and either the dense "
            "shared pipeline or routed tile512 split weight pipeline"
        )
    _validate_binary_config_option(
        "dense_specialize_tma_cga", cfg.dense_specialize_tma_cga
    )
    if cfg.dense_specialize_tma_cga and not (
        (
            cfg.dense_fuse_tma_pipeline
            or cfg.uses_routed_tile512
            or (cfg.routed_native_sfa and cfg.has_tma_route)
        )
        and cfg.fallback_cluster_shape == (2, 1, 1)
    ):
        raise ValueError(
            "dense_specialize_tma_cga requires the dense shared pipeline or "
            "routed tile512 or native-SFA TMA-A, with a 2x1 fallback"
        )
    _validate_binary_config_option("dense_single_expert", cfg.dense_single_expert)
    if cfg.dense_single_expert and not cfg.dense_fuse_tma_pipeline:
        raise ValueError(
            "dense_single_expert requires the non-routed dense shared pipeline"
        )
    _validate_binary_config_option(
        "dense_partition_tma_loads", cfg.dense_partition_tma_loads
    )
    if cfg.dense_partition_tma_loads:
        # Native paired loaders also support separate B/SFB barriers.
        if not (
            cfg.dense_fuse_tma_pipeline
            or cfg.uses_routed_weight_pipeline
            or (
                cfg.fuse_operand_sf_loads
                and not cfg.is_swap_ab
                and cfg.routed_native_sfa
            )
        ):
            raise ValueError(
                "dense_partition_tma_loads requires dense_fuse_tma_pipeline or "
                "a routed weight pipeline or fused native routed loaders"
            )
        if (
            cfg.tile_m not in (128, 256)
            or cfg.preferred_cluster_shape[1] not in (1, 2, 4)
            or cfg.fallback_cluster_shape not in (None, (2, 1, 1))
            or cfg.use_tma_oob_opt
        ):
            raise ValueError(
                "dense_partition_tma_loads requires tile_m=128/256, Y=1/2/4, "
                "fallback 2x1, no TMA OOB optimization"
            )
    _validate_binary_config_option(
        "dense_fuse_tma_pipeline", cfg.dense_fuse_tma_pipeline
    )
    if cfg.dense_fuse_tma_pipeline:
        if not (
            cfg.is_rubin
            and cfg.is_nvfp4_mma
            and not cfg.is_swap_ab
            and not cfg.has_routed_act
            and not cfg.has_routed_sfs
            and cfg.uses_two_cta_mma
            and cfg.tile_n == 256
            and cfg.tile_k == 256
            and cfg.fuse_sf_copy_to_mma
            and cfg.smem_sfa_layout == int(SfLayout.R128c4)
            and cfg.smem_sfb_layout == int(SfLayout.R128c4)
            and cfg.preferred_cluster_shape[0] in (2, 4)
            and (cfg.preferred_cluster_shape[0] == 2 or cfg.dense_fuse_tma_loads)
            and not cfg.use_pdl
        ):
            raise ValueError(
                "dense_fuse_tma_pipeline requires Rubin non-routed batch-M NVFP4 "
                "tile_n=tile_k=256, R128c4 scale factors, 2xY CGAs "
                "(or 4xY with dense_fuse_tma_loads=1), "
                "fuse_sf_copy_to_mma=1, and use_pdl=0"
            )
        if cfg.dense_split_tma_pipeline:
            if (
                cfg.num_stages_a != cfg.num_stages_smem_sfa
                or cfg.num_stages_b != cfg.num_stages_smem_sfb
            ):
                raise ValueError("split pipelines require matching operand/SF depths")
        elif (
            len(
                {
                    cfg.num_stages_a,
                    cfg.num_stages_b,
                    cfg.num_stages_smem_sfa,
                    cfg.num_stages_smem_sfb,
                }
            )
            != 1
        ):
            raise ValueError(
                "dense_fuse_tma_pipeline requires equal A/B/SFA/SFB stages"
            )
    _validate_binary_config_option(
        "dense_split_tma_pipeline", cfg.dense_split_tma_pipeline
    )
    if cfg.dense_split_tma_pipeline and (
        (not cfg.dense_fuse_tma_loads and not cfg.fuse_operand_sf_loads)
        or not cfg.is_persistent
        or cfg.use_work_throttle
    ):
        raise ValueError(
            "dense_split_tma_pipeline requires fused loaders, persistent scheduling, and no work throttling"
        )
    if (
        cfg.fuse_operand_sf_loads
        and cfg.uses_routed_weight_pipeline
        and not cfg.dense_partition_tma_loads
    ):
        raise ValueError("paired routed weights require partitioned SFB multicast")
    if cfg.uses_routed_weight_pipeline and cfg.num_stages_b != cfg.num_stages_smem_sfb:
        raise ValueError("routed weight pipeline requires matching B/SFB depths")
    if (
        cfg.partitions_routed_tma_a
        and cfg.fallback_cluster_shape
        and not cfg.dense_specialize_tma_cga
    ):
        raise ValueError(
            "partitioned routed TMA A requires static CGA specialization with a fallback"
        )
    _validate_binary_config_option(
        "fuse_operand_sf_loads",
        cfg.fuse_operand_sf_loads,
    )
    if cfg.fuse_operand_sf_loads:
        if not (
            cfg.has_input_block_scaling
            and cfg.has_gather
            and cfg.uses_two_cta_mma
            and cfg.uses_ldgsts_routed_sfs
            and cfg.tile_n >= 128
        ):
            raise ValueError(
                "fuse_operand_sf_loads requires clustered LDGSTS "
                "activation and scale-factor routing with tile_n >= 128"
            )
        if not cfg.fuse_sf_copy_to_mma:
            raise ValueError("fuse_operand_sf_loads requires fuse_sf_copy_to_mma=1")
        if cfg.warp_layout.num_load_sfa_warps != cfg.warp_layout.num_load_a_warps:
            raise ValueError(
                "fuse_operand_sf_loads requires num_load_sfa_warps == "
                "num_load_a_warps, got "
                f"{cfg.warp_layout.num_load_sfa_warps} and {cfg.warp_layout.num_load_a_warps}"
            )
        if cfg.warp_layout.num_load_sfb_warps != cfg.warp_layout.num_load_b_warps:
            raise ValueError(
                "fuse_operand_sf_loads requires num_load_sfb_warps == "
                "num_load_b_warps, got "
                f"{cfg.warp_layout.num_load_sfb_warps} and {cfg.warp_layout.num_load_b_warps}"
            )
        sfb_prefetch_depth = _ldgsts_sfb_producer_commit_prefetch_depth(cfg)
        if sfb_prefetch_depth >= cfg.num_stages_b:
            raise ValueError(
                "fuse_operand_sf_loads requires the delayed SFB commit depth "
                "to be smaller than the fused B pipeline depth, got "
                f"prefetch_depth={sfb_prefetch_depth}, "
                f"num_stages_b={cfg.num_stages_b}"
            )
    if cfg.tmem_ldst_max_num_regs not in (32, 64):
        raise ValueError(
            f"tmem_ldst_max_num_regs must be 32 or 64, got {cfg.tmem_ldst_max_num_regs}"
        )

    if cfg.use_tma_store and cfg.epi_tile_n % cfg.tma_store_cols != 0:
        raise ValueError(
            "epi_tile_n must be divisible by the layout-appropriate TMA store "
            f"N width, got epi_tile_n={cfg.epi_tile_n}, "
            f"tma_store_cols={cfg.tma_store_cols}"
        )
    if (
        cfg.use_tma_store
        and not cfg.is_swap_ab
        and cfg.c_tma_swizzle_bytes < TMA_SWIZZLE_ATOM_BYTES
    ):
        raise ValueError(
            "Non-swap TMA output requires at least one complete "
            f"{TMA_SWIZZLE_ATOM_BYTES}-byte transaction, got "
            f"tma_store_cols={cfg.tma_store_cols} and "
            f"dtype_c_bits={cfg.dtype_c_bits}"
        )

    if cfg.use_tma_oob_opt not in (0, 1):
        raise ValueError(f"use_tma_oob_opt must be 0 or 1, got {cfg.use_tma_oob_opt}")

    if cfg.use_early_exit not in (0, 1):
        raise ValueError(f"use_early_exit must be 0 or 1, got {cfg.use_early_exit}")

    if cfg.use_clc_fast_drain not in (0, 1):
        raise ValueError(
            f"use_clc_fast_drain must be 0 or 1, got {cfg.use_clc_fast_drain}"
        )
    if cfg.use_clc_fast_drain and (not cfg.is_persistent or not cfg.use_early_exit):
        raise ValueError(
            "use_clc_fast_drain requires persistent scheduling and use_early_exit=1"
        )

    if cfg.use_work_throttle not in (0, 1):
        raise ValueError(
            f"use_work_throttle must be 0 or 1, got {cfg.use_work_throttle}"
        )

    if cfg.record_runtime_cga_shapes not in (0, 1):
        raise ValueError(
            "record_runtime_cga_shapes must be 0 or 1, got "
            f"{cfg.record_runtime_cga_shapes}"
        )

    _validate_binary_config_option(
        "use_unroll_loop_2x_for_mma",
        cfg.use_unroll_loop_2x_for_mma,
    )
    if cfg.use_unroll_loop_2x_for_mma and cfg.use_deepseek_fp8:
        raise ValueError(
            "use_unroll_loop_2x_for_mma is not supported by the DeepSeek FP8 "
            "per-K accumulator pipeline"
        )

    _validate_binary_config_option(
        "use_full_tmem_dealloc_barrier",
        cfg.use_full_tmem_dealloc_barrier,
    )

    if cfg.use_pdl not in (0, 1):
        raise ValueError(f"use_pdl must be 0 or 1, got {cfg.use_pdl}")

    if cfg.skip_early_exit_pdl not in (0, 1):
        raise ValueError(
            f"skip_early_exit_pdl must be 0 or 1, got {cfg.skip_early_exit_pdl}"
        )
    if cfg.skip_early_exit_pdl and not (cfg.use_pdl and cfg.use_early_exit):
        raise ValueError("skip_early_exit_pdl requires use_pdl=1 and use_early_exit=1")

    if cfg.use_global_scales not in (0, 1):
        raise ValueError(
            f"use_global_scales must be 0 or 1, got {cfg.use_global_scales}"
        )

    if cfg.use_per_token_sf_a not in (0, 1):
        raise ValueError(
            f"use_per_token_sf_a must be 0 or 1, got {cfg.use_per_token_sf_a}"
        )

    if cfg.use_per_token_sf_b not in (0, 1):
        raise ValueError(
            f"use_per_token_sf_b must be 0 or 1, got {cfg.use_per_token_sf_b}"
        )

    for name, value in (
        ("has_gemm1_alpha", cfg.has_gemm1_alpha),
        ("has_gemm1_beta", cfg.has_gemm1_beta),
        ("has_gemm1_clamp_limit", cfg.has_gemm1_clamp_limit),
    ):
        _validate_binary_config_option(name, value)

    if cfg.has_gemm1_activation_params and cfg.act_kind not in (
        int(ActKind.SWIGLU),
        int(ActKind.SITU),
    ):
        raise ValueError("GEMM1 activation params require act_kind=SWIGLU or SITU")

    if cfg.per_token_sf_dtype not in (
        int(DType.FP32),
        int(DType.BF16),
        int(DType.FP16),
    ):
        raise ValueError(
            "per_token_sf_dtype must be FP32, BF16, or FP16, got "
            f"{cfg.per_token_sf_dtype}"
        )

    if cfg.use_per_token_sf_a != 0 and not cfg.is_fp8_mma:
        raise ValueError("Per-token sf_a is only valid for plain FP8 MMA")

    if cfg.use_per_token_sf_b != 0:
        if not (cfg.is_fp8_mma or cfg.is_nvfp4_mma):
            raise ValueError("Per-token sf_b is only valid for plain FP8 or NVFP4 MMA")
        if cfg.has_bias_m and cfg.is_fp8_mma:
            raise ValueError(
                "FP8 per-token sf_b with bias is parsed but not implemented"
            )

    if cfg.use_deepseek_fp8 not in (0, 1):
        raise ValueError(f"use_deepseek_fp8 must be 0 or 1, got {cfg.use_deepseek_fp8}")

    if cfg.use_deepseek_fp8:
        if not cfg.is_fp8_mma:
            raise ValueError(
                "use_deepseek_fp8=1 requires E4M3 MMA (dtype_a=dtype_b=E4M3)"
            )
        block = cfg.deepseek_fp8_block_size
        if block not in (32, 128):
            raise ValueError(f"deepseek_fp8_block_size must be 32 or 128, got {block}")
        if cfg.tile_k != 128:
            raise ValueError(
                "use_deepseek_fp8=1 requires tile_k=128 "
                f"for DeepSeek scale-factor blocks, got {cfg.tile_k}"
            )
        if block != 128 and not cfg.has_deepseek_fp8_two_epilogue:
            raise ValueError(
                f"deepseek_fp8_block_size={block} requires the swapAB split-M "
                "epilogue (batch_mode=BATCH_N, mma_m=64)"
            )
        if cfg.has_gated_epilogue:
            raise ValueError(
                "use_deepseek_fp8=1 is incompatible with fused gated activation"
            )
        if cfg.use_max_tmem_overlap != 0:
            raise ValueError(
                "use_deepseek_fp8=1 is incompatible with use_max_tmem_overlap"
            )
        if cfg.umma_cta_count > 1:
            raise ValueError("use_deepseek_fp8=1 is incompatible with cluster m > 1")
        if cfg.has_input_block_scaling:
            raise ValueError(
                "use_deepseek_fp8=1 is incompatible with MX/NVFP4 input block scaling"
            )
        if cfg.use_per_token_sf_a != 0 or cfg.use_per_token_sf_b != 0:
            raise ValueError(
                "use_deepseek_fp8=1 is incompatible with use_per_token_sf_a/b"
            )
        if cfg.warp_layout.num_load_sfab_warps != 1:
            raise ValueError(
                "use_deepseek_fp8=1 requires num_load_sfab_warps=1, "
                f"got {cfg.warp_layout.num_load_sfab_warps}"
            )
        if cfg.has_deepseek_fp8_c_scale and not cfg.is_swap_ab:
            raise ValueError(
                "DeepSeek FP8 dtype_c=e4m3 C-scale output requires is_swap_ab"
            )
        if cfg.deepseek_fp8_epi_subtile_cnt != 1:
            raise ValueError(
                "use_deepseek_fp8=1 requires epi_subtile_cnt=1 because the "
                "current DeepSeek FP8 epilogue stores one output-N subtile, "
                f"got {cfg.deepseek_fp8_epi_subtile_cnt} "
                f"(tile_n={cfg.tile_n}, epi_tile_n={cfg.epi_tile_n}, "
                f"num_epilogue_warps={cfg.warp_layout.num_epilogue_warps})"
            )

    if cfg.use_tma_store not in (0, 1):
        raise ValueError(f"use_tma_store must be 0 or 1, got {cfg.use_tma_store}")

    if (
        cfg.uses_deepseek_fp8_split_epilogue_tma_c_scratch
        and cfg.num_stages_c_smem != 1
    ):
        raise ValueError(
            "DeepSeek FP8 split-epilogue TMA C scratch requires "
            f"num_stages_c_smem=1, got {cfg.num_stages_c_smem}"
        )

    supported_sf_layouts = {
        int(SfLayout.R8c4),
        int(SfLayout.LINEAR),
        int(SfLayout.R128c4),
    }
    if cfg.sf_layout_a not in supported_sf_layouts:
        raise ValueError(f"Unsupported sf_layout_a: {cfg.sf_layout_a}")
    if cfg.sf_layout_b not in supported_sf_layouts:
        raise ValueError(f"Unsupported sf_layout_b: {cfg.sf_layout_b}")
    if cfg.sf_layout_c not in supported_sf_layouts:
        raise ValueError(f"Unsupported sf_layout_c: {cfg.sf_layout_c}")

    if cfg.has_input_block_scaling:
        routed_sf_operand = "b" if cfg.is_swap_ab else "a"
        for operand, layout in (("a", cfg.sf_layout_a), ("b", cfg.sf_layout_b)):
            routed = cfg.has_routed_sfs and operand == routed_sf_operand
            if layout == int(SfLayout.LINEAR) and not routed:
                raise ValueError(
                    f"sf_layout_{operand}=linear requires routed input scale factors "
                    "(route_sfs_act=tma or ldgsts); non-routed LINEAR SF has no "
                    "matching TMA descriptor or LDS+STTM pipeline. Use 8x4 or 128x4."
                )

    if cfg.has_input_block_scaling and cfg.sf_layout_a == int(SfLayout.R8c4):
        raise ValueError(
            "sf_layout_a=8x4 is not supported for input SF loads; "
            "8x4 load support is only for the non-gather B tensor with an "
            "8-row aligned tile_n <= 64 or tile_n == 192"
        )

    if (
        cfg.has_input_block_scaling
        and cfg.sf_layout_b == int(SfLayout.R8c4)
        and not cfg.uses_sfb_8x4_load
        and not (cfg.has_routed_sfs and cfg.is_swap_ab)
    ):
        raise ValueError(
            "sf_layout_b=8x4 requires non-gather B scale-factor loading with "
            "an 8-row aligned tile_n <= 64, tile_n == 192, or routed SFB"
        )

    if (
        cfg.has_input_block_scaling
        and cfg.sf_layout_b == int(SfLayout.R128c4)
        and cfg.tile_n % 128 != 0
        and not (cfg.has_routed_sfs and cfg.is_swap_ab)
    ):
        raise ValueError(
            "sf_layout_b=128x4 requires tile_n to be a multiple of 128 for "
            "non-routed B scale-factor loading"
        )

    uses_legacy_sfa_tma_desc = (
        uses_sfa_tma_desc(cfg)
        and not uses_routed_sfa_tma_desc(cfg)
        and not (
            cfg.use_tile256_tmem_overlap
            or cfg.is_mx_mma
            or cfg.has_cast_a
            or cfg.is_nvfp4_mma
        )
    )
    if uses_legacy_sfa_tma_desc and cfg.tile_m * 4 // 2 > 256:
        raise ValueError(
            "Legacy SFA TensorMap box exceeds the per-dimension limit of 256: "
            f"tile_m * 4 / 2 = {cfg.tile_m * 4 // 2}. "
            "Use a supported split R128c4/R8c4 scale-factor descriptor."
        )

    uses_legacy_sfb_tma_desc = (
        uses_sfb_tma_desc(cfg)
        and not uses_routed_sfb_tma_desc(cfg)
        and not cfg.uses_sfb_8x4_load
        and not cfg.uses_split_r128c4_sfb_tma_desc
    )
    if uses_legacy_sfb_tma_desc and cfg.tile_n * 4 // 2 > 256:
        raise ValueError(
            "Legacy SFB TensorMap box exceeds the per-dimension limit of 256: "
            f"tile_n * 4 / 2 = {cfg.tile_n * 4 // 2}. "
            "Use use_max_tmem_overlap or a supported split R128c4/R8c4 "
            "scale-factor descriptor."
        )

    if cfg.has_input_block_scaling and cfg.fuse_sf_copy_to_mma:
        fused_sf_layouts = {
            "SFA": cfg.smem_sfa_layout,
            "SFB": cfg.smem_sfb_layout,
        }
        unsupported_fused_layouts = [
            name
            for name, layout in fused_sf_layouts.items()
            if layout != int(SfLayout.R128c4)
        ]
        if unsupported_fused_layouts:
            raise ValueError(
                "fused SF copy requires effective R128c4 SMEM layout for "
                + "/".join(unsupported_fused_layouts)
            )

    if cfg.has_output_block_scaling:
        if not cfg.has_input_block_scaling:
            raise ValueError("Output block scaling requires FP4/MX input scale factors")
        if not (cfg.uses_fp4_output_quant or cfg.uses_mxfp8_output_quant):
            raise ValueError(
                "Output block scaling requires dtype_c=e2m1, mxe2m1, or mxe4m3"
            )
        if not cfg.has_gated_epilogue:
            if not cfg.is_swap_ab:
                raise ValueError(
                    "Non-gated output block scaling currently requires swapAB"
                )
            if cfg.act_kind not in (
                int(ActKind.RELU2),
                int(ActKind.SILU),
            ):
                raise ValueError(
                    "Non-gated output block scaling currently requires "
                    "act_kind=RELU2 or SILU"
                )
            if cfg.dtype_c not in (int(DType.E2M1), int(DType.MXE4M3)):
                raise ValueError(
                    "Non-gated output block scaling currently requires "
                    "dtype_c=e2m1 or mxe4m3"
                )
            if cfg.epi_tile_m != cfg.tile_m:
                raise ValueError(
                    "Non-gated output block scaling requires epi_tile_m == "
                    f"tile_m, got epi_tile_m={cfg.epi_tile_m}, "
                    f"tile_m={cfg.tile_m}"
                )
            if problem_mnk is not None:
                problem_m, _, _ = problem_mnk
                if (
                    isinstance(problem_m, int)
                    and problem_m % cfg.output_sf_block_size_c != 0
                ):
                    raise ValueError(
                        "Non-gated output block scaling requires problem_m to "
                        "contain whole output scale blocks, got "
                        f"problem_m={problem_m}, "
                        f"sf_block_size_c={cfg.output_sf_block_size_c}"
                    )
        if not cfg.is_swap_ab and cfg.dtype_c != int(DType.E2M1):
            raise ValueError(
                "Non-swap output block scaling currently requires dtype_c=e2m1"
            )
        if (
            not cfg.is_swap_ab
            and (cfg.non_swap_tmem_load_num_regs // 2) % cfg.output_sf_block_size_c != 0
        ):
            raise ValueError(
                "Non-swap output block scaling requires each TMEM load to contain "
                "a whole output scale block, got "
                f"load_regs={cfg.non_swap_tmem_load_num_regs}, "
                f"sf_block_size_c={cfg.output_sf_block_size_c}"
            )
        if cfg.sf_layout_c not in (int(SfLayout.R8c4), int(SfLayout.R128c4)):
            raise ValueError("Output block scaling requires sf_layout_c 8x4 or 128x4")

    if (
        cfg.uses_fp8_output
        and not cfg.has_activation_epilogue
        and not cfg.has_deepseek_fp8_c_scale
    ):
        raise ValueError("Plain FP8 C output requires an activation-side epilogue")

    if cfg.use_max_tmem_overlap != 0:
        if cfg.has_input_block_scaling and not cfg.fuse_sf_copy_to_mma:
            raise ValueError("use_max_tmem_overlap requires fuse_sf_copy_to_mma=1")
        if not cfg.use_tile256_tmem_overlap:
            raise ValueError(
                "use_max_tmem_overlap is only supported for block-scaled or "
                "unscaled Batch-M tile_n=256 kernels"
            )
        supported_tile_k = (
            (64,)
            if cfg.is_bf16_mma
            else ((128, 256) if cfg.is_mx_mma or cfg.is_fp8_mma else (256,))
        )
        if (
            cfg.tile_k not in supported_tile_k
            or cfg.umma_cta_count != 2
            or not cfg.is_persistent
        ):
            raise ValueError(
                "use_max_tmem_overlap currently supports the generated persistent "
                "2-CTA tile256 shape only, got "
                f"tile_k={cfg.tile_k}, umma_cta_count={cfg.umma_cta_count}, "
                f"persistent={cfg.is_persistent}"
            )
        if cfg.num_stages_tmem_acc != 1:
            raise ValueError(
                "use_max_tmem_overlap requires one accumulator TMEM stage, got "
                f"num_stages_tmem_acc={cfg.num_stages_tmem_acc}"
            )
        if cfg.has_input_block_scaling and (
            cfg.tmem_sfa_cols % TMEM_SF_UTCCP_COLS_PER_COPY != 0
            or cfg.tmem_sfb_cols % TMEM_SF_UTCCP_COLS_PER_COPY != 0
        ):
            raise ValueError(
                "use_max_tmem_overlap requires fused scale-factor footprints "
                f"to be multiples of {TMEM_SF_UTCCP_COLS_PER_COPY} TMEM "
                f"columns, got sfa_cols={cfg.tmem_sfa_cols}, "
                f"sfb_cols={cfg.tmem_sfb_cols}"
            )
        overlap_loads = cfg.tmem_overlap_loads
        if overlap_loads not in (1, 2):
            raise ValueError(
                "use_max_tmem_overlap supports one or two T2R loads for the "
                "shared TMEM region, got "
                f"{overlap_loads} (swap_ab={cfg.is_swap_ab}, "
                f"epi_tile_n={cfg.epi_tile_n}, "
                f"tmem_load_regs={cfg.tmem_load_num_regs})"
            )

    if cfg.tmem_required_cols > cfg.tmem_capacity_columns:
        raise ValueError(
            f"TMEM footprint requires {cfg.tmem_required_cols} columns, "
            f"which exceeds the {cfg.tmem_capacity_columns}-column tcgen05 "
            f"allocation limit for {cfg.arch_name}"
        )

    _validate_route_config_option("route_act", cfg.route_act)
    if cfg.route_sfs_act != int(RouteImpl.AUTO):
        _validate_route_config_option("route_sfs_act", cfg.route_sfs_act)

    if cfg.bias_type not in (int(BiasType.NONE), int(BiasType.M)):
        raise ValueError(f"Unsupported bias_type={cfg.bias_type}")
    if cfg.has_bias_m and problem_mnk is not None:
        problem_m, _, _ = problem_mnk
        if isinstance(problem_m, int) and problem_m > 0:
            bias_block_size = 32 if cfg.epi_tile_m % 128 == 0 else 16
            if problem_m % bias_block_size != 0:
                raise ValueError(
                    "BiasType.M uses TRT-LLM Gen shuffleMatrixA storage and "
                    "requires problem M to be a multiple of "
                    f"{bias_block_size}, got {problem_m}"
                )

    route_sfs_explicit = cfg.route_sfs_act not in (
        int(RouteImpl.NONE),
        int(RouteImpl.AUTO),
    )

    # Register overrides belong to operand loaders for every transport. The
    # task manager validates register allocation for the resolved warp layout.
    for name in ("load_a_regs", "load_b_regs", "load_sfa_regs", "load_sfb_regs"):
        if getattr(cfg, name) < 0:
            raise ValueError(f"{name} must be automatic (0) or a positive budget")

    # Explicit NONE uses pre-expanded SF, even with routed activations.
    if route_sfs_explicit and cfg.route_act == int(RouteImpl.NONE):
        raise ValueError(
            "route_sfs_act requires route_act != NONE; cannot gather SF without "
            "gathering activations"
        )

    if route_sfs_explicit and not cfg.has_input_block_scaling:
        raise ValueError(
            "route_sfs_act requires FP4/MX input block scale factors; no routed "
            "SF resource is instantiated for this dtype combination"
        )

    # SF TMA gather4 requires at least 32 SF elements per tile for the routed
    # operand. Use LDGSTS when tile_k is too small for the operand SF block size.
    if cfg.has_routed_sfs and cfg.uses_tma_routed_sfs:
        routed_sf_operand = "b" if cfg.is_swap_ab else "a"
        routed_sf_block_size = (
            cfg.input_sf_block_size_b if cfg.is_swap_ab else cfg.input_sf_block_size_a
        )
        sf_k = cfg.tile_k // routed_sf_block_size
        if sf_k < 32:
            raise ValueError(
                "SF TMA gather4 requires sf_k >= 32 for the routed activation "
                f"SF operand, got sf_{routed_sf_operand} with sf_k={sf_k} "
                f"for tile_k={cfg.tile_k} and "
                f"input_sf_block_size_{routed_sf_operand}={routed_sf_block_size}. "
                "Use route_sfs_act=LDGSTS."
            )
        # Only the first LoadSf warp arms the transaction barrier. A warp with
        # no gather4 rows never gates the full barrier, so its lag is bounded
        # only by the work-id ring. Two laps behind (two phases of the same
        # empty barrier), its one-bit parity wait aliases onto a phase that
        # never completes once the CTA runs out of work.
        routed_sf_rows = cfg.tile_n if cfg.is_swap_ab else cfg.tile_m
        num_routed_sf_gather4 = routed_sf_rows // 4
        num_routed_sf_load_warps = (
            cfg.warp_layout.num_load_sfb_warps
            if cfg.is_swap_ab
            else cfg.warp_layout.num_load_sfa_warps
        )
        if num_routed_sf_load_warps > num_routed_sf_gather4:
            raise ValueError(
                "SF TMA gather4 requires every load warp to own at least one "
                f"gather4 row group, got num_load_sf{routed_sf_operand}_warps="
                f"{num_routed_sf_load_warps} for {num_routed_sf_gather4} groups "
                f"(tile rows {routed_sf_rows} / 4)."
            )

    if cfg.has_gather:
        activation_operand = "b" if cfg.is_swap_ab else "a"
        activation_dtype_bits = cfg.dtype_b_bits if cfg.is_swap_ab else cfg.dtype_a_bits
        if activation_dtype_bits not in (4, 8, 16):
            raise ValueError(
                "LDGSTS activation gather requires a supported packed or "
                "byte-addressable activation "
                f"elements, got dtype_{activation_operand} with "
                f"{activation_dtype_bits} bits."
            )
        if (cfg.tile_k * activation_dtype_bits) % 128 != 0:
            raise ValueError(
                "LDGSTS activation gather requires each row tile to contain an "
                f"integer number of 16-byte copies, got tile_k={cfg.tile_k} "
                f"and dtype_{activation_operand}={activation_dtype_bits} bits."
            )

    if cfg.act_kind not in (
        int(ActKind.NONE),
        int(ActKind.SWIGLU),
        int(ActKind.GEGLU),
        int(ActKind.RELU2),
        int(ActKind.SILU),
        int(ActKind.SITU),
    ):
        raise ValueError(f"Unsupported act_kind={cfg.act_kind}")

    # Validate requested loader widths even on paths whose geometry did not
    # otherwise need the layout during validation. Resolution is immutable.
    _ = cfg.warp_layout

    # Activation is valid for any kernel (FC1 convention, but not restricted).


def _normalize_config_overrides(overrides: dict) -> dict:
    """Validate that all overrides target BatchedGemmConfig fields."""
    result = dict(overrides)

    valid_fields = {field.name for field in fields(BatchedGemmConfig) if field.init}
    unknown = sorted(key for key in result if key not in valid_fields)
    if unknown:
        raise TypeError(f"Unknown BatchedGemmConfig option(s): {', '.join(unknown)}")

    return result


def make_config(**overrides) -> BatchedGemmConfig:
    """Patch helper for tests / sweep runs."""
    overrides = _normalize_config_overrides(overrides)
    return BatchedGemmConfig(**overrides)


def compute_max_num_ctas_in_token_dim_for_moe(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    token_tile_size: int,
    cluster_dim_in_token: int,
) -> int:
    """Compute the worst-case MoE token-CTA launch extent.

    The generated testbed first gives one token to as many experts as possible,
    then packs the remaining routed tokens into one expert. This maximizes CGA
    count for a fixed ``num_tokens * top_k``. TS early-exit indexes token work
    at CTA granularity, so convert the CGA count by the token-axis cluster width.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k must be in [1, num_experts], got {top_k}")
    if token_tile_size <= 0:
        raise ValueError(f"token_tile_size must be positive, got {token_tile_size}")
    if cluster_dim_in_token <= 0:
        raise ValueError(
            f"cluster_dim_in_token must be positive, got {cluster_dim_in_token}"
        )

    cga_tile_size = token_tile_size * cluster_dim_in_token
    num_remaining_tokens = num_tokens * top_k
    num_filled_experts = min(num_experts, num_remaining_tokens)
    max_num_cgas = num_filled_experts
    num_remaining_tokens -= num_filled_experts
    if num_remaining_tokens > 0:
        max_num_cgas += num_remaining_tokens // cga_tile_size
    return max_num_cgas * cluster_dim_in_token


def resolve_early_exit_max_token_ctas(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    cfg_overrides: dict,
    explicit_early_exit_max_token_ctas: int = 0,
) -> int:
    """Return launch-time token CTA extent for persistent/static early exit."""
    if explicit_early_exit_max_token_ctas < 0:
        raise ValueError(
            "early_exit_max_token_ctas must be non-negative, got "
            f"{explicit_early_exit_max_token_ctas}"
        )
    if not cfg_overrides.get("use_early_exit", 0):
        return 0
    if explicit_early_exit_max_token_ctas:
        return explicit_early_exit_max_token_ctas

    shape_overrides = dict(cfg_overrides)
    shape_overrides["use_early_exit"] = 0
    shape_cfg = make_config(**shape_overrides)
    token_tile_size = shape_cfg.token_tile_size
    # Batch-N (swapAB) distributes the token tile along physical CGA Y,
    # while Batch-M distributes it along physical CGA X.  Pad for the
    # preferred shape; validation guarantees that a fallback shape divides
    # the preferred shape, so the same layout is valid for either launch.
    cluster_dim_in_token = shape_cfg.token_cluster_dim
    return compute_max_num_ctas_in_token_dim_for_moe(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        token_tile_size=token_tile_size,
        cluster_dim_in_token=cluster_dim_in_token,
    )


CUDA_GRID_DIM_LIMITS = (2**31 - 1, 65535, 65535)


def compute_launch_grid(
    cfg: BatchedGemmConfig,
    *,
    problem_m: int,
    problem_n: int,
    early_exit_max_token_ctas: int,
) -> tuple[int, int, int]:
    """Compute the CUDA grid used by the BatchedGemm host launcher."""
    num_tiles_m = (problem_m + cfg.tile_m - 1) // cfg.tile_m
    num_tiles_n = (problem_n + cfg.tile_n - 1) // cfg.tile_n
    launch_num_tiles_m = num_tiles_m
    launch_num_tiles_n = num_tiles_n
    if cfg.use_early_exit:
        if cfg.is_swap_ab:
            launch_num_tiles_n = early_exit_max_token_ctas
        else:
            launch_num_tiles_m = early_exit_max_token_ctas

    cluster_m, cluster_n, _ = cfg.preferred_cluster_shape
    grid_m = ((launch_num_tiles_m + cluster_m - 1) // cluster_m) * cluster_m
    grid_n = ((launch_num_tiles_n + cluster_n - 1) // cluster_n) * cluster_n
    if cfg.is_static_persistent:
        return (
            cluster_m,
            cluster_n,
            min(
                grid_m * grid_n // (cluster_m * cluster_n),
                cfg.static_persistent_clusters,
            ),
        )
    if cfg.is_persistent and not cfg.raster_along_m:
        return (
            cluster_m,
            cluster_n,
            grid_m * grid_n // (cluster_m * cluster_n),
        )
    return (grid_m, grid_n, 1)


def validate_launch_grid(
    cfg: BatchedGemmConfig,
    *,
    problem_m: int,
    problem_n: int,
    early_exit_max_token_ctas: int,
) -> None:
    """Reject a BatchedGemm launch whose CUDA grid exceeds hardware limits."""
    grid = compute_launch_grid(
        cfg,
        problem_m=problem_m,
        problem_n=problem_n,
        early_exit_max_token_ctas=early_exit_max_token_ctas,
    )
    axis_names = ("x", "y", "z")
    for axis, value, limit in zip(axis_names, grid, CUDA_GRID_DIM_LIMITS, strict=True):
        if value <= limit:
            continue
        raster_order = "along-M" if cfg.raster_along_m else "along-N"
        detail = ""
        if cfg.is_persistent and not cfg.raster_along_m and axis == "z":
            detail = (
                " Along-N rasterization stores the cluster count in gridDim.z; "
                "use along-M rasterization or a larger token tile."
            )
        raise ValueError(
            f"Prims-TS BatchedGemm {raster_order} launch gridDim.{axis}="
            f"{value} exceeds the CUDA limit {limit} (grid={grid}).{detail}"
        )


def compute_num_k_tiles(cfg: BatchedGemmConfig) -> int:
    """Compute the number of K-tiles for a given problem shape."""
    problem_k = cfg.tile_k
    padded_k = ((problem_k + cfg.tile_k - 1) // cfg.tile_k) * cfg.tile_k
    return padded_k // cfg.tile_k
