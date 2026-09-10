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

"""Integer constants shared by the FMHA decode TS implementation.

Keep non-obvious integer constants here with their rationale so config,
resource, and reduction code can use named values without duplicating comments.
"""

import math

# B200 has 148 SMs. Use this only when the runtime SM query is unavailable,
# so auto split-KV selection remains deterministic in offline/test flows.
FALLBACK_SM_COUNT_B200 = 148

# Shared-memory budget constants are in KiB because profile sizing is based on
# the hardware SMEM carveout. KV staging is capped below the full
# 218 KiB budget so Q staging, page-offset staging, and scratch can coexist.
TOTAL_SMEM_BUDGET_KIB = 218
MAX_KV_STAGE_SMEM_KIB = 144
BYTES_PER_KIB = 1024

# The M64N256 profile stages one complete 256-row K or V tile per shared-ring
# slot, so the ring depth follows the element width. A 16-bit tile is 64 KiB:
# three stages occupy 192 KiB, and with the 16-KiB Q stage, the dedicated tail
# exchange, and the metadata/barrier allocations the CTA sits at about 227 KiB,
# the SM100 carveout. A byte-wide tile (E4M3 or Int8 K with E4M3 V) is 32 KiB,
# so three stages leave about 111 KiB unused. Block-sparse Sage kernels run
# about 7.5% faster with four or five stages than with three and gain nothing
# past that: a deeper ring no longer hides load latency and only adds barrier
# traffic. Five is the byte-wide default rather than four because ptxas keeps
# every warp role spill-free at five (and six) stages, while the four-stage
# build demotes the persistent work-queue state to local memory in every
# role; re-check the ptxas spill report if that state or the register split
# per role changes. The dense static grid is the one profile that prefers
# four (a few percent over five); it can set kv_stages explicitly. Keep both
# exact-profile depths separate from the conservative, topology-independent
# MAX_KV_STAGE_SMEM_KIB inference above.
KV_TILE_256_SHARED_FIFO_STAGES = 3
KV_TILE_256_BYTE_WIDE_SHARED_FIFO_STAGES = 5

# The four semantic K64 atoms are stored in the physical K slots consumed by
# the two interleaved QK instructions in this order.
KV_TILE_256_K_SLOT_FOR_SEMANTIC_ATOM = (0, 2, 1, 3)

# Keep the old maximum as the exponent reference while a new maximum is at
# most eight log2 units larger. This avoids an output-correction round without
# letting an intermediate probability exceed 2**8; the softmax identity is
# unchanged apart from normal finite-precision rounding. As in the
# FlashInfer/TRT-LLM policy, this assumes normal model logits rather than
# adversarial values outside the qualified probability bound. Streamed KV256
# and block-sparse Keeps profiles apply it; see
# ``FmhaDecodeConfig.defers_softmax_anchor_updates``.
SOFTMAX_RESCALE_THRESHOLD_LOG2 = 8.0

# A launch bound makes ptxas honor warpgroup ``setmaxnreg`` allocations, but
# the resulting register hand-off has a fixed cost. Paired B200 measurements
# show that it is amortized once a Q64/KV256 CTA processes at least 32 tiles
# (8K dense KV tokens); shorter loops are as fast or faster without it.
KV_TILE_256_REGISTER_REALLOCATION_MIN_TILES = 32

# TMA-swizzled Q rows are padded to 128 B before computing how many KV stages
# fit in the remaining SMEM budget.
Q_ROW_ALIGNMENT_BYTES = 128
BITS_PER_BYTE = 8

# Conservative per-CTA budget for cluster distributed-SMEM reduction leader
# staging. The reducer-owner CTA holds every split's partial O/stats for its
# row band; keep staging below this cap to avoid dynamic SMEM overflow.
CLUSTER_PARTIAL_SMEM_LIMIT_KIB = 96
MAX_CLUSTER_PARTIAL_SMEM_BYTES = CLUSTER_PARTIAL_SMEM_LIMIT_KIB * BYTES_PER_KIB

# Fused GMEM/cluster partial O is staged in 16-bit elements, and each row's stats
# are a float2 pair (max, sum). The separate-GMEM workspace contract instead
# uses one FP32 log2-LSE scalar and normalized 16-bit O.
PARTIAL_O_ELEMENT_BYTES = 2
PARTIAL_STATS_VALUES_PER_ROW = 2
SEPARATE_REDUCTION_LSE_VALUES_PER_ROW = 1
FP32_BYTES = 4

# Hardware clusterDim.x limit for this decode reduction layout.
MAX_CLUSTER_DIM_X = 16

# One split-KV CTA should cover at least two loop iterations so reduction
# overhead does not dominate tiny per-split K ranges.
MIN_LOOP_ITERS_PER_SPLIT = 2

# The maximum number of warp groups per CTA.
MAX_WARP_GROUPS = 4

# Default used only when the launch helper has no resolved decode config.
# Config-aware FMHA and block-sparse callers pass their selected KV tile.
AUTO_LAUNCH_TILE_SIZE_KV = 128

# Split-KV is worthwhile below one static SM wave only when each CTA still owns
# enough K/V work to amortize GMEM reduction. The B200-qualified
# crossover is 2,048 tokens: 16 KV128 tiles or 8 KV256 tiles.
SPLIT_KV_MIN_TOKENS_PER_CTA = 2_048

# Two interleaved K/V instances form the decode cadence: instance 0 is the
# first K/P/V stream in each loop iteration and instance 1 is the second. These
# integer tags are passed through TS work calls and used in constexpr branches.
KV_INST0 = 0
KV_INST1 = 1

# Compact K/V selector used by shared SMEM resources. Keep this as an integer
# contract because the JIT work-call plumbing expects constexpr scalar values.
KV_KIND_K = 0
KV_KIND_V = 1

# One hardware warp. Barrier participant counts are expressed as warps * lanes.
WARP_THREADS = 32

# TMEM column layout for staged SwapsMmaAb O. A TMEM row holds 256 columns, and
# the tcgen05 descriptor encodes a 16-row jump in the high 16 bits.
TMEM_COLUMNS_PER_ROW = 256
TMEM_ROW_STRIDE = 16 << 16

# Number of scalar softmax/output values packed in one register for the
# supported element widths.
PACKED_REGISTER_BYTES = 4
FP8_VALUES_PER_REG = 4
FP16_VALUES_PER_REG = 2

# K elements consumed by one tcgen05 MMA instruction: 32 bytes of the K-major
# operand row, so 32 one-byte or 16 two-byte elements. A streamed K32 score
# fragment therefore feeds one FP8 PV instruction or two 16-bit ones.
FP8_MMA_K_STEP = 32
FP16_MMA_K_STEP = 16

# FP8 probabilities are quantized as 448 * p (the E4M3 maximum). Row sums and
# attention-sink terms follow the same scale; the output normalization divides
# it back out. The log2 form is the addend of the exp2-domain softmax.
FP8_P_QUANT_SCALE = 448.0
FP8_P_QUANT_LOG2_SCALE = math.log2(FP8_P_QUANT_SCALE)

# tcgen05 SMEM descriptor geometry: address offsets count 16-byte units and a
# 128-byte swizzle atom spans one 128-byte row per K or MN index.
SMEM_DESC_UNIT_BYTES = 16
SWIZZLE_128B_ROW_BYTES = 128

# Per-lane register ownership denominators for packed output fragments.
FP8_OUTPUT_ELEMENTS_PER_REG_GROUP = 512
FP16_OUTPUT_ELEMENTS_PER_REG_GROUP = 256

# SwapsMmaAb maps up to eight Q heads into one q-repetition group.
Q_REPETITION_GROUP_HEADS = 8

# Packed P register count per q-repetition in the SwapsMmaAb path.
FP8_P_PACKED_REGS_PER_Q_REPEAT = 2
FP16_P_PACKED_REGS_PER_Q_REPEAT = 4

# Standalone reducer CTA shape. Each thread owns a 16-byte vector, so one CTA
# reduces one contiguous 8 KiB slice of the partial-O buffer.
REDUCTION_THREADS_PER_CTA = 512
REDUCTION_BYTES_PER_THREAD = 16
REDUCTION_BYTES_PER_SLICE = REDUCTION_THREADS_PER_CTA * REDUCTION_BYTES_PER_THREAD

# Clustered standalone reducer shape. One 128-thread CTA covers a contiguous
# 2 KiB partial-O slice with one 16-byte vector per thread. Each cluster rank
# owns a compile-time 2, 4, or 8 split slots. Loads are batched in groups of at
# most four; padded split slots remain neutral and never form GMEM pointers.
PARALLEL_REDUCTION_THREADS_PER_CTA = 128
PARALLEL_REDUCTION_BYTES_PER_SLICE = (
    PARALLEL_REDUCTION_THREADS_PER_CTA * REDUCTION_BYTES_PER_THREAD
)
PARALLEL_REDUCTION_LOAD_BATCH = 4
PARALLEL_REDUCTION_FINAL_REDUCERS = 4

# Each reduction thread produces an 8-element O vector backed by four packed
# 16-bit registers.
OUTPUT_VALUES_PER_THREAD = 8
FP8_PACKED_OUTPUT_REGS_PER_THREAD = 2
PACKED_OUTPUT_REGS_PER_THREAD = 4
