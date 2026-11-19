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

"""Shared instruction and layout constants for MLA decode TS examples."""

# CUDA warp geometry used by lane ownership, butterfly reductions, and TMEM row
# addressing.  Keep masks and shifts named so instruction-level code does not
# hide the warp contract behind raw bit constants.
WARP_LANES = 32
WARP_LANE_SHIFT = 5
WARP_REDUCTION_BFLY_DISTANCES = (16, 8, 4, 2, 1)


TMEM_DEALLOC_MBAR_THREADS = WARP_LANES


# tcgen05 TMEM load/store instruction geometry used by MLA softmax, correction,
# and epilogue paths.
TCGEN05_32B_SHAPE = "32x32b"
TCGEN05_32B_REGS_PER_LOAD = 32

# Softmax uses base-two exponentials internally; public sparse LSE uses ln.
LN_2 = 0.6931471805599453
LOG2_E = 1.4426950408889634


# The same scale barrier in the standalone 2CTA split-KV reducers, where every
# thread of the CTA joins it. A named barrier ID reserves all lower IDs, so ID
# 4 limited these latency-bound 128-thread CTAs to three per SM (SM107); ID 1
# leaves residency to registers and threads (eight CTAs).
STANDALONE_REDUCTION_SCALE_BARRIER_ID = 1


# gather4 always copies four selected KV rows. This instruction width is
# independent of the number of issuing warps or the KV tile size.
TMA_GATHER_ROWS = 4

# Sparse 2CTA offset caches derive their sizes from cfg token geometry:
# K owns 128/2=64 rows per CTA; V needs all 128 rows. Lane-distributed
# caches therefore hold 64/32=2 K and 128/32=4 V words per lane. A uniform
# cache instead holds the issuing warp's share (64/W K and 128/W V words
# per lane). Pending V stays lane-distributed until it becomes current.

# Dense MLA kernels specialize paged-KV addressing for this explicit ABI set.
# Each listed size exactly partitions a 128-token KV tile; other page sizes are
# outside the current kernel ABI.
SUPPORTED_MLA_PAGE_SIZES = (4, 8, 16, 32, 64, 128)

# Throughput 2CTA epilogue maps 128 local threads onto two 64-row groups and
# 128-column output halves for vectorized GMEM publication.
EPILOGUE_THREAD_TILE_THREADS = 128
EPILOGUE_THREAD_TILE_MASK = EPILOGUE_THREAD_TILE_THREADS - 1
EPILOGUE_ROW_THREADS = 64
EPILOGUE_ROW_MASK = EPILOGUE_ROW_THREADS - 1
EPILOGUE_COLUMN_GROUP_SHIFT = 6
BF16_OUTPUT_VECTOR_ELEMENTS = 8
FP8_OUTPUT_VECTOR_ELEMENTS = 16
PACKED_FP8_OUTPUT_REGS = 4
