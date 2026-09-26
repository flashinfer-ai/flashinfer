# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Planned MXFP4 W4A8 routed MoE; minimum architecture SM100 (B300: SM103).

Planning binds caller-owned buffers and compiles the selected offline tactic.
Execution reads their current contents, including runtime SiTU parameters.
Use a separate plan/output/workspace for concurrently executing calls.

Rank-local layouts come from explicit metadata (``Mxfp4MoEParallelLayout`` or
the explicit local expert interval), never from tensor shapes. Expert
parallelism owns a contiguous global expert interval; MoE tensor parallelism
owns an intermediate-dimension shard of every expert. Hybrid layouts are
rejected. Every rank output is a partial sum whose reduction is external.
"""

from dataclasses import dataclass
from math import prod
from typing import Optional, Union

import cuda.bindings.driver as cuda
import os
import torch

from ...tllm_enums import ActivationType
from .fused_moe import _moe_core_impl, validate_w4a8_inputs
from .moe_utils import get_max_num_tiles, moe_sort
from .mxfp4_finalize import plan_finalize_rows
from .mxfp4_routing import (
    FUSED_ROUTE_LARGE_MAX_LOCAL_EXPERTS,
    FUSED_ROUTE_MAX_ROUTES,
    FUSED_ROUTE_MAX_ROUTES_LARGE,
    _plan_route_preprocess,
)
from .blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    blockscaled_contiguous_gather_grouped_gemm_act_fusion,
)
from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    blockscaled_contiguous_grouped_gemm_finalize_fusion,
)
from .swapab_moe import (
    SWAP_MAX_AB_STAGES,
    SWAP_ROW_TILE,
    fill_permuted_token_index,
    swap_row_tma,
    swapab_dispatch,
    swapab_gemm1_situ,
    swapab_gemm2,
)
from .tuner import DEFAULT_BLACKWELL_MOE_TACTIC, canonicalize_w4a8_tactic

# Swap-AB finalize form by token count. T <= this bound reduce-adds
# ``alpha * route_weight * acc`` straight from the GEMM2 epilogue
# (``red.global.add.bf16x2`` into the zero-filled output). Above it GEMM2
# writes ``alpha * acc`` rows in permuted order into workspace and a separate
# gather kernel applies the route weights: the per-element reductions cost
# ~80-160 G adds/s, which at TP8 T=1024 (8192 local rows x 7168) added
# ~380 us to a 244 us GEMM2, while the two-stage form streams the rows once.
SWAP_ATOMIC_FINALIZE_MAX_TOKENS = int(
    os.environ.get("SWAPAB_ATOMIC_FINALIZE_MAX_TOKENS", "16")
)
# Two-stage finalize only pays off with a narrow GEMM2 K (short per-tile
# mainloops cannot hide the epilogue's reductions): TP8 (K=384) T=128 saved
# 38 us, EP8 (K=3072) T=128..1024 lost 6-28 us against the fused epilogue.
SWAP_TWO_STAGE_MAX_SHARD = int(os.environ.get("SWAPAB_TWO_STAGE_MAX_SHARD", "512"))
# Largest token count whose swap-AB GEMM2 uses 4-K-block stages (deeper
# pipeline); above it the intermediate shard streams in one 12-block stage.
SWAP_GEMM2_SHORT_STAGE_MAX_TOKENS = int(
    os.environ.get("SWAPAB_GEMM2_SHORT_STAGE_MAX_TOKENS", "256")
)
# L2 eviction policy for the dense grouped GEMMs' weight TMA loads (CUTLASS
# SM90 TMA cache-hint encodings). Weights stream once per token tile while the
# gathered activations are re-read by every N tile of an expert: EVICT_FIRST
# keeps the activations resident (B300, TP8 T=2048 balanced: GEMM1 530 ->
# 511 us, GEMM2 345 -> 316 us; T=4096: 1049 -> 930 us total).
_DENSE_L2_HINTS = {
    "none": None,
    "first": 0x12F0000000000000,
    "last": 0x14F0000000000000,
}
DENSE_WEIGHT_L2_HINT = _DENSE_L2_HINTS[os.environ.get("MXFP4_DENSE_L2HINT", "first")]
# Hybrid prefill form (wide swap tiles, finalize=True): the swap GEMM1 runs
# ``n_tile``-row sub-tiles of 128-row sort groups (only the occupied ones, via
# a device-built work list) and writes its MXFP8 rows with the block-scaled
# SFA layout, so the dense contiguous grouped GEMM2 with the bulk-reduce
# finalize consumes them directly: no permuted partial rows, no finalize
# kernel. B300 TP8 T=2048: 434 + 314 us against 434 + 295 + 94 (two-stage).
SWAP_HYBRID = os.environ.get("SWAPAB_HYBRID", "1") != "0"
SWAP_HYBRID_GROUP_ROWS = 128
# Hybrid form from this many routes (T * top_k) up: T > 256 for top-16.
SWAP_HYBRID_MIN_ROUTES = 4096
SWAP_HYBRID_MIN_TILE = 64
# Hybrid form, mixed GEMM1 tiles: a 128-row sort group with more valid rows
# than this runs as one dense gather-GEMM1 tile (its expert's weights stream
# once for the group); the other groups run as swap sub-tiles, which stream
# the weights once per sub-tile. Measured on B300 for the MoE-TP shard: 16
# hot experts holding every route at T=128..2048 re-stream each expert 16x
# in the swap form (148 us at T=128 against 53 us for the dense form). 128
# disables the dense tiles.
SWAP_HYBRID_DENSE_MIN_ROWS = int(os.environ.get("SWAPAB_HYBRID_DENSE_MIN_ROWS", "64"))
# Mixed form below the hybrid cap (finalize only, T >= SWAP_MIXED_MIN_TOKENS):
# 128-row sort groups as in the hybrid form, dense gather-GEMM1 tiles for the
# groups above SWAP_HYBRID_DENSE_MIN_ROWS and swap GEMM1 sub-tiles for the
# rest, but the swap GEMM2 of the policy tile over every occupied sub-tile
# (reading the blocked scales) instead of the dense finalize GEMM2. Targets
# the concentrated routings (few experts holding every route) of the MoE-TP
# shard at T=128..1024; SWAPAB_MIXED=1 enables it, SWAPAB_MIXED_EP=1 also on
# expert-parallel ranks.
SWAP_MIXED = os.environ.get("SWAPAB_MIXED", "0") == "1"
# Programmatic dependent launch inside the swap-AB kernel chain (fused routing
# -> GEMM1 -> GEMM2): the routing kernel triggers its dependents at entry and
# the swap GEMMs wait before their first routing read, so the launch latency
# and prologue of each GEMM overlap the previous kernel. The caller's
# ``enable_pdl`` still governs the first launch of the op. 0 = plain launches.
SWAP_PDL = os.environ.get("SWAPAB_PDL", "1") != "0"
# Split form (swap-AB path, fused routing, finalize, T >= SWAP_SPLIT_MIN_TOKENS):
# the fused routing kernel lays out the local experts holding more than
# SWAP_SPLIT_MIN_ROWS rows in 128-row groups behind the policy-tile groups of
# the other experts, and a second pair of launches runs those groups: the
# dense gather GEMM1 (128-row tiles, block-scaled output scales) and the swap
# GEMM2 at n_tile = 128 reading those blocked scales. One weight stream per
# 128 rows instead of one per policy tile, and 8x fewer GEMM2 work items.
# Targets the routings where a few local experts hold most rows (the MoE-TP
# shard's ``empty`` rows re-stream each hot expert 8x at T=128); under
# balanced routing both wide launches find an empty list. The 128-row swap
# GEMM1 measured 2.3x slower than the dense gather tile for the same groups
# (65 vs 28 us, B300 TP8 T=128 empty), so the dense tile takes GEMM1.
# SWAPAB_SPLIT=0 disables, SWAPAB_SPLIT_EP=1 also enables it on
# expert-parallel ranks.
SWAP_SPLIT = os.environ.get("SWAPAB_SPLIT", "1") != "0"
SWAP_SPLIT_WIDE_TILE = 128
SWAP_SPLIT_MIN_ROWS = int(os.environ.get("SWAPAB_SPLIT_MIN_ROWS", "64"))
# ... and only when those experts hold this share (per mille) of the local
# rows: one wide expert (a ``hot`` routing, 1/16 of the rows) would pay the
# dense tile's single-CTA-per-N-tile latency (26 us at K=7168) for nothing.
SWAP_SPLIT_MIN_PERMILLE = int(os.environ.get("SWAPAB_SPLIT_MIN_PERMILLE", "250"))
SWAP_SPLIT_MIN_TOKENS = int(os.environ.get("SWAPAB_SPLIT_MIN_TOKENS", "128"))
# Upper bound: the fused routing kernel (which lays the split out) handles
# 16384 routes = T=1024 for top-16; there it costs the shard 17 us against 16
# for the conversion kernel plus moe_sort, and the split's balanced/hot rows
# pay 0.4-0.7 % (B300) for a 0.65x ``empty`` row. Above, the hybrid form.
SWAP_SPLIT_MAX_TOKENS = int(os.environ.get("SWAPAB_SPLIT_MAX_TOKENS", "1024"))
SWAP_SPLIT_EP = os.environ.get("SWAPAB_SPLIT_EP", "0") == "1"
# Wide launches of the split form: trigger their programmatic dependents at
# kernel entry (they often find no work items).
SWAP_SPLIT_EARLY_TRIGGER = os.environ.get("SWAPAB_SPLIT_EARLY_TRIGGER", "1") != "0"
# Plain chain (routing -> GEMM1 -> GEMM2, no split / hybrid / mixed launches):
# GEMM1 triggers its dependent right after its own dependency wait and GEMM2
# waits on GEMM1 only in the warps that load GEMM1's output, so GEMM2's CTAs
# take the SMs GEMM1's tile-less CTAs leave and stream their weight stages
# while GEMM1 runs. SWAPAB_DEP_PREFETCH=0 disables.
SWAP_DEP_PREFETCH = os.environ.get("SWAPAB_DEP_PREFETCH", "1") != "0"
# Plain-chain GEMM2 (finalize): split each weight tile's K over this many work
# items while the row's valid items fit half the SMs (device-side decision);
# the finalize reduce-add makes the partials additive. 1 disables.
SWAP_GEMM2_SPLIT_K = int(os.environ.get("SWAPAB_GEMM2_SPLIT_K", "2"))
# Plain-chain GEMM1 (SiTU): pairs of CTAs (a 2-CTA cluster) split each weight
# tile's K while the row's valid items fit half the SMs; the peer's partial
# accumulator crosses over DSMEM before the activation epilogue. 0 disables.
SWAP_GEMM1_CLUSTER_SPLIT = os.environ.get("SWAPAB_GEMM1_CLUSTER_SPLIT", "1") != "0"
# N tile of the wide dense GEMM1 by token count: with few wide groups the
# 128-wide tile doubles the streaming CTAs (B300 TP8 empty: T=128 65 -> 57 us,
# T=512 140 -> 134), from T=1024 the 256-wide tile is back ahead (236 vs 242).
SWAP_SPLIT_GEMM1_N_POLICY = ((512, 128), (1 << 62, 256))
# Wide GEMM2 of the split form on the dense finalize-fusion kernel: it
# reduce-adds the route-weighted 128-row groups into the zero-filled output
# (one 128 x N tile per weight tile, TMA-fed), and the two-stage finalize adds
# the narrow groups' rows on top instead of overwriting. The swap GEMM2 at
# ``n_tile = 128`` measured 35.9 us against 15.9 us for the dense kernel over
# the same 16 experts x 128 rows (MoE-TP shard, T=128 empty, B300).
SWAP_SPLIT_DENSE_GEMM2 = os.environ.get("SWAPAB_SPLIT_DENSE_GEMM2", "1") != "0"
# Wide chain on a side stream (fork after the routing kernel, join before the
# finalize / at the end): the two wide launches leave the narrow chain's
# critical path, so a routing without wide experts pays no dependency hops
# for them (measured 2 hops = ~5-7 us on the expert-parallel decode rows).
SWAP_SPLIT_SIDE_STREAM = os.environ.get("SWAPAB_SPLIT_SIDE_STREAM", "0") == "1"
SWAP_SPLIT_GEMM2_N = (
    int(os.environ["SWAPAB_SPLIT_GEMM2_N"])
    if os.environ.get("SWAPAB_SPLIT_GEMM2_N")
    else None
)
SWAP_SPLIT_GEMM1_N = (
    int(os.environ["SWAPAB_SPLIT_GEMM1_N"])
    if os.environ.get("SWAPAB_SPLIT_GEMM1_N")
    else None
)
SWAP_MIXED_MIN_TOKENS = int(os.environ.get("SWAPAB_MIXED_MIN_TOKENS", "128"))
# Mixed form by default on the fused-routing path (T * top_k <= 4096, i.e.
# T <= 256 for top-16) up to this token count (0 disables): there the fused
# routing kernel emits the work lists itself, so the form costs no extra
# launch besides the dense GEMM1 that may find an empty list.
SWAP_MIXED_AUTO_MAX_TOKENS = int(os.environ.get("SWAPAB_MIXED_AUTO_MAX_TOKENS", "0"))
# Timing switch: build the mixed-form lists in the fused routing kernel (1)
# or with the separate ``swapab_dispatch`` launch (0).
SWAP_MIXED_FUSED_LISTS = os.environ.get("SWAPAB_MIXED_FUSED_LISTS", "1") == "1"
SWAP_MIXED_EP = os.environ.get("SWAPAB_MIXED_EP", "0") == "1"
# Mixed form: dense tiles only when the full groups hold at least this share
# (per mille) of the valid rows (0 = always); timing-only switch to keep the
# swap GEMM1/GEMM2 scales plain (valid with the dense tiles disabled).
SWAP_MIXED_WIDE_PERMILLE = int(os.environ.get("SWAPAB_MIXED_WIDE_PERMILLE", "0"))
SWAP_MIXED_SF_PLAIN = os.environ.get("SWAPAB_MIXED_SF_PLAIN", "0") == "1"
# Mixed form: weight M-tiles per swap-GEMM2 work item. Measured on B300 (TP8
# T=256/1024 balanced): with the 128-row groups the GEMM2 of the policy tile
# loses 6-9 % at m_group 1 and is back at the 32-row-group time with 2.
SWAP_MIXED_GEMM2_MGROUP = int(os.environ.get("SWAPAB_MIXED_MGROUP2", "2"))
# Swap GEMM2 of the narrow MoE-TP shard (K = 384): two weight M-tiles per
# work item with 128-wide stages. Measured on B300 (TP8 rank 0, same GPU):
# every row from T=4 up is faster (balanced/hot -0.5..-4.6 %, empty
# -2..-7.5 %: T=1024 empty 388 -> 359 us, T=16 hot 182 -> 173 us), T=1 is
# unchanged; the wide expert-parallel shard (K = 3072) loses 7-11 % on its
# 24-40 us decode/empty rows with the same grouping and keeps one tile.
# Below SWAP_TP_GEMM2_MGROUP_MIN_TOKENS the shard keeps one tile too: with a
# single active expert (``empty`` routing) GEMM2 has 56 weight tiles and the
# grouping halves the CTAs that stream them (+0.3..0.8 us on 26-37 us rows)
# while the balanced/hot rows gain 1-5 us; the acceptance rule of this work
# forbids a slower row, so the grouping starts where every row gains.
# SWAPAB_MGROUP2 (swapab_moe.swap_m_group) still overrides both.
SWAP_TP_GEMM2_MGROUP = int(os.environ.get("SWAPAB_TP_MGROUP2", "2"))
SWAP_TP_GEMM2_MGROUP_MIN_TOKENS = int(
    os.environ.get("SWAPAB_TP_MGROUP2_MIN_TOKENS", "16")
)
# Swap-AB path: EVICT_FIRST on the weight loads helps every shape whose
# experts stream their weights once (decode -3..-15 us, single-group prefill
# -5..-14 us on B300, same-GPU pairs) and hurts a hot expert whose groups
# re-read them (EP8 T=128 hot +10 us, balanced +4 us).  The policy below is
# per layout and token count (see ``_swap_weight_l2_hint``): every layout
# uses it for decode; an expert-parallel rank skips it inside
# ``SWAP_EP_L2HINT_SKIP`` where the hot rows are thin.
SWAP_WEIGHT_L2_HINT_MAX_TOKENS = int(os.environ.get("SWAPAB_L2HINT_MAX_TOKENS", "16"))
SWAP_EP_L2HINT_SKIP = (
    SWAP_WEIGHT_L2_HINT_MAX_TOKENS + 1,
    int(os.environ.get("SWAPAB_EP_L2HINT_SKIP_MAX_TOKENS", "255")),
)
# Dense-path (T > swapab_max_tokens) W4A8 tactic measured on B300 (Kimi K3):
# 256-wide MMA N halves the GEMM2 tile count (TP8 T=2048 GEMM2 444 -> 344 us,
# T=4096 564 -> 459 us) and matches GEMM1; used when no offline table covers T.
B300_SITU_DENSE_TACTIC = (128, ((128, 256), (1, 1), False), ((128, 256), (1, 1), False))
# Dense-path W4A8 tactics by token-count bucket, measured on B300 (Kimi K3,
# candidate-only CUPTI graph medians, every tactic of a bucket timed in the
# same process on the same GPU). The rows per local expert set the M-tile
# padding. The 2-CTA M256 GEMM1 tile wins where the rows of an expert pad
# to the same 256-row multiple under both tiles, i.e. when ceil(rows / 128)
# is even: 146 rows per expert (balanced routing at T=8192 on either
# layout) pad to 256 either way and the 2-CTA tile streams each weight
# tile once (GEMM1 905 -> 760 us), while 73 rows (T=4096, or the
# remote-dominated routing at T=8192) pad 3.5x and 293 rows (T=16384) pad
# 1.75x against 1.31x, where M128 wins by 7-16%. On the narrow MoE-TP
# shard the bucket (7168, 14336] therefore takes the M256 tile with the
# 2-CTA 256-wide GEMM2 (same GPU, T=8192 balanced/empty/hot -9.1/-4.2/-0.4%
# against the cluster-2 N256 M128 tactic). The expert-parallel rank keeps
# M128 in every bucket: at T=8192 M256 gains 17/13/5% on the
# balanced/hot/empty routings but loses 34% on the remote-dominated one,
# and at T=16384 the reverse (-18% remote-dominated, +16/+6% balanced/hot),
# so its tile has to follow the rows per expert at run time (open). The
# 192-wide GEMM2 helps where GEMM2 is
# finalize-bound: cluster 1 wins 1-2% on the wide shard up to T=4096 and
# cluster 2 wins 3-5% there at T=16384 (not at 32768, where the default is
# best); on the narrow shard the cluster-2 N192 GEMM2 wins 1-7% on every
# routing from T=16384 (16384: -3.4..-4.6%, 32768: -1.2..-7.2%).
_T128_N256_C2 = (128, ((128, 256), (1, 1), False), ((128, 256), (1, 2), False))
_T128_N192 = (128, ((128, 256), (1, 1), False), ((128, 192), (1, 1), False))
_T128_N192_C2 = (128, ((128, 256), (1, 1), False), ((128, 192), (1, 2), False))
B300_SITU_DENSE_TACTIC_TABLE_WIDE = (
    (8192, _T128_N192),
    (16384, _T128_N192_C2),
    (1 << 62, B300_SITU_DENSE_TACTIC),
)
_T256_N256_C1 = (256, ((256, 256), (2, 1), False), ((256, 256), (2, 1), False))
B300_SITU_DENSE_TACTIC_TABLE_NARROW = (
    (2048, B300_SITU_DENSE_TACTIC),
    (7168, _T128_N256_C2),
    (14336, _T256_N256_C1),
    (1 << 62, _T128_N192_C2),
)
# Dual-tile dense routing. The rows per local expert decide whether the 2-CTA
# M256 tile (GEMM1 761 vs 905 us, GEMM2 431 vs 543 us on 146-row experts at
# EP8 T=8192, same padded rows) or the M128 tile (73- and 293-row experts,
# where M256 pads 2x / 1.33x) is faster, and that depends on the routing, not
# only on the token count (the remote-dominated routing at EP8 T=8192 has half
# the rows per expert of the balanced one; the MoE-TP shard's hot routing at
# T=16384 loses 7% under M256 while its empty routing gains). Above
# ``DENSE_DUAL_TILE_MIN_TOKENS`` the routing kernel therefore pads each routing
# to the 128- or 256-row tile at run time (M256 when its padded rows are within
# ``DENSE_DUAL_TILE_THRESHOLD_PERMILLE`` / 1000 of the M128 padded rows) and
# both GEMMs are launched in both tile variants; the variant the routing did
# not choose reads a zero tile count and exits (about 3 us per launch, measured
# on the mixed-form dense GEMM1). Measured on B300 (same GPU, T=8192..32768,
# both layouts): M256 wins only where the padded rows are equal (ratio 1.0,
# 4-17%), M128 wins at every measured ratio from 1.137 up (2-14%), so the
# threshold is 1.10. The bucket whose table tactic is the M256 tile runs the
# M128 base ``_T128_N256_C2`` under this rule instead.
DENSE_DUAL_TILE = os.environ.get("MXFP4_DENSE_DUAL_TILE", "1") == "1"
DENSE_DUAL_TILE_MIN_TOKENS = int(
    os.environ.get("MXFP4_DENSE_DUAL_TILE_MIN_TOKENS", "7168")
)
# The two launches of the variant the routing did not choose cost 2.6-3.9 us
# each plus their graph launch gaps (about 8 us together, measured EP8
# T=8192..32768 on B300). That is below 0.5% of every MoE-TP shard row above
# ``DENSE_DUAL_TILE_MIN_TOKENS`` (>= 1.07 ms) but 5-10% of the wide rank's
# empty routings (86-160 us), which must not run slower than the single-tile
# plan, so the default rule applies to shards up to ``DENSE_DUAL_TILE_MAX_SHARD``
# columns only; the wide rank keeps it opt-in (``dense_dual_tile=True`` or a
# larger ``MXFP4_DENSE_DUAL_TILE_MAX_SHARD``) until the unchosen launches can
# be skipped on the device (graph conditional nodes).
DENSE_DUAL_TILE_MAX_SHARD = int(
    os.environ.get("MXFP4_DENSE_DUAL_TILE_MAX_SHARD", "3072")
)
DENSE_DUAL_TILE_THRESHOLD_PERMILLE = int(
    os.environ.get("MXFP4_DENSE_DUAL_TILE_THRESHOLD_PERMILLE", "1100")
)
# Dense chain with the fused (reduce-add) finalize: the T x H BF16 output must
# be zero before GEMM2. The zero-fill runs at the HBM write rate (B300: 5.6 us
# at T=2048, 75 us at T=32768) and used to sit between GEMM1 and GEMM2 in
# stream order. With this on, ``run`` forks it onto an auxiliary stream right
# after the sort (the previous run's GEMM2 is already ordered before it) and
# joins before GEMM2, so it overlaps the sort and GEMM1; captured graphs get
# the fork/join as edges. "ep" (default) enables it on expert-parallel ranks
# only (num_local_experts < num_experts), where it was measured faster on
# every dense row; on the MoE-TP shard the finalize GEMM2 reduce-adds ran
# 2-4 % slower without the zero-fill immediately before it (its output no
# longer in L2), so the shard keeps the in-order launch. "1" = every layout,
# "0" = keep the zero-fill in the main stream everywhere.
DENSE_ASYNC_MEMSET = os.environ.get("MXFP4_DENSE_ASYNC_MEMSET", "ep")


def _dense_async_memset(num_experts: int, num_local_experts: int) -> bool:
    if DENSE_ASYNC_MEMSET == "1":
        return True
    if DENSE_ASYNC_MEMSET == "ep":
        return num_local_experts < num_experts
    return False


# Dense chain with the fused finalize: let the gather GEMM1's epilogue warps
# zero-fill the output (dynamic 16 KB chunks claimed from a counter once a
# CTA's tiles are stored) instead of a separate ``cudaMemsetAsync``. The
# memset takes every SM, so a GEMM1 launch enqueued behind it waits for the
# whole fill (B300: 20 / 54 / 74 us at T = 8192 / 16384 / 32768); inside GEMM1
# the CTAs without tiles fill while the others compute. Same vocabulary as
# DENSE_ASYNC_MEMSET: "ep" = expert-parallel ranks only, "1" = every layout,
# "0" = off (the memset path above).
DENSE_FILL_IN_GEMM1 = os.environ.get("MXFP4_DENSE_FILL_IN_GEMM1", "0")
# Below this token count the zero-fill (5.6 us at T = 2048) hides beside GEMM1
# on the auxiliary stream anyway; the in-GEMM1 fill only pays where the memset
# gates a GEMM launch.
DENSE_FILL_IN_GEMM1_MIN_TOKENS = int(
    os.environ.get("MXFP4_DENSE_FILL_IN_GEMM1_MIN_TOKENS", "8192")
)


def _dense_fill_in_gemm1(num_experts: int, num_local_experts: int, num_tokens: int) -> bool:
    if num_tokens < DENSE_FILL_IN_GEMM1_MIN_TOKENS:
        return False
    if DENSE_FILL_IN_GEMM1 == "1":
        return True
    if DENSE_FILL_IN_GEMM1 == "ep":
        return num_local_experts < num_experts
    return False


B300_SITU_DENSE_DUAL_TACTIC = _T256_N256_C1
# Experimental: dense-path GEMM2 writes expanded rows and ``moe_unpermute``
# applies the route weights (no bulk reduce-add into the output).
DENSE_TWO_STAGE_FINALIZE = os.environ.get("MXFP4_DENSE_TWO_STAGE", "0") == "1"
# Dense GEMM2 tile raster. N-fastest (the finalize kernel's default) reuses
# each 128-row A tile across the N tiles; M-fastest walks the M tiles of one
# N tile so the fused finalize's reduce-add target (one 256-column slab of
# every routed token row) stays L2-resident, and with ``swizzle`` N tiles per
# group the A tile is still reused ``swizzle`` times. Measured on B300 (r38e,
# graph, same GPU) on the 384-wide MoE-TP shard, span vs N-fastest with
# swizzle 4: T=16384 balanced 0.916 / empty 0.844 / hot 1.059, T=32768
# balanced 0.867 / empty 0.864 / hot 1.088; T=8192 loses 1-6 % on every
# routing and the wide expert-parallel rank (K = 3072) 3-5 %. The hot loss is
# specific to one dominant expert next to a long tail of small ones (a single
# giant expert plus 895 small: +4.5-6 %; 16 giant experts: -20 %; 512 equal
# small experts: -8 %), so the choice follows the routing on the device:
# ``auto`` compiles both rasters into the shard's GEMM2 from
# ``MXFP4_GEMM2_RASTER_M_MIN_TOKENS`` tokens (shards up to
# ``MXFP4_GEMM2_RASTER_M_MAX_SHARD`` columns) and the kernel's scheduler warp
# keeps N-fastest when one expert holds at least half the rows while more
# than 32 experts are active. ``1`` forces M-fastest, ``0`` N-fastest;
# ``MXFP4_GEMM2_SWIZZLE`` (default 4) sets the group.
DENSE_GEMM2_RASTER_M = os.environ.get("MXFP4_GEMM2_RASTER_M", "auto")
DENSE_GEMM2_SWIZZLE = int(os.environ.get("MXFP4_GEMM2_SWIZZLE", "4"))
DENSE_GEMM2_RASTER_M_MAX_SHARD = int(
    os.environ.get("MXFP4_GEMM2_RASTER_M_MAX_SHARD", "512")
)
DENSE_GEMM2_RASTER_M_MIN_TOKENS = int(
    os.environ.get("MXFP4_GEMM2_RASTER_M_MIN_TOKENS", "16384")
)
if DENSE_GEMM2_RASTER_M not in ("auto", "0", "1"):
    raise ValueError("MXFP4_GEMM2_RASTER_M must be auto, 0 or 1")
# Dual-tile routing: launching the alternate-tile GEMMs as programmatic
# dependents (PDL) was meant to hide the unchosen variant's launch, but the
# base kernels trigger their dependents only at their end, so the alternate
# grid launches into the base kernel's tail and waits there: its CUPTI
# duration grows to 26-61 us when unchosen (EP8 T=16384/32768) while the GPU
# span does not improve. Off by default; ``MXFP4_DUAL_ALT_PDL=1`` restores it.
DENSE_DUAL_ALT_PDL = os.environ.get("MXFP4_DUAL_ALT_PDL", "0") == "1"


_PARALLEL_MODES = ("single", "expert_parallel", "moe_tensor_parallel")


@dataclass(frozen=True)
class Mxfp4MoEParallelLayout:
    """Explicit rank placement; ``size`` ranks, this process is ``rank``.

    ``mode`` is ``"single"`` (one rank owns everything), ``"expert_parallel"``
    (rank ``r`` owns global experts ``[r*E/size, (r+1)*E/size)`` with the
    full intermediate dimension) or ``"moe_tensor_parallel"`` (every rank owns
    all experts and intermediate columns ``[r*I/size, (r+1)*I/size)`` of each).
    Hybrid expert/tensor parallelism is not representable; ``from_sizes``
    rejects it. Global ``num_experts`` and ``intermediate_size`` are resolved
    into rank-local values by ``resolve_mxfp4_moe_layout``.
    """

    mode: str = "single"
    size: int = 1
    rank: int = 0

    def __post_init__(self):
        if self.mode not in _PARALLEL_MODES:
            raise ValueError(f"parallel mode must be one of {_PARALLEL_MODES}")
        if not isinstance(self.size, int) or isinstance(self.size, bool):
            raise ValueError("parallel size must be an int")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise ValueError("parallel rank must be an int")
        if self.size < 1:
            raise ValueError("parallel size must be positive")
        if not 0 <= self.rank < self.size:
            raise ValueError(f"parallel rank must be in [0, {self.size})")
        if self.mode == "single" and (self.size, self.rank) != (1, 0):
            raise ValueError("single layout requires size 1 and rank 0")

    @classmethod
    def from_sizes(
        cls,
        *,
        ep_size: int = 1,
        ep_rank: int = 0,
        moe_tp_size: int = 1,
        moe_tp_rank: int = 0,
    ) -> "Mxfp4MoEParallelLayout":
        """Build a layout from EP/TP sizes; both above one is a hybrid error."""
        for name, size, rank in (
            ("ep", ep_size, ep_rank),
            ("moe_tp", moe_tp_size, moe_tp_rank),
        ):
            if size < 1 or not 0 <= rank < size:
                raise ValueError(
                    f"require {name}_size >= 1 and 0 <= {name}_rank < size"
                )
        if ep_size > 1 and moe_tp_size > 1:
            raise ValueError(
                "hybrid expert/tensor parallelism is unsupported: "
                f"ep_size={ep_size} and moe_tp_size={moe_tp_size} both exceed 1"
            )
        if ep_size > 1:
            return cls("expert_parallel", ep_size, ep_rank)
        if moe_tp_size > 1:
            return cls("moe_tensor_parallel", moe_tp_size, moe_tp_rank)
        return cls()


@dataclass(frozen=True)
class Mxfp4MoERankLayout:
    """Resolved rank-local geometry; ``num_experts``/``intermediate_size`` are global.

    ``parallel_size``/``parallel_rank`` are ``None`` when an explicit expert
    interval does not coincide with a uniform expert-parallel rank slice.
    """

    mode: str
    num_experts: int
    intermediate_size: int
    num_local_experts: int
    local_expert_offset: int
    intermediate_shard: int
    parallel_size: Optional[int] = None
    parallel_rank: Optional[int] = None

    @property
    def gemm1_n(self) -> int:
        """Rank-local GEMM1 output width: interleaved up and gate rows."""
        return 2 * self.intermediate_shard

    @property
    def gemm2_k(self) -> int:
        """Rank-local GEMM2 contraction length."""
        return self.intermediate_shard


def resolve_mxfp4_moe_layout(
    num_experts: int,
    intermediate_size: int,
    *,
    parallel_layout: Optional[Mxfp4MoEParallelLayout] = None,
    num_local_experts: Optional[int] = None,
    local_expert_offset: Optional[int] = None,
) -> Mxfp4MoERankLayout:
    """Derive and validate rank-local geometry from explicit metadata only.

    ``parallel_layout`` is the uniform EP/TP form. ``num_local_experts`` and
    ``local_expert_offset`` are the explicit expert-interval form, which is
    expert parallelism (or single when the interval is every expert). When
    both forms are given they must agree. Nothing is inferred from tensors.
    """
    if num_experts <= 0 or intermediate_size <= 0:
        raise ValueError("num_experts and intermediate_size must be positive")
    if parallel_layout is None:
        local = num_experts if num_local_experts is None else num_local_experts
        offset = 0 if local_expert_offset is None else local_expert_offset
        if local <= 0 or offset < 0 or offset + local > num_experts:
            raise ValueError(
                "local experts must form a nonempty contiguous global expert interval"
            )
        if local == num_experts and offset == 0:
            return Mxfp4MoERankLayout(
                "single",
                num_experts,
                intermediate_size,
                local,
                0,
                intermediate_size,
                1,
                0,
            )
        uniform = num_experts % local == 0 and offset % local == 0
        return Mxfp4MoERankLayout(
            "expert_parallel",
            num_experts,
            intermediate_size,
            local,
            offset,
            intermediate_size,
            num_experts // local if uniform else None,
            offset // local if uniform else None,
        )
    if not isinstance(parallel_layout, Mxfp4MoEParallelLayout):
        raise TypeError("parallel_layout must be an Mxfp4MoEParallelLayout")
    mode, size, rank = parallel_layout.mode, parallel_layout.size, parallel_layout.rank
    if mode == "expert_parallel":
        if num_experts % size:
            raise ValueError(
                f"expert parallelism requires num_experts ({num_experts}) divisible "
                f"by ep size ({size})"
            )
        local, offset, shard = (
            num_experts // size,
            rank * (num_experts // size),
            (intermediate_size),
        )
    elif mode == "moe_tensor_parallel":
        if intermediate_size % size or (intermediate_size // size) % 128:
            raise ValueError(
                f"MoE tensor parallelism requires intermediate_size ({intermediate_size}) "
                f"divisible by moe_tp size ({size}) into a multiple of 128"
            )
        local, offset, shard = num_experts, 0, intermediate_size // size
    else:
        local, offset, shard = num_experts, 0, intermediate_size
    if num_local_experts is not None and num_local_experts != local:
        raise ValueError(
            f"num_local_experts={num_local_experts} is inconsistent with "
            f"{mode} size {size} rank {rank}, which owns {local} local experts"
        )
    if local_expert_offset is not None and local_expert_offset != offset:
        raise ValueError(
            f"local_expert_offset={local_expert_offset} is inconsistent with "
            f"{mode} size {size} rank {rank}, whose offset is {offset}"
        )
    return Mxfp4MoERankLayout(
        mode, num_experts, intermediate_size, local, offset, shard, size, rank
    )


@dataclass(frozen=True)
class Mxfp4MoECapability:
    """Support verdict plus the resolved rank-local layout when it resolves."""

    supported: bool
    reason: str
    cuda_graph: bool
    layout: Optional[Mxfp4MoERankLayout] = None
    # Deferred finalize (``plan(..., do_finalize=False)``): rank-local GEMM2
    # rows plus the route weights and the assignment-to-row map, for callers
    # fusing the route-weight reduction with their own collectives.
    deferred_output: bool = False


def mxfp4_moe_capability(
    *,
    gpu_arch: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    local_expert_offset: Optional[int] = None,
    parallel_layout: Optional[Mxfp4MoEParallelLayout] = None,
    quantization: str = "mxfp4_w4a8",
    activation_type: ActivationType = ActivationType.Situ,
    cuda_graph: bool = True,
    do_finalize: bool = True,
) -> Mxfp4MoECapability:
    """Query support from explicit metadata without allocating or using CUDA.

    ``gpu_arch`` is 100 for SM100 or 103 for SM103. Weight scales are UE8M0
    with group size 32; activation storage is E4M3 and output is BF16.
    ``num_experts`` and ``intermediate_size`` are global model values; the
    rank-local expert interval and intermediate shard are derived from
    ``parallel_layout`` and/or the explicit ``num_local_experts`` and
    ``local_expert_offset`` (see ``resolve_mxfp4_moe_layout``). The result
    carries that resolved layout; CUDA Graph capture is supported for every
    supported configuration in both parallel modes. ``do_finalize=False``
    asks for the deferred output form (GEMM2 rows in permuted order, route
    weights and the expanded->permuted row map), which the SiTU swap-AB
    path provides at every token count; ``deferred_output`` reports it.
    """
    reason = ""
    layout = None
    deferred = activation_type == ActivationType.Situ
    if gpu_arch not in (100, 103):
        reason = "MXFP4 W4A8 requires SM100 or SM103"
    elif quantization != "mxfp4_w4a8":
        reason = "quantization must be explicitly mxfp4_w4a8"
    elif activation_type not in (ActivationType.Situ, ActivationType.Swiglu):
        reason = "planned MXFP4 supports SiTU and SwiGLU"
    elif min(hidden_size, intermediate_size) <= 0 or (
        hidden_size % 128 or intermediate_size % 128
    ):
        reason = "hidden and intermediate dimensions must be positive multiples of 128"
    elif not (1 <= top_k <= num_experts <= 1024):
        reason = "require 1 <= top_k <= num_experts <= 1024"
    elif top_k > 32:
        reason = "top_k must not exceed 32"
    else:
        try:
            layout = resolve_mxfp4_moe_layout(
                num_experts,
                intermediate_size,
                parallel_layout=parallel_layout,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
            )
        except (TypeError, ValueError) as error:
            reason = str(error)
    if not reason and not do_finalize and not deferred:
        reason = "deferred (do_finalize=False) output requires SiTU activation"
    return Mxfp4MoECapability(
        not reason, reason, not reason, layout, not reason and deferred
    )


@dataclass(frozen=True)
class _WorkspaceField:
    name: str
    shape: tuple
    dtype: torch.dtype
    offset: int
    nbytes: int


def _align(size: int) -> int:
    return (size + 255) // 256 * 256


def _byte_interval(tensor):
    first = tensor.data_ptr()
    span = 1 + sum(
        (dim - 1) * stride
        for dim, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return first, first + span * tensor.element_size()


def _overlap(left, right):
    return max(left[0], right[0]) < min(left[1], right[1])


class Mxfp4MoEPlan:
    """Executable with fixed buffer addresses, prepared outside graph capture.

    Update bound activation, routing, beta and scale tensors in-place before
    calling ``run`` or replaying a captured graph. No weights are copied.
    The output is this rank's partial sum: its local experts under expert
    parallelism, or its intermediate shard under MoE tensor parallelism.
    """

    def __init__(
        self, *, kwargs, workspace, topk_ids, topk_weights, route_ids, route_weights
    ):
        # Keep every bound tensor alive alongside the raw launch pointers.
        self._kwargs = kwargs
        self.workspace = workspace
        self.output = kwargs["moe_output"]
        self._topk_ids = topk_ids
        self._topk_weights = topk_weights
        self._route_ids = route_ids
        self._route_weights = route_weights
        self._route_preprocess = None
        self._aux_stream = None
        self._main_event = None
        self._memset_event = None
        self.device = self.output.device
        self._packed_weight_view = (
            topk_ids.view(torch.bfloat16)[:, ::2] if topk_weights is None else None
        )

    def _prepare_routing(self):
        if self._topk_weights is None:
            torch.bitwise_right_shift(self._topk_ids, 16, out=self._route_ids)
            self._route_weights.copy_(self._packed_weight_view)
        elif self._route_weights is not self._topk_weights:
            self._route_weights.copy_(self._topk_weights)

    def _prepare(self):
        # The existing path validates and warms the exact pointers/callables
        # retained below. No stream is retained: run resolves the caller's.
        launches = {}
        with torch.cuda.device(self.device):
            self._prepare_routing()
            _moe_core_impl(**self._kwargs, _prepared_launches=launches)
        self._sort, self._sort_args = launches["sort"]
        self._gather, self._gather_args, self._gather_kwargs = launches["gather"]
        # Dual-tile routing: the alternate-tile GEMMs run after the base ones;
        # the one the routing did not choose exits on a zero tile count.
        self._gather_alt = launches.get("gather_alt")
        self._finalize_alt = launches.get("finalize_alt")
        # No memset in the expanded-row (non-fused) finalize form.
        self._memset, self._memset_args = launches.get("memset", (None, None))
        # Auxiliary stream and fork/join events for the output zero-fill (see
        # DENSE_ASYNC_MEMSET); created here, outside any graph capture.
        if self._memset is not None and _dense_async_memset(
            self._kwargs["num_experts"], self._kwargs["num_local_experts"]
        ):
            self._aux_stream = torch.cuda.Stream(device=self.device)
            self._main_event = torch.cuda.Event()
            self._memset_event = torch.cuda.Event()
        self._finalize, self._finalize_args = launches["finalize"]
        self._unpermute = launches.get("unpermute")
        self._finalize_rows = None
        if "gemm2_partial" in launches:
            self._finalize_rows = plan_finalize_rows(
                launches["gemm2_partial"],
                self._kwargs["moe_sort_buffers"]["out_expanded_idx_to_permuted_idx"],
                self._route_weights,
                self.output,
                expanded_rows=True,
            )
        if self.output.shape[0] <= 16:
            self._route_preprocess = _plan_route_preprocess(
                self._topk_ids,
                self._topk_weights,
                route_ids=self._route_ids,
                route_weights=self._route_weights,
                output=self.output,
                moe_sort_buffers=(
                    None
                    if self._kwargs["enable_pdl"]
                    else self._kwargs["moe_sort_buffers"]
                ),
                num_experts=self._kwargs["num_experts"],
                num_local_experts=self._kwargs["num_local_experts"],
                local_expert_offset=self._kwargs["local_expert_offset"],
                tile_size=self._kwargs["tile_size"],
                _single_tile_per_expert=self._kwargs.get(
                    "_enable_decode_specialization", False
                ),
            )
            # Preprocessing warmup clears output. Finish the complete MoE so
            # plan retains its existing valid-output postcondition.
            self.run()
        elif not self._kwargs["enable_pdl"]:
            # T > 16: one conversion + output-clear launch replaces the torch
            # unpack kernels and the separate memset (run() skips the memset
            # whenever a route preprocess is bound).
            self._route_preprocess = _plan_route_preprocess(
                self._topk_ids,
                self._topk_weights,
                route_ids=self._route_ids,
                route_weights=self._route_weights,
                output=self.output,
                clear_output=self._kwargs.get("zero_fill_counters") is None,
            )
            self.run()

    def run(self) -> torch.Tensor:
        """Enqueue on the caller's current stream and return the bound output.

        All GPU buffers and compiled kernels were prepared by ``plan``.
        This method performs no tuning, allocation, or host synchronization.
        """
        with torch.cuda.device(self.device):
            stream_ptr = torch.cuda.current_stream().cuda_stream
            stream = cuda.CUstream(stream_ptr)
            if self._route_preprocess is None:
                self._prepare_routing()
            else:
                self._route_preprocess.run(stream)
            if (
                self._route_preprocess is None
                or not self._route_preprocess.sorts_tokens
            ):
                self._sort(*self._sort_args, stream_ptr)
            async_memset = (
                self._route_preprocess is None and self._aux_stream is not None
            )
            if async_memset:
                # Fork: the zero-fill waits for everything enqueued so far
                # (the previous run's GEMM2 included) and runs beside GEMM1.
                current = torch.cuda.current_stream()
                self._main_event.record(current)
                self._aux_stream.wait_event(self._main_event)
                self._memset(*self._memset_args, self._aux_stream.cuda_stream)
                self._memset_event.record(self._aux_stream)
            self._gather(*self._gather_args, stream=stream, **self._gather_kwargs)
            if self._gather_alt is not None:
                gather_alt, gather_alt_args, gather_alt_kwargs = self._gather_alt
                gather_alt(*gather_alt_args, stream=stream, **gather_alt_kwargs)
            if async_memset:
                # Join: GEMM2 reduce-adds into the zeroed output.
                torch.cuda.current_stream().wait_event(self._memset_event)
            elif self._route_preprocess is None and self._memset is not None:
                self._memset(*self._memset_args, stream_ptr)
            self._finalize(*self._finalize_args, stream=stream)
            if self._finalize_alt is not None:
                finalize_alt, finalize_alt_args = self._finalize_alt
                finalize_alt(*finalize_alt_args, stream=stream)
            if self._unpermute is not None:
                unpermute, unpermute_kwargs = self._unpermute
                unpermute(**unpermute_kwargs)
            if self._finalize_rows is not None:
                self._finalize_rows.run(stream)
        return self.output


class Mxfp4MoESwapAbPlan:
    """Plan on the swap-AB path: route preprocessing (ID unpack, FP32 weights,
    output zero-fill; fused with ``n_tile``-row expert grouping for T <= 16,
    followed by ``moe_sort`` above) and the two swap-AB grouped GEMMs. Same
    contract as :class:`Mxfp4MoEPlan`: fixed buffer addresses,
    graph-capturable ``run``. ``finalize=False`` is the deferred form: GEMM2
    writes ``alpha * acc`` rows in permuted order into ``output`` and the plan
    exposes ``expanded_idx_to_permuted_idx`` / ``route_weights``. With
    ``finalize=True`` and T > ``SWAP_ATOMIC_FINALIZE_MAX_TOKENS`` the same
    permuted rows go to the ``partial_rows`` workspace region and a fifth
    launch (:func:`plan_finalize_rows`) applies the route weights
    (``two_stage``); at or below the bound GEMM2 reduce-adds into ``output``.
    """

    def __init__(
        self,
        *,
        wrapper,
        buffers,
        workspace,
        x,
        x_sf,
        topk_ids,
        topk_weights,
        w1,
        w1_sf,
        w2,
        w2_sf,
        beta,
        linear_beta,
        output,
        n_tile,
        finalize=True,
    ):
        self._wrapper = wrapper
        self._buffers = buffers
        self.workspace = workspace
        self.output = output
        self.device = output.device
        self.n_tile = n_tile
        self.finalize = bool(finalize)
        self.deferred = not self.finalize
        self.hybrid = wrapper._swap_hybrid(x.shape[0], self.finalize)
        self.mixed = wrapper._swap_mixed(x.shape[0], self.finalize)
        self.split = wrapper._swap_split(x.shape[0], self.finalize)
        self.split_dense = False  # set by _prepare
        self._side_stream = None
        self._fork_event = self._join_event = None
        if self.split and SWAP_SPLIT_SIDE_STREAM:
            self._side_stream = torch.cuda.Stream(device=self.device)
            self._fork_event = torch.cuda.Event()
            self._join_event = torch.cuda.Event()
        self.group_rows = (
            SWAP_HYBRID_GROUP_ROWS if (self.hybrid or self.mixed) else n_tile
        )
        self.two_stage = (
            self.finalize
            and not self.hybrid
            and x.shape[0] > SWAP_ATOMIC_FINALIZE_MAX_TOKENS
            and wrapper.intermediate_shard <= SWAP_TWO_STAGE_MAX_SHARD
        )
        self._partial_rows = buffers["partial_rows"] if self.two_stage else None
        self._finalize_rows = None
        # Deferred-finalize outputs (valid after ``run``): row of each
        # (token, slot) assignment (-1 when not local) and FP32 route weights.
        self.expanded_idx_to_permuted_idx = buffers["out_expanded_idx_to_permuted_idx"]
        self._inputs = (x, x_sf, topk_ids, topk_weights, w1, w1_sf, w2, w2_sf)
        self._beta = beta
        self._linear_beta = linear_beta
        # Same private surface as Mxfp4MoEPlan (tests poke these buffers).
        self._topk_ids = topk_ids
        self._topk_weights = topk_weights
        self._kwargs = {
            "gemm1_out": buffers["gemm1_out"],
            "gemm1_out_scale": buffers["gemm1_out_scale"],
            "moe_output": output,
            "moe_sort_buffers": {
                name: value
                for name, value in buffers.items()
                if name.startswith("out_")
            },
        }
        self._route_ids = buffers["route_ids"] if topk_weights is None else topk_ids
        self._route_weights = (
            topk_weights
            if topk_weights is not None and topk_weights.dtype == torch.float32
            else buffers["route_weights"]
        )
        self.route_weights = self._route_weights

    def _prepare_routing(self):
        if self._topk_weights is None:
            torch.bitwise_right_shift(self._topk_ids, 16, out=self._route_ids)
            self._route_weights.copy_(self._packed_weight_view)
        elif self._route_weights is not self._topk_weights:
            self._route_weights.copy_(self._topk_weights)

    def _prepare(self):
        w = self._wrapper
        # Swap-chain kernels are PDL-launched (see SWAP_PDL); each waits before
        # its first read of a predecessor's output.
        self._pdl = pdl = w.enable_pdl or SWAP_PDL
        self._dep_prefetch = SWAP_DEP_PREFETCH and not (
            self.split or self.hybrid or self.mixed
        )
        x, x_sf, topk_ids, topk_weights, w1, w1_sf, w2, w2_sf = self._inputs
        b = self._buffers
        num_tokens = x.shape[0]
        self._route_preprocess = None
        self._sort = None
        self._dispatch = None
        self._dispatch_args = None
        self._gemm1_dense = None
        self._gemm2_wide = None
        self._token_index = None
        self._token_index_args = None
        self._packed_weight_view = (
            topk_ids.view(torch.bfloat16)[:, ::2] if topk_weights is None else None
        )
        sort_buffers = {
            name: value for name, value in b.items() if name.startswith("out_")
        }
        launches = {}
        # The route preprocess kernel clears the finalize output; in the
        # deferred form the first T rows of the row buffer stand in (they are
        # overwritten or padding, so the clear is harmless).
        # In the deferred and two-stage forms the first T permuted rows stand
        # in (overwritten or padding, so the clear is harmless): the finalize
        # kernel writes every output row itself -- except in the split form
        # with the dense wide GEMM2, which reduce-adds into the output.
        # With the fused (atomic) finalize the dense wide GEMM2 reduce-adds
        # into the same zero-filled output as the narrow swap GEMM2; with the
        # two-stage finalize the finalize kernel accumulates on top of it.
        split_dense = self.split and SWAP_SPLIT_DENSE_GEMM2
        self.split_dense = split_dense
        clear_output = not self.two_stage or split_dense
        if (self.finalize and not self.two_stage) or split_dense:
            clear_target = self.output
        elif self.two_stage:
            clear_target = self._partial_rows[:num_tokens]
        else:
            clear_target = self.output[:num_tokens]
        # Mixed form on the fused-routing path: the routing kernel emits the
        # wide / narrow / all-sub-tile lists (no ``swapab_dispatch`` launch).
        lists_from_routing = (
            self.mixed
            and SWAP_MIXED_FUSED_LISTS
            and num_tokens * w.top_k <= w._fused_route_cap(num_tokens, self.finalize)
        )
        with torch.cuda.device(self.device):
            if num_tokens * w.top_k <= w._fused_route_cap(num_tokens, self.finalize):
                # Fused routing (T <= 512 for top-16 on the shard, T <= 1024
                # on an expert-parallel rank or with the split form): ID unpack, FP32
                # weights, n_tile-row groups and the output zero-fill in one
                # single-CTA launch instead of the conversion kernel plus
                # ``moe_sort`` (about 3 us against 11 us at T=128).
                self._route_preprocess = _plan_route_preprocess(
                    topk_ids,
                    topk_weights,
                    route_ids=self._route_ids,
                    route_weights=self._route_weights,
                    output=clear_target,
                    moe_sort_buffers=sort_buffers,
                    num_experts=w.num_experts,
                    num_local_experts=w.num_local_experts,
                    local_expert_offset=w.local_expert_offset,
                    tile_size=self.group_rows,
                    _single_tile_per_expert=self.group_rows >= num_tokens,
                    clear_output=clear_output,
                    dispatch_lists=(
                        dict(
                            wide_list=b["swap_wide_list"],
                            wide_count=b["swap_wide_count"],
                            narrow_list=b["swap_row_groups"],
                            narrow_count=b["swap_row_group_count"],
                            all_list=b["swap_all_groups"],
                            all_count=b["swap_all_count"],
                            narrow_tile=self.n_tile,
                            wide_min_rows=min(
                                SWAP_HYBRID_DENSE_MIN_ROWS, self.group_rows
                            ),
                            wide_min_permille=SWAP_MIXED_WIDE_PERMILLE,
                        )
                        if lists_from_routing
                        else None
                    ),
                    split_layout=(
                        dict(
                            wide_expert=b["swap_wide_expert"],
                            wide_limit=b["swap_wide_limit"],
                            wide_list=b["swap_wide_list"],
                            wide_count=b["swap_wide_count"],
                            wide_tile=SWAP_SPLIT_WIDE_TILE,
                            wide_min_rows=SWAP_SPLIT_MIN_ROWS,
                            wide_min_permille=SWAP_SPLIT_MIN_PERMILLE,
                            rows_capacity=b["out_permuted_idx_to_expanded_idx"].shape[
                                0
                            ],
                        )
                        if self.split
                        else None
                    ),
                )
            else:
                # Generic routing: one conversion + output-clear launch, then
                # moe_sort with n_tile-row groups.
                self._route_preprocess = _plan_route_preprocess(
                    topk_ids,
                    topk_weights,
                    route_ids=self._route_ids,
                    route_weights=self._route_weights,
                    output=clear_target,
                    clear_output=clear_output,
                )
                moe_sort(
                    token_selected_experts=self._route_ids,
                    token_final_scales=self._route_weights,
                    num_experts=w.num_experts,
                    top_k=w.top_k,
                    local_expert_offset=w.local_expert_offset,
                    num_local_experts=w.num_local_experts,
                    tile_tokens_dim=self.group_rows,
                    enable_pdl=pdl,
                    _prepared_launches=launches,
                    **sort_buffers,
                )
                self._sort, self._sort_args = launches["sort"]
            fused_finalize = self.finalize and not self.two_stage
            gemm1_lists = {}
            if (self.hybrid or self.mixed) and not lists_from_routing:
                # Work lists over the 128-row sort groups: wide (dense GEMM1
                # tiles), narrow (swap GEMM1 sub-tiles) and, in the mixed
                # form, every occupied sub-tile for the swap GEMM2.
                swapab_dispatch(
                    tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
                    num_non_exiting_tiles=b["out_num_non_exiting_tiles"],
                    group_rows=self.group_rows,
                    narrow_tile=self.n_tile,
                    wide_list=b["swap_wide_list"],
                    wide_count=b["swap_wide_count"],
                    narrow_list=b["swap_row_groups"],
                    narrow_count=b["swap_row_group_count"],
                    wide_min_rows=min(SWAP_HYBRID_DENSE_MIN_ROWS, self.group_rows),
                    wide_min_permille=SWAP_MIXED_WIDE_PERMILLE if self.mixed else 0,
                    all_list=b["swap_all_groups"] if self.mixed else None,
                    all_count=b["swap_all_count"] if self.mixed else None,
                    enable_pdl=pdl,
                    _prepared_launches=launches,
                )
                self._dispatch, self._dispatch_args = launches["swap_dispatch"]
            if self.hybrid or self.mixed:
                gemm1_lists = dict(
                    tile_idx_to_row_group=b["swap_row_groups"],
                    num_non_exiting_tiles=b["swap_row_group_count"],
                    group_rows=self.group_rows,
                    sf_blocked=not (self.mixed and SWAP_MIXED_SF_PLAIN),
                )
            token_idx = None
            if swap_row_tma(self.n_tile, True):
                fill_permuted_token_index(
                    b["out_permuted_idx_to_expanded_idx"],
                    b["permuted_idx_to_token_idx"],
                    num_tokens,
                    w.top_k,
                    _prepared_launches=launches,
                )
                self._token_index, self._token_index_args = launches["swap_token_index"]
                token_idx = b["permuted_idx_to_token_idx"]
            swapab_gemm1_situ(
                w1=w1,
                w1_sf=w1_sf,
                x=x,
                x_sf=x_sf,
                permuted_idx_to_token_idx=token_idx,
                permuted_idx_to_expanded_idx=b["out_permuted_idx_to_expanded_idx"],
                act=b["gemm1_out"],
                act_sf=b["gemm1_out_scale"],
                tile_idx_to_expert_idx=b["out_tile_idx_to_expert_idx"],
                tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
                alpha=b["w1_alpha"],
                beta=self._beta,
                linear_beta=self._linear_beta,
                top_k=w.top_k,
                zero_output=None,
                n_tile=self.n_tile,
                enable_pdl=pdl,
                pdl_trigger_after_wait=pdl and self._dep_prefetch,
                weight_l2_hint=w._swap_weight_l2_hint(num_tokens),
                _prepared_launches=launches,
                cluster_split_k=SWAP_GEMM1_CLUSTER_SPLIT
                and self._dep_prefetch
                # Only a rank with remote experts can see a single active
                # local group (top_k routes per token all land locally
                # otherwise), so the cluster launch pays off only there.
                and w.num_local_experts < w.num_experts,
                **{
                    "num_non_exiting_tiles": b["out_num_non_exiting_tiles"],
                    **gemm1_lists,
                },
            )
            if self.split:
                # Split form: dense gather GEMM1 tiles over the wide experts'
                # 128-row groups (slots listed by the routing kernel); they
                # write the block-scaled row scales the wide swap GEMM2 reads.
                gemm1_tactic = w._tactic(num_tokens)[1]
                if gemm1_tactic[0][0] != SWAP_SPLIT_WIDE_TILE:
                    raise ValueError(
                        "split-form GEMM1 tactic tile must match the "
                        f"{SWAP_SPLIT_WIDE_TILE}-row wide groups, got {gemm1_tactic!r}"
                    )
                gemm1_n = SWAP_SPLIT_GEMM1_N
                if gemm1_n is None:
                    for limit, value in SWAP_SPLIT_GEMM1_N_POLICY:
                        if num_tokens <= limit:
                            gemm1_n = value
                            break
                if gemm1_tactic[0][1] != gemm1_n:
                    gemm1_tactic = ((SWAP_SPLIT_WIDE_TILE, gemm1_n), (1, 1), False)
                wide_launches = {}
                blockscaled_contiguous_gather_grouped_gemm_act_fusion(
                    a=x,
                    b=w1,
                    a_scale=x_sf,
                    b_scale=w1_sf,
                    alpha=b["w1_alpha"],
                    tile_idx_to_expert_idx=b["swap_wide_expert"],
                    tile_idx_to_mn_limit=b["swap_wide_limit"],
                    token_id_mapping=b["out_permuted_idx_to_expanded_idx"],
                    num_non_exiting_tiles=b["swap_wide_count"],
                    tile_idx_to_row_group=b["swap_wide_list"],
                    out=b["gemm1_out"],
                    out_scale=b["gemm1_out_scale"],
                    c_dtype="float8_e4m3fn",
                    a_dtype="float8_e4m3fn",
                    b_dtype="float4_e2m1fn",
                    sf_dtype="float8_e8m0fnu",
                    sf_vec_size=32,
                    quantize_output=True,
                    topk=w.top_k,
                    mma_tiler_mn=gemm1_tactic[0],
                    cluster_shape_mn=gemm1_tactic[1],
                    enable_pdl=pdl,
                    activation_type=w.activation_type.value,
                    situ_beta=self._beta,
                    situ_linear_beta=self._linear_beta,
                    weight_l2_hint=DENSE_WEIGHT_L2_HINT,
                    pdl_trigger_early=pdl and SWAP_SPLIT_EARLY_TRIGGER,
                    _prepared_launches=wide_launches,
                )
                self._gemm1_dense = wide_launches["gather"]
                if split_dense:
                    # Wide GEMM2 on the dense finalize-fusion kernel over the
                    # same 128-row groups: route-weighted reduce-add into the
                    # zero-filled output; the finalize adds the narrow rows.
                    gemm2_tactic = w._tactic(num_tokens)[2]
                    gemm2_n = SWAP_SPLIT_GEMM2_N or gemm2_tactic[0][1]
                    if gemm2_tactic[0] != (SWAP_SPLIT_WIDE_TILE, gemm2_n):
                        gemm2_tactic = ((SWAP_SPLIT_WIDE_TILE, gemm2_n), (1, 1), False)
                    blockscaled_contiguous_grouped_gemm_finalize_fusion(
                        a=b["gemm1_out"],
                        b=w2,
                        a_scale=b["gemm1_out_scale"],
                        b_scale=w2_sf,
                        alpha=b["w2_alpha"],
                        tile_idx_to_expert_idx=b["swap_wide_expert"],
                        num_non_exiting_tiles=b["swap_wide_count"],
                        tile_idx_to_mn_limit=b["swap_wide_limit"],
                        permuted_idx_to_expanded_idx=b[
                            "out_permuted_idx_to_expanded_idx"
                        ],
                        token_final_scales=self._route_weights,
                        out=self.output,
                        a_dtype="float8_e4m3fn",
                        b_dtype="float4_e2m1fn",
                        sf_dtype="float8_e8m0fnu",
                        sf_vec_size=32,
                        out_dtype="bfloat16",
                        mma_tiler_mn=gemm2_tactic[0],
                        cluster_shape_mn=gemm2_tactic[1],
                        enable_pdl=pdl,
                        use_fused_finalize=True,
                        weight_l2_hint=DENSE_WEIGHT_L2_HINT,
                        tile_idx_to_row_group=b["swap_wide_list"],
                        pdl_trigger_early=pdl and SWAP_SPLIT_EARLY_TRIGGER,
                        _prepared_launches=wide_launches,
                    )
                    self._gemm2_wide = wide_launches["finalize"]
                else:
                    self._prepare_swap_gemm2(
                        w,
                        b,
                        w2,
                        w2_sf,
                        num_tokens,
                        fused_finalize,
                        wide_launches,
                        wide=True,
                    )
                    self._gemm2_wide = wide_launches["swap_gemm2"]
            if (
                self.hybrid or self.mixed
            ) and self.group_rows > SWAP_HYBRID_DENSE_MIN_ROWS:
                # Dense gather GEMM1 over the wide list (sort groups with more
                # than SWAP_HYBRID_DENSE_MIN_ROWS valid rows); it writes the
                # same E4M3 rows and blocked scales the swap sub-tiles write
                # for the narrow groups, so GEMM2 sees one contiguous layout.
                gemm1_tactic = w._tactic(num_tokens)[1]
                if gemm1_tactic[0][0] != self.group_rows:
                    raise ValueError(
                        "mixed-tile GEMM1 tactic tile must match the "
                        f"{self.group_rows}-row sort groups, got {gemm1_tactic!r}"
                    )
                blockscaled_contiguous_gather_grouped_gemm_act_fusion(
                    a=x,
                    b=w1,
                    a_scale=x_sf,
                    b_scale=w1_sf,
                    alpha=b["w1_alpha"],
                    tile_idx_to_expert_idx=b["out_tile_idx_to_expert_idx"],
                    tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
                    token_id_mapping=b["out_permuted_idx_to_expanded_idx"],
                    num_non_exiting_tiles=b["swap_wide_count"],
                    tile_idx_to_row_group=b["swap_wide_list"],
                    out=b["gemm1_out"],
                    out_scale=b["gemm1_out_scale"],
                    c_dtype="float8_e4m3fn",
                    a_dtype="float8_e4m3fn",
                    b_dtype="float4_e2m1fn",
                    sf_dtype="float8_e8m0fnu",
                    sf_vec_size=32,
                    quantize_output=True,
                    topk=w.top_k,
                    mma_tiler_mn=gemm1_tactic[0],
                    cluster_shape_mn=gemm1_tactic[1],
                    enable_pdl=pdl,
                    activation_type=w.activation_type.value,
                    situ_beta=self._beta,
                    situ_linear_beta=self._linear_beta,
                    weight_l2_hint=DENSE_WEIGHT_L2_HINT,
                    _prepared_launches=launches,
                )
                self._gemm1_dense = launches["gather"]
            if self.hybrid:
                # Dense contiguous grouped GEMM2 over the 128-row sort groups
                # with the bulk-reduce finalize into the zero-filled output.
                gemm2_tactic = w._tactic(num_tokens)[2]
                if gemm2_tactic[0][0] != self.group_rows:
                    raise ValueError(
                        "hybrid GEMM2 tactic tile must match the "
                        f"{self.group_rows}-row sort groups, got {gemm2_tactic!r}"
                    )
                blockscaled_contiguous_grouped_gemm_finalize_fusion(
                    a=b["gemm1_out"],
                    b=w2,
                    a_scale=b["gemm1_out_scale"],
                    b_scale=w2_sf,
                    alpha=b["w2_alpha"],
                    tile_idx_to_expert_idx=b["out_tile_idx_to_expert_idx"],
                    num_non_exiting_tiles=b["out_num_non_exiting_tiles"],
                    tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
                    permuted_idx_to_expanded_idx=b["out_permuted_idx_to_expanded_idx"],
                    token_final_scales=self._route_weights,
                    out=self.output,
                    a_dtype="float8_e4m3fn",
                    b_dtype="float4_e2m1fn",
                    sf_dtype="float8_e8m0fnu",
                    sf_vec_size=32,
                    out_dtype="bfloat16",
                    mma_tiler_mn=gemm2_tactic[0],
                    cluster_shape_mn=gemm2_tactic[1],
                    enable_pdl=pdl,
                    use_fused_finalize=True,
                    weight_l2_hint=DENSE_WEIGHT_L2_HINT,
                    _prepared_launches=launches,
                )
                launches["swap_gemm2"] = launches["finalize"]
            else:
                self._prepare_swap_gemm2(
                    w, b, w2, w2_sf, num_tokens, fused_finalize, launches
                )
            self._gemm1, self._gemm1_args = launches["swap_gemm1"]
            self._gemm2, self._gemm2_args = launches["swap_gemm2"]
            if self.two_stage:
                self._finalize_rows = plan_finalize_rows(
                    self._partial_rows,
                    b["out_expanded_idx_to_permuted_idx"],
                    self._route_weights,
                    self.output,
                    accumulate=split_dense,
                    skip_wide=(
                        (
                            b["out_num_non_exiting_tiles"],
                            b["swap_wide_count"],
                            self.n_tile,
                            SWAP_SPLIT_WIDE_TILE,
                        )
                        if split_dense
                        else None
                    ),
                )

    def _prepare_swap_gemm2(
        self, w, b, w2, w2_sf, num_tokens, fused_finalize, launches, wide=False
    ):
        if wide:
            # Split form, wide launch: the 128-row groups of the wide experts
            # (block-scaled row scales written by the dense GEMM1 tiles),
            # kernel defaults for the stage depth and weight grouping.
            with torch.cuda.device(self.device):
                swapab_gemm2(
                    w2=w2,
                    w2_sf=w2_sf,
                    act=b["gemm1_out"],
                    act_sf=b["gemm1_out_scale"],
                    out=self._partial_rows if self.two_stage else self.output,
                    tile_idx_to_expert_idx=b["swap_wide_expert"],
                    tile_idx_to_mn_limit=b["swap_wide_limit"],
                    num_non_exiting_tiles=b["swap_wide_count"],
                    tile_idx_to_row_group=b["swap_wide_list"],
                    alpha=b["w2_alpha"],
                    permuted_idx_to_expanded_idx=b["out_permuted_idx_to_expanded_idx"],
                    token_final_scales=self._route_weights if fused_finalize else None,
                    top_k=w.top_k,
                    finalize=fused_finalize,
                    n_tile=SWAP_SPLIT_WIDE_TILE,
                    sf_blocked=True,
                    enable_pdl=self._pdl,
                    weight_l2_hint=w._swap_weight_l2_hint(num_tokens),
                    pdl_trigger_early=self._pdl and SWAP_SPLIT_EARLY_TRIGGER,
                    _prepared_launches=launches,
                )
            return
        with torch.cuda.device(self.device):
            # Short GEMM2 stages (4 K blocks) pipeline deeper and win 6-10 us
            # on the wide expert-parallel shard while every expert fits one
            # row group (T <= 256); the 384-wide single stage of a narrow
            # MoE-TP shard is faster from T = 512 up and for few hot experts.
            gemm2_k_blocks = None
            if (
                num_tokens <= SWAP_GEMM2_SHORT_STAGE_MAX_TOKENS
                and w.intermediate_shard > SWAP_TWO_STAGE_MAX_SHARD
                and not os.environ.get("SWAPAB_KBLOCKS2")
            ):
                gemm2_k_blocks = 4
                # dep_prefetch_full_ring: with the dependent-side prefetch the
                # weight tile is resident before GEMM1 ends only if the stage
                # ring covers K; 8-block (256-wide) stages do for K = 3072
                # (12 stages). B300 EP8 decode rows: 1.02-1.03x -> 1.04-1.05x.
                if (
                    self._dep_prefetch
                    and w.intermediate_shard % 256 == 0
                    and w.intermediate_shard // 256 <= SWAP_MAX_AB_STAGES
                ):
                    gemm2_k_blocks = 8
            gemm2_m_group = None
            if (
                w.intermediate_shard <= SWAP_TWO_STAGE_MAX_SHARD
                and SWAP_TP_GEMM2_MGROUP > 1
                and num_tokens >= SWAP_TP_GEMM2_MGROUP_MIN_TOKENS
                and not os.environ.get("SWAPAB_MGROUP2")
            ):
                gemm2_m_group = SWAP_TP_GEMM2_MGROUP
                if not os.environ.get("SWAPAB_KBLOCKS2"):
                    gemm2_k_blocks = 4
            gemm2_lists = {"num_non_exiting_tiles": b["out_num_non_exiting_tiles"]}
            if self.mixed:
                # Every occupied n_tile-row sub-tile of the 128-row groups;
                # the row scales come blocked from the mixed GEMM1 tiles.
                # Grouped weight tiles keep 128-wide (4 K-block) stages, as
                # ``gemm2_k_blocks_per_stage`` does for ``swap_m_group`` > 1.
                if SWAP_MIXED_GEMM2_MGROUP > 1 and not os.environ.get(
                    "SWAPAB_KBLOCKS2"
                ):
                    gemm2_k_blocks = 4
                gemm2_m_group = SWAP_MIXED_GEMM2_MGROUP
                gemm2_lists = dict(
                    num_non_exiting_tiles=b["swap_all_count"],
                    tile_idx_to_row_group=b["swap_all_groups"],
                    group_rows=self.group_rows,
                    sf_blocked=not SWAP_MIXED_SF_PLAIN,
                )
            swapab_gemm2(
                w2=w2,
                w2_sf=w2_sf,
                act=b["gemm1_out"],
                act_sf=b["gemm1_out_scale"],
                out=self._partial_rows if self.two_stage else self.output,
                tile_idx_to_expert_idx=b["out_tile_idx_to_expert_idx"],
                tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
                alpha=b["w2_alpha"],
                permuted_idx_to_expanded_idx=b["out_permuted_idx_to_expanded_idx"],
                token_final_scales=self._route_weights if fused_finalize else None,
                top_k=w.top_k,
                finalize=fused_finalize,
                n_tile=self.n_tile,
                k_blocks_per_stage=gemm2_k_blocks,
                enable_pdl=self._pdl,
                weight_l2_hint=w._swap_weight_l2_hint(num_tokens),
                _prepared_launches=launches,
                m_group=gemm2_m_group,
                late_dep_wait=self._pdl and self._dep_prefetch,
                split_k=SWAP_GEMM2_SPLIT_K if self._dep_prefetch else 1,
                **gemm2_lists,
            )

    def run(self) -> torch.Tensor:
        """Enqueue three (T <= 16), four (deferred), five (two-stage
        finalize or split: + wide GEMM1/GEMM2), six (hybrid: sort, dispatch,
        swap and dense GEMM1 tiles, GEMM2) or up to seven (mixed / split +
        two-stage finalize) launches on the caller's stream."""
        with torch.cuda.device(self.device):
            stream_ptr = torch.cuda.current_stream().cuda_stream
            stream = cuda.CUstream(stream_ptr)
            self._route_preprocess.run(stream)
            if self._sort is not None:
                self._sort(*self._sort_args, stream_ptr)
            if self._dispatch is not None:
                self._dispatch(*self._dispatch_args, stream_ptr)
            if self._token_index is not None:
                self._token_index(*self._token_index_args, stream=stream)
            if self.split and self._side_stream is not None:
                # Fork: the wide chain runs on the plan's side stream after
                # the routing kernel; the narrow chain keeps its PDL edges.
                main = torch.cuda.current_stream()
                self._fork_event.record(main)
                self._side_stream.wait_event(self._fork_event)
                side = cuda.CUstream(self._side_stream.cuda_stream)
                compiled, args, kwargs = self._gemm1_dense
                compiled(*args, stream=side, **kwargs)
                compiled, args = self._gemm2_wide
                compiled(*args, stream=side)
                self._join_event.record(self._side_stream)
            elif self.split:
                # Wide launches first: when they find no groups, their CTAs
                # are resident during the routing kernel and leave at once.
                compiled, args, kwargs = self._gemm1_dense
                compiled(*args, stream=stream, **kwargs)
                compiled, args = self._gemm2_wide
                compiled(*args, stream=stream)
            self._gemm1(*self._gemm1_args, stream=stream)
            if self._gemm1_dense is not None and not self.split:
                compiled, args, kwargs = self._gemm1_dense
                compiled(*args, stream=stream, **kwargs)
            self._gemm2(*self._gemm2_args, stream=stream)
            if self._side_stream is not None:
                # Join before the finalize reads the wide GEMM2's output rows
                # (or before returning, when GEMM2 reduce-adds directly).
                torch.cuda.current_stream().wait_event(self._join_event)
            if self._finalize_rows is not None:
                self._finalize_rows.run(stream)
        return self.output


class CuteDslMxfp4MoEWrapper:
    """MXFP4 runner with offline tactics and explicit caller-owned workspace.

    The wrapper is metadata only. ``get_workspace_size(T)`` may be called
    before any CUDA allocation. ``plan`` compiles and performs warmup
    execution using valid caller inputs; it must run outside CUDA Graph
    capture. Its returned plan is used both for prefill and decode.

    ``offline_tactics`` maps token-count upper bounds to W4A8 tactic tuples.
    The smallest covering bucket is selected from host shape metadata. No
    serving-time tuning or device-to-host routing inspection is performed.
    An omitted table uses the conservative existing Blackwell tactic.

    ``num_experts`` and ``intermediate_size`` are the global model values.
    The rank-local layout is explicit: ``parallel_layout`` (uniform EP or MoE
    TP) and/or the expert interval ``num_local_experts``/``local_expert_offset``
    (the expert-parallel form). Both forms may be given if they agree. The
    derived ``layout`` fixes the weight shapes ``plan`` accepts; shapes never
    select the mode. Every rank computes a partial output; reduce externally.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        num_local_experts: Optional[int] = None,
        local_expert_offset: Optional[int] = None,
        parallel_layout: Optional[Mxfp4MoEParallelLayout] = None,
        activation_type: ActivationType = ActivationType.Situ,
        quantization: str = "mxfp4_w4a8",
        enable_pdl: bool = False,
        offline_tactics: Optional[dict] = None,
        swapab_max_tokens: Optional[int] = None,
        swapab_n_tile: int = SWAP_ROW_TILE,
        swapab_tile_policy=None,
        dense_dual_tile: Optional[bool] = None,
        dense_dual_tile_min_tokens: Optional[int] = None,
        dense_dual_tile_threshold_permille: Optional[int] = None,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = ActivationType(activation_type)
        self.quantization = quantization
        self.enable_pdl = enable_pdl
        # Swap-AB decode path: token counts up to ``swapab_max_tokens`` (0
        # disables) run weights-as-M grouped GEMMs with ``swapab_n_tile``-row
        # expert groups; the fused route preprocess bounds it to T <= 16.
        if swapab_max_tokens is not None and swapab_max_tokens < 0:
            raise ValueError("swapab_max_tokens must be >= 0")
        if swapab_n_tile not in (8, 16, 32, 64, 128):
            raise ValueError("swapab_n_tile must be 8, 16, 32, 64 or 128")
        self.swapab_max_tokens = swapab_max_tokens
        self.swapab_n_tile = swapab_n_tile
        # Dual-tile dense routing (see DENSE_DUAL_TILE). ``dense_dual_tile=True``
        # forces it for every dense token count (tests); ``False`` disables it.
        self.dense_dual_tile = dense_dual_tile
        self.dense_dual_tile_min_tokens = (
            DENSE_DUAL_TILE_MIN_TOKENS
            if dense_dual_tile_min_tokens is None
            else dense_dual_tile_min_tokens
        )
        self.dense_dual_tile_threshold_permille = (
            DENSE_DUAL_TILE_THRESHOLD_PERMILLE
            if dense_dual_tile_threshold_permille is None
            else dense_dual_tile_threshold_permille
        )
        if self.dense_dual_tile_threshold_permille <= 0:
            raise ValueError("dense_dual_tile_threshold_permille must be positive")
        # (max_tokens, rows per expert group) buckets for T > 16; T <= 16 uses
        # ``swapab_n_tile``. Weights are re-streamed once per group, so the
        # group width grows with the expected rows per local expert.
        policy = swapab_tile_policy
        if policy is None:
            env_policy = os.environ.get("SWAPAB_TILE_POLICY")
            if env_policy:
                # "256:8,0:32" (0 = no upper bound)
                policy = tuple(
                    (int(t) if int(t) > 0 else (1 << 62), int(n))
                    for t, n in (item.split(":") for item in env_policy.split(","))
                )
        if policy is not None:
            policy = tuple((int(t), int(n)) for t, n in policy)
            if any(n not in (8, 16, 32, 64, 128) for _, n in policy) or any(
                policy[i][0] >= policy[i + 1][0] for i in range(len(policy) - 1)
            ):
                raise ValueError(
                    "swapab_tile_policy must be ascending "
                    "(max_tokens, tile in {8,16,32,64,128})"
                )
        self._swapab_tile_policy = policy
        supported = mxfp4_moe_capability(
            gpu_arch=103,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            parallel_layout=parallel_layout,
            quantization=quantization,
            activation_type=self.activation_type,
        )
        if not supported.supported:
            raise ValueError(supported.reason)
        self.parallel_layout = parallel_layout
        self.layout = supported.layout
        self.num_local_experts = self.layout.num_local_experts
        self.local_expert_offset = self.layout.local_expert_offset
        self.intermediate_shard = self.layout.intermediate_shard
        if self.swapab_max_tokens is None:
            env_max = os.environ.get("SWAPAB_MAX_TOKENS")
            if env_max:
                self.swapab_max_tokens = int(env_max)
            else:
                # Measured on B300 (Kimi K3 geometry, two-stage finalize): the
                # swap path beats the dense grouped GEMMs up to T=1024 in both
                # layouts; beyond that every expert needs a second 32-row
                # group (weights re-streamed) and wider groups pay more in
                # the row operand and finalize than they save. The narrow
                # MoE-TP shard continues with the hybrid form (64-row swap
                # GEMM1 sub-tiles + dense finalize GEMM2) up to T=2048; at
                # T=4096 the dense path is faster again (the 64-row GEMM1
                # re-streams every expert's weights for 2-3 groups).
                self.swapab_max_tokens = (
                    2048 if SWAP_HYBRID and self.intermediate_shard < 1024 else 1024
                )
        if self._swapab_tile_policy is None:
            # Measured on B300 (Kimi K3). The group width follows the local
            # rows per expert (T/56 under balanced routing) so that no expert
            # needs a second group, while a wider group caps the weight
            # re-streaming of a hot expert; the narrow MoE-TP shard has 8x
            # the local rows of an EP8 rank at the same T.
            if self.intermediate_shard >= 1024:
                self._swapab_tile_policy = ((128, 16), (1 << 62, 32))
            elif SWAP_HYBRID:
                self._swapab_tile_policy = (
                    (128, 8),
                    (256, 16),
                    (1024, 32),
                    (1 << 62, SWAP_HYBRID_MIN_TILE),
                )
            else:
                self._swapab_tile_policy = ((128, 8), (256, 16), (1 << 62, 32))
        self.swapab_tile_policy = self._swapab_tile_policy
        self._offline_tactics = sorted(
            (int(limit), canonicalize_w4a8_tactic(tactic))
            for limit, tactic in (offline_tactics or {}).items()
        )
        if any(limit <= 0 for limit, _ in self._offline_tactics):
            raise ValueError("offline tactic bucket bounds must be positive")

    @property
    def parallel_mode(self) -> str:
        return self.layout.mode

    def _metadata(self):
        return dict(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            parallel_layout=self.parallel_layout,
            quantization=self.quantization,
            activation_type=self.activation_type,
        )

    def _tactic(self, num_tokens):
        for limit, tactic in self._offline_tactics:
            if num_tokens <= limit:
                return tactic
        if self.activation_type == ActivationType.Situ:
            table = (
                B300_SITU_DENSE_TACTIC_TABLE_NARROW
                if self.intermediate_shard < 1024
                else B300_SITU_DENSE_TACTIC_TABLE_WIDE
            )
            for limit, tactic in table:
                if num_tokens <= limit:
                    if tactic == B300_SITU_DENSE_DUAL_TACTIC and self._dual_enabled(
                        num_tokens
                    ):
                        # The routing picks M256 at run time; base is M128.
                        return _T128_N256_C2
                    return tactic
        return DEFAULT_BLACKWELL_MOE_TACTIC

    def _dual_enabled(self, num_tokens):
        if self.activation_type != ActivationType.Situ or self.dense_dual_tile is False:
            return False
        if self.dense_dual_tile is True:
            return True
        return (
            DENSE_DUAL_TILE
            and num_tokens > self.dense_dual_tile_min_tokens
            and self.intermediate_shard <= DENSE_DUAL_TILE_MAX_SHARD
        )

    def _gemm2_raster(self, num_tokens, gemm2_tile_n):
        """(raster_along_m, swizzle_size) of the dense GEMM2 (see
        DENSE_GEMM2_RASTER_M): ``raster_along_m`` is ``"auto"`` (device-side
        choice), True (M-fastest) or False (N-fastest). The swizzle groups N
        tiles, so it must divide the N tile count; otherwise the raster runs
        ungrouped."""
        if DENSE_GEMM2_RASTER_M == "auto":
            if not (
                self.intermediate_shard <= DENSE_GEMM2_RASTER_M_MAX_SHARD
                and num_tokens >= DENSE_GEMM2_RASTER_M_MIN_TOKENS
            ):
                return False, 1
            mode = "auto"
        elif DENSE_GEMM2_RASTER_M == "1":
            mode = True
        else:
            return False, 1
        n_tiles = -(-self.hidden_size // gemm2_tile_n)
        swizzle = (
            DENSE_GEMM2_SWIZZLE
            if DENSE_GEMM2_SWIZZLE > 0 and n_tiles % DENSE_GEMM2_SWIZZLE == 0
            else 1
        )
        return mode, swizzle

    def _dual_tactic(self, num_tokens):
        """Alternate (M256, two-CTA) dense tactic the routing may select at run
        time, or None when this token count runs one tile (see DENSE_DUAL_TILE)."""
        if not self._dual_enabled(num_tokens):
            return None
        base = self._tactic(num_tokens)
        if base[0] != 128:
            return None
        return B300_SITU_DENSE_DUAL_TACTIC

    def _use_swapab(self, num_tokens, do_finalize=True):
        # Deferred output exists only on the swap-AB path, at any token count.
        return (
            (1 <= num_tokens <= self.swapab_max_tokens or not do_finalize)
            and self.activation_type == ActivationType.Situ
            and (SWAP_PDL or not self.enable_pdl)
        )

    def _dense_two_stage(self, num_tokens):
        # Dense path: expanded-row GEMM2 output + finalize kernel instead of
        # the bulk reduce-add epilogue (experimental, env-gated).
        return DENSE_TWO_STAGE_FINALIZE and num_tokens > SWAP_ATOMIC_FINALIZE_MAX_TOKENS

    def _swap_weight_l2_hint(self, num_tokens):
        """L2 policy for the swap GEMMs' weight loads at this token count."""
        if num_tokens <= SWAP_WEIGHT_L2_HINT_MAX_TOKENS:
            return _DENSE_L2_HINTS["first"]
        lo, hi = SWAP_EP_L2HINT_SKIP
        if self.parallel_mode == "expert_parallel" and lo <= num_tokens <= hi:
            return None
        return _DENSE_L2_HINTS["first"]

    def _swap_tile(self, num_tokens):
        if num_tokens <= 16:
            return self.swapab_n_tile
        for max_tokens, tile in self.swapab_tile_policy:
            if num_tokens <= max_tokens:
                return tile
        return self.swapab_tile_policy[-1][1]

    def _swap_hybrid(self, num_tokens, do_finalize=True):
        """Hybrid form: swap GEMM1 sub-tiles of 128-row sort groups + dense
        finalize GEMM2 (wide swap tiles, finalize=True, generic routing)."""
        return (
            SWAP_HYBRID
            and bool(do_finalize)
            and num_tokens * self.top_k > SWAP_HYBRID_MIN_ROUTES
            and self._swap_tile(num_tokens) >= SWAP_HYBRID_MIN_TILE
        )

    def _swap_mixed(self, num_tokens, do_finalize=True):
        """Mixed form: 128-row sort groups with dense / swap GEMM1 tiles and
        the swap GEMM2 of the policy tile over every occupied sub-tile."""
        return (
            (
                SWAP_MIXED
                or (
                    num_tokens <= SWAP_MIXED_AUTO_MAX_TOKENS
                    and num_tokens * self.top_k <= self._fused_route_cap()
                )
            )
            and bool(do_finalize)
            and num_tokens >= SWAP_MIXED_MIN_TOKENS
            and (self.intermediate_shard < 1024 or SWAP_MIXED_EP)
            and not self._swap_hybrid(num_tokens, do_finalize)
            and SWAP_HYBRID_GROUP_ROWS % self._swap_tile(num_tokens) == 0
        )

    def _fused_route_cap(self, num_tokens=None, do_finalize=True):
        """Routes (T * top_k) the fused routing kernel handles for this
        rank. Its single CTA scales with the local expert count: on B300 it
        takes 7 / 13 us for 8192 / 16384 routes over 112 local experts
        (against 12-13 us for the conversion kernel plus ``moe_sort``) but
        11 / 15 us over the 896 experts of the MoE-TP shard (16 us for the
        conversion kernel plus ``moe_sort`` at 16384 routes), so the shard
        takes 16384 routes only where the split form needs the fused
        layout (T=1024: the split saves 127 us on concentrated routings
        for +0.4-0.7 % elsewhere)."""
        if self.num_local_experts <= FUSED_ROUTE_LARGE_MAX_LOCAL_EXPERTS:
            return FUSED_ROUTE_MAX_ROUTES_LARGE
        if num_tokens is not None and self._swap_split(num_tokens, do_finalize):
            return FUSED_ROUTE_MAX_ROUTES_LARGE
        return FUSED_ROUTE_MAX_ROUTES

    def _swap_split(self, num_tokens, do_finalize=True):
        """Split form: policy-tile groups for most experts plus 128-row groups
        (a second pair of swap GEMM launches) for the experts above
        SWAP_SPLIT_MIN_ROWS rows; fused routing, finalize only."""
        return (
            SWAP_SPLIT
            and bool(do_finalize)
            and SWAP_SPLIT_MIN_TOKENS <= num_tokens <= SWAP_SPLIT_MAX_TOKENS
            and num_tokens * self.top_k <= FUSED_ROUTE_MAX_ROUTES_LARGE
            and (self.intermediate_shard < 1024 or SWAP_SPLIT_EP)
            and self._swap_tile(num_tokens) < SWAP_SPLIT_WIDE_TILE
            and not self._swap_hybrid(num_tokens, do_finalize)
            and not self._swap_mixed(num_tokens, do_finalize)
        )

    def _swap_split_capacity(self, num_tokens):
        """``(wide_groups, rows)`` of the split layout: the narrow groups'
        rows rounded up to the wide tile plus the wide groups' rows. Experts
        need more than SWAP_SPLIT_MIN_ROWS rows to be wide, which bounds their
        number and, through ``get_max_num_tiles``, their 128-row groups."""
        tile = self._swap_tile(num_tokens)
        wide = SWAP_SPLIT_WIDE_TILE
        narrow_rows = (
            get_max_num_tiles(num_tokens, self.top_k, self.num_local_experts, tile)
            * tile
        )
        wide_experts = min(
            self.num_local_experts,
            num_tokens * self.top_k // (SWAP_SPLIT_MIN_ROWS + 1),
        )
        wide_groups = max(
            1,
            get_max_num_tiles(num_tokens, self.top_k, max(1, wide_experts), wide)
            if wide_experts
            else 0,
        )
        rows = -(-narrow_rows // wide) * wide + wide_groups * wide
        return wide_groups, rows

    def _swap_group_rows(self, num_tokens, do_finalize=True):
        if self._swap_hybrid(num_tokens, do_finalize) or self._swap_mixed(
            num_tokens, do_finalize
        ):
            return SWAP_HYBRID_GROUP_ROWS
        return self._swap_tile(num_tokens)

    def _workspace_fields(self, num_tokens, do_finalize=True):
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self._use_swapab(num_tokens, do_finalize):
            tile = self._swap_tile(num_tokens)
            hybrid = self._swap_hybrid(num_tokens, do_finalize)
            mixed = self._swap_mixed(num_tokens, do_finalize)
            split = self._swap_split(num_tokens, do_finalize)
            group = self._swap_group_rows(num_tokens, do_finalize)
            tiles = get_max_num_tiles(
                num_tokens, self.top_k, self.num_local_experts, group
            )
            rows = tiles * group
            if split:
                rows = self._swap_split_capacity(num_tokens)[1]
            wide_slots = rows // SWAP_SPLIT_WIDE_TILE
            specs = [
                ("out_tile_idx_to_expert_idx", (tiles,), torch.int32, 4),
                ("out_tile_idx_to_mn_limit", (tiles,), torch.int32, 4),
                *(
                    # Split form: the wide experts' 128-row groups (the list
                    # spans every slot: the dense gather GEMM1 requires it).
                    [
                        ("swap_wide_expert", (wide_slots,), torch.int32, 4),
                        ("swap_wide_limit", (wide_slots,), torch.int32, 4),
                        ("swap_wide_list", (wide_slots,), torch.int32, 4),
                        ("swap_wide_count", (1,), torch.int32, 4),
                    ]
                    if split
                    else []
                ),
                *(
                    # Dispatch work lists of the hybrid form.
                    [
                        ("swap_row_groups", (tiles * (group // tile),), torch.int32, 4),
                        ("swap_row_group_count", (1,), torch.int32, 4),
                        ("swap_wide_list", (tiles,), torch.int32, 4),
                        ("swap_wide_count", (1,), torch.int32, 4),
                    ]
                    if (hybrid or mixed)
                    else []
                ),
                *(
                    # Mixed form: the swap GEMM2 work list (every occupied sub-tile).
                    [
                        ("swap_all_groups", (tiles * (group // tile),), torch.int32, 4),
                        ("swap_all_count", (1,), torch.int32, 4),
                    ]
                    if mixed
                    else []
                ),
                # moe_sort scratch (T > 16 path; used by the sort for T > 1024)
                ("out_expert_counts", (2 * 4096,), torch.int32, 4),
                (
                    "out_expanded_idx_to_permuted_idx",
                    (num_tokens, self.top_k),
                    torch.int32,
                    4,
                ),
                ("out_permuted_idx_to_expanded_idx", (rows,), torch.int32, 4),
                *(
                    # gather4 row coordinates of the TMA row operand
                    [("permuted_idx_to_token_idx", (rows,), torch.int32, 4)]
                    if swap_row_tma(tile, True)
                    else []
                ),
                ("out_total_num_padded_tokens", (1,), torch.int32, 4),
                ("out_num_non_exiting_tiles", (1,), torch.int32, 4),
                ("gemm1_out", (rows, self.intermediate_shard), torch.float8_e4m3fn, 1),
                (
                    "gemm1_out_scale",
                    (rows, self.intermediate_shard // 32),
                    torch.uint8,
                    1,
                ),
                ("route_ids", (num_tokens, self.top_k), torch.int32, 4),
                ("route_weights", (num_tokens, self.top_k), torch.float32, 4),
                ("w1_alpha", (self.num_local_experts,), torch.float32, 4),
                ("w2_alpha", (self.num_local_experts,), torch.float32, 4),
            ]
            if (
                do_finalize
                and not hybrid
                and num_tokens > SWAP_ATOMIC_FINALIZE_MAX_TOKENS
                and self.intermediate_shard <= SWAP_TWO_STAGE_MAX_SHARD
            ):
                # Two-stage finalize: GEMM2's alpha-scaled rows in permuted
                # order, reduced by the finalize kernel.
                specs.append(
                    ("partial_rows", (rows, self.hidden_size), torch.bfloat16, 2)
                )
            fields, offset = [], 0
            for name, shape, dtype, itemsize in specs:
                offset = _align(offset)
                size = prod(shape) * itemsize
                fields.append(_WorkspaceField(name, shape, dtype, offset, size))
                offset += size
            return fields, _align(offset)
        tile = self._tactic(num_tokens)[0]
        dual = self._dual_tactic(num_tokens)
        # Dual-tile routing: the coarser tile's padding bounds the permuted
        # rows; the base list is sized by those rows.
        cap_tile = dual[0] if dual is not None else tile
        tiles = get_max_num_tiles(
            num_tokens, self.top_k, self.num_local_experts, cap_tile
        )
        rows = tiles * cap_tile
        base_tiles = rows // tile
        specs = [
            *(
                [
                    (
                        "partial_rows",
                        (num_tokens * self.top_k, self.hidden_size),
                        torch.bfloat16,
                        2,
                    )
                ]
                if self._dense_two_stage(num_tokens)
                else []
            ),
            ("out_tile_idx_to_expert_idx", (base_tiles,), torch.int32, 4),
            ("out_tile_idx_to_mn_limit", (base_tiles,), torch.int32, 4),
            *(
                [
                    ("out_alt_tile_idx_to_expert_idx", (tiles,), torch.int32, 4),
                    ("out_alt_tile_idx_to_mn_limit", (tiles,), torch.int32, 4),
                    ("out_alt_num_non_exiting_tiles", (1,), torch.int32, 4),
                    ("out_base_active_num_non_exiting_tiles", (1,), torch.int32, 4),
                ]
                if dual is not None
                else []
            ),
            (
                "out_expanded_idx_to_permuted_idx",
                (num_tokens, self.top_k),
                torch.int32,
                4,
            ),
            ("out_permuted_idx_to_expanded_idx", (rows,), torch.int32, 4),
            ("out_total_num_padded_tokens", (1,), torch.int32, 4),
            ("out_num_non_exiting_tiles", (1,), torch.int32, 4),
            ("gemm1_out", (rows, self.intermediate_shard), torch.float8_e4m3fn, 1),
            (
                "gemm1_out_scale",
                (32, 4, rows // 128, 4, self.intermediate_shard // 128, 1),
                torch.uint8,
                1,
            ),
            ("route_ids", (num_tokens, self.top_k), torch.int32, 4),
            ("route_weights", (num_tokens, self.top_k), torch.float32, 4),
            ("w1_alpha", (self.num_local_experts,), torch.float32, 4),
            ("w2_alpha", (self.num_local_experts,), torch.float32, 4),
        ]
        if num_tokens > 1024:
            specs.append(("out_expert_counts", (2 * self.num_experts,), torch.int32, 4))
        fields, offset = [], 0
        for name, shape, dtype, itemsize in specs:
            offset = _align(offset)
            size = prod(shape) * itemsize
            fields.append(_WorkspaceField(name, shape, dtype, offset, size))
            offset += size
        return fields, _align(offset)

    def get_workspace_size(self, num_tokens: int, do_finalize: bool = True) -> int:
        """Return required workspace bytes for any routing at this token count.

        The size follows the rank-local layout: the GEMM1 intermediate region
        uses ``intermediate_shard`` columns and the per-expert regions use
        ``num_local_experts``. ``do_finalize=False`` sizes the deferred-output
        (swap-AB) plan for this token count.
        """
        return self._workspace_fields(num_tokens, do_finalize)[1]

    def get_deferred_output_rows(self, num_tokens: int) -> int:
        """Rows of the caller-owned ``[rows, hidden_size]`` BF16 buffer that
        ``plan(..., do_finalize=False)`` writes: the permuted-row capacity
        (every local expert's rows padded to the row-group width) for any
        routing at this token count."""
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if not self._use_swapab(num_tokens, do_finalize=False):
            raise ValueError(
                "deferred output requires SiTU activation (the swap-AB path)"
            )
        tile = self._swap_tile(num_tokens)
        return (
            get_max_num_tiles(num_tokens, self.top_k, self.num_local_experts, tile)
            * tile
        )

    def plan(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        w1: torch.Tensor,
        w1_sf: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        *,
        beta: Optional[torch.Tensor] = None,
        linear_beta: Optional[torch.Tensor] = None,
        workspace: torch.Tensor,
        output: torch.Tensor,
        do_finalize: bool = True,
    ) -> Union[Mxfp4MoEPlan, Mxfp4MoESwapAbPlan]:
        """Bind buffers and prepare kernels; all tensor contents must be valid.

        Weights use ``prepare_cute_dsl_mxfp4_weights`` layouts for this rank's
        shard: ``[num_local_experts, 2*intermediate_shard, H/2]`` W1 and
        ``[num_local_experts, H, intermediate_shard/2]`` W2, as produced by
        ``shard_cute_dsl_mxfp4_weights`` for the resolved layout. Shapes are
        validated against that layout and never used to select it. ``x_sf`` is
        linear UE8M0 bytes [T,H/32]. ``beta`` and optional ``linear_beta`` are
        contiguous CUDA FP32 tensors with one value or one per local expert.
        Their values must be finite and positive; they are read on the device
        at execution, so changing them requires no recompilation.

        Routing may be separate int32 IDs and BF16/FP32 weights, or packed
        int32 (expert ID in high 16 bits, BF16 weight in low 16 bits) when
        ``topk_weights=None``. IDs are global, must be in ``[0, num_experts)``,
        and must be distinct within each token. Output and workspace must be
        distinct from all inputs; workspace is a contiguous uint8 tensor whose
        address is aligned to 256 bytes.

        ``do_finalize=False`` (deferred finalize, SiTU only) skips the
        route-weight reduction: ``output`` is a caller-owned contiguous BF16
        ``[rows, H]`` buffer with ``rows >= get_deferred_output_rows(T)``
        that receives ``alpha * GEMM2`` rows in permuted order, and the plan
        exposes ``expanded_idx_to_permuted_idx`` (int32 ``[T, top_k]``, ``-1``
        for non-local routes) and ``route_weights`` (FP32 ``[T, top_k]``),
        both valid after ``run``. Rows not referenced by the map are padding.
        """
        if x.device.type != "cuda":
            raise ValueError("plan requires CUDA tensors")
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("plan must be called before CUDA Graph capture")
            major, minor = torch.cuda.get_device_capability(x.device)
            capability = mxfp4_moe_capability(
                gpu_arch=major * 10 + minor, do_finalize=do_finalize, **self._metadata()
            )
            if not capability.supported:
                raise ValueError(capability.reason)
            num_tokens = x.shape[0]
            layout = self.layout
            if not do_finalize:
                if not self._use_swapab(num_tokens, do_finalize=False):
                    raise ValueError(
                        "deferred output requires SiTU activation (the swap-AB path)"
                    )
                deferred_rows = self.get_deferred_output_rows(num_tokens)
                if (
                    output.device != x.device
                    or output.dtype != torch.bfloat16
                    or output.ndim != 2
                    or output.shape[0] < deferred_rows
                    or output.shape[1] != self.hidden_size
                    or not output.is_contiguous()
                ):
                    raise ValueError(
                        "deferred output must be contiguous BF16 [>= "
                        f"{deferred_rows}, {self.hidden_size}] on {x.device}; got "
                        f"{output.dtype} {tuple(output.shape)} on {output.device}"
                    )
            expected = {
                "x": (x, (num_tokens, self.hidden_size), torch.float8_e4m3fn),
                "x_sf": (x_sf, (num_tokens, self.hidden_size // 32), torch.uint8),
                "topk_ids": (topk_ids, (num_tokens, self.top_k), torch.int32),
                "w1": (
                    w1,
                    (
                        layout.num_local_experts,
                        2 * layout.intermediate_shard,
                        self.hidden_size // 2,
                    ),
                    torch.uint8,
                ),
                "w2": (
                    w2,
                    (
                        layout.num_local_experts,
                        self.hidden_size,
                        layout.intermediate_shard // 2,
                    ),
                    torch.uint8,
                ),
            }
            if do_finalize:
                expected["output"] = (
                    output,
                    (num_tokens, self.hidden_size),
                    torch.bfloat16,
                )
            for name, (tensor, shape, dtype) in expected.items():
                if (
                    tensor.device != x.device
                    or tensor.dtype != dtype
                    or tuple(tensor.shape) != shape
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} must be contiguous {dtype} {shape} on {x.device} "
                        f"for parallel mode {layout.mode} ({layout.num_local_experts} "
                        f"local experts at offset {layout.local_expert_offset}, "
                        f"intermediate shard {layout.intermediate_shard}); got "
                        f"{tensor.dtype} {tuple(tensor.shape)} on {tensor.device}"
                    )
            for name, tensor, rows, columns in (
                ("w1_sf", w1_sf, 2 * layout.intermediate_shard, self.hidden_size),
                ("w2_sf", w2_sf, self.hidden_size, layout.intermediate_shard),
            ):
                expected_shape = (
                    32,
                    4,
                    rows // 128,
                    4,
                    columns // 128,
                    layout.num_local_experts,
                )
                if (
                    tensor.device != x.device
                    or tensor.dtype != torch.uint8
                    or tuple(tensor.shape) != expected_shape
                ):
                    raise ValueError(
                        f"{name} must be the prepared uint8 MMA scale layout "
                        f"{expected_shape} on {x.device} for parallel mode "
                        f"{layout.mode}; got {tensor.dtype} {tuple(tensor.shape)} "
                        f"on {tensor.device}"
                    )
            if topk_weights is not None and (
                topk_weights.device != x.device
                or topk_weights.dtype not in (torch.bfloat16, torch.float32)
                or tuple(topk_weights.shape) != (num_tokens, self.top_k)
                or not topk_weights.is_contiguous()
            ):
                raise ValueError(
                    "topk_weights must be contiguous CUDA BF16/FP32 [T,top_k]"
                )
            if self.activation_type == ActivationType.Situ and beta is None:
                raise ValueError("SiTU requires runtime beta")
            if self.activation_type != ActivationType.Situ and (
                beta is not None or linear_beta is not None
            ):
                raise ValueError("SiTU parameters require ActivationType.Situ")
            for name, tensor in (("beta", beta), ("linear_beta", linear_beta)):
                if tensor is not None and (
                    tensor.device != x.device
                    or tensor.dtype != torch.float32
                    or tensor.ndim != 1
                    or tensor.numel() not in (1, self.num_local_experts)
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} must be CUDA FP32 [1] or [num_local_experts]"
                    )
            fields, size = self._workspace_fields(num_tokens, do_finalize)
            if (
                workspace.device != x.device
                or workspace.dtype != torch.uint8
                or workspace.ndim != 1
                or not workspace.is_contiguous()
                or workspace.numel() < size
                or workspace.data_ptr() % 256
            ):
                raise ValueError(
                    f"workspace requires at least {size} aligned CUDA uint8 bytes"
                )
            workspace_interval = (workspace.data_ptr(), workspace.data_ptr() + size)
            output_interval = _byte_interval(output)
            if _overlap(workspace_interval, output_interval):
                raise ValueError("output must not overlap workspace")
            for name, tensor in (
                ("x", x),
                ("x_sf", x_sf),
                ("topk_ids", topk_ids),
                ("topk_weights", topk_weights),
                ("w1", w1),
                ("w1_sf", w1_sf),
                ("w2", w2),
                ("w2_sf", w2_sf),
                ("beta", beta),
                ("linear_beta", linear_beta),
            ):
                if tensor is not None and (
                    _overlap(workspace_interval, _byte_interval(tensor))
                    or _overlap(output_interval, _byte_interval(tensor))
                ):
                    raise ValueError(f"output/workspace must not overlap {name}")
            buffers = {
                f.name: workspace.narrow(0, f.offset, f.nbytes)
                .view(f.dtype)
                .view(f.shape)
                for f in fields
            }
            buffers["w1_alpha"].fill_(1.0)
            buffers["w2_alpha"].fill_(1.0)
            route_weights = (
                topk_weights
                if topk_weights is not None and topk_weights.dtype == torch.float32
                else buffers["route_weights"]
            )
            validate_w4a8_inputs(x, x_sf, route_weights, w1, w1_sf, w2, w2_sf)
            if w1_sf.device != x.device or w2_sf.device != x.device:
                raise ValueError("weight scales must be on the input device")
            if self._use_swapab(num_tokens, do_finalize):
                if (major, minor) not in ((10, 0), (10, 3)):
                    raise ValueError("the swap-AB decode path requires SM100/SM103")
                swap_plan = Mxfp4MoESwapAbPlan(
                    wrapper=self,
                    buffers=buffers,
                    workspace=workspace,
                    x=x,
                    x_sf=x_sf,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    w1=w1,
                    w1_sf=w1_sf,
                    w2=w2,
                    w2_sf=w2_sf,
                    beta=beta,
                    linear_beta=linear_beta,
                    output=output,
                    n_tile=self._swap_tile(num_tokens),
                    finalize=do_finalize,
                )
                swap_plan._prepare()
                return swap_plan
            tile, gemm1, gemm2 = self._tactic(num_tokens)
            gemm2_raster = self._gemm2_raster(num_tokens, gemm2[0][1])
            dual = self._dual_tactic(num_tokens)
            # The public routing contract requires distinct IDs per token,
            # so an expert has at most T rows. Restrict this specialization
            # to the qualified B300 SiTU decode tactic.
            decode_specialization = (
                (major, minor) == (10, 3)
                and 1 <= num_tokens <= 16
                and self.activation_type == ActivationType.Situ
                and not self.enable_pdl
                and tile == 128
                and gemm1 == ((128, 128), (1, 1), False)
                and gemm2 == ((128, 128), (1, 1), False)
            )
            route_ids = buffers["route_ids"] if topk_weights is None else topk_ids
            dense_two_stage = self._dense_two_stage(num_tokens)
            kwargs = dict(
                x=x,
                x_sf=x_sf,
                token_selected_experts=route_ids,
                token_final_scales=route_weights,
                w1_weight=w1,
                w1_weight_sf=w1_sf,
                w1_alpha=buffers["w1_alpha"],
                fc2_input_scale=None,
                w2_weight=w2,
                w2_weight_sf=w2_sf,
                w2_alpha=buffers["w2_alpha"],
                num_experts=self.num_experts,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_offset,
                tile_size=tile,
                gemm1_mma_tiler_mn=gemm1[0],
                gemm1_cluster_shape_mn=gemm1[1],
                gemm2_mma_tiler_mn=gemm2[0],
                gemm2_cluster_shape_mn=gemm2[1],
                dual_tile_size=dual[0] if dual is not None else 0,
                dual_gemm1_mma_tiler_mn=dual[1][0] if dual is not None else (256, 256),
                dual_gemm1_cluster_shape_mn=dual[1][1] if dual is not None else (2, 1),
                dual_gemm2_mma_tiler_mn=dual[2][0] if dual is not None else (256, 256),
                dual_gemm2_cluster_shape_mn=dual[2][1] if dual is not None else (2, 1),
                dual_tile_threshold_permille=self.dense_dual_tile_threshold_permille,
                gemm2_raster_along_m=gemm2_raster[0],
                gemm2_swizzle_size=gemm2_raster[1],
                dual_alt_pdl=DENSE_DUAL_ALT_PDL,
                moe_sort_buffers={
                    name: value
                    for name, value in buffers.items()
                    if name.startswith("out_")
                },
                gemm1_out=buffers["gemm1_out"],
                gemm1_out_scale=buffers["gemm1_out_scale"],
                moe_output=output,
                output_dtype=torch.bfloat16,
                use_async_memset=False,
                zero_fill_counters=(
                    torch.zeros(2, dtype=torch.int32, device=output.device)
                    if not dense_two_stage
                    and _dense_fill_in_gemm1(
                        self.num_experts, self.num_local_experts, num_tokens
                    )
                    else None
                ),
                use_fused_finalize=not dense_two_stage,
                gemm2_partial_out=buffers["partial_rows"] if dense_two_stage else None,
                skip_unpermute=dense_two_stage,
                weight_l2_hint=DENSE_WEIGHT_L2_HINT,
                enable_pdl=self.enable_pdl,
                activation_type=self.activation_type.value,
                situ_beta=beta,
                situ_linear_beta=linear_beta,
                _enable_decode_specialization=decode_specialization,
            )
            plan = Mxfp4MoEPlan(
                kwargs=kwargs,
                workspace=workspace,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                route_ids=route_ids,
                route_weights=route_weights,
            )
            plan._prepare()
            return plan
