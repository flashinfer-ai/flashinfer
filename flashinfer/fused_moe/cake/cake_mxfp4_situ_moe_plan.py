#
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# mypy: ignore-errors
# The decision table below is the producer's plan source rendered verbatim; its typing is
# maintained upstream, not in this generated copy.
"""Host plan of the Cake MXFP4 x MXFP8 SiTU routed MoE (Kimi K3 W4A8), ``backend="cake"``.

Generated: the decision table below (constants, tile policy, workspace layout, weight
preparation and the routing configuration) is the Cake plan's own source, rendered
verbatim from the pinned Cake revision recorded in the package manifest; the launch
section binds it to the generated kernels of this package.  Do not edit.

The plan reproduces the hand-written CuTe DSL MXFP4 swap-AB plan of FlashInfer's
``fused_moe.cute_dsl.mxfp4`` module decision for decision (every ``:N`` reference below
is a line of that module): ``swapab_max_tokens``, the tile policy, group rows, stage
depths, weight grouping, the fused-routing cap, the PDL chain, the workspace layout and
the launch sequence.  Rows whose hand-written selection has no traced Cake form (mixed-192) are refused
by :meth:`CakeSwapAbPolicy.decide` with the reason, never substituted (so are the dense
rows whose chain the Cake runner does not enqueue, named by their reason); rows whose
selection is traced but not run by a shipped executable chain are refused by
:meth:`CakeMxfp4MoEWrapper.plan` (``EXECUTABLE_PATHS``).  A launch whose kernel this
package's backend does not build (the cooperative ``moe_sort`` pair under the CuTe DSL
package) is loaded from the sibling package at the same pinned Cake revision
(``SIBLING_MODULE``; the manifest's ``sibling_backend_forms``), never re-implemented.

Numerics are the hand-written ones: FP32 accumulation, the SiTU FP32 sequence, UE8M0
round-up requantization and BF16 route-weight scaling before the reduce-add.  The
device-side split-K of GEMM2 and the cluster split-K of GEMM1 are not built, so every
work item accumulates its whole K in FP32 before the single BF16 reduce-add (an
accumulation-order deviation only; recorded in the ``GemmForm`` fields).

Usage::

    from flashinfer.fused_moe.cake import CakeMxfp4MoEWrapper, prepare_cake_mxfp4_weights

    weights = prepare_cake_mxfp4_weights(w1, w1_scale, w2, w2_scale)      # once per rank, pure torch
    runner = CakeMxfp4MoEWrapper(num_experts, top_k, hidden_size, intermediate_size,
                                 num_local_experts=..., local_expert_offset=...)   # or intermediate_shard=...
    workspace = torch.empty(runner.get_workspace_size(T), dtype=torch.uint8, device="cuda")
    plan = runner.plan(x, x_sf, topk_ids, topk_weights, weights["w1"], weights["w1_sf"],
                       weights["w2"], weights["w2_sf"], beta=beta, linear_beta=linear_beta,
                       workspace=workspace, output=out)                    # outside graph capture
    plan.run()                                                             # graph-capturable
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

PACKAGE_BACKEND = "cake"
SIBLING_MODULE = "flashinfer.fused_moe.cake_cute"
# The hand-written paths the shipped executable chains run (the producer plan's declaration at the pinned
# revision): ``plain`` = ``CakeSwapAbPlan.run`` (routing -> swap-AB GEMM1 -> swap-AB GEMM2 finalize), ``dense`` =
# ``CakeDensePlan.run`` (route preprocess -> moe_sort pair -> dense GEMM1 -> dense finalize GEMM2),
# ``split_two_stage`` = ``CakeSplitPlan.run`` (split routing -> wide row-group GEMM1 / GEMM2 -> narrow swap-AB
# GEMM1 / partial GEMM2 -> finalize rows), ``hybrid`` = ``CakeHybridPlan.run`` (route preprocess -> moe_sort pair ->
# dispatch -> swap-AB row-group GEMM1 -> dense row-group GEMM1 -> dense finalize GEMM2).  ``plan()`` refuses rows
# whose decision selects another path, although
# every launch of such a row has a generated form (see the manifest's ``carried_rows`` /
# ``carried_rows_mixed_backend`` / ``uncarried_rows``).
EXECUTABLE_PATHS = ('plain', 'dense', 'split_two_stage', 'hybrid')


SWAP_ATOMIC_FINALIZE_MAX_TOKENS = 16        # mxfp4.py:63-65


SWAP_TWO_STAGE_MAX_SHARD = 512              # mxfp4.py:69


SWAP_GEMM2_SHORT_STAGE_MAX_TOKENS = 256     # mxfp4.py:72-74


SWAP_HYBRID = True                          # mxfp4.py:98


SWAP_HYBRID_GROUP_ROWS = 128                # mxfp4.py:99


SWAP_HYBRID_MIN_ROUTES = 4096               # mxfp4.py:101


SWAP_HYBRID_MIN_TILE = 64                   # mxfp4.py:102


SWAP_MIXED = False                          # mxfp4.py:119


SWAP_PDL = True                             # mxfp4.py:125


SWAP_SPLIT = True                           # mxfp4.py:141


SWAP_SPLIT_WIDE_TILE = 128                  # mxfp4.py:142


SWAP_SPLIT_MIN_ROWS = 64                    # mxfp4.py:143


SWAP_SPLIT_MIN_TOKENS = 128                 # mxfp4.py:149


SWAP_SPLIT_MAX_TOKENS = 1024                # mxfp4.py:154


SWAP_SPLIT_EP = False                       # mxfp4.py:155


SWAP_DEP_PREFETCH = True                    # mxfp4.py:165


SWAP_GEMM2_SPLIT_K = 2                      # mxfp4.py:169


SWAP_GEMM1_CLUSTER_SPLIT = True             # mxfp4.py:173


SWAP_MIXED_MIN_TOKENS = 128                 # mxfp4.py:206


SWAP_MIXED_AUTO_MAX_TOKENS = 0              # mxfp4.py:211


SWAP_MIXED_EP = False                       # mxfp4.py:215


SWAP_WIDE192_LAYOUTS = ("moe_tensor_parallel",)  # mxfp4.py:232-236 (mode "auto")


SWAP_WIDE192_MIN_TOKENS = 8192              # mxfp4.py:237


SWAP_TP_GEMM2_MGROUP = 2                    # mxfp4.py:334


SWAP_TP_GEMM2_MGROUP_MIN_TOKENS = 16        # mxfp4.py:335-337


SWAP_WEIGHT_L2_HINT_MAX_TOKENS = 16         # mxfp4.py:345


SWAP_EP_L2HINT_SKIP = (17, 255)             # mxfp4.py:346-349


TMA_L2_EVICT_FIRST = 0x12F0000000000000     # mxfp4.py:91 / swapab_moe.py:77


SWAP_ROW_TILE = 8                           # swapab_moe.py:34


SWAP_K_BLOCKS_PER_STAGE = 4                 # swapab_moe.py:37


SWAP_MAX_AB_STAGES = 12                     # swapab_moe.py:90


SWAP_TILED_WEIGHTS = 3                      # swapab_moe.py:101 (tile-major W1 and W2)


FUSED_ROUTE_MAX_ROUTES = 8192               # mxfp4_routing.py:262


FUSED_ROUTE_MAX_ROUTES_LARGE = 16384        # mxfp4_routing.py:263


FUSED_ROUTE_LARGE_MAX_LOCAL_EXPERTS = 128   # mxfp4_routing.py:264-266


WORKSPACE_ALIGN = 256                       # mxfp4.py:805-806 (_align)


SORT_SCRATCH_WORDS = 2 * 4096               # mxfp4.py:2714 (out_expert_counts)


UNBOUNDED = 1 << 62                         # mxfp4.py:2318 (policy sentinel)


DENSE_WEIGHT_L2_HINT = TMA_L2_EVICT_FIRST   # mxfp4.py:86-91 (MXFP4_DENSE_L2HINT=first)


SWAP_HYBRID_DENSE_MIN_ROWS = 64             # mxfp4.py:110


SWAP_SPLIT_MIN_PERMILLE = 250               # mxfp4.py:146


SWAP_SPLIT_EARLY_TRIGGER = True             # mxfp4.py:156


SWAP_SPLIT_GEMM1_N_POLICY = ((512, 128), (UNBOUNDED, 256))  # mxfp4.py:174


SWAP_SPLIT_DENSE_GEMM2 = True               # mxfp4.py:181


SWAP_SPLIT_SIDE_STREAM = False              # mxfp4.py:186


SWAP_MIXED_FUSED_LISTS = True               # mxfp4.py:205


SWAP_MIXED_WIDE_PERMILLE = 0                # mxfp4.py:210


SWAP_MIXED_SF_PLAIN = False                 # mxfp4.py:211


SWAP_WIDE192_TILE = 192                     # mxfp4.py:236-238


SWAP_WIDE192_MIXED = True                   # mxfp4.py:246 (MXFP4_SWAP192_MIXED=1)


SWAP_WIDE192_ROW_UNIT = 64                  # mxfp4.py:247


SWAP_WIDE192_MIXED_GEMM2 = "auto"           # mxfp4.py:259


SWAP_WIDE192_DENSE_GEMM2_MIN_TOKENS = 16384  # mxfp4.py:262-264


SWAP_WIDE192_MIXED_STREAMS = "win"          # mxfp4.py:281


SWAP_WIDE192_ZERO_FILL = "dense"            # mxfp4.py:290


SWAP_WIDE192_LISTS = "sort"                 # mxfp4.py:297


SWAP_WIDE192_MAX_ROWS = 0                   # mxfp4.py:306


SWAP_WIDE192_MIN_ROWS = 1024                # mxfp4.py:307


SWAP_WIDE192_DENSE_FIRST_MIN_TOKENS = 16384  # mxfp4.py:315-317


SWAP_MIXED_GEMM2_MGROUP = 2                 # mxfp4.py:321


B300_SITU_DENSE_TACTIC = (128, ((128, 256), (1, 1), False), ((128, 256), (1, 1), False))   # :353


_T128_N256_C2 = (128, ((128, 256), (1, 1), False), ((128, 256), (1, 2), False))            # :376


_T128_N192 = (128, ((128, 256), (1, 1), False), ((128, 192), (1, 1), False))               # :377


_T128_N192_C2 = (128, ((128, 256), (1, 1), False), ((128, 192), (1, 2), False))            # :378


_T256_N256_C1 = (256, ((256, 256), (2, 1), False), ((256, 256), (2, 1), False))            # :384


B300_SITU_DENSE_TACTIC_TABLE_WIDE = ((8192, _T128_N192), (16384, _T128_N192_C2), (UNBOUNDED, B300_SITU_DENSE_TACTIC))  # :379-383


B300_SITU_DENSE_TACTIC_TABLE_NARROW = ((2048, B300_SITU_DENSE_TACTIC), (7168, _T128_N256_C2), (14336, _T256_N256_C1),
                                       (UNBOUNDED, _T128_N192_C2))                         # :385-390


B300_SITU_DENSE_DUAL_TACTIC = _T256_N256_C1  # :525


DENSE_DUAL_TILE = True                      # mxfp4.py:408


DENSE_GEMM1_CLUSTER_SPLIT = True            # mxfp4.py:415-417


DENSE_DUAL_TILE_MIN_TOKENS = 7168           # mxfp4.py:418-420


DENSE_DUAL_TILE_MAX_SHARD = 3072            # mxfp4.py:430-432


DENSE_DUAL_TILE_THRESHOLD_PERMILLE = 1100   # mxfp4.py:433-435


DENSE_ASYNC_MEMSET = "ep"                   # mxfp4.py:448


DENSE_FILL_IN_GEMM1 = "1"                   # mxfp4.py:467


DENSE_FILL_IN_GEMM1_MIN_TOKENS = 8192       # mxfp4.py:471-473


ROUTE_PREPROCESS_PDL = True                 # mxfp4.py:478


DENSE_TWO_STAGE_FINALIZE = False            # mxfp4.py:528


DENSE_GEMM2_RASTER_M = "auto"               # mxfp4.py:548


DENSE_GEMM2_SWIZZLE = 4                     # mxfp4.py:549


DENSE_GEMM2_C_STAGES = 1                    # mxfp4.py:554


DENSE_GEMM1_A_TMA = False                   # mxfp4.py:557


DENSE_GEMM2_RASTER_M_MAX_SHARD = 512        # mxfp4.py:558-560


DENSE_GEMM2_RASTER_M_MIN_TOKENS = 16384     # mxfp4.py:561-563


DENSE_DUAL_ALT_PDL = False                  # mxfp4.py:573


MOE_SORT_EXPERT_COUNTS_MIN_TOKENS = 1024    # moe_utils.py:205-227


DENSE_CHAIN_EXECUTABLE = True


DENSE_CHAIN_MISSING = ("dense chain not executable in the Cake-tree runner: CakeSwapAbPlan enqueues the plain "
                       "fused-routing chain only (contract rows T=2048/4096 raised 'routes exceed the fused routing "
                       "cap', no kernel launched; forms traced)")


DENSE_TWO_STAGE_CHAIN_MISSING = ("dense two-stage chain not executable in the Cake-tree runner: the dense two-stage "
                                 "finalize_rows (DENSE_TWO_STAGE_FINALIZE) is traced but not enqueued by CakeDensePlan")


SPLIT_CHAIN_EXECUTABLE = True


HYBRID_CHAIN_EXECUTABLE = True


MOE_SORT_EXPERT_TIERS = (128, 160, 256, 384, 512, 576, 896, 1024)   # RoutingCustomPolicy.cuh:748-770


MOE_SORT_CAKE_TIERS = (128, 160, 256, 384, 512, 576, 896, 1024)     # kimi_k3_mxfp4_situ_moe_sort.EXPERT_TIERS


MOE_SORT_RESERVED_SMS = 8                   # RoutingKernel.cuh:49 (numBlocksCoop = SMs - 8, getCoopLaunchSMCounts)


MOE_SORT_GENERIC_PER_THREAD = 64            # RoutingKernel.cuh:1205-1207 MaxExpandedIdxPerThread (NumTop8Experts kernel)


MOE_SORT_BOUNDED_PER_THREAD = 4             # ... of the NumTop16Experts kernel of the 512..1024 tiers


MOE_SORT_HIGH_EXPERT_TIERS = (512, 1024)    # RoutingKernelTopK.cuh:326-337 isInHighExpertLaneOwnedTopKRange


MOE_SORT_HIGH_EXPERT_TOPK = (9, 16)


MOE_SORT_CLUSTER_MAX_TOKENS = 1024          # routing_common.cu:98 ClusterKernelPreferredMaxNumTokens (single-cluster kernel)


MOE_SORT_CONTIGUOUS_WINDOW_MIN_TOKENS = 65536   # mUseContiguousRouteWindows (kimi_k3_mxfp4_situ_moe_sort.CONTIGUOUS_WINDOW_MIN_TOKENS)


CONTRACT_SM_COUNT = 148                     # B300 (sm_103a): the contract's device; the K6 grid / state selection take it explicitly


ROUTE_PREPROCESS_THREADS = 256              # mxfp4_routing.py:908 (_plan_route_preprocess default)


FINALIZE_ROWS_THREADS = 256                 # mxfp4_finalize.py:297 (plan_finalize_rows default)


PARALLEL_MODES = ("single", "expert_parallel", "moe_tensor_parallel")


def moe_sort_expert_tier(num_experts: int) -> int:
    """``routingCustom::getMaxNumExperts`` (RoutingCustomPolicy.cuh:748-770) for the tiers this family can hit."""
    for tier in MOE_SORT_EXPERT_TIERS:
        if num_experts <= tier:
            return tier
    raise ValueError("num_experts above 1024 is outside the Kimi K3 family")


def align(size: int) -> int:
    """``mxfp4.py:805-806``."""
    return (size + WORKSPACE_ALIGN - 1) // WORKSPACE_ALIGN * WORKSPACE_ALIGN


def get_max_num_tiles(num_tokens: int, top_k: int, num_local_experts: int, tile_size: int) -> int:
    """``moe_utils.get_max_num_tiles``: tight bound on ``sum_e ceil(rows_e / tile)``."""
    routes = num_tokens * top_k
    if routes <= num_local_experts:
        return routes
    return (routes + (tile_size - 1) * num_local_experts) // tile_size


def gemm1_k_blocks_per_stage(n_tile: int) -> int:
    """``swapab_moe.py:40-47``: 8-row groups take 256-wide stages, wider groups 128-wide."""
    return 8 if n_tile == 8 else SWAP_K_BLOCKS_PER_STAGE


def swap_two_cta(n_tile: int) -> bool:
    """``swapab_moe.py:177-189`` (default ``SWAPAB_TWO_CTA=1``)."""
    return n_tile == 192


def swap_row_tma(n_tile: int, gather_rows: bool = True) -> bool:
    """``swapab_moe.py:161-175``: TMA row operand only at ``n_tile >= 64`` for the contiguous GEMM2 rows;
    the plan calls it with ``gather_rows=True`` (``mxfp4.py:1410`` / ``:2716``), so it is never enabled."""
    if n_tile == 192 or swap_two_cta(n_tile):
        return False
    return n_tile >= 64 and not gather_rows


def swap_m_group(n_tile: int, gemm2: bool = False) -> int:
    """``swapab_moe.py:200-215``: two weight M-tiles per GEMM2 work item for the 64-row form."""
    return 2 if (n_tile == 64 and gemm2) else 1


def gemm2_k_blocks_per_stage(k: int, n_tile: int = 8) -> int:
    """``swapab_moe.py:55-71`` (launcher default when the plan passes ``k_blocks_per_stage=None``)."""
    if swap_m_group(n_tile, gemm2=True) > 1:
        return 4
    if n_tile >= 192:
        return 4
    if k % 384 == 0:
        return 12
    return SWAP_K_BLOCKS_PER_STAGE


@dataclass(frozen=True)
class RankLayout:
    """``Mxfp4MoERankLayout`` (mxfp4.py:606-631): rank-local geometry from explicit metadata."""

    mode: str
    num_experts: int
    intermediate_size: int
    num_local_experts: int
    local_expert_offset: int
    intermediate_shard: int

    @property
    def gemm1_n(self) -> int:
        return 2 * self.intermediate_shard

    @property
    def gemm2_k(self) -> int:
        return self.intermediate_shard


def resolve_layout(num_experts: int, intermediate_size: int, *, num_local_experts: int | None = None,
                   local_expert_offset: int | None = None, intermediate_shard: int | None = None) -> RankLayout:
    """``resolve_mxfp4_moe_layout`` (mxfp4.py:634-716), explicit expert-interval / shard form.

    ``intermediate_shard`` < ``intermediate_size`` selects the MoE-TP shard (every expert local); a proper
    expert interval selects expert parallelism; everything local is ``single``.
    """
    if num_experts <= 0 or intermediate_size <= 0:
        raise ValueError("num_experts and intermediate_size must be positive")
    local = num_experts if num_local_experts is None else int(num_local_experts)
    offset = 0 if local_expert_offset is None else int(local_expert_offset)
    shard = intermediate_size if intermediate_shard is None else int(intermediate_shard)
    if local <= 0 or offset < 0 or offset + local > num_experts:
        raise ValueError("local experts must form a nonempty contiguous global expert interval")
    if shard <= 0 or shard > intermediate_size or intermediate_size % shard or shard % 128:
        raise ValueError("the intermediate shard must divide I into multiples of 128")
    if shard < intermediate_size:
        if local != num_experts or offset:
            raise ValueError("hybrid expert/tensor parallelism is unsupported")
        return RankLayout("moe_tensor_parallel", num_experts, intermediate_size, local, 0, shard)
    if local == num_experts and offset == 0:
        return RankLayout("single", num_experts, intermediate_size, local, 0, intermediate_size)
    return RankLayout("expert_parallel", num_experts, intermediate_size, local, offset, intermediate_size)


@dataclass(frozen=True)
class WorkspaceField:
    """``_WorkspaceField`` (mxfp4.py:796-802)."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    nbytes: int


_ITEMSIZE = {"int32": 4, "float32": 4, "uint8": 1, "float8_e4m3fn": 1, "bfloat16": 2}


def _layout(specs) -> tuple[list[WorkspaceField], int]:
    """``_workspace_fields`` packing (mxfp4.py:2755-2761 / :2826-2831): 256-byte aligned offsets, aligned total."""
    fields, offset = [], 0
    for name, shape, dtype in specs:
        offset = align(offset)
        size = prod(shape) * _ITEMSIZE[dtype]
        fields.append(WorkspaceField(name, tuple(shape), dtype, offset, size))
        offset += size
    return fields, align(offset)


@dataclass(frozen=True)
class GemmForm:
    """One swap-AB GEMM launch of the chain as the hand-written launcher configures it."""

    n_tile: int
    kbps: int                  # k_blocks_per_stage the kernel is built with
    m_group: int               # weight M-tiles per work item
    k_tiles: int               # K / (32 * kbps)
    use_pdl: bool              # launch attribute (mxfp4.py:1148 pdl = enable_pdl or SWAP_PDL)
    weight_l2_hint: int | None  # weight-stream L2 policy the form is built with (``gemm2_swapab.form_config``;
                                # ``_swap_weight_l2_hint`` :2503-2510, EVICT_FIRST or None -> the ``_nol2`` form)
    # Hand-written knobs the Cake forms do not build (recorded deviations, not selections):
    hw_split_k: int            # GEMM2 device-side split-K (:169, :1983-1987, swapab_moe.py:880-887)
    hw_cluster_split_k: bool   # GEMM1 2-CTA cluster split-K (:173, :1450, swapab_moe.py:716-729)
    # PDL placement the form is built with (``gemm2_swapab.form_config``; ``pdl and dep_prefetch``, :1438 / :1982):
    late_dep_wait: bool        # GEMM2 waits in the row-loading warps only (:1400-1409)
    pdl_trigger_after_wait: bool  # GEMM1 triggers dependents right after its common wait (:1411-1413)
    two_cta: bool = False
    row_tma: bool = False
    row_group_list: bool = False   # GEMM1 SiTU over ``tile_idx_to_row_group`` (hybrid / mixed192 chains)
    sf_blocked: bool = False       # GEMM1 output scales in the tcgen05 atom layout (deployed with the list)


@dataclass(frozen=True)
class Launch:
    """One kernel launch of the hand-written ``run()`` on this row, in enqueue order, and its Cake form.

    ``step`` is the plan's launch name; ``hw`` the hand-written kernel and configuration (with the line
    reference); ``form`` the traced Cake form symbol that reproduces it exactly (``None`` when the merged
    tree has no such form, then ``missing`` names the IR form that is required); ``module`` the Cake module;
    ``backends`` the backends the form lowers through (``("cuda_cpp",)`` = a mixed-backend launch under the
    ``cutedsl`` plan); ``stream`` ``main`` or ``side`` (the hand-written fork / join
    streams); ``deviations`` the hand-written knobs of this launch the Cake form does not build (recorded, the
    same class as the plain chain's ``hw_*`` knobs: PDL placement, L2 hint, device-side split-K)."""

    step: str
    hw: str
    form: str | None
    module: str | None
    backends: tuple[str, ...] = ("cuda_cpp", "cutedsl")
    missing: str = ""
    stream: str = "main"
    deviations: tuple[str, ...] = ()

    @property
    def traced(self) -> bool:
        return self.form is not None


GEMM2_SWAPAB_MODULE = "cake.examples.cake.kimi_k3_mxfp4_situ_gemm2_swapab"


GEMM1_DENSE_MODULE = "cake.examples.cake.kimi_k3_mxfp4_situ_gemm1_dense"


GEMM2_DENSE_MODULE = "cake.examples.cake.kimi_k3_mxfp4_situ_gemm2_dense"


ROUTING_MODULE = "cake.examples.cake.kimi_k3_mxfp4_situ_routing"


DISPATCH_MODULE = "cake.examples.cake.kimi_k3_mxfp4_situ_dispatch"


MOE_SORT_MODULE = "cake.examples.cake.kimi_k3_mxfp4_situ_moe_sort"


FINALIZE_MODULE = "cake.examples.cake.kimi_k3_mxfp4_situ_finalize"


BOTH = ("cuda_cpp", "cutedsl")


CUDA_ONLY = ("cuda_cpp",)


SWAPAB_SITU_FORMS = ((8, 8), (16, 4), (32, 4), (64, 4), (128, 4))   # gemm2_swapab.SITU_FORMS (n_tile, kbps)


SWAPAB_NARROW_FORMS = (8, 16, 32)                                   # gemm2_swapab.NARROW_FORMS


SWAPAB_SITU_ROWGROUP_FORMS = ((64, 4), (192, 4))                     # gemm2_swapab.SITU_ROWGROUP_FORMS (r11b: hybrid n64 / mixed192 n192 2-CTA)


SWAPAB_WIDE_CTA1_FORMS = (64, 128)                                  # gemm2_swapab.WIDE_CTA1_FORMS (row TMA)


SWAPAB_TWO_CTA_FORMS = (192,)                                       # gemm2_swapab.TWO_CTA_FORMS


GEMM2_DENSE_TRACED_N = (128, 192, 256)                              # gemm2_dense.TRACED_FORMS (cta_group 1)


GEMM1_DENSE_FORMS = {
    "gemm1_dense_situ_m128_n256": (128, 256, False, False, False, False),
    "gemm1_dense_situ_m128_n256_rowgroup": (128, 256, False, False, True, False),
    "gemm1_dense_situ_m128_n128_rowgroup": (128, 128, False, False, True, False),
    "gemm1_dense_situ_m128_n256_zero_fill": (128, 256, True, False, False, False),
    "gemm1_dense_situ_m256_n256_2cta_zero_fill_secondary": (256, 256, True, True, False, False),
    "gemm1_dense_situ_m128_n256_rowgroup_early": (128, 256, False, False, True, True),
    "gemm1_dense_situ_m128_n128_rowgroup_early": (128, 128, False, False, True, True),
}


DENSE_EARLY_SUFFIX = "_early"                                       # gemm1_dense.EARLY_SUFFIX / gemm2_dense.EARLY_SUFFIX


GEMM2_DENSE_CLUSTER12_N = (256, 192)                                # gemm2_dense.CLUSTER12_FORMS ((128, N) / cluster (1, 2))


SWAPAB_PARTIAL_FORMS = ((8, 4, 2), (16, 4, 2), (32, 4, 2))         # gemm2_swapab.PARTIAL_FORMS ((n_tile, kbps, m_group))


SWAPAB_NODP_SUFFIX = "_nodp"                                        # gemm2_swapab.NODP_SUFFIX (constructor-default PDL placement)


SWAPAB_NOL2_SUFFIX = "_nol2"                                        # gemm2_swapab.NOL2_SUFFIX (weight TMA loads without EVICT_FIRST)


def swapab_pdl_suffix(n_tile: int, dep_prefetch: bool) -> str:
    """``gemm2_swapab.pdl_symbol_suffix`` from the plan's ``pdl and dep_prefetch``: the module's default placement
    (``dep_prefetch_by_default``: narrow forms prefetch, wide / 2-CTA forms do not) carries no suffix; a narrow form
    without the prefetch (the split chain) carries ``_nodp``; the prefetch on a wide form is no hand-written selection."""
    narrow = n_tile in SWAPAB_NARROW_FORMS
    if bool(dep_prefetch) == narrow:
        return ""
    if narrow:
        return SWAPAB_NODP_SUFFIX
    raise ValueError(f"dep_prefetch on the wide swap-AB form n{n_tile} is not a hand-written plan selection")


def swapab_l2_suffix(n_tile: int, weight_l2_hint) -> str:
    """``gemm2_swapab.l2_symbol_suffix`` from the plan's ``swap_weight_l2_hint`` (:2503-2510): the EVICT_FIRST policy
    (every T <= 16 row, every MoE-TP row, the expert-parallel rows from T = 256) carries no suffix; the policy-less
    selection (expert-parallel 17 <= T <= 255, the n16 / n32 forms) carries ``_nol2``; no other form reaches it."""
    if weight_l2_hint is not None:
        return ""
    if n_tile not in (16, 32):
        raise ValueError(f"weight_l2_hint=None on the swap-AB form n{n_tile} is not a hand-written plan selection "
                         f"(mxfp4.py:2503-2510 / swapab_tile_policy)")
    return SWAPAB_NOL2_SUFFIX


def swapab_situ_form(n_tile: int, kbps: int, *, row_group_list: bool = False, sf_blocked: bool = False,
                     two_cta: bool = False, dep_prefetch: bool | None = None,
                     weight_l2_hint: int | None = TMA_L2_EVICT_FIRST) -> Launch:
    """GEMM1 SiTU swap-AB launch (hand-written ``swapab_gemm1_situ``, swapab_moe.py:665-835).

    ``dep_prefetch`` (the plan's ``pdl and dep_prefetch``, mxfp4.py:1438) selects ``pdl_trigger_after_wait``; the
    form symbol carries ``swapab_pdl_suffix`` (``_nodp`` on a narrow form without the dependent-side prefetch:
    the split chain's GEMM1). ``weight_l2_hint`` (the plan's ``swap_weight_l2_hint``, :1439 / :2503-2510) is the
    weight-stream L2 policy of the form (``_nol2`` without it, ``swapab_l2_suffix``)."""
    if dep_prefetch is None:
        dep_prefetch = n_tile in SWAPAB_NARROW_FORMS          # gemm2_swapab.dep_prefetch_by_default
    hw = f"swapab_gemm1_situ n_tile={n_tile} k_blocks_per_stage={kbps}"
    flags = []
    if row_group_list:
        flags.append("tile_idx_to_row_group (128-row sort groups)")
    if sf_blocked:
        flags.append("sf_blocked")
    if two_cta:
        flags.append("two_cta (192-row cta_group::2 pair)")
    flags.append(f"pdl_trigger_after_wait={dep_prefetch}")
    flags.append(f"weight_l2_hint={'EVICT_FIRST' if weight_l2_hint is not None else None}")
    if flags:
        hw += " " + ", ".join(flags)
    nol2 = swapab_l2_suffix(n_tile, weight_l2_hint)
    if row_group_list or sf_blocked or two_cta:
        # Round 11 (r11b): the deployed pairings of the list forms -- the hybrid 64-row sub-tile form over 128-row
        # sort groups (``swapab_gemm1_situ(tile_idx_to_row_group=swap_row_groups, group_rows=128, sf_blocked=True)``,
        # mxfp4.py:1346-1351) and the mixed192 192-row cta_group::2 window form (:1350-1357 / :1403-1408, ``row_unit``
        # 64) -- both write the blocked output scales the dense finalize GEMM2 tiles of the chain read
        # (``gemm2_swapab.SITU_ROWGROUP_FORMS``, symbols ``gemm1_swapab_situ_n64_rowgroup`` /
        # ``gemm1_swapab_situ_n192_2cta_rowgroup``). Every other combination is refused by name.
        deployed = (row_group_list and sf_blocked and (n_tile, kbps) in SWAPAB_SITU_ROWGROUP_FORMS
                    and two_cta == swap_two_cta(n_tile))
        if not deployed:
            what = ("192-row cta_group::2 SiTU epilogue form" if two_cta else
                    "SiTU form with the row-group list / blocked row-scale output")
            return Launch("gemm1_swapab_situ", hw, None, GEMM2_SWAPAB_MODULE, BOTH,
                          missing=f"gemm2_swapab.form_config(situ=True): {what} (n_tile {n_tile}, kbps {kbps}, "
                                  f"row_group_list={row_group_list}, sf_blocked={sf_blocked}, two_cta={two_cta}) is not a "
                                  f"deployed pairing; traced: {SWAPAB_SITU_ROWGROUP_FORMS} with the list and blocked scales")
        return Launch("gemm1_swapab_situ", hw,
                      f"gemm1_swapab_situ_n{n_tile}" + ("_2cta" if two_cta else "") + "_rowgroup"
                      + swapab_pdl_suffix(n_tile, dep_prefetch) + nol2, GEMM2_SWAPAB_MODULE, BOTH)
    if (n_tile, kbps) not in SWAPAB_SITU_FORMS:
        return Launch("gemm1_swapab_situ", hw, None, GEMM2_SWAPAB_MODULE, BOTH,
                      missing=f"gemm2_swapab.form_config(situ=True): SiTU form (n_tile {n_tile}, kbps {kbps}) not traced "
                              f"(traced: {SWAPAB_SITU_FORMS})")
    return Launch("gemm1_swapab_situ", hw, f"gemm1_swapab_situ_n{n_tile}" + ("" if kbps == 4 else f"_k{kbps}")
                  + swapab_pdl_suffix(n_tile, dep_prefetch) + nol2, GEMM2_SWAPAB_MODULE, BOTH)


def swapab_gemm2_form(n_tile: int, kbps: int, m_group: int, *, finalize: bool, row_group_list: bool = False,
                      sf_blocked: bool = False, two_cta: bool = False, dep_prefetch: bool | None = None,
                      weight_l2_hint: int | None = TMA_L2_EVICT_FIRST) -> Launch:
    """GEMM2 swap-AB launch (hand-written ``swapab_gemm2``, swapab_moe.py:836-1000).

    ``dep_prefetch`` (the plan's ``pdl and dep_prefetch``, mxfp4.py:1982) selects ``late_dep_wait``; the form symbol
    carries ``swapab_pdl_suffix`` (``_nodp`` on the split chain's narrow partial GEMM2). ``weight_l2_hint`` (:1979 /
    :2503-2510) is the weight-stream L2 policy (``_nol2`` without it, ``swapab_l2_suffix``)."""
    if dep_prefetch is None:
        dep_prefetch = n_tile in SWAPAB_NARROW_FORMS          # gemm2_swapab.dep_prefetch_by_default
    kind = "finalize" if finalize else "partial"
    hw = f"swapab_gemm2 n_tile={n_tile} k_blocks_per_stage={kbps} m_group={m_group} epilogue={kind}"
    if row_group_list:
        hw += " tile_idx_to_row_group"
    if sf_blocked:
        hw += " sf_blocked"
    if two_cta:
        hw += " two_cta"
    hw += f" late_dep_wait={dep_prefetch} weight_l2_hint={'EVICT_FIRST' if weight_l2_hint is not None else None}"
    nodp = swapab_pdl_suffix(n_tile, dep_prefetch) + swapab_l2_suffix(n_tile, weight_l2_hint)
    step = f"gemm2_swapab_{kind}"
    if not finalize:
        # The deferred epilogue (hand-written ``epilogue_kind="partial"``: alpha * acc BF16 rows stored in
        # permuted order into ``partial_rows`` [>= R, H] BF16 (mxfp4.py:1957-1961, :2743), no route weight) is traced
        # for the narrow cta_group-1 forms of the two-stage chain, ``build_gemm2_swapab_finalize_ir(n, 4, m_group=2,
        # partial=True)`` -> ``gemm2_swapab_partial_n<N>_m2``.
        if row_group_list or sf_blocked or two_cta:
            return Launch(step, hw, None, GEMM2_SWAPAB_MODULE, BOTH,
                          missing="gemm2_swapab partial epilogue with tile_idx_to_row_group / sf_blocked / two_cta "
                                  "(form_config refuses partial with two_cta or the wide epilogue) not traced")
        if (n_tile, kbps, m_group) not in SWAPAB_PARTIAL_FORMS:
            return Launch(step, hw, None, GEMM2_SWAPAB_MODULE, BOTH,
                          missing=f"gemm2_swapab partial epilogue form n{n_tile} kbps {kbps} m_group {m_group} not "
                                  f"traced (traced: {SWAPAB_PARTIAL_FORMS} as (n_tile, kbps, m_group))")
        return Launch(step, hw, f"gemm2_swapab_partial_n{n_tile}_m{m_group}" + nodp, GEMM2_SWAPAB_MODULE, BOTH)
    if two_cta:
        if n_tile not in SWAPAB_TWO_CTA_FORMS or kbps != 4 or m_group != 1 or not (row_group_list and sf_blocked):
            return Launch(step, hw, None, GEMM2_SWAPAB_MODULE, BOTH,
                          missing=f"gemm2_swapab two_cta form only traced as n192 kbps 4 m_group 1 with the row-group "
                                  f"list and blocked scales (mixed192 deployment); got n{n_tile} k{kbps} m{m_group}")
        return Launch(step, hw, "gemm2_swapab_finalize_n192_2cta" + nodp, GEMM2_SWAPAB_MODULE, BOTH)
    if row_group_list or sf_blocked:
        return Launch(step, hw, None, GEMM2_SWAPAB_MODULE, BOTH,
                      missing="gemm2_swapab single-CTA finalize with tile_idx_to_row_group / sf_blocked "
                              "(form_config defaults them on for the 192-row form only) not traced")
    if n_tile in SWAPAB_NARROW_FORMS and kbps in (4, 8, 12) and m_group in (1, 2):
        default_m = 1
        sym = f"gemm2_swapab_finalize_n{n_tile}" + ("" if kbps == 4 else f"_k{kbps}") + \
              ("" if m_group == default_m else f"_m{m_group}")
        return Launch(step, hw, sym + nodp, GEMM2_SWAPAB_MODULE, BOTH)
    if n_tile in SWAPAB_WIDE_CTA1_FORMS and kbps == 4 and m_group == (2 if n_tile == 64 else 1):
        return Launch(step, hw, f"gemm2_swapab_finalize_n{n_tile}" + nodp, GEMM2_SWAPAB_MODULE, BOTH)
    return Launch(step, hw, None, GEMM2_SWAPAB_MODULE, BOTH,
                  missing=f"gemm2_swapab finalize form n{n_tile} kbps {kbps} m_group {m_group} not traced")


def gemm1_dense_form_symbol(tile_m: int, n_tile: int, *, zero_fill: bool, zero_fill_secondary: bool,
                            row_group_list: bool, pdl_trigger_early: bool = False) -> str:
    """``kimi_k3_mxfp4_situ_gemm1_dense.form_symbol``:
    ``gemm1_dense_situ_m<M>_n<N>[_2cta][_rowgroup][_zero_fill[_secondary]][_early]``."""
    sym = f"gemm1_dense_situ_m{tile_m}_n{n_tile}" + ("_2cta" if tile_m == 256 else "")
    if row_group_list:
        sym += "_rowgroup"
    if zero_fill:
        sym += "_zero_fill" + ("_secondary" if zero_fill_secondary else "")
    if pdl_trigger_early:
        sym += DENSE_EARLY_SUFFIX
    return sym


def gemm1_dense_form(mma_tiler: tuple[int, int], cluster: tuple[int, int], *, cluster_split_k: bool = False,
                     zero_fill: bool = False, zero_fill_secondary: bool = False, row_group_list: bool = False,
                     pdl_trigger_early: bool = False) -> Launch:
    """Dense gather GEMM1 SiTU launch (hand-written ``blockscaled_contiguous_gather_grouped_gemm_act_fusion``).

    Traced forms = ``GEMM1_DENSE_FORMS``: the (128, 256) / (1, 1) plain, row-group and zero-fill forms,
    the (128, 128) / (1, 1) row-group form (``SWAP_SPLIT_GEMM1_N_POLICY`` at T <= 512) and the 2-CTA (256, 256) /
    (2, 1) ``zero_fill_secondary`` alternate. Flag combinations without a traced form (row-group list together with
    zero fill, the 2-CTA tile with a row-group list or without the secondary fill, ...) are refused by name.
    ``cluster_split_k`` (form (c)) is recorded as a deviation on the traced form it decorates. ``pdl_trigger_early``
    (the split chain's wide launches, ``pdl and SWAP_SPLIT_EARLY_TRIGGER``) selects the ``_early`` placement form
    (``launch_dependents`` at kernel entry right before the common wait, no footer trigger; K:1775-1777 / :5482).
    """
    hw = f"dense gather GEMM1 mma_tiler {mma_tiler} cluster {cluster}"
    flags = [name for flag, name in ((zero_fill, "zero_fill_output"), (zero_fill_secondary, "zero_fill_secondary"),
                                     (row_group_list, "tile_idx_to_row_group"), (cluster_split_k, "cluster_split_k"),
                                     (pdl_trigger_early, "pdl_trigger_early")) if flag]
    if flags:
        hw += " " + ", ".join(flags)
    deviations = ()
    missing = []
    tile_m, n = mma_tiler
    expected_cluster = (2, 1) if tile_m == 256 else (1, 1)
    if tile_m not in (128, 256) or n not in (128, 256) or cluster != expected_cluster:
        missing.append(f"gemm1_dense form with mma_tiler {mma_tiler} cluster {cluster} (traced tiles: (128, 256) / (1, 1), "
                       "(128, 128) / (1, 1), (256, 256) / (2, 1))")
    else:
        sym = gemm1_dense_form_symbol(tile_m, n, zero_fill=zero_fill, zero_fill_secondary=zero_fill_secondary,
                                      row_group_list=row_group_list, pdl_trigger_early=pdl_trigger_early)
        if sym not in GEMM1_DENSE_FORMS:
            letters = []
            if zero_fill and tile_m == 128:
                letters.append("form (b) zero_fill_output")
            if tile_m == 256:
                letters.append("form (d) M256 2-CTA")
            if row_group_list:
                letters.append("row-group list")
            if pdl_trigger_early:
                letters.append("pdl_trigger_early")
            missing.append(f"gemm1_dense form {sym} ({', '.join(letters)} together) not traced; traced forms: "
                           f"{tuple(GEMM1_DENSE_FORMS)}")
    if cluster_split_k:
        # Same class as the swap-AB GEMM1 cluster split-K recorded by the plain chain:
        # a device-side split decided per launch inside the (1, 1, 2) cluster launch; recorded, not a refusal.
        deviations = ("cluster_split_k (form (c), (1, 1, 2) cluster; device-side split when 2 * valid_tiles <= grid.z)",)
    if missing:
        return Launch("gemm1_dense", hw, None, GEMM1_DENSE_MODULE, BOTH, missing="; ".join(missing), deviations=deviations)
    return Launch("gemm1_dense", hw, sym, GEMM1_DENSE_MODULE, BOTH, deviations=deviations)


def gemm2_dense_form(mma_tiler: tuple[int, int], cluster: tuple[int, int], *, raster_along_m=False, swizzle: int = 1,
                     c_stages: int = 1, row_group_list: bool = False, pdl_trigger_early: bool = False,
                     hidden_size: int | None = None) -> Launch:
    """Dense grouped GEMM2 finalize launch (hand-written ``blockscaled_contiguous_grouped_gemm_finalize_fusion``).

    ``hidden_size`` (the GEMM2 N extent) is required for the cluster (1, 2) tactics: the form
    (``build_gemm2_dense_finalize_ir(n_tile, cluster_n=2)`` -> ``gemm2_dense_finalize_n<N>_c12``) refuses an odd
    weight-tile count ``ceil(H / N)`` (the out-of-range rank-1 CTA arm of the hand-written scheduler is untraced), and
    lays the pair along x: ``cluster_dims (2, 1, 1)``, grid ``(2 * min(cap * ceil(n_tiles / 2), SMs // 2), 1, 1)``
    (recorded geometry divergence vs the hand-written ``(1, 2, Z)``).
    """
    hw = (f"dense finalize GEMM2 mma_tiler {mma_tiler} cluster {cluster} raster_along_m={raster_along_m} "
          f"swizzle={swizzle} c_stages={c_stages}")
    if row_group_list:
        hw += " tile_idx_to_row_group"
    if pdl_trigger_early:
        hw += " pdl_trigger_early"
    missing = []
    n = mma_tiler[1]
    cta_group = 2 if mma_tiler[0] == 256 else 1
    deviations = ()
    if cluster == (1, 2):
        if cta_group != 1 or n not in GEMM2_DENSE_CLUSTER12_N:
            missing.append(f"gemm2_dense {mma_tiler} cluster (1, 2) (traced for (128, N), N in "
                           f"{GEMM2_DENSE_CLUSTER12_N} only)")
        elif row_group_list:
            missing.append(f"gemm2_dense (128, {n}) cluster (1, 2) with the tile_idx_to_row_group list (only the "
                           "plain _c12 forms are traced / registered)")
        elif hidden_size is None:
            raise ValueError("gemm2_dense_form(cluster=(1, 2)) needs hidden_size to resolve the weight-tile count")
        elif -(-hidden_size // n) % 2:
            missing.append(f"gemm2_dense (128, {n}) cluster (1, 2): odd weight-tile count {-(-hidden_size // n)} "
                           f"(H={hidden_size}); the form refuses it (the hand-written scheduler's out-of-range rank-1 "
                           "CTA arm is untraced)")
        else:
            deviations += (f"cluster (1, 2) laid along x: Cake cluster_dims (2, 1, 1), grid (2 * min(cap * "
                           f"ceil({-(-hidden_size // n)} / 2), SMs // 2), 1, 1) vs hand-written (1, 2, Z) (recorded)",)
    elif cta_group == 2 and (mma_tiler, cluster) != ((256, 256), (2, 1)):
        missing.append(f"gemm2_dense 2-CTA form {mma_tiler} / {cluster} (only (256, 256) / (2, 1) traced)")
    elif cta_group == 1 and (cluster != (1, 1) or n not in GEMM2_DENSE_TRACED_N):
        missing.append(f"gemm2_dense form {mma_tiler} / {cluster} (traced: n in {GEMM2_DENSE_TRACED_N}, cluster (1, 1))")
    if raster_along_m == "auto" or raster_along_m is True or swizzle != 1:
        missing.append(f"gemm2_dense raster_along_m={raster_along_m!r} swizzle={swizzle} (device-side scheduler mode "
                       "vote / M-fastest raster not traced)")
    if c_stages != 1:
        missing.append(f"gemm2_dense c_stages {c_stages} (form_config allows 1)")
    if missing:
        return Launch("gemm2_dense_finalize", hw, None, GEMM2_DENSE_MODULE, BOTH, missing="; ".join(missing),
                      deviations=deviations)
    sym = (f"gemm2_dense_finalize_n{n}" + ("_2cta" if cta_group == 2 else "") + ("_c12" if cluster == (1, 2) else "")
           + ("_rg" if row_group_list else "") + (DENSE_EARLY_SUFFIX if pdl_trigger_early else ""))
    if pdl_trigger_early and not row_group_list:
        missing.append(f"gemm2_dense {sym}: pdl_trigger_early is traced on the split chain's _rg form only")
        return Launch("gemm2_dense_finalize", hw, None, GEMM2_DENSE_MODULE, BOTH, missing="; ".join(missing),
                      deviations=deviations)
    return Launch("gemm2_dense_finalize", hw, sym, GEMM2_DENSE_MODULE, BOTH, deviations=deviations)


def moe_sort_bounded_state(num_tokens: int, top_k: int, tier: int, sm_count: int) -> bool:
    """``launchCoopKernelTier<Tier>`` (trtllm_fused_moe_routing_custom.cuh:1426-1451): the NumTop16Experts kernel
    (four expanded indices per thread) when the tier is in the high-expert range, ``9 <= top_k <= 16`` and
    ``T * top_k <= 4 * numBlocksCoop * Tier``; the generic 64-entry NumTop8Experts kernel otherwise."""
    lo, hi = MOE_SORT_HIGH_EXPERT_TIERS
    if not lo <= tier <= hi:
        return False
    klo, khi = MOE_SORT_HIGH_EXPERT_TOPK
    capacity = MOE_SORT_BOUNDED_PER_THREAD * (sm_count - MOE_SORT_RESERVED_SMS) * tier
    return klo <= top_k <= khi and num_tokens * top_k <= capacity


def moe_sort_max_tokens_coop(tier: int, top_k: int, sm_count: int) -> int:
    """``maxTokensCoop = numBlocksCoop * numThreadsHist * 64 / topK`` (routing_common.cu:136-141)."""
    return (sm_count - MOE_SORT_RESERVED_SMS) * min(tier, 1024) * MOE_SORT_GENERIC_PER_THREAD // top_k


def moe_sort_form_names(tier: int, *, bounded: bool, dual: bool, mixed: bool) -> tuple[str, str]:
    """The K6 module's IR export names (``kimi_k3_mxfp4_situ_moe_sort.init_form_name`` / ``coop_form_name``)."""
    coop = f"moe_sort_coop_t{tier}" + ("_bounded" if bounded else "") + ("_dual" if dual else "") + ("_mixed" if mixed else "")
    return f"moe_sort_init_t{tier}", coop


def moe_sort_forms(num_experts: int, *, num_tokens: int, top_k: int, sm_count: int, tile: int,
                   alt_tile: int | None, mixed: bool, pdl: bool) -> tuple[Launch, Launch]:
    """K6 ``moe_sort`` (init + cooperative kernel) of the generic routing path (mxfp4.py:1251-1313) exactly as
    ``runPostTopKPipeline`` (routing_common.cu:64-150) launches it on ``sm_count`` SMs: expert tier by
    ``getMaxNumExperts``, ``min(tier, 1024)`` threads, ``SMs - 8`` CTAs, and the per-thread state of
    ``launchCoopKernelTier``. ``alt_tile`` is the dual-tile alternate (``tile_tokens_dim_alt``) or None."""
    tier = moe_sort_expert_tier(num_experts)
    threads = min(tier, 1024)
    blocks = sm_count - MOE_SORT_RESERVED_SMS
    dual = alt_tile is not None
    bounded = moe_sort_bounded_state(num_tokens, top_k, tier, sm_count)
    state = (f"NumTop16Experts bounded {MOE_SORT_BOUNDED_PER_THREAD}/thread" if bounded
             else f"NumTop8Experts {MOE_SORT_GENERIC_PER_THREAD}/thread")
    hw_init = f"routingInitExpertCounts tier {tier} threads {threads} (trtllm_fused_moe_routing_custom.cuh:1499)"
    hw_coop = (f"routingIndicesCoopKernel<{tier}, {state}> grid {blocks} x {threads} tile {tile}"
               + (f" dual {alt_tile}" if dual else "") + (" mixed192 lists" if mixed else "") + f" pdl={pdl}"
               f" (launchCoopKernelTier :1426-1451, {sm_count} SMs)")
    missing = None
    if tier not in MOE_SORT_CAKE_TIERS:
        missing = f"moe_sort expert tier {tier} (E={num_experts}) is not traced by kimi_k3_mxfp4_situ_moe_sort"
    elif num_tokens <= MOE_SORT_CLUSTER_MAX_TOKENS:
        missing = (f"moe_sort T={num_tokens} <= {MOE_SORT_CLUSTER_MAX_TOKENS}: the hand-written pipeline launches the "
                   "single-cluster kernel routingIndicesClusterKernel (routing_common.cu:98-101), not the cooperative "
                   "pair; not reproduced")
    elif num_tokens > moe_sort_max_tokens_coop(tier, top_k, sm_count):
        missing = (f"moe_sort T={num_tokens} above maxTokensCoop={moe_sort_max_tokens_coop(tier, top_k, sm_count)} "
                   f"({blocks} x {threads} x 64 / {top_k}): the multi-kernel histogram / offsets path "
                   "(routing_common.cu:150-) is not reproduced")
    if missing is not None:
        return (Launch("moe_sort_init", hw_init, None, MOE_SORT_MODULE, CUDA_ONLY, missing=missing),
                Launch("moe_sort_coop", hw_coop, None, MOE_SORT_MODULE, CUDA_ONLY, missing=missing))
    init_form, coop_form = moe_sort_form_names(tier, bounded=bounded, dual=dual, mixed=mixed)
    return (Launch("moe_sort_init", hw_init, init_form, MOE_SORT_MODULE, CUDA_ONLY),
            Launch("moe_sort_coop", hw_coop, coop_form, MOE_SORT_MODULE, CUDA_ONLY))


@dataclass(frozen=True)
class DenseSelection:
    """The dense path's ``CuteDslMxfp4MoEWrapper`` selections (``_tactic`` / ``_dual_tactic`` / ``_gemm2_raster`` /
    ``_dense_gemm1_cluster_split`` / zero fill, mxfp4.py:2353-2440, :3007-3088)."""

    tile: int
    gemm1: tuple                # ((M, N), (cluster_m, cluster_n), False)
    gemm2: tuple
    dual: tuple | None          # (tile, gemm1, gemm2) of the alternate (M256 2-CTA) launches or None
    gemm1_cluster_split_k: bool
    gemm2_raster: tuple         # (raster_along_m, swizzle)
    fill_in_gemm1: bool         # zero_fill_counters bound (T >= 8192): GEMM1 zero-fills the output
    async_memset: bool          # _dense_async_memset (aux-stream memset resources; never enqueued for this family)
    two_stage: bool


@dataclass(frozen=True)
class PlanDecision:
    """Everything the hand-written plan decides from ``(layout, T)``: the path, the sub-form flags, the launch
    list in enqueue order with the Cake form of every launch, and ``supported`` = every launch has a traced
    Cake form (``missing_forms`` names the hand-written kernel configurations the merged tree lacks; no launch
    is ever substituted by a different form). ``mixed_backend``: the plan under the ``cutedsl`` backend would
    launch a ``cuda_cpp``-only module (K6)."""

    num_tokens: int
    use_swapab: bool
    tile: int
    group_rows: int
    hybrid: bool
    mixed: bool
    split: bool
    wide192: bool
    two_stage: bool
    fused_route_cap: int
    fused_routing: bool
    single_tile_per_expert: bool
    clear_output: bool
    pdl: bool
    dep_prefetch: bool
    token_index: bool
    tiles: int
    rows: int
    launches: tuple[str, ...]
    gemm1: GemmForm | None
    gemm2: GemmForm | None
    supported: bool
    reason: str = ""
    path: str = "plain"                       # plain | split_two_stage | hybrid | mixed192 | dense
    launch_plan: tuple[Launch, ...] = ()
    missing_forms: tuple[str, ...] = ()
    mixed_backend: bool = False
    mixed192: bool = False
    split_dense: bool = False
    dense: DenseSelection | None = None


class CakeSwapAbPolicy:
    """The hand-written wrapper's decision functions (mxfp4.py:2197-2615) for one rank layout."""

    def __init__(self, layout: RankLayout, *, top_k: int, hidden_size: int, enable_pdl: bool = False,
                 swapab_n_tile: int = SWAP_ROW_TILE, swapab_max_tokens: int | None = None,
                 swapab_tile_policy: tuple[tuple[int, int], ...] | None = None, sm_count: int = CONTRACT_SM_COUNT):
        if not 1 <= top_k <= min(32, layout.num_experts):
            raise ValueError("require 1 <= top_k <= min(32, num_experts)")
        if sm_count <= MOE_SORT_RESERVED_SMS:
            raise ValueError("sm_count must exceed the 8 SMs the cooperative routing kernel reserves")
        self.sm_count = int(sm_count)            # the device's SM count (K6 grid + per-thread state selection)
        if hidden_size <= 0 or hidden_size % 128:
            raise ValueError("hidden size must be a positive multiple of 128")
        if swapab_n_tile not in (8, 16, 32, 64, 128):
            raise ValueError("swapab_n_tile must be 8, 16, 32, 64 or 128")   # :2231-2232
        self.layout = layout
        self.top_k = int(top_k)
        self.hidden_size = int(hidden_size)
        self.enable_pdl = bool(enable_pdl)
        self.swapab_n_tile = int(swapab_n_tile)
        shard = layout.intermediate_shard
        if swapab_max_tokens is None:
            # :2299-2310 (no SWAPAB_MAX_TOKENS in the environment)
            swapab_max_tokens = 2048 if SWAP_HYBRID and shard < 1024 else 1024
        if swapab_max_tokens < 0:
            raise ValueError("swapab_max_tokens must be >= 0")
        self.swapab_max_tokens = int(swapab_max_tokens)
        if swapab_tile_policy is None:
            # :2311-2327
            if shard >= 1024:
                swapab_tile_policy = ((128, 16), (UNBOUNDED, 32))
            elif SWAP_HYBRID:
                swapab_tile_policy = ((128, 8), (256, 16), (1024, 32), (UNBOUNDED, SWAP_HYBRID_MIN_TILE))
            else:
                swapab_tile_policy = ((128, 8), (256, 16), (UNBOUNDED, 32))
        policy = tuple((int(t), int(n)) for t, n in swapab_tile_policy)
        if any(n not in (8, 16, 32, 64, 128) for _, n in policy) or any(
                policy[i][0] >= policy[i + 1][0] for i in range(len(policy) - 1)):
            raise ValueError("swapab_tile_policy must be ascending (max_tokens, tile in {8,16,32,64,128})")
        self.swapab_tile_policy = policy

    # -- properties of the layout used by the decisions -------------------------------------------------
    @property
    def num_local_experts(self) -> int:
        return self.layout.num_local_experts

    @property
    def intermediate_shard(self) -> int:
        return self.layout.intermediate_shard

    @property
    def parallel_mode(self) -> str:
        return self.layout.mode

    # -- decision functions (same names as the hand-written wrapper, ``_`` dropped) ---------------------
    def use_swapab(self, num_tokens: int, do_finalize: bool = True) -> bool:
        """:2442-2452."""
        return ((1 <= num_tokens <= self.swapab_max_tokens or not do_finalize
                 or self.swap_wide192(num_tokens, do_finalize))
                and (SWAP_PDL or not self.enable_pdl))

    def swap_wide192(self, num_tokens: int, do_finalize: bool = True) -> bool:
        """:2454-2465 (``MXFP4_SWAP192=auto``, layouts ``moe_tensor_parallel``, min 8192 tokens)."""
        return (self.layout.mode in SWAP_WIDE192_LAYOUTS and bool(do_finalize)
                and num_tokens > self.swapab_max_tokens and num_tokens >= SWAP_WIDE192_MIN_TOKENS)

    def swap_tile(self, num_tokens: int, do_finalize: bool = True) -> int:
        """:2512-2520."""
        if self.swap_wide192(num_tokens, do_finalize):
            return 192
        if num_tokens <= 16:
            return self.swapab_n_tile
        for max_tokens, tile in self.swapab_tile_policy:
            if num_tokens <= max_tokens:
                return tile
        return self.swapab_tile_policy[-1][1]

    def swap_hybrid(self, num_tokens: int, do_finalize: bool = True) -> bool:
        """:2522-2531."""
        return (SWAP_HYBRID and bool(do_finalize) and not self.swap_wide192(num_tokens, do_finalize)
                and num_tokens * self.top_k > SWAP_HYBRID_MIN_ROUTES
                and self.swap_tile(num_tokens, do_finalize) >= SWAP_HYBRID_MIN_TILE)

    def swap_mixed(self, num_tokens: int, do_finalize: bool = True) -> bool:
        """:2533-2550 (``SWAPAB_MIXED=0``, ``SWAPAB_MIXED_AUTO_MAX_TOKENS=0``)."""
        return ((SWAP_MIXED or (num_tokens <= SWAP_MIXED_AUTO_MAX_TOKENS
                                and num_tokens * self.top_k <= self.fused_route_cap()))
                and bool(do_finalize) and num_tokens >= SWAP_MIXED_MIN_TOKENS
                and (self.intermediate_shard < 1024 or SWAP_MIXED_EP)
                and not self.swap_hybrid(num_tokens, do_finalize)
                and not self.swap_wide192(num_tokens, do_finalize)
                and SWAP_HYBRID_GROUP_ROWS % self.swap_tile(num_tokens, do_finalize) == 0)

    def fused_route_cap(self, num_tokens: int | None = None, do_finalize: bool = True) -> int:
        """:2552-2566."""
        if self.num_local_experts <= FUSED_ROUTE_LARGE_MAX_LOCAL_EXPERTS:
            return FUSED_ROUTE_MAX_ROUTES_LARGE
        if num_tokens is not None and self.swap_split(num_tokens, do_finalize):
            return FUSED_ROUTE_MAX_ROUTES_LARGE
        return FUSED_ROUTE_MAX_ROUTES

    def swap_split(self, num_tokens: int, do_finalize: bool = True) -> bool:
        """:2568-2581."""
        return (SWAP_SPLIT and bool(do_finalize)
                and SWAP_SPLIT_MIN_TOKENS <= num_tokens <= SWAP_SPLIT_MAX_TOKENS
                and num_tokens * self.top_k <= FUSED_ROUTE_MAX_ROUTES_LARGE
                and (self.intermediate_shard < 1024 or SWAP_SPLIT_EP)
                and self.swap_tile(num_tokens, do_finalize) < SWAP_SPLIT_WIDE_TILE
                and not self.swap_hybrid(num_tokens, do_finalize)
                and not self.swap_mixed(num_tokens, do_finalize))

    def swap_split_capacity(self, num_tokens: int) -> tuple[int, int]:
        """:2583-2605."""
        tile = self.swap_tile(num_tokens)
        wide = SWAP_SPLIT_WIDE_TILE
        narrow_rows = get_max_num_tiles(num_tokens, self.top_k, self.num_local_experts, tile) * tile
        wide_experts = min(self.num_local_experts, num_tokens * self.top_k // (SWAP_SPLIT_MIN_ROWS + 1))
        wide_groups = max(1, get_max_num_tiles(num_tokens, self.top_k, max(1, wide_experts), wide)
                          if wide_experts else 0)
        rows = -(-narrow_rows // wide) * wide + wide_groups * wide
        return wide_groups, rows

    def swap_group_rows(self, num_tokens: int, do_finalize: bool = True) -> int:
        """:2607-2614 (mixed-192 is a wide-192 sub-form, off below ``SWAP_WIDE192_MIN_TOKENS``)."""
        if (self.swap_hybrid(num_tokens, do_finalize) or self.swap_mixed(num_tokens, do_finalize)
                or self.swap_wide192(num_tokens, do_finalize)):
            return SWAP_HYBRID_GROUP_ROWS
        return self.swap_tile(num_tokens, do_finalize)

    def two_stage(self, num_tokens: int, do_finalize: bool = True) -> bool:
        """``Mxfp4MoESwapAbPlan.__init__`` :1099-1105."""
        return (bool(do_finalize) and not self.swap_hybrid(num_tokens, do_finalize)
                and not self.swap_wide192(num_tokens, do_finalize)
                and num_tokens > SWAP_ATOMIC_FINALIZE_MAX_TOKENS
                and self.intermediate_shard <= SWAP_TWO_STAGE_MAX_SHARD)

    def swap_weight_l2_hint(self, num_tokens: int) -> int | None:
        """:2503-2510."""
        if num_tokens <= SWAP_WEIGHT_L2_HINT_MAX_TOKENS:
            return TMA_L2_EVICT_FIRST
        lo, hi = SWAP_EP_L2HINT_SKIP
        if self.parallel_mode == "expert_parallel" and lo <= num_tokens <= hi:
            return None
        return TMA_L2_EVICT_FIRST

    # -- mixed 192-row form (:2467-2497) ---------------------------------------------------------------------
    def swap_mixed192(self, num_tokens: int, do_finalize: bool = True) -> bool:
        """:2467-2471 (``MXFP4_SWAP192_MIXED=1``): the wide form over 128-row sort groups with dense tiles."""
        return SWAP_WIDE192_MIXED and self.swap_wide192(num_tokens, do_finalize)

    def swap_mixed192_dual(self, num_tokens: int, do_finalize: bool = True):
        """:2473-2484: ``(tile, gemm1, gemm2)`` of the dual-tile routing's coarser tile, or None."""
        if not self.swap_mixed192(num_tokens, do_finalize):
            return None
        dual = self.dual_tactic(num_tokens)
        if dual is None or dual[0] % SWAP_HYBRID_GROUP_ROWS:
            return None
        return dual

    def swap_mixed192_gemm2(self, num_tokens: int) -> str:
        """:2486-2496 (``MXFP4_SWAP192_MIXED_GEMM2=auto``): ``dense`` on the MoE-TP shard from 16384 tokens."""
        if SWAP_WIDE192_MIXED_GEMM2 != "auto":
            return SWAP_WIDE192_MIXED_GEMM2
        if self.layout.mode == "moe_tensor_parallel" and num_tokens >= SWAP_WIDE192_DENSE_GEMM2_MIN_TOKENS:
            return "dense"
        return "split"

    # -- dense path (:2353-2440, :451-497) -------------------------------------------------------------------
    def tactic(self, num_tokens: int) -> tuple:
        """:2353-2371 (no offline table; SiTU tables by shard width)."""
        table = B300_SITU_DENSE_TACTIC_TABLE_NARROW if self.intermediate_shard < 1024 else B300_SITU_DENSE_TACTIC_TABLE_WIDE
        for limit, tactic in table:
            if num_tokens <= limit:
                if tactic == B300_SITU_DENSE_DUAL_TACTIC and self.dual_enabled(num_tokens):
                    return _T128_N256_C2                                                     # :2364-2368
                return tactic
        raise AssertionError("unreachable: the tactic tables end in an unbounded bucket")

    def dual_enabled(self, num_tokens: int) -> bool:
        """:2396-2405 (``dense_dual_tile=None``)."""
        return (DENSE_DUAL_TILE and num_tokens > DENSE_DUAL_TILE_MIN_TOKENS
                and self.intermediate_shard <= DENSE_DUAL_TILE_MAX_SHARD)

    def dual_tactic(self, num_tokens: int):
        """:2432-2440."""
        if not self.dual_enabled(num_tokens):
            return None
        if self.tactic(num_tokens)[0] != 128:
            return None
        return B300_SITU_DENSE_DUAL_TACTIC

    def dense_fill_in_gemm1(self, num_tokens: int) -> bool:
        """:481-490 (``MXFP4_DENSE_FILL_IN_GEMM1=1``, min 8192 tokens)."""
        if num_tokens < DENSE_FILL_IN_GEMM1_MIN_TOKENS:
            return False
        if DENSE_FILL_IN_GEMM1 == "1":
            return True
        if DENSE_FILL_IN_GEMM1 == "ep":
            return self.num_local_experts < self.layout.num_experts
        return False

    def dense_async_memset(self) -> bool:
        """:451-456 (``MXFP4_DENSE_ASYNC_MEMSET=ep``)."""
        if DENSE_ASYNC_MEMSET == "1":
            return True
        if DENSE_ASYNC_MEMSET == "ep":
            return self.num_local_experts < self.layout.num_experts
        return False

    def dense_gemm1_cluster_split(self, gemm1_tactic: tuple, num_tokens: int) -> bool:
        """:2373-2394 (``dense_gemm1_cluster_split=None``)."""
        return bool(DENSE_GEMM1_CLUSTER_SPLIT and self.num_local_experts < self.layout.num_experts
                    and gemm1_tactic[0][0] == 128 and tuple(gemm1_tactic[1]) == (1, 1)
                    and not self.dense_fill_in_gemm1(num_tokens))

    def gemm2_raster(self, num_tokens: int, gemm2_tile_n: int) -> tuple:
        """:2407-2430 (``MXFP4_GEMM2_RASTER_M=auto``, swizzle 4)."""
        if DENSE_GEMM2_RASTER_M == "auto":
            if not (self.intermediate_shard <= DENSE_GEMM2_RASTER_M_MAX_SHARD
                    and num_tokens >= DENSE_GEMM2_RASTER_M_MIN_TOKENS):
                return False, 1
            mode = "auto"
        elif DENSE_GEMM2_RASTER_M == "1":
            mode = True
        else:
            return False, 1
        n_tiles = -(-self.hidden_size // gemm2_tile_n)
        swizzle = DENSE_GEMM2_SWIZZLE if DENSE_GEMM2_SWIZZLE > 0 and n_tiles % DENSE_GEMM2_SWIZZLE == 0 else 1
        return mode, swizzle

    def dense_two_stage(self, num_tokens: int) -> bool:
        """:2498-2501 (``MXFP4_DENSE_TWO_STAGE=0``)."""
        return DENSE_TWO_STAGE_FINALIZE and num_tokens > SWAP_ATOMIC_FINALIZE_MAX_TOKENS

    def dense_selection(self, num_tokens: int) -> DenseSelection:
        """The dense path's launch configuration (``plan`` :3007-3088)."""
        tile, gemm1, gemm2 = self.tactic(num_tokens)
        return DenseSelection(
            tile=tile, gemm1=gemm1, gemm2=gemm2, dual=self.dual_tactic(num_tokens),
            gemm1_cluster_split_k=self.dense_gemm1_cluster_split(gemm1, num_tokens),
            gemm2_raster=self.gemm2_raster(num_tokens, gemm2[0][1]),
            fill_in_gemm1=(not self.dense_two_stage(num_tokens)) and self.dense_fill_in_gemm1(num_tokens),
            async_memset=self.dense_async_memset(), two_stage=self.dense_two_stage(num_tokens))

    # -- workspace (:2616-2822) --------------------------------------------------------------------------------
    def workspace_fields(self, num_tokens: int, do_finalize: bool = True) -> tuple[list[WorkspaceField], int]:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if not self.use_swapab(num_tokens, do_finalize):
            return self._dense_workspace_fields(num_tokens)
        tile = self.swap_tile(num_tokens, do_finalize)
        hybrid = self.swap_hybrid(num_tokens, do_finalize)
        mixed = self.swap_mixed(num_tokens, do_finalize)
        mixed192 = self.swap_mixed192(num_tokens, do_finalize)
        split = self.swap_split(num_tokens, do_finalize)
        group = self.swap_group_rows(num_tokens, do_finalize)
        tiles = get_max_num_tiles(num_tokens, self.top_k, self.num_local_experts, group)
        rows = tiles * group
        mixed192_dual = self.swap_mixed192_dual(num_tokens, do_finalize)
        alt_tiles = 0
        if mixed192_dual is not None:                                                        # :2630-2639
            alt_tiles = get_max_num_tiles(num_tokens, self.top_k, self.num_local_experts, mixed192_dual[0])
            rows = alt_tiles * mixed192_dual[0]
            tiles = rows // group
        if split:
            rows = self.swap_split_capacity(num_tokens)[1]                                  # :2640-2641
        wide_slots = rows // SWAP_SPLIT_WIDE_TILE
        L, shard, K = self.num_local_experts, self.intermediate_shard, self.top_k
        specs: list[tuple[str, tuple[int, ...], str]] = [
            ("out_tile_idx_to_expert_idx", (tiles,), "int32"),                              # :2644
            ("out_tile_idx_to_mn_limit", (tiles,), "int32"),                                # :2645
        ]
        if split:                                                                            # :2646-2657
            specs += [("swap_wide_expert", (wide_slots,), "int32"), ("swap_wide_limit", (wide_slots,), "int32"),
                      ("swap_wide_list", (wide_slots,), "int32"), ("swap_wide_count", (1,), "int32")]
        if hybrid or mixed or mixed192:                                                      # :2658-2673
            specs += [("swap_row_groups", (tiles * max(group // tile, 1),), "int32"),
                      ("swap_row_group_count", (1,), "int32"), ("swap_wide_list", (tiles,), "int32"),
                      ("swap_wide_count", (1,), "int32")]
        if mixed:                                                                            # :2674-2681
            specs += [("swap_all_groups", (tiles * (group // tile),), "int32"), ("swap_all_count", (1,), "int32")]
        if mixed192_dual is not None:                                                        # :2682-2713
            specs += [("out_alt_tile_idx_to_expert_idx", (alt_tiles,), "int32"),
                      ("out_alt_tile_idx_to_mn_limit", (alt_tiles,), "int32"),
                      ("out_alt_num_non_exiting_tiles", (1,), "int32"),
                      ("out_base_active_num_non_exiting_tiles", (1,), "int32"),
                      ("swap_alt_wide_list", (alt_tiles,), "int32"), ("swap_alt_wide_count", (1,), "int32"),
                      ("swap_row_group_count_base", (1,), "int32")]
        specs += [
            ("out_expert_counts", (SORT_SCRATCH_WORDS,), "int32"),                          # :2714
            ("out_expanded_idx_to_permuted_idx", (num_tokens, K), "int32"),                 # :2715-2720
            ("out_permuted_idx_to_expanded_idx", (rows,), "int32"),                         # :2721
        ]
        if swap_row_tma(tile, True):                                                         # :2722-2727
            specs.append(("permuted_idx_to_token_idx", (rows,), "int32"))
        specs += [
            ("out_total_num_padded_tokens", (1,), "int32"),                                 # :2728
            ("out_num_non_exiting_tiles", (1,), "int32"),                                   # :2729
            ("gemm1_out", (rows, shard), "float8_e4m3fn"),                                  # :2730
            ("gemm1_out_scale", (rows, shard // 32), "uint8"),                              # :2731-2736
            ("route_ids", (num_tokens, K), "int32"),                                        # :2737
            ("route_weights", (num_tokens, K), "float32"),                                  # :2738
            ("w1_alpha", (L,), "float32"),                                                  # :2739
            ("w2_alpha", (L,), "float32"),                                                  # :2740
        ]
        if (do_finalize and not hybrid and not self.swap_wide192(num_tokens, do_finalize)
                and num_tokens > SWAP_ATOMIC_FINALIZE_MAX_TOKENS
                and shard <= SWAP_TWO_STAGE_MAX_SHARD):                                      # :2742-2754
            specs.append(("partial_rows", (rows, self.hidden_size), "bfloat16"))
        return _layout(specs)

    def _dense_workspace_fields(self, num_tokens: int) -> tuple[list[WorkspaceField], int]:
        """:2763-2822 (dense branch)."""
        tile = self.tactic(num_tokens)[0]
        dual = self.dual_tactic(num_tokens)
        cap_tile = dual[0] if dual is not None else tile
        tiles = get_max_num_tiles(num_tokens, self.top_k, self.num_local_experts, cap_tile)
        rows = tiles * cap_tile
        base_tiles = rows // tile
        L, shard, K = self.num_local_experts, self.intermediate_shard, self.top_k
        specs: list[tuple[str, tuple[int, ...], str]] = []
        if self.dense_two_stage(num_tokens):                                                 # :2775-2785
            specs.append(("partial_rows", (num_tokens * K, self.hidden_size), "bfloat16"))
        specs += [("out_tile_idx_to_expert_idx", (base_tiles,), "int32"),                   # :2786
                  ("out_tile_idx_to_mn_limit", (base_tiles,), "int32")]                      # :2787
        if dual is not None:                                                                 # :2788-2797
            specs += [("out_alt_tile_idx_to_expert_idx", (tiles,), "int32"),
                      ("out_alt_tile_idx_to_mn_limit", (tiles,), "int32"),
                      ("out_alt_num_non_exiting_tiles", (1,), "int32"),
                      ("out_base_active_num_non_exiting_tiles", (1,), "int32")]
        specs += [("out_expanded_idx_to_permuted_idx", (num_tokens, K), "int32"),           # :2798-2803
                  ("out_permuted_idx_to_expanded_idx", (rows,), "int32")]                    # :2804
        if DENSE_GEMM1_A_TMA:                                                                # :2805-2809
            specs.append(("permuted_idx_to_token_idx", (rows,), "int32"))
        specs += [("out_total_num_padded_tokens", (1,), "int32"),                           # :2810
                  ("out_num_non_exiting_tiles", (1,), "int32"),                              # :2811
                  ("gemm1_out", (rows, shard), "float8_e4m3fn"),                             # :2812
                  ("gemm1_out_scale", (32, 4, rows // 128, 4, shard // 128, 1), "uint8"),    # :2813-2818 (blocked)
                  ("route_ids", (num_tokens, K), "int32"), ("route_weights", (num_tokens, K), "float32"),
                  ("w1_alpha", (L,), "float32"), ("w2_alpha", (L,), "float32")]
        if num_tokens > MOE_SORT_EXPERT_COUNTS_MIN_TOKENS:                                  # :2823-2824
            specs.append(("out_expert_counts", (2 * self.layout.num_experts,), "int32"))
        return _layout(specs)

    def get_workspace_size(self, num_tokens: int, do_finalize: bool = True) -> int:
        """:2824-2832 (see :meth:`workspace_fields`)."""
        return self.workspace_fields(num_tokens, do_finalize)[1]

    # -- the whole decision for one token count ---------------------------------------------------------
    def decide(self, num_tokens: int, do_finalize: bool = True) -> PlanDecision:
        T = int(num_tokens)
        if T <= 0:
            raise ValueError("num_tokens must be positive")
        use = self.use_swapab(T, do_finalize)
        if not use:
            return self._decide_dense(T)
        wide192 = self.swap_wide192(T, do_finalize)
        mixed192 = self.swap_mixed192(T, do_finalize)
        tile = self.swap_tile(T, do_finalize)
        hybrid = self.swap_hybrid(T, do_finalize)
        mixed = self.swap_mixed(T, do_finalize)
        split = self.swap_split(T, do_finalize)
        group = self.swap_group_rows(T, do_finalize)
        two_stage = self.two_stage(T, do_finalize)
        cap = self.fused_route_cap(T, do_finalize)
        fused = T * self.top_k <= cap
        pdl = self.enable_pdl or SWAP_PDL                                                    # :1148
        dep_prefetch = SWAP_DEP_PREFETCH and not (split or hybrid or mixed or mixed192)      # :1149-1151
        split_dense = split and SWAP_SPLIT_DENSE_GEMM2                                       # :1172-1173
        zero_fill = SWAP_WIDE192_ZERO_FILL if wide192 else "route"                           # :1174-1176
        if zero_fill == "dense" and not mixed192:
            zero_fill = "route"
        clear_output = (not two_stage or split_dense) and zero_fill == "route"               # :1182
        token_index = swap_row_tma(tile, True)                                               # :1410
        tiles = get_max_num_tiles(T, self.top_k, self.num_local_experts, group)
        rows = tiles * group
        mixed192_dual = self.swap_mixed192_dual(T, do_finalize)
        if mixed192_dual is not None:
            rows = get_max_num_tiles(T, self.top_k, self.num_local_experts, mixed192_dual[0]) * mixed192_dual[0]
            tiles = rows // group
        if split:
            rows = self.swap_split_capacity(T)[1]
        H, shard, K = self.hidden_size, self.intermediate_shard, self.top_k
        hint = self.swap_weight_l2_hint(T)
        mode_note = "mode = packed / separate_bf16 / separate_fp32 by the routing operands"
        plan: list[Launch] = []
        # ---- routing ------------------------------------------------------------------------------------------
        if fused:                                                                            # :1201-1250
            if mixed192_dual is not None:
                raise AssertionError("the dual-tile mixed form needs the moe_sort routing path (:1315-1318)")
            cfg = (f"_FusedRoutePreprocess tile_size={group} single_tile={group >= T} clear_output={clear_output} "
                   f"dispatch_lists={'yes' if (mixed and SWAP_MIXED_FUSED_LISTS) else 'no'} "
                   f"split_layout={'yes' if split else 'no'} max_routes={FUSED_ROUTE_MAX_ROUTES if T * K <= FUSED_ROUTE_MAX_ROUTES else FUSED_ROUTE_MAX_ROUTES_LARGE} ({mode_note}; mxfp4.py:1201-1250)")
            if mixed and SWAP_MIXED_FUSED_LISTS:
                plan.append(Launch("routing_fused", cfg, None, ROUTING_MODULE, BOTH,
                                   missing="routing dispatch_lists emission is traced (RoutingConfig.dispatch_lists) but the mixed "
                                           "form is off at 281907276 (SWAPAB_MIXED=0); never selected"))
            else:
                plan.append(Launch("routing_fused" if not split else "routing_split", cfg,
                                   "kimi_k3_mxfp4_situ_routing" + ("_split" if split else ""), ROUTING_MODULE, BOTH))
        else:                                                                                # :1251-1313
            plan.append(Launch("route_preprocess",
                               f"_RoutePreprocess threads={ROUTE_PREPROCESS_THREADS} clear_output={clear_output} "
                               f"({mode_note}; mxfp4.py:1254-1261, mxfp4_routing.py:64-160)",
                               "kimi_k3_mxfp4_situ_route_preprocess", ROUTING_MODULE, BOTH))
            init, coop = moe_sort_forms(self.layout.num_experts, num_tokens=T, top_k=self.top_k,
                                        sm_count=self.sm_count, tile=group,
                                        alt_tile=mixed192_dual[0] if mixed192_dual is not None else None,
                                        mixed=bool(mixed192 and SWAP_WIDE192_LISTS == "sort"), pdl=pdl)
            plan += [init, coop]
        # ---- dispatch -----------------------------------------------------------------------------------------
        win_side = mixed192_dual is not None and SWAP_WIDE192_MIXED_STREAMS == "win"        # :1067-1072
        if (hybrid or mixed) and not (mixed and SWAP_MIXED_FUSED_LISTS and fused):          # :1324-1345
            plan.append(Launch("dispatch",
                               f"swapab_dispatch group_rows={group} narrow_tile={tile} wide_min_rows={min(SWAP_HYBRID_DENSE_MIN_ROWS, group)} "
                               f"wide_min_permille={SWAP_MIXED_WIDE_PERMILLE if mixed else 0} all_lists={mixed} pdl={pdl} (mxfp4.py:1324-1345)",
                               "kimi_k3_mxfp4_situ_dispatch", DISPATCH_MODULE, BOTH))
        if mixed192 and not (mixed192_dual is not None and SWAP_WIDE192_LISTS == "sort"):   # :1364-1420
            plan.append(Launch("dispatch_mixed192",
                               f"swapab_dispatch_mixed group_rows={group} narrow_tile={tile} row_unit={SWAP_WIDE192_ROW_UNIT} "
                               f"max_rows={SWAP_WIDE192_MAX_ROWS} min_total_rows={SWAP_WIDE192_MIN_ROWS} dual={mixed192_dual is not None} (mxfp4.py:1364-1420)",
                               "kimi_k3_mxfp4_situ_dispatch_mixed", DISPATCH_MODULE, BOTH,
                               stream="side" if win_side else "main"))
        if token_index:                                                                      # :1421-1430
            plan.append(Launch("token_index", "fill_permuted_token_index (swapab_moe.py:505)",
                               "kimi_k3_mxfp4_situ_token_index", ROUTING_MODULE, BOTH))
        # ---- GEMM1 (swap-AB) ----------------------------------------------------------------------------------
        kb1 = gemm1_k_blocks_per_stage(tile)                                                 # swapab_moe.py:40-47
        k_tiles1 = H // (32 * kb1)
        g1_two_cta = swap_two_cta(tile)
        cluster_split = (SWAP_GEMM1_CLUSTER_SPLIT and dep_prefetch                           # :1450
                         and not g1_two_cta and k_tiles1 % 2 == 0                            # swapab_moe.py:716-729
                         and swap_m_group(tile, gemm2=False) == 1)
        g1_lists = bool(hybrid or mixed or mixed192)                                         # :1346-1363
        g1_sf_blocked = g1_lists and not (mixed and SWAP_MIXED_SF_PLAIN)
        g1 = swapab_situ_form(tile, kb1, row_group_list=g1_lists, sf_blocked=g1_sf_blocked, two_cta=g1_two_cta,
                              dep_prefetch=pdl and dep_prefetch, weight_l2_hint=hint)
        g1_dev = tuple(g1.deviations)
        if cluster_split:
            g1_dev += ("cluster_split_k (swapab_moe.py:716-729)",)
        g1 = Launch(g1.step, g1.hw + f" enable_pdl={pdl}", g1.form, g1.module, g1.backends, g1.missing,
                    "side" if win_side else "main", g1_dev)
        gemm1 = GemmForm(n_tile=tile, kbps=kb1, m_group=1, k_tiles=k_tiles1, use_pdl=pdl, weight_l2_hint=hint,
                         hw_split_k=2 if cluster_split else 1, hw_cluster_split_k=cluster_split,
                         late_dep_wait=False, pdl_trigger_after_wait=pdl and dep_prefetch,
                         two_cta=g1_two_cta, row_tma=swap_row_tma(tile, True), row_group_list=g1_lists,
                         sf_blocked=g1_sf_blocked)
        # ---- split form: dense wide GEMM1 (+ dense wide GEMM2) --------------------------------------------------
        wide_plan: list[Launch] = []
        if split:                                                                            # :1462-1553
            gemm1_n = next(v for limit, v in SWAP_SPLIT_GEMM1_N_POLICY if T <= limit)
            wide_plan.append(gemm1_dense_form((SWAP_SPLIT_WIDE_TILE, gemm1_n), (1, 1), row_group_list=True,
                                              pdl_trigger_early=pdl and SWAP_SPLIT_EARLY_TRIGGER))
            if split_dense:
                gemm2_tactic = self.tactic(T)[2]
                gemm2_n = gemm2_tactic[0][1]
                cl = gemm2_tactic[1] if gemm2_tactic[0] == (SWAP_SPLIT_WIDE_TILE, gemm2_n) else (1, 1)
                wide_plan.append(gemm2_dense_form((SWAP_SPLIT_WIDE_TILE, gemm2_n), cl, hidden_size=self.hidden_size, row_group_list=True,
                                                  pdl_trigger_early=pdl and SWAP_SPLIT_EARLY_TRIGGER))
            else:
                wide_plan.append(swapab_gemm2_form(SWAP_SPLIT_WIDE_TILE, gemm2_k_blocks_per_stage(shard, SWAP_SPLIT_WIDE_TILE),
                                                   swap_m_group(SWAP_SPLIT_WIDE_TILE, gemm2=True), finalize=not two_stage,
                                                   row_group_list=True, sf_blocked=True))
            side = "side" if SWAP_SPLIT_SIDE_STREAM else "main"
            wide_plan = [Launch(l.step, l.hw, l.form, l.module, l.backends, l.missing, side, l.deviations) for l in wide_plan]
        # ---- hybrid / mixed192: dense GEMM1 tiles over the wide list ----------------------------------------------
        dense_g1: list[Launch] = []
        if ((hybrid or mixed) and group > SWAP_HYBRID_DENSE_MIN_ROWS) or mixed192:          # :1554-1677
            g1_tactic = self.tactic(T)[1]
            if mixed192 and g1_tactic[0][0] != group:
                g1_tactic = ((group, 256), (1, 1), False)
            zero_in_dense = zero_fill == "dense"
            dense_g1.append(gemm1_dense_form(g1_tactic[0], g1_tactic[1], row_group_list=True, zero_fill=zero_in_dense,
                                             cluster_split_k=(mixed192 and self.dense_gemm1_cluster_split(g1_tactic, T))))
            if mixed192_dual is not None:                                                    # :1678-1727
                alt = mixed192_dual[1]
                dense_g1.append(gemm1_dense_form(alt[0], alt[1], row_group_list=True, zero_fill=zero_in_dense,
                                                 zero_fill_secondary=zero_in_dense))
        # ---- GEMM2 --------------------------------------------------------------------------------------------
        mixed192_gemm2 = self.swap_mixed192_gemm2(T) if mixed192 else None
        mixed192_dense_gemm2 = mixed192 and mixed192_gemm2 == "dense"
        mixed192_alt_gemm2 = mixed192 and mixed192_gemm2 == "alt" and mixed192_dual is not None
        gemm2 = None
        g2_main: list[Launch] = []
        if hybrid or mixed192_dense_gemm2:                                                   # :1738-1795
            g2_tactic = self.tactic(T)[2]
            if mixed192_dense_gemm2 and g2_tactic[0][0] != group:
                g2_tactic = ((group, 192), (1, 2), False)
            raster = self.gemm2_raster(T, g2_tactic[0][1]) if mixed192_dense_gemm2 else (False, 1)
            g2_main.append(gemm2_dense_form(g2_tactic[0], g2_tactic[1], hidden_size=self.hidden_size, raster_along_m=raster[0], swizzle=raster[1],
                                            c_stages=DENSE_GEMM2_C_STAGES if mixed192_dense_gemm2 else 1))
            if mixed192_dense_gemm2 and mixed192_dual is not None:                           # :1796-1801
                raster_b = self.gemm2_raster(T, self.tactic(T)[2][0][1])
                g2_main.append(gemm2_dense_form(mixed192_dual[2][0], mixed192_dual[2][1], hidden_size=self.hidden_size, raster_along_m=raster_b[0],
                                                swizzle=raster_b[1], c_stages=DENSE_GEMM2_C_STAGES))
        else:                                                                                # _prepare_swap_gemm2 :1897-1990
            kb2 = None
            if T <= SWAP_GEMM2_SHORT_STAGE_MAX_TOKENS and shard > SWAP_TWO_STAGE_MAX_SHARD:
                kb2 = 4
                if dep_prefetch and shard % 256 == 0 and shard // 256 <= SWAP_MAX_AB_STAGES:
                    kb2 = 8
            mg2 = None
            if shard <= SWAP_TWO_STAGE_MAX_SHARD and SWAP_TP_GEMM2_MGROUP > 1 and T >= SWAP_TP_GEMM2_MGROUP_MIN_TOKENS:
                mg2 = SWAP_TP_GEMM2_MGROUP
                kb2 = 4
            g2_two_cta = swap_two_cta(tile)
            if wide192:                                                                      # :1931-1935
                kb2 = None
                mg2 = 1
            if mixed:                                                                        # :1937-1950
                kb2 = 4 if SWAP_MIXED_GEMM2_MGROUP > 1 else kb2
                mg2 = SWAP_MIXED_GEMM2_MGROUP
            if kb2 is None:
                kb2 = gemm2_k_blocks_per_stage(shard, tile)                                  # swapab_moe.py:843-847
            if mg2 is None:
                mg2 = swap_m_group(tile, gemm2=True)
            k_tiles2 = shard // (32 * kb2)
            split_k = SWAP_GEMM2_SPLIT_K if (dep_prefetch and not wide192) else 1            # :1983-1987
            fused_finalize = do_finalize and not two_stage
            if split_k > 1 and (not fused_finalize or g2_two_cta or k_tiles2 % split_k):
                split_k = 1                                                                  # swapab_moe.py:880-887
            g2_lists = bool(mixed or mixed192)
            g2 = swapab_gemm2_form(tile, kb2, mg2, finalize=fused_finalize, row_group_list=g2_lists,
                                   sf_blocked=g2_lists and not (mixed and SWAP_MIXED_SF_PLAIN), two_cta=g2_two_cta,
                                   dep_prefetch=pdl and dep_prefetch, weight_l2_hint=hint)
            g2_dev = tuple(g2.deviations)
            if split_k > 1:
                g2_dev += ("split_k=2 device-side (swapab_moe.py:880-887)",)
            g2_main.append(Launch(g2.step, g2.hw + f" enable_pdl={pdl}", g2.form, g2.module, g2.backends, g2.missing,
                                  "side" if (win_side and mixed192_gemm2 != "dense") else "main", g2_dev))
            gemm2 = GemmForm(n_tile=tile, kbps=kb2, m_group=mg2, k_tiles=k_tiles2, use_pdl=pdl, weight_l2_hint=hint,
                             hw_split_k=split_k, hw_cluster_split_k=False, late_dep_wait=pdl and dep_prefetch,
                             pdl_trigger_after_wait=False, two_cta=g2_two_cta, row_tma=False)
        g2_wide: list[Launch] = []
        if mixed192 and not mixed192_dense_gemm2:                                            # :1803-1850
            g2_tactic = self.tactic(T)[2]
            if g2_tactic[0][0] != group:
                g2_tactic = ((group, 192), (1, 2), False)
            raster = self.gemm2_raster(T, g2_tactic[0][1])
            g2_wide.append(gemm2_dense_form(g2_tactic[0], g2_tactic[1], hidden_size=self.hidden_size, raster_along_m=raster[0], swizzle=raster[1],
                                            c_stages=DENSE_GEMM2_C_STAGES, row_group_list=True))
            if mixed192_dual is not None:                                                    # :1851-1862, :1820-1861
                raster_b = self.gemm2_raster(T, self.tactic(T)[2][0][1])
                g2_wide.append(gemm2_dense_form(mixed192_dual[2][0], mixed192_dual[2][1], hidden_size=self.hidden_size, raster_along_m=raster_b[0],
                                                swizzle=raster_b[1], c_stages=DENSE_GEMM2_C_STAGES,
                                                row_group_list=not mixed192_alt_gemm2))
        # ---- finalize rows (two-stage) --------------------------------------------------------------------------
        fin: list[Launch] = []
        if two_stage:                                                                        # :1866-1882
            fin.append(Launch("finalize_rows",
                              f"plan_finalize_rows top_k={K} threads={FINALIZE_ROWS_THREADS} expanded_rows=False "
                              f"accumulate={split_dense} skip_wide={split_dense} (narrow {tile} / wide {SWAP_SPLIT_WIDE_TILE}; mxfp4.py:1866-1882)",
                              "kimi_k3_mxfp4_situ_finalize_rows" + ("_split" if split_dense else ""), FINALIZE_MODULE, BOTH))
        # ---- enqueue order of run() (:1992-2160) -------------------------------------------------------------
        if win_side:
            dense_first = T >= SWAP_WIDE192_DENSE_FIRST_MIN_TOKENS
            window = [l for l in plan if l.step == "dispatch_mixed192"] + [g1]
            plan = [l for l in plan if l.step != "dispatch_mixed192"]
            plan += (dense_g1 + window) if dense_first else (window + dense_g1)
            plan += g2_main + g2_wide
        else:
            plan += wide_plan if split else []
            plan += [g1] + dense_g1 + g2_main + g2_wide
        plan += fin
        if wide192:
            path = "mixed192" if mixed192 else "wide192"
        elif hybrid:
            path = "hybrid"
        elif mixed:
            path = "mixed"
        elif split:
            path = "split_two_stage" if two_stage else "split"
        elif two_stage:
            path = "two_stage"
        else:
            path = "plain"
        return self._finish(T, plan, path, use=True, tile=tile, group=group, hybrid=hybrid, mixed=mixed, split=split,
                            wide192=wide192, two_stage=two_stage, cap=cap, fused=fused, clear_output=clear_output,
                            pdl=pdl, dep_prefetch=dep_prefetch, token_index=token_index, tiles=tiles, rows=rows,
                            gemm1=gemm1, gemm2=gemm2, mixed192=mixed192, split_dense=split_dense, dense=None)

    def _decide_dense(self, T: int) -> PlanDecision:
        """The dense path (``Mxfp4MoEPlan``, mxfp4.py:822-995, ``plan`` :3007-3088, ``_moe_core_impl``)."""
        sel = self.dense_selection(T)
        pdl = self.enable_pdl                                                                # kwargs enable_pdl
        K = self.top_k
        cap = self.fused_route_cap(T)
        clears = not sel.fill_in_gemm1 and not pdl                                           # :917-920
        plan: list[Launch] = []
        if T <= SWAP_ATOMIC_FINALIZE_MAX_TOKENS:
            raise AssertionError("T <= 16 always takes the swap-AB path for this family (:2442-2452)")
        mode_note = "mode = packed / separate_bf16 / separate_fp32 by the routing operands"
        plan.append(Launch("route_preprocess",
                           f"_RoutePreprocess threads={ROUTE_PREPROCESS_THREADS} clear_output={clears} ({mode_note}; "
                           "mxfp4.py:912-935: T > 16, ROUTE_PREPROCESS_PDL)",
                           "kimi_k3_mxfp4_situ_route_preprocess", ROUTING_MODULE, BOTH))
        init, coop = moe_sort_forms(self.layout.num_experts, num_tokens=T, top_k=self.top_k, sm_count=self.sm_count,
                                    tile=sel.tile, alt_tile=sel.dual[0] if sel.dual is not None else None, mixed=False,
                                    pdl=pdl)
        plan += [init, coop]
        if DENSE_GEMM1_A_TMA:
            plan.append(Launch("token_index", "fill_permuted_token_index (DENSE_GEMM1_A_TMA)",
                               "kimi_k3_mxfp4_situ_token_index", ROUTING_MODULE, BOTH))
        # ``moe_output_memset_inplace`` is prepared (use_async_memset=False) but ``run`` skips it: the route
        # preprocess clears below 8192 tokens and the gather GEMM1 zero-fills from 8192 (:961-969).
        plan.append(gemm1_dense_form(sel.gemm1[0], sel.gemm1[1], cluster_split_k=sel.gemm1_cluster_split_k,
                                     zero_fill=sel.fill_in_gemm1))
        if sel.dual is not None:                                                             # fused_moe.py:481-541
            plan.append(gemm1_dense_form(sel.dual[1][0], sel.dual[1][1], zero_fill=sel.fill_in_gemm1,
                                         zero_fill_secondary=sel.fill_in_gemm1))
        plan.append(gemm2_dense_form(sel.gemm2[0], sel.gemm2[1], hidden_size=self.hidden_size, raster_along_m=sel.gemm2_raster[0],
                                     swizzle=sel.gemm2_raster[1], c_stages=DENSE_GEMM2_C_STAGES))
        if sel.dual is not None:                                                             # fused_moe.py:697-736
            plan.append(gemm2_dense_form(sel.dual[2][0], sel.dual[2][1], hidden_size=self.hidden_size, raster_along_m=sel.gemm2_raster[0],
                                         swizzle=sel.gemm2_raster[1], c_stages=DENSE_GEMM2_C_STAGES))
        if sel.two_stage:
            plan.append(Launch("finalize_rows", "plan_finalize_rows expanded_rows=True (:890-897)",
                               None, FINALIZE_MODULE, BOTH,
                               missing="finalize_rows expanded_rows form (FinalizeConfig.expanded_rows=True not registered)"))
        tiles = get_max_num_tiles(T, K, self.num_local_experts, sel.dual[0] if sel.dual else sel.tile)
        rows = tiles * (sel.dual[0] if sel.dual else sel.tile)
        return self._finish(T, plan, "dense", use=False, tile=sel.tile, group=sel.tile, hybrid=False, mixed=False,
                            split=False, wide192=False, two_stage=sel.two_stage, cap=cap, fused=False,
                            clear_output=clears, pdl=pdl, dep_prefetch=False, token_index=DENSE_GEMM1_A_TMA,
                            tiles=rows // sel.tile, rows=rows, gemm1=None, gemm2=None, mixed192=False,
                            split_dense=False, dense=sel,
                            extra_missing=self._dense_extra_missing(sel))

    @staticmethod
    def _dense_extra_missing(sel: "DenseSelection") -> tuple[str, ...]:
        """Why a traced dense decision is still not executable here (empty = ``CakeDensePlan`` runs it)."""
        if not DENSE_CHAIN_EXECUTABLE:
            return (DENSE_CHAIN_MISSING,)
        if sel.two_stage:
            return (DENSE_TWO_STAGE_CHAIN_MISSING,)
        return ()

    def _finish(self, T, plan, path, *, use, tile, group, hybrid, mixed, split, wide192, two_stage, cap, fused,
                clear_output, pdl, dep_prefetch, token_index, tiles, rows, gemm1, gemm2, mixed192, split_dense,
                dense, extra_missing: tuple[str, ...] = ()) -> PlanDecision:
        plan = tuple(plan)
        missing = tuple(dict.fromkeys([l.missing for l in plan if not l.traced] + list(extra_missing)))
        mixed_backend = any("cutedsl" not in l.backends for l in plan)
        reasons = []
        if not use:
            reasons.append(f"dense grouped-GEMM path (T > swapab_max_tokens {self.swapab_max_tokens}, :2442-2452)")
        if wide192:
            reasons.append("wide 192-row 2-CTA swap form (:2454-2465)" + (" mixed192 (:2467-2471)" if mixed192 else ""))
        if hybrid:
            reasons.append("hybrid form: 128-row sort groups + dense finalize GEMM2 (:2522-2531)")
        if mixed:
            reasons.append("mixed form (:2533-2550)")
        if split:
            reasons.append("split form: dense gather GEMM1 + dense finalize GEMM2 over the wide experts (:2568-2581, :1462-1553)")
        if two_stage:
            reasons.append("two-stage finalize: GEMM2 partial epilogue + plan_finalize_rows kernel (:1099-1105, :1866-1882)")
        if use and not fused:
            reasons.append("generic routing: conversion + moe_sort launches (:1251-1313)")
        reason = ""
        if missing:
            reason = "; ".join(reasons) + " -- missing IR forms: " + " | ".join(missing)
        return PlanDecision(
            num_tokens=T, use_swapab=use, tile=tile, group_rows=group, hybrid=hybrid, mixed=mixed, split=split,
            wide192=wide192, two_stage=two_stage, fused_route_cap=cap, fused_routing=use and fused,
            single_tile_per_expert=bool(use and group >= T),                                 # :1227
            clear_output=clear_output, pdl=pdl, dep_prefetch=dep_prefetch, token_index=token_index,
            tiles=tiles, rows=rows, launches=tuple(l.step for l in plan), gemm1=gemm1, gemm2=gemm2,
            supported=not missing, reason=reason, path=path, launch_plan=plan, missing_forms=missing,
            mixed_backend=mixed_backend, mixed192=mixed192, split_dense=split_dense, dense=dense)


def plan_decision(*, num_tokens: int, top_k: int, hidden_size: int, num_experts: int, intermediate_size: int,
                  num_local_experts: int, local_expert_offset: int, intermediate_shard: int | None = None,
                  do_finalize: bool = True, sm_count: int = CONTRACT_SM_COUNT) -> PlanDecision:
    """Host-only decision for one problem (no torch, no CUDA); ``sm_count`` = the target device's SMs."""
    layout = resolve_layout(num_experts, intermediate_size, num_local_experts=num_local_experts,
                            local_expert_offset=local_expert_offset, intermediate_shard=intermediate_shard)
    return CakeSwapAbPolicy(layout, top_k=top_k, hidden_size=hidden_size, sm_count=sm_count).decide(num_tokens,
                                                                                                    do_finalize)


def interleave_up_gate(x: Any, group_size: int = 64, dim: int = 1) -> Any:
    """``prepare.py:2477-2488 _interleave_linear_and_gate``: [up rows | gate rows] -> 64-row up/gate groups."""
    sizes = x.size()
    dim = dim % x.dim()
    if sizes[dim] % (group_size * 2):
        raise ValueError("the interleaved dimension must be a multiple of 2 * group_size")
    prev_sizes, post_sizes = sizes[:dim], sizes[dim + 1:]
    x = x.view(*prev_sizes, 2, sizes[dim] // (group_size * 2), group_size, *post_sizes)
    return x.transpose(dim, dim + 1).contiguous().view(*sizes)


def pack_weights_bytes(packed: Any, scales: Any) -> tuple[Any, Any]:
    """Packed E2M1 ``(L, rows, K/2)`` bytes + linear UE8M0 ``(L, rows, K/32)`` -> the Cake kernels' TMA operands.

    Weights: ``(L * rows/128, K/128, 128, 64)`` tile-major (the hand-written ``tile_major_weights``,
    ``swapab_moe.py:114-141``, one contiguous 8 KB block per 128 x 128 MMA tile). Scales: ``(L * rows/128, K/128,
    4, 128)`` 512-byte atoms with byte ``(row % 32) * 16 + (row // 32) * 4 + kblock`` (the hand-written
    ``to_mma_layout`` bytes, ``prepare.py:2547-2558``, viewed per tile).  Same bytes as
    ``kimi_k3_mxfp4_situ_gemm2_swapab.pack_weights_tile_major`` produces from unpacked codes.
    """
    import torch

    L, rows, kb = packed.shape
    if rows % 128 or kb % 64 or scales.shape != (L, rows, kb * 2 // 32):
        raise ValueError("tile-major weights need rows % 128 == 0, K % 128 == 0 and matching linear scales")
    m_tiles, k_tiles = rows // 128, kb // 64
    tiled = packed.view(L, m_tiles, 128, k_tiles, 64).permute(0, 1, 3, 2, 4).reshape(
        L * m_tiles, k_tiles, 128, 64).contiguous()
    atoms = scales.view(L, m_tiles, 4, 32, k_tiles, 4).permute(0, 1, 4, 3, 2, 5).reshape(
        L * m_tiles, k_tiles, 4, 128).contiguous()
    return tiled.to(torch.uint8), atoms.to(torch.uint8)


def prepare_cake_mxfp4_weights(w1: Any, w1_scale: Any, w2: Any, w2_scale: Any) -> dict[str, Any]:
    """Canonical rank-local [up, gate] MXFP4 bank -> the Cake chain's operands.

    ``w1`` ``[L, 2*I_shard, H/2]`` (up rows then gate rows), ``w1_scale`` ``[L, 2*I_shard, H/32]``, ``w2``
    ``[L, H, I_shard/2]``, ``w2_scale`` ``[L, H, I_shard/32]`` (the contract's ``physical_abi``).  W1 rows and
    scales are interleaved in 64-row up/gate groups exactly as ``prepare_cute_dsl_mxfp4_weights`` does; both
    weights then take the tile-major layout and the 512-byte scale atoms.  All payload bytes are preserved.
    """
    for name, t in (("w1", w1), ("w1_scale", w1_scale), ("w2", w2), ("w2_scale", w2_scale)):
        if t.ndim != 3 or str(t.dtype) != "torch.uint8" or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous 3-D uint8 tensor")
    L, rows_w1, hk = w1.shape
    if w1_scale.shape != (L, rows_w1, hk * 2 // 32) or w2.shape[0] != L or w2_scale.shape[:2] != w2.shape[:2]:
        raise ValueError("inconsistent canonical MXFP4 shapes")
    a1, sfa1 = pack_weights_bytes(interleave_up_gate(w1), interleave_up_gate(w1_scale))
    a2, sfa2 = pack_weights_bytes(w2, w2_scale)
    return {"w1": a1, "w1_sf": sfa1, "w2": a2, "w2_sf": sfa2, "num_local_experts": L,
            "intermediate_shard": rows_w1 // 2, "hidden_size": hk * 2}


def routing_config(policy: CakeSwapAbPolicy, decision: PlanDecision, *, mode: str, split_layout: bool = False):
    """The ``RoutingConfig`` the hand-written ``_plan_route_preprocess`` selects for this row (:1201-1250).

    ``split_layout`` is the caller's statement of the chain it enqueues (the split chain's two-granularity row
    layout, :1236-1249); the plain chain run on a split decision keeps the plain layout."""
    rt = _routing_module()
    return rt.plan_config(tokens=decision.num_tokens, top_k=policy.top_k, num_local_experts=policy.num_local_experts,
                          tile_size=decision.group_rows, mode=mode, clear=decision.clear_output,
                          dispatch_lists=False, rows_capacity=decision.rows, scratch_words=SORT_SCRATCH_WORDS,
                          split_layout=split_layout)


_TORCH_DTYPE = {"int32": "int32", "float32": "float32", "uint8": "uint8", "float8_e4m3fn": "float8_e4m3fn",
                "bfloat16": "bfloat16"}


def _byte_interval(tensor) -> tuple[int, int]:
    """:809-816."""
    first = tensor.data_ptr()
    span = 1 + sum((dim - 1) * stride for dim, stride in zip(tensor.shape, tensor.stride(), strict=True))
    return first, first + span * tensor.element_size()


def _overlap(left, right) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


UNUSED_ROUTING_LISTS = ("wide_list", "wide_count", "narrow_list", "narrow_count", "all_list", "all_count",
                        "wide_expert", "wide_limit")


UNUSED_SORT_OPERANDS = ("alt_expert", "alt_limit", "alt_active", "base_active", "alt_wide_list", "alt_wide_count",
                        "narrow_count_base")


UNUSED_GEMM1_DENSE_OPERANDS = ("zero_fill_words", "zero_fill_other_tiles")


def unused_launch_operands(device, *, group_capacity: int) -> dict[str, Any]:
    """Operands of the kernel ABI the plain chain never dereferences.

    Routing (``dispatch_lists=False``, ``split_layout=False``): the six dispatch lists and the two wide-slot
    arrays are never written (static ``cfg`` guards) -> one-word buffers. GEMM template (narrow forms):
    ``tile_idx_to_row_group`` is read only by the mixed192 window form -> the identity list the module's own
    ``make_problem`` / ``make_gemm1_problem`` bind ("identity (no window list)").
    """
    import torch

    ops = {name: torch.zeros(1, dtype=torch.int32, device=device)
           for name in UNUSED_ROUTING_LISTS + UNUSED_SORT_OPERANDS}
    ops["tile_idx_to_row_group"] = torch.arange(group_capacity, dtype=torch.int32, device=device)
    ops["zero_buf"] = torch.zeros(2, dtype=torch.float32, device=device)   # GEMM1 zero source (zero_words = 0)
    for name in UNUSED_GEMM1_DENSE_OPERANDS:
        ops[name] = torch.zeros(1, dtype=torch.int32, device=device)
    ops["zero_fill_words"] = torch.zeros(1, dtype=torch.uint32, device=device)   # the kernel's u32 word pointer
    ops["zero_fill_counters"] = torch.zeros(2, dtype=torch.int32, device=device)
    return ops


def routing_launch_bindings(b: dict[str, Any], *, topk_ids, weights_src, w_strides, output, num_tokens: int,
                            top_k: int, num_experts: int, local_experts: int, local_offset: int, group_rows: int,
                            unused: dict[str, Any], split_layout: bool = False) -> dict[str, Any]:
    """Named bindings of the fused routing launch (K2, ``dispatch_lists=False``).

    Plain layout: ``wide_tile`` is the kernel's "rows per wide group (``tile_size`` otherwise)" operand and equals
    ``tile_size`` (routing module ``run_cake_routing``: ``wide_tile=p.get("wide_tile", p["tile_size"])``);
    ``wide_min_permille=0`` and ``wide_min_rows=narrow_tile=tile_size`` as in the plain chain; the wide slots and
    lists are the one-word dummies. Split layout (hand-written ``_plan_route_preprocess(split_layout=dict(...))``,
    mxfp4.py:1236-1249): the ``swap_wide_expert`` / ``swap_wide_limit`` slots at ``SWAP_SPLIT_WIDE_TILE``
    granularity from row 0, the compacted ``swap_wide_list`` / ``swap_wide_count``, wide experts above
    ``SWAP_SPLIT_MIN_ROWS`` rows holding ``SWAP_SPLIT_MIN_PERMILLE`` of the rows.
    """
    import torch

    output_words = num_tokens * output.shape[1] // 2
    lists = {name: unused[name] for name in UNUSED_ROUTING_LISTS}
    wide = dict(wide_min_rows=group_rows, wide_min_permille=0, wide_tile=group_rows)
    if split_layout:
        lists.update(wide_expert=b["swap_wide_expert"], wide_limit=b["swap_wide_limit"], wide_list=b["swap_wide_list"],
                     wide_count=b["swap_wide_count"])
        wide = dict(wide_min_rows=SWAP_SPLIT_MIN_ROWS, wide_min_permille=SWAP_SPLIT_MIN_PERMILLE,
                    wide_tile=SWAP_SPLIT_WIDE_TILE)
    return dict(
        ids_src=topk_ids, weights_src=weights_src, ids_dst=b["route_ids"].reshape(-1),
        weights_dst=b["route_weights"].reshape(-1), output=output.view(torch.uint32).reshape(-1),
        tile_expert=b["out_tile_idx_to_expert_idx"], tile_limit=b["out_tile_idx_to_mn_limit"],
        expanded=b["out_expanded_idx_to_permuted_idx"].reshape(-1), permuted=b["out_permuted_idx_to_expanded_idx"],
        padded_total=b["out_total_num_padded_tokens"], active_total=b["out_num_non_exiting_tiles"],
        **lists, chunk_counts=b["out_expert_counts"],
        num_routes=num_tokens * top_k, top_k=top_k, output_words=output_words, num_experts=num_experts,
        local_experts=local_experts, local_offset=local_offset, tile_size=group_rows, narrow_tile=group_rows,
        **wide, ids_stride0=topk_ids.stride(0), ids_stride1=topk_ids.stride(1), w_stride0=w_strides[0],
        w_stride1=w_strides[1])


def moe_sort_launch_bindings(b: dict[str, Any], *, topk_ids, num_tokens: int, top_k: int, num_experts: int,
                             local_experts: int, local_offset: int, tile: int, alt_tile: int | None,
                             permille: int, mixed: bool, narrow_tile: int,
                             unused: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Named bindings ``(init, coop)`` of the K6 launch pair, the hand-written ``moe_sort`` call (mxfp4.py:1269-
    1313): ``sort_buffers`` = the ``out_*`` workspace fields, the mixed lists = the ``swap_*`` dispatch fields
    (``SWAP_WIDE192_LISTS == "sort"``), ``tile_tokens_dim`` = the padding tile, ``tile_tokens_dim_alt`` /
    ``dual_tile_threshold_permille`` of the dual form. Same operand order as the module's ``coop_kwargs``."""
    dual = alt_tile is not None
    init = dict(expert_counts=b["out_expert_counts"], num_experts=num_experts)
    coop = dict(
        topk_ids=topk_ids, expert_counts=b["out_expert_counts"], tile_expert=b["out_tile_idx_to_expert_idx"],
        tile_limit=b["out_tile_idx_to_mn_limit"], expanded=b["out_expanded_idx_to_permuted_idx"].reshape(-1),
        permuted=b["out_permuted_idx_to_expanded_idx"], padded_total=b["out_total_num_padded_tokens"],
        active_total=b["out_num_non_exiting_tiles"],
        alt_expert=b["out_alt_tile_idx_to_expert_idx"] if dual else unused["alt_expert"],
        alt_limit=b["out_alt_tile_idx_to_mn_limit"] if dual else unused["alt_limit"],
        alt_active=b["out_alt_num_non_exiting_tiles"] if dual else unused["alt_active"],
        base_active=b["out_base_active_num_non_exiting_tiles"] if dual else unused["base_active"],
        wide_list=b["swap_wide_list"] if mixed else unused["wide_list"],
        wide_count=b["swap_wide_count"] if mixed else unused["wide_count"],
        alt_wide_list=b["swap_alt_wide_list"] if mixed else unused["alt_wide_list"],
        alt_wide_count=b["swap_alt_wide_count"] if mixed else unused["alt_wide_count"],
        narrow_list=b["swap_row_groups"] if mixed else unused["narrow_list"],
        narrow_count=b["swap_row_group_count"] if mixed else unused["narrow_count"],
        narrow_count_base=b["swap_row_group_count_base"] if mixed else unused["narrow_count_base"],
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k, local_offset=local_offset,
        local_experts=local_experts, stride_log2=0, padding_log2=tile.bit_length() - 1,
        padding_log2_alt=alt_tile.bit_length() - 1 if dual else 0, permille=permille if dual else 0,
        mixed_narrow_tile=narrow_tile if mixed else 0, mixed_row_unit=SWAP_WIDE192_ROW_UNIT if mixed else 0,
        mixed_max_rows=SWAP_WIDE192_MAX_ROWS, mixed_min_total_rows=SWAP_WIDE192_MIN_ROWS,
        contiguous_windows=1 if num_tokens >= MOE_SORT_CONTIGUOUS_WINDOW_MIN_TOKENS else 0)
    return init, coop


def _gemm_common_bindings(b: dict[str, Any], *, route_weights, tiles: int, top_k: int, dbg, unused) -> dict[str, Any]:
    return dict(
        tile_idx_to_expert_idx=b["out_tile_idx_to_expert_idx"], tile_idx_to_mn_limit=b["out_tile_idx_to_mn_limit"],
        num_non_exiting_tiles=b["out_num_non_exiting_tiles"], tile_idx_to_row_group=unused["tile_idx_to_row_group"],
        permuted_idx_to_expanded_idx=b["out_permuted_idx_to_expanded_idx"],
        token_final_scales=route_weights.reshape(-1), group_capacity=tiles, top_k=top_k, dbg=dbg)


def gemm1_launch_bindings(b: dict[str, Any], *, weights: dict[str, Any], x, x_sf, route_weights, hidden: int,
                          shard: int, top_k: int, num_tokens: int, tiles: int, k_tiles: int, beta, linear_beta, dbg,
                          unused: dict[str, Any], row_group_list: bool = False) -> dict[str, Any]:
    """Named bindings of the swap-AB SiTU GEMM1 launch (:1596-1640).

    ``row_group_list`` (the ``_rowgroup`` forms of the hybrid / mixed192 chains, hand-written :1346-1351 / :1403-1408):
    the work list is the dispatch kernel's narrow list ``swap_row_groups`` with its count ``swap_row_group_count``
    (``tile_idx_to_row_group`` / ``num_non_exiting_tiles``), ``group_capacity`` is the list capacity (hand-written
    ``tile_idx_to_row_group.shape[0]``, swapab_moe.py:797), and the expert / row-bound tables stay the 128-row sort
    groups' ``out_tile_idx_to_*``. ``tiles`` is the sort-group capacity either way.
    """
    import torch

    common = _gemm_common_bindings(b, route_weights=route_weights, tiles=tiles, top_k=top_k, dbg=dbg, unused=unused)
    if row_group_list:
        common.update(tile_idx_to_row_group=b["swap_row_groups"], num_non_exiting_tiles=b["swap_row_group_count"],
                      group_capacity=int(b["swap_row_groups"].numel()))
    return dict(
        A=weights["w1"], SFA=weights["w1_sf"], B=x.view(torch.uint8), SFB=x_sf, alpha=b["w1_alpha"], **common,
        num_m_tiles=2 * shard // 128, k_tiles=k_tiles, k_cols=hidden, sf_cols=hidden // 32, out_cols=shard,
        situ_beta=beta, situ_linear_beta=linear_beta, act_sf=b["gemm1_out_scale"], zero_buf=unused["zero_buf"],
        zero_words=0, num_rows_b=num_tokens, act_cols=shard, act_sf_cols=shard // 32,
        out=b["gemm1_out"].view(torch.uint8))


def dispatch_launch_bindings(b: dict[str, Any], *, group_rows: int, narrow_tile: int, wide_min_rows: int,
                             wide_min_permille: int, unused: dict[str, Any]) -> dict[str, Any]:
    """Named bindings of the K3 ``swapab_dispatch`` launch of the hybrid chain (hand-written mxfp4.py:1324-1345,
    ``all_list=None``): the wide list (dense GEMM1 tiles: sort groups with more than ``wide_min_rows`` valid rows)
    and the narrow list (swap-AB sub-tiles) over the K6 groups; the ``all`` list pointers are the one-word dummies
    (``DispatchConfig(all_lists=False)`` never writes them)."""
    return dict(mn_limit=b["out_tile_idx_to_mn_limit"], num_groups_ptr=b["out_num_non_exiting_tiles"],
                wide_list=b["swap_wide_list"], wide_count=b["swap_wide_count"], narrow_list=b["swap_row_groups"],
                narrow_count=b["swap_row_group_count"], all_list=unused["all_list"], all_count=unused["all_count"],
                group_rows=int(group_rows), narrow_tile=int(narrow_tile), wide_min_rows=int(wide_min_rows),
                wide_min_permille=int(wide_min_permille))


def gemm1_dense_launch_bindings(b: dict[str, Any], *, weights: dict[str, Any], x, x_sf, hidden: int, shard: int,
                                top_k: int, num_tokens: int, rows: int, tile_m: int, n_tile: int, row_group_list: bool,
                                zero_fill: bool, secondary: bool, output, beta, linear_beta, beta_stride: int,
                                linear_beta_stride: int, unused: dict[str, Any], split_layout: bool = False,
                                dual: bool = False) -> dict[str, Any]:
    """Named bindings of a dense gather GEMM1 launch (hand-written
    ``blockscaled_contiguous_gather_grouped_gemm_act_fusion`` operands, mxfp4.py / fused_moe.py:481-541).

    Tile lists: the single-tile base launch reads ``out_tile_idx_to_*`` / ``out_num_non_exiting_tiles``. In a dual-tile
    pair (``dual``) the base launch reads ``out_tile_idx_to_*`` with the valid count
    ``out_base_active_num_non_exiting_tiles`` (zero when the routing chose the 256-row padding, fused_moe.py:496-500)
    and fills the other tiles of ``out_alt_num_non_exiting_tiles``; the 2-CTA ``secondary`` launch reads
    ``out_alt_tile_idx_to_*`` / ``out_alt_num_non_exiting_tiles`` (256-row groups) and
    ``zero_fill_other_tiles = out_base_active_num_non_exiting_tiles`` (fused_moe.py:546-593). Row-group forms (split /
    hybrid / mixed192) take the compacted
    ``swap_wide_list`` / ``swap_wide_count``; plain forms take the identity list and the zero-fill dummies.
    The kernel indexes ``tile_idx_to_expert_idx`` / ``tile_idx_to_mn_limit`` by the row GROUP the list names: in the
    split layout the groups are the ``SWAP_SPLIT_WIDE_TILE``-row wide slots from row 0, so the lists are the
    routing kernel's ``swap_wide_expert`` / ``swap_wide_limit`` (hand-written mxfp4.py:1478-1482); the 128-row sort
    groups of the hybrid / mixed192 forms use ``out_tile_idx_to_*``.
    """
    import torch

    if secondary and not dual:
        raise ValueError("the zero_fill_secondary launch is the alternate of a dual-tile pair (dual=True)")
    if row_group_list:
        if split_layout:
            tile_expert, tile_limit = b["swap_wide_expert"], b["swap_wide_limit"]
        else:
            tile_expert, tile_limit = b["out_tile_idx_to_expert_idx"], b["out_tile_idx_to_mn_limit"]
        row_group, valid = b["swap_wide_list"], b["swap_wide_count"]
    elif secondary:
        tile_expert, tile_limit = b["out_alt_tile_idx_to_expert_idx"], b["out_alt_tile_idx_to_mn_limit"]
        row_group, valid = unused["tile_idx_to_row_group"], b["out_alt_num_non_exiting_tiles"]
    else:
        tile_expert, tile_limit = b["out_tile_idx_to_expert_idx"], b["out_tile_idx_to_mn_limit"]
        row_group = unused["tile_idx_to_row_group"]
        valid = b["out_base_active_num_non_exiting_tiles"] if dual else b["out_num_non_exiting_tiles"]
    if zero_fill:
        zf_words = output.view(torch.uint32).reshape(-1)
        zf_num = num_tokens * hidden // 2
        if secondary:
            zf_other = b["out_base_active_num_non_exiting_tiles"]
        elif dual:
            zf_other = b["out_alt_num_non_exiting_tiles"]
        else:
            zf_other = b["out_num_non_exiting_tiles"]
    else:
        zf_words, zf_num, zf_other = unused["zero_fill_words"], 0, unused["zero_fill_other_tiles"]
    return dict(
        X=x.view(torch.uint8), XSF=x_sf, W=weights["w1"], WSF=weights["w1_sf"], C=b["gemm1_out"].view(torch.uint8),
        CSF=b["gemm1_out_scale"], alpha=b["w1_alpha"], situ_beta=beta, situ_linear_beta=linear_beta, tile_idx_to_expert_idx=tile_expert,
        tile_idx_to_mn_limit=tile_limit, tile_idx_to_row_group=row_group,
        token_id_mapping=b["out_permuted_idx_to_expanded_idx"], num_non_exiting_tiles=valid,
        zero_fill_words=zf_words, zero_fill_counters=unused["zero_fill_counters"], zero_fill_other_tiles=zf_other,
        zero_fill_num_words=zf_num, num_m_tiles=rows // tile_m, n_tiles=2 * shard // n_tile, k_tiles=hidden // 128,
        k_cols=hidden, sf_cols=hidden // 32, top_k=top_k, sf_n_blocks=-(-shard // 128), beta_stride=beta_stride,
        linear_beta_stride=linear_beta_stride)


def gemm2_launch_bindings(b: dict[str, Any], *, weights: dict[str, Any], route_weights, hidden: int, shard: int,
                          top_k: int, tiles: int, k_tiles: int, output, dbg, unused: dict[str, Any]) -> dict[str, Any]:
    """Named bindings of the swap-AB finalize GEMM2 launch (:1957-1990); SiTU operands are the module's placeholders."""
    import torch

    return dict(
        A=weights["w2"], SFA=weights["w2_sf"], B=b["gemm1_out"].view(torch.uint8), SFB=b["gemm1_out_scale"],
        alpha=b["w2_alpha"],
        **_gemm_common_bindings(b, route_weights=route_weights, tiles=tiles, top_k=top_k, dbg=dbg, unused=unused),
        num_m_tiles=hidden // 128, k_tiles=k_tiles, k_cols=shard, sf_cols=shard // 32, out_cols=hidden,
        out=output, **_gemm_module()._unused_situ_operands(output.device))


def preprocess_launch_bindings(b: dict[str, Any], *, topk_ids, weights_src, w_strides, output, num_tokens: int,
                               top_k: int, hidden: int) -> dict[str, Any]:
    """Named bindings of the K1 ``_RoutePreprocess`` launch (mxfp4_routing.py:908-935): packed IDs unpack into
    ``route_ids``, BF16 / packed weights convert into ``route_weights`` (both bound in every mode; the static
    ``cfg`` guards skip the writes the mode does not need), the output is cleared as 32-bit words."""
    import torch

    return dict(
        ids_src=topk_ids, weights_src=weights_src, ids_dst=b["route_ids"].reshape(-1),
        weights_dst=b["route_weights"].reshape(-1), output=output.view(torch.uint32).reshape(-1),
        num_routes=num_tokens * top_k, top_k=top_k, output_words=num_tokens * hidden // 2,
        ids_stride0=topk_ids.stride(0), ids_stride1=topk_ids.stride(1), w_stride0=w_strides[0],
        w_stride1=w_strides[1])


def gemm2_dense_launch_bindings(b: dict[str, Any], *, w2, w2_sf, route_weights, hidden: int, shard: int, top_k: int,
                                tiles: int, rows: int, n_tile: int, output, dbg, unused: dict[str, Any],
                                row_group: bool = False, split_layout: bool = False, dual: bool = False,
                                secondary: bool = False) -> dict[str, Any]:
    """Named bindings of the dense finalize GEMM2 launch (hand-written
    ``blockscaled_contiguous_grouped_gemm_finalize_fusion``, fused_moe.py:697-736): ``A`` = the W1 activation
    (``gemm1_out``, ``(R, I)`` E4M3) with its blocked scales viewed as ``(R/128, I/128, 4, 128)`` atoms (W1
    ``output_scales_unblock`` byte order == W11 ``blocked_scales``; the descriptor's rank), ``B`` = row-major
    ``(L, H, I/2)`` weights, ``SFB`` = blocked ``(L, H/128, I/128, 4, 128)`` scales, ``num_n_tiles = ceil(H / N)``
    (the N192 ragged tail). ``row_group`` (the ``_rg`` form, split chain wide GEMM2, mxfp4.py:1506-1530): the
    scheduler slots are ``tiles`` = the ``swap_wide_list`` capacity, the valid count ``swap_wide_count``, and the
    expert / limit lists are indexed by the listed group (``swap_wide_expert`` / ``swap_wide_limit`` in the split
    layout). ``dual``: the launch belongs to a dual-tile pair -- the base launch's valid count is
    ``out_base_active_num_non_exiting_tiles`` (fused_moe.py:654-658), the ``secondary`` (2-CTA (256, 256) alternate)
    launch reads ``out_alt_tile_idx_to_*`` / ``out_alt_num_non_exiting_tiles`` over 256-row groups with ``tiles`` =
    the 256-row capacity (fused_moe.py:684-717)."""
    import torch

    if secondary and (not dual or row_group):
        raise ValueError("the alternate GEMM2 launch is the 2-CTA member of a dual-tile pair (dual=True, no row-group list)")
    if row_group:
        tile_expert, tile_limit = ((b["swap_wide_expert"], b["swap_wide_limit"]) if split_layout
                                   else (b["out_tile_idx_to_expert_idx"], b["out_tile_idx_to_mn_limit"]))
        valid, group_list = b["swap_wide_count"], b["swap_wide_list"]
    elif secondary:
        tile_expert, tile_limit = b["out_alt_tile_idx_to_expert_idx"], b["out_alt_tile_idx_to_mn_limit"]
        valid, group_list = b["out_alt_num_non_exiting_tiles"], unused["tile_idx_to_row_group"]
    else:
        tile_expert, tile_limit = b["out_tile_idx_to_expert_idx"], b["out_tile_idx_to_mn_limit"]
        valid = b["out_base_active_num_non_exiting_tiles"] if dual else b["out_num_non_exiting_tiles"]
        group_list = unused["tile_idx_to_row_group"]
    return dict(
        A=b["gemm1_out"].view(torch.uint8), SFA=b["gemm1_out_scale"].reshape(rows // 128, shard // 128, 4, 128),
        B=w2, SFB=w2_sf, tile_idx_to_expert_idx=tile_expert,
        tile_idx_to_mn_limit=tile_limit, num_non_exiting_tiles=valid,
        alpha=b["w2_alpha"], permuted_idx_to_expanded_idx=b["out_permuted_idx_to_expanded_idx"],
        token_final_scales=route_weights.reshape(-1), tile_idx_to_row_group=group_list,
        num_m_tiles=tiles, num_n_tiles=-(-hidden // n_tile), k_tiles=shard // 128, out_cols=hidden, top_k=top_k,
        dbg=dbg, out=output)


def finalize_rows_launch_bindings(b: dict[str, Any], *, route_weights, output, num_tokens: int, hidden: int,
                                  narrow_tile: int) -> dict[str, Any]:
    """Named bindings of the K7 ``finalize_rows`` launch of the split two-stage chain (hand-written
    ``plan_finalize_rows(partial_rows, out_expanded_idx_to_permuted_idx, route_weights, output, accumulate=True,
    skip_wide=(out_num_non_exiting_tiles, swap_wide_count, n_tile, SWAP_SPLIT_WIDE_TILE))``, mxfp4.py:1866-1882)."""
    import torch

    fin = _finalize_module()
    chunks = (hidden // 2) // fin.WORDS_PER_CHUNK
    return dict(rows=b["partial_rows"].view(torch.uint32).reshape(-1), perm=b["out_expanded_idx_to_permuted_idx"].reshape(-1),
                weights=route_weights.reshape(-1), out=output.view(torch.uint32).reshape(-1),
                narrow_count=b["out_num_non_exiting_tiles"], wide_count=b["swap_wide_count"], chunks=chunks,
                tasks=num_tokens * chunks, narrow_tile=narrow_tile, wide_tile=SWAP_SPLIT_WIDE_TILE)


def dense_weight_operands(weights: dict[str, Any], *, local_experts: int, hidden: int, shard: int) -> dict[str, Any]:
    """The dense GEMMs' weight operands from the prepared swap-AB operands (``prepare_cake_mxfp4_weights``).

    Weights: tile-major ``(L * rows/128, K/128, 128, 64)`` -> row-major packed ``(L, rows, K/2)`` (the W1 ``W`` /
    W11 ``B`` TMA tensors; W1 rows are the 64-row up/gate interleave the swap-AB prep already applied). Scales: the
    512-byte atoms are shared -- free views in the **rank each kernel's TMA descriptor derives its global dims
    from** (``LM.axis[-k]``): W1 ``WSF`` is the physical 4-D ``(L, rows/128, K/128, 512)`` tensor
    (``gemm1_dense.weight_scales_blocked``; axis[-2] = K atoms), W11 ``SFB`` the 5-D ``(L, rows/128, K/128, 4,
    128)`` tensor (``gemm2_dense.blocked_scales``; axis[-3] = K atoms). Round-10 root cause of the uncorrelated
    dense output: ``w1_sf`` was bound as the 5-D view, so the W1 descriptor read axis[-2] = 4 as
    the K-atom extent (K atoms >= 4 out of bounds -> zero-filled UE8M0 code 0 -> x 2^-127) with wrong atom strides."""
    def rows_major(t, rows, k):
        return (t.view(local_experts, rows // 128, k // 128, 128, 64).permute(0, 1, 3, 2, 4)
                .reshape(local_experts, rows, k // 2).contiguous())

    return {"w1": rows_major(weights["w1"], 2 * shard, hidden),
            "w1_sf": weights["w1_sf"].view(local_experts, 2 * shard // 128, hidden // 128, 512),
            "w2": rows_major(weights["w2"], hidden, shard),
            "w2_sf": weights["w2_sf"].view(local_experts, hidden // 128, shard // 128, 4, 128)}


def dense_launch_config(policy: CakeSwapAbPolicy, decision: PlanDecision) -> dict[str, Any]:
    """Host-only resolution of the dense chain's module configurations from the decision's launch plan.

    Nothing is inferred from names: the K6 configuration is rebuilt from the decision inputs and must reproduce the
    plan's form names exactly; the GEMM forms come from the pinned tables. The dual-tile chain (``sel.dual``) adds
    ``alt_tile`` (256), ``gemm1_alt`` (the 2-CTA ``zero_fill_secondary`` form args), ``gemm2_alt`` = ``(256, 2, 1)``
    (``n_tile``, ``cta_group``, ``cluster_n``) and ``pdl_alt`` = ``enable_pdl or DENSE_DUAL_ALT_PDL`` (fused_moe.py:579,
    :707); the single-tile chain carries ``None`` for the three. Raises ``NotImplementedError`` for the dense two-stage
    finalize (traced, not enqueued; never selected for this family)."""
    if decision.path != "dense" or decision.dense is None:
        raise ValueError(f"not a dense decision (path {decision.path!r})")
    if not decision.supported:
        raise NotImplementedError(decision.reason)
    sel = decision.dense
    if sel.two_stage:
        raise NotImplementedError(f"T={decision.num_tokens}: {DENSE_TWO_STAGE_CHAIN_MISSING}")
    dual = sel.dual is not None
    steps = [l.step for l in decision.launch_plan]
    expected = (["route_preprocess", "moe_sort_init", "moe_sort_coop", "gemm1_dense"] + (["gemm1_dense"] if dual else [])
                + ["gemm2_dense_finalize"] + (["gemm2_dense_finalize"] if dual else []))
    if steps != expected:
        raise NotImplementedError(f"dense launch sequence {decision.launches} is not the executable chain {expected}")
    forms = [l.form for l in decision.launch_plan]
    ms = _moe_sort_module()
    tier = moe_sort_expert_tier(policy.layout.num_experts)
    sort_cfg = ms.MoeSortConfig(tier=tier, dual=dual, mixed=False, narrow_count_base=False, pdl=decision.pdl,
                                bounded=moe_sort_bounded_state(decision.num_tokens, policy.top_k, tier, policy.sm_count))
    names = (ms.init_form_name(sort_cfg), ms.coop_form_name(sort_cfg))
    if names != (forms[1], forms[2]):
        raise AssertionError(f"K6 configuration {names} does not reproduce the plan's forms")
    gemm1 = GEMM1_DENSE_FORMS[forms[3]]
    if gemm1[2] != sel.fill_in_gemm1 or gemm1[3] or gemm1[4] or gemm1[5]:
        raise AssertionError(f"base GEMM1 form {forms[3]} does not match the selection (zero_fill={sel.fill_in_gemm1})")
    n2, cluster_n = sel.gemm2[0][1], sel.gemm2[1][1]
    g2_form = f"gemm2_dense_finalize_n{n2}" + ("_c12" if cluster_n == 2 else "")
    g2_at = 5 if dual else 4
    if forms[g2_at] != g2_form or sel.gemm2[1][0] != 1:
        raise AssertionError(f"GEMM2 form {forms[g2_at]} != {g2_form}")
    cfg = {"sort": sort_cfg, "gemm1": gemm1, "gemm2": (n2, cluster_n), "clear": decision.clear_output,
           "pdl": decision.pdl, "alt_tile": None, "gemm1_alt": None, "gemm2_alt": None, "pdl_alt": None}
    if dual:
        alt_tile, (g1_alt_tiler, g1_alt_cluster, _), (g2_alt_tiler, g2_alt_cluster, _) = sel.dual
        if alt_tile != 256 or (g1_alt_tiler, g1_alt_cluster) != ((256, 256), (2, 1)) or \
                (g2_alt_tiler, g2_alt_cluster) != ((256, 256), (2, 1)):
            raise AssertionError(f"dual tactic {sel.dual} is not the B300 SiTU dual tactic (256-row 2-CTA pair)")
        gemm1_alt = GEMM1_DENSE_FORMS[forms[4]]
        if gemm1_alt != (256, 256, True, True, False, False) or not sel.fill_in_gemm1:
            raise AssertionError(f"alternate GEMM1 form {forms[4]} is not the 2-CTA zero_fill_secondary form "
                                 f"(fill_in_gemm1={sel.fill_in_gemm1})")
        if forms[6] != "gemm2_dense_finalize_n256_2cta":
            raise AssertionError(f"alternate GEMM2 form {forms[6]} != gemm2_dense_finalize_n256_2cta")
        cfg.update(alt_tile=alt_tile, gemm1_alt=gemm1_alt, gemm2_alt=(256, 2, 1),
                   pdl_alt=bool(decision.pdl or DENSE_DUAL_ALT_PDL))
    return cfg


SPLIT_CHAIN_STEPS = ("routing_split", "gemm1_dense", "gemm2_dense_finalize", "gemm1_swapab_situ",
                     "gemm2_swapab_partial", "finalize_rows")


def split_launch_config(policy: CakeSwapAbPolicy, decision: PlanDecision) -> dict[str, Any]:
    """Host-only resolution of the split two-stage chain's module configurations from the decision's launch plan
    (hand-written ``Mxfp4MoESwapAbPlan`` split form, mxfp4.py:1462-1553 + :1866-1882; enqueue order :2075-2160
    without the side stream, ``SWAP_SPLIT_SIDE_STREAM=0``).

    Nothing is inferred from names: every form symbol of the plan is re-derived from the decision inputs and must
    match (the W1 row-group form of ``SWAP_SPLIT_GEMM1_N_POLICY``, the W11 ``_rg`` form of the dense tactic's GEMM2
    N at cluster (1, 1), the swap-AB partial form ``(tile, kbps, m_group)``, the split routing and finalize forms).
    """
    if decision.path != "split_two_stage":
        raise ValueError(f"not a split two-stage decision (path {decision.path!r})")
    if not decision.supported:
        raise NotImplementedError(decision.reason)
    if not decision.split_dense:
        raise NotImplementedError("split chain without the dense wide GEMM2 (SWAP_SPLIT_DENSE_GEMM2=0) is not enqueued")
    steps = tuple(l.step for l in decision.launch_plan)
    if steps != SPLIT_CHAIN_STEPS:
        raise NotImplementedError(f"split launch sequence {steps} is not the executable chain {SPLIT_CHAIN_STEPS}")
    forms = {l.step: l.form for l in decision.launch_plan}
    T = decision.num_tokens
    if forms["routing_split"] != "kimi_k3_mxfp4_situ_routing_split":
        raise AssertionError(f"routing form {forms['routing_split']}")
    gemm1_n = next(v for limit, v in SWAP_SPLIT_GEMM1_N_POLICY if T <= limit)
    # Both wide launches trigger their programmatic dependents at kernel entry (``pdl_trigger_early = pdl and
    # SWAP_SPLIT_EARLY_TRIGGER``, mxfp4.py:1500 / :1537): the ``_early`` placement forms.
    early = bool(decision.pdl and SWAP_SPLIT_EARLY_TRIGGER)
    g1_sym = gemm1_dense_form_symbol(SWAP_SPLIT_WIDE_TILE, gemm1_n, zero_fill=False, zero_fill_secondary=False,
                                     row_group_list=True, pdl_trigger_early=early)
    if forms["gemm1_dense"] != g1_sym or \
            GEMM1_DENSE_FORMS[g1_sym] != (SWAP_SPLIT_WIDE_TILE, gemm1_n, False, False, True, early):
        raise AssertionError(f"wide GEMM1 form {forms['gemm1_dense']} != {g1_sym}")
    gemm2_tactic = policy.tactic(T)[2]
    n2 = gemm2_tactic[0][1]
    cluster = gemm2_tactic[1] if gemm2_tactic[0] == (SWAP_SPLIT_WIDE_TILE, n2) else (1, 1)
    if cluster != (1, 1):
        raise NotImplementedError(f"split wide GEMM2 with cluster {cluster} (the _c12_rg form is not traced)")
    g2_dense_sym = f"gemm2_dense_finalize_n{n2}_rg" + (DENSE_EARLY_SUFFIX if early else "")
    if forms["gemm2_dense_finalize"] != g2_dense_sym:
        raise AssertionError(f"wide GEMM2 form {forms['gemm2_dense_finalize']} != {g2_dense_sym}")
    g1, g2 = decision.gemm1, decision.gemm2
    # The split chain runs without the dependent-side prefetch (:1149-1151): both narrow forms carry the
    # constructor-default PDL placement, ``swapab_pdl_suffix`` of ``pdl and dep_prefetch`` (= the forms' flags).
    if g1.pdl_trigger_after_wait or g2.late_dep_wait:
        raise AssertionError("the split chain selects the dependent-side prefetch placement")
    narrow_g1_sym = f"gemm1_swapab_situ_n{g1.n_tile}" + ("" if g1.kbps == 4 else f"_k{g1.kbps}") + \
        swapab_pdl_suffix(g1.n_tile, g1.pdl_trigger_after_wait) + swapab_l2_suffix(g1.n_tile, g1.weight_l2_hint)
    if forms["gemm1_swapab_situ"] != narrow_g1_sym:
        raise AssertionError(f"narrow GEMM1 form {forms['gemm1_swapab_situ']} != {narrow_g1_sym}")
    g2_sym = f"gemm2_swapab_partial_n{g2.n_tile}_m{g2.m_group}" + swapab_pdl_suffix(g2.n_tile, g2.late_dep_wait) \
        + swapab_l2_suffix(g2.n_tile, g2.weight_l2_hint)
    if (g2.n_tile, g2.kbps, g2.m_group) not in SWAPAB_PARTIAL_FORMS or forms["gemm2_swapab_partial"] != g2_sym:
        raise AssertionError(f"narrow partial GEMM2 form {forms['gemm2_swapab_partial']} != {g2_sym} for "
                             f"{(g2.n_tile, g2.kbps, g2.m_group)}")
    if forms["finalize_rows"] != "kimi_k3_mxfp4_situ_finalize_rows_split":
        raise AssertionError(f"finalize form {forms['finalize_rows']}")
    if not decision.clear_output:
        raise AssertionError("the split chain's routing launch clears the output (:1182 with split_dense)")
    return {"gemm1_dense": GEMM1_DENSE_FORMS[g1_sym], "gemm2_dense": (n2, 1, early), "gemm1": g1, "gemm2": g2,
            "wide_slots": decision.rows // SWAP_SPLIT_WIDE_TILE, "pdl": decision.pdl, "clear": decision.clear_output}


HYBRID_CHAIN_STEPS = ("route_preprocess", "moe_sort_init", "moe_sort_coop", "dispatch", "gemm1_swapab_situ",
                      "gemm1_dense", "gemm2_dense_finalize")


def hybrid_launch_config(policy: CakeSwapAbPolicy, decision: PlanDecision) -> dict[str, Any]:
    """Host-only resolution of the hybrid chain's module configurations from the decision's launch plan (hand-written
    ``Mxfp4MoESwapAbPlan`` hybrid form on the MoE-TP shard, 1024 < T <= 2048: generic routing :1251-1313, dispatch
    :1324-1345, swap-AB SiTU GEMM1 over the narrow list :1346-1351 / :1430-1461, dense GEMM1 over the wide list
    :1554-1629, dense finalize GEMM2 over every 128-row group :1686-1735; enqueue order ``run`` :1992-2160).

    Nothing is inferred from names: the K6 configuration (tier, bounded state, PDL) is rebuilt from the decision inputs
    and must reproduce the plan's form names, the dense GEMM forms come from the pinned tables, the swap-AB GEMM1 is
    ``decision.gemm1`` with the row-group list and blocked scales, the dispatch constants are the hand-written ones
    (``group_rows`` 128, ``narrow_tile`` = the swap tile, ``wide_min_rows = min(SWAP_HYBRID_DENSE_MIN_ROWS, group_rows)``,
    ``wide_min_permille`` 0, no ``all`` list). Every kernel of the chain launches with the plan's PDL attribute
    (``decision.pdl`` = ``enable_pdl or SWAP_PDL``, :1148; the hand-written hybrid launches all take ``enable_pdl=pdl``).
    """
    if decision.path != "hybrid":
        raise ValueError(f"not a hybrid decision (path {decision.path!r})")
    if not decision.supported:
        raise NotImplementedError(decision.reason)
    steps = tuple(l.step for l in decision.launch_plan)
    if steps != HYBRID_CHAIN_STEPS:
        raise NotImplementedError(f"hybrid launch sequence {steps} is not the executable chain {HYBRID_CHAIN_STEPS}")
    forms = {l.step: l.form for l in decision.launch_plan}
    T, group = decision.num_tokens, decision.group_rows
    if group != SWAP_HYBRID_GROUP_ROWS or decision.fused_routing or decision.two_stage or decision.token_index:
        raise AssertionError("hybrid chain: 128-row sort groups, generic routing, fused finalize, no token index (:1324-1461)")
    ms = _moe_sort_module()
    tier = moe_sort_expert_tier(policy.layout.num_experts)
    sort_cfg = ms.MoeSortConfig(tier=tier, dual=False, mixed=False, narrow_count_base=False, pdl=decision.pdl,
                                bounded=moe_sort_bounded_state(T, policy.top_k, tier, policy.sm_count))
    names = (ms.init_form_name(sort_cfg), ms.coop_form_name(sort_cfg))
    if names != (forms["moe_sort_init"], forms["moe_sort_coop"]):
        raise AssertionError(f"K6 configuration {names} does not reproduce the plan's forms")
    if forms["route_preprocess"] != "kimi_k3_mxfp4_situ_route_preprocess" or forms["dispatch"] != "kimi_k3_mxfp4_situ_dispatch":
        raise AssertionError(f"routing / dispatch forms {forms['route_preprocess']} / {forms['dispatch']}")
    g1 = decision.gemm1
    if g1 is None or not (g1.row_group_list and g1.sf_blocked) or g1.two_cta or g1.n_tile != decision.tile:
        raise AssertionError("hybrid swap-AB GEMM1 is the single-CTA SiTU form over the row-group list with blocked scales")
    g1_sym = f"gemm1_swapab_situ_n{g1.n_tile}" + ("" if g1.kbps == 4 else f"_k{g1.kbps}") + "_rowgroup" \
        + swapab_l2_suffix(g1.n_tile, g1.weight_l2_hint)
    if forms["gemm1_swapab_situ"] != g1_sym:
        raise AssertionError(f"swap-AB GEMM1 form {forms['gemm1_swapab_situ']} != {g1_sym}")
    g1_tactic = policy.tactic(T)[1]
    dense_sym = gemm1_dense_form_symbol(g1_tactic[0][0], g1_tactic[0][1], zero_fill=False, zero_fill_secondary=False,
                                        row_group_list=True)
    if forms["gemm1_dense"] != dense_sym or GEMM1_DENSE_FORMS[dense_sym] != (group, g1_tactic[0][1], False, False, True, False) \
            or tuple(g1_tactic[1]) != (1, 1):
        raise AssertionError(f"wide GEMM1 form {forms['gemm1_dense']} != {dense_sym} (tactic {g1_tactic})")
    g2_tactic = policy.tactic(T)[2]
    n2, cluster_n = g2_tactic[0][1], g2_tactic[1][1]
    if g2_tactic[0][0] != group or g2_tactic[1][0] != 1:
        raise AssertionError(f"hybrid GEMM2 tactic tile must match the {group}-row sort groups, got {g2_tactic!r}")   # :1690-1694
    g2_sym = f"gemm2_dense_finalize_n{n2}" + ("_c12" if cluster_n == 2 else "")
    if forms["gemm2_dense_finalize"] != g2_sym:
        raise AssertionError(f"GEMM2 form {forms['gemm2_dense_finalize']} != {g2_sym}")
    if not decision.clear_output:
        raise AssertionError("the hybrid chain's route preprocess clears the output (:1182, zero_fill 'route')")
    return {"sort": sort_cfg, "dispatch": (group, decision.tile, min(SWAP_HYBRID_DENSE_MIN_ROWS, group), 0),
            # The hybrid dense launches trigger in the footer (mxfp4.py:1594 / :1659 pass ``enable_pdl`` only).
            "gemm1": g1, "gemm1_dense": GEMM1_DENSE_FORMS[dense_sym], "gemm2_dense": (n2, cluster_n, False),
            "clear": decision.clear_output, "pdl": decision.pdl}


@dataclass(frozen=True)
class ExecutedLaunch:
    """One kernel the Cake-tree runner actually enqueues: plan step, IR form symbol, lowering backend, grid."""

    step: str
    form: str
    backend: str
    grid: tuple[int, int, int]


def executed_chain_record(chain: str, decision: PlanDecision, launches: tuple[ExecutedLaunch, ...]) -> dict[str, Any]:
    """The ``executed_chain`` label of one planned row: the chain the runner enqueued (``chain``), the chain the
    hand-written decision table selects (``decided_path``), the per-kernel backends, and whether they agree."""
    backends = sorted({l.backend for l in launches})
    return {"chain": chain, "decided_path": decision.path, "executes_decided_chain": chain == decision.path,
            "launches": [{"step": l.step, "form": l.form, "backend": l.backend, "grid": list(l.grid)} for l in launches],
            "backends": backends, "mixed_backend": len(backends) > 1}


def _route_operands(b: dict[str, Any], topk_ids, topk_weights) -> tuple[str, Any, tuple[int, int], Any, Any]:
    """Routing operands (mxfp4.py:1128-1136): ``(mode, weights_src, w_strides, route_weights, route_ids)``.

    Packed IDs unpack into ``route_ids`` and their BF16 halves convert into ``route_weights``; separate BF16 weights
    convert into ``route_weights``; separate FP32 weights and int32 IDs bind directly."""
    import torch

    if topk_weights is None:
        return ("packed", topk_ids.view(torch.bfloat16), (2 * topk_ids.stride(0), 2 * topk_ids.stride(1)),
                b["route_weights"], b["route_ids"])
    if topk_weights.dtype == torch.float32:
        return "separate_fp32", topk_weights, tuple(topk_weights.stride()), topk_weights, topk_ids
    return "separate_bf16", topk_weights, tuple(topk_weights.stride()), b["route_weights"], topk_ids


class CakeSwapAbPlan:
    """Fixed-address executable: ``run()`` enqueues routing -> GEMM1 -> GEMM2 on the caller's current stream.

    Same contract as the hand-written plan (:996-1008): caller-owned output and workspace, prepared outside
    CUDA-graph capture (the constructor runs the chain once as the warmup / validity postcondition, :1214),
    ``run`` performs no allocation, JIT or host synchronisation and is capturable.
    """

    def __init__(self, *, wrapper: CakeMxfp4MoEWrapper, decision: PlanDecision, buffers: dict[str, Any],
                 workspace, x, x_sf, topk_ids, topk_weights, weights: dict[str, Any], beta, linear_beta, output):
        import torch

        self._wrapper = wrapper
        self.decision = decision
        self.backend = wrapper.backend
        self.workspace = workspace
        self.output = output
        self.device = output.device
        self.n_tile = decision.tile
        self.finalize = True
        self.group_rows = decision.group_rows
        self._buffers = b = buffers
        self._inputs = (x, x_sf, topk_ids, topk_weights)
        self._weights = weights
        T = x.shape[0]
        pol = wrapper.policy
        self.mode, weights_src, w_strides, self.route_weights, self.route_ids = _route_operands(b, topk_ids, topk_weights)
        self.expanded_idx_to_permuted_idx = b["out_expanded_idx_to_permuted_idx"]
        # Compile (or fetch the cached) kernel modules of the three launches (plain routing layout: this plan is
        # also what a split decision runs while SPLIT_CHAIN_EXECUTABLE is off).
        self.routing_config = cfg = routing_config(pol, decision, mode=self.mode, split_layout=False)
        self._routing = build_routing_module(self.backend, cfg)
        self._gemm1 = build_gemm1_module(self.backend, decision.gemm1)
        self._gemm2 = build_gemm2_module(self.backend, decision.gemm2)
        rt = _routing_module()
        gm = _gemm_module()
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        H, I, K, L = pol.hidden_size, pol.intermediate_shard, pol.top_k, pol.num_local_experts
        output_words = T * H // 2
        self._routing_grid = (rt.launch_grid(cfg, output_words), 1, 1)
        # Kernel ABI operands the plain chain never reads (the kernels carry them for the dispatch-list, split
        # and mixed192 forms): the routing lists / wide slots and the GEMM row-group list. Same convention as the
        # modules' own launchers (routing ``allocate_buffers`` / ``run_cake_routing``, GEMM ``make_problem``).
        self._unused = unused_launch_operands(self.device, group_capacity=decision.tiles)
        self._routing_args = routing_launch_bindings(
            b, topk_ids=topk_ids, weights_src=weights_src, w_strides=w_strides, output=output,
            num_tokens=T, top_k=K, num_experts=pol.layout.num_experts, local_experts=L,
            local_offset=pol.layout.local_expert_offset, group_rows=decision.group_rows, unused=self._unused)
        g1, g2 = decision.gemm1, decision.gemm2
        tiles = decision.tiles
        # dbg words are written only by DEBUG_PROGRESS builds; the pointer must exist.
        self._dbg = torch.zeros(64 * sms, dtype=torch.int32, device=self.device)
        self._gemm1_grid = (min((2 * I // 128) * tiles, sms), 1, 1)
        self._gemm1_args = gemm1_launch_bindings(
            b, weights=weights, x=x, x_sf=x_sf, route_weights=self.route_weights, hidden=H, shard=I, top_k=K,
            num_tokens=T, tiles=tiles, k_tiles=g1.k_tiles, beta=beta, linear_beta=linear_beta, dbg=self._dbg,
            unused=self._unused)
        m_chunks = gm.m_chunks_of(H // 128, g2.m_group)
        self._gemm2_grid = (min(m_chunks * tiles, sms), 1, 1)
        self._gemm2_args = gemm2_launch_bindings(
            b, weights=weights, route_weights=self.route_weights, hidden=H, shard=I, top_k=K, tiles=tiles,
            k_tiles=g2.k_tiles, output=output, dbg=self._dbg, unused=self._unused)
        self._beta = beta
        self._linear_beta = linear_beta
        self.launches = (ExecutedLaunch("routing_fused", "kimi_k3_mxfp4_situ_routing", self.backend, self._routing_grid),
                         ExecutedLaunch("gemm1_swapab_situ", f"gemm1_swapab_situ_n{g1.n_tile}" + ("" if g1.kbps == 4 else f"_k{g1.kbps}")
                                        + swapab_l2_suffix(g1.n_tile, g1.weight_l2_hint),
                                        self.backend, self._gemm1_grid),
                         ExecutedLaunch("gemm2_swapab_finalize", f"gemm2_swapab_finalize_n{g2.n_tile}"
                                        + ("" if g2.kbps == 4 else f"_k{g2.kbps}") + ("" if g2.m_group == 1 else f"_m{g2.m_group}")
                                        + swapab_l2_suffix(g2.n_tile, g2.weight_l2_hint),
                                        self.backend, self._gemm2_grid))
        # Warmup run = the hand-written planning postcondition (valid output after ``plan``, :1214 / :1216-1218).
        with torch.cuda.device(self.device):
            self.run()
            torch.cuda.synchronize()

    @property
    def executed_chain(self) -> dict[str, Any]:
        return executed_chain_record("plain", self.decision, self.launches)

    def launch_sequence(self) -> tuple[tuple[str, tuple[int, int, int]], ...]:
        return (("routing_fused", self._routing_grid), ("gemm1_swapab_situ", self._gemm1_grid),
                ("gemm2_swapab_finalize", self._gemm2_grid))

    def run(self):
        """Enqueue the three launches on the caller's current stream and return the bound output (:1992-2160
        plain chain: ``_route_preprocess.run`` :2000, ``_gemm1`` :2137, ``_gemm2`` :2159)."""
        import torch

        with torch.cuda.device(self.device):
            self._routing.launch(grid=self._routing_grid, **self._routing_args)
            self._gemm1.launch(grid=self._gemm1_grid, **self._gemm1_args)
            self._gemm2.launch(grid=self._gemm2_grid, **self._gemm2_args)
        return self.output


class CakeDensePlan:
    """The executable dense chain of one planned problem (hand-written ``Mxfp4MoEPlan.run`` / ``_moe_core_impl``,
    mxfp4.py:912-995): K1 ``_RoutePreprocess`` (conversion + output clear), K6 ``moe_sort`` init + cooperative
    kernel, the W1 dense gather GEMM1 (SiTU, blocked output scales), the W11 dense finalize GEMM2. Dual-tile rows
    (``dense_launch_config(...)["alt_tile"]``, T > 7168 on a shard <= 3072) add the K6 256-row alternate padding and
    the 2-CTA alternate GEMM1 / GEMM2 launches over the ``out_alt_*`` lists, enqueued after their base launch
    (gather, gather_alt, finalize, finalize_alt; mxfp4.py:967-980); the base launches read
    ``out_base_active_num_non_exiting_tiles`` and the pair member the routing did not choose exits on a zero tile
    count. Same buffer and operand conventions as :class:`CakeSwapAbPlan`; the dense workspace fields of
    ``_dense_workspace_fields``."""

    def __init__(self, *, wrapper: CakeMxfp4MoEWrapper, decision: PlanDecision, buffers: dict[str, Any],
                 workspace, x, x_sf, topk_ids, topk_weights, weights: dict[str, Any], beta, linear_beta, output):
        import torch

        self.backend = wrapper.backend
        self.decision = decision
        self.workspace = workspace
        self.output = output
        self.device = output.device
        self.n_tile = decision.tile
        self.finalize = True
        self.group_rows = decision.group_rows
        self._buffers = b = buffers
        self._inputs = (x, x_sf, topk_ids, topk_weights)
        self._weights = weights
        T = x.shape[0]
        pol = wrapper.policy
        H, I, K, L = pol.hidden_size, pol.intermediate_shard, pol.top_k, pol.num_local_experts
        E, offset = pol.layout.num_experts, pol.layout.local_expert_offset
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        if sms != pol.sm_count:
            raise RuntimeError(f"the plan was decided for {pol.sm_count} SMs (K6 tier state / grid); the device has {sms}")
        cfg = dense_launch_config(pol, decision)
        self.mode, weights_src, w_strides, self.route_weights, self.route_ids = _route_operands(b, topk_ids, topk_weights)
        self.expanded_idx_to_permuted_idx = b["out_expanded_idx_to_permuted_idx"]
        rt, ms, g1d, g2d = _routing_module(), _moe_sort_module(), _gemm1_dense_module(), _gemm2_dense_module()
        # K1: conversion + (below 8192 tokens) the output clear (mxfp4.py:912-935).
        pre_cfg = rt.PreprocessConfig(mode=self.mode, threads=ROUTE_PREPROCESS_THREADS, clear=cfg["clear"])
        self._pre = rt.build_preprocess_module(pre_cfg, self.backend)
        self._pre_grid = (rt.preprocess_grid(pre_cfg, T, K, H), 1, 1)
        self._pre_args = preprocess_launch_bindings(b, topk_ids=topk_ids, weights_src=weights_src, w_strides=w_strides,
                                                    output=output, num_tokens=T, top_k=K, hidden=H)
        # K6: init + cooperative kernel (mxfp4.py:1269-1313), the tier-896 bounded state of this family.
        sort_cfg = cfg["sort"]
        self._sort_init = ms.build_init_module(sort_cfg)
        self._sort_coop = ms.build_coop_module(sort_cfg)
        self._sort_init_grid = (ms.init_grid(sort_cfg, E), 1, 1)
        self._sort_coop_grid = (ms.coop_grid(sms), 1, 1)
        self._unused = unused_launch_operands(self.device, group_capacity=decision.tiles)
        alt_tile = cfg["alt_tile"]
        dual = alt_tile is not None
        self._sort_init_args, self._sort_coop_args = moe_sort_launch_bindings(
            b, topk_ids=self.route_ids, num_tokens=T, top_k=K, num_experts=E, local_experts=L, local_offset=offset,
            tile=decision.dense.tile, alt_tile=alt_tile, permille=DENSE_DUAL_TILE_THRESHOLD_PERMILLE, mixed=False,
            narrow_tile=0, unused=self._unused)
        # Dense GEMM operands: row-major packed weights + the shared 512-byte scale atoms.
        dense_w = dense_weight_operands(weights, local_experts=L, hidden=H, shard=I)
        self._dbg = torch.zeros(64 * sms, dtype=torch.int32, device=self.device)
        g1_weights = {"w1": dense_w["w1"], "w1_sf": dense_w["w1_sf"]}
        # W1 dense gather GEMM1 (fused_moe.py:481-541): persistent grid over (row tiles x N tiles) up to the SMs.
        tile_m, n1, zero_fill, secondary, row_group, early = cfg["gemm1"]
        self._gemm1 = g1d.build_module(self.backend, tile_m, n1, zero_fill, secondary, row_group, use_pdl=cfg["pdl"],
                                       pdl_trigger_early=early)
        self._gemm1_grid = (min((decision.rows // tile_m) * (2 * I // n1), sms), 1, 1)
        self._gemm1_args = gemm1_dense_launch_bindings(
            b, weights=g1_weights, x=x, x_sf=x_sf, hidden=H, shard=I, top_k=K,
            num_tokens=T, rows=decision.rows, tile_m=tile_m, n_tile=n1, row_group_list=row_group, zero_fill=zero_fill,
            secondary=secondary, output=output, beta=beta, linear_beta=linear_beta, beta_stride=1,
            linear_beta_stride=1, unused=self._unused, dual=dual)
        # Dual-tile alternate GEMM1 (fused_moe.py:546-595): the 2-CTA (256, 256) / (2, 1) ``zero_fill_secondary``
        # form over the ``out_alt_*`` lists; one cluster per work item up to the co-resident cluster capacity
        # (hand-written ``max_active_clusters``; ``gemm1_dense.launch_grid``).
        self._gemm1_alt = self._gemm1_alt_grid = self._gemm1_alt_args = None
        self._gemm2_alt = self._gemm2_alt_grid = self._gemm2_alt_args = None
        if dual:
            alt_m, alt_n1, alt_zf, alt_sec, alt_rg, alt_early = cfg["gemm1_alt"]
            alt_cap = decision.rows // alt_tile
            self._gemm1_alt = g1d.build_module(self.backend, alt_m, alt_n1, alt_zf, alt_sec, alt_rg, use_pdl=cfg["pdl_alt"],
                                               pdl_trigger_early=alt_early)
            self._gemm1_alt_grid = (2 * min(alt_cap * (2 * I // alt_n1), self._gemm1_alt.max_active_clusters()), 1, 1)
            self._gemm1_alt_args = gemm1_dense_launch_bindings(
                b, weights=g1_weights, x=x, x_sf=x_sf, hidden=H, shard=I, top_k=K,
                num_tokens=T, rows=decision.rows, tile_m=alt_m, n_tile=alt_n1, row_group_list=alt_rg, zero_fill=alt_zf,
                secondary=alt_sec, output=output, beta=beta, linear_beta=linear_beta, beta_stride=1,
                linear_beta_stride=1, unused=self._unused, dual=True)
        # W11 dense finalize GEMM2 (fused_moe.py:697-736): one CTA (cluster) per raster unit up to the SMs.
        n2, cluster_n = cfg["gemm2"]
        self._gemm2 = g2d.build_module(self.backend, n2, use_pdl=cfg["pdl"], cta_group=1, cluster_n=cluster_n)
        n2_tiles = -(-H // n2)
        self._gemm2_grid = (cluster_n * min(decision.tiles * (-(-n2_tiles // cluster_n)), sms // cluster_n), 1, 1)
        self._gemm2_args = gemm2_dense_launch_bindings(
            b, w2=dense_w["w2"], w2_sf=dense_w["w2_sf"], route_weights=self.route_weights, hidden=H, shard=I, top_k=K,
            tiles=decision.tiles, rows=decision.rows, n_tile=n2, output=output, dbg=self._dbg, unused=self._unused,
            dual=dual)
        if dual:
            # Dual-tile alternate GEMM2 (fused_moe.py:684-717): the (256, 256) / (2, 1) cta_group::2 form over the
            # ``out_alt_*`` 256-row groups; one CTA pair per raster unit up to SMs // 2 (``gemm2_dense.grid_for``).
            alt_n2, alt_cg, alt_cn = cfg["gemm2_alt"]
            alt_cap = decision.rows // alt_tile
            self._gemm2_alt = g2d.build_module(self.backend, alt_n2, use_pdl=cfg["pdl_alt"], cta_group=alt_cg,
                                               cluster_n=alt_cn)
            pair = alt_cg * alt_cn
            self._gemm2_alt_grid = (pair * min(alt_cap * (-(-H // alt_n2)), sms // pair), 1, 1)
            self._gemm2_alt_args = gemm2_dense_launch_bindings(
                b, w2=dense_w["w2"], w2_sf=dense_w["w2_sf"], route_weights=self.route_weights, hidden=H, shard=I,
                top_k=K, tiles=alt_cap, rows=decision.rows, n_tile=alt_n2, output=output, dbg=self._dbg,
                unused=self._unused, dual=True, secondary=True)
        self._dense_weights = dense_w
        self._beta = beta
        self._linear_beta = linear_beta
        forms = [l.form for l in decision.launch_plan]
        launches = [ExecutedLaunch("route_preprocess", forms[0], self.backend, self._pre_grid),
                    ExecutedLaunch("moe_sort_init", forms[1], "cuda_cpp", self._sort_init_grid),
                    ExecutedLaunch("moe_sort_coop", forms[2], "cuda_cpp", self._sort_coop_grid),
                    ExecutedLaunch("gemm1_dense", forms[3], self.backend, self._gemm1_grid)]
        if dual:
            launches += [ExecutedLaunch("gemm1_dense", forms[4], self.backend, self._gemm1_alt_grid),
                         ExecutedLaunch("gemm2_dense_finalize", forms[5], self.backend, self._gemm2_grid),
                         ExecutedLaunch("gemm2_dense_finalize", forms[6], self.backend, self._gemm2_alt_grid)]
        else:
            launches.append(ExecutedLaunch("gemm2_dense_finalize", forms[4], self.backend, self._gemm2_grid))
        self.launches = tuple(launches)
        # Warmup run = the hand-written planning postcondition (valid output after ``plan``, :1214 / :1216-1218).
        with torch.cuda.device(self.device):
            self.run()
            torch.cuda.synchronize()

    @property
    def executed_chain(self) -> dict[str, Any]:
        return executed_chain_record("dense", self.decision, self.launches)

    @property
    def dual(self) -> bool:
        """True when the chain is the dual-tile pair (alternate GEMM1 / GEMM2 launches enqueued)."""
        return self._gemm1_alt is not None

    def launch_sequence(self) -> tuple[tuple[str, tuple[int, int, int]], ...]:
        return tuple((l.step, l.grid) for l in self.launches)

    def run(self):
        """Enqueue the launches on the caller's current stream and return the bound output (mxfp4.py:912-995 dense
        ``run``: preprocess :935, ``moe_sort`` :1269-1313, GEMM1 :481-541 then the alternate :546-595, GEMM2
        :697-736 then the alternate :684-717)."""
        import torch

        with torch.cuda.device(self.device):
            self._pre.launch(grid=self._pre_grid, **self._pre_args)
            self._sort_init.launch(grid=self._sort_init_grid, **self._sort_init_args)
            self._sort_coop.launch(grid=self._sort_coop_grid, **self._sort_coop_args)
            self._gemm1.launch(grid=self._gemm1_grid, **self._gemm1_args)
            if self._gemm1_alt is not None:
                self._gemm1_alt.launch(grid=self._gemm1_alt_grid, **self._gemm1_alt_args)
            self._gemm2.launch(grid=self._gemm2_grid, **self._gemm2_args)
            if self._gemm2_alt is not None:
                self._gemm2_alt.launch(grid=self._gemm2_alt_grid, **self._gemm2_alt_args)
        return self.output


class CakeSplitPlan:
    """The executable split two-stage chain of one planned problem (hand-written ``Mxfp4MoESwapAbPlan`` with
    ``split`` and ``two_stage``, mxfp4.py:1462-1553, :1866-1882, run :2075-2160 without the side stream): the
    split-layout fused routing (clears the output), the W1 row-group gather GEMM1 and the W11 row-group finalize
    GEMM2 over the ``SWAP_SPLIT_WIDE_TILE``-row groups of the wide experts (reduce-add into the output), the swap-AB
    SiTU GEMM1 and the swap-AB partial GEMM2 over the narrow groups (``partial_rows``), and the K7 ``finalize_rows``
    accumulate (narrow rows added onto the wide GEMM2's partial combine). Same buffer / operand conventions as
    :class:`CakeSwapAbPlan`; the shared ``gemm1_out`` / ``gemm1_out_scale`` hold the wide groups' rows in the
    blocked scale layout and the narrow groups' rows in the plain row-scale layout (disjoint 128-row blocks)."""

    def __init__(self, *, wrapper: CakeMxfp4MoEWrapper, decision: PlanDecision, buffers: dict[str, Any],
                 workspace, x, x_sf, topk_ids, topk_weights, weights: dict[str, Any], beta, linear_beta, output):
        import torch

        self._wrapper = wrapper
        self.decision = decision
        self.backend = wrapper.backend
        self.workspace = workspace
        self.output = output
        self.device = output.device
        self.n_tile = decision.tile
        self.finalize = True
        self.group_rows = decision.group_rows
        self._buffers = b = buffers
        self._inputs = (x, x_sf, topk_ids, topk_weights)
        self._weights = weights
        T = x.shape[0]
        pol = wrapper.policy
        cfg = split_launch_config(pol, decision)
        self.mode, weights_src, w_strides, self.route_weights, self.route_ids = _route_operands(b, topk_ids, topk_weights)
        self.expanded_idx_to_permuted_idx = b["out_expanded_idx_to_permuted_idx"]
        rt, gm, g1d, g2d, fin = _routing_module(), _gemm_module(), _gemm1_dense_module(), _gemm2_dense_module(), _finalize_module()
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        H, I, K, L = pol.hidden_size, pol.intermediate_shard, pol.top_k, pol.num_local_experts
        tiles, rows, wide_slots = decision.tiles, decision.rows, cfg["wide_slots"]
        self._unused = unused_launch_operands(self.device, group_capacity=tiles)
        self._dbg = torch.zeros(64 * sms, dtype=torch.int32, device=self.device)
        # Routing (split layout, clears the output): :1201-1250 with ``split_layout=dict(...)``.
        self.routing_config = rcfg = routing_config(pol, decision, mode=self.mode, split_layout=True)
        self._routing = build_routing_module(self.backend, rcfg)
        self._routing_grid = (rt.launch_grid(rcfg, T * H // 2), 1, 1)
        self._routing_args = routing_launch_bindings(
            b, topk_ids=topk_ids, weights_src=weights_src, w_strides=w_strides, output=output, num_tokens=T, top_k=K,
            num_experts=pol.layout.num_experts, local_experts=L, local_offset=pol.layout.local_expert_offset,
            group_rows=decision.group_rows, unused=self._unused, split_layout=True)
        # Wide GEMM1: W1 row-group form over ``swap_wide_list`` (:1473-1503); grid one CTA per list slot x N tile.
        dense_w = dense_weight_operands(weights, local_experts=L, hidden=H, shard=I)
        tile_m, n1, zf, zfs, rg, early = cfg["gemm1_dense"]
        self._gemm1_dense = g1d.build_module(self.backend, tile_m, n1, zf, zfs, rg, use_pdl=cfg["pdl"],
                                             pdl_trigger_early=early)
        self._gemm1_dense_grid = (min(wide_slots * (2 * I // n1), sms), 1, 1)
        self._gemm1_dense_args = gemm1_dense_launch_bindings(
            b, weights={"w1": dense_w["w1"], "w1_sf": dense_w["w1_sf"]}, x=x, x_sf=x_sf, hidden=H, shard=I, top_k=K,
            num_tokens=T, rows=rows, tile_m=tile_m, n_tile=n1, row_group_list=True, zero_fill=False, secondary=False,
            output=output, beta=beta, linear_beta=linear_beta, beta_stride=1, linear_beta_stride=1, unused=self._unused,
            split_layout=True)
        # Wide GEMM2: W11 ``_rg`` finalize form over the same list (:1506-1530), reduce-add into the cleared output.
        n2, cluster_n, g2_early = cfg["gemm2_dense"]
        self._gemm2_dense = g2d.build_module(self.backend, n2, use_pdl=cfg["pdl"], cta_group=1, cluster_n=cluster_n,
                                             row_group=True, pdl_trigger_early=g2_early)
        self._gemm2_dense_grid = (min(wide_slots * (-(-H // n2)), sms), 1, 1)
        self._gemm2_dense_args = gemm2_dense_launch_bindings(
            b, w2=dense_w["w2"], w2_sf=dense_w["w2_sf"], route_weights=self.route_weights, hidden=H, shard=I, top_k=K,
            tiles=wide_slots, rows=rows, n_tile=n2, output=output, dbg=self._dbg, unused=self._unused, row_group=True,
            split_layout=True)
        # Narrow swap-AB GEMM1 SiTU (:1596-1640) and partial GEMM2 (:1957-1990, ``out = partial_rows``).
        g1, g2 = cfg["gemm1"], cfg["gemm2"]
        self._gemm1 = build_gemm1_module(self.backend, g1)
        self._gemm1_grid = (min((2 * I // 128) * tiles, sms), 1, 1)
        self._gemm1_args = gemm1_launch_bindings(
            b, weights=weights, x=x, x_sf=x_sf, route_weights=self.route_weights, hidden=H, shard=I, top_k=K,
            num_tokens=T, tiles=tiles, k_tiles=g1.k_tiles, beta=beta, linear_beta=linear_beta, dbg=self._dbg,
            unused=self._unused)
        self._gemm2 = build_gemm2_partial_module(self.backend, g2)
        self._gemm2_grid = (min(gm.m_chunks_of(H // 128, g2.m_group) * tiles, sms), 1, 1)
        self._gemm2_args = gemm2_launch_bindings(
            b, weights=weights, route_weights=self.route_weights, hidden=H, shard=I, top_k=K, tiles=tiles,
            k_tiles=g2.k_tiles, output=b["partial_rows"], dbg=self._dbg, unused=self._unused)
        # K7 finalize_rows accumulate (:1866-1882).
        fcfg = fin.FinalizeConfig(top_k=K, threads=FINALIZE_ROWS_THREADS, expanded_rows=False, accumulate=True,
                                  skip_wide=True)
        self._finalize = fin.build_finalize_module(fcfg, self.backend)
        self._finalize_grid = (fin.finalize_grid(fcfg, T, H), 1, 1)
        self._finalize_args = finalize_rows_launch_bindings(b, route_weights=self.route_weights, output=output,
                                                            num_tokens=T, hidden=H, narrow_tile=decision.tile)
        self._dense_weights = dense_w
        self._beta = beta
        self._linear_beta = linear_beta
        forms = {l.step: l.form for l in decision.launch_plan}
        self.launches = tuple(ExecutedLaunch(step, forms[step], self.backend, grid) for step, grid in (
            ("routing_split", self._routing_grid), ("gemm1_dense", self._gemm1_dense_grid),
            ("gemm2_dense_finalize", self._gemm2_dense_grid), ("gemm1_swapab_situ", self._gemm1_grid),
            ("gemm2_swapab_partial", self._gemm2_grid), ("finalize_rows", self._finalize_grid)))
        # Warmup run = the hand-written planning postcondition (valid output after ``plan``, :1214 / :1216-1218).
        with torch.cuda.device(self.device):
            self.run()
            torch.cuda.synchronize()

    @property
    def executed_chain(self) -> dict[str, Any]:
        return executed_chain_record("split_two_stage", self.decision, self.launches)

    def launch_sequence(self) -> tuple[tuple[str, tuple[int, int, int]], ...]:
        return tuple((l.step, l.grid) for l in self.launches)

    def run(self):
        """Enqueue the six launches on the caller's current stream in the hand-written order (:2075-2160: routing,
        wide GEMM1, wide GEMM2, swap GEMM1, swap GEMM2 partial, finalize_rows) and return the bound output."""
        import torch

        with torch.cuda.device(self.device):
            self._routing.launch(grid=self._routing_grid, **self._routing_args)
            self._gemm1_dense.launch(grid=self._gemm1_dense_grid, **self._gemm1_dense_args)
            self._gemm2_dense.launch(grid=self._gemm2_dense_grid, **self._gemm2_dense_args)
            self._gemm1.launch(grid=self._gemm1_grid, **self._gemm1_args)
            self._gemm2.launch(grid=self._gemm2_grid, **self._gemm2_args)
            self._finalize.launch(grid=self._finalize_grid, **self._finalize_args)
        return self.output


class CakeHybridPlan:
    """The executable hybrid chain of one planned problem (hand-written ``Mxfp4MoESwapAbPlan`` with ``hybrid``,
    MoE-TP shard 1024 < T <= 2048, mxfp4.py:1324-1461 / :1554-1629 / :1686-1735, run :1992-2160): K1 route preprocess
    (conversion + output clear), the K6 ``moe_sort`` pair over 128-row groups, the K3 ``swapab_dispatch`` wide /
    narrow lists, the swap-AB SiTU GEMM1 ``_rowgroup`` form over the narrow 64-row sub-tiles, the W1 dense gather
    GEMM1 over the wide list (both write E4M3 rows + blocked UE8M0 scales into the shared ``gemm1_out`` /
    ``gemm1_out_scale``), and the W11 dense finalize GEMM2 over every valid group (bulk reduce-add into the cleared
    output). Same buffer / operand conventions as :class:`CakeDensePlan`; no side stream (the hand-written hybrid
    chain is single-stream too)."""

    def __init__(self, *, wrapper: CakeMxfp4MoEWrapper, decision: PlanDecision, buffers: dict[str, Any],
                 workspace, x, x_sf, topk_ids, topk_weights, weights: dict[str, Any], beta, linear_beta, output):
        import torch

        self._wrapper = wrapper
        self.backend = wrapper.backend
        self.decision = decision
        self.workspace = workspace
        self.output = output
        self.device = output.device
        self.n_tile = decision.tile
        self.finalize = True
        self.group_rows = decision.group_rows
        self._buffers = b = buffers
        self._inputs = (x, x_sf, topk_ids, topk_weights)
        self._weights = weights
        T = x.shape[0]
        pol = wrapper.policy
        H, I, K, L = pol.hidden_size, pol.intermediate_shard, pol.top_k, pol.num_local_experts
        E, offset = pol.layout.num_experts, pol.layout.local_expert_offset
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        if sms != pol.sm_count:
            raise RuntimeError(f"the plan was decided for {pol.sm_count} SMs (K6 tier state / grid); the device has {sms}")
        cfg = hybrid_launch_config(pol, decision)
        self.mode, weights_src, w_strides, self.route_weights, self.route_ids = _route_operands(b, topk_ids, topk_weights)
        self.expanded_idx_to_permuted_idx = b["out_expanded_idx_to_permuted_idx"]
        rt, ms, dp, gm = _routing_module(), _moe_sort_module(), _dispatch_module(), _gemm_module()
        g1d, g2d = _gemm1_dense_module(), _gemm2_dense_module()
        tiles, rows = decision.tiles, decision.rows
        # K1: conversion + output clear (:1251-1261; hybrid: zero_fill "route", clear_output True).
        pre_cfg = rt.PreprocessConfig(mode=self.mode, threads=ROUTE_PREPROCESS_THREADS, clear=cfg["clear"])
        self._pre = rt.build_preprocess_module(pre_cfg, self.backend)
        self._pre_grid = (rt.preprocess_grid(pre_cfg, T, K, H), 1, 1)
        self._pre_args = preprocess_launch_bindings(b, topk_ids=topk_ids, weights_src=weights_src, w_strides=w_strides,
                                                    output=output, num_tokens=T, top_k=K, hidden=H)
        # K6: init + cooperative kernel over 128-row groups (:1269-1313; no dual tile, no mixed lists).
        sort_cfg = cfg["sort"]
        self._sort_init = ms.build_init_module(sort_cfg)
        self._sort_coop = ms.build_coop_module(sort_cfg)
        self._sort_init_grid = (ms.init_grid(sort_cfg, E), 1, 1)
        self._sort_coop_grid = (ms.coop_grid(sms), 1, 1)
        self._unused = unused_launch_operands(self.device, group_capacity=tiles)
        self._sort_init_args, self._sort_coop_args = moe_sort_launch_bindings(
            b, topk_ids=self.route_ids, num_tokens=T, top_k=K, num_experts=E, local_experts=L, local_offset=offset,
            tile=decision.group_rows, alt_tile=None, permille=DENSE_DUAL_TILE_THRESHOLD_PERMILLE, mixed=False,
            narrow_tile=0, unused=self._unused)
        # K3: wide / narrow work lists over the sort groups (:1324-1345, ``all_list=None``); one CTA.
        group, narrow_tile, wide_min_rows, wide_min_permille = cfg["dispatch"]
        self._dispatch = dp.build_dispatch_module(dp.DispatchConfig(pdl=cfg["pdl"], all_lists=False), self.backend)
        self._dispatch_grid = (1, 1, 1)
        self._dispatch_args = dispatch_launch_bindings(b, group_rows=group, narrow_tile=narrow_tile,
                                                       wide_min_rows=wide_min_rows, wide_min_permille=wide_min_permille,
                                                       unused=self._unused)
        self._dbg = torch.zeros(64 * sms, dtype=torch.int32, device=self.device)
        # Swap-AB SiTU GEMM1 over the narrow list (:1430-1461 with ``gemm1_lists``): one CTA per (weight M tile, list
        # slot) up to the SMs; the raster extent is the list capacity (swapab_moe.py:797).
        g1 = cfg["gemm1"]
        self._gemm1 = build_gemm1_module(self.backend, g1)
        list_capacity = int(b["swap_row_groups"].numel())
        self._gemm1_grid = (min(gm.m_chunks_of(2 * I // 128, 1) * list_capacity, sms), 1, 1)
        self._gemm1_args = gemm1_launch_bindings(
            b, weights=weights, x=x, x_sf=x_sf, route_weights=self.route_weights, hidden=H, shard=I, top_k=K,
            num_tokens=T, tiles=tiles, k_tiles=g1.k_tiles, beta=beta, linear_beta=linear_beta, dbg=self._dbg,
            unused=self._unused, row_group_list=True)
        # W1 dense gather GEMM1 over the wide list (:1554-1629): the row-group form over ``swap_wide_list`` /
        # ``swap_wide_count`` indexing the 128-row sort groups' expert / limit tables; grid one CTA per (list slot,
        # N tile) up to the SMs (the list capacity is the group capacity).
        dense_w = dense_weight_operands(weights, local_experts=L, hidden=H, shard=I)
        tile_m, n1, zf, zfs, rg, early = cfg["gemm1_dense"]
        self._gemm1_dense = g1d.build_module(self.backend, tile_m, n1, zf, zfs, rg, use_pdl=cfg["pdl"],
                                             pdl_trigger_early=early)
        self._gemm1_dense_grid = (min(tiles * (2 * I // n1), sms), 1, 1)
        self._gemm1_dense_args = gemm1_dense_launch_bindings(
            b, weights={"w1": dense_w["w1"], "w1_sf": dense_w["w1_sf"]}, x=x, x_sf=x_sf, hidden=H, shard=I, top_k=K,
            num_tokens=T, rows=rows, tile_m=tile_m, n_tile=n1, row_group_list=True, zero_fill=False, secondary=False,
            output=output, beta=beta, linear_beta=linear_beta, beta_stride=1, linear_beta_stride=1, unused=self._unused)
        # W11 dense finalize GEMM2 over every valid group (:1686-1735): reduce-add into the cleared output.
        n2, cluster_n, g2_early = cfg["gemm2_dense"]
        self._gemm2 = g2d.build_module(self.backend, n2, use_pdl=cfg["pdl"], cta_group=1, cluster_n=cluster_n,
                                       pdl_trigger_early=g2_early)
        n2_tiles = -(-H // n2)
        self._gemm2_grid = (cluster_n * min(tiles * (-(-n2_tiles // cluster_n)), sms // cluster_n), 1, 1)
        self._gemm2_args = gemm2_dense_launch_bindings(
            b, w2=dense_w["w2"], w2_sf=dense_w["w2_sf"], route_weights=self.route_weights, hidden=H, shard=I, top_k=K,
            tiles=tiles, rows=rows, n_tile=n2, output=output, dbg=self._dbg, unused=self._unused)
        self._dense_weights = dense_w
        self._beta = beta
        self._linear_beta = linear_beta
        forms = {l.step: l.form for l in decision.launch_plan}
        self.launches = tuple(ExecutedLaunch(step, forms[step], backend, grid) for step, backend, grid in (
            ("route_preprocess", self.backend, self._pre_grid), ("moe_sort_init", "cuda_cpp", self._sort_init_grid),
            ("moe_sort_coop", "cuda_cpp", self._sort_coop_grid), ("dispatch", self.backend, self._dispatch_grid),
            ("gemm1_swapab_situ", self.backend, self._gemm1_grid), ("gemm1_dense", self.backend, self._gemm1_dense_grid),
            ("gemm2_dense_finalize", self.backend, self._gemm2_grid)))
        # Warmup run = the hand-written planning postcondition (valid output after ``plan``, :1214 / :1216-1218).
        with torch.cuda.device(self.device):
            self.run()
            torch.cuda.synchronize()

    @property
    def executed_chain(self) -> dict[str, Any]:
        return executed_chain_record("hybrid", self.decision, self.launches)

    def launch_sequence(self) -> tuple[tuple[str, tuple[int, int, int]], ...]:
        return tuple((l.step, l.grid) for l in self.launches)

    def run(self):
        """Enqueue the seven launches on the caller's current stream in the hand-written order (:1992-2160: route
        preprocess, sort, dispatch, swap GEMM1, dense GEMM1, dense finalize GEMM2) and return the bound output."""
        import torch

        with torch.cuda.device(self.device):
            self._pre.launch(grid=self._pre_grid, **self._pre_args)
            self._sort_init.launch(grid=self._sort_init_grid, **self._sort_init_args)
            self._sort_coop.launch(grid=self._sort_coop_grid, **self._sort_coop_args)
            self._dispatch.launch(grid=self._dispatch_grid, **self._dispatch_args)
            self._gemm1.launch(grid=self._gemm1_grid, **self._gemm1_args)
            self._gemm1_dense.launch(grid=self._gemm1_dense_grid, **self._gemm1_dense_args)
            self._gemm2.launch(grid=self._gemm2_grid, **self._gemm2_args)
        return self.output


class CakeMxfp4MoEWrapper:
    """Metadata-only runner (hand-written ``CuteDslMxfp4MoEWrapper`` :2176-2335) for one backend.

    ``num_experts`` / ``intermediate_size`` are the global model values; the rank layout is explicit
    (``num_local_experts`` / ``local_expert_offset`` for expert parallelism, ``intermediate_shard`` for the
    MoE-TP shard).  ``get_workspace_size`` needs no CUDA; ``plan`` compiles the forms and runs one warmup.
    """

    def __init__(self, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, *,
                 num_local_experts: int | None = None, local_expert_offset: int | None = None,
                 intermediate_shard: int | None = None, backend: str = PACKAGE_BACKEND, enable_pdl: bool = False,
                 swapab_n_tile: int = SWAP_ROW_TILE, swapab_max_tokens: int | None = None,
                 swapab_tile_policy=None, sm_count: int = CONTRACT_SM_COUNT):
        if backend != PACKAGE_BACKEND:
            raise ValueError(f"this package serves backend={PACKAGE_BACKEND!r}, got {backend!r}")
        self.backend = backend
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.enable_pdl = bool(enable_pdl)
        if hidden_size % 128 or intermediate_size % 128:
            raise ValueError("hidden and intermediate dimensions must be positive multiples of 128")   # :761-764
        if not 1 <= top_k <= num_experts <= 1024 or top_k > 32:
            raise ValueError("require 1 <= top_k <= min(32, num_experts) and num_experts <= 1024")   # :765-768
        self.layout = resolve_layout(num_experts, intermediate_size, num_local_experts=num_local_experts,
                                     local_expert_offset=local_expert_offset, intermediate_shard=intermediate_shard)
        self.num_local_experts = self.layout.num_local_experts
        self.local_expert_offset = self.layout.local_expert_offset
        self.intermediate_shard = self.layout.intermediate_shard
        self.policy = CakeSwapAbPolicy(self.layout, top_k=top_k, hidden_size=hidden_size, enable_pdl=enable_pdl,
                                       swapab_n_tile=swapab_n_tile, swapab_max_tokens=swapab_max_tokens,
                                       swapab_tile_policy=swapab_tile_policy, sm_count=sm_count)
        self.swapab_max_tokens = self.policy.swapab_max_tokens
        self.swapab_tile_policy = self.policy.swapab_tile_policy

    @property
    def parallel_mode(self) -> str:
        return self.layout.mode

    def decide(self, num_tokens: int) -> PlanDecision:
        return self.policy.decide(num_tokens)

    def get_workspace_size(self, num_tokens: int) -> int:
        return self.policy.get_workspace_size(num_tokens)

    def plan(self, x, x_sf, topk_ids, topk_weights, w1, w1_sf, w2, w2_sf, *, beta, linear_beta, workspace,
             output) -> CakeSwapAbPlan | CakeDensePlan | CakeSplitPlan | CakeHybridPlan:
        """Bind buffers, compile the forms, run one warmup (:2851-3088).

        ``w1`` / ``w1_sf`` / ``w2`` / ``w2_sf`` are the :func:`prepare_cake_mxfp4_weights` operands (tile-major
        weights and 512-byte scale atoms) for this rank's shard.
        """
        import torch

        if x.device.type != "cuda":
            raise ValueError("plan requires CUDA tensors")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan must be called before CUDA Graph capture")
        T = x.shape[0]
        H, I, L, K = self.hidden_size, self.intermediate_shard, self.num_local_experts, self.top_k
        decision = self.decide(T)
        if not decision.supported:
            raise NotImplementedError(f"T={T} on {self.layout.mode} (shard {I}, {L} local experts): {decision.reason}")
        expected = {
            "x": (x, (T, H), torch.float8_e4m3fn),
            "x_sf": (x_sf, (T, H // 32), torch.uint8),
            "topk_ids": (topk_ids, (T, K), torch.int32),
            "w1": (w1, (L * (2 * I // 128), H // 128, 128, 64), torch.uint8),
            "w1_sf": (w1_sf, (L * (2 * I // 128), H // 128, 4, 128), torch.uint8),
            "w2": (w2, (L * (H // 128), I // 128, 128, 64), torch.uint8),
            "w2_sf": (w2_sf, (L * (H // 128), I // 128, 4, 128), torch.uint8),
            "output": (output, (T, H), torch.bfloat16),
        }
        for name, (tensor, shape, dtype) in expected.items():
            if (tensor.device != x.device or tensor.dtype != dtype or tuple(tensor.shape) != shape
                    or not tensor.is_contiguous()):
                raise ValueError(f"{name} must be contiguous {dtype} {shape} on {x.device}; got {tensor.dtype} "
                                 f"{tuple(tensor.shape)} on {tensor.device}")
        if topk_weights is not None and (topk_weights.device != x.device
                                         or topk_weights.dtype not in (torch.bfloat16, torch.float32)
                                         or tuple(topk_weights.shape) != (T, K) or not topk_weights.is_contiguous()):
            raise ValueError("topk_weights must be contiguous CUDA BF16/FP32 [T,top_k]")
        if beta is None:
            raise ValueError("SiTU requires runtime beta")
        for name, tensor in (("beta", beta), ("linear_beta", linear_beta)):
            if tensor is None or tensor.device != x.device or tensor.dtype != torch.float32 or tensor.ndim != 1 \
                    or tensor.numel() != L or not tensor.is_contiguous():
                # The traced GEMM1 form indexes both arrays per expert (BETA_EXPERT_STRIDE 1); the hand-written
                # broadcast variants (numel 1) are separate trace-time forms not built.
                raise ValueError(f"{name} must be CUDA FP32 [num_local_experts] (per-expert form)")
        fields, size = self.policy.workspace_fields(T)
        if (workspace.device != x.device or workspace.dtype != torch.uint8 or workspace.ndim != 1
                or not workspace.is_contiguous() or workspace.numel() < size or workspace.data_ptr() % 256):
            raise ValueError(f"workspace requires at least {size} aligned CUDA uint8 bytes")
        ws_interval = (workspace.data_ptr(), workspace.data_ptr() + size)
        out_interval = _byte_interval(output)
        if _overlap(ws_interval, out_interval):
            raise ValueError("output must not overlap workspace")
        for name, tensor in (("x", x), ("x_sf", x_sf), ("topk_ids", topk_ids), ("topk_weights", topk_weights),
                             ("w1", w1), ("w1_sf", w1_sf), ("w2", w2), ("w2_sf", w2_sf), ("beta", beta),
                             ("linear_beta", linear_beta)):
            if tensor is not None and (_overlap(ws_interval, _byte_interval(tensor))
                                       or _overlap(out_interval, _byte_interval(tensor))):
                raise ValueError(f"output/workspace must not overlap {name}")
        buffers = {f.name: workspace.narrow(0, f.offset, f.nbytes).view(getattr(torch, _TORCH_DTYPE[f.dtype]))
                   .view(f.shape) for f in fields}
        buffers["w1_alpha"].fill_(1.0)                                                      # :3058
        buffers["w2_alpha"].fill_(1.0)                                                      # :3059
        weights = {"w1": w1, "w1_sf": w1_sf, "w2": w2, "w2_sf": w2_sf}
        kwargs = dict(wrapper=self, decision=decision, buffers=buffers, workspace=workspace, x=x, x_sf=x_sf,
                      topk_ids=topk_ids, topk_weights=topk_weights, weights=weights, beta=beta,
                      linear_beta=linear_beta, output=output)
        if decision.path == "dense":
            return CakeDensePlan(**kwargs)
        if decision.path == "hybrid":
            if not HYBRID_CHAIN_EXECUTABLE:
                raise NotImplementedError(f"T={T}: the hybrid chain is traced but not enqueued (HYBRID_CHAIN_EXECUTABLE off)")
            return CakeHybridPlan(**kwargs)
        if decision.path == "split_two_stage":
            if SPLIT_CHAIN_EXECUTABLE:
                return CakeSplitPlan(**kwargs)
            # Round-10 behaviour kept until the split chain is verified: the plain chain with the split's narrow
            # tile; the plan's ``executed_chain`` labels the row "plain" (decided_path "split_two_stage").
            return CakeSwapAbPlan(**kwargs)
        if decision.path not in EXECUTABLE_PATHS:
            raise NotImplementedError(f"T={T}: the {decision.path} chain is traced but the Cake-tree runner does not "
                                      f"enqueue it (launches {decision.launches})")
        return CakeSwapAbPlan(**kwargs)


FUSED_ROUTE_SMEM_LIMIT = 232448


FUSED_ROUTE_STAGE_MIN_ROWS = 8192


FUSED_ROUTE_CLEAR_CTAS = 120


FUSED_ROUTE_CLUSTER = 8


MODES = ("packed", "separate_bf16", "separate_fp32")


@dataclass(frozen=True)
class RoutingConfig:
    """Build-time configuration (the hand-written ``_FusedRoutePreprocess`` constructor arguments)."""

    mode: str = "packed"
    threads: int = 1024
    single_tile_per_expert: bool = False
    max_routes: int = FUSED_ROUTE_MAX_ROUTES_LARGE
    clear: bool = True
    dispatch_lists: bool = True
    max_rows: int | None = None
    cluster: int = 1
    split_layout: bool = False      # two-granularity row layout of the swap-AB split form (exclusive with the lists)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.dispatch_lists and self.split_layout:
            raise ValueError("dispatch_lists and split_layout are exclusive")
        if self.split_layout and self.single_tile_per_expert:
            raise ValueError("the split layout never runs the single-tile prefix (mxfp4_routing.py:1305-1310)")
        if self.threads not in (1024, 2048) or self.threads % 32:
            raise ValueError("the fused routing kernel runs 1024 threads (2048 above 1024 local experts)")
        if not 1 <= self.cluster <= 8:
            raise ValueError("cluster must be 1..8")
        if self.max_routes not in (FUSED_ROUTE_MAX_ROUTES, FUSED_ROUTE_MAX_ROUTES_LARGE):
            raise ValueError("max_routes must be 8192 or 16384")

    # Derived geometry (constructor lines 271-345 of the hand-written module).
    @property
    def warps(self) -> int:
        return self.threads // 32

    @property
    def chunk_routes(self) -> int:
        routes_per_cta = (self.max_routes + self.cluster - 1) // self.cluster
        return (routes_per_cta + self.threads - 1) // self.threads * self.threads

    @property
    def fixed_smem(self) -> int:
        return 4 * (3 * self.threads + 2 * self.warps) + 4 * self.chunk_routes

    @property
    def stage_rows(self) -> bool:
        return self.max_routes > FUSED_ROUTE_MAX_ROUTES and self.cluster == 1

    @property
    def staged_rows(self) -> int:
        fit = (FUSED_ROUTE_SMEM_LIMIT - self.fixed_smem) // 2
        if fit <= 0:
            raise ValueError("fused routing scratch exceeds the shared memory limit")
        wanted = self.max_rows if self.max_rows is not None else self.max_routes
        return max(1, min(wanted, fit)) if self.stage_rows else 0

    @property
    def smem_bytes(self) -> int:
        return self.fixed_smem + 2 * self.staged_rows

    @property
    def packed(self) -> bool:
        return self.mode == "packed"

    @property
    def convert_weights(self) -> bool:
        return self.mode != "separate_fp32"


def plan_config(*, tokens, top_k, num_local_experts, tile_size, mode="packed", clear=True,
                dispatch_lists=True, rows_capacity=None, fused_route_cluster=FUSED_ROUTE_CLUSTER,
                scratch_words=None, split_layout=False):
    """The configuration ``_plan_route_preprocess`` selects for one problem (mxfp4_routing.py:1010-1373).

    ``rows_capacity`` = ``get_max_num_tiles(...) * tile_size`` (the caller's permuted-row capacity);
    ``scratch_words`` = size of the ``out_expert_counts`` scratch (``None`` = no cluster).
    """
    required = max(1024, num_local_experts)
    threads = 1 << (required - 1).bit_length()
    routes = tokens * top_k
    if routes > FUSED_ROUTE_MAX_ROUTES_LARGE:
        raise ValueError("routes exceed the fused routing cap (generic moe_sort path)")
    # mxfp4_routing.py:975-979: the 8192-route variant up to 8192 routes, the large variant above.
    max_routes = FUSED_ROUTE_MAX_ROUTES if routes <= FUSED_ROUTE_MAX_ROUTES else FUSED_ROUTE_MAX_ROUTES_LARGE
    cluster = 1
    if fused_route_cluster > 1 and routes > threads and scratch_words is not None:
        wanted = min(fused_route_cluster, -(-routes // threads))
        if scratch_words >= wanted * threads:
            cluster = wanted
    if rows_capacity is None:
        rows_capacity = get_max_num_tiles(tokens, top_k, num_local_experts, tile_size) * tile_size
    max_rows = -(-rows_capacity // 4096) * 4096 if max_routes > FUSED_ROUTE_MAX_ROUTES else 0
    return RoutingConfig(mode=mode, threads=threads, single_tile_per_expert=tile_size >= tokens and not split_layout,
                         max_routes=max_routes, clear=clear, dispatch_lists=dispatch_lists and not split_layout,
                         max_rows=max_rows, cluster=cluster, split_layout=split_layout)


def launch_grid(cfg: RoutingConfig, output_words: int) -> int:
    """CTA count of the hand-written launch (mxfp4_routing.py:429-441)."""
    blocks = cfg.cluster
    if cfg.clear:
        if cfg.cluster > 1:
            clear_ctas = min(-(-output_words // (4 * cfg.threads)), FUSED_ROUTE_CLEAR_CTAS)
            blocks = -(-(cfg.cluster + clear_ctas) // cfg.cluster) * cfg.cluster
        else:
            blocks = -(-output_words // cfg.threads)
    return blocks


@dataclass(frozen=True)
class PreprocessConfig:
    """``_RoutePreprocess(mode, threads, clear)``: the non-fused conversion + clear launch (generic ``moe_sort`` path)."""

    mode: str = "packed"
    threads: int = 256
    clear: bool = True

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.threads not in (128, 256):
            raise ValueError("route preprocessing supports 128 or 256 threads per block")

    @property
    def packed(self) -> bool:
        return self.mode == "packed"

    @property
    def convert_weights(self) -> bool:
        return self.mode != "separate_fp32"


def preprocess_grid(cfg: PreprocessConfig, tokens: int, top_k: int, hidden: int) -> int:
    """``ceil_div(tasks, threads)``: tasks = ``T * (H / 8)`` when clearing, else ``T * top_k`` (mxfp4_routing.py:110-117)."""
    tasks = tokens * (hidden // 8) if cfg.clear else tokens * top_k
    return -(-tasks // cfg.threads)


def m_chunks_of(num_m_tiles, tiles_per_chunk):
    """Weight M-tile chunks of the raster (hand-written ``ceil_div(num_m_tiles, m_group)``, and for the 2-CTA
    form the pair's two CTA tiles share one chunk: ``cur[0] // cta_v``, :1619-1621); ``tiles_per_chunk`` is
    ``m_group * cta_v``, passed explicitly (module helpers see the module defaults, not the clone's)."""
    return num_m_tiles if tiles_per_chunk == 1 else (num_m_tiles + (tiles_per_chunk - 1)) // tiles_per_chunk


def _unused_situ_operands(device):
    """Placeholders for the SiTU-only parameters of a finalize launch (never dereferenced: ``zero_words`` is 0)."""
    import torch

    return dict(situ_beta=torch.zeros(1, dtype=torch.float32, device=device),
                situ_linear_beta=torch.zeros(1, dtype=torch.float32, device=device),
                act_sf=torch.zeros(1, dtype=torch.uint8, device=device),
                zero_buf=torch.zeros(2, dtype=torch.float32, device=device),
                zero_words=0, num_rows_b=0, act_cols=0, act_sf_cols=0)


MAX_EXPANDED_PER_THREAD = 64          # MaxExpandedIdxPerThread of the generic (NumTop8Experts) state (RoutingKernel.cuh:1205-1207)


BOUNDED_EXPANDED_PER_THREAD = 4       # ... of the bounded NumTop16Experts state of the high-expert tiers (:1206)


RESERVED_SMS = 8                      # kDefaultReservedSMsForOverlapping (RoutingKernel.cuh:49)


EXPERT_TIERS = (128, 160, 256, 384, 512, 576, 896, 1024)   # routingCustom::getMaxNumExperts tiers (RoutingCustomPolicy.cuh:748-770)


HIGH_EXPERT_TIER_RANGE = (512, 1024)


def high_expert_tier(tier: int) -> bool:
    """``isInHighExpertLaneOwnedTopKRange(tier, NumTop16Experts)``: the tier has a bounded-state kernel."""
    return HIGH_EXPERT_TIER_RANGE[0] <= tier <= HIGH_EXPERT_TIER_RANGE[1]


@dataclass(frozen=True)
class MoeSortConfig:
    """Build-time shape of the coop kernel: the tier and the ``Data`` pointers ``moe_sort`` binds."""

    tier: int = 384
    dual: bool = True                 # routingDualTileEnabled: mPaddingLog2Alt > mPaddingLog2 && alt list bound
    mixed: bool = True                # mMixedNarrowTile > 0 && mPtrMixedNarrowList bound (needs dual)
    narrow_count_base: bool = True    # mPtrMixedNarrowCountBase bound
    pdl: bool = True                  # mUsePdl
    bounded: bool = False             # the NumTop16Experts kernel of the 512..1024 tiers: 4 expanded indices per thread

    def __post_init__(self):
        if self.tier not in EXPERT_TIERS:
            raise ValueError(f"tier must be one of {EXPERT_TIERS}")
        if self.mixed and not self.dual:
            raise ValueError("the mixed work lists need the dual-tile routing")
        if self.narrow_count_base and not self.mixed:
            raise ValueError("narrow_count_base needs the mixed work lists")
        if self.bounded and not high_expert_tier(self.tier):
            raise ValueError("the bounded four-entry state exists for the 512..1024 tiers only")

    @property
    def threads(self) -> int:
        return self.tier

    @property
    def expanded_per_thread(self) -> int:
        """``MaxExpandedIdxPerThread`` (RoutingKernel.cuh:1205-1207)."""
        return BOUNDED_EXPANDED_PER_THREAD if self.bounded else MAX_EXPANDED_PER_THREAD

    @property
    def warps(self) -> int:
        return self.tier // 32


def init_grid(cfg: MoeSortConfig, num_experts: int) -> int:
    """``(2 * numExperts - 1) / numThreadsHist + 1`` (trtllm_fused_moe_routing_custom.cuh:1501)."""
    return (2 * num_experts - 1) // cfg.threads + 1


def coop_grid(sm_count: int) -> int:
    """``getCoopLaunchSMCounts(smCount).moeSms`` with the default eight reserved SMs."""
    return sm_count - RESERVED_SMS


def init_form_name(cfg: MoeSortConfig) -> str:
    """IR export name of the init kernel of a tier (``moe_sort_init_t<tier>``; tier-only)."""
    return f"moe_sort_init_t{cfg.tier}"


def coop_form_name(cfg: MoeSortConfig) -> str:
    """IR export name of a cooperative-kernel configuration: ``moe_sort_coop_t<tier>[_bounded][_dual][_mixed]``
    (``narrow_count_base`` and ``pdl`` are launch-time operands / attributes of the same kernel)."""
    return (f"moe_sort_coop_t{cfg.tier}" + ("_bounded" if cfg.bounded else "") + ("_dual" if cfg.dual else "")
            + ("_mixed" if cfg.mixed else ""))


FINALIZE_VEC = 8          # BF16 elements per thread task (mxfp4_finalize.py:34)


WORDS_PER_CHUNK = FINALIZE_VEC // 2


@dataclass(frozen=True)
class FinalizeConfig:
    """``_FinalizeRows(top_k, threads, expanded_rows, accumulate, skip_wide)``."""

    top_k: int = 8
    threads: int = 256
    expanded_rows: bool = False
    accumulate: bool = False
    skip_wide: bool = False

    def __post_init__(self):
        if not 1 <= self.top_k <= 64:
            raise ValueError("top_k must be in [1, 64]")
        if self.threads % 32 or self.threads > 1024:
            raise ValueError("threads must be a multiple of 32 up to 1024")
        if self.accumulate and not self.skip_wide:
            raise ValueError("accumulate requires skip_wide (the wide group count)")

    @property
    def warps(self) -> int:
        return self.threads // 32


def finalize_grid(cfg: FinalizeConfig, tokens: int, hidden: int) -> int:
    chunks = (hidden // 2) // WORDS_PER_CHUNK
    return -(-(tokens * chunks) // cfg.threads)


@dataclass(frozen=True)
class DispatchConfig:
    """K3 build-time variant: ``kPdl`` and whether the ``all`` list pointers are given (non-null)."""

    pdl: bool = True
    all_lists: bool = True


# ----------------------------------------------------------------------------------------------------------
# Package binding: the Cake-module accessors and kernel builders the plan below reaches for, served by the
# generated modules of this package (``cake_mxfp4_situ_moe_kernels``) instead of the producer's compilers.
# ----------------------------------------------------------------------------------------------------------
class _RoutingModule:
    """The routing helpers the plan reads through ``_routing_module()`` (extracted above): the fused-routing
    configuration and the K1 route-preprocess configuration of the generic (``moe_sort``) routing path."""

    RoutingConfig = RoutingConfig
    PreprocessConfig = PreprocessConfig
    MODES = MODES
    plan_config = staticmethod(plan_config)
    launch_grid = staticmethod(launch_grid)
    preprocess_grid = staticmethod(preprocess_grid)

    @staticmethod
    def build_preprocess_module(cfg, backend: str) -> "PackageLaunch":
        _check_backend(backend)
        return _package_launch(_kernels().find_kernel(kind="route_preprocess", mode=cfg.mode, threads=int(cfg.threads),
                                                      clear=bool(cfg.clear)))


class _GemmModule:
    """The swap-AB GEMM helpers the plan reads through ``_gemm_module()`` (extracted above); the kernel builders
    of the swap-AB forms are ``build_gemm1_module`` / ``build_gemm2_module`` / ``build_gemm2_partial_module``."""

    m_chunks_of = staticmethod(m_chunks_of)
    _unused_situ_operands = staticmethod(_unused_situ_operands)


class _MoeSortModule:
    """The K6 ``moe_sort`` helpers the plan reads through ``_moe_sort_module()`` (extracted above); the kernels
    are CUDA-only and load from the sibling package under the CuTe DSL package (``_kernels_for``)."""

    MoeSortConfig = MoeSortConfig
    init_form_name = staticmethod(init_form_name)
    coop_form_name = staticmethod(coop_form_name)
    init_grid = staticmethod(init_grid)
    coop_grid = staticmethod(coop_grid)

    @staticmethod
    def build_init_module(cfg) -> "PackageLaunch":
        stage = _k6_stage(init_form_name(cfg), cfg.pdl)
        return _package_launch(_kernels_for(stage).load(stage), use_pdl=cfg.pdl)

    @staticmethod
    def build_coop_module(cfg) -> "PackageLaunch":
        stage = _k6_stage(coop_form_name(cfg), cfg.pdl)
        return _package_launch(_kernels_for(stage).load(stage), use_pdl=cfg.pdl)


def _k6_stage(name: str, pdl: bool) -> str:
    """The generated stage of a K6 form name under its PDL attribute: ``cfg.pdl`` is a trace-time constant of the
    kernel (griddepcontrol emission), so the PDL-off kernel is its own module, ``<name>_nopdl``."""
    return name if pdl else name + "_nopdl"


class _Gemm1DenseModule:
    """The dense gather GEMM1 builder the plan reaches through ``_gemm1_dense_module()``; the module is selected
    by the form's trace-time fields and the launch attribute the chain asks for (the dense chain launches with the
    wrapper's ``enable_pdl``; both attributes are rendered)."""

    @staticmethod
    def build_module(backend: str, tile_m: int, n_tile: int, zero_fill: bool, zero_fill_secondary: bool,
                     row_group_list: bool, *, use_pdl: bool, pdl_trigger_early: bool = False) -> "PackageLaunch":
        _check_backend(backend)
        kernel = _kernels().find_kernel(kind="gemm1_dense", tile_m=int(tile_m), n_tile=int(n_tile),
                                        zero_fill=bool(zero_fill), zero_fill_secondary=bool(zero_fill_secondary),
                                        row_group_list=bool(row_group_list), pdl_trigger_early=bool(pdl_trigger_early),
                                        use_pdl=bool(use_pdl))
        return _package_launch(kernel, use_pdl=use_pdl)


class _FinalizeModule:
    """The K7 ``finalize_rows`` helpers the split two-stage chain reaches through ``_finalize_module()``
    (extracted above: the configuration, the chunk width of the launch bindings, the launch grid); the kernel is
    the generated module whose ``route.form`` carries the configuration's trace-time fields."""

    FinalizeConfig = FinalizeConfig
    WORDS_PER_CHUNK = WORDS_PER_CHUNK
    finalize_grid = staticmethod(finalize_grid)

    @staticmethod
    def build_finalize_module(cfg, backend: str) -> "PackageLaunch":
        _check_backend(backend)
        return _package_launch(_kernels().find_kernel(kind="finalize", top_k=int(cfg.top_k), threads=int(cfg.threads),
                                                      expanded_rows=bool(cfg.expanded_rows),
                                                      accumulate=bool(cfg.accumulate), skip_wide=bool(cfg.skip_wide)))


class _DispatchModule:
    """The K3 ``swapab_dispatch`` helpers the hybrid chain reaches through ``_dispatch_module()`` (extracted above:
    the configuration); the kernel is the generated module whose ``route.form`` carries the configuration's
    trace-time fields (``pdl``, ``all_lists``)."""

    DispatchConfig = DispatchConfig

    @staticmethod
    def build_dispatch_module(cfg, backend: str) -> "PackageLaunch":
        _check_backend(backend)
        return _package_launch(_kernels().find_kernel(kind="dispatch", pdl=bool(cfg.pdl), all_lists=bool(cfg.all_lists)))


class _Gemm2DenseModule:
    """The dense finalize GEMM2 builder the plan reaches through ``_gemm2_dense_module()``."""

    @staticmethod
    def build_module(backend: str, n_tile: int, *, use_pdl: bool, cta_group: int = 1, cluster_n: int = 1,
                     row_group: bool = False, pdl_trigger_early: bool = False) -> "PackageLaunch":
        _check_backend(backend)
        kernel = _kernels().find_kernel(kind="gemm2_dense", n_tile=int(n_tile), cta_group=int(cta_group),
                                        cluster_n=int(cluster_n), row_group=bool(row_group),
                                        pdl_trigger_early=bool(pdl_trigger_early), use_pdl=bool(use_pdl))
        return _package_launch(kernel, use_pdl=use_pdl)


def _routing_module():
    return _RoutingModule


def _gemm_module():
    return _GemmModule


def _moe_sort_module():
    return _MoeSortModule


def _gemm1_dense_module():
    return _Gemm1DenseModule


def _gemm2_dense_module():
    return _Gemm2DenseModule


def _finalize_module():
    return _FinalizeModule


def _dispatch_module():
    return _DispatchModule


def _kernels():
    from . import cake_mxfp4_situ_moe_kernels as kernels

    return kernels


def _sibling_kernels():
    """The sibling package's kernel loader (``SIBLING_MODULE``), pinned to the same Cake revision."""
    import importlib

    sibling = importlib.import_module(f"{SIBLING_MODULE}.cake_mxfp4_situ_moe_kernels")
    mine = _kernels()
    if sibling.cake_revision() != mine.cake_revision():
        raise RuntimeError(
            f"sibling package {SIBLING_MODULE} is pinned to Cake revision {sibling.cake_revision()}, "
            f"this package to {mine.cake_revision()}; install both packages from one export")
    return sibling


def _kernels_for(stage: str):
    """The loader that serves ``stage``: this package, or -- for a form this package's backend does not build
    (``unsupported_forms``) -- the sibling package named by the manifest's ``sibling_backend_forms`` (mixed
    backend per kernel: the same generated module the sibling backend runs; numerics unchanged)."""
    kernels = _kernels()
    if stage not in kernels.unsupported_forms():
        return kernels
    sibling = dict(kernels.manifest()["contract"].get("sibling_backend_forms", {})).get(stage)
    if sibling is None:
        raise KeyError(f"{stage}: not in this package ({kernels.unsupported_forms()[stage]}) and no sibling package serves it")
    if sibling["module"] != SIBLING_MODULE:
        raise RuntimeError(f"{stage}: the manifest names sibling package {sibling['module']!r}, expected {SIBLING_MODULE!r}")
    return _sibling_kernels()


def is_available() -> bool:
    """``True`` when this package's kernels can be loaded on this installation."""
    return bool(_kernels().is_available())


class PackageLaunch:
    """One generated module of this package as the plan's launch handle (``launch(grid=..., **bindings)``).

    The kernel is bound to the device of the first launch's tensors (the plan's warmup run, before any CUDA-graph
    capture); later launches perform no allocation.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.stage: str = kernel.stage
        self._bound = None

    def launch(self, grid, **bindings) -> None:
        if self._bound is None:
            device = next(value.device for value in bindings.values() if hasattr(value, "device"))
            self._bound = self.kernel.bind(device)
        self._bound.launch(tuple(grid), **bindings)


def _check_backend(backend: str) -> None:
    if backend != PACKAGE_BACKEND:
        raise ValueError(f"this package serves backend={PACKAGE_BACKEND!r}, the plan asks for {backend!r}")


def _package_launch(kernel, *, use_pdl: bool | None = None, pdl_placement: dict | None = None) -> PackageLaunch:
    if use_pdl is not None and bool(kernel.use_pdl) != bool(use_pdl):
        raise ValueError(
            f"{kernel.stage}: this package's module was generated with use_pdl={kernel.use_pdl}, the plan asks "
            f"for use_pdl={use_pdl}")
    for key, wanted in (pdl_placement or {}).items():
        # The griddepcontrol placement (late_dep_wait / pdl_trigger_after_wait) and the weight-stream L2 policy
        # (weight_l2_hint) are trace-time constants of the generated module (``route.form``); the plan's selection
        # must be the one the module was generated with.
        if key not in kernel.form or bool(kernel.form[key]) != bool(wanted):
            raise ValueError(
                f"{kernel.stage}: this package's module was generated with {key}={kernel.form.get(key)}, the plan "
                f"asks for {key}={wanted}")
    return PackageLaunch(kernel)


def _pdl_placement(form: GemmForm) -> dict:
    """The swap-AB form's trace-time variant selection: griddepcontrol placement + weight-stream L2 policy."""
    return {"late_dep_wait": form.late_dep_wait, "pdl_trigger_after_wait": form.pdl_trigger_after_wait,
            "weight_l2_hint": form.weight_l2_hint is not None}


def build_gemm1_module(backend: str, form: GemmForm) -> PackageLaunch:
    """The swap-AB SiTU GEMM1 module of ``form`` (``route.form`` match on ``n_tile`` / ``kbps`` / ``row_group_list``
    -- the hybrid / mixed192 ``_rowgroup`` forms pair the list with the blocked output scales -- on the
    griddepcontrol placement ``pdl_trigger_after_wait`` (the split chain's narrow forms exist under both) and on the
    weight-stream L2 policy ``weight_l2_hint`` (the n16 / n32 forms exist under both))."""
    _check_backend(backend)
    if form.sf_blocked != form.row_group_list:
        raise ValueError("the swap-AB SiTU list forms pair tile_idx_to_row_group with sf_blocked")
    kernel = _kernels().select_gemm1(form.n_tile, form.kbps, row_group_list=form.row_group_list,
                                     pdl_trigger_after_wait=form.pdl_trigger_after_wait,
                                     weight_l2_hint=form.weight_l2_hint is not None)
    return _package_launch(kernel, use_pdl=form.use_pdl, pdl_placement=_pdl_placement(form))


def build_gemm2_module(backend: str, form: GemmForm) -> PackageLaunch:
    """The swap-AB finalize GEMM2 module of ``form`` (``n_tile`` / ``kbps`` / ``m_group``, the placement
    ``late_dep_wait`` and the weight-stream L2 policy ``weight_l2_hint``)."""
    _check_backend(backend)
    kernel = _kernels().select_gemm2(form.n_tile, form.kbps, form.m_group, late_dep_wait=form.late_dep_wait,
                                     weight_l2_hint=form.weight_l2_hint is not None)
    return _package_launch(kernel, use_pdl=form.use_pdl, pdl_placement=_pdl_placement(form))


def build_gemm2_partial_module(backend: str, form: GemmForm) -> PackageLaunch:
    """The swap-AB *partial* GEMM2 module of ``form`` (the deferred epilogue of the two-stage chains, ``out =
    partial_rows``; ``n_tile`` / ``kbps`` / ``m_group`` match on the partial form table)."""
    _check_backend(backend)
    kernel = _kernels().find_kernel(kind="gemm2_swapab_partial", n_tile=int(form.n_tile), kbps=int(form.kbps),
                                    m_group=int(form.m_group), late_dep_wait=bool(form.late_dep_wait),
                                    weight_l2_hint=form.weight_l2_hint is not None)
    return _package_launch(kernel, use_pdl=form.use_pdl, pdl_placement=_pdl_placement(form))


def build_routing_module(backend: str, cfg) -> PackageLaunch:
    """The fused routing module of ``cfg`` (every trace-time ``RoutingConfig`` field must match)."""
    _check_backend(backend)
    return _package_launch(_kernels().select_routing(cfg))


def build_launch_module(backend: str, form: str, *, use_pdl: bool | None = None) -> PackageLaunch:
    """The generated module of one plan ``Launch.form`` (a ``route.stage``, or a registry id for the kernels the
    plan names by their registry entry), from this package or -- for a CUDA-only form under the CuTe DSL
    package -- from the sibling package (``_kernels_for``)."""
    _check_backend(backend)
    kernels = _kernels()
    stage = form
    if not any(dict(item["route"]).get("stage") == form for item in kernels.modules()) and form not in kernels.unsupported_forms():
        matches = [item for item in kernels.modules() if dict(item["route"].get("form", {})).get("registry") == form]
        if len(matches) != 1:
            raise KeyError(f"{form}: not a stage or registry id of one module of this package")
        stage = str(matches[0]["route"]["stage"])
    return _package_launch(_kernels_for(stage).load(stage), use_pdl=use_pdl)


def kernel_stages(plan) -> tuple[str, ...]:
    """``route.stage`` of the generated modules an executable plan launches, in construction order (the plain
    chain: routing, swap-AB GEMM1, swap-AB GEMM2 finalize; the dense chain: route preprocess, the K6 pair, dense
    GEMM1, dense finalize GEMM2; the split two-stage chain: split routing, wide row-group GEMM1 / GEMM2, narrow
    swap-AB GEMM1 / partial GEMM2, finalize rows; the hybrid chain: route preprocess, the K6 pair, dispatch, swap-AB
    row-group GEMM1, dense row-group GEMM1, dense finalize GEMM2)."""
    return tuple(value.stage for value in vars(plan).values() if isinstance(value, PackageLaunch))
