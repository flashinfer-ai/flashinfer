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

Backend of :mod:`flashinfer.cake_vsa_sm90`: Hopper (SM90) variable block-sparse
attention on the generated Cake kernels as one self-contained package (the host
planner :mod:`.cake_planner`, the plan object :class:`CakeVsaSm90Plan`, the
``cake_jit`` registry / loader and the generated CUDA sources under ``csrc/``).
``backend="cake"`` of :class:`flashinfer.sparse.VariableBlockSparseAttentionWrapper`
reaches :func:`create_plan` through the :mod:`flashinfer.cake_vsa_sm90` facade;
import that module for the user-facing documentation.

Supported scope: BF16 HND ``q/k/v`` with shape ``(H, S, 128)``, or Sage FP8
``q/k/v`` (``torch.float8_e4m3fn``, same shapes) with their FP32 scales
``q_scale (1, H, Sq)`` per token, ``k_scale (1, H, ceil(Sk / 16))`` per 16-token
group and ``v_scale (H, 128)`` per channel bound at ``plan()`` (the output stays
BF16); ``num_qo_heads = g x num_kv_heads`` with ``g`` in {1, 2, 4, 8} (``g = 1``:
the MHA programs; ``g`` in {2, 4, 8}: the grouped-query programs, query head
``h`` attending kv head ``h // g``; other ratios are refused), the block mask and
the K / V scales shaped by the kv heads (``block_mask_map (num_kv_heads, MB,
NB)``, ``k_scale (1, num_kv_heads, ceil(Sk / 16))``, ``v_scale (num_kv_heads,
128)``) and ``q`` / ``q_scale`` / the output by the query heads; noncausal
64-token query and KV blocks, 1..64 selected KV blocks per query block (ragged
counts allowed), custom finite softmax scales (including 0 and negative), no LSE
/ PDL / positional encoding / logits soft cap.  The data type and the head ratio
are route decisions of the plan (``q_data_type``, ``gqa``), never kernel flags:
the FP8 stages are twins of the BF16 stages that consume the same planner bytes,
and the GQA stages consume the GQA planner's bytes (the kv head in plan word 2 of
a persistent plan; the ``g_shift`` scalar of a small plan).
:func:`sage_fp8_quantize_sm90` produces the FP8 inputs and scales from BF16 HND
tensors with the package's two quantizer kernels (``k`` / ``v`` may carry the kv
heads of a GQA problem).

Two kernels serve the contract.  Small problems (at most 6 selected KV blocks
per query block, a grid that fits one wave, a plan that fits the kernel
parameter budget; :func:`small_route`) run the *small-selection* kernel: one
CTA per query block whose plan (``[meta, qtile, blk...]`` int16 halfwords,
:func:`plan_small`) travels in the launch parameters, so no metadata upload and
no dependent global load precede the K/V loads.  Grids too small for the
persistent kernel's two warpgroups per CTA take the kernel's *split-KV*
variant (a query block's selection sliced across ``ceil(count / KMAX)`` CTAs
that merge through a plan-owned workspace; :func:`split_kmax`) or, when its
modelled cost is lower, its *cluster* variant (the slices form one thread-block
cluster and merge through distributed shared memory; :func:`cluster_variant_for`);
a split-KV route whose slices fit one cluster and save enough per-CTA ingest
takes the *split-cluster* variant instead (the slices form one cluster and
merge their workspace partials column-set by column-set;
:func:`split_cluster_variant_for`).  Everything else runs the persistent
kernel: the plan (:func:`plan_vsa_sm90`) is uploaded once as a per-CTA list of
tiles (``grid = min(tiles, SMs)``), either one 64-row query block whose KV list
alternates between the two consumer warpgroups (``split``) or two query blocks
of one head sharing the KV ring (``pair``), and plans with enough tiles per CTA
keep one static tile per CTA and hand the rest out through a device queue in
head-major order (``attention_queue``); uniform pair plans of short tiles with many tiles
per CTA run the compact static program (``attention_compact``).  The plan layouts mirror the Cake kernel modules
and are validated by the kernels' own tests; both planners are dependency-free
ports of the Cake planners and must stay byte-identical to them.

This package imports only itself, ``torch``, ``tvm_ffi`` and ``cuda.bindings``
(the cluster-capacity probe); it reads no environment variable.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import tvm_ffi

from . import cake_jit as _loader
from .cake_jit import (  # noqa: F401
    ARCH_BY_CAPABILITY,
    ARCHES,
    FP8_GQA_PERSISTENT_STAGES,
    FP8_GQA_SMALL_STAGES,
    FP8_PERSISTENT_STAGES,
    FP8_SMALL_STAGES,
    GQA_PERSISTENT_STAGES,
    GQA_SMALL_STAGES,
    LOGICAL_PARAM_ARRAYS,
    LOGICAL_SCALARS,
    LOGICAL_TENSORS,
    MODULES,
    PERSISTENT_STAGES,
    QUANTIZE_STAGES,
    SMALL_STAGES,
    STAGES,
    arch_for_capability,
    persistent_stage,
    small_stage,
    stage_dtype,
    stage_gqa,
    stage_record,
)
from .cake_planner import (  # noqa: F401
    BLOCK,
    CLUSTER_VARIANTS,
    GQA_GROUPS,
    HDR_HALFWORDS,
    HEAD_DIM,
    LOG2E,
    MAX_NSPLIT,
    MAX_SELECTED,
    PLAN_HALFWORDS,
    PLAN_META_SPLIT,
    PLAN_META_UNSPLIT,
    POOL_MIN_TILES,
    POOL_RAGGED_MIN_TILES,
    POOL_ROUNDS,
    SMALL_ITEM_ELEMS,
    SMALL_KMAX_VARIANTS,
    SMALL_OCCUPANCY,
    SMALL_STATS_FLOATS,
    SPLIT_CLUSTER_VARIANTS,
    SPLIT_KMAX_VARIANTS,
    _FP32_MAX,
    cluster_capacity,
    cluster_cost,
    cluster_variant_for,
    fp8_gqa_pin,
    fp8_gqa_xshare,
    gqa_group,
    plan_small,
    plan_vsa_sm90,
    small_kmax_for,
    small_prefetch,
    small_route,
    split_cluster_variant_for,
    split_cost,
    split_kmax,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
# Sage FP8 route: E4M3 q/k/v, FP32 scales per query token / per 16-token KV group /
# per V channel (the SM100 Sage package's layout); the quantizer kernels handle
# 64 tokens per CTA and index with 32-bit offsets.
FP8_DTYPE = torch.float8_e4m3fn
K_SCALE_GROUP = 16
QUANTIZE_TOKENS_PER_CTA = 64
_INT32_ELEMENTS = 2**31 - 1


def is_cake_vsa_sm90_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device with an exported build (Hopper, compute capability 9.0)."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    return (
        arch_for_capability(tuple(torch.cuda.get_device_capability(device))) is not None
    )


# ---------------------------------------------------------------------------
# Wrapper-facing plan object
# ---------------------------------------------------------------------------


def _check_descriptors(block_mask_map, block_row_sz, block_col_sz, num_kv_heads):
    """The sparse descriptors are keyed by the kv heads: ``block_mask_map (num_kv_heads, MB, NB)``,
    ``block_row_sz (num_kv_heads, MB)``, ``block_col_sz (num_kv_heads, NB)`` of 64-token blocks (the ``g``
    query heads of a kv head share its pattern; ``num_kv_heads == num_qo_heads`` for MHA)."""
    if (
        not isinstance(block_mask_map, torch.Tensor)
        or block_mask_map.dtype != torch.bool
    ):
        raise ValueError("cake (SM90) requires a boolean block_mask_map")
    if block_mask_map.ndim != 3:
        raise ValueError("block_mask_map must have shape (num_kv_heads, MB, NB)")
    h, mb, nb = block_mask_map.shape
    if h != num_kv_heads:
        raise ValueError(
            "cake (SM90) requires one sparse pattern per kv head: block_mask_map.shape[0] "
            f"({h}) must equal num_kv_heads ({num_kv_heads})"
        )
    for name, sizes, count in (
        ("block_row_sz", block_row_sz, mb),
        ("block_col_sz", block_col_sz, nb),
    ):
        if not isinstance(sizes, torch.Tensor) or tuple(sizes.shape) != (h, count):
            raise ValueError(f"{name} must have shape ({h}, {count})")
        if not bool((sizes.to("cpu") == BLOCK).all()):
            raise ValueError(f"cake (SM90) requires {BLOCK}-token blocks in {name}")


class CakeVsaSm90Plan:
    """Wrapper-owned plan: uploaded tile metadata plus the launch geometry.

    ``run`` issues exactly one kernel launch on the current stream (after an
    event wait on the metadata upload), so a fixed plan is CUDA-graph
    capturable; ``plan()`` may not replace a plan that a graph has captured.

    The stage the plan launches (``self.stage``, a record of
    :data:`cake_jit.MODULES`) is fixed when the plan is built -- the route and
    the data type (``self.fp8``) -- its library is loaded then (outside graph
    capture), and its ffi arguments are bound once: ``run`` only substitutes
    ``q``, ``k``, ``v`` and ``out``.  An FP8 plan binds ``q_scale`` / ``k_scale``
    / ``v_scale`` here; a caller that re-quantizes per forward writes the same
    tensors in place (graph-safe) and ``run`` takes no scales.

    ``num_qo_heads = g x num_kv_heads`` with ``g`` in {1, 2, 4, 8} (``self.gqa``):
    ``g = 1`` plans and runs the MHA stages exactly as before; ``g`` in {2, 4, 8}
    plans the GQA stage of the same route on the kv-head-shaped mask (query head
    ``h`` attends kv head ``h // g``), with ``q`` / ``out`` / ``q_scale`` shaped
    by the query heads and ``k`` / ``v`` / ``k_scale`` / ``v_scale`` by the kv
    heads; a small GQA plan binds its ``g_shift`` scalar.  Other ratios are
    refused with ``ValueError``.
    """

    def __init__(
        self,
        device: torch.device,
        block_mask_map: torch.Tensor,
        block_row_sz: torch.Tensor,
        block_col_sz: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: Optional[torch.dtype] = None,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        non_blocking: bool = True,
        route: Optional[str] = None,
    ):
        """``route`` ``None`` applies :func:`small_route`; ``"small"`` / ``"smallsplit"`` / ``"smallcluster"`` /
        ``"smallsplitcluster"`` / ``"persistent"`` force one kernel (the forced split-cluster route takes the
        smallest variant that holds the selection and fits the plan parameter).  ``q_data_type``
        ``torch.float8_e4m3fn`` (with ``kv_data_type`` equal or ``None``) selects the Sage FP8 twin of the
        route and requires the three FP32 scales ``q_scale (1, num_qo_heads, Sq)``, ``k_scale (1, num_kv_heads,
        ceil(Sk / 16))``, ``v_scale (num_kv_heads, 128)``, contiguous on the plan device; BF16 plans take none."""
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("cake (SM90) requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (9, 0):
            raise ValueError("cake (SM90) VSA requires Hopper compute capability 9.0")
        if causal:
            raise ValueError("cake (SM90) VSA supports noncausal attention only")
        if pos_encoding_mode != "NONE":
            raise ValueError("cake (SM90) VSA does not apply positional encodings")
        if use_fp16_qk_reduction:
            raise ValueError("cake (SM90) VSA accumulates QK in FP32 only")
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise ValueError("cake (SM90) VSA does not support logits_soft_cap")
        if q_data_type not in (torch.bfloat16, FP8_DTYPE):
            raise ValueError(
                "cake (SM90) VSA supports BF16 or Sage FP8 (float8_e4m3fn) q/k/v only"
            )
        if kv_data_type is not None and kv_data_type != q_data_type:
            raise ValueError(
                "cake (SM90) VSA requires kv_data_type equal to q_data_type (BF16 or "
                f"float8_e4m3fn q/k/v only), got q_data_type={q_data_type}, "
                f"kv_data_type={kv_data_type}"
            )
        self.fp8 = q_data_type == FP8_DTYPE
        self.q_dtype = q_data_type
        scales = {"q_scale": q_scale, "k_scale": k_scale, "v_scale": v_scale}
        given = [name for name, tensor in scales.items() if tensor is not None]
        if self.fp8 and len(given) != 3:
            missing = [name for name in scales if name not in given]
            raise ValueError(
                "cake (SM90) Sage FP8 requires q_scale, k_scale and v_scale together "
                f"(missing: {', '.join(missing)})"
            )
        if not self.fp8 and given:
            raise ValueError(
                "cake (SM90) VSA accepts q_scale/k_scale/v_scale only with "
                f"q_data_type=float8_e4m3fn (Sage FP8); got q_data_type={q_data_type}"
            )
        if num_qo_heads < 1 or num_kv_heads < 1:
            raise ValueError("cake (SM90) VSA requires positive query/KV head counts")
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_qo_heads ({num_qo_heads}) must be a multiple of num_kv_heads "
                f"({num_kv_heads})"
            )
        # The head ratio is a route decision: 1 = the MHA stages, 2 / 4 / 8 = the
        # GQA stages (query head h attends kv head h >> g_shift); others refused.
        self.gqa = gqa_group(num_qo_heads // num_kv_heads)
        self.g_shift = self.gqa.bit_length() - 1
        self.num_qo_heads = int(num_qo_heads)
        self.num_kv_heads = int(num_kv_heads)
        if head_dim != HEAD_DIM:
            raise ValueError(f"cake (SM90) VSA requires head_dim {HEAD_DIM}")
        _check_descriptors(block_mask_map, block_row_sz, block_col_sz, num_kv_heads)
        if self.fp8:
            kv_heads, mb, nb = (int(x) for x in block_mask_map.shape)
            self.k_scale_groups = -(-(nb * BLOCK) // K_SCALE_GROUP)
            for name, tensor, shape in (
                ("q_scale", q_scale, (1, self.num_qo_heads, mb * BLOCK)),
                ("k_scale", k_scale, (1, kv_heads, self.k_scale_groups)),
                ("v_scale", v_scale, (kv_heads, HEAD_DIM)),
            ):
                self._check_scale(name, tensor, shape)
            self.q_scale, self.k_scale, self.v_scale = q_scale, k_scale, v_scale
        scale = 1.0 / math.sqrt(HEAD_DIM) if sm_scale is None else float(sm_scale)
        # The kernel consumes scale * log2(e) as FP32.
        if not math.isfinite(scale * LOG2E) or abs(scale * LOG2E) > _FP32_MAX:
            raise ValueError("sm_scale must be finite in FP32")
        self.sm_scale = scale
        self.scale_log2 = scale * LOG2E
        props = torch.cuda.get_device_properties(self.device)
        sms = int(props.multi_processor_count)
        if route not in (
            None,
            "small",
            "smallsplit",
            "smallcluster",
            "smallsplitcluster",
            "persistent",
        ):
            raise ValueError(
                f"unknown route {route!r}; expected None, 'small', 'smallsplit', 'smallcluster', "
                "'smallsplitcluster' or 'persistent'"
            )
        small: Optional[tuple[int, bool, int]] = None
        # The forced small routes size their variant on the mask the kernels
        # see: one row per query head (the kv-head mask replicated per group).
        mask_rep = block_mask_map.to("cpu")
        if self.gqa > 1:
            mask_rep = mask_rep.repeat_interleave(self.gqa, dim=0)
        if route is None:
            small = small_route(
                block_mask_map,
                sms=sms,
                cluster_capacity=cluster_capacity(self.device.index),
                gqa=self.gqa,
            )
        elif route == "small":
            small = (
                small_kmax_for(int(mask_rep.sum(dim=-1).max())),
                False,
                0,
            )
        elif route == "smallsplit":
            counts = mask_rep.sum(dim=-1).reshape(-1).tolist()
            small = (split_kmax(counts), True, 0)
        elif route == "smallcluster":
            counts = mask_rep.sum(dim=-1).reshape(-1).tolist()
            variant = cluster_variant_for(counts)
            if variant is None:
                raise ValueError("no cluster variant holds this selection")
            small = (variant[0], False, variant[1])
        elif route == "smallsplitcluster":
            counts = mask_rep.sum(dim=-1).reshape(-1).tolist()
            fitting = [
                (kmax, csize)
                for kmax, csize in SPLIT_CLUSTER_VARIANTS
                if kmax * csize >= max(counts)
                and len(counts) * (kmax * csize + PLAN_META_UNSPLIT) <= PLAN_HALFWORDS
            ]
            if not fitting:
                raise ValueError("no split-cluster variant holds this selection")
            small = (fitting[0][0], True, fitting[0][1])
        self.captured = False
        self.small_kmax: Optional[int] = None if small is None else small[0]
        self.small_split = False
        self.small_cluster = 0
        self.small_prefetch: Optional[int] = None
        # whether a persistent grouped-query plan holds re-paired split tiles
        # (``plan_vsa_sm90(..., repair_split)``; never on the small route)
        self.repair_split = False
        self.arch = _loader.device_arch(self.device.index)
        # Plan-owned ffi arguments by logical name (``cake_jit.LOGICAL_*``);
        # ``run`` adds q / k / v / out and the grid.
        bound: dict[str, Any] = {
            "scale_log2": float(self.scale_log2),
        }
        if self.fp8:
            # Scales are plan data (R3): bound once, read by every launch; the
            # per-head stride of ``k_scale`` travels as a kernel scalar.
            bound["q_scale"] = self.q_scale
            bound["k_scale"] = self.k_scale
            bound["v_scale"] = self.v_scale
            bound["k_scale_groups"] = int(self.k_scale_groups)
        if small is not None:
            plan = plan_small(
                block_mask_map,
                kmax=small[0],
                split=small[1],
                cluster=small[2],
                gqa=self.gqa,
            )
            self.small_split = bool(plan["split"])
            self.small_cluster = int(plan["cluster"])
            if self.small_split and self.small_cluster:
                self.mode = "smallsplitcluster"
            elif self.small_cluster:
                self.mode = "smallcluster"
            else:
                self.mode = "smallsplit" if self.small_split else "small"
            self.num_tiles = plan["num_tiles"]
            self.num_items = plan["num_items"]
            self.num_ctas = plan["num_items"]
            self.tile_stride = 0
            self.num_heads = plan["H"]
            self.qo_len = plan["MB"] * BLOCK
            self.kv_len = plan["NB"] * BLOCK
            # The depth-2 grouped-query FP8 k4 stage on the grids ``small_prefetch`` names
            # (same plan, same output bits; ``None`` = the stage without a depth suffix).
            self.small_prefetch = small_prefetch(
                self.small_kmax,
                self.small_split,
                self.small_cluster,
                num_items=plan["num_items"],
                fp8=self.fp8,
                gqa=self.gqa,
            )
            self.stage = small_stage(
                self.small_kmax,
                self.small_split,
                self.small_cluster,
                fp8=self.fp8,
                gqa=self.gqa > 1,
                prefetch=self.small_prefetch,
            )
            if self.gqa > 1:
                # The small GQA stages take the group as a kernel scalar
                # (``kv_head = head >> g_shift``); the plan rows are the
                # replicated mask's.
                assert int(plan["g_shift"]) == self.g_shift
                bound["g_shift"] = self.g_shift
            # The plan is a launch parameter (CPU int16): nothing is uploaded,
            # so an unsplit or cluster plan has no stream dependency and a
            # captured graph replays it.  A split plan owns device workspace
            # whose arrival counters must be zero before the first run
            # (``ready``); the kernel leaves them zero, so replays need no host
            # reset.  One run per split plan may be in flight at a time.  A
            # split-cluster plan owns the same workspace (its slices publish
            # their partials there and rendezvous on the cluster barrier; the
            # counters stay zero and are bound only because the kernel's
            # parameter list is the split kernel's).
            self.plan_param = plan["plan"].contiguous()
            bound["plan"] = self.plan_param
            if self.small_split:
                rule = _loader.workspace_rule(self.stage)
                assert rule is not None and rule["kind"] == "split"
                with torch.cuda.device(self.device):
                    self.partial_o = torch.empty(
                        (self.num_items * int(rule["item_floats"]),),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.partial_stats = torch.empty(
                        (self.num_items * int(rule["stats_floats"]),),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.counters = torch.zeros(
                        (int(getattr(self, rule["counter_words"])),),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    # The rendered binding takes ``u32`` arrival counters; keep the
                    # int32 storage (uint32 lacks most reductions) and pass a view.
                    self.counters_u32 = self.counters.view(torch.uint32)
                    self.ready = torch.cuda.Event(external=True)
                    self.ready.record(torch.cuda.current_stream(self.device))
                bound["partial_o"] = self.partial_o
                bound["partial_stats"] = self.partial_stats
                bound["counters"] = self.counters_u32
        else:
            plan = plan_vsa_sm90(
                block_mask_map,
                sms=sms,
                gqa=self.gqa,
                # the Sage-FP8 route keeps the group-major split order (its split
                # rows measured slower head-major); the BF16 route takes the default
                split_tile_order="group" if self.fp8 else None,
            )
            self.mode = plan["mode"]
            self.repair_split = bool(plan["repair_split"])
            self.num_ctas = plan["num_ctas"]
            self.num_tiles = plan["num_tiles"]
            self.tile_stride = plan["tile_stride"]
            self.queue = plan["queue"]
            self.compact = plan["compact"]
            self.tagged = plan["tagged"]
            self.num_rows = plan["num_rows"]
            self.num_heads = plan["H"]
            self.qo_len = plan["MB"] * BLOCK
            self.kv_len = plan["NB"] * BLOCK
            self.stage = persistent_stage(
                bool(self.queue),
                bool(self.compact),
                bool(self.tagged),
                fp8=self.fp8,
                gqa=self.gqa > 1,
                # the Sage-FP8 route runs head-pair queue plans on the shared-transposer
                # queue program; split-mode queue plans keep the queue program
                xshare=self.fp8 and fp8_gqa_xshare(plan),
                # ... and static plans whose lightest CTA runs at least PIN_MIN_CTA_LOOP_ITERS
                # main-loop iterations on the position-selected static program (a plan key)
                pin=self.fp8 and fp8_gqa_pin(plan),
            )
            rule = _loader.workspace_rule(self.stage)
            assert rule is not None and rule["kind"] == "persistent"
            with torch.cuda.device(self.device):
                host_meta = plan["meta"]
                if non_blocking:
                    host_meta = host_meta.pin_memory()
                self.meta = host_meta.to(self.device, non_blocking=non_blocking)
                # By-value kernel parameter in both builds: stays on the host.
                self.hdr = plan["hdr"].contiguous()
                # The timeline buffer is inert in the exported build; ``dbg`` holds
                # the queue kernel's two tile-queue words (zero between launches:
                # the last CTA to drain the pool resets them, so a captured graph
                # replays without a memset).  Like the split-KV counters, the words
                # belong to this plan: two launches of one plan must not overlap.
                self.dbg = torch.zeros(
                    (int(rule["dbg_words"]),), dtype=torch.int32, device=self.device
                )
                self.tl = torch.zeros(
                    (int(rule["tl_words"]),), dtype=torch.uint64, device=self.device
                )
                # Capture must retain an external wait node for the metadata
                # upload, which was recorded outside the captured stream.
                self.ready = torch.cuda.Event(external=True)
                self.ready.record(torch.cuda.current_stream(self.device))
            self._host_meta = host_meta  # keeps pinned storage alive for the copy
            bound["meta"] = self.meta
            bound["hdr"] = self.hdr
            bound["tile_stride"] = int(self.tile_stride)
            bound["dbg"] = self.dbg
            bound["tl"] = self.tl
        bound["seqlen_q"] = int(self.qo_len)
        bound["seqlen_k"] = int(self.kv_len)
        self.grid = _loader.launch_grid(
            self.stage,
            {"num_ctas": self.num_ctas, "num_items": getattr(self, "num_items", 0)},
        )
        bound.update(zip(_loader.GRID_NAMES, self.grid, strict=True))
        self._bound = bound
        self._names = _loader.argument_names(self.stage)
        # Loaded here, outside graph capture (``create_plan`` refuses capture):
        # the prebuilt library or FlashInfer's JIT build of the stage.
        self._entry = getattr(
            _loader.load(self.stage, self.arch), stage_record(self.stage)["ffi_entry"]
        )

    def check_replan(self) -> None:
        if self.captured:
            raise RuntimeError(
                "A captured VSA plan cannot be replaced; create a new wrapper and "
                "keep the original wrapper alive while its graph may replay"
            )

    def _check_tensor(self, name, tensor, shape, dtype=torch.bfloat16) -> None:
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            raise ValueError(
                f"{name} must have shape {tuple(shape)} and dtype {str(dtype).removeprefix('torch.')}"
            )
        if tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous on the planned device {self.device}"
            )

    def _check_scale(self, name, tensor, shape) -> None:
        """An FP32 Sage scale of the planned problem: exact shape, contiguous on the plan device."""
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a float32 tensor of shape {tuple(shape)}")
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != torch.float32:
            raise ValueError(
                f"{name} must have shape {tuple(shape)} and dtype float32, got "
                f"{tuple(tensor.shape)} {str(tensor.dtype).removeprefix('torch.')}"
            )
        if tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous on the planned device {self.device}"
            )

    @staticmethod
    def _aligned(tensor: torch.Tensor) -> torch.Tensor:
        # TMA needs 16-byte aligned bases; an offset view is copied on the
        # current stream (graph-capturable).
        if tensor.data_ptr() % 16 == 0:
            return tensor
        return tensor.clone(memory_format=torch.contiguous_format)

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        if return_lse or lse is not None:
            raise ValueError("cake (SM90) VSA does not support log-sum-exp outputs")
        if enable_pdl:
            raise ValueError("cake (SM90) VSA does not support PDL")
        for name, tensor, heads, length in (
            ("q", q, self.num_heads, self.qo_len),
            ("k", k, self.num_kv_heads, self.kv_len),
            ("v", v, self.num_kv_heads, self.kv_len),
        ):
            self._check_tensor(name, tensor, (heads, length, HEAD_DIM), self.q_dtype)
        # Preserve VariableBlockSparseAttentionWrapper's existing DPS ABI: the
        # provided output is [H*M, 1, D], while the return value is HND.  The
        # output is BF16 for both data types.
        if out is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "CUDA graph capture requires a preallocated out buffer"
                )
            result = torch.empty_like(q, dtype=torch.bfloat16)
        else:
            self._check_tensor("out", out, (self.num_heads * self.qo_len, 1, HEAD_DIM))
            result = out.view(self.num_heads, self.qo_len, HEAD_DIM)
            begin = out.data_ptr()
            end = begin + out.numel() * out.element_size()
            inputs: tuple[torch.Tensor, ...] = (q, k, v)
            if self.fp8:
                scales = (self.q_scale, self.k_scale, self.v_scale)
                inputs += tuple(scale for scale in scales if scale is not None)
            for tensor in inputs:
                tensor_begin = tensor.data_ptr()
                tensor_end = tensor_begin + tensor.numel() * tensor.element_size()
                if begin < tensor_end and tensor_begin < end:
                    raise ValueError("out must not overlap Q/K/V or scale storage")

        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if torch.cuda.is_current_stream_capturing():
                self.captured = True
            if self.small_kmax is None or self.small_split:
                # The device meta table (persistent route) or the split
                # workspace may have been built on another stream: an event
                # wait enqueues the dependency without a CPU synchronization.
                # Unsplit / cluster plans are by-value launch arguments and
                # need no wait.
                stream.wait_event(self.ready)
            aligned_q = self._aligned(q)
            aligned_k = self._aligned(k)
            aligned_v = self._aligned(v)
            target = result if result.data_ptr() % 16 == 0 else torch.empty_like(result)
            self._launch(aligned_q, aligned_k, aligned_v, target)
            if target is not result:
                result.copy_(target)
        return result

    def _launch(self, aligned_q, aligned_k, aligned_v, target) -> None:
        values = self._bound
        call = {"q": aligned_q, "k": aligned_k, "v": aligned_v, "out": target}
        args = [call[name] if name in call else values[name] for name in self._names]
        with tvm_ffi.use_torch_stream():
            self._entry(*args)


def create_plan(device, *args, **kwargs) -> CakeVsaSm90Plan:
    """Validate the request and build the uploaded plan (outside graph capture)."""
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("cake (SM90) VSA requires Hopper compute capability 9.0")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Call plan() outside CUDA graph capture")
        return CakeVsaSm90Plan(device, *args, **kwargs)


# ---------------------------------------------------------------------------
# Sage FP8 quantizer (the SM100 Sage package's kernels on HND tensors)
# ---------------------------------------------------------------------------


def _check_hnd(name, tensor, device, shape) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise ValueError(f"{name} must be a rank-3 HND tensor (H, S, {HEAD_DIM})")
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != torch.bfloat16:
        raise ValueError(
            f"{name} must have shape {tuple(shape)} and dtype bfloat16, got "
            f"{tuple(tensor.shape)} {str(tensor.dtype).removeprefix('torch.')}"
        )
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")
    if tensor.numel() > _INT32_ELEMENTS:
        raise ValueError(f"{name} exceeds 2**31 - 1 elements (32-bit kernel offsets)")


def sage_fp8_quantize_sm90(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fused Sage FP8 quantization of BF16 HND ``q/k/v`` for the FP8 route of :class:`CakeVsaSm90Plan`.

    Returns ``(q8, k8, v8, q_scale, k_scale, v_scale)``: E4M3 tensors of the input shapes
    and FP32 scales ``q_scale (1, H, Sq)`` per query token, ``k_scale (1, H_kv, ceil(Sk / 16))``
    per 16-token KV group (shared by its 16 tokens and all 128 channels) and ``v_scale (H_kv, 128)``
    per V channel over the whole sequence -- the shapes ``plan()`` takes.  ``k`` and ``v`` carry
    the kv heads: ``H_kv = H`` (MHA) or ``H = g x H_kv`` with ``g`` in {2, 4, 8} (the GQA routes;
    every tensor is quantized per its own head, so the result equals the three tensors quantized
    separately).  A scale is
    ``max(amax, 1e-6) / 448`` (the kernels floor the absolute maximum at ``1e-6`` and multiply by
    the reciprocal), so the torch recipe ``amax.clamp_min(1e-12) / 448`` agrees to about 1e-5
    relative on the scales and to one E4M3 step on the values.  Two kernel launches of the
    package's ``quantize_qk_vamax`` / ``quantize_v`` stages behind one ``torch.zeros`` of the V
    amax scratch (three launches on a GQA problem: ``quantize_qk_vamax`` once over the query
    heads with its K side idle and once over the kv heads with its Q side idle).  HND
    ``(H, S, 128)`` is the kernels' BSHD layout with ``batch := H`` and one head, so the inputs
    are passed as they are and the scale outputs come out in the plan's shapes.  Each input
    holds at most ``2**31 - 1`` elements (32-bit kernel offsets).
    """
    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise ValueError(f"q must be a rank-3 HND tensor (H, S, {HEAD_DIM})")
    device = q.device
    if device.type != "cuda":
        raise ValueError("q must be a CUDA tensor")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    heads, seqlen_q, dim = (int(x) for x in q.shape)
    if dim != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}, got {dim}")
    if not isinstance(k, torch.Tensor) or k.ndim != 3:
        raise ValueError(f"k must be a rank-3 HND tensor (H, S, {HEAD_DIM})")
    kv_heads, seqlen_k = int(k.shape[0]), int(k.shape[1])
    if heads < 1 or kv_heads < 1 or seqlen_q < 1 or seqlen_k < 1:
        raise ValueError("q/k/v must have positive head and sequence counts")
    if heads % kv_heads != 0:
        raise ValueError(
            f"k must have shape (H_kv, {seqlen_k}, {HEAD_DIM}) with q's head count ({heads}) a "
            f"multiple of H_kv, got {tuple(k.shape)}"
        )
    gqa_group(heads // kv_heads)
    _check_hnd("q", q, device, (heads, seqlen_q, HEAD_DIM))
    _check_hnd("k", k, device, (kv_heads, seqlen_k, HEAD_DIM))
    _check_hnd("v", v, device, (kv_heads, seqlen_k, HEAD_DIM))
    groups = -(-seqlen_k // K_SCALE_GROUP)
    q_chunks = -(-seqlen_q // QUANTIZE_TOKENS_PER_CTA)
    k_chunks = -(-seqlen_k // QUANTIZE_TOKENS_PER_CTA)
    with torch.cuda.device(device):
        q8 = torch.empty_like(q, dtype=FP8_DTYPE)
        k8 = torch.empty_like(k, dtype=FP8_DTYPE)
        v8 = torch.empty_like(v, dtype=FP8_DTYPE)
        q_scale = torch.empty((1, heads, seqlen_q), dtype=torch.float32, device=device)
        k_scale = torch.empty((1, kv_heads, groups), dtype=torch.float32, device=device)
        v_scale = torch.empty((kv_heads, HEAD_DIM), dtype=torch.float32, device=device)
        v_amax = torch.zeros((kv_heads, HEAD_DIM), dtype=torch.float32, device=device)
        values = {
            "q_in": q,
            "k_in": k,
            "v_in": v,
            "q_out": q8,
            "k_out": k8,
            "q_scale": q_scale,
            "k_scale": k_scale,
            "v_amax": v_amax,
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "num_heads": 1,
            "num_kv_heads": 1,
            "grid_heads": 1,
            "k_scale_groups": groups,
        }
        if kv_heads == heads:
            # One launch over the common heads: the chunk count is the longer side's.
            launches = [(heads, max(q_chunks, k_chunks), q_chunks, k_chunks)]
        else:
            # GQA: the kernel gates its Q side on ``chunk < q_chunks`` and its K / V
            # side on ``chunk < k_chunks`` per CTA, so one launch over the query heads
            # with no K chunks quantizes q alone and one over the kv heads with no Q
            # chunks quantizes k and the V amax alone (``batch := heads`` either way).
            launches = [
                (heads, q_chunks, q_chunks, 0),
                (kv_heads, k_chunks, 0, k_chunks),
            ]
        for batch, chunks, q_active, k_active in launches:
            _loader.run_module(
                "quantize_qk_vamax",
                {
                    **values,
                    "chunks_per_bh": chunks,
                    "q_chunks": q_active,
                    "k_chunks": k_active,
                },
                grid=(batch * chunks, 1, 1),
                device_index=device.index,
            )
        _loader.run_module(
            "quantize_v",
            {
                "v_in": v,
                "v_out": v8,
                "v_amax": v_amax,
                "v_scale": v_scale,
                "seqlen_k": seqlen_k,
                "num_heads": 1,
                "chunks_per_bh": k_chunks,
            },
            grid=(kv_heads * k_chunks, 1, 1),
            device_index=device.index,
        )
    return q8, k8, v8, q_scale, k_scale, v_scale


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "FP8_DTYPE",
    "FP8_GQA_PERSISTENT_STAGES",
    "FP8_GQA_SMALL_STAGES",
    "FP8_PERSISTENT_STAGES",
    "FP8_SMALL_STAGES",
    "GQA_GROUPS",
    "GQA_PERSISTENT_STAGES",
    "GQA_SMALL_STAGES",
    "K_SCALE_GROUP",
    "MODULES",
    "QUANTIZE_STAGES",
    "STAGES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeVsaSm90Plan",
    "arch_for_capability",
    "cluster_capacity",
    "cluster_variant_for",
    "create_plan",
    "fp8_gqa_pin",
    "fp8_gqa_xshare",
    "is_cake_vsa_sm90_supported",
    "persistent_stage",
    "plan_small",
    "plan_vsa_sm90",
    "sage_fp8_quantize_sm90",
    "small_prefetch",
    "small_route",
    "small_stage",
    "split_cluster_variant_for",
    "split_kmax",
    "stage_dtype",
    "stage_gqa",
]
