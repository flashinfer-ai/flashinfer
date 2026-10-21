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

Supported scope: BF16 HND ``q/k/v`` with shape ``(H, S, 128)``, equal query/KV
head counts, noncausal 64-token query and KV blocks, 1..64 selected KV blocks
per query block (ragged counts allowed), custom finite softmax scales
(including 0 and negative), no LSE / PDL / positional encoding / logits soft cap.

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
    LOGICAL_PARAM_ARRAYS,
    LOGICAL_SCALARS,
    LOGICAL_TENSORS,
    MODULES,
    PERSISTENT_STAGES,
    SMALL_STAGES,
    STAGES,
    arch_for_capability,
    persistent_stage,
    small_stage,
    stage_record,
)
from .cake_planner import (  # noqa: F401
    BLOCK,
    CLUSTER_VARIANTS,
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
    plan_small,
    plan_vsa_sm90,
    small_kmax_for,
    small_route,
    split_cluster_variant_for,
    split_cost,
    split_kmax,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)


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


def _check_descriptors(block_mask_map, block_row_sz, block_col_sz, num_qo_heads):
    if (
        not isinstance(block_mask_map, torch.Tensor)
        or block_mask_map.dtype != torch.bool
    ):
        raise ValueError("cake (SM90) requires a boolean block_mask_map")
    if block_mask_map.ndim != 3:
        raise ValueError("block_mask_map must have shape (num_heads, MB, NB)")
    h, mb, nb = block_mask_map.shape
    if h != num_qo_heads:
        raise ValueError(
            "cake (SM90) requires one sparse pattern per head: block_mask_map.shape[0] "
            f"({h}) must equal num_qo_heads ({num_qo_heads})"
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
    :data:`cake_jit.MODULES`) is fixed when the plan is built, its library is
    loaded then (outside graph capture), and its ffi arguments are bound once:
    ``run`` only substitutes ``q``, ``k``, ``v`` and ``out``.
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
        non_blocking: bool = True,
        route: Optional[str] = None,
    ):
        """``route`` ``None`` applies :func:`small_route`; ``"small"`` / ``"smallsplit"`` / ``"smallcluster"`` /
        ``"smallsplitcluster"`` / ``"persistent"`` force one kernel (the forced split-cluster route takes the
        smallest variant that holds the selection and fits the plan parameter)."""
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
        if q_data_type != torch.bfloat16 or (
            kv_data_type is not None and kv_data_type != torch.bfloat16
        ):
            raise ValueError("cake (SM90) VSA supports BF16 q/k/v only")
        if num_qo_heads != num_kv_heads or num_qo_heads < 1:
            raise ValueError(
                "cake (SM90) VSA requires equal, positive query/KV head counts"
            )
        if head_dim != HEAD_DIM:
            raise ValueError(f"cake (SM90) VSA requires head_dim {HEAD_DIM}")
        _check_descriptors(block_mask_map, block_row_sz, block_col_sz, num_qo_heads)
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
        if route is None:
            small = small_route(
                block_mask_map,
                sms=sms,
                cluster_capacity=cluster_capacity(self.device.index),
            )
        elif route == "small":
            small = (
                small_kmax_for(int(block_mask_map.to("cpu").sum(dim=-1).max())),
                False,
                0,
            )
        elif route == "smallsplit":
            counts = block_mask_map.to("cpu").sum(dim=-1).reshape(-1).tolist()
            small = (split_kmax(counts), True, 0)
        elif route == "smallcluster":
            counts = block_mask_map.to("cpu").sum(dim=-1).reshape(-1).tolist()
            variant = cluster_variant_for(counts)
            if variant is None:
                raise ValueError("no cluster variant holds this selection")
            small = (variant[0], False, variant[1])
        elif route == "smallsplitcluster":
            counts = block_mask_map.to("cpu").sum(dim=-1).reshape(-1).tolist()
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
        self.arch = _loader.device_arch(self.device.index)
        # Plan-owned ffi arguments by logical name (``cake_jit.LOGICAL_*``);
        # ``run`` adds q / k / v / out and the grid.
        bound: dict[str, Any] = {
            "scale_log2": float(self.scale_log2),
        }
        if small is not None:
            plan = plan_small(
                block_mask_map, kmax=small[0], split=small[1], cluster=small[2]
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
            self.stage = small_stage(
                self.small_kmax, self.small_split, self.small_cluster
            )
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
            plan = plan_vsa_sm90(block_mask_map, sms=sms)
            self.mode = plan["mode"]
            self.num_ctas = plan["num_ctas"]
            self.num_tiles = plan["num_tiles"]
            self.tile_stride = plan["tile_stride"]
            self.queue = plan["queue"]
            self.compact = plan["compact"]
            self.num_rows = plan["num_rows"]
            self.num_heads = plan["H"]
            self.qo_len = plan["MB"] * BLOCK
            self.kv_len = plan["NB"] * BLOCK
            self.stage = persistent_stage(bool(self.queue), bool(self.compact))
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

    def _check_tensor(self, name, tensor, shape) -> None:
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"{name} must have shape {tuple(shape)} and dtype bfloat16"
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
        for name, tensor, length in (
            ("q", q, self.qo_len),
            ("k", k, self.kv_len),
            ("v", v, self.kv_len),
        ):
            self._check_tensor(name, tensor, (self.num_heads, length, HEAD_DIM))
        # Preserve VariableBlockSparseAttentionWrapper's existing DPS ABI: the
        # provided output is [H*M, 1, D], while the return value is HND.
        if out is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "CUDA graph capture requires a preallocated out buffer"
                )
            result = torch.empty_like(q)
        else:
            self._check_tensor("out", out, (self.num_heads * self.qo_len, 1, HEAD_DIM))
            result = out.view(self.num_heads, self.qo_len, HEAD_DIM)
            begin = out.data_ptr()
            end = begin + out.numel() * out.element_size()
            for tensor in (q, k, v):
                tensor_begin = tensor.data_ptr()
                tensor_end = tensor_begin + tensor.numel() * tensor.element_size()
                if begin < tensor_end and tensor_begin < end:
                    raise ValueError("out must not overlap Q/K/V storage")

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


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "MODULES",
    "STAGES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeVsaSm90Plan",
    "arch_for_capability",
    "cluster_capacity",
    "cluster_variant_for",
    "create_plan",
    "is_cake_vsa_sm90_supported",
    "persistent_stage",
    "plan_small",
    "plan_vsa_sm90",
    "small_route",
    "small_stage",
    "split_cluster_variant_for",
    "split_kmax",
]
