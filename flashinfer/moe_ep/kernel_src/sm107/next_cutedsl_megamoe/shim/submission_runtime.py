# SPDX-License-Identifier: Apache-2.0
"""Host side of NVIDIA's CuTe DSL ``BlockScaledSwapAbMegaMoeKernel`` for one EP group.

One runtime serves every NVFP4 MegaMoE layer of a process (they share shapes):

* one compiled kernel per token bucket (``max_tokens_per_rank`` is compile-time),
  each with its own local and symmetric workspace (their layouts depend on it);
* one torch symmetric-memory slab per process holding the operands peers read or
  write: the staged NVFP4 tokens and their scales, the top-k scores and every
  bucket's symmetric workspace, carved in the same order on every rank and
  addressed through one ``SymmetricBufferHost`` (``peer = local + offset[rank]``);
* local buffers: top-k ids, the MXFP8 shared-expert input and its swizzled scales,
  the shared-expert output.

A call stages the tokens (one Triton launch, ``staging.py``) and launches the
compiled variant, which runs the router, the persistent fused kernel and the
top-k reduce (adding the shared-expert output) on the current stream. Workspaces
are zeroed once at allocation; each launch's tail resets its own counters behind
an all-rank NVLink barrier, so consecutive layers reuse them without host work and
the whole call can be captured in a CUDA graph.

Kernels from cutedsl_megamoe main 0611b4d on keep their per-launch data in a
caller-owned tmp buffer (``get_tmp_buffer_bytes()``; any content at launch, nothing
survives it) instead of the local workspace. Launches of one process run in stream
order, so one tmp buffer of the largest bucket's size serves every bucket and layer.
Older checkouts (no ``get_tmp_buffer_bytes``) get no tmp buffer, as before.

The kernel source is not part of this package: it is imported from the checkout
named by ``VLLM_NVFP4_MEGAMOE_KERNEL_DIR``.
"""

import logging
import os
import time
import types

import torch

from . import submission_staging as staging
from .submission_config import bucket_for, kernel_settings, round_up, token_buckets

logger = logging.getLogger("vllm.plugins.nvfp4_megamoe")

_SLAB_ALIGN = 256
_kernel_api = None


def import_kernel() -> types.SimpleNamespace:
    """Load the pinned Rubin CuTeDSL kernel API."""
    global _kernel_api
    if _kernel_api is not None:
        return _kernel_api
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as cutlass_utils
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.typing import AddressSpace
    from ..submission_6cb79ec1.api import ImplDesc, ProblemDesc
    from ..submission_6cb79ec1.communication.nvlink_domain.symmetric_buffer import SymmetricBufferHost
    from ..submission_6cb79ec1 import RubinInferenceMegaMoE
    from ..submission_6cb79ec1.quant_def import CombineFormat, QuantKind

    _kernel_api = types.SimpleNamespace(
        cuda=cuda, cutlass=cutlass, cute=cute, cutlass_torch=cutlass_torch, cutlass_utils=cutlass_utils,
        OperandMajorMode=OperandMajorMode, AddressSpace=AddressSpace, ImplDesc=ImplDesc,
        ProblemDesc=ProblemDesc, SymmetricBufferHost=SymmetricBufferHost,
        Kernel=RubinInferenceMegaMoE, CombineFormat=CombineFormat, QuantKind=QuantKind,
    )
    return _kernel_api


def to_cute(tensor: torch.Tensor, assumed_align: int = 16):
    """torch -> cute tensor, leading dimension dynamic (one compiled kernel for every layer)."""
    api = import_kernel()
    cute_tensor = api.cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    return cute_tensor.mark_layout_dynamic(leading_dim=api.cutlass_torch.get_leading_dim(tensor))


def to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 16):
    """A raw Uint8 global-memory pointer for an opaque byte workspace."""
    api = import_kernel()
    return api.cute.runtime.make_ptr(
        api.cutlass.Uint8, tensor.data_ptr(), api.AddressSpace.gmem, assumed_align=assumed_align
    )


def current_cu_stream():
    return import_kernel().cuda.CUstream(torch.cuda.current_stream().cuda_stream)


class _ClusterCounts:
    """Co-resident cluster counts (probe compiles; whole device), cached per size."""

    def __init__(self) -> None:
        self._max_active: dict[int, int] = {}

    def max_active(self, size: int) -> int:
        if size not in self._max_active:
            api = import_kernel()
            self._max_active[size] = int(api.cutlass_utils.HardwareInfo().get_max_active_clusters(size))
        return self._max_active[size]

    def launch(self, preferred: tuple[int, int], fallback: tuple[int, int] | None):
        """(launch_cluster_count, preferred_cluster_count, fallback_cluster_count) as NVIDIA's solver."""
        preferred_size = preferred[0] * preferred[1]
        if fallback is None or tuple(fallback) == tuple(preferred):
            return self.max_active(preferred_size), None, None
        fallback_size = fallback[0] * fallback[1]
        fallback_capacity = self.max_active(fallback_size) * fallback_size
        preferred_count = self.max_active(preferred_size)
        preferred_capacity = preferred_count * preferred_size
        if fallback_capacity < preferred_capacity:
            raise ValueError("fallback cluster CTA capacity is below the preferred clusters' capacity")
        split = preferred_size // fallback_size
        fallback_count = (fallback_capacity - preferred_capacity) // fallback_size // split * split
        return preferred_count + fallback_count // split, preferred_count, fallback_count


def make_kernel(api, *, bucket, settings, counts, experts, hidden, intermediate, shared_intermediate, topk,
                world_size, clamp):
    launch_count, preferred_count, fallback_count = counts.launch(
        settings.cluster_shape_mn, settings.fallback_cluster_shape_mn
    )
    tile = tuple(settings.mma_tiler_mnk)
    instruction_k = api.QuantKind("nvfp4").instruction_k("2x")
    # The fused MXFP8 shared expert runs (tile K / routed instruction K) instructions of its own K per stage, and that
    # tile must hold whole block-scaled chunks (4 x 32 values): the routed tile K must be a multiple of 2 instructions.
    if tile[2] % (2 * instruction_k):
        tile = (tile[0], tile[1], (tile[2] // (2 * instruction_k) + 1) * 2 * instruction_k)
        logger.info("NVFP4 MegaMoE bucket %d: tile K raised to %d for the fused MXFP8 shared expert", bucket, tile[2])
    problem = api.ProblemDesc(
        {
            "expert_count": experts,
            "intermediate_gateup_size": 2 * intermediate,
            "hidden_size": hidden,
            "quant_kind": "nvfp4",
            "a_major_mode": api.OperandMajorMode.K,
            "b_major_mode": api.OperandMajorMode.K,
            "combine_format": api.CombineFormat.parse("bf16"),
            "gate_up_clamp": clamp,
            "situ_beta": None,
            "situ_linear_beta": None,
            "world_size": world_size,
            "topk": topk,
            "topk_index_dtype": api.cutlass.Int32,
            "max_tokens_per_rank": bucket,
            "apply_topk_at_fc1": True,
            "shared_expert": True,
            "shared_quant_kind": "mxfp8_e4m3",
            "shared_hidden_size": hidden,
            "shared_intermediate_gateup_size": 2 * shared_intermediate,
            "shared_output_size": hidden,
            "reduce_shared_output": True,
        }
    )
    fields = {
        "mma_instruction_mnk": (tile[0], tile[1], instruction_k),
        "mma_tiler_mnk": tile,
        "mma_k_mode": "2x",
        "cluster_shape_mn": tuple(settings.cluster_shape_mn),
        "use_2cta_instrs": tile[0] == 256,
        "schedule_policy": tuple(settings.schedule_policy),
        "token_padding_block": settings.token_padding_block,
        "sf_padding_block": settings.sf_padding_block,
        "work_id_mode": settings.work_id_mode,
        "fc2_use_bulk": settings.fc2_use_bulk,
        "fc2_tma_stages": settings.fc2_tma_stages,
        "epi_flag_batches": tuple(settings.epi_flag_batches),
        "launch_cluster_count": launch_count,
        "token_in_flag_batch": settings.token_in_flag_batch,
        "token_back_mode": settings.token_back_mode,
        "token_back_on_compute_ctas": False,
        "reduce_topk_in_kernel": False,
    }
    if preferred_count is not None:
        fields.update(
            fallback_cluster_shape_mn=tuple(settings.fallback_cluster_shape_mn),
            preferred_cluster_count=preferred_count,
            fallback_cluster_count=fallback_count,
        )
    # Optional kernel fields, passed only when set so the default kernels stay exactly as before.
    if settings.combine_dedup:
        fields["combine_dedup"] = True
    if settings.shared_l2_position is not None:
        fields["shared_l2_position"] = settings.shared_l2_position
    # Kernel options of cutedsl_megamoe main (72796e9, 2026-11-12 dedup knobs), passed only to a kernel that declares
    # them: ImplDesc validation ignores undeclared fields, so an older checkout (441ca70) would silently run without.
    declared = api.Kernel.impl_desc_require()
    for name in ("epilogue_warpgroups", "dedup_copies_per_fill", "dedup_poll_max_sleep_ns",
                 "dedup_epilogue_tail_combine"):
        value = getattr(settings, name)
        if value is None:
            continue
        if name in declared:
            fields[name] = value
        else:
            logger.info("NVFP4 MegaMoE bucket %d: this kernel checkout has no %s; not set", bucket, name)
    try:
        return api.Kernel(problem, api.ImplDesc(fields))
    except ValueError as error:
        mode, hint = fields["schedule_policy"]
        if hint is None or "hint" not in str(error).lower():
            raise
        logger.warning("NVFP4 MegaMoE bucket %d: schedule hint %d refused (%s); using the minimum", bucket, hint,
                       error)
        fields["schedule_policy"] = (mode, None)
        return api.Kernel(problem, api.ImplDesc(fields))


class Nvfp4MegaMoERuntime:
    """Kernel variants, symmetric slab and staging buffers of one EP group on one device."""

    kernel_kind = "general"

    def __init__(self, *, ep_group, num_experts: int, hidden: int, intermediate: int, shared_intermediate: int,
                 topk: int, clamp: float | None, max_num_tokens: int, num_logical_experts: int | None = None) -> None:
        """``num_experts`` is the PHYSICAL count (logical + EPLB redundant), which the kernel addresses and
        splits over the ranks; ``num_logical_experts`` (default: the same) only selects the tuned settings."""
        api = import_kernel()
        os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
        self.api = api
        self.ep_group = ep_group
        self.world_size = ep_group.world_size
        self.rank = ep_group.rank_in_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.hidden, self.intermediate, self.topk = hidden, intermediate, topk
        self.shared_intermediate = shared_intermediate
        self.buckets = token_buckets(max_num_tokens)
        self.max_tokens = self.buckets[-1]
        if num_experts % self.world_size:
            raise ValueError(f"{num_experts} experts do not split over {self.world_size} ranks")
        self.num_experts = num_experts
        self.num_logical_experts = num_experts if num_logical_experts is None else num_logical_experts
        self.num_local_experts = num_experts // self.world_size

        started = time.time()
        counts = _ClusterCounts()
        self.kernels = {}
        self.settings = {}  # bucket -> KernelSettings the kernel was built with (checked by the GPU tests)
        for bucket in self.buckets:
            # Tables by the logical count (settings do not depend on E); the kernel gets the physical count.
            settings = kernel_settings(bucket, hidden, intermediate, self.num_logical_experts, topk)
            self.settings[bucket] = settings
            logger.info("NVFP4 MegaMoE bucket %d (%d physical / %d logical experts, EP%d): %s", bucket,
                        num_experts, self.num_logical_experts, self.world_size, settings)
            kernel = make_kernel(
                api, bucket=bucket, settings=settings, counts=counts, experts=num_experts, hidden=hidden,
                intermediate=intermediate, shared_intermediate=shared_intermediate, topk=topk,
                world_size=self.world_size, clamp=clamp,
            )
            # The router drops ids >= expert_count and the top-k reduce sums every cell unmasked: a kernel built
            # for another expert count (or with the kernel's own helper slots, which assume a fixed home/helper
            # layout EPLB does not keep) would silently mix stale data into the output.
            built = (getattr(kernel, "expert_count", num_experts), getattr(kernel, "local_expert_count",
                     self.num_local_experts), getattr(kernel, "helper_expert_count", 0))
            if built != (num_experts, self.num_local_experts, 0):
                raise RuntimeError(f"NVFP4 MegaMoE bucket {bucket}: kernel built for (experts, local experts, "
                                   f"helpers) {built}, expected {(num_experts, self.num_local_experts, 0)}")
            self.kernels[bucket] = kernel
        sizes = {b: tuple(int(v) for v in k.get_workspace_sizes()) for b, k in self.kernels.items()}
        tmp_sizes = {b: int(k.get_tmp_buffer_bytes()) if hasattr(k, "get_tmp_buffer_bytes") else 0
                     for b, k in self.kernels.items()}
        sf_cols = int(self.kernels[self.max_tokens].token_comm.activation_sf_hidden_padded)
        if sf_cols != hidden // 16:
            raise NotImplementedError(f"activation scale rows padded to {sf_cols} columns (staging writes {hidden // 16})")

        # symmetric slab, carved identically on every rank
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0

        def carve(name: str, nbytes: int) -> None:
            nonlocal cursor
            cursor = round_up(cursor, _SLAB_ALIGN)
            offsets[name] = (cursor, nbytes)
            cursor += nbytes

        carve("act", self.max_tokens * hidden // 2)
        carve("act_sf", self.max_tokens * sf_cols)
        carve("topk_w", self.max_tokens * topk * 4)
        for bucket in self.buckets:
            carve(f"ws{bucket}", sizes[bucket][1])
        total = round_up(cursor, _SLAB_ALIGN)

        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.slab = symm_mem.empty(total, dtype=torch.uint8, device=self.device)
        self.handle = symm_mem.rendezvous(self.slab, group=ep_group.device_group)
        pointers = [int(p) for p in self.handle.buffer_ptrs]
        base = self.slab.data_ptr()
        if len(pointers) != self.world_size or pointers[self.rank] != base:
            raise RuntimeError("symmetric slab rendezvous returned an unexpected peer table")
        self.peer_offsets = tuple(p - base for p in pointers)
        self.slab.zero_()

        def view(name: str) -> torch.Tensor:
            start, nbytes = offsets[name]
            return self.slab.narrow(0, start, nbytes)

        self.act = view("act").view(self.max_tokens, hidden // 2)
        self.act_sf = view("act_sf").view(self.max_tokens, sf_cols).view(torch.float8_e4m3fn)
        self.topk_w = view("topk_w").view(torch.float32).view(self.max_tokens, topk)
        self.shared_ws = {b: view(f"ws{b}") for b in self.buckets}
        self.topk_idx = torch.full((self.max_tokens, topk), -1, dtype=torch.int32, device=self.device)
        self.sh_act = torch.zeros((self.max_tokens, hidden), dtype=torch.float8_e4m3fn, device=self.device)
        self.sh_sf = torch.zeros(round_up(self.max_tokens, 128) * (hidden // 32), dtype=torch.uint8,
                                 device=self.device)
        self.sh_out = torch.zeros((self.max_tokens, hidden), dtype=torch.bfloat16, device=self.device)
        self.local_ws = {b: torch.zeros(sizes[b][0], dtype=torch.uint8, device=self.device) for b in self.buckets}
        # Per-launch data (kernels with a caller-owned tmp buffer): one buffer for every bucket and layer.
        tmp_bytes = max(tmp_sizes.values())
        self.tmp_buffer = (torch.empty(round_up(tmp_bytes, _SLAB_ALIGN), dtype=torch.uint8, device=self.device)
                           if tmp_bytes else None)
        for bucket in self.buckets:
            for ws in (self.local_ws[bucket], self.shared_ws[bucket]):
                if ws.data_ptr() % 128:
                    raise RuntimeError("MegaMoE workspaces must be 128-byte aligned")
        if self.tmp_buffer is not None and self.tmp_buffer.data_ptr() % 128:
            raise RuntimeError("the MegaMoE tmp buffer must be 128-byte aligned")
        self.peer = api.SymmetricBufferHost(
            base_address=base, offsets=self.peer_offsets, rank=self.rank, max_ranks=self.world_size
        )
        # vLLM's DeepGEMM warmup reads ``get_symm_buffer().x.device``
        self.x = self.act
        self._buffer_kwargs = {b: self._make_buffer_kwargs(b) for b in self.buckets}
        self.compiled: dict[int, object] = {}
        torch.cuda.synchronize()
        dist.barrier(group=ep_group.cpu_group)
        logger.info(
            "NVFP4 MegaMoE runtime: buckets %s, symmetric slab %.2f GiB, local workspaces %.2f GiB, tmp buffer "
            "%.2f GiB (%.1f s)",
            self.buckets, total / 2**30, sum(s[0] for s in sizes.values()) / 2**30, tmp_bytes / 2**30,
            time.time() - started,
        )

    # ------------------------------------------------------------------------------------------------------ kwargs
    def _make_buffer_kwargs(self, bucket: int) -> dict:
        fp4 = torch.float4_e2m1fn_x2
        sf_elems = round_up(bucket, 128) * (self.hidden // 32)
        extra = {} if self.tmp_buffer is None else {"tmp_buffer": to_cute_ptr(self.tmp_buffer, assumed_align=128)}
        return {
            **extra,
            "activation": to_cute(self.act[:bucket].view(fp4)),
            "activation_sf": to_cute(self.act_sf[:bucket]),
            "topk_indices": to_cute(self.topk_idx[:bucket]),
            "topk_scores": to_cute(self.topk_w[:bucket], assumed_align=4),
            "local_workspace": to_cute_ptr(self.local_ws[bucket]),
            "shared_workspace": to_cute_ptr(self.shared_ws[bucket]),
            "peer_rank_ptr_mapper_host": self.peer,
            "shared_activation": to_cute(self.sh_act[:bucket]),
            "shared_activation_sf": to_cute(self.sh_sf[:sf_elems].view(torch.float8_e8m0fnu)),
            "shared_output": to_cute(self.sh_out[:bucket], assumed_align=32),
        }

    @staticmethod
    def layer_kwargs(weights) -> dict:
        """CuTe views of one layer's kernel weights and scalars (persistent tensors)."""
        fp4 = torch.float4_e2m1fn_x2
        e4m3, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
        return {
            "fc1_weight": to_cute(weights.w13.view(fp4).permute(0, 2, 1)),
            "fc1_weight_sf": to_cute(weights.w13_sf.view(e4m3)),
            "fc2_weight": to_cute(weights.w2.view(fp4).permute(0, 2, 1)),
            "fc2_weight_sf": to_cute(weights.w2_sf.view(e4m3)),
            "fc1_alpha": to_cute(weights.fc1_alpha, assumed_align=4),
            "fc2_alpha": to_cute(weights.fc2_alpha, assumed_align=4),
            "fc1_norm_const": to_cute(weights.fc1_norm_const, assumed_align=4),
            "shared_fc1_weight": to_cute(weights.shared_w13.view(e4m3).unsqueeze(0).permute(0, 2, 1)),
            "shared_fc1_weight_sf": to_cute(weights.shared_w13_sf.view(e8m0).unsqueeze(0)),
            "shared_fc2_weight": to_cute(weights.shared_w2.view(e4m3).unsqueeze(0).permute(0, 2, 1)),
            "shared_fc2_weight_sf": to_cute(weights.shared_w2_sf.view(e8m0).unsqueeze(0)),
        }

    # --------------------------------------------------------------------------------------------------- compile
    def compile_all(self, layer_kwargs: dict) -> None:
        """Compile every bucket variant (eagerly, before any CUDA graph capture)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NVFP4 MegaMoE kernels must be compiled before CUDA graph capture")
        import torch.distributed as dist

        for bucket in self.buckets:
            if bucket in self.compiled:
                continue
            started = time.time()
            y = torch.empty((bucket, self.hidden), dtype=torch.bfloat16, device=self.device)
            kwargs = {**self._buffer_kwargs[bucket], **layer_kwargs, "output_activation": to_cute(y, assumed_align=32),
                      "stream": current_cu_stream()}
            self.compiled[bucket] = self.api.cute.compile(self.kernels[bucket], **kwargs)
            logger.info("NVFP4 MegaMoE: compiled %s in %.1f s", self.kernels[bucket].name()[:96],
                        time.time() - started)
        torch.cuda.synchronize()
        dist.barrier(group=self.ep_group.cpu_group)

    # ------------------------------------------------------------------------------------------------------- call
    def bucket(self, num_tokens: int) -> int:
        return bucket_for(self.buckets, num_tokens)

    def run(self, x, topk_ids, topk_weights, is_padding, *, bucket: int, layer_kwargs: dict, norm: float,
            scale_mul: float, eplb: tuple | None = None) -> torch.Tensor:
        """Stage ``x`` and launch the ``bucket`` variant; returns ``[bucket, H]`` BF16 (routed + shared).

        ``eplb``: the layer's ``eplb.eplb_kernel_args``; staging maps ``topk_ids`` (logical) to physical
        replicas and records the expert load. None or inert: the ids are passed through.
        """
        staging.stage_nvfp4_mxfp8(
            x, topk_ids, topk_weights, is_padding, norm=norm, scale_mul=scale_mul, rows=bucket,
            act=self.act, act_sf=self.act_sf, shared_act=self.sh_act, shared_act_sf=self.sh_sf,
            ids_out=self.topk_idx, weights_out=self.topk_w, eplb=eplb,
        )
        y = torch.empty((bucket, self.hidden), dtype=torch.bfloat16, device=x.device)
        kwargs = {**self._buffer_kwargs[bucket], **layer_kwargs, "output_activation": to_cute(y, assumed_align=32),
                  "stream": current_cu_stream()}
        self.compiled[bucket](**kwargs)
        return y
