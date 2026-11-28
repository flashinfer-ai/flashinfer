# SPDX-License-Identifier: Apache-2.0
"""Decode servers: NVIDIA's generation-phase MegaMoE kernel (``BlockScaledSwapAbGenphaseMoeKernel``) in place of the
general one.

A server whose token budget is at most 1024 tokens per rank (the disaggregated decode servers) can run cutedsl_megamoe
main's generation-phase kernel instead of ``BlockScaledSwapAbMegaMoeKernel`` (``VLLM_NVFP4_MEGAMOE_DECODE_KERNEL=genphase``,
see config.py). It takes the same operands (weights, scalars, staged NVFP4 tokens, MXFP8 shared-expert input, top-k
ids and scores, symmetric and local workspaces) without a tmp buffer, and launches its own communication grid ahead
of the main kernel (programmatic dependent launch), so one compiled call per bucket replaces the general kernel's.

What differs from ``Nvfp4MegaMoERuntime`` is only how the kernels are built:

* buckets are powers of two up to 1024 (``config.genphase_buckets``), the keys of the kernel checkout's tuned tables;
* every bucket takes the checkout's tuned decode tactic for DeepSeek-V4.1-Flash with the fused shared expert
  (``repo_internal_only/genphase_tactics.py``, preset ``dsv4_1_flash_genphase``, keyed on this EP size and
  ``VLLM_NVFP4_MEGAMOE_DECODE_ROUTE``, default balanced), as the checkout's runner
  ``test_genphase_megamoe_rubin.py --tuned ... --shared_expert`` builds it;
* ``reduce_shared_output``: the separate top-k reduce adds the shared expert's output into the layer output (the
  plugin returns routed + shared in one tensor), so the top-k reduce never runs inside the kernel;
* the routing weights are applied at FC1, before the FC2-input requantization (``pre_fc1_quant``), whatever the
  tuned entry's position: the checkpoint's static FC2-input global scale (w2 ``input_scale``) is calibrated for
  weighted FC1 outputs, as vLLM and the general kernel produce them. ``post_fc2`` would quantize unweighted
  activations against it (tests/nvfp4_gpu_check.py: rel-L2 0.113 instead of 2.4e-3 at bucket 16, EP4, job 1121600).
  ``VLLM_NVFP4_MEGAMOE_DECODE_TOPK_POSITION=tuned`` restores the tuned positions (experiments only).

Staging, the symmetric slab, the weight views, compilation and the call are the parent's. The kernel's activation
planes must have the staging's row stride (H/2 bytes of tokens, H/16 bytes of scales); a checkout that pads them is
refused rather than mis-staged.
"""

import os
import time
import types

import torch

from .submission_config import genphase_buckets, round_up
from .submission_runtime import _SLAB_ALIGN, Nvfp4MegaMoERuntime, _ClusterCounts, import_kernel, logger

TUNED_PRESET = "dsv4_1_flash_genphase"
TUNED_SHAPE = (5120, 2304, 384, 6)  # hidden, intermediate, logical experts, top-k of the preset
TUNED_EPS = (4, 8, 16, 32)
ROUTE_ENV = "VLLM_NVFP4_MEGAMOE_DECODE_ROUTE"
POSITION_ENV = "VLLM_NVFP4_MEGAMOE_DECODE_TOPK_POSITION"


from ..submission_6cb79ec1 import RubinInferenceGenphaseMegaMoE
from ..submission_6cb79ec1 import genphase_tactics as tactics

FUSED_ONLY_KNOBS = ("shared_l2_position", "shared_work_id_mode", "shared_fc1_swizzle")


class GenphaseRuntime(Nvfp4MegaMoERuntime):
    """Generation-phase kernel variants (one per bucket), symmetric slab and staging buffers of one EP group."""

    kernel_kind = "genphase"

    def __init__(self, *, ep_group, num_experts: int, hidden: int, intermediate: int, shared_intermediate: int,
                 topk: int, clamp: float | None, max_num_tokens: int, num_logical_experts: int | None = None) -> None:
        api = import_kernel()
        os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
        Kernel = RubinInferenceGenphaseMegaMoE
        self.api = api
        self.ep_group = ep_group
        self.world_size = ep_group.world_size
        self.rank = ep_group.rank_in_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.hidden, self.intermediate, self.topk = hidden, intermediate, topk
        self.shared_intermediate = shared_intermediate
        self.buckets = genphase_buckets(max_num_tokens)
        self.max_tokens = self.buckets[-1]
        if num_experts % self.world_size:
            raise ValueError(f"{num_experts} experts do not split over {self.world_size} ranks")
        self.num_experts = num_experts
        self.num_logical_experts = num_experts if num_logical_experts is None else num_logical_experts
        self.num_local_experts = num_experts // self.world_size
        if (hidden, intermediate, self.num_logical_experts, topk) != TUNED_SHAPE:
            raise NotImplementedError(f"NVFP4 MegaMoE decode kernel: tuned for {TUNED_SHAPE} (H, I, logical experts, "
                                      f"top-k), got {(hidden, intermediate, self.num_logical_experts, topk)}")
        if self.world_size not in TUNED_EPS:
            raise NotImplementedError(f"NVFP4 MegaMoE decode kernel: tuned for EP {TUNED_EPS}, got EP{self.world_size}")
        if shared_intermediate != intermediate:
            raise NotImplementedError("NVFP4 MegaMoE decode kernel: one shared expert of the routed width expected")
        route = os.environ.get(ROUTE_ENV, "balanced").strip()
        position_mode = os.environ.get(POSITION_ENV, "pre_fc1_quant").strip()
        if position_mode not in ("pre_fc1_quant", "tuned"):
            raise ValueError(f"{POSITION_ENV} must be pre_fc1_quant or tuned, got {position_mode!r}")

        started = time.time()
        counts = _ClusterCounts()
        # The runner's launch_cluster_count for the default (4, 1) cluster: every co-resident cluster.
        cluster_m = Kernel.default_cluster_shape_mn[0]
        launch_clusters = counts.max_active(cluster_m * Kernel.default_cluster_shape_mn[1])
        instruction_k = api.QuantKind("nvfp4").instruction_k("2x")
        self.kernels, self.settings = {}, {}
        for bucket in self.buckets:
            key = dict(max_tokens_per_rank=bucket, ep=self.world_size, route=route)
            tactic, _reduce_in_kernel, tuned_position = tactics.resolve_tuned_tactic(TUNED_PRESET, shared=True, **key)
            position = tuned_position if position_mode == "tuned" else "pre_fc1_quant"
            tile, schedule, work_id_mode, fc2_stages, epi_flags = tactic
            knobs = tactics.resolve_shared_knobs(TUNED_PRESET, **key)
            shared_knobs = {k: v for k, v in knobs.items()
                            if k in FUSED_ONLY_KNOBS}
            problem = api.ProblemDesc({
                "expert_count": num_experts,
                "intermediate_gateup_size": 2 * intermediate,
                "hidden_size": hidden,
                "quant_kind": "nvfp4",
                "a_major_mode": api.OperandMajorMode.K,
                "b_major_mode": api.OperandMajorMode.K,
                "combine_format": api.CombineFormat.parse("bf16"),
                "gate_up_clamp": clamp,
                "situ_beta": None,
                "situ_linear_beta": None,
                "world_size": self.world_size,
                "topk": topk,
                "topk_index_dtype": api.cutlass.Int32,
                "max_tokens_per_rank": bucket,
                "apply_topk_at_fc1": position == "pre_fc1_quant",
                "shared_expert": True,
                "shared_quant_kind": "mxfp8_e4m3",
                "shared_hidden_size": hidden,
                "shared_intermediate_gateup_size": 2 * shared_intermediate,
                "shared_output_size": hidden,
                "reduce_shared_output": True,
            })
            impl = {
                "mma_instruction_mnk": (tile[0], tile[1], instruction_k),
                "mma_tiler_mnk": tuple(tile),
                "use_2cta_instrs": tile[0] == 256,
                "cluster_shape_mn": tuple(Kernel.default_cluster_shape_mn),
                "schedule_policy": tuple(schedule),
                "token_padding_block": 64,
                "sf_padding_block": 128,
                "work_id_mode": "atomic_counter",  # the decode scheduler claims from atomic counters only
                "fc2_use_bulk": True,
                **({"fc2_tma_stages": fc2_stages} if fc2_stages is not None else {}),
                "epi_flag_batches": tuple(epi_flags),
                "launch_cluster_count": launch_clusters,
                "reduce_topk_in_kernel": False,  # the separate reduce adds the shared output
                "claim_gate_on_tma_a": tactics.resolve_claim_gate(TUNED_PRESET, shared=True, **key),
                **({"sf_plain_transport": True}
                   if tactics.resolve_sf_plain_transport(TUNED_PRESET, shared=True, **key) else {}),
                **shared_knobs,
            }
            if schedule[0] != "decode" or work_id_mode != "atomic_counter":
                raise RuntimeError(f"tuned gen-phase entry for bucket {bucket} is not a decode tactic: {tactic}")
            kernel = Kernel(problem, api.ImplDesc(impl))
            built = (getattr(kernel, "expert_count", num_experts), getattr(kernel, "local_expert_count",
                     self.num_local_experts), getattr(kernel, "helper_expert_count", 0))
            if built != (num_experts, self.num_local_experts, 0):
                raise RuntimeError(f"NVFP4 MegaMoE decode bucket {bucket}: kernel built for (experts, local experts, "
                                   f"helpers) {built}, expected {(num_experts, self.num_local_experts, 0)}")
            if not getattr(kernel, "shared_expert", False):
                raise RuntimeError("this cutedsl_megamoe checkout's gen-phase kernel does not fuse the shared expert")
            comm = kernel.token_comm
            rows = (int(comm.plain_activation_row_elements), int(comm.plain_activation_sf_row_elements))
            if rows != (hidden, hidden // 16):
                raise NotImplementedError(f"gen-phase activation rows padded to {rows} elements (staging writes "
                                          f"{(hidden, hidden // 16)})")
            self.kernels[bucket] = kernel
            # Attribute access as KernelSettings (the GPU tests read combine_dedup); the gen-phase kernel has no dedup.
            self.settings[bucket] = types.SimpleNamespace(**impl, apply_topk_at_fc1=position == "pre_fc1_quant",
                                                          tuned_position=tuned_position, route=route,
                                                          combine_dedup=False)
            logger.info("NVFP4 MegaMoE decode bucket %d (gen-phase, %d physical / %d logical experts, EP%d, %s): %s",
                        bucket, num_experts, self.num_logical_experts, self.world_size, route, self.settings[bucket])
        sizes = {b: tuple(int(v) for v in k.get_workspace_sizes()) for b, k in self.kernels.items()}

        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0

        def carve(name: str, nbytes: int) -> None:
            nonlocal cursor
            cursor = round_up(cursor, _SLAB_ALIGN)
            offsets[name] = (cursor, nbytes)
            cursor += nbytes

        sf_cols = hidden // 16
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
        # Zeroed once: every launch's tail resets the counters it used (the kernel's workspace contract).
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
        self.tmp_buffer = None
        for bucket in self.buckets:
            for ws in (self.local_ws[bucket], self.shared_ws[bucket]):
                if ws.data_ptr() % 128:
                    raise RuntimeError("MegaMoE workspaces must be 128-byte aligned")
        if self.topk_idx.data_ptr() % 16:
            raise RuntimeError("the routing index tensor must be 16-byte aligned")
        self.peer = api.SymmetricBufferHost(
            base_address=base, offsets=self.peer_offsets, rank=self.rank, max_ranks=self.world_size
        )
        self.x = self.act
        self._buffer_kwargs = {b: self._make_buffer_kwargs(b) for b in self.buckets}
        self.compiled: dict[int, object] = {}
        torch.cuda.synchronize()
        dist.barrier(group=ep_group.cpu_group)
        logger.info("NVFP4 MegaMoE decode runtime (gen-phase): buckets %s, symmetric slab %.2f GiB, local "
                    "workspaces %.2f GiB (%.1f s)", self.buckets, total / 2**30,
                    sum(s[0] for s in sizes.values()) / 2**30, time.time() - started)
