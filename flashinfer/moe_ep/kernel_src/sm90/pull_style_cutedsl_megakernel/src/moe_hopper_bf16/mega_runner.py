# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone multi-rank MegaMoE host driver for the BF16 GLU path.

Mirror of ``moe_nvfp4_swapab.mega_runner`` (NVFP4) specialised for BF16, with
selectable non-swap and swap-AB kernels.
It subclasses the NVFP4 ``MegaMoETester`` and reuses all of its distributed
bootstrap, routing-table generation, symmetric-heap allocation, workspace
allocation, validation and teardown; only the three kind-specific stages are
overridden:

  * ``generate_inputs``   -- bf16 input + weight generation + sym staging
  * ``compute_reference`` -- ``compute_megamoe_reference_bf16``
  * ``run_kernel``        -- instantiate ``Sm90MegaMoEBf16Kernel``

Topk weighting follows the NVFP4/MXFP8 compute graphs. ``deepgemm`` folds each
routing weight into the SwiGLU output before the BF16 FC1-output store;
``transformers`` keeps the staged FC2 terms unweighted and applies routing
weights in the standalone ``TopkReduce`` kernel. In-kernel reduce
(``--in_kernel_fc2_reduce``) requires ``deepgemm``.

Launcher::

    torchrun --nproc_per_node=4 -m moe_hopper_bf16.mega_runner \\
        --kind bf16 --num_total_experts 32 --route_distribution balanced
"""

import argparse
import gc
import os
import sys
from typing import List, Optional

import torch

## TODO: some common modules currently live in moe_nvfp4_swapab; these path
## dependencies can be removed once the modules move to a shared package.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PKG_DIR)
_NVFP4_DIR = os.path.join(_PARENT_DIR, "moe_nvfp4_swapab")
for _p in (_PARENT_DIR, _NVFP4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.host_utils import compare_and_report_mismatches
from tester.host_utils import (
    reduce_add_deterministic_check_dim_size_limit,
    reduce_add_ordering_match,
)
from moe_nvfp4_swapab.mega_runner import (
    TokenCommProblemDesc,
    MiscDesc,
    MegaMoETester,
    _generate_topk_idx_balanced,
    _generate_topk_idx_power_law,
    _generate_topk_weights,
    _print_remote_rank_comm_matrices,
    _sym_zeros,
    _compute_peer_offsets,
    _parse_tuple,
    _parse_output_dtype,
    _NO_DIST,
)
from moe_hopper_bf16.runner_fc12 import ImplDesc
from moe_hopper_bf16.mega_reference_bf16 import compute_megamoe_reference_bf16
from moe_hopper_bf16.hopper_moe_utils import (
    BF16_KIND_CHOICES,
    bf16_kind_to_cutlass_dtype,
    bf16_kind_to_torch_dtype,
    create_bf16_tensor,
)
from moe_hopper_bf16.heuristic_config import resolve_hopper_bf16_config


# =============================================================================
# BF16 MegaMoE tester
# =============================================================================


class MegaMoEBf16Tester(MegaMoETester):
    """BF16 specialisation of the multi-rank MegaMoE host driver."""

    def __init__(
        self,
        problem: TokenCommProblemDesc,
        impl: ImplDesc,
        misc: MiscDesc,
        *,
        rank: int,
        kind: str = "bf16",
        swap_ab: bool = False,
        pingpong: bool = False,
        use_cuda_profiler_api: bool = False,
    ) -> None:
        self.swap_ab = swap_ab
        self.pingpong = pingpong
        self._use_cuda_profiler_api = use_cuda_profiler_api
        super().__init__(problem, impl, misc, rank=rank)
        if kind not in BF16_KIND_CHOICES:
            raise ValueError(
                f"kind must be one of {list(BF16_KIND_CHOICES)}, got {kind!r}."
            )
        self.kind = kind
        self.torch_ab_dtype = bf16_kind_to_torch_dtype(kind)

    # ------------------------------------------------------------------
    # Step 1: deterministic BF16 input + weight generation
    # ------------------------------------------------------------------

    def generate_inputs(self) -> None:
        problem = self.problem
        rng = self._np_rng
        num_ranks = problem.world_size
        num_tokens_per_rank = problem.num_tokens_per_rank
        num_topk = problem.num_topk
        hidden = problem.hidden
        intermediate = problem.intermediate
        intermediate_downproj = intermediate // 2
        num_experts_per_rank = problem.num_experts_per_rank
        nonzero_pct = float(
            os.environ.get("BF16_NONZERO_PCT", "1")
        ) / 100.0

        # ---- Activation (bf16).  There is no scale-factor leg: BF16 operands
        # carry their real values, so the dispatch warps move data only.
        self._global_activation = create_bf16_tensor(
            (num_ranks, num_tokens_per_rank, hidden),
            perf_run=self.misc.perf_run,
            positive_prob=nonzero_pct,
            negative_prob=nonzero_pct,
            generator=self._torch_cuda_rng,
            perf_positive_only=True,
        )

        # ---- Routing table.
        if problem.route_distribution == "balanced":
            topk_idx_np = _generate_topk_idx_balanced(
                num_ranks, num_tokens_per_rank, num_topk,
                problem.num_total_experts, rng,
            )
        else:
            topk_idx_np = _generate_topk_idx_power_law(
                num_ranks, num_tokens_per_rank, num_topk,
                problem.num_total_experts, problem.power_law_exponent, rng,
            )
        topk_weights = _generate_topk_weights(
            num_ranks, num_tokens_per_rank, num_topk, self._torch_cuda_rng,
        )
        if self.rank == 0:
            _print_remote_rank_comm_matrices(
                topk_idx_np, num_ranks, problem.num_total_experts,
            )
        self._global_topk_idx = torch.from_numpy(topk_idx_np).cuda()
        self._global_topk_weights = topk_weights

        # ---- Weights.  fc1: logical (experts, intermediate, hidden) permuted
        # to (experts, hidden, intermediate) with hidden stride-1 (K-major).
        # fc2: logical (experts, hidden, inter//2) permuted to
        # (experts, inter//2, hidden) with inter//2 stride-1.
        self._global_fc1_weight = create_bf16_tensor(
            (num_ranks, num_experts_per_rank, intermediate, hidden),
            perf_run=self.misc.perf_run,
            positive_prob=nonzero_pct,
            negative_prob=nonzero_pct,
            generator=self._torch_cuda_rng,
            perf_positive_only=True,
        ).permute(0, 1, 3, 2)
        self._global_fc2_weight = create_bf16_tensor(
            (num_ranks, num_experts_per_rank, hidden, intermediate_downproj),
            perf_run=self.misc.perf_run,
            positive_prob=nonzero_pct,
            negative_prob=nonzero_pct,
            generator=self._torch_cuda_rng,
            perf_positive_only=True,
        ).permute(0, 1, 3, 2)

        # ---- Stage own-rank inputs into the symmetric heap.
        own_activation = self._global_activation[self.rank]
        own_topk_idx = self._global_topk_idx[self.rank]
        own_topk_weights = self._global_topk_weights[self.rank]

        self.my_activation = _sym_zeros(
            (num_tokens_per_rank, hidden), torch.bfloat16,
        )
        self.my_activation.copy_(own_activation)

        self.my_topk_idx = _sym_zeros(tuple(own_topk_idx.shape), torch.int64)
        self.my_topk_idx.copy_(own_topk_idx)

        self.my_topk_weights = _sym_zeros(tuple(own_topk_weights.shape), torch.float32)
        self.my_topk_weights.copy_(own_topk_weights)

        # ---- Own-rank weights stay on regular cuda.  DO NOT ``.contiguous()``:
        # the permute above puts the K dim mid-tensor (stride-1); contiguity
        # would re-pack to row-major and break the K-as-stride-1 invariant the
        # GEMM path depends on.
        self.my_fc1_weight = self._global_fc1_weight[self.rank]
        self.my_fc2_weight = self._global_fc2_weight[self.rank]

        # ---- Public final output. The per-topk (T, K, H) combine plane is an
        # internal shared-workspace region in separate-reduce mode.
        if problem.fc2_output_dtype != torch.bfloat16:
            raise ValueError(
                "the Hopper combine (REDG / cp.reduce push / separate "
                f"TopkReduce) is BF16-only, got {problem.fc2_output_dtype}."
            )
        if self.impl.in_kernel_fc2_reduce:
            # In-kernel reduce (epi_warps REDG or dispatch cp.reduce push)
            # accumulates across ranks from zero, so the output must live on
            # the symmetric heap.
            self.output_activation = _sym_zeros(
                (num_tokens_per_rank, hidden), problem.fc2_output_dtype,
            )
        else:
            self.output_activation = torch.empty(
                (num_tokens_per_rank, hidden),
                dtype=problem.fc2_output_dtype,
                device="cuda",
            )

        torch.cuda.synchronize()
        self._check_cuda_rng_consistency()

    # ------------------------------------------------------------------
    # Step 2: BF16 reference
    # ------------------------------------------------------------------

    def compute_reference(self) -> None:
        if self.misc.skip_ref_check:
            return
        if self._global_activation is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")

        ref_result = compute_megamoe_reference_bf16(
            input_activation=self._global_activation,
            input_topk_idx=self._global_topk_idx,
            input_topk_weights=self._global_topk_weights,
            fc1_weight=self._global_fc1_weight,
            fc2_weight=self._global_fc2_weight,
            norm_const=1.0,
            ref_compute_graph=self.misc.ref_compute_graph,
            fc2_output_dtype=self.problem.fc2_output_dtype,
            gate_up_clamp=self.problem.gate_up_clamp,
            return_fc1_gateup=self.impl.generate_c,
        )

        if self.impl.generate_c:
            combine_ref_global, fc1_gateup_global = ref_result
            expert_start = self.rank * self.problem.num_experts_per_rank
            self._ref_fc1_gateup_per_expert = {
                e: fc1_gateup_global.get(expert_start + e)
                for e in range(self.problem.num_experts_per_rank)
            }
        else:
            combine_ref_global = ref_result
            self._ref_fc1_gateup_per_expert = None

        self.combine_output_ref = combine_ref_global[self.rank].contiguous()

    # ------------------------------------------------------------------
    # Step 5: BF16 validation
    # ------------------------------------------------------------------

    def _validate_c_output(self) -> None:
        """Compare kernel ``fc1_c`` vs the reference pre-SwiGLU gate+up.

        Rows are compared positionally after a keyed permutation.  The
        multi-rank dispatch order of pool rows is non-deterministic, but each
        pool row's source coordinates are recorded by the dispatch warps in
        the ``token_src_metadata`` workspace region (one packed
        ``(src_rank, src_token, src_topk)`` Int64 per row -- see
        ``src.token_comm.TokenSrcMetadata``).  The reference rows are produced
        in (rank, token, topk) row-major order, so keying both sides by
        ``(rank * tokens_per_rank + token) * num_topk + topk`` aligns them.
        """
        if not self.impl.generate_c:
            return
        if self.misc.skip_ref_check:
            return
        c = getattr(self, "_c_output", None)
        if c is None:
            if self.rank == 0:
                print("[generate_c] c_output not allocated -- skipped.")
            return
        ref_map = getattr(self, "_ref_fc1_gateup_per_expert", None)
        if not ref_map:
            if self.rank == 0:
                print("[generate_c] reference fc1 gate+up not available -- skipped.")
            return

        valid = self._c_valid_tokens_per_expert
        doff = self._c_data_physical_offsets

        kernel = self._kernel
        md_off = kernel._local_offsets["token_src_metadata"]
        pool_cap = kernel.pool_token_capacity
        metadata = (
            self.local_workspace[md_off : md_off + pool_cap * 8]
            .view(torch.int64)
            .cpu()
        )
        tpb = kernel.token_padding_block
        num_topk = self.problem.num_topk
        tokens_per_rank = self.problem.num_tokens_per_rank
        topk_cpu = self._global_topk_idx.cpu()

        print(f"\n{'=' * 60}")
        print(
            f"[generate_c][rank{self.rank}] kernel c_output vs reference "
            f"fc1 gate+up:"
        )
        any_checked = False
        pool_base = 0
        for e in range(self.problem.num_experts_per_rank):
            v_e = valid[e]
            ref = ref_map.get(e)
            expert_pool_base = pool_base
            pool_base += -(-v_e // tpb) * tpb
            if v_e == 0 or ref is None:
                continue
            any_checked = True

            # Kernel row i of expert e is pool row (expert_pool_base + i);
            # key each row by its (rank, token, topk) source coordinate.
            packed = metadata[expert_pool_base : expert_pool_base + v_e]
            src_token = packed & 0xFFFFFFFF
            hi = packed >> 32
            src_rank = (hi >> 16) & 0xFFFF
            src_topk = hi & 0xFFFF
            kernel_keys = (
                src_rank * tokens_per_rank + src_token
            ) * num_topk + src_topk

            global_expert = self.rank * self.problem.num_experts_per_rank + e
            routed = (topk_cpu == global_expert).nonzero(as_tuple=False)
            ref_keys = (
                routed[:, 0] * tokens_per_rank + routed[:, 1]
            ) * num_topk + routed[:, 2]

            # The routing-table keys are unique, so sorted kernel keys must
            # equal the reference key list element-wise; that simultaneously
            # checks equal counts, both-side membership, and that no pool row
            # duplicates another row's source coordinate.
            if not torch.equal(torch.sort(kernel_keys).values, ref_keys):
                raise RuntimeError(
                    f"[generate_c][rank{self.rank}] expert {e}: pool metadata "
                    f"keys do not biject onto the routing table "
                    f"(kernel rows={v_e}, reference rows={ref_keys.numel()})."
                )
            pos = torch.searchsorted(ref_keys, kernel_keys)

            # ``compare_and_report_mismatches`` raises on the first failing
            # expert, so a mismatch fails the run.
            compare_and_report_mismatches(
                c[doff[e] : doff[e] + v_e].to(torch.float32).cpu(),
                ref.to(torch.float32).cpu()[pos],
                name=f"c_output[rank{self.rank}]expert{e}",
                atol=1e-2,
                rtol=1e-2,
                max_mismatches=5,
            )
        if not any_checked:
            print("  (no valid tokens routed to any local expert)")
        print("=" * 60)

    def validate(self) -> None:
        """Compare the public 2D output against the topk-reduced reference."""
        self._validate_c_output()
        if self.misc.skip_ref_check:
            return
        if self.output_activation is None:
            raise RuntimeError("validate requires run_kernel first.")
        if self.combine_output_ref is None:
            raise RuntimeError("validate requires compute_reference first.")

        actual_reduced = self.output_activation.to(torch.float32)
        ref_terms = self.combine_output_ref.to(torch.float32)
        if self.misc.ref_compute_graph == "transformers":
            ref_terms = ref_terms * self._global_topk_weights[
                self.rank, :, :, None
            ].to(torch.float32)
        ref_reduced = ref_terms.sum(dim=1)

        atol = 1e-2
        rtol = 1e-2
        if (
            self.impl.in_kernel_fc2_reduce
            and not torch.allclose(actual_reduced, ref_reduced, atol=atol, rtol=rtol)
        ):
            ref_terms = self.combine_output_ref.to(torch.bfloat16)
            num_topk = ref_terms.shape[1]
            if num_topk <= reduce_add_deterministic_check_dim_size_limit:
                match_mask, num_orderings = reduce_add_ordering_match(
                    actual_reduced, ref_terms,
                )
                if bool(match_mask.all().item()):
                    if self.rank == 0:
                        print(
                            "Validation PASSED: in-kernel reduce output "
                            "matches a legal BF16 atomic-add ordering "
                            f"({num_orderings} orderings)."
                        )
                    return
            else:
                unit_roundoff = torch.finfo(torch.bfloat16).eps / 2.0
                gamma = (
                    (num_topk - 1) * unit_roundoff
                    / (1.0 - (num_topk - 1) * unit_roundoff)
                )
                exact = ref_terms.sum(dim=1, dtype=torch.float32)
                bound = gamma * ref_terms.abs().sum(dim=1, dtype=torch.float32)
                if not bool(((actual_reduced - exact).abs() > bound).any().item()):
                    if self.rank == 0:
                        print(
                            "Validation PASSED: in-kernel reduce output is "
                            "within the BF16 atomic-add roundoff envelope."
                        )
                    return

        # bf16-grade tolerance; the K-axis fp32 sum adds at most ~K bf16 ULPs,
        # well below this band for the v1 configurations.
        # TODO: replace to silent check numerical dist when kernel stable.
        compare_and_report_mismatches(
            actual_reduced,
            ref_reduced,
            name=f"output_activation[rank{self.rank}]",
            atol=atol,
            rtol=rtol,
        )

    # ------------------------------------------------------------------
    # Step 4: BF16 kernel launch
    # ------------------------------------------------------------------

    def run_kernel(self) -> None:
        """Compile + launch ``Sm90MegaMoEBf16Kernel`` on the current stream.

        Mirrors ``MegaMoETester.run_kernel`` step-for-step; only the kernel
        instantiation (class + ``ab_dtype``) is Hopper-specific.
        """
        if (
            self.my_activation is None
            or self.my_topk_idx is None
            or self.my_topk_weights is None
            or self.my_fc1_weight is None
            or self.my_fc2_weight is None
            or self.output_activation is None
        ):
            raise RuntimeError("run_kernel requires generate_inputs first.")

        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        from moe_hopper_bf16.megamoe_kernel_bf16 import (
            Sm90MegaMoEBf16Kernel,
            Sm90MegaMoESwapABBf16Kernel,
        )
        from src.sym_buffer import SymBufferHost

        # -- 1. Kernel instance (Hopper requires static_expert_shape != None) --
        static_expert_shape = (
            self.problem.num_experts_per_rank,
            self.problem.intermediate,
            self.problem.hidden,
        )

        cluster_size = (
            self.impl.cluster_shape_mnk[0] * self.impl.cluster_shape_mnk[1]
        )
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
            cluster_size
        )
        group_hint = self.impl.group_hint
        if group_hint is None:
            group_hint = max_active_clusters

        Kernel = (
            Sm90MegaMoESwapABBf16Kernel
            if self.swap_ab
            else Sm90MegaMoEBf16Kernel
        )
        # Keep the dispatch pool and scheduler on the physical token tile: M
        # for the native layout and N after swapping A/B.
        token_padding_block = (
            self.impl.mma_tiler_mnk[1]
            if self.swap_ab
            else self.impl.mma_tiler_mnk[0]
        )
        self._kernel = Kernel(
            mma_tiler_mnk=self.impl.mma_tiler_mnk,
            cluster_shape_mnk=self.impl.cluster_shape_mnk,
            use_2cta_instrs=self.impl.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            load_balance_mode=self.impl.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=self.impl.force_static_sched,
            clc_bundle_size=self.impl.clc_bundle_size,
            num_sched_stages=self.impl.num_sched_stages,
            ab_dtype=bf16_kind_to_cutlass_dtype(self.kind),
            pingpong=self.pingpong,
            world_size=self.world_size,
            local_rank=self.rank,
            num_topk=self.problem.num_topk,
            max_tokens_per_rank=self.problem.num_tokens_per_rank,
            hidden=self.problem.hidden,
            fc2_in_kernel_topk_reduce=self.impl.in_kernel_fc2_reduce,
            apply_topk_in_fc1=self.misc.ref_compute_graph == "deepgemm",
            token_back_mode=self.impl.token_back_mode,
            epi_flag_batch=self.impl.epi_flag_batch,
            flag_batch=self.impl.flag_batch,
            gate_up_clamp=self.problem.gate_up_clamp,
            generate_c=self.impl.generate_c,
            tail_split_pairs=self.impl.tail_split_pairs,
        )

        # -- 1b. generate_c: raw gate+up output tensor + per-expert offsets --
        #
        # Row space must match the kernel's per-expert pool exactly: the
        # scheduler's ``cumulative_data_physical_row`` is the running sum of
        # ``round_up(valid_tokens_e, token_padding_block)``, so use the same
        # padding block the kernel was constructed with (not a fixed 128).
        self._c_output = None
        self._c_valid_tokens_per_expert = None
        self._c_data_physical_offsets = None
        if self.impl.generate_c:
            expert_start = self.rank * self.problem.num_experts_per_rank
            valid_tokens = [
                int((self._global_topk_idx == expert_start + e).sum().item())
                for e in range(self.problem.num_experts_per_rank)
            ]
            doff = [0]
            for v in valid_tokens:
                doff.append(doff[-1] + -(-v // token_padding_block)
                            * token_padding_block)
            tokens_sum = max(1, doff[-1])
            # ``problem.intermediate`` is the full gate+up width here.
            self._c_output = torch.zeros(
                (tokens_sum, self.problem.intermediate),
                dtype=torch.bfloat16, device="cuda",
            )
            self._c_valid_tokens_per_expert = valid_tokens
            self._c_data_physical_offsets = doff[:-1]

        # -- 2. Workspaces (local cuda + sym-heap) --
        self.allocate_workspaces()

        # -- 3. Torch -> cute --
        def _to_cute(tensor: torch.Tensor, assumed_align: int = 16, force_static_layout=False):
            cute_tensor = cutlass_torch.from_dlpack(
                tensor, assumed_align=assumed_align,
            )
            if force_static_layout:
                return cute_tensor
            leading_dim = cutlass_torch.get_leading_dim(tensor)
            return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

        activation_cute = _to_cute(self.my_activation)
        topk_idx_cute = _to_cute(self.my_topk_idx)
        topk_weights_cute = _to_cute(self.my_topk_weights)
        fc1_weight_cute = _to_cute(self.my_fc1_weight)
        fc2_weight_cute = _to_cute(self.my_fc2_weight)
        output_activation_cute = _to_cute(self.output_activation)

        # The internal combine plane can push shared_workspace beyond 2 GiB.
        # Pass opaque workspaces as raw pointers so no 32-bit tensor shape is
        # materialized; the wrapper partitions them with Int64 byte offsets.
        from cutlass.cute.typing import AddressSpace as _AddressSpace

        def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 16):
            return cute.runtime.make_ptr(
                cutlass.Uint8,
                tensor.data_ptr(),
                _AddressSpace.gmem,
                assumed_align=assumed_align,
            )

        local_workspace_cute = _to_cute_ptr(self.local_workspace)
        shared_workspace_cute = _to_cute_ptr(self.shared_workspace)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # -- 4. cute.compile --
        peer_rank_ptr_mapper_host = SymBufferHost(
            base_addr=self.symmetric_base,
            offsets=tuple(self.peer_offsets_list),
            rank_idx=self.rank,
            num_max_ranks=self.world_size,
        )

        runtime_kwargs = dict(
            activation=activation_cute,
            topk_idx=topk_idx_cute,
            topk_weights=topk_weights_cute,
            fc1_weight=fc1_weight_cute,
            fc2_weight=fc2_weight_cute,
            output_activation=output_activation_cute,
            local_workspace=local_workspace_cute,
            shared_workspace=shared_workspace_cute,
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )
        # ``fc1_c`` is a positional kernel argument (Blackwell training ABI):
        # always present, None on the inference path.
        if self.impl.generate_c and self._c_output is not None:
            runtime_kwargs["fc1_c"] = _to_cute(self._c_output)
        else:
            runtime_kwargs["fc1_c"] = None
        compile_kwargs = dict(runtime_kwargs)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if self.misc.enable_iket:
            compile_kwargs["options"] = "iket"

        if self.misc.profile_friendly and self._use_cuda_profiler_api:
            torch.cuda.synchronize()
            _dist_active = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            if _dist_active:
                torch.distributed.barrier()
                torch.cuda.synchronize()
            if self.rank == 0:
                profile_cudart = torch.cuda.cudart()
                torch.cuda.check_error(profile_cudart.cudaProfilerStart())
            # Nsys uses a global multi-process range. Keep all ranks before
            # compile until rank 0 has made that range effective.
            if _dist_active:
                torch.distributed.barrier()
                torch.cuda.synchronize()

        self._compiled_kernel = cute.compile(self._kernel, **compile_kwargs)

        # -- 5. Launch (with optional profile-friendly barriers) --
        if self.misc.profile_friendly:
            import nvtx

            torch.cuda.synchronize()
            _dist_active = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            if _dist_active:
                torch.distributed.barrier()
                torch.cuda.synchronize()
            with nvtx.annotate("cute_dsl_prof"):
                self._launch_target_kernels_with_optional_torch_profiler(
                    runtime_kwargs,
                )
            if _dist_active:
                torch.distributed.barrier()
                torch.cuda.synchronize()
            # Do not call cudaProfilerStop here. With legacy IKET warp-phase
            # tracing, stop-shutdown can discard the target as incomplete.
            # Process teardown closes and flushes this profiling-only range.
        else:
            self._launch_target_kernels_with_optional_torch_profiler(
                runtime_kwargs,
            )


# =============================================================================
# CLI entry point
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MegaMoE BF16 GLU multi-rank fused dispatch+fc12+combine runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kind", type=str, default="bf16",
        choices=list(BF16_KIND_CHOICES),
        help="Element format for activations and weights (BF16 only).",
    )
    parser.add_argument("--num_tokens_per_rank", type=int, default=128)
    parser.add_argument("--num_topk", type=int, default=4)
    parser.add_argument("--num_total_experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument(
        "--fc2_output_dtype", type=_parse_output_dtype, default=torch.bfloat16,
    )
    parser.add_argument(
        "--route_distribution", type=str, default="balanced",
        choices=["balanced", "power_law"],
    )
    parser.add_argument(
        "--power_law_exponent", type=float, default=1.0,
        help="Zipf exponent for --route_distribution power_law.",
    )
    parser.add_argument(
        "--gate_up_clamp", type=float, default=None,
        help="DeepSeek-V4 swiglu_limit: clamp gate/up pre-activations before SiLU.",
    )
    swap_group = parser.add_mutually_exclusive_group()
    swap_group.add_argument(
        "--swap_ab", dest="swap_ab", action="store_true",
        help="Use the Hopper weight-as-A M128/M256xN swap-AB kernel.",
    )
    swap_group.add_argument(
        "--no_swap_ab", dest="swap_ab", action="store_false",
        help="Force the non-swap Hopper kernel and disable token heuristics.",
    )
    pingpong_group = parser.add_mutually_exclusive_group()
    pingpong_group.add_argument(
        "--pingpong", dest="pingpong", action="store_true",
        help="Alternate complete task tiles across two WGMMA+epilogue warpgroups.",
    )
    pingpong_group.add_argument(
        "--no_pingpong", dest="pingpong", action="store_false",
        help="Force legacy scheduling and disable token heuristics.",
    )
    parser.set_defaults(swap_ab=None, pingpong=None)

    parser.add_argument(
        "--generate_c", action="store_true", default=False,
        help="Training forward: also store the raw pre-SwiGLU fc1 accumulator "
             "(gate+up, BF16, raw GEMM N order) into a separate fc1_c tensor "
             "and validate it against the reference.",
    )
    parser.add_argument(
        "--tail_split_pairs", action="store_true", default=False,
        help="Split the tail cluster block of an expert with an odd CTA-tile "
             "count into pair tasks (both CTAs compute the single valid token "
             "tile against adjacent weight tiles).  Requires a token-side "
             "cluster of 2 (swap-AB cga 1,2,1 or non-swap cga 2,1,1).  "
             "Default: the heuristic table entry, or BF16_TAIL_SPLIT=1 when "
             "the selected geometry qualifies.",
    )

    parser.add_argument(
        "--mma_tiler_mnk",
        type=str,
        default=None,
        help="Manual M,N,K tile; setting it disables token heuristics.",
    )
    parser.add_argument(
        "--cluster_shape_mnk",
        type=str,
        default=None,
        help="Manual M,N,K cluster shape; setting it disables token heuristics.",
    )
    parser.add_argument("--use_2cta_instrs", action="store_true", default=False)
    parser.add_argument("--enable_static_expert_shape", action="store_true", default=False)
    parser.add_argument("--dynamic_sched", action="store_true", default=False)
    parser.add_argument("--clc_bundle_size", type=int, default=None)
    parser.add_argument("--num_sched_stages", type=int, default=None)
    parser.add_argument(
        "--load_balance_mode", type=str, default="static",
        choices=["static", "atomic_counter"],
    )
    parser.add_argument("--group_hint", type=int, default=None)
    parser.add_argument("--perf_run", action="store_true", default=False)
    parser.add_argument("--skip_ref_check", action="store_true", default=False)
    parser.add_argument("--profile_friendly", action="store_true", default=False)
    parser.add_argument(
        "--use_cuda_profiler_api",
        action="store_true",
        default=False,
        help="Start CUDA profiling before the profile-friendly launch.",
    )
    parser.add_argument("--use_torch_profiler", action="store_true", default=False)
    parser.add_argument("--perf_warmup", type=int, default=1)
    parser.add_argument("--perf_iters", type=int, default=10)
    parser.add_argument("--enable_debug_checks", action="store_true", default=False)
    parser.add_argument(
        "--ref_compute_graph", type=str, default="deepgemm",
        choices=["transformers", "deepgemm"],
    )
    parser.add_argument("--enable_iket", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--in_kernel_fc2_reduce", action="store_true", default=False,
        help="Collapse topk in kernel: epi_warps issues REDG into "
             "output_activation[t, :]; the dispatch token-back modes push "
             "with cp.reduce.async.bulk (token_back_reduce_topk).",
    )
    parser.add_argument(
        "--token_back_mode",
        type=str,
        default=None,
        choices=["epi_warps", "standalone_warps", "reuse_dispatch_warps"],
        help="Where the cross-rank fc2 write-back runs: epi_warps (epilogue "
             "STG/REDG straight to the source rank), standalone_warps (four "
             "dedicated token-back warps), or reuse_dispatch_warps (dispatch "
             "warps push after pull).  Default: the token-bucket heuristic "
             "table's per-bucket winner (heuristic_config.py).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.use_cuda_profiler_api and not args.profile_friendly:
        parser.error("--use_cuda_profiler_api requires --profile_friendly")

    if _NO_DIST:
        torch.cuda.set_device(0)
        rank = 0
        world_size = 1
    else:
        from src.bootstrap import (
            init_dist_and_nvshmem,
            finalize_dist_and_nvshmem,
        )
        _local_rank, rank, world_size, _ = init_dist_and_nvshmem()

    problem = TokenCommProblemDesc(
        world_size=world_size,
        num_tokens_per_rank=args.num_tokens_per_rank,
        num_topk=args.num_topk,
        num_total_experts=args.num_total_experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        fc2_output_dtype=args.fc2_output_dtype,
        route_distribution=args.route_distribution,
        power_law_exponent=args.power_law_exponent,
        gate_up_clamp=args.gate_up_clamp,
    )

    config_selection = resolve_hopper_bf16_config(
        args.num_tokens_per_rank,
        swap_ab=args.swap_ab,
        pingpong=args.pingpong,
        mma_tiler_mnk=(
            _parse_tuple(args.mma_tiler_mnk)
            if args.mma_tiler_mnk is not None
            else None
        ),
        cluster_shape_mnk=(
            _parse_tuple(args.cluster_shape_mnk)
            if args.cluster_shape_mnk is not None
            else None
        ),
    )
    launch_config = config_selection.config
    # Explicit CLI choice wins; else the table's per-bucket winner.
    token_back_mode = (
        args.token_back_mode
        if args.token_back_mode is not None
        else launch_config.token_back_mode
    )
    group_hint = (
        args.group_hint if args.group_hint is not None else launch_config.group_hint
    )
    # CLI flag or table entry (with the BF16_TAIL_SPLIT override applied).
    tail_split_pairs = bool(args.tail_split_pairs or launch_config.tail_split_pairs)
    if rank == 0:
        bucket = (
            str(config_selection.token_bucket)
            if config_selection.token_bucket is not None
            else "n/a"
        )
        print(
            "[mega_runner_bf16] "
            f"config_source={config_selection.source} token_bucket={bucket} "
            f"swap_ab={launch_config.swap_ab} "
            f"pingpong={launch_config.pingpong} "
            f"mma_tiler_mnk={launch_config.mma_tiler_mnk} "
            f"cluster_shape_mnk={launch_config.cluster_shape_mnk} "
            f"token_back_mode={token_back_mode} group_hint={group_hint} "
            f"tail_split_pairs={tail_split_pairs}",
            flush=True,
        )

    impl = ImplDesc(
        mma_tiler_mnk=launch_config.mma_tiler_mnk,
        cluster_shape_mnk=launch_config.cluster_shape_mnk,
        use_2cta_instrs=args.use_2cta_instrs,
        enable_static_expert_shape=args.enable_static_expert_shape,
        force_static_sched=not args.dynamic_sched,
        clc_bundle_size=args.clc_bundle_size,
        num_sched_stages=args.num_sched_stages,
        load_balance_mode=args.load_balance_mode,
        group_hint=group_hint,
        non_ubulk_fc2_store=True,
        in_kernel_fc2_reduce=args.in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        epi_flag_batch=(2, 4),
        flag_batch=1,
        generate_c=args.generate_c,
        tail_split_pairs=tail_split_pairs,
    )

    misc = MiscDesc(
        perf_run=args.perf_run,
        skip_ref_check=args.skip_ref_check,
        run_target_kernel_only=args.profile_friendly,
        enable_debug_checks=args.enable_debug_checks,
        ref_compute_graph=args.ref_compute_graph,
        enable_iket=args.enable_iket,
        seed=args.seed,
    )

    tester = MegaMoEBf16Tester(
        problem,
        impl,
        misc,
        rank=rank,
        kind=args.kind,
        swap_ab=launch_config.swap_ab,
        pingpong=launch_config.pingpong,
        use_cuda_profiler_api=args.use_cuda_profiler_api,
    )
    tester.set_torch_profiler_enabled(args.use_torch_profiler)
    tester.set_perf_iters(args.perf_warmup, args.perf_iters)

    return_code = 0
    try:
        tester.run()
    except NotImplementedError as exc:
        if rank == 0:
            print(f"[mega_runner_bf16] kernel launch skipped: {exc}")

    if not _NO_DIST:
        tester._compiled_kernel = None
        tester._kernel = None
        gc.collect()
        torch.cuda.synchronize()
        try:
            import nvshmem.core
            for sym_tensor in (
                tester.my_activation,
                tester.my_topk_idx, tester.my_topk_weights,
                tester.output_activation, tester.shared_workspace,
            ):
                if sym_tensor is not None:
                    try:
                        nvshmem.core.free_tensor(sym_tensor)
                    except Exception:  # noqa: BLE001
                        pass
            tester.my_activation = None
            tester.my_topk_idx = None
            tester.my_topk_weights = None
            tester.output_activation = None
            tester.shared_workspace = None
        except ImportError:
            pass

        gc.collect()
        finalize_dist_and_nvshmem()
    return return_code


if __name__ == "__main__":
    sys.exit(main())
