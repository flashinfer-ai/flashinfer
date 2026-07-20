# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone multi-rank MegaMoE host driver for the BF16 GLU path.

BF16 A/B operands (2 bytes/element, no scale factors), non-swap-AB.  It
subclasses the NVFP4 ``MegaMoETester`` and reuses all of its distributed
bootstrap, routing-table generation, symmetric-heap allocation, workspace
allocation and teardown; only the kind-specific stages are overridden:

  * ``generate_inputs``   -- BF16 input + weight generation + sym staging
  * ``compute_reference`` -- ``compute_megamoe_reference`` (bit-exact dense
                             GEMM launcher)
  * ``run_kernel``        -- instantiate ``Sm100MegaMoEBf16Kernel``
  * ``validate``          -- per-rank combine compare (form A K-sum / form B
                             squeeze)

Launcher::

    torchrun --nproc_per_node=4 -m moe_bf16_glu.mega_runner \\
        --kind bf16 --num_total_experts 32 --route_distribution balanced
"""

import argparse
import gc
import os
import sys
from typing import List, Optional, Tuple

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
from moe_nvfp4_swapab.runner_common import round_up
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
from moe_bf16_glu.runner_common import TrainingImplDesc
from moe_bf16_glu.mega_reference_bf16 import compute_megamoe_reference


# =============================================================================
# kind <-> dtype maps
# =============================================================================

_KIND_TO_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
}


def _kind_to_cutlass_dtype(kind: str):
    import cutlass
    return {
        "bf16": cutlass.BFloat16,
    }[kind]


# =============================================================================
# BF16 host-side tensor builders (torch rng)
# =============================================================================


def _make_bf16_tensor(
    torch_rng: torch.Generator,
    shape: Tuple[int, ...],
    *,
    perf_run: bool,
) -> torch.Tensor:
    """Build a BF16 data tensor.

    perf:        near-uniform random BF16 bit patterns (timing only): uniform
                 int16 bit patterns with the NaN/Inf encodings (exponent
                 all-ones, 0.4% of patterns) remapped to the max finite
                 exponent by clearing one exponent bit.
    correctness: sparse {0, +0.5, -0.5} (exactly representable in BF16).  The
                 nonzero density keeps the fc2 combine terms bounded, which
                 the tolerance-based form-B (REDG) compare relies on.  Tune
                 via ``BF16_NONZERO_PCT`` (each sign gets PCT%; default
                 1% + 1% = 2% nonzero).
    """
    n = 1
    for s in shape:
        n *= s
    if perf_run:
        bits = torch.randint(
            -32768, 32768, (n,), dtype=torch.int16, device="cuda",
            generator=torch_rng,
        )
        nan_inf = (bits & 0x7F80) == 0x7F80
        bits = torch.where(nan_inf, bits & ~0x0080, bits)
        return bits.view(torch.bfloat16).reshape(shape)
    nz_pct = float(os.environ.get("BF16_NONZERO_PCT", "1")) / 100.0
    fp32 = torch.zeros((n,), dtype=torch.float32, device="cuda")
    rand = torch.rand((n,), device="cuda", generator=torch_rng)
    fp32[rand < nz_pct] = 0.5
    fp32[(rand >= nz_pct) & (rand < 2.0 * nz_pct)] = -0.5
    return fp32.to(torch.bfloat16).reshape(shape)


# =============================================================================
# BF16 MegaMoE tester
# =============================================================================


class MegaMoEBf16Tester(MegaMoETester):
    """BF16 specialisation of the multi-rank MegaMoE host driver."""

    def __init__(
        self,
        problem: TokenCommProblemDesc,
        impl: TrainingImplDesc,
        misc: MiscDesc,
        *,
        rank: int,
        kind: str = "bf16",
    ) -> None:
        super().__init__(problem, impl, misc, rank=rank)
        if kind not in _KIND_TO_TORCH_DTYPE:
            raise ValueError(
                f"kind must be one of {sorted(_KIND_TO_TORCH_DTYPE)}, got {kind!r}."
            )
        self.kind = kind
        self.torch_ab_dtype = _KIND_TO_TORCH_DTYPE[kind]
        self._apply_topk_in_fc1 = True
        if problem.fc2_output_dtype is not torch.bfloat16:
            raise ValueError(
                f"BF16 pipeline requires fc2_output_dtype=torch.bfloat16; "
                f"got {problem.fc2_output_dtype}."
            )
        # Early host-side mirror of the kernel's store-granularity checks:
        # fc1 STG (use_stg_fc1) stores full 256-wide gate+up tiles; the
        # default TMA store path clamps at the tensor extent, so its only
        # hard requirement is the gate/up interleave pair unit (2 x 32 = 64
        # gate+up columns).  fc2 hidden subtiles are 32 wide.
        _gateup_granularity = 256 if impl.use_stg_fc1 else 64
        if problem.intermediate % _gateup_granularity != 0:
            raise ValueError(
                f"intermediate (gate+up width) must be a multiple of "
                f"{_gateup_granularity} (use_stg_fc1={impl.use_stg_fc1}); "
                f"got {problem.intermediate}."
            )
        if problem.hidden % 32 != 0:
            raise ValueError(
                f"hidden must be a multiple of 32; got {problem.hidden}."
            )
        if not (1 <= problem.num_topk <= 32):
            raise ValueError(
                f"num_topk must be in [1, 32]; got {problem.num_topk}."
            )

    # ------------------------------------------------------------------
    # Step 1: deterministic input + weight generation (BF16)
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

        # ---- Activation (BF16).
        self._global_activation = _make_bf16_tensor(
            self._torch_cuda_rng, (num_ranks, num_tokens_per_rank, hidden),
            perf_run=self.misc.perf_run,
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
        self._global_fc1_weight = _make_bf16_tensor(
            self._torch_cuda_rng,
            (num_ranks, num_experts_per_rank, intermediate, hidden),
            perf_run=self.misc.perf_run,
        ).permute(0, 1, 3, 2)

        self._global_fc2_weight = _make_bf16_tensor(
            self._torch_cuda_rng,
            (num_ranks, num_experts_per_rank, hidden, intermediate_downproj),
            perf_run=self.misc.perf_run,
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

        # ---- Combine output on sym heap.
        # Form B (token-back reduce): (T, 1, H); kernel reduces topk axis.
        # Form A / token_back_by_dispatch: (T, K, H); host reduces topk axis.
        combine_topk = 1 if self.impl.in_kernel_fc2_reduce else num_topk
        self.combine_output = _sym_zeros(
            (num_tokens_per_rank, combine_topk, hidden),
            problem.fc2_output_dtype,
        )

        torch.cuda.synchronize()
        self._check_cuda_rng_consistency()

    # ------------------------------------------------------------------
    # Step 2: reference (BF16, bit-exact dense GEMM)
    # ------------------------------------------------------------------

    def compute_reference(self) -> None:
        if self.misc.skip_ref_check:
            return
        if self._global_activation is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")

        ref_result = compute_megamoe_reference(
            input_activation=self._global_activation,
            input_topk_idx=self._global_topk_idx,
            input_topk_weights=self._global_topk_weights,
            fc1_weight=self._global_fc1_weight,
            fc2_weight=self._global_fc2_weight,
            ref_compute_graph=self.misc.ref_compute_graph,
            fc2_output_dtype=self.problem.fc2_output_dtype,
            gate_up_clamp=self.problem.gate_up_clamp,
            apply_topk_in_fc1=self._apply_topk_in_fc1,
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
    # Step 5: validation (BF16)
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Compare the rank's combine output against the reference.

        * **Form A / token_back_by_dispatch**: kernel writes per-(token, topk)
          cells into ``(T, K, H)``; host reduces the topk axis on both sides.
        * **Form B (token-back reduce)**: kernel writes topk-reduced into
          ``(T, 1, H)``; compare directly against the reference's topk-summed
          view.
        """
        if self.misc.skip_ref_check:
            return
        if self.combine_output is None:
            raise RuntimeError("validate requires run_kernel first.")
        if self.combine_output_ref is None:
            raise RuntimeError("validate requires compute_reference first.")

        if self.impl.in_kernel_fc2_reduce:
            # Form B: combine_output is (T, 1, H); squeeze topk=1.
            actual_reduced = self.combine_output[:, 0, :].to(torch.float32)
            # Reference is (T, K, H); sum over K to match.
            ref_reduced = self.combine_output_ref.to(torch.float32).sum(dim=1)
        else:
            # Form A / token_back_by_dispatch: (T, K, H) -> sum over K.
            actual_reduced = self.combine_output.to(torch.float32).sum(dim=1)
            ref_reduced = self.combine_output_ref.to(torch.float32).sum(dim=1)

        # bf16-grade tolerance; the K-axis fp32 sum adds at most ~K bf16 ULPs,
        # well below this band for the v1 configurations.
        # TODO: replace to silent check numerical dist when kernel stable.
        compare_and_report_mismatches(
            actual_reduced,
            ref_reduced,
            name=f"combine_output[rank{self.rank}]",
            atol=1e-2,
            rtol=1e-2,
        )

        self._validate_c_output()

    def _validate_c_output(self) -> None:
        """Compare kernel c_output vs reference pre-SwiGLU fc1 gate+up per expert.

        Rows are compared positionally.  The multi-rank dispatch order of pool
        rows is non-deterministic, but each pool row's source coordinates are
        recorded by the dispatch warps in the ``token_src_metadata`` workspace
        region (one packed ``(src_rank, src_token, src_topk)`` int64 per row).
        The reference rows are produced in (rank, token, topk) row-major order
        (``routing_mask.nonzero()`` in ``compute_megamoe_reference``), so a
        keyed permutation aligns the two sides for an element-wise comparison.
        """
        if not self.impl.generate_c:
            return
        c = getattr(self, "_c_output", None)
        if c is None:
            if self.rank == 0:
                print("[generate_c] c_output not allocated — skipped.")
            return
        ref_map = getattr(self, "_ref_fc1_gateup_per_expert", None)
        if not ref_map:
            if self.rank == 0:
                print("[generate_c] reference fc1 gate+up not available — skipped.")
            return

        valid = self._c_valid_tokens_per_expert
        doff = self._c_data_physical_offsets

        # Per-pool-row source coordinates written by dispatch_pull.
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
        print(f"[generate_c][rank{self.rank}] kernel c_output vs reference fc1 gate+up:")
        any_checked = False
        pool_base = 0
        for e in range(self.problem.num_experts_per_rank):
            v_e = valid[e]
            ref = ref_map.get(e)
            expert_pool_base = pool_base
            pool_base += round_up(v_e, tpb)
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

            # Reference rows for this expert, in ascending-key order.
            global_expert = self.rank * self.problem.num_experts_per_rank + e
            routed = (topk_cpu == global_expert).nonzero(as_tuple=False)
            ref_keys = (
                routed[:, 0] * tokens_per_rank + routed[:, 1]
            ) * num_topk + routed[:, 2]

            # Bijection: the routing-table keys are unique ((rank, token,
            # topk) rows are distinct), so sorted kernel keys must equal the
            # reference key list element-wise -- this simultaneously checks
            # equal counts, both-side membership, and that no pool row
            # duplicates another row's source coordinate.
            if not torch.equal(torch.sort(kernel_keys).values, ref_keys):
                raise RuntimeError(
                    f"[generate_c][rank{self.rank}] expert {e}: pool metadata "
                    f"keys do not biject onto the routing table "
                    f"(kernel rows={v_e}, reference rows={ref_keys.numel()})."
                )
            # With the bijection established, searchsorted gives each kernel
            # row's exact reference row.
            pos = torch.searchsorted(ref_keys, kernel_keys)

            kernel_rows = c[doff[e] : doff[e] + v_e].float().cpu()
            ref_rows = ref.float().cpu()[pos]
            compare_and_report_mismatches(
                kernel_rows,
                ref_rows,
                name=f"c_output[rank{self.rank}]expert{e}",
                atol=1e-5,
                rtol=1e-2,
                max_mismatches=5,
            )
        if not any_checked:
            print("  (no valid tokens routed to any local expert)")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Step 4: kernel launch (BF16)
    # ------------------------------------------------------------------

    def run_kernel(self) -> None:
        """Compile + launch ``Sm100MegaMoEBf16Kernel`` on the current stream.

        Mirrors ``MegaMoETester.run_kernel`` step-for-step; only the kernel
        instantiation (class + ``ab_dtype``) is BF16-specific.
        """
        if (
            self.my_activation is None
            or self.my_topk_idx is None
            or self.my_topk_weights is None
            or self.my_fc1_weight is None
            or self.my_fc2_weight is None
            or self.combine_output is None
        ):
            raise RuntimeError("run_kernel requires generate_inputs first.")

        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        from moe_nvfp4_swapab.epilogue import EpilogueTokenTile
        from moe_bf16_glu.megamoe_kernel_bf16 import Sm100MegaMoEBf16Kernel
        from src.sym_buffer import SymBufferHost

        # -- 1. Kernel instance (BF16 requires static_expert_shape != None) --
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

        # generate_c TMA tile height is cta_tile_m=128; physical row offsets must
        # be 128-aligned so the scheduler's data_physical_offsets land on tile
        # boundaries.  Fall back to the default EpilogueTokenTile otherwise.
        c_token_padding = 128 if self.impl.generate_c else EpilogueTokenTile

        self._kernel = Sm100MegaMoEBf16Kernel(
            mma_tiler_mnk=self.impl.mma_tiler_mnk,
            cluster_shape_mnk=self.impl.cluster_shape_mnk,
            use_2cta_instrs=self.impl.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=c_token_padding,
            load_balance_mode=self.impl.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=self.impl.force_static_sched,
            clc_bundle_size=self.impl.clc_bundle_size,
            num_sched_stages=self.impl.num_sched_stages,
            ab_dtype=_kind_to_cutlass_dtype(self.kind),
            world_size=self.world_size,
            local_rank=self.rank,
            num_topk=self.problem.num_topk,
            max_tokens_per_rank=self.problem.num_tokens_per_rank,
            hidden=self.problem.hidden,
            fc2_in_kernel_topk_reduce=self.impl.in_kernel_fc2_reduce,
            token_back_by_dispatch=self.impl.token_back_by_dispatch,
            token_back_mode=self.impl.token_back_mode,
            epi_flag_batch=self.impl.epi_flag_batch,
            flag_batch=self.impl.flag_batch,
            gate_up_clamp=self.problem.gate_up_clamp,
            apply_topk_in_fc1=self._apply_topk_in_fc1,
            generate_c=self.impl.generate_c,
            use_stg_fc1=self.impl.use_stg_fc1,
        )

        # -- generate_c: allocate output tensor and compute per-expert offsets --
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
                doff.append(doff[-1] + round_up(v, 128))
            tokens_sum = max(1, doff[-1])
            # problem.intermediate is already the full gate+up width here
            # (unlike lean fc12 where it's the post-SwiGLU half-size).
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
        combine_output_cute = _to_cute(self.combine_output)
        local_workspace_cute = _to_cute(self.local_workspace, force_static_layout=True)
        shared_workspace_cute = _to_cute(self.shared_workspace)

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
            combine_output=combine_output_cute,
            local_workspace=local_workspace_cute,
            shared_workspace=shared_workspace_cute,
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )
        if self.impl.generate_c and self._c_output is not None:
            runtime_kwargs["fc1_c"] = _to_cute(self._c_output)
        else:
            runtime_kwargs["fc1_c"] = None
        compile_kwargs = dict(runtime_kwargs)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if self.misc.enable_iket:
            compile_kwargs["options"] = "iket"

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
        choices=["bf16"],
        help="Data element format for activations and weights.",
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

    # BF16 fused fc12 is validated for the (256, 256, 64) 2-CTA tile only.
    parser.add_argument("--mma_tiler_mnk", type=str, default="256,256,64")
    parser.add_argument("--cluster_shape_mnk", type=str, default="2,1,1")
    parser.add_argument("--use_2cta_instrs", action="store_true", default=True)
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
        help="Form B: in-kernel topk-reduce to combine_output[t, 0, :] — "
             "epilogue REDG under epi_warps, cp.reduce bulk-add under the "
             "token-back modes.",
    )
    parser.add_argument(
        "--generate_c", action="store_true", default=False,
        help="Store raw pre-SwiGLU fc1 accumulator (gate+up, BF16) to a separate tensor.",
    )
    parser.add_argument(
        "--use_stg_fc1", action="store_true", default=False,
        help="Write fc1 BF16 output directly to GMEM via STG instead of R2S+TMA. "
             "Eliminates sD SMEM staging; may increase AB pipeline stages.",
    )
    parser.add_argument(
        "--token_back_mode", type=str, default="epi_warps",
        choices=["epi_warps", "standalone_warps", "reuse_dispatch_warps"],
        help="Where the cross-rank fc2 push-back runs: epi_warps (epilogue "
             "STG redirect, form A, default), standalone_warps (dedicated "
             "warps 12-15), or reuse_dispatch_warps (dispatch warps 8-11).",
    )
    parser.add_argument(
        "--epi_flag_batch", type=str, default="2,4",
        help="Done-counter publish batching as 'fc1,fc2' (e.g. '2,4'). "
             "Each component must be in [1, 32].",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

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

    impl = TrainingImplDesc(
        mma_tiler_mnk=_parse_tuple(args.mma_tiler_mnk),
        cluster_shape_mnk=_parse_tuple(args.cluster_shape_mnk),
        use_2cta_instrs=args.use_2cta_instrs,
        enable_static_expert_shape=args.enable_static_expert_shape,
        force_static_sched=not args.dynamic_sched,
        clc_bundle_size=args.clc_bundle_size,
        num_sched_stages=args.num_sched_stages,
        load_balance_mode=args.load_balance_mode,
        group_hint=args.group_hint,
        non_ubulk_fc2_store=True,
        in_kernel_fc2_reduce=args.in_kernel_fc2_reduce,
        token_back_mode=args.token_back_mode,
        epi_flag_batch=_parse_tuple(args.epi_flag_batch),
        flag_batch=1,
        generate_c=args.generate_c,
        use_stg_fc1=args.use_stg_fc1,
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

    if (
        args.use_torch_profiler
        and not args.skip_ref_check
        and not args.profile_friendly
    ):
        if rank == 0:
            print(
                "[mega_runner_bf16] torch.profiler disabled: "
                "reference validation is enabled."
            )
        args.use_torch_profiler = False

    tester = MegaMoEBf16Tester(problem, impl, misc, rank=rank, kind=args.kind)
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
                tester.combine_output, tester.shared_workspace,
            ):
                if sym_tensor is not None:
                    try:
                        nvshmem.core.free_tensor(sym_tensor)
                    except Exception:  # noqa: BLE001
                        pass
            tester.my_activation = None
            tester.my_topk_idx = None
            tester.my_topk_weights = None
            tester.combine_output = None
            tester.shared_workspace = None
        except ImportError:
            pass

        gc.collect()
        finalize_dist_and_nvshmem()
    return return_code


if __name__ == "__main__":
    sys.exit(main())
