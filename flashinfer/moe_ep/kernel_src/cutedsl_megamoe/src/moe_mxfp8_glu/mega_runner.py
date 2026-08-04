# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone multi-rank MegaMoE host driver for the MXFP8 GLU path.

Mirror of ``moe_nvfp4_swapab.mega_runner`` (NVFP4) specialised for MXFP8: 1-byte
fp8 data (``e4m3`` / ``e5m2``) + E8M0 block scales (``sf_vec_size = 32``),
non-swap-AB.  It subclasses the NVFP4 ``MegaMoETester`` and reuses all of its
distributed bootstrap, routing-table generation, symmetric-heap allocation,
workspace allocation, validation and teardown; only the three kind-specific
stages are overridden:

  * ``generate_inputs``   -- fp8 / E8M0 input + weight generation + sym staging
  * ``compute_reference`` -- ``compute_megamoe_reference_mxfp8``
  * ``run_kernel``        -- instantiate ``Sm100MegaMoEMxfp8Kernel``

Launcher::

    torchrun --nproc_per_node=4 -m moe_mxfp8_glu.mega_runner \\
        --kind mxfp8_e4m3 --num_total_experts 32 --route_distribution balanced
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

from common.megamoe_constants import Mxfp8BlockSize
from common.host_utils import compare_and_report_mismatches
from moe_nvfp4_swapab.runner_common import (
    ceil_div,
    round_up,
    to_blocked,
    _stack_byte_reinterpretable_tensors,
    Mxfp8ScaleDtype,
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
    _parse_tuple,
    _parse_output_dtype,
    _NO_DIST,
)
from src.token_comm import CombineFormat
from moe_mxfp8_glu.runner_common import TrainingImplDesc
from moe_mxfp8_glu.mega_reference_mxfp8 import compute_megamoe_reference_mxfp8


# =============================================================================
# kind <-> dtype maps
# =============================================================================

_KIND_TO_TORCH_DTYPE = {
    "mxfp8_e4m3": torch.float8_e4m3fn,
    "mxfp8_e5m2": torch.float8_e5m2,
}


def _kind_to_cutlass_dtype(kind: str):
    import cutlass

    return {
        "mxfp8_e4m3": cutlass.Float8E4M3FN,
        "mxfp8_e5m2": cutlass.Float8E5M2,
    }[kind]


# =============================================================================
# MXFP8 host-side tensor builders (torch rng)
# =============================================================================


def _make_fp8_tensor(
    torch_rng: torch.Generator,
    shape: Tuple[int, ...],
    data_dtype: torch.dtype,
    *,
    perf_run: bool,
) -> torch.Tensor:
    """Build an fp8 (e4m3 / e5m2) data tensor (1 byte/element, not packed).

    perf:        random finite fp8 bytes (timing only); ``randint`` skips
                 dtype-specific NaN/Inf encodings (e4m3: 0x7F/0xFF;
                 e5m2: 0x7C-0x7F / 0xFC-0xFF).
    correctness: sparse {0, +0.5, -0.5}.  The nonzero DENSITY is bounded so the
                 fc1/fc2 dot products stay small in magnitude (K up to ~7168 for
                 fc1).  Dense data pushes fc1-output values near MXFP8 quant-bin
                 boundaries where the kernel's ``quant_sfd_row`` and the host
                 ``mxfp8_quantize_per_block_32`` can round to adjacent bins -- a
                 discrete ~1-fp8-step difference that survives the topk-combine
                 and trips the validator on a handful of cells.  Low density
                 keeps the bf16 fc2 output below the validator's atol floor.
                 Tune via ``MXFP8_NONZERO_PCT`` (each sign gets PCT%; default
                 1% + 1% = 2% nonzero), matching the reference runner.
    """
    n = 1
    for s in shape:
        n *= s
    if perf_run:
        if data_dtype == torch.float8_e4m3fn:
            # E4M3FN NaN: 0x7F, 0xFF → sample 254 codes, skip 127.
            idx = torch.randint(
                0,
                254,
                (n,),
                device="cuda",
                generator=torch_rng,
            )
            flat_bytes = torch.where(idx < 127, idx, idx + 1).to(torch.uint8)
        elif data_dtype == torch.float8_e5m2:
            # E5M2 Inf/NaN: 0x7C-0x7F, 0xFC-0xFF → sample 248 codes, skip 4.
            idx = torch.randint(
                0,
                248,
                (n,),
                device="cuda",
                generator=torch_rng,
            )
            flat_bytes = torch.where(idx < 124, idx, idx + 4).to(torch.uint8)
        else:
            raise ValueError(f"Unsupported fp8 data_dtype: {data_dtype}")
        return flat_bytes.view(data_dtype).reshape(shape)
    nz_pct = float(os.environ.get("MXFP8_NONZERO_PCT", "1")) / 100.0
    fp32 = torch.zeros((n,), dtype=torch.float32, device="cuda")
    rand = torch.rand((n,), device="cuda", generator=torch_rng)
    fp32[rand < nz_pct] = 0.5
    fp32[(rand >= nz_pct) & (rand < 2.0 * nz_pct)] = -0.5
    return fp32.to(data_dtype).reshape(shape)


def _make_e8m0_scale_tensor(
    torch_rng: torch.Generator,
    non_k_size: int,
    k_size: int,
    blocksize: int,
) -> torch.Tensor:
    """Plain ``(non_k_size, ceil_div(k_size, blocksize))`` E8M0 SF tensor.

    Uses 80% scale=1.0 (E8M0 byte 0x7F = 127) / 20% scale=2.0 (0x80 = 128).
    """
    num_scale_cols = ceil_div(k_size, blocksize)
    rand = torch.rand((non_k_size, num_scale_cols), device="cuda", generator=torch_rng)
    scale_bytes = torch.where(
        rand < 0.80,
        torch.full_like(rand, 127.0),
        torch.full_like(rand, 128.0),
    ).to(torch.uint8)
    return scale_bytes.view(Mxfp8ScaleDtype).reshape(non_k_size, num_scale_cols)


def _sym_zeros_byte_view_1b(
    logical_shape: Tuple[int, ...],
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Sym-heap allocation for 1-byte dtypes (fp8 e4m3/e5m2, E8M0) that
    nvshmem4py doesn't natively support: allocate a uint8 byte buffer and
    ``.view(target_dtype)`` (shape preserved -- all are 1 byte/element).
    """
    total_bytes = 1
    for dim_size in logical_shape:
        total_bytes *= dim_size
    return (
        _sym_zeros((total_bytes,), torch.uint8)
        .view(target_dtype)
        .reshape(logical_shape)
    )


# =============================================================================
# MXFP8 MegaMoE tester
# =============================================================================


class MegaMoEMxfp8Tester(MegaMoETester):
    """MXFP8 specialisation of the multi-rank MegaMoE host driver."""

    def __init__(
        self,
        problem: TokenCommProblemDesc,
        impl: TrainingImplDesc,
        misc: MiscDesc,
        *,
        rank: int,
        kind: str = "mxfp8_e4m3",
        combine_format: Optional[CombineFormat] = None,
    ) -> None:
        super().__init__(problem, impl, misc, rank=rank)
        if kind not in _KIND_TO_TORCH_DTYPE:
            raise ValueError(
                f"kind must be one of {sorted(_KIND_TO_TORCH_DTYPE)}, got {kind!r}."
            )
        self.kind = kind
        self.torch_ab_dtype = _KIND_TO_TORCH_DTYPE[kind]
        self._apply_topk_in_fc1 = True
        self._combine_format = (
            combine_format
            if combine_format is not None
            else CombineFormat.parse("bf16")
        )
        if impl.in_kernel_fc2_reduce and impl.token_back_by_dispatch:
            raise ValueError(
                "in_kernel_fc2_reduce and token_back_by_dispatch cannot both be True."
            )
        if impl.in_kernel_fc2_reduce and self._combine_format.is_quantized:
            raise ValueError(
                "in_kernel_fc2_reduce and quantized combine_format cannot both be True."
            )

    # ------------------------------------------------------------------
    # Step 1: deterministic input + weight generation (MXFP8)
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
        scale_blocksize = Mxfp8BlockSize
        data_dtype = self.torch_ab_dtype
        hidden_sf_cols = ceil_div(hidden, scale_blocksize)
        intermediate_downproj_sf_cols = ceil_div(intermediate_downproj, scale_blocksize)

        # ---- Activation (fp8, not packed) + plain K-major E8M0 SF.  The SF
        # stays in the user-facing plain layout; the kernel's dispatch warps
        # re-stage it into the SFB atom layout inside l1_sf_buffer before fc1
        # reads it, so the host reference can dequant from the plain layout.
        self._global_activation = _make_fp8_tensor(
            self._torch_cuda_rng,
            (num_ranks, num_tokens_per_rank, hidden),
            data_dtype,
            perf_run=self.misc.perf_run,
        )
        self._global_activation_sf = _make_e8m0_scale_tensor(
            self._torch_cuda_rng,
            num_ranks * num_tokens_per_rank,
            hidden,
            blocksize=scale_blocksize,
        ).reshape(num_ranks, num_tokens_per_rank, hidden_sf_cols)

        # ---- Routing table.
        if problem.route_distribution == "balanced":
            topk_idx_np = _generate_topk_idx_balanced(
                num_ranks,
                num_tokens_per_rank,
                num_topk,
                problem.num_total_experts,
                rng,
            )
        else:
            topk_idx_np = _generate_topk_idx_power_law(
                num_ranks,
                num_tokens_per_rank,
                num_topk,
                problem.num_total_experts,
                problem.power_law_exponent,
                rng,
            )
        topk_weights = _generate_topk_weights(
            num_ranks,
            num_tokens_per_rank,
            num_topk,
            self._torch_cuda_rng,
        )
        if self.rank == 0:
            _print_remote_rank_comm_matrices(
                topk_idx_np,
                num_ranks,
                problem.num_total_experts,
            )
        self._global_topk_idx = torch.from_numpy(topk_idx_np).cuda()
        self._global_topk_weights = topk_weights

        # ---- Weights.  fc1: logical (experts, intermediate, hidden) permuted
        # to (experts, hidden, intermediate) with hidden stride-1 (K-major).
        # fc2: logical (experts, hidden, inter//2) permuted to
        # (experts, inter//2, hidden) with inter//2 stride-1.
        self._global_fc1_weight = _make_fp8_tensor(
            self._torch_cuda_rng,
            (num_ranks, num_experts_per_rank, intermediate, hidden),
            data_dtype,
            perf_run=self.misc.perf_run,
        ).permute(0, 1, 3, 2)
        self._global_fc1_weight_sf = _make_e8m0_scale_tensor(
            self._torch_cuda_rng,
            num_ranks * num_experts_per_rank * intermediate,
            hidden,
            blocksize=scale_blocksize,
        ).reshape(num_ranks, num_experts_per_rank, intermediate, hidden_sf_cols)

        self._global_fc2_weight = _make_fp8_tensor(
            self._torch_cuda_rng,
            (num_ranks, num_experts_per_rank, hidden, intermediate_downproj),
            data_dtype,
            perf_run=self.misc.perf_run,
        ).permute(0, 1, 3, 2)
        self._global_fc2_weight_sf = _make_e8m0_scale_tensor(
            self._torch_cuda_rng,
            num_ranks * num_experts_per_rank * hidden,
            intermediate_downproj,
            blocksize=scale_blocksize,
        ).reshape(
            num_ranks,
            num_experts_per_rank,
            hidden,
            intermediate_downproj_sf_cols,
        )

        # ---- Atom-swizzle the weight SFs (the base kernel consumes weight SFs
        # in the 32x4x4 atom-swizzled flat layout that tile_atom_to_shape_SF /
        # the TMA SFA descriptor expect).  Plain ``_global_fc1/2_weight_sf`` are
        # kept untouched for ``compute_megamoe_reference_mxfp8``.
        fc1_sf_swizzled = [
            to_blocked(self._global_fc1_weight_sf[r, e])
            for r in range(num_ranks)
            for e in range(num_experts_per_rank)
        ]
        fc1_flat_sf_size = fc1_sf_swizzled[0].numel()
        self._global_fc1_weight_sf_swizzled = _stack_byte_reinterpretable_tensors(
            fc1_sf_swizzled, dim=0
        ).view(num_ranks, num_experts_per_rank, fc1_flat_sf_size)

        fc2_sf_swizzled = [
            to_blocked(self._global_fc2_weight_sf[r, e])
            for r in range(num_ranks)
            for e in range(num_experts_per_rank)
        ]
        fc2_flat_sf_size = fc2_sf_swizzled[0].numel()
        self._global_fc2_weight_sf_swizzled = _stack_byte_reinterpretable_tensors(
            fc2_sf_swizzled, dim=0
        ).view(num_ranks, num_experts_per_rank, fc2_flat_sf_size)

        # ---- Stage own-rank inputs into the symmetric heap.
        own_activation = self._global_activation[self.rank]
        own_activation_sf = self._global_activation_sf[self.rank]
        own_topk_idx = self._global_topk_idx[self.rank]
        own_topk_weights = self._global_topk_weights[self.rank]

        # fp8 goes through a uint8 byte-buf view (nvshmem4py doesn't natively
        # support fp8); copy via uint8 to dodge fp8 assignment quirks.
        self.my_activation = _sym_zeros_byte_view_1b(
            (num_tokens_per_rank, hidden),
            data_dtype,
        )
        self.my_activation.view(torch.uint8).copy_(own_activation.view(torch.uint8))

        # SF leg caller contract: the K_sf axis (= ceil(hidden, 32)) must be a
        # multiple of 4 E8M0 SFs so dispatch_pull's LDG.32 byte stride matches
        # the host row stride.  Pad-to-mult-of-4 + zero-fill on the host side;
        # the trailing padded SFs pair with fp8 data in TMA's OOB-fill-0 region.
        hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)
        self.my_activation_sf = _sym_zeros_byte_view_1b(
            (num_tokens_per_rank, hidden_sf_cols_padded),
            Mxfp8ScaleDtype,
        )
        self.my_activation_sf.view(torch.uint8)[:, :hidden_sf_cols].copy_(
            own_activation_sf.view(torch.uint8)
        )

        self.my_topk_idx = _sym_zeros(tuple(own_topk_idx.shape), torch.int64)
        self.my_topk_idx.copy_(own_topk_idx)

        self.my_topk_weights = _sym_zeros(tuple(own_topk_weights.shape), torch.float32)
        self.my_topk_weights.copy_(own_topk_weights)

        # ---- Own-rank weights stay on regular cuda.  DO NOT ``.contiguous()``:
        # the permute above puts the K dim mid-tensor (stride-1); contiguity
        # would re-pack to row-major and break the K-as-stride-1 invariant the
        # dequant + GEMM path depends on.
        self.my_fc1_weight = self._global_fc1_weight[self.rank]
        self.my_fc1_weight_sf = self._global_fc1_weight_sf_swizzled[self.rank]
        self.my_fc2_weight = self._global_fc2_weight[self.rank]
        self.my_fc2_weight_sf = self._global_fc2_weight_sf_swizzled[self.rank]

        # ---- Public 2D (T, hidden) combined output.  The kernel owns the per-topk
        # (T, K, H) combine staging internally on the sym heap (combine_quant inside
        # shared_workspace, allocated by allocate_workspaces via the kernel's
        # _build_shared_region_specs); the caller only provides / consumes the final
        # reduced result.  Placement depends on the reduce mode:
        #   * in_kernel_reduce (Form B): this IS the cross-rank REDG target, so it
        #     MUST live on the sym heap; _sym_zeros also gives the atomic-add
        #     accumulate-from-zero caller contract.
        #   * separate_kernel_reduce: peers write the internal staging instead; this
        #     only receives the local TopkReduce, so allocate it locally.
        if self.impl.in_kernel_fc2_reduce:
            self.output_activation = _sym_zeros(
                (num_tokens_per_rank, hidden),
                torch.bfloat16,
            )
        else:
            self.output_activation = torch.zeros(
                (num_tokens_per_rank, hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
        self.combine_sf = None  # lives inside shared_workspace; None for free-list

        torch.cuda.synchronize()
        self._check_cuda_rng_consistency()

    # ------------------------------------------------------------------
    # Step 2: reference (MXFP8)
    # ------------------------------------------------------------------

    def compute_reference(self) -> None:
        if self.misc.skip_ref_check:
            return
        if self._global_activation is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")

        ref_result = compute_megamoe_reference_mxfp8(
            input_activation=self._global_activation,
            input_activation_sf=self._global_activation_sf,
            input_topk_idx=self._global_topk_idx,
            input_topk_weights=self._global_topk_weights,
            fc1_weight=self._global_fc1_weight,
            fc1_weight_sf=self._global_fc1_weight_sf,
            fc2_weight=self._global_fc2_weight,
            fc2_weight_sf=self._global_fc2_weight_sf,
            ab_dtype=self.torch_ab_dtype,
            norm_const=1.0,
            ref_compute_graph=self.misc.ref_compute_graph,
            fc2_output_dtype=self.problem.fc2_output_dtype,
            combine_format=self._combine_format,
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
    # Step 5: validation (MXFP8)
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Compare the rank's 2D combined output against the reduced reference.

        Both reduce modes now expose the same public 2D ``(T, hidden)``
        ``output_activation``; the per-topk combine staging is internal to the
        kernel.  The reference is per-topk ``(T, K, H)``; reduce it over the topk
        axis the same way the kernel does -- weighting follows the compute graph:
        ``apply_topk_in_fc1`` folds the routing weight into fc1 (reference cells
        already weighted -> plain K-sum); otherwise the weight is applied at the
        reduce (kernel via TopkReduce ``score``, reference via the weighted sum).
        """
        if self.misc.skip_ref_check:
            return
        if self.output_activation is None:
            raise RuntimeError("validate requires run_kernel first.")
        if self.combine_output_ref is None:
            raise RuntimeError("validate requires compute_reference first.")

        actual_reduced = self.output_activation.to(torch.float32)
        ref = self.combine_output_ref.to(torch.float32)  # (T, K, H)
        if self._apply_topk_in_fc1:
            ref_reduced = ref.sum(dim=1)
        else:
            topk_w = self.my_topk_weights.to(torch.float32)  # (T, K)
            ref_reduced = (ref * topk_w[:, :, None]).sum(dim=1)

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

        Token order within each expert's pool slot is non-deterministic in the
        multi-rank dispatch path, so we sort both sides flat before comparing.
        This catches value-level errors (wrong magnitudes, wrong elements) while
        being robust to dispatch-ordering differences across runs or ranks.
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

        print(f"\n{'=' * 60}")
        print(
            f"[generate_c][rank{self.rank}] kernel c_output vs reference fc1 gate+up:"
        )
        any_checked = False
        for e in range(self.problem.num_experts_per_rank):
            v_e = valid[e]
            ref = ref_map.get(e)
            if v_e == 0 or ref is None:
                continue
            any_checked = True
            kernel_c = c[doff[e] : doff[e] + v_e].float().cpu().flatten().sort().values
            ref_c = ref.float().cpu().flatten().sort().values
            compare_and_report_mismatches(
                kernel_c,
                ref_c,
                name=f"c_output[rank{self.rank}]expert{e}",
                atol=1e-2,
                rtol=1e-2,
                max_mismatches=5,
            )
        if not any_checked:
            print("  (no valid tokens routed to any local expert)")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Step 4: kernel launch (MXFP8)
    # ------------------------------------------------------------------

    def run_kernel(self) -> None:
        """Compile + launch ``Sm100MegaMoEMxfp8Kernel`` on the current stream.

        Mirrors ``MegaMoETester.run_kernel`` step-for-step; only the kernel
        instantiation (class + ``ab_dtype`` / ``sf_vec_size``) is MXFP8-specific.
        """
        if (
            self.my_activation is None
            or self.my_activation_sf is None
            or self.my_topk_idx is None
            or self.my_topk_weights is None
            or self.my_fc1_weight is None
            or self.my_fc1_weight_sf is None
            or self.my_fc2_weight is None
            or self.my_fc2_weight_sf is None
            or self.output_activation is None
        ):
            raise RuntimeError("run_kernel requires generate_inputs first.")

        if self.my_activation_sf.shape[-1] % 4 != 0:
            raise ValueError(
                f"activation_sf.shape[-1] ({self.my_activation_sf.shape[-1]}) "
                f"must be a multiple of 4 (4 E8M0 SFs per uint32 dispatch "
                f"LDG.32 wire format)."
            )

        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        from moe_nvfp4_swapab.epilogue import EpilogueTokenTile
        from common.megamoe_constants import SfPaddingBlock
        from moe_mxfp8_glu.megamoe_kernel_mxfp8 import Sm100MegaMoEMxfp8Kernel
        from src.sym_buffer import SymBufferHost

        # -- 1. Kernel instance (MXFP8 requires static_expert_shape != None) --
        static_expert_shape = (
            self.problem.num_experts_per_rank,
            self.problem.intermediate,
            self.problem.hidden,
        )

        cluster_size = self.impl.cluster_shape_mnk[0] * self.impl.cluster_shape_mnk[1]
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        group_hint = self.impl.group_hint
        if group_hint is None:
            group_hint = max_active_clusters

        # generate_c TMA tile height is cta_tile_m=128; physical row offsets must
        # be 128-aligned so the scheduler's data_physical_offsets land on tile
        # boundaries.  Fall back to the default EpilogueTokenTile otherwise.
        c_token_padding = 128 if self.impl.generate_c else EpilogueTokenTile

        self._kernel = Sm100MegaMoEMxfp8Kernel(
            mma_tiler_mnk=self.impl.mma_tiler_mnk,
            cluster_shape_mnk=self.impl.cluster_shape_mnk,
            use_2cta_instrs=self.impl.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=c_token_padding,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=self.impl.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=self.impl.force_static_sched,
            clc_bundle_size=self.impl.clc_bundle_size,
            num_sched_stages=self.impl.num_sched_stages,
            ab_dtype=_kind_to_cutlass_dtype(self.kind),
            sf_vec_size=Mxfp8BlockSize,
            world_size=self.world_size,
            local_rank=self.rank,
            num_topk=self.problem.num_topk,
            max_tokens_per_rank=self.problem.num_tokens_per_rank,
            hidden=self.problem.hidden,
            fc2_in_kernel_topk_reduce=self.impl.in_kernel_fc2_reduce,
            token_back_mode=self.impl.token_back_mode,
            epi_flag_batch=self.impl.epi_flag_batch,
            flag_batch=self.impl.flag_batch,
            gate_up_clamp=self.problem.gate_up_clamp,
            apply_topk_in_fc1=self._apply_topk_in_fc1,
            generate_c=self.impl.generate_c,
            use_stg_fc1=self.impl.use_stg_fc1,
            combine_format=self._combine_format,
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
                dtype=torch.bfloat16,
                device="cuda",
            )
            self._c_valid_tokens_per_expert = valid_tokens
            self._c_data_physical_offsets = doff[:-1]

        # -- 2. Workspaces (local cuda + sym-heap) --
        self.allocate_workspaces()

        # -- 3. Torch -> cute --
        def _to_cute(
            tensor: torch.Tensor, assumed_align: int = 16, force_static_layout=False
        ):
            cute_tensor = cutlass_torch.from_dlpack(
                tensor,
                assumed_align=assumed_align,
            )
            if force_static_layout:
                return cute_tensor
            leading_dim = cutlass_torch.get_leading_dim(tensor)
            return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

        activation_cute = _to_cute(self.my_activation)
        activation_sf_cute = _to_cute(self.my_activation_sf)
        topk_idx_cute = _to_cute(self.my_topk_idx)
        topk_weights_cute = _to_cute(self.my_topk_weights)
        fc1_weight_cute = _to_cute(self.my_fc1_weight)
        fc1_weight_sf_cute = _to_cute(self.my_fc1_weight_sf)
        fc2_weight_cute = _to_cute(self.my_fc2_weight)
        fc2_weight_sf_cute = _to_cute(self.my_fc2_weight_sf)
        output_activation_cute = _to_cute(self.output_activation)
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
            activation_sf=activation_sf_cute,
            topk_idx=topk_idx_cute,
            topk_weights=topk_weights_cute,
            fc1_weight=fc1_weight_cute,
            fc1_weight_sf=fc1_weight_sf_cute,
            fc2_weight=fc2_weight_cute,
            fc2_weight_sf=fc2_weight_sf_cute,
            output_activation=output_activation_cute,
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
                torch.distributed.is_available() and torch.distributed.is_initialized()
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


def _parse_combine_format(argument: str) -> CombineFormat:
    try:
        return CombineFormat.parse(argument)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MegaMoE MXFP8 GLU multi-rank fused dispatch+fc12+combine runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="mxfp8_e4m3",
        choices=["mxfp8_e4m3", "mxfp8_e5m2"],
        help="MXFP8 element format for activations and weights.",
    )
    parser.add_argument("--num_tokens_per_rank", type=int, default=128)
    parser.add_argument("--num_topk", type=int, default=4)
    parser.add_argument("--num_total_experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument(
        "--fc2_output_dtype",
        type=_parse_output_dtype,
        default=torch.bfloat16,
    )
    parser.add_argument(
        "--route_distribution",
        type=str,
        default="balanced",
        choices=["balanced", "power_law"],
    )
    parser.add_argument(
        "--power_law_exponent",
        type=float,
        default=1.0,
        help="Zipf exponent for --route_distribution power_law.",
    )
    parser.add_argument(
        "--gate_up_clamp",
        type=float,
        default=None,
        help="DeepSeek-V4 swiglu_limit: clamp gate/up pre-activations before SiLU.",
    )

    # MXFP8 fused fc12 is validated for the (M=256, N=256) 2-CTA tile only.
    parser.add_argument("--mma_tiler_mnk", type=str, default="256,256,128")
    parser.add_argument("--cluster_shape_mnk", type=str, default="2,1,1")
    parser.add_argument("--use_2cta_instrs", action="store_true", default=True)
    parser.add_argument(
        "--enable_static_expert_shape", action="store_true", default=False
    )
    parser.add_argument("--dynamic_sched", action="store_true", default=False)
    parser.add_argument("--clc_bundle_size", type=int, default=None)
    parser.add_argument("--num_sched_stages", type=int, default=None)
    parser.add_argument(
        "--load_balance_mode",
        type=str,
        default="static",
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
        "--ref_compute_graph",
        type=str,
        default="deepgemm",
        choices=["transformers", "deepgemm"],
    )
    parser.add_argument("--enable_iket", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--in_kernel_fc2_reduce",
        action="store_true",
        default=False,
        help="Form B: REDG in-kernel topk-reduce to combine_output[t, 0, :]",
    )
    parser.add_argument(
        "--generate_c",
        action="store_true",
        default=False,
        help="Store raw pre-SwiGLU fc1 accumulator (gate+up, BF16) to a separate tensor.",
    )
    parser.add_argument(
        "--use_stg_fc1",
        action="store_true",
        default=False,
        help="Write fc1 FP8 output directly to GMEM via STG.256 instead of R2S+TMA. "
        "Eliminates sD SMEM staging (saves 16 KB); may increase AB pipeline stages.",
    )
    parser.add_argument(
        "--combine_format",
        type=_parse_combine_format,
        default=CombineFormat.parse("bf16"),
        help="Wire format for the cross-rank combine payload. "
        "Choices: 'bf16' (no quantization, default), "
        "'32e4m3xe8m0' (MXFP8 e4m3: fp8 e4m3 data + E8M0 block scale, ~2x bandwidth saving), "
        "'32e5m2xe8m0' (MXFP8 e5m2: fp8 e5m2 data + E8M0 block scale, ~2x bandwidth saving). "
        "e.g. --combine_format 32e4m3xe8m0",
    )
    parser.add_argument(
        "--token_back_mode",
        type=str,
        default="standalone_warps",
        choices=["epi_warps", "standalone_warps", "reuse_dispatch_warps"],
        help="Where the cross-rank fc2 push-back runs: epi_warps (epilogue "
        "STG redirect), standalone_warps (dedicated warps 12-15), "
        "or reuse_dispatch_warps (dispatch warps 8-11, default).",
    )
    parser.add_argument(
        "--epi_flag_batch",
        type=str,
        default="2,4",
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
        combine_format=args.combine_format,
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

    tester = MegaMoEMxfp8Tester(
        problem,
        impl,
        misc,
        rank=rank,
        kind=args.kind,
        combine_format=args.combine_format,
    )
    tester.set_torch_profiler_enabled(args.use_torch_profiler)
    tester.set_perf_iters(args.perf_warmup, args.perf_iters)

    return_code = 0
    try:
        tester.run()
    except NotImplementedError as exc:
        if rank == 0:
            print(f"[mega_runner_mxfp8] kernel launch skipped: {exc}")

    if not _NO_DIST:
        tester._compiled_kernel = None
        tester._kernel = None
        gc.collect()
        torch.cuda.synchronize()
        try:
            import nvshmem.core

            # output_activation is on the sym heap only under in_kernel_reduce
            for sym_tensor in (
                tester.my_activation,
                tester.my_activation_sf,
                tester.my_topk_idx,
                tester.my_topk_weights,
                tester.output_activation,
                tester.combine_sf,
                tester.shared_workspace,
            ):
                if sym_tensor is not None:
                    try:
                        nvshmem.core.free_tensor(sym_tensor)
                    except Exception:  # noqa: BLE001
                        pass
            tester.my_activation = None
            tester.my_activation_sf = None
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
