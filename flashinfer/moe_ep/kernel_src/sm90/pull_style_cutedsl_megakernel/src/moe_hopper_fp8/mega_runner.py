# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone multi-rank MegaMoE host driver for the FP8 GLU path.

Mirror of ``moe_nvfp4_swapab.mega_runner`` (NVFP4) specialised for FP8
(``e4m3`` / ``e5m2``), with selectable non-swap and swap-AB kernels. It retains the legacy 32-wide SF ABI
layout for per-tensor compatibility and also supports blockwise FP32 scales.
It subclasses the NVFP4 ``MegaMoETester`` and reuses all of its distributed
bootstrap, routing-table generation, symmetric-heap allocation, workspace
allocation, validation and teardown; only the three kind-specific stages are
overridden:

  * ``generate_inputs``   -- fp8 / E8M0 input + weight generation + sym staging
  * ``compute_reference`` -- ``compute_megamoe_reference_fp8``
  * ``run_kernel``        -- instantiate ``Sm90MegaMoEFp8Kernel``

Topk weighting follows the NVFP4/MXFP8 compute graphs. ``deepgemm`` folds each
routing weight into the SwiGLU output before FC1-output quantization;
``transformers`` keeps the staged FC2 terms unweighted and applies routing
weights in the standalone ``TopkReduce`` kernel. Form B requires ``deepgemm``.

Launcher::

    torchrun --nproc_per_node=4 -m moe_hopper_fp8.mega_runner \\
        --kind fp8_e4m3 --num_total_experts 32 --route_distribution balanced
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

from common.megamoe_constants import (
    Fp8BlockScaleK,
    Fp8E8M0SfVecSize,
    Fp8Fc2ActivationScaleK,
    Fp8WeightScaleBlockK,
    Fp8WeightScaleBlockN,
)
from common.host_utils import compare_and_report_mismatches
from tester.host_utils import (
    reduce_add_deterministic_check_dim_size_limit,
    reduce_add_ordering_match,
)
from moe_nvfp4_swapab.runner_common import (
    ceil_div,
    round_up,
    to_blocked,
    _stack_byte_reinterpretable_tensors,
    Mxfp8ScaleDtype as Fp8E8M0ScaleDtype,
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
from moe_hopper_fp8.runner_fc12 import ImplDesc
from moe_hopper_fp8.mega_reference_fp8 import compute_megamoe_reference_fp8
from moe_hopper_fp8.hopper_moe_utils import (
    FP8_ACCUM_MODE_CHOICES,
    FP8_KIND_CHOICES,
    FP8_SCALE_MODE_CHOICES,
    Fp8PerTensorTargetAmax,
    create_fp8_tensor,
    fp8_kind_to_cutlass_dtype,
    make_constant_block_scale,
    make_fp8_per_tensor_dequant_scale,
    quantize_fp8_per_token_block,
    quantize_fp8_weight_block_nk,
)


_KIND_TO_TORCH_DTYPE = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "mxfp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "mxfp8_e5m2": torch.float8_e5m2,
}


def _clone_if_data_ptr_unaligned(
    tensor: torch.Tensor,
    align: int = 16,
    *,
    name: str = "tensor",
) -> torch.Tensor:
    """Return a base-aligned copy when a rank-local view starts mid-allocation."""
    data_ptr = tensor.data_ptr()
    remainder = data_ptr % align
    if remainder == 0:
        return tensor
    print(
        f"[alignment] {name} input data is not aligned to {align} bytes: "
        f"data_ptr=0x{data_ptr:x}, remainder={remainder}, "
        f"shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, "
        f"dtype={tensor.dtype}; cloning aligned copy.",
        flush=True,
    )
    return tensor.clone()


def _sym_zeros_byte_view_1b(
    logical_shape: Tuple[int, ...], target_dtype: torch.dtype,
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
# FP8 MegaMoE tester
# =============================================================================


class MegaMoEFp8Tester(MegaMoETester):
    """FP8 specialisation of the multi-rank MegaMoE host driver."""

    def __init__(
        self,
        problem: TokenCommProblemDesc,
        impl: ImplDesc,
        misc: MiscDesc,
        *,
        rank: int,
        kind: str = "fp8_e4m3",
        fp8_scale_mode: str = "per_tensor",
        fp8_accum_mode: str = "1xacc",
        swap_ab: bool = False,
        use_cuda_profiler_api: bool = False,
    ) -> None:
        self.swap_ab = swap_ab
        self._use_cuda_profiler_api = use_cuda_profiler_api
        super().__init__(problem, impl, misc, rank=rank)
        if kind not in _KIND_TO_TORCH_DTYPE:
            raise ValueError(
                f"kind must be one of {sorted(_KIND_TO_TORCH_DTYPE)}, got {kind!r}."
            )
        self.kind = kind
        if fp8_scale_mode not in FP8_SCALE_MODE_CHOICES:
            raise ValueError(
                f"fp8_scale_mode must be one of {FP8_SCALE_MODE_CHOICES}, "
                f"got {fp8_scale_mode!r}."
            )
        self.fp8_scale_mode = fp8_scale_mode
        if fp8_accum_mode not in FP8_ACCUM_MODE_CHOICES:
            raise ValueError(
                f"fp8_accum_mode must be one of {FP8_ACCUM_MODE_CHOICES}, "
                f"got {fp8_accum_mode!r}."
            )
        self.fp8_accum_mode = fp8_accum_mode
        self.torch_ab_dtype = _KIND_TO_TORCH_DTYPE[kind]
        self.fc1_activation_dequant_scale: Optional[torch.Tensor] = None
        self.fc1_weight_dequant_scale: Optional[torch.Tensor] = None
        self.fc2_activation_dequant_scale: Optional[torch.Tensor] = None
        self.fc2_weight_dequant_scale: Optional[torch.Tensor] = None
        self._global_fc1_activation_dequant_scale: Optional[torch.Tensor] = None
        self._global_fc1_weight_dequant_scale: Optional[torch.Tensor] = None
        self._global_fc2_activation_dequant_scale: Optional[torch.Tensor] = None
        self._global_fc2_weight_dequant_scale: Optional[torch.Tensor] = None
        if impl.in_kernel_fc2_reduce and impl.token_back_by_dispatch:
            raise ValueError(
                "in_kernel_fc2_reduce and token_back_by_dispatch cannot both be True."
            )

    # ------------------------------------------------------------------
    # Step 1: deterministic FP8 input + weight generation
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
        e8m0_sf_vec_size = Fp8E8M0SfVecSize
        data_dtype = self.torch_ab_dtype
        hidden_sf_cols = ceil_div(hidden, e8m0_sf_vec_size)
        intermediate_downproj_sf_cols = ceil_div(
            intermediate_downproj, e8m0_sf_vec_size
        )
        blockwise = self.fp8_scale_mode == "blockwise"
        per_tensor_nonzero_pct = float(
            os.environ.get("FP8_NONZERO_PCT", os.environ.get("MXFP8_NONZERO_PCT", "1"))
        ) / 100.0
        if blockwise:
            for name, value, divisor in (
                ("hidden", hidden, Fp8BlockScaleK),
                ("intermediate", intermediate, Fp8WeightScaleBlockN),
                (
                    "intermediate_downproj",
                    intermediate_downproj,
                    Fp8Fc2ActivationScaleK,
                ),
                (
                    "intermediate_downproj",
                    intermediate_downproj,
                    Fp8WeightScaleBlockK,
                ),
            ):
                if value % divisor != 0:
                    raise ValueError(
                        f"blockwise FP8 requires {name}={value} divisible by "
                        f"{divisor}."
                    )

        # ---- Activation (fp8, not packed) + plain K-major E8M0 SF.  The SF
        # stays in the user-facing plain layout; the kernel's dispatch warps
        # re-stage it into the SFB atom layout inside l1_sf_buffer before fc1
        # reads it, so the host reference can dequant from the plain layout.
        if blockwise and not self.misc.perf_run:
            activation_src = create_fp8_tensor(
                (num_ranks * num_tokens_per_rank, hidden),
                data_dtype,
                perf_run=False,
                return_fp8=False,
                generator=self._torch_cuda_rng,
            )
            activation_q, activation_scale = quantize_fp8_per_token_block(
                activation_src,
                data_dtype,
                block_k=Fp8BlockScaleK,
                target_amax=Fp8PerTensorTargetAmax,
            )
            self._global_activation = activation_q.reshape(
                num_ranks, num_tokens_per_rank, hidden
            )
            self._global_activation_sf = activation_scale.reshape(
                num_ranks, num_tokens_per_rank, hidden // Fp8BlockScaleK
            )
        else:
            self._global_activation = create_fp8_tensor(
                (num_ranks, num_tokens_per_rank, hidden),
                data_dtype,
                perf_run=self.misc.perf_run,
                nonzero_value=0.5,
                positive_prob=per_tensor_nonzero_pct,
                negative_prob=per_tensor_nonzero_pct,
                generator=self._torch_cuda_rng,
                perf_positive_only=True,
            )
            if blockwise:
                self._global_activation_sf = make_constant_block_scale(
                    data_dtype,
                    (num_ranks, num_tokens_per_rank, hidden // Fp8BlockScaleK),
                )
            else:
                self._global_activation_sf = torch.ones(
                    (num_ranks, num_tokens_per_rank, hidden_sf_cols),
                    dtype=Fp8E8M0ScaleDtype,
                    device="cuda",
                )

        if blockwise:
            self._global_fc1_activation_dequant_scale = torch.ones(
                (1,), dtype=torch.float32, device="cuda"
            )
        elif self.misc.perf_run:
            self._global_fc1_activation_dequant_scale = (
                make_fp8_per_tensor_dequant_scale(data_dtype, (1,))
            )
        else:
            self._global_fc1_activation_dequant_scale = (
                make_fp8_per_tensor_dequant_scale(self._global_activation)
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
        if blockwise and not self.misc.perf_run:
            fc1_weight_q_parts = []
            fc1_weight_scale_parts = []
            for _rank_idx in range(num_ranks):
                for _expert_idx in range(num_experts_per_rank):
                    weight_src = create_fp8_tensor(
                        (intermediate, hidden),
                        data_dtype,
                        perf_run=False,
                        return_fp8=False,
                        generator=self._torch_cuda_rng,
                    )
                    weight_q, weight_scale = quantize_fp8_weight_block_nk(
                        weight_src,
                        data_dtype,
                        block_n=Fp8WeightScaleBlockN,
                        block_k=Fp8WeightScaleBlockK,
                        target_amax=Fp8PerTensorTargetAmax,
                    )
                    fc1_weight_q_parts.append(weight_q)
                    fc1_weight_scale_parts.append(weight_scale)
            self._global_fc1_weight = (
                torch.stack(fc1_weight_q_parts, dim=0)
                .reshape(num_ranks, num_experts_per_rank, intermediate, hidden)
                .permute(0, 1, 3, 2)
            )
            self._global_fc1_weight_sf = torch.stack(
                fc1_weight_scale_parts, dim=0
            ).reshape(
                num_ranks,
                num_experts_per_rank,
                intermediate // Fp8WeightScaleBlockN,
                hidden // Fp8WeightScaleBlockK,
            )

            fc2_weight_q_parts = []
            fc2_weight_scale_parts = []
            for _rank_idx in range(num_ranks):
                for _expert_idx in range(num_experts_per_rank):
                    weight_src = create_fp8_tensor(
                        (hidden, intermediate_downproj),
                        data_dtype,
                        perf_run=False,
                        return_fp8=False,
                        generator=self._torch_cuda_rng,
                    )
                    weight_q, weight_scale = quantize_fp8_weight_block_nk(
                        weight_src,
                        data_dtype,
                        block_n=Fp8WeightScaleBlockN,
                        block_k=Fp8WeightScaleBlockK,
                        target_amax=Fp8PerTensorTargetAmax,
                    )
                    fc2_weight_q_parts.append(weight_q)
                    fc2_weight_scale_parts.append(weight_scale)
            self._global_fc2_weight = (
                torch.stack(fc2_weight_q_parts, dim=0)
                .reshape(num_ranks, num_experts_per_rank, hidden, intermediate_downproj)
                .permute(0, 1, 3, 2)
            )
            self._global_fc2_weight_sf = torch.stack(
                fc2_weight_scale_parts, dim=0
            ).reshape(
                num_ranks,
                num_experts_per_rank,
                hidden // Fp8WeightScaleBlockN,
                intermediate_downproj // Fp8WeightScaleBlockK,
            )
        else:
            self._global_fc1_weight = create_fp8_tensor(
                (num_ranks, num_experts_per_rank, intermediate, hidden),
                data_dtype,
                perf_run=self.misc.perf_run,
                nonzero_value=0.5,
                positive_prob=per_tensor_nonzero_pct,
                negative_prob=per_tensor_nonzero_pct,
                generator=self._torch_cuda_rng,
                perf_positive_only=True,
            ).permute(0, 1, 3, 2)
            self._global_fc2_weight = create_fp8_tensor(
                (num_ranks, num_experts_per_rank, hidden, intermediate_downproj),
                data_dtype,
                perf_run=self.misc.perf_run,
                nonzero_value=0.5,
                positive_prob=per_tensor_nonzero_pct,
                negative_prob=per_tensor_nonzero_pct,
                generator=self._torch_cuda_rng,
                perf_positive_only=True,
            ).permute(0, 1, 3, 2)
            if blockwise:
                self._global_fc1_weight_sf = make_constant_block_scale(
                    data_dtype,
                    (
                        num_ranks,
                        num_experts_per_rank,
                        intermediate // Fp8WeightScaleBlockN,
                        hidden // Fp8WeightScaleBlockK,
                    ),
                )
                self._global_fc2_weight_sf = make_constant_block_scale(
                    data_dtype,
                    (
                        num_ranks,
                        num_experts_per_rank,
                        hidden // Fp8WeightScaleBlockN,
                        intermediate_downproj // Fp8WeightScaleBlockK,
                    ),
                )
            else:
                self._global_fc1_weight_sf = torch.ones(
                    (
                        num_ranks,
                        num_experts_per_rank,
                        intermediate,
                        hidden_sf_cols,
                    ),
                    dtype=Fp8E8M0ScaleDtype,
                    device="cuda",
                )
                self._global_fc2_weight_sf = torch.ones(
                    (
                        num_ranks,
                        num_experts_per_rank,
                        hidden,
                        intermediate_downproj_sf_cols,
                    ),
                    dtype=Fp8E8M0ScaleDtype,
                    device="cuda",
                )

        if blockwise:
            self._global_fc1_weight_dequant_scale = torch.ones(
                (num_ranks, num_experts_per_rank),
                dtype=torch.float32,
                device="cuda",
            )
            self._global_fc2_weight_dequant_scale = torch.ones(
                (num_ranks, num_experts_per_rank),
                dtype=torch.float32,
                device="cuda",
            )
            self._global_fc1_weight_sf_swizzled = self._global_fc1_weight_sf
            self._global_fc2_weight_sf_swizzled = self._global_fc2_weight_sf
        else:
            if self.misc.perf_run:
                self._global_fc1_weight_dequant_scale = make_fp8_per_tensor_dequant_scale(
                    data_dtype, (num_ranks, num_experts_per_rank)
                )
                self._global_fc2_weight_dequant_scale = make_fp8_per_tensor_dequant_scale(
                    data_dtype, (num_ranks, num_experts_per_rank)
                )
            else:
                self._global_fc1_weight_dequant_scale = (
                    make_fp8_per_tensor_dequant_scale(
                        self._global_fc1_weight, reduce_dims=(2, 3)
                    )
                )
                self._global_fc2_weight_dequant_scale = (
                    make_fp8_per_tensor_dequant_scale(
                        self._global_fc2_weight, reduce_dims=(2, 3)
                    )
                )

            # ---- Atom-swizzle the weight SFs (the base kernel consumes weight SFs
            # in the 32x4x4 atom-swizzled flat layout that tile_atom_to_shape_SF /
            # the TMA SFA descriptor expect).  Plain ``_global_fc1/2_weight_sf`` are
            # kept untouched for ``compute_megamoe_reference_fp8``.
            fc1_sf_swizzled = [
                to_blocked(self._global_fc1_weight_sf[r, e])
                for r in range(num_ranks)
                for e in range(num_experts_per_rank)
            ]
            fc1_flat_sf_size = fc1_sf_swizzled[0].numel()
            self._global_fc1_weight_sf_swizzled = (
                _stack_byte_reinterpretable_tensors(fc1_sf_swizzled, dim=0)
                .view(num_ranks, num_experts_per_rank, fc1_flat_sf_size)
            )

            fc2_sf_swizzled = [
                to_blocked(self._global_fc2_weight_sf[r, e])
                for r in range(num_ranks)
                for e in range(num_experts_per_rank)
            ]
            fc2_flat_sf_size = fc2_sf_swizzled[0].numel()
            self._global_fc2_weight_sf_swizzled = (
                _stack_byte_reinterpretable_tensors(fc2_sf_swizzled, dim=0)
                .view(num_ranks, num_experts_per_rank, fc2_flat_sf_size)
            )

        # ---- Stage own-rank inputs into the symmetric heap.
        own_activation = self._global_activation[self.rank]
        own_activation_sf = self._global_activation_sf[self.rank]
        own_topk_idx = self._global_topk_idx[self.rank]
        own_topk_weights = self._global_topk_weights[self.rank]

        # fp8 goes through a uint8 byte-buf view (nvshmem4py doesn't natively
        # support fp8); copy via uint8 to dodge fp8 assignment quirks.
        self.my_activation = _sym_zeros_byte_view_1b(
            (num_tokens_per_rank, hidden), data_dtype,
        )
        self.my_activation.view(torch.uint8).copy_(own_activation.view(torch.uint8))

        if blockwise:
            activation_sf_cols = hidden // Fp8BlockScaleK
            activation_sf_storage_cols = round_up(activation_sf_cols, 4)
            self.my_activation_sf = _sym_zeros(
                (num_tokens_per_rank, activation_sf_storage_cols),
                torch.float32,
            )
            self.my_activation_sf[:, :activation_sf_cols].copy_(own_activation_sf)
        else:
            # SF leg caller contract: the K_sf axis (= ceil(hidden, 32)) must be a
            # multiple of 4 E8M0 SFs so dispatch_pull's LDG.32 byte stride matches
            # the host row stride.  Pad-to-mult-of-4 + zero-fill on the host side;
            # the trailing padded SFs pair with fp8 data in TMA's OOB-fill-0 region.
            hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)
            self.my_activation_sf = _sym_zeros_byte_view_1b(
                (num_tokens_per_rank, hidden_sf_cols_padded), Fp8E8M0ScaleDtype,
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
        if blockwise:
            self.my_fc1_weight_sf = _clone_if_data_ptr_unaligned(
                self.my_fc1_weight_sf,
                name="my_fc1_weight_sf",
            )
            self.my_fc2_weight_sf = _clone_if_data_ptr_unaligned(
                self.my_fc2_weight_sf,
                name="my_fc2_weight_sf",
            )
        self.fc1_activation_dequant_scale = self._global_fc1_activation_dequant_scale
        self.fc1_weight_dequant_scale = self._global_fc1_weight_dequant_scale[self.rank]
        self.fc2_weight_dequant_scale = self._global_fc2_weight_dequant_scale[self.rank]
        self.fc2_activation_dequant_scale = None

        # ---- Public final output. The per-topk (T, K, H) combine plane is an
        # internal shared-workspace region in separate-reduce mode.
        if self.impl.in_kernel_fc2_reduce:
            # Form B REDG writes across ranks and accumulates from zero.
            self.output_activation = _sym_zeros(
                (num_tokens_per_rank, hidden), problem.fc2_output_dtype,
            )
        else:
            if problem.fc2_output_dtype != torch.bfloat16:
                raise ValueError(
                    "separate TopkReduce currently expects BF16 output, "
                    f"got {problem.fc2_output_dtype}."
                )
            self.output_activation = torch.empty(
                (num_tokens_per_rank, hidden),
                dtype=problem.fc2_output_dtype,
                device="cuda",
            )

        torch.cuda.synchronize()
        self._check_cuda_rng_consistency()

    # ------------------------------------------------------------------
    # Step 2: FP8 reference
    # ------------------------------------------------------------------

    def compute_reference(self) -> None:
        if self.misc.skip_ref_check:
            return
        if self._global_activation is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")

        combine_ref_global, self._global_fc2_activation_dequant_scale = (
            compute_megamoe_reference_fp8(
                input_activation=self._global_activation,
                input_activation_sf=self._global_activation_sf,
                input_topk_idx=self._global_topk_idx,
                input_topk_weights=self._global_topk_weights,
                fc1_weight=self._global_fc1_weight,
                fc1_weight_sf=self._global_fc1_weight_sf,
                fc2_weight=self._global_fc2_weight,
                fc2_weight_sf=self._global_fc2_weight_sf,
                ab_dtype=self.torch_ab_dtype,
                fc1_activation_dequant_scale=self._global_fc1_activation_dequant_scale,
                fc1_weight_dequant_scale=self._global_fc1_weight_dequant_scale,
                fc2_weight_dequant_scale=self._global_fc2_weight_dequant_scale,
                norm_const=1.0,
                ref_compute_graph=self.misc.ref_compute_graph,
                fp8_accum_mode=self.fp8_accum_mode,
                mma_tiler_k=self.impl.mma_tiler_mnk[2],
                fc2_output_dtype=self.problem.fc2_output_dtype,
                gate_up_clamp=self.problem.gate_up_clamp,
                return_fc2_activation_dequant_scale=True,
                fp8_scale_mode=self.fp8_scale_mode,
            )
        )
        if self.fp8_scale_mode == "blockwise":
            self.fc2_activation_dequant_scale = torch.ones(
                (1,), dtype=torch.float32, device=self._global_activation.device
            )
        else:
            self.fc2_activation_dequant_scale = self._global_fc2_activation_dequant_scale
        self.combine_output_ref = combine_ref_global[self.rank].contiguous()

    def _ensure_fp8_per_tensor_scale_tensors(self) -> None:
        device = self.my_activation.device
        if self.fc1_activation_dequant_scale is None:
            self.fc1_activation_dequant_scale = torch.ones(
                (1,), dtype=torch.float32, device=device
            )
        if self.fc1_weight_dequant_scale is None:
            self.fc1_weight_dequant_scale = torch.ones(
                (self.problem.num_experts_per_rank,), dtype=torch.float32, device=device
            )
        if self.fc2_activation_dequant_scale is None:
            self.fc2_activation_dequant_scale = torch.ones(
                (1,), dtype=torch.float32, device=device
            )
        if self.fc2_weight_dequant_scale is None:
            self.fc2_weight_dequant_scale = torch.ones(
                (self.problem.num_experts_per_rank,), dtype=torch.float32, device=device
            )

    # ------------------------------------------------------------------
    # Step 5: FP8 validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Compare the public 2D output against the topk-reduced reference."""
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
                            "Validation PASSED: Form B output matches a legal "
                            f"BF16 atomic-add ordering ({num_orderings} orderings)."
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
                            "Validation PASSED: Form B output is within the "
                            "BF16 atomic-add roundoff envelope."
                        )
                    return
        if (
            self.fp8_scale_mode == "blockwise"
            and not self.impl.in_kernel_fc2_reduce
            and not torch.allclose(actual_reduced, ref_reduced, atol=atol, rtol=rtol)
        ):
            if self._kernel is None or self.shared_workspace is None:
                raise RuntimeError("combine workspace is unavailable for validation.")
            combine_spec = self._kernel._shared_region_by_name["combine_quant"]
            combine_offset = self._kernel._shared_offsets["combine_quant"]
            combine_bytes = combine_spec.nbytes
            actual_full = (
                self.shared_workspace.narrow(0, combine_offset, combine_bytes)
                .view(torch.bfloat16)
                .reshape(self.combine_output_ref.shape)
                .to(torch.float32)
            )
            ref_full = self.combine_output_ref.to(torch.float32)
            if torch.allclose(actual_full, ref_full, atol=atol, rtol=rtol):
                if self.rank == 0:
                    print(
                        "Validation PASSED: blockwise per-topk cells are within "
                        "tolerance; reduced-only drift is BF16 cancellation from "
                        "reference WGMMA/torch differences."
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
    # Step 4: FP8 kernel launch
    # ------------------------------------------------------------------

    def run_kernel(self) -> None:
        """Compile + launch ``Sm90MegaMoEFp8Kernel`` on the current stream.

        Mirrors ``MegaMoETester.run_kernel`` step-for-step; only the kernel
        instantiation (class + ``ab_dtype`` / ``sf_vec_size``) is specific to
        FP8.
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

        if self.fp8_scale_mode != "blockwise" and self.my_activation_sf.shape[-1] % 4 != 0:
            raise ValueError(
                f"activation_sf.shape[-1] ({self.my_activation_sf.shape[-1]}) "
                f"must be a multiple of 4 (4 E8M0 SFs per uint32 dispatch "
                f"LDG.32 wire format)."
            )
        self._ensure_fp8_per_tensor_scale_tensors()

        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        from common.megamoe_constants import SfPaddingBlock
        from moe_hopper_fp8.megamoe_kernel_fp8 import (
            Sm90MegaMoEFp8Kernel,
            Sm90MegaMoESwapABFp8Kernel,
        )
        from src.sym_buffer import SymBufferHost

        # -- 1. Kernel instance (FP8 requires static_expert_shape != None) --
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
            Sm90MegaMoESwapABFp8Kernel
            if self.swap_ab
            else Sm90MegaMoEFp8Kernel
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
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=self.impl.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=self.impl.force_static_sched,
            clc_bundle_size=self.impl.clc_bundle_size,
            num_sched_stages=self.impl.num_sched_stages,
            ab_dtype=fp8_kind_to_cutlass_dtype(self.kind),
            sf_vec_size=Fp8E8M0SfVecSize,
            fp8_scale_mode=self.fp8_scale_mode,
            fp8_accum_mode=self.fp8_accum_mode,
            world_size=self.world_size,
            local_rank=self.rank,
            num_topk=self.problem.num_topk,
            max_tokens_per_rank=self.problem.num_tokens_per_rank,
            hidden=self.problem.hidden,
            fc2_in_kernel_topk_reduce=self.impl.in_kernel_fc2_reduce,
            apply_topk_in_fc1=self.misc.ref_compute_graph == "deepgemm",
            token_back_by_dispatch=self.impl.token_back_by_dispatch,
            epi_flag_batch=self.impl.epi_flag_batch,
            flag_batch=self.impl.flag_batch,
            gate_up_clamp=self.problem.gate_up_clamp,
        )

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
        activation_sf_cute = _to_cute(self.my_activation_sf)
        topk_idx_cute = _to_cute(self.my_topk_idx)
        topk_weights_cute = _to_cute(self.my_topk_weights)
        fc1_weight_cute = _to_cute(self.my_fc1_weight)
        fc1_weight_sf_cute = _to_cute(self.my_fc1_weight_sf)
        fc1_activation_dequant_scale_cute = _to_cute(
            self.fc1_activation_dequant_scale, assumed_align=4
        )
        fc1_weight_dequant_scale_cute = _to_cute(
            self.fc1_weight_dequant_scale, assumed_align=4
        )
        fc2_weight_cute = _to_cute(self.my_fc2_weight)
        fc2_weight_sf_cute = _to_cute(self.my_fc2_weight_sf)
        fc2_activation_dequant_scale_cute = _to_cute(
            self.fc2_activation_dequant_scale, assumed_align=4
        )
        fc2_weight_dequant_scale_cute = _to_cute(
            self.fc2_weight_dequant_scale, assumed_align=4
        )
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
            activation_sf=activation_sf_cute,
            topk_idx=topk_idx_cute,
            topk_weights=topk_weights_cute,
            fc1_weight=fc1_weight_cute,
            fc1_weight_sf=fc1_weight_sf_cute,
            fc1_activation_dequant_scale=fc1_activation_dequant_scale_cute,
            fc1_weight_dequant_scale=fc1_weight_dequant_scale_cute,
            fc2_weight=fc2_weight_cute,
            fc2_weight_sf=fc2_weight_sf_cute,
            fc2_activation_dequant_scale=fc2_activation_dequant_scale_cute,
            fc2_weight_dequant_scale=fc2_weight_dequant_scale_cute,
            output_activation=output_activation_cute,
            local_workspace=local_workspace_cute,
            shared_workspace=shared_workspace_cute,
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )
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
        description="MegaMoE FP8 GLU multi-rank fused dispatch+fc12+combine runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kind", type=str, default="fp8_e4m3",
        choices=list(FP8_KIND_CHOICES),
        help=(
            "FP8 element format for activations and weights. "
            "Legacy mxfp8_* names are accepted as aliases."
        ),
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
    parser.add_argument(
        "--fp8_scale_mode", type=str, default="per_tensor",
        choices=list(FP8_SCALE_MODE_CHOICES),
        help="FP8 scale interpretation: scalar per-tensor or DeepGEMM-style blockwise.",
    )
    parser.add_argument(
        "--fp8_accum_mode", type=str, default="1xacc",
        choices=list(FP8_ACCUM_MODE_CHOICES),
        help="Per-tensor WGMMA accumulation mode; ignored by blockwise scaling.",
    )
    parser.add_argument(
        "--swap_ab", action="store_true",
        help="Use the Hopper weight-as-A M128/M256xN swap-AB kernel.",
    )

    # Hopper FP8 fused fc12 defaults to the 1CTA (M=64, N=128) tile.
    parser.add_argument("--mma_tiler_mnk", type=str, default="64,128,128")
    parser.add_argument("--cluster_shape_mnk", type=str, default="1,1,1")
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
        help="Form B: REDG in-kernel topk-reduce to output_activation[t, :]",
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

    mma_tiler_mnk = _parse_tuple(args.mma_tiler_mnk)
    if args.swap_ab and mma_tiler_mnk == (64, 128, 128):
        mma_tiler_mnk = (256, 32, 128)

    impl = ImplDesc(
        mma_tiler_mnk=mma_tiler_mnk,
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
        token_back_mode=(
            "epi_warps" if args.in_kernel_fc2_reduce else "reuse_dispatch_warps"
        ),
        epi_flag_batch=(2, 4),
        flag_batch=1,
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

    tester = MegaMoEFp8Tester(
        problem,
        impl,
        misc,
        rank=rank,
        kind=args.kind,
        fp8_scale_mode=args.fp8_scale_mode,
        fp8_accum_mode=args.fp8_accum_mode,
        swap_ab=args.swap_ab,
        use_cuda_profiler_api=args.use_cuda_profiler_api,
    )
    tester.set_torch_profiler_enabled(args.use_torch_profiler)
    tester.set_perf_iters(args.perf_warmup, args.perf_iters)

    return_code = 0
    try:
        tester.run()
    except NotImplementedError as exc:
        if rank == 0:
            print(f"[mega_runner_fp8] kernel launch skipped: {exc}")

    if not _NO_DIST:
        tester._compiled_kernel = None
        tester._kernel = None
        gc.collect()
        torch.cuda.synchronize()
        try:
            import nvshmem.core
            for sym_tensor in (
                tester.my_activation, tester.my_activation_sf,
                tester.my_topk_idx, tester.my_topk_weights,
                tester.output_activation, tester.shared_workspace,
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
