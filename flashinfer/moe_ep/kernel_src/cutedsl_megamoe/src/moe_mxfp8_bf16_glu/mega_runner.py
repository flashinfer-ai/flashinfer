# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MegaMoE host runner for MXFP8 weights and BF16 activations.

This runner deliberately derives from the BF16 MegaMoE runner. Dispatch,
routing, token-back buffers, BF16 activations, symmetric-heap allocations, and
validation therefore keep the BF16 contract. Only the two expert weights are
replaced with MXFP8 data plus per-K32 E8M0 scale planes.

The mixed implementation is intentionally narrow: static expert shape,
static scheduler, BF16 Form-A or Form-B combine through either the epilogue
warps or reused dispatch warps, and the DeepGEMM computation graph with the
top-k score folded into FC1.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from common.host_utils import (
    compare_and_report_mismatches,
    mxfp8_quantize_per_block_32_row,
)
from common.megamoe_constants import Mxfp8BlockSize, SfPaddingBlock
from moe_bf16_glu.mega_runner import (
    MegaMoEBf16Tester,
    _NO_DIST,
    _build_arg_parser as _build_bf16_arg_parser,
)
from moe_bf16_glu.runner_common import TrainingImplDesc
from moe_nvfp4_swapab.mega_runner import (
    MiscDesc,
    TokenCommProblemDesc,
    _parse_tuple,
)
from moe_nvfp4_swapab.runner_common import (
    assemble_raw_scales_stacked_expert,
    round_up,
)
from src.token_comm import CombineFormat


WeightKind = str

_WEIGHT_DTYPE_BY_KIND = {
    "mxfp8_bf16_e4m3": torch.float8_e4m3fn,
    "mxfp8_bf16_e5m2": torch.float8_e5m2,
}

# Public runtime ABI of the mixed kernel. In particular, there is no
# activation-SF leg and both weight-SF planes are explicit.
MIXED_RUNTIME_TENSOR_ATTRS = {
    "activation": "my_activation",
    "topk_idx": "my_topk_idx",
    "topk_weights": "my_topk_weights",
    "fc1_weight": "my_fc1_weight",
    "fc1_weight_sf": "my_fc1_weight_sf",
    "fc2_weight": "my_fc2_weight",
    "fc2_weight_sf": "my_fc2_weight_sf",
    "combine_output": "combine_output",
}


@dataclass(frozen=True)
class _RuntimeBindings:
    """Lazily imported GPU dependencies, factored for CPU orchestration tests."""

    cuda: object
    cutlass: object
    cute: object
    cutlass_torch: object
    utils: object
    kernel_class: object
    epilogue_token_tile: int
    sym_buffer_host: object


def _load_runtime_bindings() -> _RuntimeBindings:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils

    from moe_mxfp8_bf16_glu.epilogue_mxfp8_bf16 import (
        EpilogueTokenTile,
    )
    from moe_mxfp8_bf16_glu.megamoe_kernel_mxfp8_bf16 import (
        Sm100MegaMoEMxfp8Bf16Kernel,
    )
    from src.sym_buffer import SymBufferHost

    return _RuntimeBindings(
        cuda=cuda,
        cutlass=cutlass,
        cute=cute,
        cutlass_torch=cutlass_torch,
        utils=utils,
        kernel_class=Sm100MegaMoEMxfp8Bf16Kernel,
        epilogue_token_tile=EpilogueTokenTile,
        sym_buffer_host=SymBufferHost,
    )


def _validate_mixed_config(
    problem: TokenCommProblemDesc,
    impl: TrainingImplDesc,
    misc: MiscDesc,
) -> None:
    """Reject configurations unsupported by the mixed kernel."""

    if not impl.enable_static_expert_shape:
        raise ValueError(
            "mixed kernel requires enable_static_expert_shape=True."
        )
    if not impl.force_static_sched:
        raise ValueError("mixed kernel requires force_static_sched=True.")
    if problem.fc2_output_dtype is not torch.bfloat16:
        raise ValueError("mixed kernel requires BF16 FC2 output.")
    if problem.combine_format.is_quantized:
        raise ValueError("mixed kernel supports BF16 combine only.")
    if impl.token_back_mode == "standalone_warps":
        raise ValueError(
            "mixed transform warps occupy the standalone token-back "
            "warp ids; use 'epi_warps' or 'reuse_dispatch_warps'."
        )
    if impl.token_back_mode not in (
        "epi_warps",
        "reuse_dispatch_warps",
    ):
        raise ValueError(
            "mixed kernel supports token_back_mode='epi_warps' or "
            "'reuse_dispatch_warps'."
        )
    expected_dispatch_token_back = (
        impl.token_back_mode == "reuse_dispatch_warps"
    )
    if impl.token_back_by_dispatch != expected_dispatch_token_back:
        raise ValueError(
            "token_back_by_dispatch must match token_back_mode."
        )
    if not impl.non_ubulk_fc2_store:
        raise ValueError(
            "mixed kernel requires the direct epilogue-warp FC2 store."
        )
    if impl.generate_c:
        raise ValueError("mixed kernel requires generate_c=False.")
    if impl.use_stg_fc1:
        raise ValueError("mixed kernel requires use_stg_fc1=False.")
    if misc.ref_compute_graph != "deepgemm":
        raise ValueError(
            "mixed kernel requires ref_compute_graph='deepgemm'."
        )


def _swizzled_sf_numel(output_size: int, reduction_size: int) -> int:
    """Byte count of one atom-swizzled E8M0 scale plane."""

    raw_cols = reduction_size // Mxfp8BlockSize
    padded_rows = round_up(output_size, SfPaddingBlock)
    padded_cols = round_up(raw_cols, 4)
    return padded_rows * padded_cols


@dataclass(frozen=True)
class Mxfp8Bf16MegaWeightPlan:
    """Pure shape plan for the mixed public weight ABI."""

    global_fc1_weight: Tuple[int, int, int, int]
    global_fc2_weight: Tuple[int, int, int, int]
    global_fc1_sf_swizzled: Tuple[int, int, int]
    global_fc2_sf_swizzled: Tuple[int, int, int]
    local_fc1_weight: Tuple[int, int, int]
    local_fc2_weight: Tuple[int, int, int]
    local_fc1_sf_swizzled: Tuple[int, int]
    local_fc2_sf_swizzled: Tuple[int, int]


def plan_mxfp8_bf16_mega_weights(
    *,
    world_size: int,
    num_experts_per_rank: int,
    hidden: int,
    intermediate: int,
) -> Mxfp8Bf16MegaWeightPlan:
    """Return all global/local weight and SF shapes without allocating CUDA."""

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if num_experts_per_rank <= 0:
        raise ValueError(
            "num_experts_per_rank must be positive, got "
            f"{num_experts_per_rank}."
        )
    if hidden <= 0 or hidden % Mxfp8BlockSize != 0:
        raise ValueError(
            f"hidden must be a positive multiple of {Mxfp8BlockSize}, "
            f"got {hidden}."
        )
    if intermediate <= 0 or intermediate % 2 != 0:
        raise ValueError(
            f"intermediate must be positive and even, got {intermediate}."
        )
    intermediate_down = intermediate // 2
    if intermediate_down % Mxfp8BlockSize != 0:
        raise ValueError(
            f"intermediate/2 must be a multiple of {Mxfp8BlockSize}, "
            f"got {intermediate_down}."
        )

    r = world_size
    e = num_experts_per_rank
    fc1_sf_size = _swizzled_sf_numel(intermediate, hidden)
    fc2_sf_size = _swizzled_sf_numel(hidden, intermediate_down)
    return Mxfp8Bf16MegaWeightPlan(
        global_fc1_weight=(r, e, hidden, intermediate),
        global_fc2_weight=(r, e, intermediate_down, hidden),
        global_fc1_sf_swizzled=(r, e, fc1_sf_size),
        global_fc2_sf_swizzled=(r, e, fc2_sf_size),
        local_fc1_weight=(e, hidden, intermediate),
        local_fc2_weight=(e, intermediate_down, hidden),
        local_fc1_sf_swizzled=(e, fc1_sf_size),
        local_fc2_sf_swizzled=(e, fc2_sf_size),
    )


def _quantize_global_weight(
    *,
    torch_rng: torch.Generator,
    world_size: int,
    num_experts_per_rank: int,
    output_size: int,
    reduction_size: int,
    weight_dtype: torch.dtype,
    perf_run: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create K-major FP8 weights and swizzled E8M0 scale planes."""

    logical_weights = []
    raw_scales = []
    source_scale = 1.0 if perf_run else 0.25
    for _rank in range(world_size):
        for _expert in range(num_experts_per_rank):
            source = torch.randn(
                (output_size, reduction_size),
                dtype=torch.float32,
                device="cuda",
                generator=torch_rng,
            )
            if source_scale != 1.0:
                source.mul_(source_scale)
            quantized, raw_scale = mxfp8_quantize_per_block_32_row(
                source,
                weight_dtype,
            )
            logical_weights.append(quantized)
            raw_scales.append(raw_scale)

    # Stack logical (N,K), then expose public K-major (K,N) without repacking.
    logical = torch.stack(logical_weights).reshape(
        world_size,
        num_experts_per_rank,
        output_size,
        reduction_size,
    )
    physical = logical.transpose(2, 3)
    swizzled = assemble_raw_scales_stacked_expert(raw_scales).reshape(
        world_size,
        num_experts_per_rank,
        -1,
    )
    return physical, swizzled


class MegaMoEMxfp8Bf16Tester(MegaMoEBf16Tester):
    """BF16 MegaMoE communication with MXFP8-only expert weights."""

    def __init__(
        self,
        problem: TokenCommProblemDesc,
        impl: TrainingImplDesc,
        misc: MiscDesc,
        *,
        rank: int,
        kind: WeightKind = "mxfp8_bf16_e4m3",
    ) -> None:
        _validate_mixed_config(problem, impl, misc)
        # The parent owns the BF16 activation/combine and distributed contract.
        super().__init__(problem, impl, misc, rank=rank, kind="bf16")
        if kind not in _WEIGHT_DTYPE_BY_KIND:
            raise ValueError(
                f"kind must be one of {tuple(_WEIGHT_DTYPE_BY_KIND)}, "
                f"got {kind!r}."
            )

        plan_mxfp8_bf16_mega_weights(
            world_size=problem.world_size,
            num_experts_per_rank=problem.num_experts_per_rank,
            hidden=problem.hidden,
            intermediate=problem.intermediate,
        )
        self.kind = kind
        self.weight_torch_dtype = _WEIGHT_DTYPE_BY_KIND[kind]
        self._perf_sleep_ms = 0.0

    def set_perf_sleep_ms(self, sleep_ms: float) -> None:
        """Set the post-launch delay used by the profiler timing loop."""

        sleep_ms = float(sleep_ms)
        if not math.isfinite(sleep_ms) or sleep_ms < 0.0:
            raise ValueError(
                "perf_sleep_ms must be finite and non-negative; "
                f"got {sleep_ms!r}."
            )
        self._perf_sleep_ms = sleep_ms

    def generate_inputs(self) -> None:
        """Reuse BF16 activation/routing/sym-heap setup, then replace weights."""

        super().generate_inputs()

        # Drop the temporary BF16 weights created by the parent.  Activation,
        # routing, top-k weights, combine output, and every symmetric allocation
        # remain exactly those of the BF16 runner.
        self.my_fc1_weight = None
        self.my_fc2_weight = None
        self._global_fc1_weight = None
        self._global_fc2_weight = None
        gc.collect()

        problem = self.problem
        r = problem.world_size
        e = problem.num_experts_per_rank
        h = problem.hidden
        i = problem.intermediate

        (
            self._global_fc1_weight,
            self._global_fc1_weight_sf,
        ) = _quantize_global_weight(
            torch_rng=self._torch_cuda_rng,
            world_size=r,
            num_experts_per_rank=e,
            output_size=i,
            reduction_size=h,
            weight_dtype=self.weight_torch_dtype,
            perf_run=self.misc.perf_run,
        )
        (
            self._global_fc2_weight,
            self._global_fc2_weight_sf,
        ) = _quantize_global_weight(
            torch_rng=self._torch_cuda_rng,
            world_size=r,
            num_experts_per_rank=e,
            output_size=h,
            reduction_size=i // 2,
            weight_dtype=self.weight_torch_dtype,
            perf_run=self.misc.perf_run,
        )

        # Expert weights are local CUDA allocations, never symmetric-heap data.
        self.my_fc1_weight = self._global_fc1_weight[self.rank]
        self.my_fc1_weight_sf = self._global_fc1_weight_sf[self.rank]
        self.my_fc2_weight = self._global_fc2_weight[self.rank]
        self.my_fc2_weight_sf = self._global_fc2_weight_sf[self.rank]

        self._validate_weight_abi()
        torch.cuda.synchronize()
        self._check_cuda_rng_consistency()

    def _validate_weight_abi(self) -> None:
        plan = plan_mxfp8_bf16_mega_weights(
            world_size=self.problem.world_size,
            num_experts_per_rank=self.problem.num_experts_per_rank,
            hidden=self.problem.hidden,
            intermediate=self.problem.intermediate,
        )
        expected = {
            "_global_fc1_weight": plan.global_fc1_weight,
            "_global_fc2_weight": plan.global_fc2_weight,
            "_global_fc1_weight_sf": plan.global_fc1_sf_swizzled,
            "_global_fc2_weight_sf": plan.global_fc2_sf_swizzled,
            "my_fc1_weight": plan.local_fc1_weight,
            "my_fc2_weight": plan.local_fc2_weight,
            "my_fc1_weight_sf": plan.local_fc1_sf_swizzled,
            "my_fc2_weight_sf": plan.local_fc2_sf_swizzled,
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tensor is None:
                raise RuntimeError(f"{name} was not generated.")
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got "
                    f"{tuple(tensor.shape)}."
                )

        for name in ("_global_fc1_weight", "_global_fc2_weight"):
            tensor = getattr(self, name)
            if tensor.dtype is not self.weight_torch_dtype:
                raise TypeError(
                    f"{name} must have dtype {self.weight_torch_dtype}, "
                    f"got {tensor.dtype}."
                )
            if tensor.stride(2) != 1:
                raise ValueError(
                    f"{name} public K axis (dimension 2) must have stride 1; "
                    f"got {tensor.stride()}."
                )
        for name in ("my_fc1_weight", "my_fc2_weight"):
            tensor = getattr(self, name)
            if tensor.stride(1) != 1:
                raise ValueError(
                    f"{name} public K axis (dimension 1) must have stride 1; "
                    f"got {tensor.stride()}."
                )
        for name in (
            "_global_fc1_weight_sf",
            "_global_fc2_weight_sf",
            "my_fc1_weight_sf",
            "my_fc2_weight_sf",
        ):
            tensor = getattr(self, name)
            if tensor.dtype is not torch.float8_e8m0fnu:
                raise TypeError(
                    f"{name} must have dtype torch.float8_e8m0fnu, "
                    f"got {tensor.dtype}."
                )
            if tensor.stride(-1) != 1:
                raise ValueError(
                    f"{name} atom-swizzled plane must be flat-contiguous."
                )

    def compute_reference(self) -> None:
        if self.misc.skip_ref_check:
            return
        if self._global_activation is None:
            raise RuntimeError(
                "compute_reference requires generate_inputs first."
            )
        self._validate_weight_abi()

        from moe_mxfp8_bf16_glu.mega_reference_mxfp8_bf16 import (
            compute_megamoe_reference_mxfp8_bf16,
        )

        ref_result = compute_megamoe_reference_mxfp8_bf16(
            input_activation=self._global_activation,
            input_topk_idx=self._global_topk_idx,
            input_topk_weights=self._global_topk_weights,
            fc1_weight=self._global_fc1_weight,
            fc1_weight_sf=self._global_fc1_weight_sf,
            fc2_weight=self._global_fc2_weight,
            fc2_weight_sf=self._global_fc2_weight_sf,
            ref_compute_graph=self.misc.ref_compute_graph,
            fc2_output_dtype=self.problem.fc2_output_dtype,
            gate_up_clamp=self.problem.gate_up_clamp,
            apply_topk_in_fc1=self._apply_topk_in_fc1,
            return_fc1_gateup=self.impl.generate_c,
        )

        if self.impl.generate_c:
            combine_ref_global, fc1_gateup_global = ref_result
            expert_start = (
                self.rank * self.problem.num_experts_per_rank
            )
            self._ref_fc1_gateup_per_expert = {
                expert: fc1_gateup_global.get(expert_start + expert)
                for expert in range(self.problem.num_experts_per_rank)
            }
        else:
            combine_ref_global = ref_result
            self._ref_fc1_gateup_per_expert = None
        self.combine_output_ref = combine_ref_global[self.rank].contiguous()

    def validate(self) -> None:
        """Validate the combine shape selected by the FC2 reduction mode.

        Form A compares every independently routed ``(token, topk, hidden)``
        cell, so slot permutations cannot hide behind a top-k sum. Form B
        compares the kernel's singleton-topk REDG result against the
        reference reduced over its top-k axis.
        """

        if self.misc.skip_ref_check:
            return
        if self.combine_output is None:
            raise RuntimeError("validate requires run_kernel first.")
        if self.combine_output_ref is None:
            raise RuntimeError("validate requires compute_reference first.")

        if self.impl.in_kernel_fc2_reduce:
            actual = self.combine_output[:, 0, :].to(torch.float32)
            reference_terms = self.combine_output_ref.to(torch.float32)
            reference = reference_terms.sum(dim=1)

            # Form B uses BF16 remote atomic adds. Their arrival order is not
            # deterministic, and every addition rounds to BF16, while the
            # host reference sum above is exact in FP32. Bound the rounding
            # error for any sequential order with the standard gamma(n)
            # model instead of hiding it behind a large fixed atol.
            unit_roundoff = torch.finfo(torch.bfloat16).eps / 2
            additions = max(reference_terms.shape[1] - 1, 0)
            gamma = (additions * unit_roundoff) / (
                1.0 - additions * unit_roundoff
            )
            reduction_roundoff = gamma * reference_terms.abs().sum(dim=1)
            base_tolerance = 1e-2 + 1e-2 * reference.abs()
            allowed_error = base_tolerance + reduction_roundoff
            normalized_error = (actual - reference) / allowed_error
            compare_and_report_mismatches(
                normalized_error,
                torch.zeros_like(normalized_error),
                name=f"combine_output_form_b[rank{self.rank}]",
                atol=1.0,
                rtol=0.0,
            )
        else:
            actual = self.combine_output.to(torch.float32)
            reference = self.combine_output_ref.to(torch.float32)
            compare_and_report_mismatches(
                actual,
                reference,
                name=f"combine_output_form_a[rank{self.rank}]",
                atol=1e-2,
                rtol=1e-2,
            )
        self._validate_c_output()

    def _runtime_tensors(self) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        missing = []
        for argument, attribute in MIXED_RUNTIME_TENSOR_ATTRS.items():
            value = getattr(self, attribute, None)
            if value is None:
                missing.append(attribute)
            else:
                tensors[argument] = value
        if missing:
            raise RuntimeError(
                "run_kernel requires generate_inputs first; missing "
                + ", ".join(missing)
            )
        self._validate_combine_output_abi()
        return tensors

    def _validate_combine_output_abi(self) -> None:
        """Mirror the kernel's mode-dependent public combine tensor contract."""

        if self.combine_output is None:
            raise RuntimeError("combine_output has not been allocated.")
        combine_topk = (
            1
            if self.impl.in_kernel_fc2_reduce
            else self.problem.num_topk
        )
        expected_shape = (
            self.problem.num_tokens_per_rank,
            combine_topk,
            self.problem.hidden,
        )
        if tuple(self.combine_output.shape) != expected_shape:
            raise ValueError(
                f"combine_output must have shape {expected_shape} for "
                f"in_kernel_fc2_reduce={self.impl.in_kernel_fc2_reduce}; "
                f"got {tuple(self.combine_output.shape)}."
            )
        if self.combine_output.dtype is not torch.bfloat16:
            raise TypeError(
                "combine_output must have dtype torch.bfloat16; got "
                f"{self.combine_output.dtype}."
            )

    def _kernel_constructor_kwargs(
        self,
        *,
        group_hint: int,
        token_padding_block: int,
        ab_dtype,
    ) -> Dict[str, object]:
        """Return the fixed mixed-kernel constructor contract."""

        return {
            "mma_tiler_mnk": self.impl.mma_tiler_mnk,
            "cluster_shape_mnk": self.impl.cluster_shape_mnk,
            "use_2cta_instrs": self.impl.use_2cta_instrs,
            "group_hint": group_hint,
            "token_padding_block": token_padding_block,
            "load_balance_mode": self.impl.load_balance_mode,
            "static_expert_shape": (
                self.problem.num_experts_per_rank,
                self.problem.intermediate,
                self.problem.hidden,
            ),
            "force_static_sched": True,
            "clc_bundle_size": self.impl.clc_bundle_size,
            "num_sched_stages": self.impl.num_sched_stages,
            "transform_buffer": getattr(
                self.impl, "transform_buffer", "tmem"
            ),
            "accumulator_overlap": getattr(
                self.impl, "accumulator_overlap", False
            ),
            "transform_k_tile": getattr(
                self.impl, "transform_k_tile", 128
            ),
            "ab_dtype": ab_dtype,
            "world_size": self.world_size,
            "local_rank": self.rank,
            "num_topk": self.problem.num_topk,
            "max_tokens_per_rank": self.problem.num_tokens_per_rank,
            "hidden": self.problem.hidden,
            "fc2_in_kernel_topk_reduce": self.impl.in_kernel_fc2_reduce,
            "token_back_by_dispatch": self.impl.token_back_by_dispatch,
            "token_back_mode": self.impl.token_back_mode,
            "epi_flag_batch": self.impl.epi_flag_batch,
            "flag_batch": self.impl.flag_batch,
            "gate_up_clamp": self.problem.gate_up_clamp,
            "apply_topk_in_fc1": True,
            "generate_c": False,
            "use_stg_fc1": False,
        }

    def _prepare_combine_output_for_launch(self) -> None:
        """Reset the REDG destination and make the reset collective-safe.

        Form-B uses relaxed system-scope reduction, either directly from the
        epilogue warps or from reused dispatch warps. Every rank must finish
        clearing its own destination before any rank starts a launch;
        otherwise an early peer reduction can be erased by a late local zero.
        """

        if not self.impl.in_kernel_fc2_reduce:
            return
        if self.combine_output is None:
            raise RuntimeError(
                "Form-B launch requires an allocated combine_output."
            )

        self.combine_output.zero_()
        if self.world_size == 1:
            return

        torch.cuda.synchronize()
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "multi-rank Form-B output reset requires initialized "
                "torch.distributed."
            )
        torch.distributed.barrier()
        torch.cuda.synchronize()

    def _launch_target_kernels_with_optional_torch_profiler(
        self,
        runtime_kwargs,
    ) -> None:
        """Launch with a fresh Form-B REDG destination on every iteration."""

        if self._compiled_kernel is None:
            raise RuntimeError("compiled kernel is not available")

        def _launch() -> None:
            self._prepare_combine_output_for_launch()
            self._compiled_kernel(**runtime_kwargs)

        sleep_seconds = self._perf_sleep_ms / 1000.0

        def _launch_with_optional_cooldown() -> None:
            if sleep_seconds > 0.0:
                torch.cuda.synchronize()
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.barrier()
                    torch.cuda.synchronize()
            _launch()
            if sleep_seconds > 0.0:
                torch.cuda.synchronize()
                time.sleep(sleep_seconds)

        if not self._use_torch_profiler:
            _launch_with_optional_cooldown()
            torch.cuda.synchronize()
            return

        for _ in range(self._perf_warmup):
            _launch_with_optional_cooldown()
        torch.cuda.synchronize()

        n_iters = max(1, self._perf_iters)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            for _ in range(n_iters):
                _launch_with_optional_cooldown()
            torch.cuda.synchronize()
        self._report_torch_profiler_kernel_time(prof, num_iters=n_iters)

    def _run_debug_relaunch(self, runtime_kwargs) -> None:
        """Exercise a second launch with mode-appropriate output handling."""

        if self.impl.in_kernel_fc2_reduce:
            # REDG accumulation order is not byte-deterministic. The launch
            # wrapper zeros the destination, and validate() checks this second
            # result numerically against the reference.
            self._launch_target_kernels_with_optional_torch_profiler(
                runtime_kwargs,
            )
            print(
                "✓ mixed Form-B repeated launch passed output-reset "
                "and workspace-reset checks."
            )
            return

        first_output = self.combine_output.view(torch.uint8).clone()
        self.combine_output.fill_(float("nan"))
        self._launch_target_kernels_with_optional_torch_profiler(
            runtime_kwargs,
        )
        second_output = self.combine_output.view(torch.uint8)
        if not torch.equal(first_output, second_output):
            byte_mismatches = int(
                torch.count_nonzero(first_output != second_output).item()
            )
            raise AssertionError(
                "mixed repeated launch is not byte deterministic: "
                f"{byte_mismatches} output bytes differ."
            )
        print(
            "✓ mixed Form-A repeated launch passed byte-determinism "
            "and workspace-reset checks."
        )

    def run_kernel(self) -> None:
        """Compile and launch ``Sm100MegaMoEMxfp8Bf16Kernel``."""

        _validate_mixed_config(self.problem, self.impl, self.misc)
        torch_runtime_tensors = self._runtime_tensors()
        bindings = _load_runtime_bindings()

        cluster_size = (
            self.impl.cluster_shape_mnk[0]
            * self.impl.cluster_shape_mnk[1]
        )
        max_active_clusters = (
            bindings.utils.HardwareInfo().get_max_active_clusters(
                cluster_size
            )
        )
        group_hint = self.impl.group_hint
        if group_hint is None:
            group_hint = max_active_clusters

        kernel_kwargs = self._kernel_constructor_kwargs(
            group_hint=group_hint,
            token_padding_block=bindings.epilogue_token_tile,
            ab_dtype=bindings.cutlass.BFloat16,
        )
        self._kernel = bindings.kernel_class(**kernel_kwargs)

        # The kernel owns the exact local/shared region plan. The inherited
        # allocator places the shared byte buffer on the symmetric heap and
        # derives the host-side peer pointer deltas.
        self.allocate_workspaces()

        def _to_cute(
            tensor: torch.Tensor,
            *,
            assumed_align: int = 16,
            force_static_layout: bool = False,
        ):
            result = bindings.cutlass_torch.from_dlpack(
                tensor,
                assumed_align=assumed_align,
            )
            if force_static_layout:
                return result
            leading_dim = bindings.cutlass_torch.get_leading_dim(tensor)
            return result.mark_layout_dynamic(leading_dim=leading_dim)

        runtime_kwargs = {
            name: _to_cute(tensor)
            for name, tensor in torch_runtime_tensors.items()
        }
        runtime_kwargs["local_workspace"] = _to_cute(
            self.local_workspace,
            force_static_layout=True,
        )
        runtime_kwargs["shared_workspace"] = _to_cute(
            self.shared_workspace,
        )

        peer_rank_ptr_mapper_host = bindings.sym_buffer_host(
            base_addr=self.symmetric_base,
            offsets=tuple(self.peer_offsets_list),
            rank_idx=self.rank,
            num_max_ranks=self.world_size,
        )
        runtime_kwargs["peer_rank_ptr_mapper_host"] = (
            peer_rank_ptr_mapper_host
        )
        runtime_kwargs["stream"] = bindings.cuda.CUstream(
            torch.cuda.current_stream().cuda_stream
        )

        # Keep the compile-time call aligned with the kernel's public
        # ``__call__`` order: the constexpr cluster bound precedes the stream.
        compile_kwargs = dict(runtime_kwargs)
        compile_stream = compile_kwargs.pop("stream")
        compile_kwargs["max_active_clusters"] = max_active_clusters
        compile_kwargs["stream"] = compile_stream
        if self.misc.enable_iket:
            compile_kwargs["options"] = "iket"
        self._compiled_kernel = bindings.cute.compile(
            self._kernel,
            **compile_kwargs,
        )

        if self.misc.profile_friendly:
            import nvtx

            torch.cuda.synchronize()
            dist_active = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            if dist_active:
                torch.distributed.barrier()
                torch.cuda.synchronize()
            with nvtx.annotate("cute_dsl_prof"):
                self._launch_target_kernels_with_optional_torch_profiler(
                    runtime_kwargs,
                )
            if dist_active:
                torch.distributed.barrier()
                torch.cuda.synchronize()
        else:
            self._launch_target_kernels_with_optional_torch_profiler(
                runtime_kwargs,
            )

        # Exercise the communication tail's counter reset and NVLink-barrier
        # epoch on the exact same workspaces. Form A poisons the destination so
        # a missed store cannot inherit the first launch. Form B must instead
        # zero its REDG accumulation target before every launch.
        if (
            getattr(self.misc, "enable_debug_checks", False)
            and self.world_size == 1
            and not self.misc.profile_friendly
        ):
            self._run_debug_relaunch(runtime_kwargs)


_SYMMETRIC_RUNTIME_TENSOR_ATTRS = (
    "my_activation",
    "my_topk_idx",
    "my_topk_weights",
    "combine_output",
    "shared_workspace",
)


def _cleanup_distributed_runtime(
    tester: Optional[MegaMoEMxfp8Bf16Tester],
    finalize_dist_and_nvshmem,
) -> None:
    """Best-effort symmetric-tensor cleanup followed by collective finalize.

    Every cleanup step is attempted even if an earlier one fails.  The first
    cleanup failure is reported after ``finalize_dist_and_nvshmem`` has been
    called; ``main`` suppresses that cleanup failure only when preserving an
    already-propagating compile, launch, or validation exception.
    """

    cleanup_errors: List[BaseException] = []

    def attempt(action) -> None:
        try:
            action()
        except BaseException as exc:  # noqa: BLE001
            cleanup_errors.append(exc)

    if tester is not None:
        attempt(lambda: setattr(tester, "_compiled_kernel", None))
        attempt(lambda: setattr(tester, "_kernel", None))
        attempt(gc.collect)
        attempt(torch.cuda.synchronize)

        free_tensor = None
        try:
            import nvshmem.core

            free_tensor = nvshmem.core.free_tensor
        except ImportError:
            pass
        except BaseException as exc:  # noqa: BLE001
            cleanup_errors.append(exc)

        for attr in _SYMMETRIC_RUNTIME_TENSOR_ATTRS:
            sym_tensor = getattr(tester, attr, None)
            if free_tensor is not None and sym_tensor is not None:
                # One unavailable/already-freed view must not prevent the
                # remaining collective frees from being attempted.
                try:
                    free_tensor(sym_tensor)
                except Exception:  # noqa: BLE001
                    pass
            attempt(lambda attr=attr: setattr(tester, attr, None))
        attempt(gc.collect)

    attempt(finalize_dist_and_nvshmem)
    if cleanup_errors:
        raise cleanup_errors[0]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Reuse the BF16 MegaMoE CLI while replacing only mixed-path policy."""

    parser = _build_bf16_arg_parser()
    parser.description = (
        "MegaMoE MXFP8-weight/BF16-activation host runner"
    )
    kind_action = next(
        action for action in parser._actions if action.dest == "kind"
    )
    kind_action.default = "mxfp8_bf16_e4m3"
    kind_action.choices = tuple(_WEIGHT_DTYPE_BY_KIND)
    kind_action.help = (
        "MXFP8 encoding for FC1/FC2 weights; activations and combine stay BF16."
    )
    parser.set_defaults(
        kind="mxfp8_bf16_e4m3",
        mma_tiler_mnk="256,128,128",
        cluster_shape_mnk="2,1,1",
        use_2cta_instrs=True,
        enable_static_expert_shape=True,
    )
    parser.add_argument(
        "--transform_buffer",
        choices=("smem", "tmem"),
        default="tmem",
        help="Storage backing the transformed-A pipeline.",
    )
    parser.add_argument(
        "--accumulator_overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the two physical accumulator stages overlap in TMEM.",
    )
    parser.add_argument(
        "--transform_k_tile",
        type=int,
        choices=(64, 128),
        default=128,
        help="Internal transform/MMA K tile; raw TMA K remains MNK.K.",
    )
    parser.add_argument(
        "--perf_sleep_ms",
        type=float,
        default=0.0,
        help=(
            "Post-launch sleep in each warm-up/measured profiler iteration. "
            "A positive value also enables a pre-launch rank barrier and "
            "post-launch CUDA synchronization."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    distributed_initialized = False
    finalize_dist_and_nvshmem = None
    if _NO_DIST:
        torch.cuda.set_device(0)
        rank = 0
        world_size = 1
    else:
        from src.bootstrap import (
            finalize_dist_and_nvshmem,
            init_dist_and_nvshmem,
        )

        _local_rank, rank, world_size, _ = init_dist_and_nvshmem()
        distributed_initialized = True

    tester: Optional[MegaMoEMxfp8Bf16Tester] = None
    execution_failed = False
    try:
        problem = TokenCommProblemDesc(
            world_size=world_size,
            num_tokens_per_rank=args.num_tokens_per_rank,
            num_topk=args.num_topk,
            num_total_experts=args.num_total_experts,
            hidden=args.hidden,
            intermediate=args.intermediate,
            fc2_output_dtype=args.fc2_output_dtype,
            combine_format=CombineFormat.parse("bf16"),
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
        # Mixed implementation knobs are intentionally local to this runner
        # instead of widening the shared BF16 TrainingImplDesc API.
        impl.transform_buffer = args.transform_buffer
        impl.accumulator_overlap = args.accumulator_overlap
        impl.transform_k_tile = args.transform_k_tile
        misc = MiscDesc(
            perf_run=args.perf_run,
            skip_ref_check=args.skip_ref_check,
            run_target_kernel_only=args.profile_friendly,
            enable_debug_checks=args.enable_debug_checks,
            ref_compute_graph=args.ref_compute_graph,
            enable_iket=args.enable_iket,
            seed=args.seed,
        )

        tester = MegaMoEMxfp8Bf16Tester(
            problem,
            impl,
            misc,
            rank=rank,
            kind=args.kind,
        )
        tester.set_torch_profiler_enabled(args.use_torch_profiler)
        tester.set_perf_iters(args.perf_warmup, args.perf_iters)
        tester.set_perf_sleep_ms(args.perf_sleep_ms)
        if rank == 0 and args.use_torch_profiler:
            print(
                "MegaMoETester: "
                f"perf_sleep_ms={args.perf_sleep_ms:g}"
            )

        # Compilation/launch/validation failures propagate unchanged.
        tester.run()
    except BaseException:  # noqa: BLE001
        execution_failed = True
        raise
    finally:
        if distributed_initialized:
            assert finalize_dist_and_nvshmem is not None
            try:
                _cleanup_distributed_runtime(
                    tester,
                    finalize_dist_and_nvshmem,
                )
            except BaseException:  # noqa: BLE001
                if not execution_failed:
                    raise
    return 0


__all__ = [
    "MegaMoEMxfp8Bf16Tester",
    "Mxfp8Bf16MegaWeightPlan",
    "MIXED_RUNTIME_TENSOR_ATTRS",
    "plan_mxfp8_bf16_mega_weights",
]


if __name__ == "__main__":
    sys.exit(main())
