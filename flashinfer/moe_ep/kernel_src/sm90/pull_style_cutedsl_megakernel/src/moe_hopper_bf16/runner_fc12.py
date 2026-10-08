# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Host driver for the MegaMoE BF16 GLU fused fc1+fc2 kernel.
"""


import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

## TODO: currently some common modules are located in moe_nvfp4_swapab,
## which will be moved to common package later. These paths dependency
## could be removed once the modules are moved.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PKG_DIR)
_NVFP4_DIR = os.path.join(_PARENT_DIR, "moe_nvfp4_swapab")
for _p in (_PARENT_DIR, _NVFP4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from moe_nvfp4_swapab.runner_fc12_common import (
    ProblemDesc as _BaseProblemDesc,
    ImplDesc as _BaseImplDesc,
    MiscDesc,
    Fc12TesterBase,
    add_common_fc12_arguments,
    parse_tuple,
)
from moe_hopper_bf16.epilogue_bf16_swapab import (
    SwapABTileMChoices,
    SwapABTokenTileNChoices,
)
from moe_hopper_bf16.epilogue_bf16 import (
    NonSwapTileMChoices,
    NonSwapTileNChoices,
)
from moe_nvfp4_swapab.runner_common import (
    offs_to_group_sizes,
    swiglu_fold_interleave,
)
from common.megamoe_constants import (
    Fp8GateUpInterleave,
    SfPaddingBlock,
)
from moe_hopper_bf16.hopper_moe_utils import (
    BF16_KIND_CHOICES,
    bf16_kind_to_cutlass_dtype,
    bf16_reference_mm,
    create_bf16_tensor,
)


# =============================================================================
# BF16 tester
# =============================================================================


@dataclass
class ProblemDesc(_BaseProblemDesc):
    """Hopper problem descriptor whose ``kind`` collapses to a single BF16."""

    kind: str = "bf16"

    def __post_init__(self) -> None:
        if self.kind != "bf16":
            raise ValueError(
                f"Hopper MoE is BF16-only; kind must be 'bf16', got {self.kind!r}."
            )
        # The shared descriptor accepts kind='bf16' and applies the checks
        # BF16 needs (gate/up interleave granularity and the intermediate/2
        # block multiple).
        super().__post_init__()


@dataclass
class ImplDesc(_BaseImplDesc):
    """Hopper BF16 impl descriptor with the swap-AB short-N specializations.

    ``generate_c`` selects the training forward: the FC1 epilogue additionally
    writes the raw pre-SwiGLU gate+up GEMM accumulator (BF16, raw N order) to a
    separate ``fc1_c`` tensor.  It defaults to False so the inference path and
    every existing test case are unchanged.

    ``tail_split_pairs`` splits the tail cluster block of an expert with an
    odd CTA-tile count into pair tasks (both CTAs compute the single valid
    token tile against adjacent weight tiles).  It requires a token-side
    cluster of 2 (swap-AB cga (1,2,1), non-swap cga (2,1,1)) and defaults to
    False so the default build is unchanged.
    """

    generate_c: bool = False
    tail_split_pairs: bool = False

    def _validate_mma_cta_mode(self, m: int) -> None:
        if self.use_2cta_instrs:
            raise ValueError(
                "Hopper BF16 only supports 1CTA MMA; "
                f"got mma_tiler_m={m}, use_2cta_instrs=True."
            )

    def __post_init__(self) -> None:
        m, n, k = self.mma_tiler_mnk
        supported_clusters = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1))
        if self.cluster_shape_mnk not in supported_clusters:
            raise ValueError(
                "Hopper BF16 cluster_shape_mnk must be one of "
                f"{supported_clusters}, got {self.cluster_shape_mnk}."
            )
        non_swap_geometry = (
            m in NonSwapTileMChoices and n in NonSwapTileNChoices
        )
        if n not in SwapABTokenTileNChoices and not non_swap_geometry:
            raise ValueError(
                "Hopper BF16 mma_tiler_mnk must use swap-AB N in "
                f"{SwapABTokenTileNChoices} or a non-swap geometry in "
                f"M={NonSwapTileMChoices}, N={NonSwapTileNChoices}; "
                f"got ({m}, {n}, {k})."
            )

        # The shared NVFP4/MXFP8 descriptor intentionally keeps its original
        # N choices. Validate all common fields through it using an equivalent
        # supported N, then restore this Hopper-only compile-time N. This
        # temporary value never becomes the physical token padding.
        original_tiler = self.mma_tiler_mnk
        original_cluster = self.cluster_shape_mnk
        if n < 64:
            self.mma_tiler_mnk = (m, 64, k)
        # The shared v1 descriptor keeps cluster-N fixed at one. Hopper owns
        # the wider cluster matrix above and validates it independently.
        self.cluster_shape_mnk = (original_cluster[0], 1, 1)
        try:
            super().__post_init__()
        finally:
            self.mma_tiler_mnk = original_tiler
            self.cluster_shape_mnk = original_cluster


class SwigluBf16Fc12Tester(Fc12TesterBase):
    """BF16 host-side input/reference/launch/validation driver."""

    def __init__(
        self,
        problem: ProblemDesc,
        impl: ImplDesc,
        misc: MiscDesc,
        *,
        swap_ab: bool = False,
        pingpong: bool = False,
    ) -> None:
        self.swap_ab = swap_ab
        self.pingpong = pingpong
        super().__init__(problem, impl, misc)
        # Non-swap publicly supports M=64; the retained M=128 implementation
        # is intentionally disabled. Swap-AB uses weight-M=128/256.
        m, n, _k = impl.mma_tiler_mnk
        valid_geometry = (
            m in SwapABTileMChoices and n in SwapABTokenTileNChoices
            if self.swap_ab
            else m in NonSwapTileMChoices and n in NonSwapTileNChoices
        )
        if not valid_geometry or impl.use_2cta_instrs:
            raise ValueError(
                "Hopper BF16 fused fc12 geometry does not match swap_ab="
                f"{self.swap_ab}; got "
                f"mma_tiler_mnk={impl.mma_tiler_mnk}, "
                f"use_2cta_instrs={impl.use_2cta_instrs}."
            )

    @property
    def _epilogue_token_tile(self) -> int:
        # A non-empty FC1 tail issues one full physical token tile: M for the
        # native layout and N after swapping A/B.
        return (
            self.impl.mma_tiler_mnk[1]
            if self.swap_ab
            else self.impl.mma_tiler_mnk[0]
        )

    # ------------------------------------------------------------------
    # Kind hooks: input / output tensor creation
    # ------------------------------------------------------------------

    def _fc2_output_shape(self, data_total_rows: int) -> Tuple[int, ...]:
        # BF16: 2D (token_max, hidden) -- epilogue_bf16.py uses shape[1].
        return (data_total_rows, self.problem.hidden)

    def _create_input_data_tensors(self, data_total_rows: int) -> None:
        problem = self.problem
        hidden = problem.hidden
        intermediate = problem.intermediate
        experts = problem.experts

        # -- activation: (data_total_rows, hidden) bf16, hidden stride-1 --
        self.activation = create_bf16_tensor(
            (data_total_rows, hidden),
            perf_run=self.misc.perf_run,
        )

        # -- fc1_weight: (experts, intermediate, hidden) -> permute ->
        #    (experts, hidden, intermediate), hidden stride-1 --
        self.fc1_weight = create_bf16_tensor(
            (experts, intermediate, hidden),
            perf_run=self.misc.perf_run,
        ).permute(0, 2, 1)

        # -- fc2_weight: (experts, hidden, inter//2) -> permute ->
        #    (experts, inter//2, hidden), inter//2 stride-1 --
        self.fc2_weight = create_bf16_tensor(
            (experts, hidden, intermediate // 2),
            perf_run=self.misc.perf_run,
        ).permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Input construction (SF-free overrides of the shared driver)
    # ------------------------------------------------------------------

    def generate_inputs(self) -> None:
        """Build ``offs`` plus every BF16 input / output tensor.

        Mirrors ``Fc12TesterBase.generate_inputs`` with the block-scale-factor
        steps removed: BF16 operands carry no per-block SF leg, so no raw
        scales are generated and no atom-layout SF buffer is assembled.
        """
        self.offs = self._generate_offs()
        valid_tokens = offs_to_group_sizes(self.offs)
        self.valid_tokens_per_expert = valid_tokens

        data_offsets, sf_offsets = self._compute_physical_offsets(valid_tokens)
        self.data_physical_offsets = data_offsets
        self.sf_physical_offsets = sf_offsets
        data_total_rows = data_offsets[-1]
        sf_total_rows = sf_offsets[-1]

        # ``run_target_kernel_only`` measures kernel-only latency against
        # undefined inputs; ``data_total_rows == 0`` is a high-EP rank with no
        # routed tokens.  Both take the un-initialized skeleton path.
        if self.misc.run_target_kernel_only or data_total_rows == 0:
            self._generate_inputs_skeleton(
                valid_tokens, data_total_rows, sf_total_rows
            )
            return

        self._create_input_data_tensors(data_total_rows)
        self._init_global_scales_and_norm()
        self._init_topk_scores(data_total_rows)
        self._alloc_fc2_output(data_total_rows)
        self._alloc_workspace_placeholder()

        torch.cuda.synchronize()

    def _generate_inputs_skeleton(
        self,
        valid_tokens_per_expert: List[int],
        data_total_rows: int,
        sf_total_rows: int,
    ) -> None:
        """Allocate BF16 tensors with correct shape / stride but no data init."""
        problem = self.problem
        hidden = problem.hidden
        intermediate = problem.intermediate
        experts = problem.experts

        self.activation = torch.empty(
            (data_total_rows, hidden), dtype=torch.bfloat16, device="cuda",
        )
        self.fc1_weight = torch.empty(
            (experts, intermediate, hidden), dtype=torch.bfloat16, device="cuda",
        ).permute(0, 2, 1)
        self.fc2_weight = torch.empty(
            (experts, hidden, intermediate // 2),
            dtype=torch.bfloat16, device="cuda",
        ).permute(0, 2, 1)

        self.activation_global_scale = torch.empty(
            (experts,), dtype=torch.float32, device="cuda"
        )
        self.fc1_weight_global_scale = torch.empty(
            (experts,), dtype=torch.float32, device="cuda"
        )
        self.fc2_weight_global_scale = torch.empty(
            (experts,), dtype=torch.float32, device="cuda"
        )
        self.norm_const = torch.empty((1,), dtype=torch.float32, device="cuda")

        self.topk_scores = torch.empty(
            (data_total_rows,), dtype=torch.float32, device="cuda",
        )

        self._alloc_fc2_output_skeleton(data_total_rows)

        self.workspace = torch.zeros(
            (1 << 20,), dtype=torch.uint8, device="cpu"
        ).to("cuda")

    def _print_layout_info(self) -> None:
        """Print every host tensor's shape/stride/dtype (no SF legs in BF16)."""
        for name in ("activation", "fc1_weight", "fc2_weight",
                     "topk_scores", "fc2_output", "workspace"):
            tensor = getattr(self, name)
            print(
                f"{name}: shape={tuple(tensor.shape)}  "
                f"stride={tensor.stride()}  dtype={tensor.dtype}"
            )
        print(
            f"offs: {self.offs.cpu().tolist()}  "
            f"valid_tokens_per_expert={self.valid_tokens_per_expert}  "
            f"data_physical_offsets={self.data_physical_offsets}"
        )
        self._print_scheduler_layout()

    def _init_topk_scores(self, data_total_rows: int) -> None:
        """Fused fc12 does not apply topk weighting; use 1.0 on valid rows."""
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        self.topk_scores = torch.zeros(
            (data_total_rows,), dtype=torch.float32, device="cuda",
        )
        for e in range(self.problem.experts):
            v_e = valid_tokens[e]
            if v_e == 0:
                continue
            phys = data_offsets[e]
            self.topk_scores[phys : phys + v_e] = 1.0

    def _alloc_fc2_output(self, data_total_rows: int) -> None:
        # 0xFF byte fill: bf16/fp16 0xFFFF = NaN -- kernel output overwriting
        # valid rows is easy to distinguish from "kernel never touched this
        # row".  Hopper stores a flat 2D ``(token_max, hidden)`` output.
        problem = self.problem
        hidden = problem.hidden
        fc2_output_bytes = torch.full(
            (data_total_rows, hidden * problem.fc2_output_dtype.itemsize),
            0xFF,
            dtype=torch.uint8, device="cuda",
        )
        self.fc2_output = fc2_output_bytes.view(problem.fc2_output_dtype).reshape(
            data_total_rows, hidden
        )

    # ------------------------------------------------------------------
    # Kind hooks: reference compute
    # ------------------------------------------------------------------

    def compute_reference(self) -> None:
        """BF16 fused fc1+fc2 reference dispatch."""
        self._compute_reference_bf16()

    def _compute_reference_bf16(self) -> None:
        """BF16 fused fc1+fc2 reference.

        Both GEMMs read BF16 operands and accumulate in FP32, mirroring the
        kernel's WGMMA.  The post-SwiGLU FC1 result is rounded back to BF16
        before FC2 consumes it, which models the kernel's BF16 ``fc1_output``
        staging buffer -- the reference therefore stays faithful to the data
        path at the fc1 / fc2 hand-off point.
        """
        if self.activation is None or self.offs is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")
        if self.misc.skip_ref_check:
            return

        problem = self.problem
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        data_total_rows = data_offsets[-1]
        gate_up_clamp = getattr(problem, "gate_up_clamp", None)

        ref_bytes = torch.zeros(
            (data_total_rows, problem.hidden * problem.fc2_output_dtype.itemsize),
            dtype=torch.uint8, device="cuda",
        )
        self.fc2_output_ref = ref_bytes.view(problem.fc2_output_dtype).reshape(
            data_total_rows, problem.hidden
        )

        self._ref_fc1_q_per_expert = [None] * problem.experts
        # Raw pre-SwiGLU gate+up accumulator (BF16), consumed by the
        # ``generate_c`` check; column order is the kernel's raw GEMM N order
        # (the gate/up interleave-8 order of ``fc1_weight``).
        self._ref_fc1_gateup_per_expert = [None] * problem.experts

        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for expert_idx in range(problem.experts):
                v_e = valid_tokens[expert_idx]
                if v_e == 0:
                    continue

                d_start = data_offsets[expert_idx]
                act_slice = self.activation[d_start : d_start + v_e]
                fc1_fp32 = bf16_reference_mm(
                    act_slice, self.fc1_weight[expert_idx],
                )

                self._ref_fc1_gateup_per_expert[expert_idx] = fc1_fp32.to(
                    torch.bfloat16
                )

                swiglu = swiglu_fold_interleave(
                    fc1_fp32,
                    Fp8GateUpInterleave,
                    gate_up_clamp=gate_up_clamp,
                )
                fc2_activation_bf16 = swiglu.to(torch.bfloat16)

                fc2_fp32 = bf16_reference_mm(
                    fc2_activation_bf16, self.fc2_weight[expert_idx],
                )
                self.fc2_output_ref[d_start : d_start + v_e] = fc2_fp32.to(
                    problem.fc2_output_dtype
                )

                self._ref_fc1_q_per_expert[expert_idx] = fc2_activation_bf16
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    # ------------------------------------------------------------------
    # Kind hooks: kernel + validation
    # ------------------------------------------------------------------

    def _instantiate_kernel(self, common_kwargs: dict):
        if self.swap_ab:
            from moe_hopper_bf16.kernel_bf16_glu_fc12_swapab import (
                Sm90SwapABSwigluBf16Fc12Kernel as Kernel,
            )
        else:
            from moe_hopper_bf16.kernel_bf16_glu_fc12 import (
                Sm90SwigluBf16Fc12Kernel as Kernel,
            )

        return Kernel(
            **common_kwargs,
            ab_dtype=bf16_kind_to_cutlass_dtype(self.problem.kind),
            pingpong=self.pingpong,
            gate_up_clamp=self.problem.gate_up_clamp,
            generate_c=self.impl.generate_c,
            tail_split_pairs=self.impl.tail_split_pairs,
        )

    def _extra_runtime_kwargs(self) -> dict:
        """Allocate and inject the ``generate_c`` raw gate+up output tensor.

        Rows are the same padded per-expert pool rows as the FC1 output
        workspace (``data_physical_offsets`` / ``valid_tokens_per_expert``);
        columns are the full gate+up width in the kernel's raw N order.
        """
        self._c_output = None
        if not self.impl.generate_c:
            return {}
        import cutlass.torch as cutlass_torch

        data_total_rows = int(self.data_physical_offsets[-1])
        self._c_output = torch.zeros(
            (data_total_rows, self.problem.intermediate),
            dtype=torch.bfloat16, device="cuda",
        )
        c_cute = cutlass_torch.from_dlpack(self._c_output, assumed_align=16)
        leading_dim = cutlass_torch.get_leading_dim(self._c_output)
        return {"fc1_c": c_cute.mark_layout_dynamic(leading_dim=leading_dim)}

    def _partition_workspace(self, counter_token_tile: int):
        """Carve the workspace into the BF16 kernel's regions.

        Layout, matching ``get_workspace_size_in_bytes`` on the kernel side::

          0:                    fc1_output (BF16, 2 bytes/element)
          fc1_output_end:       fc1_done_counter (Int32 1D)
          fc1_done_counter_end: load_balance_counter (Int32 scalar,
                                atomic_counter mode only)

        There is no scale-factor region: BF16 operands carry no per-block SF.
        The returned ``fc1_output_sf`` slot is an empty view kept only so the
        shared driver's 4-tuple contract (and its byte-level determinism
        check) still applies unchanged.
        """
        problem = self.problem
        intermediate_downproj = problem.intermediate // 2
        data_total_rows = int(self.data_physical_offsets[-1])

        # Hopper cluster peers execute independent WGMMA tiles, so FC1-done
        # counters are indexed at physical CTA token-tile granularity.
        counter_token_tile = (
            self.impl.mma_tiler_mnk[1]
            if self.swap_ab
            else self.impl.mma_tiler_mnk[0]
        )
        counter_slots_upper = (
            (data_total_rows + counter_token_tile - 1) // counter_token_tile
            + problem.experts
        )

        fc1_output_byte_count = (
            data_total_rows * intermediate_downproj
            * torch.bfloat16.itemsize
        )
        fc1_done_counter_byte_count = counter_slots_upper * 4

        ws = self.workspace
        offset = 0

        fc1_output_torch = (
            ws[offset : offset + fc1_output_byte_count]
            .view(torch.uint8)
            .view(torch.bfloat16)
            .reshape(data_total_rows, intermediate_downproj)
        )
        offset += fc1_output_byte_count

        fc1_output_sf_torch = ws[offset : offset + 0].view(torch.uint8)

        # fc1_done_counter relies on the zero-init ``run_kernel`` performs.
        fc1_done_counter_torch = (
            ws[offset : offset + fc1_done_counter_byte_count].view(torch.int32)
        )
        offset += fc1_done_counter_byte_count

        if self.impl.load_balance_mode == "atomic_counter":
            load_balance_counter_torch = ws[offset : offset + 4].view(torch.int32)
            offset += 4
        else:
            load_balance_counter_torch = None

        return (
            fc1_output_torch,
            fc1_output_sf_torch,
            fc1_done_counter_torch,
            load_balance_counter_torch,
        )

    def run_kernel(self) -> None:
        """Instantiate, size, partition, compile and launch the BF16 kernel.

        Mirrors ``Fc12TesterBase.run_kernel`` step for step; the BF16 kernel's
        ``__call__`` has no scale-factor or dequant-scale operands, so the
        runtime kwargs are built here instead of by the shared driver.
        """
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        required = (
            self.activation, self.fc1_weight, self.fc2_weight,
            self.topk_scores, self.fc2_output, self.offs,
        )
        if any(t is None for t in required):
            raise RuntimeError("run_kernel requires generate_inputs first.")

        cluster_size = (
            self.impl.cluster_shape_mnk[0] * self.impl.cluster_shape_mnk[1]
        )
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
            cluster_size
        )
        group_hint = self.impl.group_hint
        if group_hint is None:
            group_hint = max_active_clusters

        if self.impl.enable_static_expert_shape:
            static_expert_shape = (
                self.problem.experts,
                self.problem.intermediate,
                self.problem.hidden,
            )
        else:
            static_expert_shape = None

        # -- 1. Instantiate the kernel (kind hook) --
        common_kwargs = dict(
            mma_tiler_mnk=self.impl.mma_tiler_mnk,
            cluster_shape_mnk=self.impl.cluster_shape_mnk,
            use_2cta_instrs=self.impl.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=self._epilogue_token_tile,
            # No SF plane survives on the BF16 path, but the fused-fc12
            # scheduler still divides by ``sf_padding_block`` and rejects
            # non-positive values, so the constant is passed through inert.
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=self.impl.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=self.impl.force_static_sched,
            clc_bundle_size=self.impl.clc_bundle_size,
            num_sched_stages=self.impl.num_sched_stages,
        )
        kernel = self._instantiate_kernel(common_kwargs)

        # -- 2. Workspace sizing + zero-init (fc1_done_counter and, in
        # atomic_counter mode, load_balance_counter both need it) --
        required_workspace_bytes = kernel.get_workspace_size_in_bytes(
            self.activation, self.fc1_weight
        )
        self.workspace = torch.zeros(
            (required_workspace_bytes,), dtype=torch.uint8, device="cpu"
        ).to("cuda")

        # -- 3. Workspace partition (torch views) --
        (
            fc1_output_torch,
            fc1_output_sf_torch,
            fc1_done_counter_torch,
            load_balance_counter_torch,
        ) = self._partition_workspace(self.impl.mma_tiler_mnk[0])

        # -- 4. Torch -> cute --
        def _to_cute(tensor: torch.Tensor, assumed_align: int = 16):
            cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
            leading_dim = cutlass_torch.get_leading_dim(tensor)
            return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

        activation_cute = _to_cute(self.activation)
        fc1_weight_cute = _to_cute(self.fc1_weight)
        fc1_output_cute = _to_cute(fc1_output_torch)
        fc2_weight_cute = _to_cute(self.fc2_weight)
        fc2_output_cute = _to_cute(self.fc2_output)
        topk_scores_cute = _to_cute(self.topk_scores)
        fc1_done_counter_cute = _to_cute(fc1_done_counter_torch, assumed_align=4)
        offs_cute = _to_cute(self.offs)

        # ``load_balance_counter`` is required iff ``load_balance_mode ==
        # 'atomic_counter'``; in 'static' mode the kwarg is omitted and the
        # kernel const_expr-skips the counter wiring.
        load_balance_counter_cute = (
            _to_cute(load_balance_counter_torch, assumed_align=4)
            if load_balance_counter_torch is not None
            else None
        )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # -- 5. cute.compile --
        runtime_kwargs = dict(
            activation=activation_cute,
            fc1_weight=fc1_weight_cute,
            fc1_output=fc1_output_cute,
            fc2_weight=fc2_weight_cute,
            fc2_output=fc2_output_cute,
            topk_scores=topk_scores_cute,
            fc1_done_counter=fc1_done_counter_cute,
            offs=offs_cute,
            stream=stream,
        )
        if load_balance_counter_cute is not None:
            runtime_kwargs["load_balance_counter"] = load_balance_counter_cute

        for k, v in self._extra_runtime_kwargs().items():
            runtime_kwargs[k] = v

        compile_kwargs = dict(runtime_kwargs)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if self.misc.enable_iket:
            compile_kwargs["options"] = "iket"

        compiled_kernel = cute.compile(kernel, **compile_kwargs)
        self._compiled_kernel = compiled_kernel

        self._ws_fc1_output_torch = fc1_output_torch
        self._ws_fc1_output_sf_torch = fc1_output_sf_torch
        self._launch_runtime_kwargs = runtime_kwargs

        # -- 6. Launch --
        if self.misc.run_target_kernel_only:
            compiled_kernel(**runtime_kwargs)
            torch.cuda.synchronize()
        else:
            self._launch_compiled_kernel_with_torch_profiler(
                compiled_kernel,
                runtime_kwargs,
            )

    def _fc2_tolerance(self) -> Tuple[float, float]:
        # BF16 output: minimum representable relative error = 1/128 ≈ 0.78%
        # (1 BF16 ULP at any value v is v/128, which always exceeds
        # rtol=1e-5 × v regardless of magnitude).  With {0.5, 1.0} input
        # scales the fc2 output reaches ±3K; at that scale 1 BF16 ULP = 8
        # while 1e-5 × 3K = 0.03 — a 267× gap.  1e-2 covers 1 BF16 ULP
        # at all magnitudes (1/128 ≈ 0.78% < 1%) without masking real bugs
        # (genuine GEMM errors are O(1) relative, not 0.78%).
        return 1e-5, 1e-2

    def _fc1_tolerance(self) -> Tuple[float, float]:
        """Tolerance for the BF16 fc1 staging buffer comparison.

        Both sides store ``bf16(fp32_accumulate(...))``; they can differ only
        by FP32 summation order, which shows up as at most a small number of
        BF16 ULPs after the final round.  One BF16 ULP is a relative 2^-8 =
        0.39%, so ``rtol=1e-2`` admits ~2.5 ULP and ``atol=1e-2`` covers
        cancellation-dominated entries.
        """
        return 1e-2, 1e-2

    def validate(self) -> None:
        super().validate()
        self._validate_c_output()

    def _validate_c_output(self) -> None:
        """Compare the kernel ``fc1_c`` tensor to the reference raw gate+up.

        Reference is ``activation[rows].float() @ fc1_weight[e].float()``
        rounded to BF16 -- no clamp, no SwiGLU fold, no topk scaling -- so
        both sides are ``bf16(fp32_accumulate(...))`` and can differ only by
        FP32 summation order.  Same tolerance argument as
        :meth:`_fc1_tolerance`.
        """
        if not self.impl.generate_c:
            return
        if self.misc.skip_ref_check:
            return
        c = getattr(self, "_c_output", None)
        if c is None:
            print("[generate_c] c_output not allocated -- skipped.")
            return
        ref_map = getattr(self, "_ref_fc1_gateup_per_expert", None)
        if not ref_map:
            print("[generate_c] reference fc1 gate+up not available -- skipped.")
            return

        from common.host_utils import compare_and_report_mismatches

        valid = self.valid_tokens_per_expert
        doff = self.data_physical_offsets
        atol, rtol = self._fc1_tolerance()

        print("\n" + "=" * 60)
        print("[generate_c] kernel c_output vs reference fc1 gate+up:")
        # ``compare_and_report_mismatches`` raises AssertionError on the
        # first failing expert, so a mismatch fails the run.
        any_checked = False
        for e in range(self.problem.experts):
            v_e = valid[e]
            ref = ref_map[e]
            if v_e == 0 or ref is None:
                continue
            any_checked = True
            compare_and_report_mismatches(
                c[doff[e] : doff[e] + v_e].to(torch.float32).cpu(),
                ref.to(torch.float32).cpu(),
                name=f"c_output_expert{e}",
                atol=atol, rtol=rtol, max_mismatches=5,
            )
        if not any_checked:
            print("  (no valid tokens routed to any expert)")
        print("=" * 60)

    def _validate_fc1_phase(self) -> None:
        """Compare the kernel-written BF16 fc1 workspace to the reference."""
        if (
            self._ws_fc1_output_torch is None
            or not self._ref_fc1_q_per_expert
        ):
            print("[fc1 phase ablation] skipped (workspace or ref not populated)")
            return

        from common.host_utils import compare_and_report_mismatches

        valid = self.valid_tokens_per_expert
        doff = self.data_physical_offsets
        atol, rtol = self._fc1_tolerance()

        print("\n" + "=" * 60)
        print("[DEBUG fc1] compare_and_report_mismatches per expert:")
        for e in range(self.problem.experts):
            v_e = valid[e]
            ref_bf16 = self._ref_fc1_q_per_expert[e]
            if v_e == 0 or ref_bf16 is None:
                continue
            kernel_bf16 = self._ws_fc1_output_torch[doff[e] : doff[e] + v_e]
            ### TODO: replace to silent check when kernel stable.
            compare_and_report_mismatches(
                kernel_bf16.to(torch.float32).cpu(),
                ref_bf16.to(torch.float32).cpu(),
                name=f"fc1_expert{e}",
                atol=atol, rtol=rtol, max_mismatches=5,
            )
        print("=" * 60)


# =============================================================================
# CLI entry point
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    """argparse setup for the BF16 fused fc12 path."""
    parser = argparse.ArgumentParser(
        description="MoE BF16 GLU fused fc1+fc2 SwiGLU (host-ready harness)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_common_fc12_arguments(parser)
    parser.set_defaults(
        mma_tiler_mnk="64,128,128",
        cluster_shape_mnk="1,1,1",
        use_2cta_instrs=False,
    )

    # -- BF16 Problem --
    parser.add_argument(
        "--kind", type=str, default="bf16",
        choices=list(BF16_KIND_CHOICES),
        help="Element format for activations and weights (BF16 only).",
    )
    parser.add_argument(
        "--flag_batch", type=int, default=1,
        help="dispatch_pull release-flag batch size; 1 == per-token "
        "baseline, larger amortizes the device fence over more tokens.",
    )
    parser.add_argument(
        "--gate_up_clamp", type=float, default=None,
        help="DeepSeek-V4 swiglu_limit: clamp gate/up pre-activations before SiLU.",
    )
    parser.add_argument(
        "--swap_ab", action="store_true",
        help="Use the Hopper weight-as-A M128/M256xN swap-AB kernel.",
    )
    parser.add_argument(
        "--pingpong", action="store_true",
        help="Alternate complete task tiles across two WGMMA+epilogue warpgroups.",
    )
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
             "cluster of 2: --swap_ab with --cluster_shape_mnk 1,2,1 or "
             "non-swap with --cluster_shape_mnk 2,1,1.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    problem = ProblemDesc(
        tokens_after_topk=args.tokens_after_topk,
        experts=args.experts,
        balance_route=args.balance_route,
        hidden=args.hidden,
        intermediate=args.intermediate,
        simulate_ep=args.simulate_ep,
        fc2_output_dtype=args.fc2_output_dtype,
        kind=args.kind,
        gate_up_clamp=args.gate_up_clamp,
    )

    mma_tiler_mnk = parse_tuple(args.mma_tiler_mnk)
    if args.swap_ab and mma_tiler_mnk == (64, 128, 128):
        mma_tiler_mnk = (128, 32, 128) if args.pingpong else (256, 32, 128)

    impl = ImplDesc(
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mnk=parse_tuple(args.cluster_shape_mnk),
        use_2cta_instrs=args.use_2cta_instrs,
        enable_static_expert_shape=args.enable_static_expert_shape,
        force_static_sched=not args.dynamic_sched,
        clc_bundle_size=args.clc_bundle_size,
        num_sched_stages=args.num_sched_stages,
        load_balance_mode=args.load_balance_mode,
        group_hint=args.group_hint,
        flag_batch=args.flag_batch,
        generate_c=args.generate_c,
        tail_split_pairs=args.tail_split_pairs,
    )

    misc = MiscDesc(
        perf_run=args.perf_run,
        skip_ref_check=args.skip_ref_check,
        run_target_kernel_only=args.run_target_kernel_only,
        enable_debug_checks=args.enable_debug_checks,
        ref_compute_graph=args.ref_compute_graph,
        enable_iket=args.enable_iket,
        seed=args.seed,
        verbose=args.verbose,
        perf_warmup=args.perf_warmup,
        perf_iters=args.perf_iters,
    )

    tester = SwigluBf16Fc12Tester(
        problem,
        impl,
        misc,
        swap_ab=args.swap_ab,
        pingpong=args.pingpong,
    )
    tester.run()


if __name__ == "__main__":
    main()
    exit(0)
