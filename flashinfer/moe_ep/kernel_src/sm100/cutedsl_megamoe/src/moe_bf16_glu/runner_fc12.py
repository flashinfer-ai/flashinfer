# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Host driver for the MegaMoE BF16 GLU fused fc1+fc2 kernel.
"""


import argparse
import os
import sys
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
    ProblemDesc,
    MiscDesc,
    Fc12TesterBase,
    add_common_fc12_arguments,
    parse_tuple,
)
from moe_bf16_glu.runner_common import TrainingImplDesc
from moe_bf16_glu.epilogue_bf16 import Fc1GateUpInterleave
from moe_nvfp4_swapab.runner_common import (
    offs_to_group_sizes,
)
from common.host_utils import compare_and_report_mismatches


# =============================================================================
# BF16 tester
# =============================================================================


class SwigluBf16Fc12Tester(Fc12TesterBase):
    """BF16 host-side input/reference/launch/validation driver."""

    def __init__(
        self,
        problem: ProblemDesc,
        impl: TrainingImplDesc,
        misc: MiscDesc,
    ) -> None:
        super().__init__(problem, impl, misc)
        # BF16 pipeline: inputs, the fc1 hand-off and the fc2 output are all
        # BF16.  The kernel's epilogue cast and c_dtype are hardwired to
        # BFloat16, so both the problem kind and the fc2 output dtype (which
        # drives the runner-side output allocation) must be bf16.
        if problem.kind != "bf16":
            raise ValueError(
                f"SwigluBf16Fc12Tester requires problem.kind == 'bf16'; "
                f"got {problem.kind!r}."
            )
        if problem.fc2_output_dtype is not torch.bfloat16:
            raise ValueError(
                f"SwigluBf16Fc12Tester requires "
                f"fc2_output_dtype=torch.bfloat16; "
                f"got {problem.fc2_output_dtype}."
            )
        # Store-granularity shape checks: the fc1 STG path (use_stg_fc1)
        # stores full 256-wide gate+up tiles without N predication; the
        # default TMA store path clamps at the tensor extent, so its only
        # hard requirement is the gate/up interleave pair unit (2 x 32 = 64
        # gate+up columns).  fc2 stores 32-wide hidden subtiles.
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
        # BF16 currently only supports the (M=256, N=256) mma tile with
        # 2-CTA instructions.
        m, n, _k = impl.mma_tiler_mnk
        if (m, n) != (256, 256) or not impl.use_2cta_instrs:
            raise ValueError(
                "BF16 fused fc12 currently only supports mma_tiler (M, N) = "
                "(256, 256) with use_2cta_instrs=True; "
                f"got mma_tiler_mnk={impl.mma_tiler_mnk}, "
                f"use_2cta_instrs={impl.use_2cta_instrs}."
            )

    @property
    def _epilogue_token_tile(self) -> int:
        # generate_c stores a (cta_tile_m=128)-row TMA tile, so physical row
        # offsets must be 128-aligned.  Override to 128 when generate_c=True so
        # the scheduler uses 128-aligned data_physical_offsets for all tensors.
        if self.impl.generate_c:
            return 128
        from moe_nvfp4_swapab.epilogue import EpilogueTokenTile
        return EpilogueTokenTile

    # ------------------------------------------------------------------
    # BF16 tensor creation
    # ------------------------------------------------------------------

    def _create_bf16_tensor(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create a BF16 tensor.

        Two modes (selected by ``misc.perf_run``):
          - correctness: sparse {0, +1, -1} (80 % zeros / 10 % +1 / 10 % -1);
            every value is exactly representable in BF16.
          - perf: near-uniform random BF16 bit patterns (timing only).
            ``torch.rand`` cannot reproduce bit-level entropy (sign always 0,
            exponent concentrated around 126), so draw uniform int16 bit
            patterns and remap the NaN/Inf encodings (exponent all-ones,
            0.4 % of patterns) to the max finite exponent by clearing one
            exponent bit.
        """
        if self.misc.perf_run:
            n = 1
            for s in shape:
                n *= s
            bits = torch.randint(
                -32768, 32768, (n,), dtype=torch.int16, device="cuda"
            )
            nan_inf = (bits & 0x7F80) == 0x7F80
            bits = torch.where(nan_inf, bits & ~0x0080, bits)
            return bits.view(torch.bfloat16).reshape(shape)
        fp32 = torch.zeros(shape, dtype=torch.float32, device="cuda")
        rand = torch.rand(shape, device="cuda")
        fp32[rand < 0.10] = 1.0
        fp32[(rand >= 0.10) & (rand < 0.20)] = -1.0
        return fp32.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Kind hooks: input / output tensor creation
    # ------------------------------------------------------------------

    def _fc2_output_shape(self, data_total_rows: int) -> Tuple[int, ...]:
        # Flat 2D (token_max, hidden); epilogue_bf16.py reads shape[1].
        return (data_total_rows, self.problem.hidden)

    def _create_input_data_tensors(self, data_total_rows: int) -> None:
        problem = self.problem
        hidden = problem.hidden
        intermediate = problem.intermediate
        experts = problem.experts

        # -- activation: (data_total_rows, hidden) bf16, hidden stride-1 --
        self.activation = self._create_bf16_tensor((data_total_rows, hidden))

        # -- fc1_weight: (experts, intermediate, hidden) -> permute ->
        #    (experts, hidden, intermediate), hidden stride-1 --
        self.fc1_weight = self._create_bf16_tensor(
            (experts, intermediate, hidden)
        ).permute(0, 2, 1)

        # -- fc2_weight: (experts, hidden, inter//2) -> permute ->
        #    (experts, inter//2, hidden), inter//2 stride-1 --
        self.fc2_weight = self._create_bf16_tensor(
            (experts, hidden, intermediate // 2)
        ).permute(0, 2, 1)

    # ``_init_topk_scores`` is NOT overridden: the base fills random
    # [0.5, 1.5] scores on valid rows (padding rows 0), and topk gets
    # exercised in both graph modes:
    #   deepgemm     -> kernel folds the weight in the fc1 epilogue
    #                   (``apply_topk_in_fc1`` derived in _instantiate_kernel)
    #   transformers -> kernel stores unweighted output; the reference is
    #                   weighted post-fc2 (``_apply_topk_post_fc2``) and the
    #                   comparison weights the actual (``_actual_for_compare``)

    def _apply_topk_post_fc2(
        self, fc2_fp32: torch.Tensor, topk_slice: torch.Tensor
    ) -> torch.Tensor:
        # HF transformers semantics: topk weight applied AFTER fc2 GEMM.
        # The kernel stores the unweighted output in BF16 and the downstream
        # reduce (emulated by ``_actual_for_compare``) multiplies that
        # BF16-rounded value -- mirror the same rounding chain here so the
        # comparison stays bit-exact.
        if self.misc.ref_compute_graph == "transformers":
            fc2_bf16 = fc2_fp32.to(torch.bfloat16).float()
            return fc2_bf16 * topk_slice.unsqueeze(-1)
        return fc2_fp32

    def _alloc_fc2_output(self, data_total_rows: int) -> None:
        # 0xFF byte fill: bf16/fp16 0xFFFF = NaN -- kernel output overwriting
        # valid rows is easy to distinguish from "kernel never touched this
        # row".  The output is a flat 2D ``(token_max, hidden)`` buffer.
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
    # Input generation
    # ------------------------------------------------------------------

    def generate_inputs(self) -> None:
        """Build offs and all input / output tensors."""
        # -- 1. offs (valid cumsum) --
        self.offs = self._generate_offs()
        valid_tokens = offs_to_group_sizes(self.offs)
        self.valid_tokens_per_expert = valid_tokens

        # -- 2. Physical offsets (per-expert rows padded to
        # ``_epilogue_token_tile``) --
        data_offsets, _ = self._compute_physical_offsets(valid_tokens)
        self.data_physical_offsets = data_offsets
        data_total_rows = data_offsets[-1]

        # Short-circuit paths that don't need fully-initialized data
        # (perf simulator + zero-routed-token EP rank).
        if self.misc.run_target_kernel_only or data_total_rows == 0:
            self._generate_inputs_skeleton(valid_tokens, data_total_rows)
            return

        # -- 3. activation / fc1_weight / fc2_weight --
        self._create_input_data_tensors(data_total_rows)

        # -- 4. topk_scores --
        self._init_topk_scores(data_total_rows)

        # -- 5. fc2_output (0xFF sentinel fill) --
        self._alloc_fc2_output(data_total_rows)

        # -- 6. Workspace placeholder --
        self._alloc_workspace_placeholder()

        torch.cuda.synchronize()

    def _generate_inputs_skeleton(
        self,
        valid_tokens_per_expert: List[int],
        data_total_rows: int,
    ) -> None:
        """Allocate BF16 tensors with correct shape / stride but no data init.
        """
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
            (experts, hidden, intermediate // 2), dtype=torch.bfloat16,
            device="cuda",
        ).permute(0, 2, 1)

        # -- topk_scores --
        self.topk_scores = torch.empty(
            (data_total_rows,), dtype=torch.float32, device="cuda",
        )

        # -- fc2_output --
        self._alloc_fc2_output_skeleton(data_total_rows)

        # -- Workspace placeholder (run_kernel reallocates to precise size) --
        self.workspace = torch.zeros((1 << 20,), dtype=torch.uint8, device="cpu").to(
            "cuda"
        )

    # ------------------------------------------------------------------
    # Reference (bit-exact dense GEMM, per expert)
    # ------------------------------------------------------------------

    def compute_reference(self) -> None:
        """Per-expert reference for the BF16 fused fc1+fc2.

        Both GEMMs run on the bit-exact dense launcher
        (``_DenseGemmReferenceLauncher``), which accumulates with the same
        tcgen05 fp32 semantics as the fused kernel -- the comparison is free
        of fp32 accumulation-order noise.  Per expert:

          1. ``fc1_fp32 = ref_mm(act, w1)``               (v_e, intermediate)
          2. SwiGLU fold over the gate/up interleave (+ optional clamp)
          3. topk pre-multiply in the ``deepgemm`` graph (mirrors the kernel,
             whose ``apply_topk_in_fc1`` is derived from the same knob); the
             ``transformers`` graph instead post-multiplies after fc2
             (``_apply_topk_post_fc2``)
          4. BF16 hand-off cast: ``fc1_bf16 = swiglu.to(bf16)`` -- the same
             value the kernel writes to the ``fc1_output`` workspace
          5. ``fc2_fp32 = ref_mm(fc1_bf16, w2)`` (v_e, hidden) -- the BF16
             hand-off is fc2's A operand in its native dtype
          6. cast to ``fc2_output_dtype`` into ``fc2_output_ref``

        Populates ``fc2_output_ref``, ``_ref_fc1_q_per_expert`` (the BF16
        hand-off values) and ``_ref_fc1_gateup_per_expert`` (pre-SwiGLU
        gate+up, BF16, consumed by the generate_c check).
        """
        if self.activation is None or self.offs is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")
        if self.misc.skip_ref_check:
            return

        from moe_bf16_glu.mega_reference_bf16 import (
            _DenseGemmReferenceLauncher,
            reference_expert_fc12,
        )

        problem = self.problem
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        data_total_rows = data_offsets[-1]
        gate_up_clamp = getattr(problem, "gate_up_clamp", None)

        # Bit-exact dense GEMM launcher, shared across this expert sweep
        # (compiles per dtype-key on first use, cached internally).
        if getattr(self, "_ref_mm", None) is None:
            self._ref_mm = _DenseGemmReferenceLauncher(
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
            )

        # Allocate via uint8 bytes then reinterpret (same bulletproof trick
        # as ``_alloc_fc2_output``).
        ref_bytes = torch.zeros(
            (data_total_rows, problem.hidden * problem.fc2_output_dtype.itemsize),
            dtype=torch.uint8, device="cuda",
        )
        self.fc2_output_ref = ref_bytes.view(problem.fc2_output_dtype).reshape(
            data_total_rows, problem.hidden
        )

        self._ref_fc1_q_per_expert = [None] * problem.experts
        self._ref_fc1_gateup_per_expert = [None] * problem.experts

        for expert_idx in range(problem.experts):
            v_e = valid_tokens[expert_idx]
            if v_e == 0:
                continue
            d_start = data_offsets[expert_idx]

            act_slice = self.activation[d_start : d_start + v_e]
            topk_slice = self.topk_scores[d_start : d_start + v_e]

            # K-major (n, k) weight views: fc1_weight[e] is (hidden, inter)
            # with hidden stride-1; its transpose is the contiguous
            # (inter, hidden) launcher-B layout.  Same for fc2_weight[e].
            w1_nk = self.fc1_weight[expert_idx].transpose(0, 1)
            w2_nk = self.fc2_weight[expert_idx].transpose(0, 1)

            fc2_fp32, fc1_bf16, fc1_fp32 = reference_expert_fc12(
                ref_mm=self._ref_mm,
                act=act_slice,
                fc1_weight=w1_nk,
                fc2_weight=w2_nk,
                intermediate=problem.intermediate,
                hidden=problem.hidden,
                gate_up_interleave=Fc1GateUpInterleave,
                gate_up_clamp=gate_up_clamp,
                topk_weights=topk_slice,
                ref_compute_graph=self.misc.ref_compute_graph,
            )

            # Raw pre-SwiGLU gate+up snapshot (kernel c_dtype is BFloat16).
            self._ref_fc1_gateup_per_expert[expert_idx] = fc1_fp32.to(
                torch.bfloat16
            )
            self._ref_fc1_q_per_expert[expert_idx] = fc1_bf16

            fc2_fp32 = self._apply_topk_post_fc2(fc2_fp32, topk_slice)
            self.fc2_output_ref[d_start : d_start + v_e] = fc2_fp32.to(
                problem.fc2_output_dtype
            )

    # ------------------------------------------------------------------
    # Workspace partition
    # ------------------------------------------------------------------

    def _partition_workspace(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Slice the opaque byte ``self.workspace`` into per-section views.

        Layout MUST match ``Sm100SwigluBf16Fc12Kernel.get_workspace_size_in_bytes``:

          0:                    fc1_output (BF16, (rows, intermediate/2))
          fc1_output_end:       fc1_done_counter (Int32 1D)
          fc1_done_counter_end: load_balance_counter (Int32 scalar,
                                atomic_counter mode only)

        ``fc1_done_counter`` is indexed per cluster token block: the token
        axis is M in this kernel (A = activations), so one slot covers
        ``per-CTA tile M x cluster_m`` tokens, matching the scheduler's
        ``cumulative_token_block_count`` units
        (``ceil(valid_e / cluster_tile_m)`` per expert).

        Returns ``(fc1_output_torch, fc1_done_counter_torch,
        load_balance_counter_torch_or_None)`` as torch views over
        ``self.workspace``.
        """
        problem = self.problem
        experts = problem.experts
        intermediate_downproj = problem.intermediate // 2
        data_total_rows = int(self.data_physical_offsets[-1])

        per_cta_tile_m = self.impl.mma_tiler_mnk[0] // (
            2 if self.impl.use_2cta_instrs else 1
        )
        cluster_tile_tokens = per_cta_tile_m * self.impl.cluster_shape_mnk[0]
        counter_slots_upper = (
            (data_total_rows + cluster_tile_tokens - 1) // cluster_tile_tokens
            + experts
        )

        fc1_output_byte_count = data_total_rows * intermediate_downproj * 2  # BF16
        fc1_done_counter_byte_count = counter_slots_upper * 4

        ws = self.workspace
        offset = 0

        # -- fc1_output --
        fc1_output_torch = (
            ws[offset : offset + fc1_output_byte_count]
            .view(torch.uint8)
            .view(torch.bfloat16)
            .reshape(data_total_rows, intermediate_downproj)
        )
        offset += fc1_output_byte_count

        # -- fc1_done_counter: Int32 1D, zero-init (host responsibility,
        # done by the workspace zero-alloc in run_kernel).
        fc1_done_counter_torch = (
            ws[offset : offset + fc1_done_counter_byte_count]
            .view(torch.int32)
        )
        offset += fc1_done_counter_byte_count

        # -- load_balance_counter: Int32 scalar (atomic_counter mode only).
        if self.impl.load_balance_mode == "atomic_counter":
            load_balance_counter_torch = ws[offset : offset + 4].view(torch.int32)
            offset += 4
        else:
            load_balance_counter_torch = None

        return (
            fc1_output_torch,
            fc1_done_counter_torch,
            load_balance_counter_torch,
        )

    # ------------------------------------------------------------------
    # Kernel call
    # ------------------------------------------------------------------

    def run_kernel(self) -> None:
        """Instantiate the fused BF16 kernel, partition workspace, compile,
        launch.

          1. Instantiate the kernel via :meth:`_instantiate_kernel`.
          2. Query workspace size, reallocate + zero-init ``self.workspace``.
          3. Partition the workspace into per-section torch views.
          4. Convert every torch tensor to a cute tensor.
          5. ``cute.compile`` once and stash the compiled callable.
          6. Launch.
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

        # Cluster size + max_active_clusters + group_hint default fill.
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

        # -- 1. Instantiate the kernel --
        common_kwargs = dict(
            mma_tiler_mnk=self.impl.mma_tiler_mnk,
            cluster_shape_mnk=self.impl.cluster_shape_mnk,
            use_2cta_instrs=self.impl.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=self._epilogue_token_tile,
            load_balance_mode=self.impl.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=self.impl.force_static_sched,
            clc_bundle_size=self.impl.clc_bundle_size,
            num_sched_stages=self.impl.num_sched_stages,
        )
        kernel = self._instantiate_kernel(common_kwargs)

        # -- 2. Workspace sizing + zero-init --
        # Zero-init is REQUIRED before each launch (fc1_done_counter and,
        # under atomic_counter mode, load_balance_counter both rely on
        # zero-init semantics).
        required_workspace_bytes = kernel.get_workspace_size_in_bytes(
            self.activation, self.fc1_weight
        )
        self.workspace = torch.zeros(
            (required_workspace_bytes,), dtype=torch.uint8, device="cpu"
        ).to("cuda")

        # -- 3. Workspace partition (torch views) --
        (
            fc1_output_torch,
            fc1_done_counter_torch,
            load_balance_counter_torch,
        ) = self._partition_workspace()

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

        # Subclass hook: inject extra tensor kwargs (e.g. generate_c output).
        for k, v in self._extra_runtime_kwargs().items():
            runtime_kwargs[k] = v

        compile_kwargs = dict(runtime_kwargs)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if self.misc.enable_iket:
            compile_kwargs["options"] = "iket"

        compiled_kernel = cute.compile(kernel, **compile_kwargs)
        self._compiled_kernel = compiled_kernel

        # Stash launch-side handles + workspace partition torch views for the
        # debug paths (determinism re-launch + fc1 workspace readback).
        self._ws_fc1_output_torch = fc1_output_torch
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

    def _instantiate_kernel(self, common_kwargs: dict):
        import cutlass
        from moe_bf16_glu.kernel_bf16_glu_fc12 import Sm100SwigluBf16Fc12Kernel

        kernel = Sm100SwigluBf16Fc12Kernel(
            **common_kwargs,
            ab_dtype=cutlass.BFloat16,
            epi_flag_batch=self.impl.epi_flag_batch,
            gate_up_clamp=self.problem.gate_up_clamp,
            # The reference graph decides where topk lands: deepgemm folds it
            # into the fc1 epilogue; transformers leaves the kernel output
            # unweighted (a downstream reduce applies it).
            apply_topk_in_fc1=self.misc.ref_compute_graph == "deepgemm",
            generate_c=self.impl.generate_c,
            use_stg_fc1=self.impl.use_stg_fc1,
        )
        self._kernel_c_dtype = kernel.c_dtype
        return kernel

    def _extra_runtime_kwargs(self) -> dict:
        if not self.impl.generate_c:
            return {}
        import cutlass
        import cutlass.torch as cutlass_torch

        _cutlass_to_torch = {
            cutlass.BFloat16: torch.bfloat16,
            cutlass.Float32: torch.float32,
            cutlass.Float16: torch.float16,
        }
        torch_c_dtype = _cutlass_to_torch[self._kernel_c_dtype]
        tokens_total = self.fc2_output.shape[0]
        intermediate_gateup = self.problem.intermediate
        self._c_output = torch.zeros(
            (tokens_total, intermediate_gateup), dtype=torch_c_dtype, device="cuda"
        )
        c_cute = cutlass_torch.from_dlpack(self._c_output, assumed_align=16)
        leading_dim = cutlass_torch.get_leading_dim(self._c_output)
        c_cute = c_cute.mark_layout_dynamic(leading_dim=leading_dim)
        return {"fc1_c": c_cute}

    # ------------------------------------------------------------------
    # Debug checks + validation
    # ------------------------------------------------------------------

    def _check_kernel_determinism(self) -> None:
        """Relaunch and byte-compare fc2 output plus the fc1 workspace."""
        if (
            self._compiled_kernel is None
            or self._launch_runtime_kwargs is None
            or self.fc2_output is None
            or self._ws_fc1_output_torch is None
        ):
            print("[determinism check] skipped (kernel not launched yet)")
            return

        # Snapshot first-launch outputs as raw bytes.
        fc2_first = self.fc2_output.view(torch.uint8).clone()
        fc1_out_first = self._ws_fc1_output_torch.view(torch.uint8).clone()

        # Counters require zero-init per launch.
        self.workspace.zero_()

        self._compiled_kernel(**self._launch_runtime_kwargs)
        torch.cuda.synchronize()

        fc2_curr = self.fc2_output.view(torch.uint8)
        fc1_out_curr = self._ws_fc1_output_torch.view(torch.uint8)

        fc2_byte_diff = int((fc2_first != fc2_curr).sum().item())
        fc1_out_byte_diff = int((fc1_out_first != fc1_out_curr).sum().item())

        print("=" * 60)
        print("[determinism check] re-launched kernel with identical inputs")
        print(
            f"  fc2_output     byte-diff: {fc2_byte_diff:>10d} / "
            f"{fc2_first.numel():>10d} "
            f"({fc2_byte_diff / max(fc2_first.numel(), 1) * 100:7.4f}%)"
        )
        print(
            f"  fc1_output     byte-diff: {fc1_out_byte_diff:>10d} / "
            f"{fc1_out_first.numel():>10d} "
            f"({fc1_out_byte_diff / max(fc1_out_first.numel(), 1) * 100:7.4f}%)"
        )
        if fc2_byte_diff == 0 and fc1_out_byte_diff == 0:
            print(
                "  -> kernel is BIT-DETERMINISTIC across re-launches; any "
                "mismatch with ref must come from kernel-vs-host "
                "accumulation order, not HW race"
            )
        else:
            print(
                "  -> kernel is NON-DETERMINISTIC (different bytes on "
                "back-to-back identical launches); race condition / stale "
                "barrier suspected"
            )
        print("=" * 60)

    def _fc2_tolerance(self) -> Tuple[float, float]:
        # BF16 output: minimum representable relative error = 1/128 ≈ 0.78%
        # (1 BF16 ULP at any value v is v/128, which always exceeds
        # rtol=1e-5 × v regardless of magnitude).  1e-2 covers 1 BF16 ULP
        # at all magnitudes without masking real bugs (genuine GEMM errors
        # are O(1) relative, not 0.78%).
        return 1e-5, 1e-2

    def _validate_fc1_phase(self) -> None:
        """Compare kernel-written BF16 fc1 workspace to reference per expert.

        Both sides are BF16 rounded from fp32 SwiGLU results; residual
        disagreement is fp32 accumulation order inside the fc1 GEMM.
        """
        if (
            self._ws_fc1_output_torch is None
            or not self._ref_fc1_q_per_expert
        ):
            print("[fc1 phase ablation] skipped (workspace or ref not populated)")
            return

        valid = self.valid_tokens_per_expert
        doff = self.data_physical_offsets

        print("\n" + "=" * 60)
        print("[DEBUG fc1] compare_and_report_mismatches per expert:")
        for e in range(self.problem.experts):
            v_e = valid[e]
            ref_q = self._ref_fc1_q_per_expert[e]
            if v_e == 0 or ref_q is None:
                continue
            kq = self._ws_fc1_output_torch[doff[e] : doff[e] + v_e]
            ### TODO: replace with silent check when kernel stable.
            compare_and_report_mismatches(
                kq.float().cpu(),
                ref_q.float().cpu(),
                name=f"fc1_expert{e}",
                atol=1e-2, rtol=1e-2, max_mismatches=5,
            )
        print("=" * 60)

    def validate(self) -> None:
        super().validate()
        self._validate_c_output()

    def _validate_c_output(self) -> None:
        """Per-element comparison of kernel C output vs reference fc1 gate+up."""
        if not self.impl.generate_c:
            return
        c = getattr(self, "_c_output", None)
        if c is None:
            print("[generate_c] c_output not allocated — skipped.")
            return
        if not self._ref_fc1_gateup_per_expert:
            print("[generate_c] reference fc1 gate+up not available — skipped.")
            return

        valid = self.valid_tokens_per_expert
        doff = self.data_physical_offsets

        print("\n" + "=" * 60)
        print("[generate_c] kernel c_output vs reference fc1 gate+up (per-element):")
        any_checked = False
        for e in range(self.problem.experts):
            v_e = valid[e]
            ref = self._ref_fc1_gateup_per_expert[e]
            if v_e == 0 or ref is None:
                continue
            any_checked = True
            kernel_c = c[doff[e] : doff[e] + v_e].float().cpu()
            ref_c = ref.float().cpu()
            compare_and_report_mismatches(
                kernel_c, ref_c,
                name=f"c_output_expert{e}",
                atol=1e-5, rtol=1e-2, max_mismatches=5,
            )
        if not any_checked:
            print("  (no valid tokens routed to any expert)")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Verbose information dump
    # ------------------------------------------------------------------

    def _print_layout_info(self) -> None:
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets

        for name in (
            "activation", "fc1_weight", "fc2_weight",
            "topk_scores", "fc2_output", "workspace",
        ):
            t = getattr(self, name)
            print(
                f"{name}: shape={tuple(t.shape)}  "
                f"stride={t.stride()}  dtype={t.dtype}"
            )
        print(f"offs (valid cumsum): {self.offs.cpu().tolist()}")
        print(f"  valid_tokens_per_expert: {valid_tokens}")
        print(f"  data_physical_offsets:   {data_offsets}")

        self._print_scheduler_layout()


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

    # -- BF16-only Problem --
    parser.add_argument(
        "--kind", type=str, default="bf16",
        choices=["bf16"],
        help="Data element format: bf16 (BF16 A/B operands).",
    )
    parser.add_argument(
        "--flag_batch", type=int, default=1,
        help="dispatch_pull release-flag batch size; 1 == per-token "
        "baseline, larger amortizes the device fence over more tokens.",
    )
    parser.add_argument(
        "--token_back_mode", type=str, default="epi_warps",
        choices=["epi_warps", "standalone_warps", "reuse_dispatch_warps"],
        help="Where the cross-rank fc2 push-back runs: epi_warps (epilogue "
             "STG redirect, default), standalone_warps (dedicated warps 12-15), "
             "or reuse_dispatch_warps (dispatch warps 8-11).",
    )
    parser.add_argument(
        "--epi_flag_batch", type=str, default="1,1",
        help="Done-counter publish batching as 'fc1,fc2' (e.g. '2,4'). "
             "Each component must be in [1, 32].",
    )
    parser.add_argument(
        "--gate_up_clamp", type=float, default=None,
        help="DeepSeek-V4 swiglu_limit: clamp gate/up pre-activations before SiLU.",
    )
    parser.add_argument(
        "--generate_c", action="store_true", default=False,
        help="Save raw fc1 accumulator (gate+up, Float32) to a separate C tensor "
             "before SwiGLU.  Allocates extra SMEM; reduces AB pipeline stages.",
    )
    parser.add_argument(
        "--use_stg_fc1", action="store_true", default=False,
        help="Write fc1 BF16 output directly to GMEM via STG (RMEM→GMEM) "
             "instead of the default R2S+TMA path.  Eliminates sD SMEM staging; "
             "may increase AB pipeline stages.",
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

    impl = TrainingImplDesc(
        mma_tiler_mnk=parse_tuple(args.mma_tiler_mnk),
        cluster_shape_mnk=parse_tuple(args.cluster_shape_mnk),
        use_2cta_instrs=args.use_2cta_instrs,
        enable_static_expert_shape=args.enable_static_expert_shape,
        force_static_sched=not args.dynamic_sched,
        clc_bundle_size=args.clc_bundle_size,
        num_sched_stages=args.num_sched_stages,
        load_balance_mode=args.load_balance_mode,
        group_hint=args.group_hint,
        flag_batch=args.flag_batch,
        token_back_mode=args.token_back_mode,
        epi_flag_batch=parse_tuple(args.epi_flag_batch),
        generate_c=args.generate_c,
        use_stg_fc1=args.use_stg_fc1,
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

    tester = SwigluBf16Fc12Tester(problem, impl, misc)
    tester.run()


if __name__ == "__main__":
    main()
    exit(0)
