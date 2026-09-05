# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""ReLU2 specialization for the vendored NVFP4 MegaMoE kernel.

The kernel-team drop is intentionally immutable.  This module specializes its
epilogue objects while deliberately keeping the top-level callable an exact
``Sm100MegaMoEKernel`` instance.  CuTeDSL resolves the vendor kernel's nested
``super().__call__`` from the runtime class; passing a Python subclass makes
that call recurse into ``Sm100MegaMoEKernel.__call__`` with the FC12-only
arguments.  An instance-bound setup hook avoids that unsupported inheritance
edge while retaining the exact FC1 ``2 * I`` physical layout.

For ReLU2, the first plane is the semantic W1 projection and the second plane
is padding.  The device epilogue never reads the padding values when it
computes ``relu(x) ** 2``.

Keep this adapter close to the upstream class structure: a future vendor drop
that changes the epilogue orchestration should fail the focused source tests
instead of silently falling back to SwiGLU.
"""

# Do not enable ``from __future__ import annotations`` in this module.  CuTeDSL
# consumes the live annotations on @cute.jit functions during tracing; PEP 563
# strings are not a supported substitute (the immutable vendor modules follow
# the same rule).

from types import MethodType
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64

from moe_nvfp4_swapab.epilogue_refactor import (
    NvFp4OptinalEpiArgs,
    SwapABFc1Epilogue,
    SwapABFc2Epilogue,
    SwapABSwigluFp4Epilogue,
)
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase
from moe_nvfp4_swapab.megamoe_kernel import Sm100MegaMoEKernel
from moe_nvfp4_swapab.moe_persistent_scheduler import (
    MoESchedConsumer,
    MoESchedExtension,
)
from src.flag_batch import GpuReleaseFlagBatchTracker
from src.iket_compat import iket


class SwapABRelu2Fc1Epilogue(SwapABFc1Epilogue):
    """FC1 epilogue that consumes only W1 and applies ungated ReLU2."""

    @cute.jit
    def alpha_swiglu_clamp(
        self,
        gate_rmem: cute.Tensor,
        up_rmem: cute.Tensor,
        alpha_val: Optional[cutlass.Float32],
    ) -> cute.Tensor:
        """Return ``relu(alpha * gate_rmem) ** 2``.

        The inherited FC1 orchestration calls this historical method name.
        ``up_rmem`` is deliberately used only for compile-time shape checks;
        none of its values enter the result.  The host weight preprocessor
        supplies a zero padding plane there so the physical 2*I FC1 contract is
        explicit even though the semantic model has a single I-wide W1.
        """
        for name, tensor in (("gate_rmem", gate_rmem), ("up_rmem", up_rmem)):
            if cutlass.const_expr(tensor.element_type is not cutlass.Float32):
                raise TypeError(
                    f"relu2: {name} must be Float32, got {tensor.element_type}"
                )
            if cutlass.const_expr(tensor.memspace != AddressSpace.rmem):
                raise ValueError(
                    f"relu2: {name} must be a register (rmem) tensor, "
                    f"got address space {tensor.memspace}"
                )
            if cutlass.const_expr(cute.rank(tensor) != 1):
                raise ValueError(
                    f"relu2: {name} must be 1D, got rank {cute.rank(tensor)}"
                )
            if cutlass.const_expr(cute.size(tensor) % 2 != 0):
                raise ValueError(
                    f"relu2: {name} element count must be even, got {cute.size(tensor)}"
                )
        if cutlass.const_expr(cute.size(gate_rmem) != cute.size(up_rmem)):
            raise ValueError(
                "relu2: semantic and padding planes must have equal size, got "
                f"{cute.size(gate_rmem)} vs {cute.size(up_rmem)}"
            )

        n = cute.size(gate_rmem)
        out = cute.make_rmem_tensor((n,), cutlass.Float32)
        zero = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(0, n, 2):
            g0 = gate_rmem[i]
            g1 = gate_rmem[i + 1]
            if cutlass.const_expr(alpha_val is not None):
                g0, g1 = cute.arch.mul_packed_f32x2((g0, g1), (alpha_val, alpha_val))
            r0 = cute.arch.fmax(g0, zero)
            r1 = cute.arch.fmax(g1, zero)
            sq0, sq1 = cute.arch.mul_packed_f32x2((r0, r1), (r0, r1))
            out[i] = sq0
            out[i + 1] = sq1
        return out


class SwapABRelu2Fp4Epilogue(SwapABSwigluFp4Epilogue):
    """Swap-AB epilogue orchestration with the ReLU2 FC1 specialization."""

    @cute.jit
    def run(
        self,
        epi_smem_storage,
        tmem_ptr: cute.Pointer,
        acc_pipeline,
        sched_consumer: MoESchedConsumer,
        sched_ext: MoESchedExtension,
        tma_atom_fc1_output: cute.CopyAtom,
        fc1_output: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        fc2_output: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        tidx: cutlass.Int32,
        optional_epi_args: NvFp4OptinalEpiArgs = None,
        token_comm_args=None,
    ):
        # This mirrors SwapABSwigluFp4Epilogue.run from the immutable source
        # drop.  The only semantic change is the FC1 epilogue class below.
        if cutlass.const_expr(optional_epi_args is None):
            optional_epi_args = NvFp4OptinalEpiArgs(
                fc1_alpha=None,
                fc2_alpha=None,
                fc1_norm_const=None,
                topk_scores=None,
            )
        tmem_acc = cute.make_tensor(
            cute.recast_ptr(tmem_ptr, dtype=cutlass.Float32),
            cute.make_layout(
                self.tmem_acc_layout_py_obj[0],
                stride=self.tmem_acc_layout_py_obj[1],
            ),
        )

        fc1_epi = SwapABRelu2Fc1Epilogue(
            self,
            tidx,
            epi_smem_storage,
            sched_ext,
            tma_atom_fc1_output,
            fc1_output,
            fc1_output_sf,
            fc1_done_counter,
            optional_epi_args,
        )
        fc2_epi = SwapABFc2Epilogue(
            self,
            tidx,
            epi_smem_storage,
            fc2_output,
            token_comm_args,
            optional_epi_args,
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_pipeline_stages
        )
        wait_only_named_barrier = pipeline.NamedBarrier(
            barrier_id=self._EpilogueSyncWaitBarId,
            num_threads=32 * self._EpilogueWarpCnt,
        )
        is_odd_turn = cutlass.Int32(1)
        work_tile_info = sched_consumer.consume_work()

        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            tid=tidx % (self._EpilogueWarpCnt * 32),
        )

        while work_tile_info.is_valid_tile:
            if cutlass.const_expr(self.overlapping_accum):
                tmem_stage_idx = acc_consumer_state.phase
            else:
                tmem_stage_idx = acc_consumer_state.index
            tmem_acc_current = tmem_acc[None, None, tmem_stage_idx]
            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                fc1_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                )
            else:
                fc2_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                )
            iket.range_pop()

            prev_work_tile_info = work_tile_info
            cur_was_linear1 = prev_work_tile_info.phase == cutlass.Int32(
                BlockPhase.Linear1
            )
            acc_consumer_state.advance()
            if cutlass.const_expr(self.overlapping_accum):
                is_odd_turn = cutlass.Int32(1) - is_odd_turn
            work_tile_info = sched_consumer.consume_work()

            if cur_was_linear1:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            wait_only_named_barrier.arrive_and_wait()
            if cur_was_linear1:
                flag_tracker = fc1_epi.signal_fc1_done(
                    prev_work_tile_info, work_tile_info, flag_tracker
                )
            else:
                flag_tracker = fc2_epi.signal_fc2_done(
                    prev_work_tile_info, work_tile_info, flag_tracker
                )
        flag_tracker.fire()


_STRUCTURAL_EPILOGUE_FIELDS = (
    "epi_smem_bytes",
    "overlapping_accum",
    "num_acc_pipeline_stages",
    "num_acc_stage",
    "overlapped_tmem_cols",
    "acc_sf_cols",
    "cta_tile_m",
    "cta_tile_n",
    "cta_tile_k",
    "subtile_cnt",
    "tmem_acc_layout_py_obj",
)


def _relu2_kernel_name(self: Sm100MegaMoEKernel) -> str:
    """Activation-bearing vendor cache name without subclass dispatch."""
    return f"{Sm100MegaMoEKernel.name(self)}_activation_relu2"


def _setup_relu2_attributes(self: Sm100MegaMoEKernel) -> None:
    """Run vendor setup, then replace only its structurally identical epilogue."""
    if self.gate_up_clamp is not None:
        raise ValueError("ReLU2 MegaMoE does not support gate_up_clamp.")

    # Call the vendor implementation lexically.  Calling the instance hook here
    # would recurse; ``super()`` is intentionally avoided because this object
    # must remain the exact vendor class for CuTeDSL's top-level trace.
    Sm100MegaMoEKernel._setup_attributes(self)
    swiglu_epilogue = self.epilogue
    relu2_epilogue = SwapABRelu2Fp4Epilogue(
        mma_tiler_mnk=self.mma_tiler,
        cluster_shape_mn=self.cluster_shape_mn,
        use_2cta_instrs=self.use_2cta_instrs,
        sf_vec_size=self.sf_vec_size,
        fc1_output_dtype=self.fc1_output_dtype,
        combine_format=self.combine_format,
        non_ubulk_fc2_store=self.non_ubulk_fc2_store,
        in_kernel_fc2_reduce=self.in_kernel_fc2_reduce,
        token_back_by_dispatch=self.token_back_by_dispatch,
        epi_flag_batch=self.epi_flag_batch,
        acc_dtype=self.acc_dtype,
        allow_overlap_acc=True,
        static_expert_shape=self.static_expert_shape,
        gate_up_clamp=None,
    )
    # The specialization changes math only.  These properties govern storage,
    # TMEM allocation, and pipeline staging already derived by vendor setup.
    for field in _STRUCTURAL_EPILOGUE_FIELDS:
        if getattr(relu2_epilogue, field) != getattr(swiglu_epilogue, field):
            raise RuntimeError(
                f"ReLU2 epilogue changed the physical kernel contract: {field} differs."
            )
    self.epilogue = relu2_epilogue


def configure_sm100_relu2_megamoe_kernel(
    kernel: Sm100MegaMoEKernel,
) -> Sm100MegaMoEKernel:
    """Attach ReLU2 host hooks while preserving the exact vendor class.

    ``cute.compile`` extracts ``kernel.__call__.__func__`` and traces it with
    ``kernel`` as ``self``.  The exact-type guard is therefore semantic, not a
    style preference: a subclass changes resolution of the vendor method's
    nested ``super().__call__`` and makes it call itself with ``fc1_output``.
    """
    if type(kernel) is not Sm100MegaMoEKernel:
        raise TypeError(
            "ReLU2 MegaMoE adapter requires an exact Sm100MegaMoEKernel "
            f"instance, got {type(kernel).__name__}."
        )
    if kernel.gate_up_clamp is not None:
        raise ValueError("ReLU2 MegaMoE does not support gate_up_clamp.")

    # These are host-only methods consulted during tracing / cache naming.
    # Special lookup of __call__ still comes from the exact vendor class.
    kernel._setup_attributes = MethodType(_setup_relu2_attributes, kernel)
    kernel.name = MethodType(_relu2_kernel_name, kernel)
    kernel._flashinfer_activation = "relu2"
    return kernel


def make_sm100_relu2_megamoe_kernel(**kwargs) -> Sm100MegaMoEKernel:
    """Construct an exact vendor kernel and opt it into the ReLU2 epilogue."""
    return configure_sm100_relu2_megamoe_kernel(Sm100MegaMoEKernel(**kwargs))


__all__ = [
    "SwapABRelu2Fc1Epilogue",
    "SwapABRelu2Fp4Epilogue",
    "configure_sm100_relu2_megamoe_kernel",
    "make_sm100_relu2_megamoe_kernel",
]
