# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ---------------------------------------------------------------------------
# Modifications Copyright (c) 2026 by FlashInfer team.
#
# This file is derived from the NVIDIA CUTLASS Python DSL example
# `examples/python/CuTeDSL/hopper/grouped_gemm.py` (BSD-3-Clause, retained
# above). FlashInfer adapts it into a Hopper (SM90) W4A8 MXFP4 grouped GEMM:
# the B (weight) operand is loaded as packed MXFP4 (FP4 e2m1, 2 nibbles/byte)
# with a per-32 UE8M0 block scale, dequantized to FP8 e4m3 in a transform
# stage, and consumed by the FP8 wgmma mainloop. The FP4 -> FP8 + E8M0
# converter logic is derived from the IEEE-style format specs.
# ---------------------------------------------------------------------------

import argparse
import os
from typing import List, Optional, Tuple, Type
from inspect import isclass
import math
import cuda.bindings.driver as cuda

import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math  # silu in the fused-SwiGLU epilogue
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.torch as cutlass_torch
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values
from cutlass.cute.core import (
    AddressSpace as _CuteAddressSpace,
    make_ptr as _cute_make_ptr,
)
from cutlass._mlir.dialects import nvvm as _nvvm_d
from cutlass._mlir.dialects._nvvm_enum_gen import (
    CpAsyncBulkTensorLoadMode as _CpAsyncBulkTensorLoadMode,
)
from cutlass.cutlass_dsl import dsl_user_op as _dsl_user_op, T as _T
from cutlass.cute.typing import Int32 as _Int32, Pointer as _Pointer

from ...api_logging import flashinfer_api


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# Debug switch: force `cute.copy` path for non-mcast loads.
_ENABLE_NVVM_NON_MCAST_LOAD = not _env_flag("GROUPED_GEMM_FORCE_CUTE_COPY", False)
# Experimental switch: enable true SMEM tensor map update/publish path in
# _FixedTensorMapManager for investigation.
_ENABLE_TRUE_SMEM_TMAP = _env_flag("GROUPED_GEMM_ENABLE_TRUE_SMEM_TMAP", False)
_ENABLE_TRUE_SMEM_TMAP_PREUPDATE = _env_flag(
    "GROUPED_GEMM_ENABLE_TRUE_SMEM_TMAP_PREUPDATE", True
)
_ENABLE_TRUE_SMEM_TMAP_PUBLISH = _env_flag(
    "GROUPED_GEMM_ENABLE_TRUE_SMEM_TMAP_PUBLISH", True
)

"""
Grouped GEMM (C_g = A_g * B_g for each group g) for the NVIDIA Hopper architecture
using CuTe DSL.

This kernel extends hopper/dense_gemm_persistent.py with per-group TMA tensor map updates
and a group-aware persistent tile scheduler (StaticPersistentGroupTileScheduler).

Key features:
    - WGMMA + TMA + persistent warp-specialized kernel (inherited from dense_gemm_persistent)
    - Per-group A/B/C TMA descriptor updates (tensor map) via GMEM or SMEM mode
    - DMA warp group: loads A/B tiles, updates tensor maps A/B on group boundary
    - MMA warp group: performs WGMMA, updates tensor map C on group boundary, stores C

To run:

.. code-block:: bash

    python hopper/grouped_gemm.py                                             \\
      --num_groups 4                                                           \\
      --problem_sizes_mnkl "(8192,1280,32,1),(16,384,1536,1),(640,1280,16,1),(640,160,16,1)" \\
      --tile_shape_mn 128,256 --cluster_shape_mn 1,1                          \\
      --a_dtype Float16 --b_dtype Float16 --c_dtype Float16 --acc_dtype Float32 \\
      --a_major k --b_major k --c_major n                                      \\
      --tensormap_update_mode SMEM

Constraints (same as dense_gemm_persistent.py plus):
* Only fp16/bf16 inputs are supported for grouped mode
* l (batch) must be 1 for each group
* CTA tile M: 64/128, N: 64/128/256
* Cluster shape M/N: power of 2, total <= 4
* Contiguous dim must be 16-byte aligned

Debug environment options:
* `GROUPED_GEMM_FORCE_CUTE_COPY=1`
    Disable the non-mcast NVVM TMA load path and always use `cute.copy`.
"""


@_dsl_user_op
def _tma_load_ab_nvvm_no_mcast(
    k_coord: _Int32,
    k_coord_b: _Int32,
    m_coord: _Int32,
    n_coord: _Int32,
    desc_a: _Pointer,
    desc_b: _Pointer,
    smem_a: _Pointer,
    smem_b: _Pointer,
    mbar: _Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue TMA A + TMA B loads via NVVM dialect ops for the non-mcast case.

    By passing the elect_sync predicate directly to the NVVM TMA op (instead of
    using scf.IfOp), all operands (k_coord, m_coord, desc_a, smem_a, mbar) are
    computed unconditionally at the MLIR/LLVM level.  PTXAS therefore emits any
    required R2UR conversions outside the predicated ELECT block, which is legal
    on sm_90a.  The scf.IfOp path, by contrast, causes PTXAS to sink the R2UR
    into the @P0-predicated elected-thread block, producing the illegal
    "@P0 R2UR" instruction (CUDA_ERROR_ILLEGAL_INSTRUCTION / error 715).
    """
    l_coord = _Int32(0).ir_value(loc=loc, ip=ip)
    # llvm_ptr is a @property on _Pointer — access without call syntax.
    smem_a_llvm = smem_a.llvm_ptr
    smem_b_llvm = smem_b.llvm_ptr
    mbar_llvm = mbar.llvm_ptr
    desc_a_llvm = desc_a.llvm_ptr
    desc_b_llvm = desc_b.llvm_ptr
    # TMA A: elect one thread and issue the load with predicate.
    is_elected_a = _nvvm_d.elect_sync(_T.bool(), loc=loc, ip=ip)
    _nvvm_d.CpAsyncBulkTensorGlobalToSharedClusterOp(
        dstMem=smem_a_llvm,
        tmaDescriptor=desc_a_llvm,
        coordinates=[
            k_coord.ir_value(loc=loc, ip=ip),
            m_coord.ir_value(loc=loc, ip=ip),
            l_coord,
        ],
        mbar=mbar_llvm,
        im2colOffsets=[],
        predicate=is_elected_a,
        loadMode=_CpAsyncBulkTensorLoadMode.TILE,
        loc=loc,
        ip=ip,
    )
    # TMA B: elect one thread and issue the load with predicate.
    is_elected_b = _nvvm_d.elect_sync(_T.bool(), loc=loc, ip=ip)
    _nvvm_d.CpAsyncBulkTensorGlobalToSharedClusterOp(
        dstMem=smem_b_llvm,
        tmaDescriptor=desc_b_llvm,
        coordinates=[
            k_coord_b.ir_value(loc=loc, ip=ip),
            n_coord.ir_value(loc=loc, ip=ip),
            l_coord,
        ],
        mbar=mbar_llvm,
        im2colOffsets=[],
        predicate=is_elected_b,
        loadMode=_CpAsyncBulkTensorLoadMode.TILE,
        loc=loc,
        ip=ip,
    )


def _tma_load_b_nvvm_no_mcast(
    k_coord_b: _Int32,
    n_coord: _Int32,
    desc_b: _Pointer,
    smem_b: _Pointer,
    mbar: _Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """B-only NVVM TMA load (fused-gather: A is cp.async-gathered in the MMA warp)."""
    l_coord = _Int32(0).ir_value(loc=loc, ip=ip)
    smem_b_llvm = smem_b.llvm_ptr
    mbar_llvm = mbar.llvm_ptr
    desc_b_llvm = desc_b.llvm_ptr
    is_elected_b = _nvvm_d.elect_sync(_T.bool(), loc=loc, ip=ip)
    _nvvm_d.CpAsyncBulkTensorGlobalToSharedClusterOp(
        dstMem=smem_b_llvm,
        tmaDescriptor=desc_b_llvm,
        coordinates=[
            k_coord_b.ir_value(loc=loc, ip=ip),
            n_coord.ir_value(loc=loc, ip=ip),
            l_coord,
        ],
        mbar=mbar_llvm,
        im2colOffsets=[],
        predicate=is_elected_b,
        loadMode=_CpAsyncBulkTensorLoadMode.TILE,
        loc=loc,
        ip=ip,
    )


class _GroupedWorkTileInfo:
    """Work tile info for grouped GEMM: carries is_valid_tile + group_search_result."""

    def __init__(self, is_valid_tile, group_search_result):
        self._is_valid_tile = is_valid_tile
        self.group_search_result = group_search_result

    @property
    def is_valid_tile(self):
        return self._is_valid_tile

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self._is_valid_tile)
        values.extend(extract_mlir_values(self.group_search_result))
        return values

    def __new_from_mlir_values__(self, values):
        n_valid = len(extract_mlir_values(self._is_valid_tile))
        is_valid = new_from_mlir_values(self._is_valid_tile, values[:n_valid])
        gsr = new_from_mlir_values(self.group_search_result, values[n_valid:])
        return _GroupedWorkTileInfo(is_valid, gsr)


class StaticPersistentGroupTileScheduler:
    """Grouped-GEMM-aware persistent tile scheduler.

    Wraps StaticPersistentTileScheduler + GroupedGemmTileSchedulerHelper.
    This class is not yet in cutlass.utils 4.3.5, so it is defined locally.
    """

    def __init__(self, tile_sched, group_helper, problem_sizes_mnkl):
        self._tile_sched = tile_sched
        self._group_helper = group_helper
        self._problem_sizes_mnkl = problem_sizes_mnkl

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self._tile_sched)
        values.extend(extract_mlir_values(self._group_helper))
        return values

    def __new_from_mlir_values__(self, values):
        n_tile = len(extract_mlir_values(self._tile_sched))
        tile_sched = new_from_mlir_values(self._tile_sched, values[:n_tile])
        group_helper = new_from_mlir_values(self._group_helper, values[n_tile:])
        return StaticPersistentGroupTileScheduler(
            tile_sched, group_helper, self._problem_sizes_mnkl
        )

    @staticmethod
    def create(
        tile_sched_params,
        bid,
        grid_dim,
        cluster_tile_shape_mnk,
        search_state,
        group_count,
        problem_sizes_mnkl,
    ):
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, bid, grid_dim
        )
        group_helper = utils.GroupedGemmTileSchedulerHelper(
            group_count, tile_sched_params, cluster_tile_shape_mnk, search_state
        )
        return StaticPersistentGroupTileScheduler(
            tile_sched, group_helper, problem_sizes_mnkl
        )

    @staticmethod
    def get_grid_shape(tile_sched_params, max_active_clusters):
        return utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

    def initial_work_tile_info(self):
        return self.get_current_work()

    def get_current_work(self):
        base = self._tile_sched.get_current_work()
        # For invalid tiles (linear_idx >= total_tiles), delinearize_z's inner
        # while-loop would infinite-loop.  Clamp the z coordinate to 0 by
        # multiplying with is_valid_tile (i1 zero-extended to i32).  z=0 is
        # always a valid tile index, so the group search terminates cleanly;
        # the resulting GroupSearchResult is discarded because the caller only
        # accesses it inside "while work_tile.is_valid_tile:".
        valid_int = base.is_valid_tile.to(cutlass.Int32)
        safe_tile_idx = (
            base.tile_idx[0],
            base.tile_idx[1],
            base.tile_idx[2] * valid_int,
        )
        gsr = self._group_helper.delinearize_z(safe_tile_idx, self._problem_sizes_mnkl)
        return _GroupedWorkTileInfo(base.is_valid_tile, gsr)

    def advance_to_next_work(self, *, advance_count=1):
        self._tile_sched.advance_to_next_work(advance_count=advance_count)

    @property
    def num_tiles_executed(self):
        return self._tile_sched.num_tiles_executed


class _FixedTensorMapManager(utils.TensorMapManager):
    """Local stability manager for environments using older cutlass.utils.

    By default, SMEM update/publish is routed through the GMEM branch for
    stability. Set GROUPED_GEMM_ENABLE_TRUE_SMEM_TMAP=1 to test the true SMEM
    path during investigation.
    """

    @_dsl_user_op
    @cute.jit
    def update_tensormap(
        self,
        tensor_gmem,
        tma_copy_atom,
        tensormap_gmem_ptr,
        warp_id: int,
        tensormap_smem_ptr,
        *,
        loc=None,
        ip=None,
    ) -> None:
        warp_idx = cute.arch.make_warp_uniform(
            cute.arch.warp_idx(loc=loc, ip=ip), loc=loc, ip=ip
        )
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            # Hoist SMEM pointer uniformization outside predicated blocks to avoid
            # predicated R2UR generation on sm_90a.
            uniform_smem_ptrs = tuple(
                _cute_make_ptr(
                    p.dtype,
                    cute.arch.make_warp_uniform(p.toint(), loc=loc, ip=ip),
                    mem_space=_CuteAddressSpace.smem,
                    assumed_align=p.alignment,
                )
                for p in tensormap_smem_ptr
            )
        else:
            uniform_smem_ptrs = tensormap_smem_ptr
        if warp_idx == warp_id:
            if cutlass.const_expr(
                self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
                and _ENABLE_TRUE_SMEM_TMAP
                and _ENABLE_TRUE_SMEM_TMAP_PREUPDATE
            ):
                for atom, tensor, sptr in zip(
                    tma_copy_atom, tensor_gmem, uniform_smem_ptrs, strict=False
                ):
                    cute.nvgpu.cpasync.update_tma_descriptor(
                        atom, tensor, sptr, loc=loc, ip=ip
                    )
            with cute.arch.elect_one(loc=loc, ip=ip):
                cute.arch.cp_async_bulk_commit_group(loc=loc, ip=ip)
                cute.arch.cp_async_bulk_wait_group(0, read=True, loc=loc, ip=ip)
            cute.arch.sync_warp(loc=loc, ip=ip)
            if cutlass.const_expr(
                self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
                and _ENABLE_TRUE_SMEM_TMAP
                and _ENABLE_TRUE_SMEM_TMAP_PUBLISH
            ):
                for gptr, sptr in zip(
                    tensormap_gmem_ptr, uniform_smem_ptrs, strict=False
                ):
                    cute.nvgpu.cpasync.cp_fence_tma_desc_release(
                        gptr, sptr, loc=loc, ip=ip
                    )
            else:
                for atom, tensor, gptr in zip(
                    tma_copy_atom, tensor_gmem, tensormap_gmem_ptr, strict=False
                ):
                    cute.nvgpu.cpasync.update_tma_descriptor(
                        atom, tensor, gptr, loc=loc, ip=ip
                    )
                cute.arch.sync_warp(loc=loc, ip=ip)
                cute.nvgpu.cpasync.fence_tma_desc_release(loc=loc, ip=ip)


class HopperGroupedGemmPersistentKernel:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Supported A/B data types:
        - Float16
          A and B must have the same data type
        - Float8E4M3FN/Float8E5M2
          A and B can have different types (Float8E4M3FN/Float8E5M2)
          only support k-major layout
        - Int8/Uint8
          A and B can have different types (Int8/Uint8)
          only support k-major layout

    :note: Supported accumulation types:
        - Float32/Float16 (for all floating point inputs)
        - Int32 (for Int8/Uint8 inputs)

    :note: Constraints:
        - CTA tile M must be 64/128
        - CTA tile N must be 64/128/256
        - CTA tile K must be 64
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperGroupedGemmPersistentKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mn=(1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    bytes_per_tensormap = 128
    num_tensormaps = 3  # A, B, C

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM,
        # W4A8 MoE behavior, passed explicitly by w4a8_mxfp4_grouped_gemm. Each is None
        # by default and then falls back to its FI_W4A8_* env flag (the CLI / benchmark
        # bring-up path); the public API passes booleans so no os.environ toggling is
        # needed. real_scale: read the per-(N,K/32) UE8M0 scale; token_scatter /
        # fused_scatter: scatter-add output rows to a shared MoE output; fused_gather /
        # real_gather: cp.async-gather A rows by a per-group route map.
        use_real_scale: bool = None,
        use_token_scatter: bool = None,
        use_fused_scatter: bool = None,
        use_fused_gather: bool = None,
        use_real_gather: bool = None,
        # When each output token is written by exactly one expert (top_k == 1, or any
        # disjoint routing), the token-scatter epilogue can use a plain store instead
        # of atomicAdd. ncu showed the FP32 atomicAdd is latency-bound (the persistent
        # kernel's low occupancy can't hide the RMW round-trip -> ~90% idle cycles), so
        # this is the single biggest fused-MoE win. Default None -> env fallback.
        scatter_no_accumulate: bool = None,
        # Fused SwiGLU epilogue (GEMM1): the weight columns are interleaved gate/up so
        # each (gate, up) pair lands in one thread's adjacent C registers; the epilogue
        # computes silu(gate)*up and writes it straight to FP8 [M, N/2] -- no [M, N] FP16
        # round-trip, no separate activation/requant kernel. c_dtype must be Float8E4M3FN.
        use_swiglu: bool = None,
        # Clamped SwiGLU (GPT-OSS / DeepSeek-V4), matching cutlass "SwiGLUBias"
        # (tests/moe/test_trtllm_cutlass_fused_moe.py:317): gate clamped to (-inf, limit],
        # up clamped to [-limit, limit] then + beta, silu uses sigmoid(alpha*gate). These
        # are model constants (uniform per call), baked in as compile-time consts;
        # swiglu_limit None => plain silu(gate)*up. alpha/beta default to 1.0/0.0.
        swiglu_alpha: float = None,
        swiglu_beta: float = None,
        swiglu_limit: float = None,
        # Dequant exponent re-centering. The FP4->FP8 dequant encodes
        # value = FP4 x 2^(scale-127) by ADDING the scale exponent to the FP8 exponent
        # bits -- exact in e4m3's NORMAL range, but real checkpoints carry small scales
        # (DSv4: ~2^-8..2^-4) that push products below 2^-6 into the subnormal band,
        # where exponent-add mis-encodes (1.5*2^-7 -> 0.5*2^-6) or underflows to 0.
        # Fix: the caller biases the UE8M0 scale bytes by +dequant_exp_bias (lifting
        # every product back into the normal range) and the epilogue multiplies the
        # FP32 accumulator by 2^-dequant_exp_bias (exact) before any epilogue math
        # (the SwiGLU clamp thresholds are in model units). 0 = off (bit-identical).
        dequant_exp_bias: int = 0,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype = acc_dtype

        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.mma_inst_shape_mn = None
        # K dimension is deferred in _setup_attributes
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        # For large tile size, using two warp groups is preferred because using only one warp
        # group may result in register spill
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = (
            self.num_dma_warp_groups * self.num_warps_per_warp_group
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        # Dedicated transform warps (perf restructure): move the FP4->FP8 transform
        # off the MMA warpgroup onto its own warpgroup, so the wgmma runs without
        # the per-k-tile transform_sync_barrier over all MMA threads (the structural
        # 8x loss). Defaults below = off (2 warpgroups). _setup_attributes() reads
        # use_transform (known only once b_dtype is set) and, when enabled, finalizes
        # the 3-warpgroup counts + barriers, overriding these defaults. The transform
        # warpgroup is placed AFTER the MMA warpgroups so the DMA/MMA warp ids and
        # tensormap barriers stay unchanged.
        self._xform_warps_requested = _env_flag("FI_W4A8_XFORM_WARPS", False)
        self.use_xform_warps = False
        self.num_transform_warp_groups = 0
        self.num_transform_threads = 0
        self.transform_warp_id = None

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage: Optional[type] = None
        self.buffer_align_bytes = 1024

        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )
        # barrier_id=3: sync all MMA threads between writing the FP4->FP8
        # transform buffer (sB_fp8) and the wgmma that consumes it.
        self.transform_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=self.num_mma_threads
        )

        # Grouped GEMM: tensor map update mode
        self.tensormap_update_mode = tensormap_update_mode
        # W4A8 MoE behavior overrides (None -> fall back to the FI_W4A8_* env flag in
        # __call__, where is_w4a8 is known). Set explicitly by the public API.
        self._opt_real_scale = use_real_scale
        self._opt_token_scatter = use_token_scatter
        self._opt_fused_scatter = use_fused_scatter
        self._opt_fused_gather = use_fused_gather
        self._opt_real_gather = use_real_gather
        self._opt_scatter_no_accumulate = scatter_no_accumulate
        self._opt_swiglu = use_swiglu
        # Clamped-SwiGLU constants (uniform per call). None limit => plain silu(gate)*up;
        # a float limit selects the cutlass SwiGLUBias formula in the fused epilogue.
        self.swiglu_alpha = 1.0 if swiglu_alpha is None else float(swiglu_alpha)
        self.swiglu_beta = 0.0 if swiglu_beta is None else float(swiglu_beta)
        self.swiglu_clamp = None if swiglu_limit is None else float(swiglu_limit)
        # Dequant exponent re-centering (see constructor docstring): the epilogue
        # multiplies the accumulator by 2^-bias; the caller pre-biased the scales.
        self.dequant_exp_bias = int(dequant_exp_bias)
        # Delegate A/B tensor map init to MMA warp for better latency hiding (SMEM mode)
        self.delegate_tensormap_ab_init = (
            tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        )
        # barrier_id=2 (barrier_id=1 is already used by epilog_sync_barrier)
        # Only the load warp (32 threads) + all MMA threads participate:
        # DMA warps 1-3 are idle and never reach this barrier.
        self.tensormap_ab_init_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_threads + 32,
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        # check the cta tile shape
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")

        # For W4A8 the wgmma runs in FP8 for both operands: B is loaded as
        # packed FP4 and dequantized to FP8 e4m3 in the transform stage, so the
        # MMA (and the wgmma-side B smem layout) use FP8, not the FP4 load type.
        self.mma_b_dtype = cutlass.Float8E4M3FN if self.is_w4a8 else self.b_dtype
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.mma_b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Cluster tile shape used by group tile scheduler
        self.cluster_tile_shape_mnk = (
            self.tile_shape_mnk[0] * self.cluster_shape_mn[0],
            self.tile_shape_mnk[1] * self.cluster_shape_mn[1],
            self.tile_shape_mnk[2],
        )

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            transform_dtype=self.mma_b_dtype if self.use_transform else None,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

        # When the FP4 -> FP8 transform is active, the wgmma B operand (sB_fp8)
        # uses an FP8 smem layout, distinct from sB's (possibly FP4) load layout.
        # Reuse _make_smem_layouts with mma_b_dtype to build it (a/c discarded).
        self.b_fp8_smem_layout_staged = self.b_smem_layout_staged
        # Single-stage FP8 B layout (same swizzle, no stage dim). A fresh tensor with
        # this layout has flat (n,k) coords -- the staged slice's coords are
        # hierarchical and not linearizable. Used by the prmt+real-scale path to map
        # each output group to (n, k) for the per-block scale lookup (validated in
        # tests/moe/spike_realscale_coord.py).
        self.b_fp8_smem_layout_single = self.b_smem_layout_staged
        if cutlass.const_expr(self.use_transform):
            (_, self.b_fp8_smem_layout_staged, _) = self._make_smem_layouts(
                self.tile_shape_mnk,
                self.epi_tile,
                self.a_dtype,
                self.a_layout,
                self.mma_b_dtype,
                self.b_layout,
                self.ab_stage,
                self.c_dtype,
                self.c_layout,
                self.epi_stage,
            )
            _b_k_major = (
                self.b_layout.sm90_mma_major_mode()
                == cute.nvgpu.warpgroup.OperandMajorMode.K
            )
            _b_fp8_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(
                    self.b_layout,
                    self.mma_b_dtype,
                    self.tile_shape_mnk[2 if _b_k_major else 1],
                ),
                self.mma_b_dtype,
            )
            self.b_fp8_smem_layout_single = cute.tile_to_shape(
                _b_fp8_atom,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                order=(0, 1) if _b_k_major else (1, 0),
            )

        # W4A8: SM90 has no native FP4 TMA, so load B as packed Uint8 [.., bK//2]
        # (1 byte = 2 nibbles). sB holds raw bytes; the transform expands them to
        # FP8 in sB_fp8. Rebuild b_smem_layout_staged as Uint8 with a half-K tile.
        self.b_load_dtype = self.b_dtype
        if cutlass.const_expr(self.is_w4a8):
            self.b_load_dtype = cutlass.Uint8
            b_load_tile = (
                self.tile_shape_mnk[0],
                self.tile_shape_mnk[1],
                self.tile_shape_mnk[2] // 2,
            )
            (_, self.b_smem_layout_staged, _) = self._make_smem_layouts(
                b_load_tile,
                self.epi_tile,
                self.a_dtype,
                self.a_layout,
                cutlass.Uint8,
                self.b_layout,
                self.ab_stage,
                self.c_dtype,
                self.c_layout,
                self.epi_stage,
            )

        # Finalize the dedicated-transform-warps layout now that use_transform is
        # known. Only enabled for the single-MMA-warpgroup config: with 2 MMA
        # warpgroups (large tiles) adding a 3rd warpgroup would blow the per-CTA
        # register budget. The transform warpgroup consumes the load pipeline
        # (wait-only -- it never releases it; the trans2mma pipeline guarantees the
        # transform finishes reading sB before the MMA frees the stage) and produces
        # sB_fp8 into a new trans2mma PipelineAsync the MMA warpgroup consumes.
        self.use_xform_warps = (
            self._xform_warps_requested
            and self.use_transform
            and self.num_mma_warp_groups == 1
        )
        if self.use_xform_warps:
            self.num_transform_warp_groups = 1
            self.num_transform_threads = (
                self.num_transform_warp_groups * self.num_threads_per_warp_group
            )
            self.threads_per_cta = (
                self.num_dma_warp_groups
                + self.num_mma_warp_groups
                + self.num_transform_warp_groups
            ) * self.num_threads_per_warp_group
            # Transform warps come last (after DMA + MMA): first transform warp id.
            self.transform_warp_id = (
                self.num_dma_warp_groups + self.num_mma_warp_groups
            ) * self.num_warps_per_warp_group
            # The transform_sync_barrier now orders the sB_fp8 smem write vs the
            # trans2mma signal among the TRANSFORM warps only -- not all MMA
            # threads. Removing that all-MMA barrier from the wgmma path is the fix.
            self.transform_sync_barrier = pipeline.NamedBarrier(
                barrier_id=3, num_threads=self.num_transform_threads
            )

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        total_num_clusters: cutlass.Int32,
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        """Execute the grouped GEMM operation.

        :param initial_a: Carries dtype+majorness only (shape irrelevant).
        :param initial_b: Carries dtype+majorness only (shape irrelevant).
        :param initial_c: Carries dtype+majorness only (shape irrelevant).
        :param group_count: Number of GEMM groups (compile-time constant).
        :param problem_shape_mnkl: Device tensor of shape (G, 4) Int32 with (M,N,K,L) per group.
        :param strides_abc: Device tensor of shape (G, 3, 2) Int32 with strides per group.
        :param tensor_address_abc: Device tensor of shape (G, 3) Int64 with base ptrs per group.
        :param total_num_clusters: Total clusters across all groups (RUNTIME value: the
            per-call routing changes it every MoE forward in serving, so baking it as a
            Constexpr would force a recompile per routing; the persistent scheduler only
            compares it against tile indices on device). The launch grid is static
            (max_active_clusters); surplus CTAs find no valid tile and exit.
        :param tensormap_cute_tensor: Tensor map workspace, shape (num_sms, 3, 16) Int64.
        :param max_active_clusters: Max active clusters (compile-time constant).
        :param stream: CUDA stream.
        """

        # Setup static attributes from initial tensor dtype/layout
        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.c_dtype = initial_c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(initial_a)
        self.b_layout = utils.LayoutEnum.from_tensor(initial_b)
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)

        # W4A8 MXFP4: B is packed FP4 (e2m1), dequantized to FP8 e4m3 in a
        # transform stage before the FP8 wgmma. A and B therefore differ in
        # input width here, so the homogeneous-width checks below do not apply.
        self.is_w4a8 = self.b_dtype == cutlass.Float4E2M1FN
        # The FP4 -> FP8 transform stage is active for W4A8. It can also be
        # forced on with FP8 B (identity copy) to validate the transform
        # plumbing independently of the dequant -- see Step X bring-up.
        self.use_transform = self.is_w4a8 or _env_flag("FI_W4A8_FORCE_TRANSFORM", False)
        # Each W4A8 MoE behavior is taken from the explicit constructor override when
        # given (the public API), else from its FI_W4A8_* env flag (CLI / bring-up).
        # Read the real per-(N, K/32) UE8M0 block scale (4th metadata operand, gmem)
        # instead of the deterministic/constant bring-up scale.
        self.use_real_scale = self.is_w4a8 and (
            _env_flag("FI_W4A8_REAL_SCALE", False)
            if self._opt_real_scale is None
            else self._opt_real_scale
        )
        # Fused MoE scatter (FS-0): raw-address store of the accumulator to a plain C
        # tensor (built from the group's raw C pointer), via an identity coordinate
        # tensor -- not the TMA-store staging. FS-1 (token scatter): each output row
        # goes to a token position via a per-group route map, scaled by a per-group
        # routing weight, into a shared MoE output. Implies the fused-scatter epilogue.
        self.use_token_scatter = (
            _env_flag("FI_W4A8_TOKEN_SCATTER", False)
            if self._opt_token_scatter is None
            else self._opt_token_scatter
        )
        # Plain store instead of atomicAdd in the token-scatter epilogue (safe when each
        # token is written once -- top_k == 1 / disjoint routing). Removes the
        # latency-bound RMW the profile flagged as the #1 fused-MoE cost.
        self.scatter_no_accumulate = (
            _env_flag("FI_W4A8_NO_ACCUMULATE", False)
            if self._opt_scatter_no_accumulate is None
            else self._opt_scatter_no_accumulate
        )
        # Fused SwiGLU epilogue (GEMM1): silu(gate)*up -> FP8 [M, N/2], gate/up
        # interleaved in the weight columns. Uses the raw-store epilogue path.
        self.use_swiglu = (
            _env_flag("FI_W4A8_SWIGLU", False)
            if self._opt_swiglu is None
            else self._opt_swiglu
        )
        # A/B knob: force the old per-element scalar atomicAdd in the accumulating
        # token-scatter epilogue instead of the vectorized v2.f32 atomic (default).
        self.use_scalar_scatter = _env_flag("FI_W4A8_SCALAR_SCATTER", False)
        self.use_fused_scatter = (
            _env_flag("FI_W4A8_FUSED_SCATTER", False)
            if self._opt_fused_scatter is None
            else self._opt_fused_scatter
        ) or self.use_token_scatter
        # Fused MoE gather: load A rows with cp.async (LDGSTS) indexed by a per-group
        # route map, replacing A's TMA load (Hopper has no TMA-gather). The MMA
        # warpgroup gathers A's tile into the swizzled sA just-in-time (no separate
        # cp.async pipeline); B keeps its TMA.
        self.use_fused_gather = (
            _env_flag("FI_W4A8_FUSED_GATHER", False)
            if self._opt_fused_gather is None
            else self._opt_fused_gather
        )
        # Real per-token gather: A base (operand 0) is the shared activation tensor and
        # a per-group route map (operand 6) maps each local row -> source row. Without
        # it, fused gather is the identity (contiguous) bring-up == TMA load.
        self.use_real_gather = self.use_fused_gather and (
            _env_flag("FI_W4A8_REAL_GATHER", False)
            if self._opt_real_gather is None
            else self._opt_real_gather
        )
        # Vectorized FP4->FP8 transform (#5): replace the scalar per-byte loop +
        # 1-byte swizzled smem stores with a vectorized, swizzle-aware register
        # round-trip (make_cotiled_copy + autovec_copy, validated in
        # tests/moe/spike_vectorized_xform_full.py). DEFAULT ON for W4A8 (2.16x over
        # scalar); FI_W4A8_SCALAR_XFORM=1 forces the scalar fallback. The vec path
        # handles both constant and (with prmt) real per-block scale.
        self.use_vec_transform = self.is_w4a8 and not _env_flag(
            "FI_W4A8_SCALAR_XFORM", False
        )
        # prmt-LUT FP4->FP8 inside the vectorized transform: replace the per-nibble
        # branchless arithmetic with a prmt byte-LUT (~2 prmt + a few ops per 8
        # nibbles, validated in tests/moe/spike_prmt_lut_fp4_fp8.py). The per-block
        # scale folds into a per-group LUT cached per output copy (compile-time %2
        # unroll). DEFAULT ON with the vec path (3.3x over vec-arith); FI_W4A8_NO_PRMT=1
        # falls back to the arithmetic dequant.
        self.use_prmt = self.use_vec_transform and not _env_flag(
            "FI_W4A8_NO_PRMT", False
        )
        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(
            not self.is_w4a8 and self.a_dtype.width != self.b_dtype.width
        ):
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
            )
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16, float8, or int8")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            initial_a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )

        # W4A8: load B as packed Uint8 [N, K/2] via a byte view (no FP4 TMA).
        if cutlass.const_expr(self.is_w4a8):
            b_tma_input = cute.recast_tensor(initial_b, cutlass.Uint8)
            b_tma_tile = (self.tile_shape_mnk[1], self.tile_shape_mnk[2] // 2)
        else:
            b_tma_input = initial_b
            b_tma_tile = (self.tile_shape_mnk[1], self.tile_shape_mnk[2])
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b_tma_input,
            self.b_smem_layout_staged,
            b_tma_tile,
            self.cluster_shape_mn[0],
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            initial_c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            total_num_clusters,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        # Number of Int64 words needed for the SMEM tensor map buffer (0 in GMEM mode)
        self.size_tensormap_in_i64 = (
            0
            if self.tensormap_update_mode == utils.TensorMapUpdateMode.GMEM
            else HopperGroupedGemmPersistentKernel.num_tensormaps
            * HopperGroupedGemmPersistentKernel.bytes_per_tensormap
            // 8
        )

        # sB_fp8 buffer size (plain Python int; precomputed to avoid a NESTED ternary
        # inside the @cute.struct body -- the DSL AST ifExp transformer mis-scopes the
        # inner then-block, "name 'ifexp_then_block_N' is not defined"). Route A's
        # transform needs the staged buffer; homogeneous-precision needs none.
        if cutlass.const_expr(self.use_transform):
            _sB_fp8_cosize = cute.cosize(self.b_fp8_smem_layout_staged)
        else:
            _sB_fp8_cosize = 0

        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[
                cutlass.Int64, self.size_tensormap_in_i64
            ]
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            # Transform -> MMA pipeline mbarriers (full+empty per stage). Sized 0
            # unless the dedicated transform warps are active.
            trans2mma_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, (self.ab_stage * 2) if self.use_xform_warps else 0
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_load_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # FP8 transform buffer: B is dequantized FP4 -> FP8 into this STAGED buffer,
            # which the wgmma then consumes. Sized 0 on the homogeneous-precision path so
            # no extra smem is used.
            sB_fp8: cute.struct.Align[
                cute.struct.MemRange[self.mma_b_dtype, _sB_fp8_cosize],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.b_fp8_smem_layout_staged,
            self.b_fp8_smem_layout_single,
            self.epi_smem_layout_staged,
            tile_sched_params,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            tensormap_cute_tensor,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

    @cute.jit
    def _transform_b_tile(
        self, sB, sB_fp8, stage_index, tidx, k_tile, real_scale, n_offset, sB_fp8_single
    ):
        """Fill the FP8 transform buffer sB_fp8 for the current pipeline stage.

        For W4A8 (B is packed MXFP4 / FP4 e2m1), dequantize each FP4 nibble to an
        FP8 e4m3 byte with a branchless bit conversion (E8M0 block scale is not
        applied yet on this scalar bring-up path -- unit scale). For the
        homogeneous-precision bring-up path (FP8 B) it is an identity cast.

        All MMA threads cooperate; a barrier + async-shared proxy fence order the
        smem writes before the wgmma that consumes sB_fp8.
        """
        # Cooperative thread base/count: the dedicated transform warpgroup (placed
        # after DMA+MMA) when active, else the MMA warpgroup (inline path).
        if cutlass.const_expr(self.use_xform_warps):
            mma_tid = (
                tidx
                - (self.num_dma_warp_groups + self.num_mma_warp_groups)
                * self.num_threads_per_warp_group
            )
            n_xform_threads = self.num_transform_threads
        else:
            mma_tid = tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            n_xform_threads = self.num_mma_threads
        if cutlass.const_expr(self.is_w4a8):
            # sB is the packed MXFP4 weight loaded as raw Uint8 bytes [bN, bK/2]
            # (SM90 has no FP4 TMA); sB_fp8 is the FP8 output [bN, bK]. Expand
            # each byte's two nibbles to FP8 e4m3 (spike-validated logic).
            src_u8 = sB[None, None, stage_index]
            dst = sB_fp8[None, None, stage_index]  # FP8 tensor, swizzle intact
            bN = cute.size(src_u8, mode=[0])
            bKh = cute.size(src_u8, mode=[1])
            # MXFP4 UE8M0 block scale applied as an integer add on the e4m3
            # exponent field (matches csrc fp8_apply_exp_offset). Currently a
            # single constant exponent for bring-up; per-32-block scale data
            # plumbing (4th TMA operand) is the next step. _se=0 -> no scale.
            _se_const = int(os.getenv("FI_W4A8_SCALE_EXP", "0"))
            # Deterministic per-32-block scale (validates per-block application
            # without scale-data plumbing): _se(block) = (block % 4) - 1.
            _use_block = os.getenv("FI_W4A8_BLOCK_SCALE") is not None
            _blocks_per_tile = (bKh * 2) // 32
            # The vectorized path handles constant scale (folded into the LUT) and,
            # with prmt, real per-block scale (folded into a per-group LUT using flat
            # (n,k) coords from a fresh single-stage cotiled -- see the prmt branch).
            # The per-block-scale debug path (_use_block) stays scalar.
            if cutlass.const_expr(
                self.use_vec_transform
                and not _use_block
                and (self.use_prmt or not self.use_real_scale)
            ):
                # Vectorized swizzle-aware FP4->FP8 (constant scale _se_const).
                # cotiled-copy + autovec_copy round-trip (validated in
                # tests/moe/spike_vectorized_xform_full.py): each thread loads VI
                # packed bytes from the swizzled sB into registers, expands each
                # byte to 2 FP8 (lo nibble -> 2*i, hi -> 2*i+1), and stores VO=2*VI
                # FP8 to the swizzled sB_fp8 -- the input/output cotiled copies tile
                # K consistently so out FP8 at K=2j,2j+1 come from in byte at K=j.
                VI = 8  # packed bytes per vectorized input copy (64-bit)
                VO = 16  # FP8 per vectorized output copy (128-bit = swizzle granule)
                in_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.Uint8,
                    num_bits_per_copy=VI * 8,
                )
                in_thr = cute.make_cotiled_copy(
                    in_atom,
                    cute.make_layout((n_xform_threads, VI), stride=(VI, 1)),
                    src_u8.layout,
                ).get_slice(mma_tid)
                out_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.mma_b_dtype,
                    num_bits_per_copy=VO * 8,
                )
                out_thr = cute.make_cotiled_copy(
                    out_atom,
                    cute.make_layout((n_xform_threads, VO), stride=(VO, 1)),
                    dst.layout,
                ).get_slice(mma_tid)
                src_part = in_thr.partition_S(src_u8)
                frag_in = cute.make_rmem_tensor(src_part.shape, cutlass.Uint8)
                cute.autovec_copy(src_part, frag_in)
                dst_part = out_thr.partition_D(dst)
                frag_out = cute.make_rmem_tensor(dst_part.shape, self.mma_b_dtype)
                # Distinct variable names from the scalar branch below: the DSL
                # treats a name assigned in both branches as one loop-carried var
                # and rejects the None->Int32 type change even though only one
                # const_expr branch is traced.
                if cutlass.const_expr(self.use_prmt):
                    # prmt byte-LUT (validated in spike_prmt_lut_fp4_fp8.py): the 8
                    # magnitude FP8 bytes packed into two i32; prmt.b32 selects 4 mag
                    # bytes from the 4 magnitude nibbles; sign (nibble bit3 -> FP8
                    # bit7) is OR'd. Same frag_out order as the arithmetic path:
                    # byte i -> frag_out[2i]=lo, [2i+1]=hi.
                    #
                    # The per-block scale exponent is folded into a per-group
                    # magnitude LUT (no per-element scale ALU -> the prmt win survives;
                    # a prmt-group = 4 packed bytes = 8 K, 8-aligned -> one 32-K block
                    # -> one scale). Constant scale folds at trace; real scale reads
                    # real_scale[n, k//32] using the group's flat (n,k) from a fresh
                    # single-stage cotiled (the staged dst coord is hierarchical;
                    # validated in spike_realscale_coord.py).
                    # Per-block scale folded into a per-group magnitude LUT. (A
                    # loop-carried LUT cache keyed on (n,block) was tried but is a net
                    # loss here: the loop is a dynamic scf.for, so the cache needs a
                    # runtime block-change `if` whose warp divergence + iter-arg cost
                    # exceeds the ~28-op rebuild it saves -- the cotiled (n,k) order has
                    # poor block locality. Rebuilding every group is faster.)
                    if cutlass.const_expr(self.use_real_scale):
                        ocoords = (
                            cute.make_cotiled_copy(
                                out_atom,
                                cute.make_layout((n_xform_threads, VO), stride=(VO, 1)),
                                sB_fp8_single.layout,
                            )
                            .get_slice(mma_tid)
                            .partition_S(cute.make_identity_tensor((bN, 2 * bKh)))
                        )
                    # PYTHON-unrolled loop (compile-time bound) so the scaled LUT can
                    # be cached per output copy: a copy is VO=16 contiguous K = 2
                    # prmt-groups (vg even/odd) that share one (n, 32-K block) and so
                    # one scale exponent -> rebuild the LUT (and read real_scale) only
                    # on even vg, reuse on odd vg (`_llo`/`_lhi` persist across unrolled
                    # iterations -- no runtime branch). Halves the LUT rebuild + scale
                    # read. A dynamic scf.for (range_constexpr over a dynamic size)
                    # would not let the python vars persist, so the bound is computed
                    # as a python int from the tile shape.
                    # Group count as a PYTHON int (range_constexpr only unrolls a
                    # compile-time bound; tile_shape_mnk[2] is a DSL value via
                    # cute.size, so derive the K tile from the wgmma K-instruction:
                    # 32 for 8-bit, 16 for 16-bit, x4 tiles; bKh = K//2).
                    _bKh_py = (32 if self.mma_b_dtype.width == 8 else 16) * 4 // 2
                    _grp_count = (self.tile_shape_mnk[1] * _bKh_py) // (
                        n_xform_threads * 4
                    )
                    for vg in cutlass.range_constexpr(_grp_count):
                        # Rebuild per output copy (2 groups = 16 K, one (n, block));
                        # %4 (2 copies) is NOT correct -- they cross n/block boundaries.
                        if cutlass.const_expr(vg % 2 == 0):
                            if cutlass.const_expr(self.use_real_scale):
                                _oc = ocoords[8 * vg]
                                _pblk = k_tile * _blocks_per_tile + (_oc[1] // 32)
                                _psng = cutlass.min(
                                    n_offset + _oc[0],
                                    cute.size(real_scale, mode=[0]) - 1,
                                )
                                _se_g = (
                                    real_scale[(_psng, _pblk, 0)].to(cutlass.Int32)
                                    & 0xFF
                                ) - W4A8_SCALE_BASE
                            else:
                                _se_g = cutlass.Int32(_se_const)
                            # scaled magnitude LUT (branchless): code 0 -> 0 via mc_nz;
                            # codes 1..7 exponent+_se_g clamped to e4m3 [0,15].
                            _lb = []
                            for _m in cutlass.range_constexpr(8):
                                _em = (_m >> 1) & 3
                                _Mm = ((_m & 1) << 2) * ((_em + 3) >> 2)
                                _mcnz = ((_m & 7) + 7) >> 3
                                _Eg = cutlass.max(cutlass.min(_em + 6 + _se_g, 15), 0)
                                _lb.append((((_Eg << 3) | _Mm) * _mcnz))
                            _llo = (
                                _lb[0] | (_lb[1] << 8) | (_lb[2] << 16) | (_lb[3] << 24)
                            )
                            _lhi = (
                                _lb[4] | (_lb[5] << 8) | (_lb[6] << 16) | (_lb[7] << 24)
                            )
                        pb0 = frag_in[4 * vg + 0].to(cutlass.Int32) & 0xFF
                        pb1 = frag_in[4 * vg + 1].to(cutlass.Int32) & 0xFF
                        pb2 = frag_in[4 * vg + 2].to(cutlass.Int32) & 0xFF
                        pb3 = frag_in[4 * vg + 3].to(cutlass.Int32) & 0xFF
                        c0 = (
                            (pb0 & 7)
                            | (((pb0 >> 4) & 7) << 4)
                            | ((pb1 & 7) << 8)
                            | (((pb1 >> 4) & 7) << 12)
                        )
                        w0 = cutlass.Int32(cute.arch.prmt(_llo, _lhi, c0)) | (
                            ((pb0 & 8) << 4)
                            | ((pb0 & 0x80) << 8)
                            | ((pb1 & 8) << 20)
                            | ((pb1 & 0x80) << 24)
                        )
                        c1 = (
                            (pb2 & 7)
                            | (((pb2 >> 4) & 7) << 4)
                            | ((pb3 & 7) << 8)
                            | (((pb3 >> 4) & 7) << 12)
                        )
                        w1 = cutlass.Int32(cute.arch.prmt(_llo, _lhi, c1)) | (
                            ((pb2 & 8) << 4)
                            | ((pb2 & 0x80) << 8)
                            | ((pb3 & 8) << 20)
                            | ((pb3 & 0x80) << 24)
                        )
                        for vj in cutlass.range_constexpr(4):
                            frag_out[8 * vg + vj] = (
                                ((w0 >> (8 * vj)) & 0xFF)
                                .to(cutlass.Uint8)
                                .bitcast(cutlass.Float8E4M3FN)
                            )
                            frag_out[8 * vg + 4 + vj] = (
                                ((w1 >> (8 * vj)) & 0xFF)
                                .to(cutlass.Uint8)
                                .bitcast(cutlass.Float8E4M3FN)
                            )
                else:
                    for vi in cutlass.range_constexpr(cute.size(frag_in)):
                        vb = frag_in[vi].to(cutlass.Int32)
                        vlo = vb & 0xF
                        vlo_e = (vlo >> 1) & 3
                        vlo_E = cutlass.max(cutlass.min(vlo_e + 6 + _se_const, 15), 0)
                        vlo_mag = (vlo_E << 3) | (((vlo & 1) << 2) * ((vlo_e + 3) >> 2))
                        vlo_fp8 = ((vlo & 8) << 4) | (vlo_mag * (((vlo & 7) + 7) >> 3))
                        frag_out[2 * vi] = vlo_fp8.to(cutlass.Uint8).bitcast(
                            cutlass.Float8E4M3FN
                        )
                        vhi = (vb >> 4) & 0xF
                        vhi_e = (vhi >> 1) & 3
                        vhi_E = cutlass.max(cutlass.min(vhi_e + 6 + _se_const, 15), 0)
                        vhi_mag = (vhi_E << 3) | (((vhi & 1) << 2) * ((vhi_e + 3) >> 2))
                        vhi_fp8 = ((vhi & 8) << 4) | (vhi_mag * (((vhi & 7) + 7) >> 3))
                        frag_out[2 * vi + 1] = vhi_fp8.to(cutlass.Uint8).bitcast(
                            cutlass.Float8E4M3FN
                        )
                cute.autovec_copy(frag_out, dst_part)
            else:
                for bi in cutlass.range(mma_tid, bN * bKh, n_xform_threads):
                    crd = cute.idx2crd(bi, (bN, bKh))
                    byte = src_u8[crd].to(cutlass.Int32)
                    n0 = crd[0]
                    k0 = 2 * crd[1]
                    if cutlass.const_expr(self.use_real_scale):
                        # Real per-(N, K/32) UE8M0 scale: 2^(byte - base). Mask to
                        # 0xFF: the Uint8 read sign-extends, so bytes >=128 must be
                        # forced unsigned (UE8M0 spans the full 0..255 byte range).
                        # Clamp the row index: a CTA tile may overhang N (padding
                        # rows), masked in the epilogue.
                        _blk = k_tile * _blocks_per_tile + (k0 // 32)
                        _sn = cutlass.min(
                            n_offset + n0, cute.size(real_scale, mode=[0]) - 1
                        )
                        _se = (
                            real_scale[(_sn, _blk, 0)].to(cutlass.Int32) & 0xFF
                        ) - W4A8_SCALE_BASE
                    elif cutlass.const_expr(_use_block):
                        _se = ((k_tile * _blocks_per_tile + (k0 // 32)) & 3) - 1
                    else:
                        _se = _se_const
                    # low nibble -> FP8 e4m3 at column 2j (write the byte's bit
                    # pattern directly as FP8 to keep sB_fp8's wgmma swizzle).
                    # E = e + 6 + scale, clamped to e4m3 finite [0,15] so a large
                    # block scale can't overflow into the sign bit.
                    lo = byte & 0xF
                    lo_e = (lo >> 1) & 3
                    lo_E = cutlass.max(cutlass.min(lo_e + 6 + _se, 15), 0)
                    lo_mag = (lo_E << 3) | (((lo & 1) << 2) * ((lo_e + 3) >> 2))
                    lo_fp8 = ((lo & 8) << 4) | (lo_mag * (((lo & 7) + 7) >> 3))
                    dst[(n0, k0)] = lo_fp8.to(cutlass.Uint8).bitcast(
                        cutlass.Float8E4M3FN
                    )
                    # high nibble -> FP8 e4m3 at column 2j+1
                    hi = (byte >> 4) & 0xF
                    hi_e = (hi >> 1) & 3
                    hi_E = cutlass.max(cutlass.min(hi_e + 6 + _se, 15), 0)
                    hi_mag = (hi_E << 3) | (((hi & 1) << 2) * ((hi_e + 3) >> 2))
                    hi_fp8 = ((hi & 8) << 4) | (hi_mag * (((hi & 7) + 7) >> 3))
                    dst[(n0, k0 + 1)] = hi_fp8.to(cutlass.Uint8).bitcast(
                        cutlass.Float8E4M3FN
                    )
        else:
            src = sB[None, None, stage_index]
            dst = sB_fp8[None, None, stage_index]
            for i in cutlass.range(mma_tid, cute.size(src), n_xform_threads):
                dst[i] = src[i].to(self.mma_b_dtype)
        self.transform_sync_barrier.arrive_and_wait()
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def _gather_a_finish(self):
        """Drain the deferred cp.async gather: wait for the async copies to land in
        sA, sync the MMA threads, and fence so the wgmma sees the gathered rows."""
        cute.arch.cp_async_wait_group(0)
        self.transform_sync_barrier.arrive_and_wait()
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def _gather_a_tile(
        self,
        sA,
        stage_index,
        k_tile,
        a_gmem_ptr,
        route_map,
        m_tile_base,
        m_bound,
        k_dim,
        tidx,
        do_finish=True,
    ):
        """cp.async-gather the A tile for this stage into the wgmma-swizzled sA.

        Replaces A's TMA load (Hopper has no TMA-gather). Each MMA thread owns one
        row; partition_D(sA) is swizzle-aware and the per-thread source base is
        shifted so this thread's row partition lands on the gathered row -- the
        pattern validated in tests/moe/derisk_cutedsl_cpasync_gather_swizzle.py.
        route_map is None for the identity (contiguous) bring-up that must equal the
        TMA load; a real route map gathers per token. Assumes num_mma_threads == bM
        (one MMA warpgroup, the (128,128) tile).
        """
        bM = self.tile_shape_mnk[0]
        bK = self.tile_shape_mnk[2]
        mma_tid = tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
        sA_tile = sA[None, None, stage_index]  # (bM, bK) swizzled
        a_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.a_dtype, num_bits_per_copy=128
        )
        thr_layout = cute.make_layout((bM, 1), stride=(1, 0))
        val_layout = cute.make_layout((1, bK), stride=(0, 1))
        tiled_copy = cute.make_tiled_copy_tv(a_atom, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(mma_tid)
        tAsA = thr_copy.partition_D(sA_tile)
        local_row = m_tile_base + mma_tid
        if cutlass.const_expr(route_map is None):
            global_row = local_row
        else:
            global_row = route_map[cutlass.min(local_row, m_bound - 1)].to(
                cutlass.Int32
            )
        k_base = k_tile * bK
        # Per-thread shifted source: shift A's base so thread mma_tid's row-partition
        # lands on (global_row, k_base). delta is a multiple of 16 elements (k_dim
        # and bK are), so the cp.async.128 source stays 16-byte aligned.
        delta = cute.assume((global_row - mma_tid) * k_dim + k_base, divby=16)
        # The row stride is dynamic K; assert it is 16-aligned (K is a multiple of
        # 32) so partition_S's per-thread base (mma_tid * k_dim) stays 16-byte
        # aligned for the cp.async.128 source.
        k_dim_a = cute.assume(k_dim, divby=16)
        mA_t = cute.make_tensor(
            a_gmem_ptr + delta, cute.make_layout((bM, bK), stride=(k_dim_a, 1))
        )
        tAgA = thr_copy.partition_S(mA_t)
        cute.copy(tiled_copy, tAgA, tAsA)
        cute.arch.cp_async_commit_group()
        # When do_finish=False the caller defers the wait/barrier/fence past the
        # FP4->FP8 weight transform so the transform compute overlaps the cp.async
        # gather latency (ncu showed the immediate wait was ~56% smem-scoreboard stall
        # the persistent kernel's low occupancy could not hide). _gather_a_finish()
        # then drains it before the wgmma.
        if cutlass.const_expr(do_finish):
            self._gather_a_finish()

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        b_fp8_smem_layout_staged: cute.ComposedLayout,
        b_fp8_smem_layout_single: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr[int],
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        tensormaps: cute.Tensor,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param tile_sched_params: Parameters for the persistent tile scheduler
        :type tile_sched_params: utils.PersistentTileSchedulerParams
        """

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch Tma desc
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        # B is loaded with b_load_dtype (Uint8 for W4A8), so its TMA byte count
        # must use that, not b_dtype (FP4) -- otherwise the pipeline's expected
        # transaction bytes mismatch the actual TMA and consumer_wait deadlocks.
        # Fused gather: A is cp.async-gathered (not TMA), so the mainloop pipeline's
        # tx-count tracks B only, else consumer_wait waits for A bytes that never
        # arrive via TMA and deadlocks.
        tma_copy_bytes = cute.size_in_bytes(self.b_load_dtype, b_smem_layout)
        if cutlass.const_expr(not self.use_fused_gather):
            tma_copy_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)

        # Alloc and init AB full/empty + ACC full mbar (pipeline)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count with the number of mcast size
        # (num_mcast_ctas_a/b are set to ints in _setup_attributes before launch; mypy
        # only sees the None initializer).
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1  # type: ignore[operator]
        consumer_arrive_cnt = (
            mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )

        # Transform -> MMA pipeline (dedicated transform warps). Producer = the
        # transform warpgroup (writes sB_fp8), consumer = the MMA warpgroup (wgmma
        # reads sB_fp8). AsyncThread both sides; the mbarrier handoff replaces the
        # per-k-tile transform_sync_barrier on the wgmma path. defer_sync=True so it
        # shares the single mbarrier-init fence + cluster arrive below.
        if cutlass.const_expr(self.use_xform_warps):
            trans2mma_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.trans2mma_pipeline_array_ptr.data_ptr(),
                num_stages=self.ab_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_transform_threads
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_mma_threads
                ),
                defer_sync=True,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Generate smem tensor A/B
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # FP8 transform buffer (same staged layout as sB for Step X bring-up).
        # When the transform is active, B is dequantized FP4 -> FP8 into this
        # buffer and the wgmma consumes it instead of the raw sB.
        if cutlass.const_expr(self.use_transform):
            sB_fp8 = storage.sB_fp8.get_tensor(
                b_fp8_smem_layout_staged.outer,
                swizzle=b_fp8_smem_layout_staged.inner,
            )
            # Fresh single-stage alias (flat (n,k) coords) of sB_fp8 for the
            # prmt+real-scale per-group scale lookup. Aliases stage 0; the coord
            # mapping is stage-independent (validated in spike_realscale_coord.py).
            sB_fp8_single = storage.sB_fp8.get_tensor(
                b_fp8_smem_layout_single.outer,
                swizzle=b_fp8_smem_layout_single.inner,
            )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # Local_tile partition global tensors
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, RestN, RestK, RestL)
        # W4A8: B is loaded as packed Uint8 with a half-K tile (bN, bK//2).
        b_tile_shape = self.tile_shape_mnk
        if cutlass.const_expr(self.is_w4a8):
            b_tile_shape = (
                self.tile_shape_mnk[0],
                self.tile_shape_mnk[1],
                self.tile_shape_mnk[2] // 2,
            )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(b_tile_shape, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition shared tensor for TMA load A/B
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # Partition global tensor for TiledMMA_A/B/C
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(
            mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
        )

        # Make fragments
        tCsA = thr_mma.partition_A(sA)
        # The wgmma B operand comes from the FP8 transform buffer when active,
        # otherwise directly from sB (homogeneous-precision path).
        if cutlass.const_expr(self.use_transform):
            tCsB = thr_mma.partition_B(sB_fp8)
        else:
            tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # Cluster wait for barrier init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Setup per-SM tensor map pointers (shared by DMA and MMA warps)
        #
        grid_dim = cute.arch.grid_dim()
        bid = cute.arch.block_idx()
        sm_idx = bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]

        tensormap_manager = _FixedTensorMapManager(
            self.tensormap_update_mode,
            HopperGroupedGemmPersistentKernel.bytes_per_tensormap,
        )
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(sm_idx, 0, None)].iterator
        )
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(sm_idx, 1, None)].iterator
        )
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(sm_idx, 2, None)].iterator
        )

        # SMEM buffer pointers for tensor maps (only non-None in SMEM mode)
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            smem_tm_base = storage.tensormap_buffer.data_ptr()
            tensormap_a_smem_ptr = smem_tm_base
            tensormap_b_smem_ptr = (
                smem_tm_base
                + HopperGroupedGemmPersistentKernel.bytes_per_tensormap // 8
            )
            tensormap_c_smem_ptr = (
                smem_tm_base
                + 2 * HopperGroupedGemmPersistentKernel.bytes_per_tensormap // 8
            )
        else:
            tensormap_a_smem_ptr = None
            tensormap_b_smem_ptr = None
            tensormap_c_smem_ptr = None

        tile_sched_params_for_sched = tile_sched_params

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        # Dedicated transform warpgroup (placed after DMA + MMA). It consumes the
        # load pipeline (wait-only) and produces the trans2mma pipeline; it keeps the
        # launch-default register count (no setmaxnreg), so DMA's freed registers
        # cover the MMA warpgroup's increase.
        is_transform_warp_group = (
            self.use_xform_warps
            and warp_group_idx >= self.num_dma_warp_groups + self.num_mma_warp_groups
        )
        if is_dma_warp_group:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)

        #
        # DMA warp group (load A/B with TMA, update tensor maps A/B per group)
        #
        if warp_idx == self.load_warp_id:
            # Initialize tensor maps A/B (either here or delegated to MMA warp)
            if cutlass.const_expr(not self.delegate_tensormap_ab_init):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_ptr, self.load_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_ptr, self.load_warp_id
                )
                tensormap_manager.fence_tensormap_initialization()
            else:
                # Delegate path: wait for MMA warp to finish A/B tensor map init.
                # Must be unconditional (before the tile loop) so every CTA
                # participates even when it processes zero tiles.
                self.tensormap_ab_init_barrier.arrive_and_wait()

            last_group_idx = cutlass.Int32(-1)

            # Create a per-warp scheduler (same state — each warp runs its own instance)
            tile_sched = StaticPersistentGroupTileScheduler.create(
                tile_sched_params_for_sched,
                bid,
                grid_dim,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
                group_count,
                problem_sizes_mnkl,
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )

            while work_tile.is_valid_tile:
                grouped_info = work_tile.group_search_result
                cur_group_idx = grouped_info.group_idx
                cur_k_tile_cnt = grouped_info.cta_tile_count_k

                if cur_k_tile_cnt != 0:
                    is_group_changed = cur_group_idx != last_group_idx

                    if is_group_changed:
                        real_a = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.a_dtype,
                            (
                                grouped_info.problem_shape_m,
                                grouped_info.problem_shape_n,
                                grouped_info.problem_shape_k,
                            ),
                            strides_abc,
                            ptrs_abc,
                            0,
                        )
                        # W4A8: B is loaded as packed Uint8 [N, K/2], so its
                        # tensormap descriptor uses b_load_dtype and half the K.
                        real_b = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.b_load_dtype,
                            (
                                grouped_info.problem_shape_m,
                                grouped_info.problem_shape_n,
                                grouped_info.problem_shape_k // 2
                                if cutlass.const_expr(self.is_w4a8)
                                else grouped_info.problem_shape_k,
                            ),
                            strides_abc,
                            ptrs_abc,
                            1,
                        )
                        tensormap_manager.update_tensormap(
                            (real_a, real_b),
                            (tma_atom_a, tma_atom_b),
                            (tensormap_a_ptr, tensormap_b_ptr),
                            self.load_warp_id,
                            (tensormap_a_smem_ptr, tensormap_b_smem_ptr),
                        )
                        tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_b_ptr)

                    mma_tile_coord_mnl = (
                        grouped_info.cta_tile_idx_m,
                        grouped_info.cta_tile_idx_n,
                        0,
                    )
                    tAgA_slice = tAgA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    tBgB_slice = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]

                    # Cache loop-invariant TMA descriptor pointers before K-loop.
                    # Keep two variants:
                    # - gmem descriptors for direct NVVM cp.async.bulk.tensor ops
                    # - generic descriptors for cute.copy fallback (mcast path)
                    #
                    # Using explicit gmem descriptors in the direct NVVM path avoids
                    # relying on generic-pointer lowering for the descriptor operand.
                    tma_a_desc_ptr_nvvm = tensormap_manager.get_tensormap_ptr(
                        tensormap_a_ptr, cute.AddressSpace.gmem
                    )
                    tma_b_desc_ptr_nvvm = tensormap_manager.get_tensormap_ptr(
                        tensormap_b_ptr, cute.AddressSpace.gmem
                    )
                    tma_a_desc_ptr_copy = tensormap_manager.get_tensormap_ptr(
                        tensormap_a_ptr, cute.AddressSpace.generic
                    )
                    tma_b_desc_ptr_copy = tensormap_manager.get_tensormap_ptr(
                        tensormap_b_ptr, cute.AddressSpace.generic
                    )
                    # Pre-compute loop-invariant TMA coordinates (m, n).
                    # For the non-mcast case (cluster 1x1), the TMA box offset is
                    # simply cta_tile_idx * tile_size.  k_coord is computed inside
                    # the loop because it varies per K-tile.
                    _tile_k = self.tile_shape_mnk[2]
                    _tile_m = self.tile_shape_mnk[0]
                    _tile_n = self.tile_shape_mnk[1]
                    use_nvvm_non_mcast_load = cutlass.const_expr(
                        _ENABLE_NVVM_NON_MCAST_LOAD
                        and not self.is_a_mcast
                        and not self.is_b_mcast
                    )
                    mainloop_producer_state.reset_count()
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        if use_nvvm_non_mcast_load:
                            # Non-mcast path: use NVVM dialect TMA op with
                            # predicate= so operands are computed outside any
                            # predicated block.  This prevents PTXAS from
                            # generating the illegal @P0 R2UR instruction on
                            # sm_90a (CUDA_ERROR_ILLEGAL_INSTRUCTION / 715).
                            if cutlass.const_expr(self.use_fused_gather):
                                # Fused gather: B-only TMA; A is cp.async-gathered
                                # in the MMA warp (Hopper has no TMA-gather).
                                _tma_load_b_nvvm_no_mcast(
                                    k_tile * (_tile_k // 2)
                                    if cutlass.const_expr(self.is_w4a8)
                                    else k_tile * _tile_k,
                                    mma_tile_coord_mnl[1] * _tile_n,
                                    tma_b_desc_ptr_nvvm,
                                    tBsB[
                                        (None, mainloop_producer_state.index)
                                    ].iterator,
                                    mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state
                                    ),
                                )
                            else:
                                _tma_load_ab_nvvm_no_mcast(
                                    k_tile * _tile_k,
                                    # W4A8: B is packed Uint8 [N, K/2], so its TMA
                                    # k-coordinate advances by half the GEMM K tile.
                                    k_tile * (_tile_k // 2)
                                    if cutlass.const_expr(self.is_w4a8)
                                    else k_tile * _tile_k,
                                    mma_tile_coord_mnl[0] * _tile_m,
                                    mma_tile_coord_mnl[1] * _tile_n,
                                    tma_a_desc_ptr_nvvm,
                                    tma_b_desc_ptr_nvvm,
                                    tAsA[
                                        (None, mainloop_producer_state.index)
                                    ].iterator,
                                    tBsB[
                                        (None, mainloop_producer_state.index)
                                    ].iterator,
                                    mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state
                                    ),
                                )
                        else:
                            # Mcast path: fall back to cute.copy which handles
                            # the multicast mask for multi-CTA clusters.
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, k_tile)],
                                tAsA[(None, mainloop_producer_state.index)],
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=a_mcast_mask,
                                tma_desc_ptr=tma_a_desc_ptr_copy,
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, k_tile)],
                                tBsB[(None, mainloop_producer_state.index)],
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=b_mcast_mask,
                                tma_desc_ptr=tma_b_desc_ptr_copy,
                            )
                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()
                else:
                    pass  # k_tile_cnt == 0: tensor map init already done before loop

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        #
        # MMA warp group (WGMMA + epilogue, update tensor map C per group)
        #
        if not is_dma_warp_group and not is_transform_warp_group:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)

            # MMA warp always initializes tensor map C
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c, tensormap_c_ptr, self.epi_store_warp_id
            )
            # When delegating, MMA warp also initializes A/B and signals DMA warp
            if cutlass.const_expr(self.delegate_tensormap_ab_init):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_ptr, self.epi_store_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_ptr, self.epi_store_warp_id
                )
                self.tensormap_ab_init_barrier.arrive_and_wait()

            tensormap_manager.fence_tensormap_initialization()

            tile_sched = StaticPersistentGroupTileScheduler.create(
                tile_sched_params_for_sched,
                bid,
                grid_dim,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
                group_count,
                problem_sizes_mnkl,
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            # trans2mma consumer states (dedicated transform warps): sB_fp8 is
            # produced by the transform warpgroup. These advance in lockstep with
            # the mainloop states (same ab_stage), so their .index coincides and the
            # existing tCrB[..., mainloop index] reads the right sB_fp8 stage.
            if cutlass.const_expr(self.use_xform_warps):
                trans2mma_consumer_read_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                trans2mma_consumer_release_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Partition for epilogue
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )

            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.c_layout.is_m_major_c(),
                    4,
                ),
                self.c_dtype,
            )

            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s,
                tiled_copy_C_Atom,
            )

            # (R2S, R2S_M, R2S_N, PIPE_D)
            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            )
            # (t)hread-partition for (r)egister to (s)mem copy (tRS_)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            # (R2S, R2S_M, R2S_N)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            # Allocate D registers.
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            # Number of in-flight wgmma groups before wait_group. Deeper pipelining
            # can hide the FP4->FP8 transform latency behind the async wgmma; tunable
            # for the perf study (default 1 = the original scaffold).
            k_pipe_mmas = int(os.getenv("FI_W4A8_KPIPE", "1"))

            # Initialize tma store pipeline
            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_threads,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            last_group_idx_mma = cutlass.Int32(-1)

            while work_tile.is_valid_tile:
                grouped_info = work_tile.group_search_result
                cur_group_idx = grouped_info.group_idx
                cur_k_tile_cnt = grouped_info.cta_tile_count_k

                # Per-group tensor map C update (only epi_store warp issues it)
                is_group_changed = cur_group_idx != last_group_idx_mma
                if is_group_changed and warp_idx == self.epi_store_warp_id:
                    real_c = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        (
                            grouped_info.problem_shape_m,
                            grouped_info.problem_shape_n,
                            grouped_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        2,
                    )
                    tensormap_manager.update_tensormap(
                        (real_c,),
                        (tma_atom_c,),
                        (tensormap_c_ptr,),
                        self.epi_store_warp_id,
                        (tensormap_c_smem_ptr,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_c_ptr)

                mma_tile_coord_mnl = (
                    grouped_info.cta_tile_idx_m,
                    grouped_info.cta_tile_idx_n,
                    0,
                )
                gC_mnl_slice = gC_mnl[(None, None, *mma_tile_coord_mnl)]

                # W4A8 real block scale: build the per-group UE8M0 scale gmem
                # tensor [N, K/32] (4th metadata operand) for the transform to
                # read; n_offset maps this CTA's local rows to global N.
                if cutlass.const_expr(self.use_real_scale):
                    scale_gptr = cute.make_ptr(
                        cutlass.Uint8,
                        ptrs_abc[(cur_group_idx, 3)],
                        cute.AddressSpace.gmem,
                        assumed_align=1,
                    )
                    real_scale = cute.make_tensor(
                        scale_gptr,
                        cute.make_layout(
                            (
                                grouped_info.problem_shape_n,
                                grouped_info.problem_shape_k // 32,
                                cutlass.Int32(1),
                            ),
                            stride=(
                                grouped_info.problem_shape_k // 32,
                                1,
                                cutlass.Int32(0),
                            ),
                        ),
                    )
                    n_offset = grouped_info.cta_tile_idx_n * self.tile_shape_mnk[1]
                else:
                    real_scale = None
                    n_offset = cutlass.Int32(0)

                # Fused gather: per-group A base + tile/shape info for the MMA-warp
                # cp.async gather (identity route for now -> must equal the TMA load).
                if cutlass.const_expr(self.use_fused_gather):
                    gather_a_ptr = cute.make_ptr(
                        self.a_dtype,
                        ptrs_abc[(cur_group_idx, 0)],
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    gather_m_base = mma_tile_coord_mnl[0] * self.tile_shape_mnk[0]
                    gather_m_bound = grouped_info.problem_shape_m
                    gather_k_dim = grouped_info.problem_shape_k
                    if cutlass.const_expr(self.use_real_gather):
                        # Per-group route map [M] int32 (operand 6): local row ->
                        # source row in the shared activation tensor (operand 0).
                        gather_route = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Int32,
                                ptrs_abc[(cur_group_idx, 6)],
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout((gather_m_bound,), stride=(1,)),
                        )
                    else:
                        gather_route = None

                # MAINLOOP
                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()
                if cutlass.const_expr(self.use_xform_warps):
                    trans2mma_consumer_read_state.reset_count()
                    trans2mma_consumer_release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()

                prologue_mma_cnt = cutlass.min(k_pipe_mmas, cur_k_tile_cnt)

                for k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                    # Wait for TMA copies to complete (B only when fused gather)
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    # Wait for the transform warpgroup to fill sB_fp8 for this stage.
                    if cutlass.const_expr(self.use_xform_warps):
                        trans2mma_pipeline.consumer_wait(trans2mma_consumer_read_state)
                    # Fused gather: cp.async-gather A's tile into the swizzled sA.
                    # Defer the wait/barrier/fence past the transform (below) so the
                    # transform compute overlaps the gather latency.
                    _gather_defer = self.use_transform and not self.use_xform_warps
                    if cutlass.const_expr(self.use_fused_gather):
                        self._gather_a_tile(
                            sA,
                            mainloop_consumer_read_state.index,
                            k_tile,
                            gather_a_ptr,
                            gather_route,
                            gather_m_base,
                            gather_m_bound,
                            gather_k_dim,
                            tidx,
                            do_finish=not _gather_defer,
                        )
                    # FP4 -> FP8 transform into sB_fp8 (Step X: identity copy).
                    # With dedicated transform warps the transform warpgroup does
                    # this; the MMA warpgroup only waits on the trans2mma pipeline.
                    if cutlass.const_expr(
                        self.use_transform and not self.use_xform_warps
                    ):
                        self._transform_b_tile(
                            sB,
                            sB_fp8,
                            mainloop_consumer_read_state.index,
                            tidx,
                            k_tile,
                            real_scale,
                            n_offset,
                            sB_fp8_single,
                        )
                        if cutlass.const_expr(self.use_fused_gather):
                            self._gather_a_finish()
                    # WGMMA
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )

                    cute.nvgpu.warpgroup.commit_group()
                    mainloop_consumer_read_state.advance()
                    if cutlass.const_expr(self.use_xform_warps):
                        trans2mma_consumer_read_state.advance()

                for k_tile in cutlass.range(
                    prologue_mma_cnt, cur_k_tile_cnt, 1, unroll=1
                ):
                    # Wait for TMA copies to complete (B only when fused gather)
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    # Wait for the transform warpgroup to fill sB_fp8 for this stage.
                    if cutlass.const_expr(self.use_xform_warps):
                        trans2mma_pipeline.consumer_wait(trans2mma_consumer_read_state)
                    # Fused gather: cp.async-gather A's tile into the swizzled sA.
                    # Deferred wait/barrier/fence (overlaps the transform below).
                    _gather_defer = self.use_transform and not self.use_xform_warps
                    if cutlass.const_expr(self.use_fused_gather):
                        self._gather_a_tile(
                            sA,
                            mainloop_consumer_read_state.index,
                            k_tile,
                            gather_a_ptr,
                            gather_route,
                            gather_m_base,
                            gather_m_bound,
                            gather_k_dim,
                            tidx,
                            do_finish=not _gather_defer,
                        )
                    # FP4 -> FP8 transform into sB_fp8 (Step X: identity copy).
                    # With dedicated transform warps the transform warpgroup does
                    # this; the MMA warpgroup only waits on the trans2mma pipeline.
                    if cutlass.const_expr(
                        self.use_transform and not self.use_xform_warps
                    ):
                        self._transform_b_tile(
                            sB,
                            sB_fp8,
                            mainloop_consumer_read_state.index,
                            tidx,
                            k_tile,
                            real_scale,
                            n_offset,
                            sB_fp8_single,
                        )
                        if cutlass.const_expr(self.use_fused_gather):
                            self._gather_a_finish()
                    # WGMMA
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )

                    cute.nvgpu.warpgroup.commit_group()
                    # Wait on the wgmma barrier for WGMMA to complete
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()
                    if cutlass.const_expr(self.use_xform_warps):
                        trans2mma_pipeline.consumer_release(
                            trans2mma_consumer_release_state
                        )
                        trans2mma_consumer_release_state.advance()
                        trans2mma_consumer_read_state.advance()

                cute.nvgpu.warpgroup.wait_group(0)
                for _k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    if cutlass.const_expr(self.use_xform_warps):
                        trans2mma_pipeline.consumer_release(
                            trans2mma_consumer_release_state
                        )
                        trans2mma_consumer_release_state.advance()

                # Dequant exponent re-centering: the caller biased the UE8M0 weight
                # scales by +dequant_exp_bias so every FP4*scale product stays in FP8
                # e4m3's NORMAL range (real DSv4 scales otherwise land subnormal, where
                # the exponent-add dequant mis-encodes or underflows). Undo it here on
                # the FP32 accumulator (exact), BEFORE any epilogue math -- the SwiGLU
                # clamp thresholds and routing weights are in model units. One spot
                # covers every epilogue branch (raw-store / SwiGLU / scatter / TMA
                # staging all read these registers).
                if cutlass.const_expr(self.dequant_exp_bias != 0):
                    _descale = 2.0 ** (-self.dequant_exp_bias)
                    for _dq_i in cutlass.range_constexpr(cute.size(accumulators)):
                        _dq_c = cute.idx2crd(_dq_i, accumulators.shape)
                        accumulators[_dq_c] = accumulators[_dq_c] * _descale

                # Epilogue
                if cutlass.const_expr(self.use_fused_scatter or self.use_swiglu):
                    # FS-0 raw-address store (validated; gated off by default).
                    # Writes each accumulator element to a plain C tensor (built from
                    # the group's raw C pointer) at the global (m, n) from an identity
                    # coordinate tensor, bypassing the sC + TMA-store staging and the
                    # coordinate-tagged gC partitions (which cannot be a non-TMA store
                    # dest). Equivalent to the staged store for full tiles; verified
                    # single- and multi-group. NOTE: requires M a multiple of the
                    # M-tile (no partial-tile predication yet). This is the store
                    # foundation -- real token scatter (out -> shared MoE output, row
                    # -> token via the route map, atomicAdd x routing weight) and
                    # partial-tile predication build on top.
                    c_gptr = cute.make_ptr(
                        self.c_dtype,
                        ptrs_abc[(cur_group_idx, 2)],
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    out_c = cute.make_tensor(
                        c_gptr,
                        cute.make_layout(
                            (
                                grouped_info.problem_shape_m,
                                grouped_info.problem_shape_n,
                            ),
                            stride=(grouped_info.problem_shape_n, 1),
                        ),
                    )
                    cC = cute.make_identity_tensor(gC_mnl_slice.shape)
                    # Slice the MMA by the actual per-thread lane (not the warpgroup
                    # base, as thr_mma is) so partition_C gives each thread's own
                    # fragment coords -- aligned with how that thread's accumulator
                    # registers map to C. Slicing by warpgroup gave all 128 threads
                    # identical coords -> they overwrote the same cells.
                    mma_tidx = (
                        tidx
                        - self.num_dma_warp_groups * self.num_threads_per_warp_group
                    )
                    tTR_cC = tiled_mma.get_slice(mma_tidx).partition_C(cC)
                    m_base = mma_tile_coord_mnl[0] * self.tile_shape_mnk[0]
                    n_base = mma_tile_coord_mnl[1] * self.tile_shape_mnk[1]
                    # Shift the output origin to this tile's (m_base, n_base) so the
                    # local tile coordinate from the identity tensor indexes gmem
                    # directly (Blackwell-finalize domain_offset idiom). Index the
                    # accumulator and coord tensor by the same logical fragment
                    # coordinate to keep value <-> (m, n) aligned.
                    out_tile = cute.domain_offset((m_base, n_base), out_c)
                    m_bound = grouped_info.problem_shape_m
                    if cutlass.const_expr(self.use_swiglu):
                        # Fused SwiGLU epilogue. The weight columns are interleaved
                        # gate/up, so each WGMMA register PAIR (slots _i,_i+1 = adjacent
                        # N cols c (even = gate) and c+1 (= up), same row) is one thread's
                        # (gate, up) pair (same fact the vectorized scatter uses). Compute
                        # silu(gate)*up = gate*sigmoid(gate)*up and write it straight to
                        # the FP8 output [M, N/2] at column (n_base + c)/2 -- no [M, N]
                        # FP16 round-trip, no separate activation/requant kernel.
                        n_dim = grouped_info.problem_shape_n  # = 2*I (the GEMM N)
                        out_n = n_dim // 2  # = I (output cols)
                        out_ptr = cute.make_ptr(
                            self.c_dtype,  # Float8E4M3FN
                            ptrs_abc[(cur_group_idx, 2)],
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        # Pass 1: silu(gate)*up for each (gate,up) register pair into an
                        # FP32 fragment. Pass 2: ONE vectorized FP32->FP8 convert (a
                        # scalar f32->fp8 cvt is illegal -- it needs a 1-d vector). Pass
                        # 3: store the FP8 bytes to the (non-contiguous, stride-4) output
                        # columns -- a scalar FP8 *store* is fine, only the convert isn't.
                        _np = cute.size(accumulators) // 2
                        _o32 = cute.make_rmem_tensor((_np,), self.acc_dtype)
                        for _p in cutlass.range_constexpr(_np):
                            _gate = accumulators[
                                cute.idx2crd(2 * _p, accumulators.shape)
                            ]
                            _up = accumulators[
                                cute.idx2crd(2 * _p + 1, accumulators.shape)
                            ]
                            if cutlass.const_expr(self.swiglu_clamp is not None):
                                # Clamped SwiGLU (cutlass SwiGLUBias): clamp gate to
                                # (-inf, L], clamp up to [-L, L] then + beta, and scale the
                                # sigmoid argument by alpha. min/max act on the dynamic FP32
                                # register against the const limit (cutlass.min/max).
                                _L = self.swiglu_clamp
                                _g = cutlass.min(_gate, _L)
                                _u = (
                                    cutlass.max(cutlass.min(_up, _L), -_L)
                                    + self.swiglu_beta
                                )
                                _sig = 1.0 / (
                                    1.0
                                    + cute_math.exp(
                                        -self.swiglu_alpha * _g, fastmath=True
                                    )
                                )
                                _o32[_p] = _g * _sig * _u
                            else:
                                _sig = 1.0 / (
                                    1.0 + cute_math.exp(-_gate, fastmath=True)
                                )
                                _o32[_p] = _gate * _sig * _up
                        _o8 = cute.make_rmem_tensor((_np,), self.c_dtype)
                        _o8.store(_o32.load().to(self.c_dtype))
                        for _p in cutlass.range_constexpr(_np):
                            _crd = tTR_cC[cute.idx2crd(2 * _p, accumulators.shape)]
                            _gm = m_base + _crd[0]
                            if _gm < m_bound:
                                _off = _gm * out_n + (n_base + _crd[1]) // 2
                                _d = cute.make_tensor(
                                    out_ptr + _off,
                                    cute.make_layout((1,), stride=(1,)),
                                )
                                _d[0] = _o8[_p]
                    elif cutlass.const_expr(self.use_token_scatter):
                        # FS-1 token scatter: each output row scatter-ADDS to a token
                        # position via the per-group route map, scaled by the per-row
                        # routing weight, into the shared FP32 MoE output. Metadata
                        # cols: 2 = shared FP32 output base, 4 = route map [M] int32
                        # (local row -> output/token row), 5 = weights [M] f32. The
                        # FP32 atomicAdd lets top_k>1 experts accumulate into the same
                        # token row (caller zeroes the output). Output addressed flat
                        # as out_row*N + n.
                        n_dim = grouped_info.problem_shape_n
                        route_map = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Int32,
                                ptrs_abc[(cur_group_idx, 4)],
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout((m_bound,), stride=(1,)),
                        )
                        weights = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float32,
                                ptrs_abc[(cur_group_idx, 5)],
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout((m_bound,), stride=(1,)),
                        )
                        out_ptr = cute.make_ptr(
                            cutlass.Float32,
                            ptrs_abc[(cur_group_idx, 2)],
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        if cutlass.const_expr(self.scatter_no_accumulate):
                            # Vectorized token scatter (top_k==1 / disjoint output
                            # rows). WGMMA's FP32 C fragment holds 2 contiguous N
                            # columns per register pair (same token row), and idx2crd
                            # walks the innermost N-pair first, so consecutive
                            # accumulator slots (_i, _i+1) map to (out_row, n..n+1) --
                            # one v2.f32 store. route_map[_gm]/weights[_gm] are constant
                            # across a row's N, so they're loaded once per pair instead
                            # of per element (the per-element gmem index reload was the
                            # bulk of the scatter's residual cost after Fix #1). Pairs
                            # start on an even N (the WGMMA lane col base 2*(lane%4) is
                            # even) so the FP32 dest is 8B-aligned for the wide store.
                            _pair = cute.make_rmem_tensor((2,), self.acc_dtype)
                            for _i in cutlass.range_constexpr(
                                0, cute.size(accumulators), 2
                            ):
                                _mc0 = cute.idx2crd(_i, accumulators.shape)
                                _mc1 = cute.idx2crd(_i + 1, accumulators.shape)
                                _crd = tTR_cC[_mc0]
                                _ml = _crd[0]
                                _nl = _crd[1]
                                _gm = m_base + _ml
                                if _gm < m_bound:
                                    _w = weights[_gm]
                                    _off = route_map[_gm] * n_dim + n_base + _nl
                                    _pair[0] = _w * accumulators[_mc0]
                                    _pair[1] = _w * accumulators[_mc1]
                                    _d = cute.make_tensor(
                                        out_ptr + _off,
                                        cute.make_layout((2,), stride=(1,)),
                                    )
                                    cute.autovec_copy(_pair, _d)
                        elif cutlass.const_expr(not self.use_scalar_scatter):
                            # Vectorized accumulating scatter (top_k>=2): same per-pair
                            # route_map/weights hoist + 8B-aligned v2 layout as the plain
                            # store, but a v2.f32 atomic add (red.global.add.v2.f32, valid
                            # on sm_90 -- atomic_add takes a vector value) -- one RMW per
                            # pair sharing the index load, vs the old scalar atomicAdd per
                            # element (latency-bound, the profile's #1 fused-MoE cost).
                            # FI_W4A8_SCALAR_SCATTER forces the scalar path below (A/B).
                            _pair = cute.make_rmem_tensor((2,), self.acc_dtype)
                            for _i in cutlass.range_constexpr(
                                0, cute.size(accumulators), 2
                            ):
                                _mc0 = cute.idx2crd(_i, accumulators.shape)
                                _mc1 = cute.idx2crd(_i + 1, accumulators.shape)
                                _crd = tTR_cC[_mc0]
                                _gm = m_base + _crd[0]
                                if _gm < m_bound:
                                    _w = weights[_gm]
                                    _off = route_map[_gm] * n_dim + n_base + _crd[1]
                                    _pair[0] = _w * accumulators[_mc0]
                                    _pair[1] = _w * accumulators[_mc1]
                                    cute.arch.atomic_add(
                                        out_ptr + _off,
                                        _pair.load(),
                                        sem="relaxed",
                                        scope="gpu",
                                    )
                        else:
                            # Scalar path: the FI_W4A8_SCALAR_SCATTER A/B baseline
                            # (per-element atomicAdd).
                            for _i in cutlass.range_constexpr(cute.size(accumulators)):
                                _mc = cute.idx2crd(_i, accumulators.shape)
                                _crd = tTR_cC[_mc]
                                _ml = _crd[0]
                                _nl = _crd[1]
                                _gm = m_base + _ml
                                if _gm < m_bound:
                                    _off = route_map[_gm] * n_dim + n_base + _nl
                                    _val = weights[_gm] * accumulators[_mc]
                                    if cutlass.const_expr(self.scatter_no_accumulate):
                                        _d = cute.make_tensor(
                                            out_ptr + _off,
                                            cute.make_layout((1,), stride=(1,)),
                                        )
                                        _d[0] = _val
                                    else:
                                        cute.arch.atomic_add(
                                            out_ptr + _off,
                                            _val,
                                            sem="relaxed",
                                            scope="gpu",
                                        )
                    else:
                        for _i in cutlass.range_constexpr(cute.size(accumulators)):
                            _mc = cute.idx2crd(_i, accumulators.shape)
                            _crd = tTR_cC[_mc]
                            _ml = _crd[0]
                            _nl = _crd[1]
                            # Partial-tile predication: skip rows past the group's M
                            # (per-expert token counts are not tile multiples). N is a
                            # tile multiple (weight columns), so no column predication.
                            if m_base + _ml < m_bound:
                                out_tile[(_ml, _nl)] = accumulators[_mc].to(
                                    self.c_dtype
                                )
                else:
                    tCgC_for_tma_partition = cute.zipped_divide(
                        gC_mnl_slice, self.epi_tile
                    )

                    # thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
                    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        tCgC_for_tma_partition,
                    )

                    epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
                    epi_tile_shape = tCgC_for_tma_partition.shape[1]
                    epi_tile_layout = cute.make_layout(
                        epi_tile_shape, stride=(epi_tile_shape[1], 1)
                    )

                    num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
                    for epi_idx in cutlass.range_constexpr(epi_tile_num):
                        # Copy from accumulators to D registers
                        for epi_v in cutlass.range_constexpr(size_tRS_rD):
                            tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                        # Type conversion
                        acc_vec = tRS_rD.load()
                        tRS_rD_out.store(acc_vec.to(self.c_dtype))

                        # Copy from D registers to shared memory
                        epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(
                            tRS_sD, mode=[3]
                        )
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )

                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                        # Copy from shared memory to global memory (TMA store)
                        if warp_idx == self.epi_store_warp_id:
                            cute.copy(
                                tma_atom_c,
                                bSG_sD[(None, epi_buffer)],
                                bSG_gD[(None, gmem_coord)],
                                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                    tensormap_c_ptr, cute.AddressSpace.generic
                                ),
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()

                        self.epilog_sync_barrier.arrive_and_wait()

                last_group_idx_mma = cur_group_idx
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Fused scatter stores directly to gmem (no TMA store pipeline).
            if cutlass.const_expr(not self.use_fused_scatter):
                tma_store_pipeline.producer_tail()

        #
        # Dedicated transform warp group: dequantize FP4 -> FP8 into sB_fp8 and hand
        # it to the MMA warpgroup via the trans2mma pipeline, so the wgmma path has
        # no per-k-tile transform barrier. It consumes the load pipeline wait-only
        # (never releases it): the MMA can't process stage s until trans2mma[s] is
        # ready, which means the transform finished reading sB[s], so when the MMA
        # releases the load stage the transform is already done with sB[s].
        #
        # const_expr gate: when the dedicated transform warps are off, this block
        # (which references the trans2mma pipeline) must not be traced at all -- a
        # runtime `if is_transform_warp_group` would still trace its body.
        if cutlass.const_expr(self.use_xform_warps):
            if is_transform_warp_group:
                tile_sched = StaticPersistentGroupTileScheduler.create(
                    tile_sched_params_for_sched,
                    bid,
                    grid_dim,
                    self.cluster_tile_shape_mnk,
                    utils.create_initial_search_state(),
                    group_count,
                    problem_sizes_mnkl,
                )
                work_tile = tile_sched.initial_work_tile_info()

                trans_mainloop_read_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                trans2mma_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )

                while work_tile.is_valid_tile:
                    grouped_info = work_tile.group_search_result
                    cur_group_idx = grouped_info.group_idx
                    cur_k_tile_cnt = grouped_info.cta_tile_count_k

                    # Same per-group UE8M0 scale tensor the inline transform builds.
                    if cutlass.const_expr(self.use_real_scale):
                        scale_gptr = cute.make_ptr(
                            cutlass.Uint8,
                            ptrs_abc[(cur_group_idx, 3)],
                            cute.AddressSpace.gmem,
                            assumed_align=1,
                        )
                        real_scale = cute.make_tensor(
                            scale_gptr,
                            cute.make_layout(
                                (
                                    grouped_info.problem_shape_n,
                                    grouped_info.problem_shape_k // 32,
                                    cutlass.Int32(1),
                                ),
                                stride=(
                                    grouped_info.problem_shape_k // 32,
                                    1,
                                    cutlass.Int32(0),
                                ),
                            ),
                        )
                        n_offset = grouped_info.cta_tile_idx_n * self.tile_shape_mnk[1]
                    else:
                        real_scale = None
                        n_offset = cutlass.Int32(0)

                    trans_mainloop_read_state.reset_count()
                    trans2mma_producer_state.reset_count()

                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        # sB[stage] loaded by the DMA warpgroup (wait only -- the
                        # MMA warpgroup owns the load pipeline's consumer_release).
                        mainloop_pipeline.consumer_wait(trans_mainloop_read_state)
                        # sB_fp8[stage] slot free (MMA released its previous use).
                        trans2mma_pipeline.producer_acquire(trans2mma_producer_state)
                        self._transform_b_tile(
                            sB,
                            sB_fp8,
                            trans_mainloop_read_state.index,
                            tidx,
                            k_tile,
                            real_scale,
                            n_offset,
                            sB_fp8_single,
                        )
                        trans2mma_pipeline.producer_commit(trans2mma_producer_state)
                        trans_mainloop_read_state.advance()
                        trans2mma_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                trans2mma_pipeline.producer_tail(trans2mma_producer_state)

    @cute.jit
    def make_tensor_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
    ):
        """Construct a global tensor for tensormap update from per-group metadata.

        :param group_idx: Index of the current group.
        :param dtype: Element type of the tensor (A, B, or C).
        :param problem_shape_mnk: (M, N, K) of the current group.
        :param strides_abc: Tensor of strides, shape (G, 3, 2), dtype Int32.
        :param tensor_address_abc: Tensor of base ptrs, shape (G, 3), dtype Int64.
        :param tensor_index: 0=A, 1=B, 2=C.
        """
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )

        strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_rmem_tensor(
            cute.make_layout(2),
            strides_abc.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        if cutlass.const_expr(tensor_index == 0):  # tensor A
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif cutlass.const_expr(tensor_index == 1):  # tensor B
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:  # tensor C
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
        transform_dtype: Optional[type[cutlass.Numeric]] = None,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        # The FP4 -> FP8 transform keeps a second (FP8) B buffer per stage.
        if transform_dtype is not None:
            ab_bytes_per_stage += cute.size(b_shape) * transform_dtype.width // 8
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_stage = 4
        epi_bytes = c_bytes_per_stage * epi_stage

        mbar_helpers_bytes = 1024

        ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: tuple[int, int, int],
        element_type: type[cutlass.Numeric],
        is_cooperative: bool = False,
        epi_tile_override: Optional[tuple[int, int]] = None,
    ) -> tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum for the output matrix C
        :type c_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = (
            a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        b_is_k_major = (
            b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        total_num_clusters: cutlass.Int32,
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple]:
        """Compute tile scheduler params and grid shape for grouped GEMM.

        :param total_num_clusters: Total clusters across all groups (runtime value).
        :type total_num_clusters: cutlass.Int32
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: (tile_sched_params, grid)
        :rtype: tuple
        """
        # The scheduler params carry the (runtime) total: the persistent loop's
        # validity check compares tile indices against it on device.
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
        )
        # The LAUNCH GRID must stay static (compile-time): build it from a params
        # whose z-extent is the static max_active_clusters. A persistent kernel never
        # usefully launches more than max_active clusters anyway; when the (runtime)
        # total is smaller, the surplus CTAs see no valid tile and exit immediately.
        grid_params = utils.PersistentTileSchedulerParams(
            (
                cluster_shape_mn[0],
                cluster_shape_mn[1],
                cutlass.Int32(max_active_clusters),
            ),
            (*cluster_shape_mn, 1),
        )
        grid = StaticPersistentGroupTileScheduler.get_grid_shape(
            grid_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # W4A8 MXFP4 mixed-input path: B is packed MXFP4 (FP4 e2m1) and is
        # dequantized to FP8 e4m3 in a transform stage before the FP8 wgmma;
        # A is FP8 e4m3. The MMA runs in FP8, so A and B intentionally differ
        # in *input* width here (the width-equality / same-kind checks below
        # are for the homogeneous-precision paths and do not apply). The epilogue
        # casts the FP32 accumulator to c_dtype generically (.to(c_dtype)), so
        # BF16/FP16/FP32 outputs are all supported.
        if b_dtype == cutlass.Float4E2M1FN:
            return (
                a_dtype == cutlass.Float8E4M3FN
                and acc_dtype == cutlass.Float32
                # Float8E4M3FN C is the fused-SwiGLU epilogue (GEMM1 writes the
                # silu(gate)*up activation straight to FP8 for GEMM2).
                and c_dtype
                in (
                    cutlass.Float16,
                    cutlass.BFloat16,
                    cutlass.Float32,
                    cutlass.Float8E4M3FN,
                )
                and a_major == "k"
                and b_major == "k"
            )

        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
            cutlass.Uint8,
            cutlass.Int8,
        }
        if a_dtype not in valid_ab_dtypes:
            is_valid = False
        if b_dtype not in valid_ab_dtypes:
            is_valid = False

        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        if a_dtype.width != b_dtype.width:
            is_valid = False
        if not a_dtype.is_same_kind(b_dtype):
            is_valid = False

        # for 8-bit types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (
            b_dtype.width == 8 and b_major != "k"
        ):
            is_valid = False

        # Define compatibility mapping between accumulator type and AB type
        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {cutlass.Uint8, cutlass.Int8},
        }
        # Check compatibility between accumulator type and A type
        if a_dtype not in acc_ab_compatibility[acc_dtype]:
            is_valid = False

        # Define compatibility mapping between accumulator type and C type
        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
        }
        # Check compatibility between accumulator type and C type
        if c_dtype not in acc_c_compatibility[acc_dtype]:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid


# ---------------------------------------------------------------------------
# Helper functions for tensor creation (ported from blackwell/grouped_gemm.py)
# ---------------------------------------------------------------------------

# MXFP4 (FP4 e2m1) code -> value, for SM90 reference dequant (no Blackwell cvt).
_FP4_LUT = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]

# UE8M0 block-scale baseline: stored byte v -> scale 2^(v - W4A8_SCALE_BASE).
W4A8_SCALE_BASE = 127

# Compiled-kernel cache for w4a8_mxfp4_grouped_gemm. cute.compile is expensive
# (seconds); the compiled kernel only specializes on compile-time params (operands
# are passed at call time via pointer/stride/size metadata tensors), so the same
# compiled kernel is reused across calls with the same config. Keyed in
# w4a8_mxfp4_grouped_gemm (dtypes, tile/cluster shape, total clusters, scatter/gather,
# env flags). Mirrors the _gather_kernel_cache pattern in the blockscaled DSL GEMMs.
_W4A8_KERNEL_CACHE: dict = {}

# GPU occupancy info (sm_count, max_active_clusters) is GPU-fixed, but the
# HardwareInfo().get_max_active_clusters() driver query is slow (tens of ms), so cache
# it per cluster shape to cut per-call host overhead.
_W4A8_HW_CACHE: dict = {}

# Per-call operand-metadata tensors (problem sizes / strides / pointers) are rebuilt
# every call via cutlass_torch.cute_tensor_like, which does a host->device copy_ -- the
# bulk of the per-call host overhead AND a CUDA-graph-capture blocker (copy_ on a
# capture/side stream raises cudaErrorInvalidAddressSpace). When the caller reuses the
# same operand buffers across calls (steady-state serving, CUDA-graph replay), the
# pointers/strides/sizes are identical, so the built cute tensors are reusable. Memoize
# them keyed on the exact (ptrs, strides, sizes) content: a hit skips the rebuild +
# copy_ entirely. Correct by construction -- the key IS the metadata, and the metadata
# encodes only pointers, not operand content (which the kernel reads through them at
# launch). Bounded to avoid unbounded growth when a caller churns fresh buffers.
_W4A8_META_CACHE: dict = {}
# Sized for serving: a model holds (layers x capacity-buckets x 2 GEMMs) distinct
# operand-metadata sets (e.g. DSv4: 44 MoE layers x buckets x 2 can exceed 256), each a
# few KB of device tensors -- the cap guards runaway churn, not memory.
_W4A8_META_CACHE_CAP = 4096


def _w4a8_hw_info(cluster_shape_mn):
    key = tuple(cluster_shape_mn)
    if key not in _W4A8_HW_CACHE:
        hw = utils.HardwareInfo()
        _W4A8_HW_CACHE[key] = (
            hw.get_max_active_clusters(1),
            hw.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
        )
    return _W4A8_HW_CACHE[key]


def create_tensor_and_stride(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    dtype: Type[cutlass.Numeric],
    is_dynamic_layout: bool = True,
    torch_tensor_cpu: "torch.Tensor" = None,
) -> tuple:
    """Create a GPU tensor and return its pointer, torch tensor, cute tensor, CPU tensor, and strides."""
    if dtype == cutlass.Float4E2M1FN:
        # SM90 has no f32->FP4 hardware cvt (Blackwell-only), so cutlass_torch's
        # convert path fails. Build a packed MXFP4 weight directly instead: random
        # FP4 codes packed 2 nibbles/byte into the first half of an int8 buffer
        # whose logical shape matches the FP4 tensor. The seed (codes) is the CPU
        # tensor so the benchmark loop can reproduce it. Layout matches
        # cutlass_torch.matrix's (l, mode0, mode1) -> (mode0, mode1, l) for k-major.
        if is_mode0_major:
            raise NotImplementedError(
                "MXFP4 tensor creation only supports k-major B (is_mode0_major=False)"
            )
        from cutlass.cute.runtime import from_dlpack as _from_dlpack
        from cutlass.torch import get_leading_dim as _get_leading_dim

        n_, k_ = mode0, mode1
        if torch_tensor_cpu is None:
            if os.getenv("FI_W4A8_CONST_B"):
                # Debug: uniform B (all code 2 = value 1.0) -> ref = rowsum(A),
                # isolating value-correctness from K-ordering.
                torch_tensor_cpu = torch.full((l, n_, k_), 2, dtype=torch.uint8)
            else:
                torch_tensor_cpu = torch.randint(0, 16, (l, n_, k_), dtype=torch.uint8)
        codes = torch_tensor_cpu
        flat = codes.reshape(-1)
        packed = (flat[0::2] | (flat[1::2] << 4)).to(torch.int8)
        buf_lnk = torch.zeros((l, n_, k_), dtype=torch.int8, device="cuda")
        buf_lnk.view(-1)[: (l * n_ * k_) // 2] = packed.cuda()
        dev = buf_lnk.permute(1, 2, 0)  # (n, k, l), strides (k, 1, n*k)
        cute_tensor = _from_dlpack(dev, assumed_align=16)
        cute_tensor.element_type = cutlass.Float4E2M1FN
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=_get_leading_dim(dev)
            )
        # Strides are consumed by the kernel's tensormap update, which loads B as
        # packed Uint8 [N, K/2]: the N stride is k/2 bytes (one packed row), not k.
        return (dev.data_ptr(), dev, cute_tensor, codes, (k_ // 2, 1))
    if torch_tensor_cpu is None:
        torch_tensor_cpu = cutlass_torch.matrix(l, mode0, mode1, is_mode0_major, dtype)
    cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
        torch_tensor_cpu, dtype, is_dynamic_layout, assumed_align=16
    )
    return (
        torch_tensor.data_ptr(),
        torch_tensor,
        cute_tensor,
        torch_tensor_cpu,
        torch_tensor.stride()[:-1],
    )


def create_tensors_for_all_groups(
    problem_sizes_mnkl: List[Tuple[int, int, int, int]],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    torch_fp32_tensors_abc: List[List] = None,
) -> tuple:
    """Create A/B/C tensors for all groups."""
    if torch_fp32_tensors_abc is not None and len(torch_fp32_tensors_abc) != len(
        problem_sizes_mnkl
    ):
        raise ValueError("torch_fp32_tensors_abc must have one entry per group")

    new_torch_fp32_tensors_abc = (
        [] if torch_fp32_tensors_abc is None else torch_fp32_tensors_abc
    )
    torch_tensors_abc = []
    cute_tensors_abc = []
    strides_abc = []
    ptrs_abc = []

    for group_idx, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        existing_cpu_a = (
            torch_fp32_tensors_abc[group_idx][0] if torch_fp32_tensors_abc else None
        )
        existing_cpu_b = (
            torch_fp32_tensors_abc[group_idx][1] if torch_fp32_tensors_abc else None
        )
        existing_cpu_c = (
            torch_fp32_tensors_abc[group_idx][2] if torch_fp32_tensors_abc else None
        )

        ptr_a, torch_a, cute_a, fp32_a, stride_mk_a = create_tensor_and_stride(
            l, m, k, a_major == "m", a_dtype, torch_tensor_cpu=existing_cpu_a
        )
        ptr_b, torch_b, cute_b, fp32_b, stride_nk_b = create_tensor_and_stride(
            l, n, k, b_major == "n", b_dtype, torch_tensor_cpu=existing_cpu_b
        )
        ptr_c, torch_c, cute_c, fp32_c, stride_mn_c = create_tensor_and_stride(
            l, m, n, c_major == "m", c_dtype, torch_tensor_cpu=existing_cpu_c
        )

        if torch_fp32_tensors_abc is None:
            new_torch_fp32_tensors_abc.append([fp32_a, fp32_b, fp32_c])

        ptrs_abc.append([ptr_a, ptr_b, ptr_c])
        torch_tensors_abc.append([torch_a, torch_b, torch_c])
        strides_abc.append([stride_mk_a, stride_nk_b, stride_mn_c])
        cute_tensors_abc.append((cute_a, cute_b, cute_c))

    return (
        ptrs_abc,
        torch_tensors_abc,
        cute_tensors_abc,
        strides_abc,
        new_torch_fp32_tensors_abc,
    )


def create_group_metadata(
    problem_sizes_mnkl: List[Tuple[int, int, int, int]],
    a_major: str,
    b_major: str,
    c_major: str,
) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    """Create per-group pointer/stride metadata without allocating operand tensors."""

    def get_stride(mode0: int, mode1: int, is_mode0_major: bool) -> tuple[int, int]:
        # Matches the layout produced by cutlass_torch.matrix(...).permute(...).
        return (1, mode0) if is_mode0_major else (mode1, 1)

    ptrs_abc = []
    strides_abc = []

    for m, n, k, _ in problem_sizes_mnkl:
        ptrs_abc.append([0, 0, 0])
        strides_abc.append(
            [
                get_stride(m, k, a_major == "m"),
                get_stride(n, k, b_major == "n"),
                get_stride(m, n, c_major == "m"),
            ]
        )

    return ptrs_abc, strides_abc


def _to_reference_operand_fp32(
    tensor: "torch.Tensor", dtype: Type[cutlass.Numeric]
) -> "torch.Tensor":
    """Convert an operand tensor to fp32 for host-side reference GEMM.

    For FP8 dtypes, tensors are stored as int8 bit-patterns by
    `cutlass_torch.matrix`, so we must reinterpret before casting.
    """
    tensor_cpu = tensor.cpu()
    if dtype == cutlass.Float8E4M3FN:
        return tensor_cpu.view(torch.float8_e4m3fn).to(dtype=torch.float32)
    if dtype == cutlass.Float8E5M2:
        return tensor_cpu.view(torch.float8_e5m2).to(dtype=torch.float32)
    if dtype == cutlass.Float4E2M1FN:
        # tensor is the (n, k, l) int8 view of an (l, n, k)-contiguous buffer with
        # MXFP4 packed 2 nibbles/byte in its first half. Recover the (l, n, k) flat
        # packed bytes, unpack to FP4 codes (FP4-offset order), LUT-dequant, and
        # reshape back to (n, k, l) to match the host reference einsum.
        n_, k_, l_ = tensor_cpu.shape
        flat = tensor_cpu.permute(2, 0, 1).contiguous().reshape(-1)
        packed = flat[: (l_ * n_ * k_) // 2].to(torch.int32) & 0xFF
        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
        nibbles = torch.stack([lo, hi], dim=-1).reshape(-1)
        lut = torch.tensor(_FP4_LUT, dtype=torch.float32)
        vals = lut[nibbles.long()].reshape(l_, n_, k_).permute(1, 2, 0)  # (n,k,l)
        if os.getenv("FI_W4A8_BLOCK_SCALE") is not None:
            # Per-32-block scale 2^((block%4)-1), matching the kernel transform.
            se = ((torch.arange(k_) // 32) % 4 - 1).float()
            return vals * (2.0**se).view(1, k_, 1)
        # Constant MXFP4 block scale 2^_se (bring-up; matches the kernel's
        # per-element exponent offset until per-32-block scale data is plumbed).
        return vals * (2.0 ** int(os.getenv("FI_W4A8_SCALE_EXP", "0")))
    return tensor_cpu.to(dtype=torch.float32)


def _compute_total_num_clusters(
    problem_sizes_mnkl: List[Tuple[int, int, int, int]],
    cluster_tile_shape_mn: Tuple[int, int],
) -> int:
    """Total cluster tiles across all groups (persistent-scheduler grid size)."""
    total = 0
    for m, n, _, _ in problem_sizes_mnkl:
        nm = (m + cluster_tile_shape_mn[0] - 1) // cluster_tile_shape_mn[0]
        nn = (n + cluster_tile_shape_mn[1] - 1) // cluster_tile_shape_mn[1]
        total += nm * nn
    return total


@flashinfer_api
def w4a8_mxfp4_grouped_gemm(
    a_fp8_list: "List[torch.Tensor]",
    b_packed_list: "List[torch.Tensor]",
    scale_list: "List[torch.Tensor]",
    c_list: "List[torch.Tensor]",
    problem_sizes_mnkl: List[Tuple[int, int, int, int]],
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    c_dtype: Type[cutlass.Numeric] = cutlass.Float16,
    tile_shape_mn: Tuple[int, int] = None,
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM,
    route_maps: "List[torch.Tensor]" = None,
    weights: "List[torch.Tensor]" = None,
    output: "torch.Tensor" = None,
    activations: "torch.Tensor" = None,
    gather_route_maps: "List[torch.Tensor]" = None,
    no_accumulate: bool = False,
    swiglu: bool = False,
    swiglu_alpha: float = None,
    swiglu_beta: float = None,
    swiglu_limit: float = None,
    dequant_exp_bias: int = 0,
) -> None:
    """Callable W4A8 MXFP4 grouped GEMM: C[g] = A[g] @ dequant(B[g], scale[g])^T.

    This is the Python entry point an MoE wrapper drives (gather/route happens in
    the caller for now; the GEMM stays a plain grouped GEMM). Per group g with
    shape (M_g, N, K, 1), the caller must pass, all on CUDA and kept alive until
    after the launch synchronizes:

    - ``a_fp8_list[g]``    : FP8 e4m3 activation ``[M_g, K]``, row-major (K contiguous).
    - ``b_packed_list[g]`` : MXFP4 weight packed as Uint8 ``[N, K // 2]`` (2 nibbles
                             per byte, low nibble = even K index), row-major.
    - ``scale_list[g]``    : UE8M0 block scale ``[N, K // 32]`` as Uint8, row-major.
    - ``c_list[g]``        : preallocated output ``[M_g, N]`` in ``c_dtype``, row-major.

    Output is written in place into ``c_list``. K must be a multiple of 32 and the
    contiguous dims 16-byte aligned (K % 16 == 0 for FP8 A, N % … for C).

    ``swiglu=True`` fuses the GEMM1 SwiGLU activation into the epilogue: the weight columns
    must be interleaved gate/up (row 2j = gate_j, 2j+1 = up_j), ``c_dtype`` must be
    ``Float8E4M3FN``, and each ``c_list[g]`` is ``[M_g, N//2]``. ``swiglu_limit`` (a uniform
    scalar) additionally selects the clamped SwiGLUBias variant (gate clamped to
    ``(-inf, limit]``, up to ``[-limit, limit]`` then ``+ swiglu_beta``, sigmoid argument
    scaled by ``swiglu_alpha``); ``None`` => plain ``silu(gate)*up``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for w4a8_mxfp4_grouped_gemm")
    # CTA tile selection. The (128,256) cooperative (2-MMA-warpgroup) tile is ~1.2x
    # faster than (128,128) on large-N MoE GEMMs -- more warps/CTA hide the
    # dequant->smem->wgmma latency (the GEMM is latency-bound at low occupancy, not
    # compute/BW-bound), and the bigger tile amortizes the dequant overhead. It needs
    # N % 256 == 0, so the auto default upgrades to it only when every group's N
    # qualifies, else falls back to (128,128). FI_W4A8_TILE_MN="M,N" forces a tile;
    # an explicit tile_shape_mn arg is respected as-is. See spike_w4a8_tile_sweep.py.
    _tile_override = os.environ.get("FI_W4A8_TILE_MN")
    if _tile_override:
        _tm, _tn = _tile_override.split(",")
        tile_shape_mn = (int(_tm), int(_tn))
    elif tile_shape_mn is None:
        if all(n % 256 == 0 for (_, n, _, _) in problem_sizes_mnkl):
            tile_shape_mn = (128, 256)
        else:
            tile_shape_mn = (128, 128)
    num_groups = len(problem_sizes_mnkl)
    # c_list is unused in token-scatter mode; a_fp8_list is unused in fused-gather
    # mode (A is the shared `activations` tensor).
    _c_len_ok = output is not None or len(c_list) == num_groups
    _a_len_ok = activations is not None or len(a_fp8_list) == num_groups
    if not (
        len(b_packed_list) == len(scale_list) == num_groups and _c_len_ok and _a_len_ok
    ):
        raise ValueError("operand lists and problem_sizes_mnkl must have equal length")

    a_dtype = cutlass.Float8E4M3FN
    b_dtype = cutlass.Float4E2M1FN
    a_major, b_major, c_major = "k", "k", "n"

    # Token-scatter (FS-1) mode: when an `output` tensor is given, each group's
    # output rows are scattered into it via a per-group route map (local row ->
    # output/token row), scaled by a per-row routing weight. The per-group C
    # pointer becomes the shared `output` base, and route map + weights are
    # appended as the 5th/6th per-group operands. Default (output is None) keeps
    # the direct per-group-C path.
    scatter = output is not None
    if scatter and not (route_maps is not None and weights is not None):
        raise ValueError("token-scatter mode requires route_maps and weights")

    # Fused-gather (FS-1 input side) mode: when an `activations` tensor is given,
    # operand 0 (A) is its shared base and a per-group `gather_route_maps[g]` maps
    # each local row -> source row in `activations` (operand 6). a_fp8_list is then
    # unused. Default (activations is None) keeps the per-group A path.
    gather = activations is not None
    if gather and gather_route_maps is None:
        raise ValueError("fused-gather mode requires gather_route_maps")

    # The kernel reads the per-(N, K/32) UE8M0 scale as a 4th per-group operand. This
    # callable always supplies a real scale, and token-scatter / fused-gather follow
    # from the presence of `output` / `activations`; these are passed straight to the
    # kernel constructor as explicit booleans (no os.environ signaling).

    # Per-group pointer/stride metadata: scale is the 4th operand -> (G, 4); in
    # token-scatter mode route map + weights append -> (G, 6).
    ptrs_abc: List[List[int]] = []
    strides_abc: List[List[Tuple[int, int]]] = []
    for g, (_m, n, k, _l) in enumerate(problem_sizes_mnkl):
        b_g, s_g = b_packed_list[g], scale_list[g]
        a_ptr = activations.data_ptr() if gather else a_fp8_list[g].data_ptr()
        c_ptr = output.data_ptr() if scatter else c_list[g].data_ptr()
        row = [a_ptr, b_g.data_ptr(), c_ptr, s_g.data_ptr()]
        # A [M,K] k-major -> (K,1); B packed Uint8 [N,K/2] -> (K/2,1);
        # C [M,N] n-major -> (N,1); scale [N,K/32] -> (K/32,1).
        stride_row = [(k, 1), (k // 2, 1), (n, 1), (k // 32, 1)]
        # operands 4,5 = scatter route map + weights (padded when not scattering but
        # gathering, so the gather route map keeps a fixed column index 6).
        if scatter:
            row += [route_maps[g].data_ptr(), weights[g].data_ptr()]
            stride_row += [(1, 0), (1, 0)]
        elif gather:
            row += [0, 0]
            stride_row += [(1, 0), (1, 0)]
        # operand 6 = gather route map [M] int32 (local row -> source row in A).
        if gather:
            row += [gather_route_maps[g].data_ptr()]
            stride_row += [(1, 0)]
        ptrs_abc.append(row)
        strides_abc.append(stride_row)

    alignment = 16
    min_ab_size = alignment * 8 // a_dtype.width
    min_c_size = alignment * 8 // c_dtype.width

    sm_count, max_active_clusters = _w4a8_hw_info(cluster_shape_mn)

    # Per-call metadata tensors: the actual operand pointers/strides/sizes, passed to
    # the (cached) compiled kernel at launch time -- the kernel reads all operands
    # through them, not through the type-spec dummies. cute_tensor_like does a
    # host->device copy_, so memoize on the exact metadata content (see _W4A8_META_CACHE
    # above): when operand buffers are reused across calls the rebuild + copy_ is skipped
    # (cuts host overhead; makes the launch CUDA-graph-capturable since a hit issues no
    # copy_). Tuple-ify the nested lists for a hashable key.
    _meta_key = (
        tuple(map(tuple, ptrs_abc)),
        tuple(tuple(map(tuple, sr)) for sr in strides_abc),
        tuple(problem_sizes_mnkl),
        c_dtype,
    )
    _meta = _W4A8_META_CACHE.get(_meta_key)
    if _meta is None:
        tensor_of_dim_size_mnkl, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(problem_sizes_mnkl, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        tensor_of_strides_abc, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(strides_abc, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        tensor_of_ptrs_abc, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(ptrs_abc, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        if len(_W4A8_META_CACHE) >= _W4A8_META_CACHE_CAP:
            _W4A8_META_CACHE.clear()  # churny caller: drop and rebuild rather than grow
        _W4A8_META_CACHE[_meta_key] = (
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc,
            tensor_of_ptrs_abc,
        )
    else:
        tensor_of_dim_size_mnkl, tensor_of_strides_abc, tensor_of_ptrs_abc = _meta

    cluster_tile_shape_mn = (
        tile_shape_mn[0] * cluster_shape_mn[0],
        tile_shape_mn[1] * cluster_shape_mn[1],
    )
    total_num_clusters = _compute_total_num_clusters(
        problem_sizes_mnkl, cluster_tile_shape_mn
    )

    # Launch on torch's CURRENT stream, not the default stream. Serving frameworks
    # (sglang) run the model forward on a non-default stream; launching the GEMM on
    # the default stream while the surrounding torch/triton ops (activation cast,
    # permute, requant, finalize) run on the current stream gives ZERO ordering
    # between the two -- e.g. the requant kernel can read the GEMM1 output buffer
    # before the GEMM wrote it (uninitialized torch.empty garbage -> huge per-token
    # scales -> ~1e37 layer outputs, observed in-situ on sglang with healthy inputs).
    # On the current stream all ordering is plain stream program order.
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Reuse a cached compiled kernel when the compile-time config matches; only the
    # runtime metadata tensors (built above) and the runtime total_num_clusters change
    # between such calls. The key captures every input cute.compile specializes on:
    # num_groups, operand dtypes (it specializes on pointer types), tile/cluster shape,
    # tensormap mode, max_active_clusters/sm_count (GPU-fixed in the persistent
    # scheduler), the scatter/gather operand-column layout (which also drives the
    # kernel's behavior flags below), and a snapshot of the FI_W4A8_* debug env flags
    # the kernel reads at trace time (SCALAR_XFORM/NO_PRMT/... change code).
    # total_num_clusters is deliberately NOT in the key: it is a runtime kernel arg
    # (per-call routing changes it every MoE forward in serving; keying on it caused a
    # cute.compile per MoE call -- a multi-second recompile storm).
    _env_snapshot = tuple(
        sorted((k, v) for k, v in os.environ.items() if k.startswith("FI_W4A8_"))
    )
    cache_key = (
        num_groups,
        a_dtype,
        b_dtype,
        c_dtype,
        acc_dtype,
        tile_shape_mn,
        cluster_shape_mn,
        tensormap_update_mode,
        max_active_clusters,
        sm_count,
        scatter,
        gather,
        no_accumulate,
        swiglu,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        dequant_exp_bias,
        _env_snapshot,
    )
    _cached = _W4A8_KERNEL_CACHE.get(cache_key)
    if _cached is None:
        # On a miss, build the compile-time-only type-spec dummies (their data is never
        # read -- the kernel reads operands via the pointer metadata) and the tensormap
        # workspace, then compile; cache all three so cached calls skip rebuilding them.
        # NOTE: the tensormap workspace is reused across cached calls of the same config
        # -- safe for sequential / same-stream launches (the kernel re-inits it each
        # launch); a concurrent multi-stream caller would need a separate workspace.
        initial_cute_tensors_abc = [
            create_tensor_and_stride(
                1, min_ab_size, min_ab_size, a_major == "m", a_dtype
            )[2],
            create_tensor_and_stride(
                1, min_ab_size, min_ab_size, b_major == "n", b_dtype
            )[2],
            create_tensor_and_stride(
                1, min_c_size, min_c_size, c_major == "m", c_dtype
            )[2],
        ]
        tensor_of_tensormap, _ = cutlass_torch.cute_tensor_like(
            torch.empty(
                (
                    sm_count,
                    HopperGroupedGemmPersistentKernel.num_tensormaps,
                    HopperGroupedGemmPersistentKernel.bytes_per_tensormap // 8,
                ),
                dtype=torch.int64,
            ),
            cutlass.Int64,
            is_dynamic_layout=False,
        )
        # Behavior is passed explicitly (no os.environ signaling): this API always
        # supplies a real UE8M0 scale, and token-scatter / fused-gather follow from
        # the presence of `output` / `activations`.
        grouped_gemm = HopperGroupedGemmPersistentKernel(
            acc_dtype,
            tile_shape_mn,
            cluster_shape_mn,
            swizzle_size=1,
            raster_along_m=True,
            tensormap_update_mode=tensormap_update_mode,
            use_real_scale=True,
            use_token_scatter=scatter,
            use_fused_scatter=scatter,
            use_fused_gather=gather,
            use_real_gather=gather,
            scatter_no_accumulate=no_accumulate,
            use_swiglu=swiglu,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            dequant_exp_bias=dequant_exp_bias,
        )
        compiled_grouped_gemm = cute.compile(
            grouped_gemm,
            initial_cute_tensors_abc[0],
            initial_cute_tensors_abc[1],
            initial_cute_tensors_abc[2],
            num_groups,
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc,
            tensor_of_ptrs_abc,
            cutlass.Int32(total_num_clusters),
            tensor_of_tensormap,
            max_active_clusters,
            current_stream,
        )
        _W4A8_KERNEL_CACHE[cache_key] = (
            compiled_grouped_gemm,
            initial_cute_tensors_abc,
            tensor_of_tensormap,
        )
    else:
        compiled_grouped_gemm, initial_cute_tensors_abc, tensor_of_tensormap = _cached
    compiled_grouped_gemm(
        initial_cute_tensors_abc[0],
        initial_cute_tensors_abc[1],
        initial_cute_tensors_abc[2],
        tensor_of_dim_size_mnkl,
        tensor_of_strides_abc,
        tensor_of_ptrs_abc,
        cutlass.Int32(total_num_clusters),
        tensor_of_tensormap,
        current_stream,
    )


def run(
    num_groups: int,
    problem_sizes_mnkl: List[Tuple[int, int, int, int]],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    tile_shape_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM,
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    """Prepare per-group tensors, compile, launch, and validate the Hopper grouped GEMM kernel.

    :return: Execution time in microseconds.
    :rtype: float
    """
    print("Running Hopper Grouped GEMM test with:")
    print(f"{num_groups} groups")
    for i, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        print(f"Group {i}: {m}x{n}x{k}x{l}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Tile Shape: {tile_shape_mn}, Cluster Shape: {cluster_shape_mn}")
    print(f"Tensor map update mode: {tensormap_update_mode}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Validate dtypes (reuse existing static method, check each group)
    for m, n, k, l in problem_sizes_mnkl:
        if not HopperGroupedGemmPersistentKernel.is_valid_dtypes(
            a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major
        ):
            raise TypeError(
                f"unsupported dtype combination: A {a_dtype}, B {b_dtype}, "
                f"Acc {acc_dtype}, C {c_dtype}, {a_major=}, {b_major=}"
            )
        if not HopperGroupedGemmPersistentKernel.is_valid_tensor_alignment(
            m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
        ):
            raise TypeError(
                f"Group {m}x{n}x{k}x{l}: contiguous dimension not 16-byte aligned"
            )

    compile_only = skip_ref_check and iterations <= 0

    if compile_only:
        ptrs_abc, strides_abc = create_group_metadata(
            problem_sizes_mnkl, a_major, b_major, c_major
        )
        torch_tensors_abc = []
        torch_fp32_tensors_abc = []
    else:
        # Create per-group tensors only when we will execute or validate.
        (
            ptrs_abc,
            torch_tensors_abc,
            _,
            strides_abc,
            torch_fp32_tensors_abc,
        ) = create_tensors_for_all_groups(
            problem_sizes_mnkl, a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
        )

    # W4A8: per-(N, K/32) UE8M0 block scale for B, plumbed as a 4th per-group
    # operand (gmem-read by the kernel -- the scale's 4-byte contiguous dim is too
    # small for TMA). Fully gated on FI_W4A8_REAL_SCALE: default path is unchanged
    # (metadata stays (G,3), kernel does no scale read). When on, metadata becomes
    # (G,4) and the kernel reads scale[n, k//32] for each B element.
    w4a8_scales = None
    if b_dtype == cutlass.Float4E2M1FN and os.getenv("FI_W4A8_REAL_SCALE"):
        w4a8_scales = []
        for g, (_m, n, k, _l) in enumerate(problem_sizes_mnkl):
            if os.getenv("FI_W4A8_UNIFORM_SCALE"):
                sc = torch.full(
                    (n, k // 32), W4A8_SCALE_BASE + 1, dtype=torch.uint8, device="cuda"
                )
            else:
                sc = torch.randint(
                    W4A8_SCALE_BASE - 1,
                    W4A8_SCALE_BASE + 3,
                    (n, k // 32),
                    dtype=torch.uint8,
                    device="cuda",
                )
            w4a8_scales.append(sc)
            ptrs_abc[g] = list(ptrs_abc[g]) + [sc.data_ptr()]
            strides_abc[g] = list(strides_abc[g]) + [(k // 32, 1)]

    # Build small "initial" tensors that carry only dtype+majorness (used for TMA atom init)
    alignment = 16
    min_ab_size = alignment * 8 // a_dtype.width
    min_c_size = alignment * 8 // c_dtype.width
    initial_cute_tensors_abc = [
        create_tensor_and_stride(1, min_ab_size, min_ab_size, a_major == "m", a_dtype)[
            2
        ],
        create_tensor_and_stride(1, min_ab_size, min_ab_size, b_major == "n", b_dtype)[
            2
        ],
        create_tensor_and_stride(1, min_c_size, min_c_size, c_major == "m", c_dtype)[2],
    ]

    sm_count, max_active_clusters = _w4a8_hw_info(cluster_shape_mn)

    # Tensor map workspace: (num_sms, 3, bytes_per_tensormap // 8) of Int64
    tensormap_shape = (
        sm_count,
        HopperGroupedGemmPersistentKernel.num_tensormaps,
        HopperGroupedGemmPersistentKernel.bytes_per_tensormap // 8,
    )
    tensor_of_tensormap, tensor_of_tensormap_torch = cutlass_torch.cute_tensor_like(
        torch.empty(tensormap_shape, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
    )

    grouped_gemm = HopperGroupedGemmPersistentKernel(
        acc_dtype,
        tile_shape_mn,
        cluster_shape_mn,
        swizzle_size=1,
        raster_along_m=True,
        tensormap_update_mode=tensormap_update_mode,
    )

    # Build device tensors for problem shapes, strides, and pointers
    tensor_of_dim_size_mnkl, tensor_of_dim_size_mnkl_torch = (
        cutlass_torch.cute_tensor_like(
            torch.tensor(problem_sizes_mnkl, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
    )
    tensor_of_strides_abc, tensor_of_strides_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(strides_abc, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    tensor_of_ptrs_abc, tensor_of_ptrs_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(ptrs_abc, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # Compute total number of cluster tiles across all groups
    def compute_total_num_clusters(
        problem_sizes: List[Tuple[int, int, int, int]],
        cluster_tile_shape_mn: Tuple[int, int],
    ) -> int:
        total = 0
        for m, n, _, _ in problem_sizes:
            nm = (m + cluster_tile_shape_mn[0] - 1) // cluster_tile_shape_mn[0]
            nn = (n + cluster_tile_shape_mn[1] - 1) // cluster_tile_shape_mn[1]
            total += nm * nn
        return total

    # cluster tile shape for Hopper: tile_shape_mn * cluster_shape_mn
    cluster_tile_shape_mn = (
        tile_shape_mn[0] * cluster_shape_mn[0],
        tile_shape_mn[1] * cluster_shape_mn[1],
    )
    total_num_clusters = compute_total_num_clusters(
        problem_sizes_mnkl, cluster_tile_shape_mn
    )

    # Current stream for the same reason as the public API (identical to the default
    # stream in this single-script CLI, but keeps the two paths consistent).
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Compile kernel
    _compiler = cute.compile
    if os.environ.get("CUTE_DSL_KEEP_PTX"):
        _compiler = cute.compile[cute.KeepPTX()]
    compiled_grouped_gemm = _compiler(
        grouped_gemm,
        initial_cute_tensors_abc[0],
        initial_cute_tensors_abc[1],
        initial_cute_tensors_abc[2],
        num_groups,
        tensor_of_dim_size_mnkl,
        tensor_of_strides_abc,
        tensor_of_ptrs_abc,
        cutlass.Int32(total_num_clusters),
        tensor_of_tensormap,
        max_active_clusters,
        current_stream,
    )

    if not skip_ref_check:
        compiled_grouped_gemm(
            initial_cute_tensors_abc[0],
            initial_cute_tensors_abc[1],
            initial_cute_tensors_abc[2],
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc,
            tensor_of_ptrs_abc,
            cutlass.Int32(total_num_clusters),
            tensor_of_tensormap,
            current_stream,
        )
        torch.cuda.synchronize()

        for i, (a_t, b_t, c_t) in enumerate(torch_tensors_abc):
            a_ref = _to_reference_operand_fp32(a_t, a_dtype)
            b_ref = _to_reference_operand_fp32(b_t, b_dtype)
            if w4a8_scales is not None:
                # Apply the per-(N, K/32) UE8M0 block scale: 2^(byte - base).
                sc = w4a8_scales[i].cpu().to(torch.int32) - W4A8_SCALE_BASE  # (n, k/32)
                sc_full = sc.repeat_interleave(32, dim=1).float()  # (n, k)
                b_ref = b_ref * (2.0**sc_full).unsqueeze(-1)  # broadcast over l
            ref = torch.einsum(
                "mkl,nkl->mnl",
                a_ref,
                b_ref,
            )
            print(f"Checking group {i}...")
            if os.getenv("FI_W4A8_DEBUG"):
                _c = c_t.cpu().float()
                _r = ref.float()
                print("  c[0,:6,0] =", _c[0, :6, 0].tolist())
                print("  r[0,:6,0] =", _r[0, :6, 0].tolist())
                print("  c[:6,0,0] =", _c[:6, 0, 0].tolist())
                print("  r[:6,0,0] =", _r[:6, 0, 0].tolist())
                _ct = _c.transpose(0, 1)  # check M<->N transpose (square M==N?)
                if _ct.shape == _r.shape:
                    print(
                        "  close(c^T, r)=",
                        torch.isclose(_ct, _r, atol=0.5).float().mean().item(),
                    )
                print(
                    "  close(c, r)=",
                    torch.isclose(_c, _r, atol=0.5).float().mean().item(),
                    " |c|mean=",
                    _c.abs().mean().item(),
                    " |r|mean=",
                    _r.abs().mean().item(),
                )
            torch.testing.assert_close(
                c_t.cpu(),
                ref.to(cutlass_torch.dtype(c_dtype)),
                atol=tolerance,
                rtol=1e-03,
            )

    if iterations <= 0:
        return 0

    def generate_tensors():
        (
            ptrs_abc_ws,
            torch_tensors_abc_ws,
            _,
            strides_abc_ws,
            __,
        ) = create_tensors_for_all_groups(
            problem_sizes_mnkl,
            a_dtype,
            b_dtype,
            c_dtype,
            a_major,
            b_major,
            c_major,
            torch_fp32_tensors_abc,
        )
        # W4A8: the kernel was compiled with use_real_scale, so it reads a 4th
        # per-group operand. The benchmark reuses the same B codes (via
        # torch_fp32_tensors_abc), so reuse the alive w4a8_scales and extend the
        # workspace metadata to (G, 4) -- otherwise the kernel reads ptrs[g, 3] OOB.
        if w4a8_scales is not None:
            for g, (_m, _n, k, _l) in enumerate(problem_sizes_mnkl):
                ptrs_abc_ws[g] = list(ptrs_abc_ws[g]) + [w4a8_scales[g].data_ptr()]
                strides_abc_ws[g] = list(strides_abc_ws[g]) + [(k // 32, 1)]
        init_ws = [
            create_tensor_and_stride(
                1, min_ab_size, min_ab_size, a_major == "m", a_dtype
            )[2],
            create_tensor_and_stride(
                1, min_ab_size, min_ab_size, b_major == "n", b_dtype
            )[2],
            create_tensor_and_stride(
                1, min_c_size, min_c_size, c_major == "m", c_dtype
            )[2],
        ]
        strides_ws, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(strides_abc_ws, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        ptrs_ws, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(ptrs_abc_ws, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        tensormap_ws, _ = cutlass_torch.cute_tensor_like(
            torch.empty(tensormap_shape, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
        )
        args = testing.JitArguments(
            init_ws[0],
            init_ws[1],
            init_ws[2],
            tensor_of_dim_size_mnkl,
            strides_ws,
            ptrs_ws,
            cutlass.Int32(total_num_clusters),
            tensormap_ws,
            current_stream,
        )
        return args

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = sum(
            t.numel() * t.element_size() for group in torch_tensors_abc for t in group
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_grouped_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    runtime_s = exec_time / 1.0e6
    fmas = sum(m * n * k for m, n, k, _ in problem_sizes_mnkl)
    gflops = (2 * fmas / 1.0e9) / runtime_s
    print(f"Average Runtime : {exec_time / 1000:.3f} ms")
    print(f"GFLOPS          : {gflops:.1f}")

    return exec_time


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_comma_separated_ints(s: str) -> tuple:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from None


def _parse_problem_sizes(s: str) -> List[Tuple[int, ...]]:
    """Parse e.g. '(4096,4096,4096,1),(512,512,512,1)' into a list of tuples."""
    s = s.strip()
    if s.startswith("("):
        tuples = s.strip("()").split("),(")
        result = []
        for t in tuples:
            nums = [int(x.strip()) for x in t.split(",")]
            result.append(tuple(nums))
        return result
    raise argparse.ArgumentTypeError(
        "Expected a list of tuples like '(M,N,K,L),(M,N,K,L)'"
    )


def _validate_problem_sizes_args(args, parser: argparse.ArgumentParser) -> None:
    if len(args.problem_sizes_mnkl) not in (0, args.num_groups):
        parser.error("--problem_sizes_mnkl must contain exactly --num_groups tuples")

    for _, _, _, l in args.problem_sizes_mnkl:
        if l != 1:
            parser.error("l (batch size) must be 1 for all groups")


def _resolve_tensormap_update_mode(
    mode: str, parser: argparse.ArgumentParser
) -> utils.TensorMapUpdateMode:
    if mode == "GMEM":
        return utils.TensorMapUpdateMode.GMEM
    if mode == "SMEM":
        return utils.TensorMapUpdateMode.SMEM
    parser.error("--tensormap_update_mode must be GMEM or SMEM")
    return utils.TensorMapUpdateMode.SMEM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hopper Grouped GEMM (CuTe DSL)")
    parser.add_argument("--num_groups", type=int, default=1, help="Number of groups")
    parser.add_argument(
        "--problem_sizes_mnkl",
        type=_parse_problem_sizes,
        default=((4096, 4096, 4096, 1),),
        help="Problem sizes per group, e.g. '(4096,4096,4096,1),(512,512,512,1)'",
    )
    parser.add_argument(
        "--tile_shape_mn",
        type=_parse_comma_separated_ints,
        choices=[(128, 128), (128, 256), (128, 64), (64, 64)],
        default=(128, 128),
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=_parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)],
        default=(1, 1),
    )
    parser.add_argument(
        "--tensormap_update_mode",
        type=str,
        choices=["GMEM", "SMEM"],
        default="SMEM",
        help="Tensor map update mode",
    )
    parser.add_argument("--a_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--b_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument("--a_major", choices=["k", "m"], default="k")
    parser.add_argument("--b_major", choices=["k", "n"], default="k")
    parser.add_argument("--c_major", choices=["n", "m"], default="n")
    parser.add_argument("--tolerance", type=float, default=1e-1)
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2", action="store_true", default=False, help="Use cold L2"
    )

    args = parser.parse_args()

    _validate_problem_sizes_args(args, parser)
    tensormap_update_mode = _resolve_tensormap_update_mode(
        args.tensormap_update_mode, parser
    )

    torch.manual_seed(2025)

    run(
        args.num_groups,
        args.problem_sizes_mnkl,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.tile_shape_mn,
        args.cluster_shape_mn,
        tensormap_update_mode,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
