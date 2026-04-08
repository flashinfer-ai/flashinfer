# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CollectiveBuilder — factory functions for kernel launch infrastructure.

Analogous to C++ CUTLASS's CollectiveBuilder templates, these functions
select MMA atoms, create SMEM layouts, TMA descriptors, and SharedStorage
structs based on the MainloopSpec and input tensor types.

This separates "what to compute" (roles, config) from "how to set up
hardware" (MMA atoms, TMA, shared memory), keeping the kernel __call__
focused on wiring and launch.
"""

from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int64

from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
import cutlass.cute.nvgpu.cpasync as cpasync

from .mainloop_spec import MainloopSpec, MLAMainloopSpec
from .mla_warp_schedule import MLAWarpSchedule, MLAWarpScheduleFP8


def build_fmha_launch_params(
    mainloop: MainloopSpec,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    q_dtype,
    k_dtype,
    v_dtype,
    o_dtype,
    q_major_mode,
    k_major_mode,
    v_major_mode,
    o_layout,
) -> SimpleNamespace:
    """Build all MMA atoms, SMEM layouts, TMA atoms, and SharedStorage for FMHA prefill.

    :param mainloop: Resolved MainloopSpec (resolve() must have been called).
    :param q: Query tensor.
    :param k: Key tensor.
    :param v: Value tensor.
    :param o: Output tensor.
    :param q_dtype: Element type of Q.
    :param k_dtype: Element type of K.
    :param v_dtype: Element type of V.
    :param o_dtype: Element type of O.
    :param q_major_mode: MMA major mode for Q operand.
    :param k_major_mode: MMA major mode for K operand.
    :param v_major_mode: MMA major mode for V operand.
    :param o_layout: Layout enum for output.
    :returns: SimpleNamespace with all derived objects needed for kernel launch.
    """
    config = mainloop.config

    cta_group = tcgen05.CtaGroup.ONE
    p_major_mode = tcgen05.OperandMajorMode.K
    p_source = tcgen05.OperandSource.TMEM

    qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        q_dtype,
        q_major_mode,
        k_major_mode,
        config.qk_acc_dtype,
        cta_group,
        config.qk_mma_tiler[:2],
    )
    pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        v_dtype,
        p_major_mode,
        v_major_mode,
        config.pv_acc_dtype,
        cta_group,
        config.pv_mma_tiler[:2],
        p_source,
    )

    cluster_shape_mnk = (*config.cluster_shape_mn, 1)
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout(cluster_shape_mnk),
        (qk_tiled_mma.thr_id.shape,),
    )
    epi_tile = config.pv_mma_tiler[:2]

    # SMEM layouts
    q_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_tiled_mma,
        config.qk_mma_tiler,
        q_dtype,
        mainloop.q_stages,
    )
    k_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma,
        config.qk_mma_tiler,
        k_dtype,
        mainloop.kv_stages,
    )
    p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
        pv_tiled_mma,
        config.pv_mma_tiler,
        q_dtype,
        mainloop.acc_stage,
    )
    v_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma,
        config.pv_mma_tiler,
        v_dtype,
        mainloop.kv_stages,
    )
    o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
        o_dtype,
        o_layout,
        epi_tile,
        mainloop.epi_stage,
    )

    # TMA atoms
    tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
    tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

    q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        q,
        q_smem_layout,
        config.qk_mma_tiler,
        qk_tiled_mma,
        cluster_layout_vmnk.shape,
    )
    k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op,
        k,
        k_smem_layout,
        config.qk_mma_tiler,
        qk_tiled_mma,
        cluster_layout_vmnk.shape,
    )
    v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op,
        v,
        v_smem_layout,
        config.pv_mma_tiler,
        pv_tiled_mma,
        cluster_layout_vmnk.shape,
    )
    o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
    tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
        tma_store_op,
        o,
        o_smem_layout,
        epi_tile,
    )

    tma_copy_q_bytes = cute.size_in_bytes(q_dtype, q_smem_layout)
    tma_copy_kv_bytes = cute.size_in_bytes(k_dtype, k_smem_layout)

    # SharedStorage struct
    align = mainloop.buffer_align_bytes
    sched = mainloop.warp_schedule

    # Minimize barrier storage for unused paths
    s0_corr_stages = (
        mainloop.softmax_corr_stage if not mainloop.has_logits_transform else 1
    )
    mma_corr_stages = (
        mainloop.mma_corr_stage if not mainloop.has_logits_transform else 1
    )
    s0_epi_stages = mainloop.epi_stage if mainloop.has_logits_transform else 1

    @cute.struct
    class SharedStorage:
        load_q_mbar_ptr: cute.struct.MemRange[Int64, mainloop.q_stages * 2]
        load_kv_mbar_ptr: cute.struct.MemRange[Int64, mainloop.kv_stages * 2]
        mma_s0_mbar_ptr: cute.struct.MemRange[Int64, mainloop.mma_softmax_stage * 2]
        mma_s1_mbar_ptr: cute.struct.MemRange[Int64, mainloop.mma_softmax_stage * 2]
        s0_corr_mbar_ptr: cute.struct.MemRange[Int64, s0_corr_stages * 2]
        s1_corr_mbar_ptr: cute.struct.MemRange[Int64, s0_corr_stages * 2]
        s0_s1_sequence_mbar_ptr: cute.struct.MemRange[
            Int64, sched.softmax_warpgroup_count
        ]
        corr_epi_mbar_ptr: cute.struct.MemRange[Int64, mainloop.epi_stage * 2]
        mma_corr_mbar_ptr: cute.struct.MemRange[Int64, mma_corr_stages * 2]
        s0_epi_mbar_ptr: cute.struct.MemRange[Int64, s0_epi_stages * 2]
        s1_epi_mbar_ptr: cute.struct.MemRange[Int64, s0_epi_stages * 2]
        tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
        tmem_holding_buf: cutlass.Int32
        sO: cute.struct.Align[
            cute.struct.MemRange[o_dtype, cute.cosize(o_smem_layout_staged)],
            align,
        ]
        sQ: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(q_smem_layout_staged)],
            align,
        ]
        sK: cute.struct.Align[
            cute.struct.MemRange[k_dtype, cute.cosize(k_smem_layout_staged)],
            align,
        ]

        @classmethod
        def size_in_bytes(cls) -> int: ...  # noqa: F811

    return SimpleNamespace(
        qk_tiled_mma=qk_tiled_mma,
        pv_tiled_mma=pv_tiled_mma,
        tma_atom_q=tma_atom_q,
        tma_tensor_q=tma_tensor_q,
        tma_atom_k=tma_atom_k,
        tma_tensor_k=tma_tensor_k,
        tma_atom_v=tma_atom_v,
        tma_tensor_v=tma_tensor_v,
        tma_atom_o=tma_atom_o,
        tma_tensor_o=tma_tensor_o,
        q_smem_layout_staged=q_smem_layout_staged,
        k_smem_layout_staged=k_smem_layout_staged,
        p_tmem_layout_staged=p_tmem_layout_staged,
        v_smem_layout_staged=v_smem_layout_staged,
        o_smem_layout_staged=o_smem_layout_staged,
        SharedStorage=SharedStorage,
        tma_copy_q_bytes=tma_copy_q_bytes,
        tma_copy_kv_bytes=tma_copy_kv_bytes,
        cluster_shape_mnk=cluster_shape_mnk,
        cluster_layout_vmnk=cluster_layout_vmnk,
        epi_tile=epi_tile,
        o_layout=o_layout,
    )


def make_paged_tiled_tma_atom(
    tma_load_op: cpasync.CopyBulkTensorTileG2SOp,
    gmem: cute.Tensor,
    smem_layout: cute.Layout,
    mma_tiler,
    tiled_mma: cute.TiledMma,
    page_size: int,
    is_k_load: bool,
):
    """Create a paged TMA atom for tiled memory access with page table indirection.

    Extracted from the monolithic MLA kernel's make_paged_tiled_tma_atom method.
    Builds a non-executable TMA descriptor that tiles the global memory tensor
    into page-aligned chunks for paged KV cache access.

    :param tma_load_op: TMA copy operation (G2S with CTA group).
    :param gmem: Global memory tensor to create the TMA descriptor for.
    :param smem_layout: Shared memory layout for the TMA tile.
    :param mma_tiler: MMA tile dimensions (M, K) or (M, N).
    :param tiled_mma: The TiledMma atom used for CTA partitioning.
    :param page_size: Number of tokens per page in the KV cache.
    :param is_k_load: True for K-operand loads, False for V-operand loads.
    :returns: Tuple of (CopyAtom, TMA tensor descriptor).
    """
    ident = cute.make_identity_layout(gmem.shape)
    g_tile = cute.composition(ident, mma_tiler)
    cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
    cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
    cta_v_map = cute.select(cta_v_map, mode=[0, 2])
    page_tile_size = (
        min(page_size, cta_mn) if is_k_load else min(page_size, mma_tiler[1])
    )
    cta_v_map = cute.zipped_divide(
        cta_v_map,
        (page_tile_size, mma_tiler[1]) if is_k_load else (cta_mn, page_tile_size),
    )
    cta_v_map = cute.select(cta_v_map, mode=[0])
    from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir

    res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
        gmem.value,
        smem_layout.value,
        cta_v_map,
        tma_load_op._to_ir(),
        num_multicast=1,
    )
    return cute.CopyAtom(
        tma_load_op, cpasync.CopyBulkTensorTileG2SNonExecTrait(res[0])
    ), res[1]


def build_mla_launch_params(
    mainloop: MLAMainloopSpec,
    schedule: MLAWarpSchedule,
    q_latent: cute.Tensor,
    q_rope: cute.Tensor,
    c_latent: cute.Tensor,
    c_rope: cute.Tensor,
    c_latent_transpose: cute.Tensor,
    page_table: cute.Tensor,
    o: cute.Tensor,
    lse: cute.Tensor,
    acc_o: cute.Tensor,
    acc_lse: cute.Tensor,
    q_dtype,
    k_dtype,
    v_dtype,
    o_dtype,
) -> SimpleNamespace:
    """Build all MMA atoms, SMEM layouts, TMA atoms, and SharedStorage for MLA decode.

    :param mainloop: Resolved MLAMainloopSpec.
    :param schedule: MLAWarpSchedule with warp role assignments.
    :param q_latent: Query latent tensor (reinterpreted as [H, D, S_q, B]).
    :param q_rope: Query RoPE tensor (reinterpreted as [H, D, S_q, B]).
    :param c_latent: KV latent tensor (reinterpreted as [page_size, D, num_pages]).
    :param c_rope: KV RoPE tensor (reinterpreted as [page_size, D, num_pages]).
    :param c_latent_transpose: Transposed KV latent (reinterpreted as [D, page_size, num_pages]).
    :param page_table: Page table tensor.
    :param o: Output tensor.
    :param lse: LSE tensor.
    :param acc_o: Accumulator output tensor (for split-KV).
    :param acc_lse: Accumulator LSE tensor (for split-KV).
    :param q_dtype: Element type of Q.
    :param k_dtype: Element type of K.
    :param v_dtype: Element type of V.
    :param o_dtype: Element type of O.
    :returns: SimpleNamespace with all derived objects needed for kernel launch.
    """
    config = mainloop.config

    cta_group = tcgen05.CtaGroup.TWO
    q_major_mode = tcgen05.OperandMajorMode.K
    k_major_mode = tcgen05.OperandMajorMode.K
    v_major_mode = tcgen05.OperandMajorMode.MN
    p_major_mode = tcgen05.OperandMajorMode.K

    qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        q_dtype,
        q_major_mode,
        k_major_mode,
        config.acc_dtype,
        cta_group,
        config.mma_qk_tiler[:2],
    )
    pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        v_dtype,
        p_major_mode,
        v_major_mode,
        config.acc_dtype,
        cta_group,
        config.mma_pv_tiler[:2],
    )

    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(config.cluster_shape_mnk),
        (qk_tiled_mma.thr_id.shape,),
    )
    epi_tile = config.mma_pv_tiler[:2]

    # SMEM layouts
    q_latent_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_tiled_mma,
        config.mma_qk_tiler,
        q_dtype,
        (config.iterations_qk_latent * config.load_q_stage),
    )
    q_latent_smem_layout_staged = cute.logical_divide(
        q_latent_smem_layout_staged,
        (None, None, None, config.iterations_qk_latent),
    )
    q_rope_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_tiled_mma,
        config.mma_qk_rope_tiler,
        q_dtype,
        config.load_q_stage,
    )

    kc_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma,
        config.mma_qk_tiler,
        k_dtype,
        config.load_kv_stage,
    )
    kc_page_tile_size = min(
        config.page_size,
        qk_tiled_mma.op.shape_mnk[0] // qk_tiled_mma.thr_id.shape,
    )
    kc_smem_layout_for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.K,
        (config.mma_qk_tiler[0] // qk_tiled_mma.thr_id.shape, config.mma_qk_tiler[2]),
        k_dtype,
        config.load_kv_stage,
    )
    kc_smem_layout_for_tma = cute.tiled_divide(
        kc_smem_layout_for_tma, (kc_page_tile_size, config.mma_qk_tiler[2])
    )

    p_smem_layout_staged = sm100_utils.make_smem_layout_a(
        pv_tiled_mma,
        config.mma_pv_tiler,
        q_dtype,
        (config.iterations_pv_k * config.p_mma_stage),
    )
    p_smem_layout_staged = cute.logical_divide(
        p_smem_layout_staged, (None, None, None, config.iterations_pv_k)
    )

    vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma,
        config.mma_pv_tiler,
        v_dtype,
        config.load_kv_stage,
    )
    vc_page_tile_size = min(config.page_size, config.mma_pv_tiler[2])
    vc_smem_layout_for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.MN,
        (config.mma_pv_tiler[1] // pv_tiled_mma.thr_id.shape, config.mma_pv_tiler[2]),
        v_dtype,
        config.load_kv_stage,
    )
    vc_smem_layout_for_tma = cute.tiled_divide(
        vc_smem_layout_for_tma,
        (
            pv_tiled_mma.op.shape_mnk[1] // pv_tiled_mma.thr_id.shape,
            vc_page_tile_size,
        ),
    )

    # TMA atoms
    tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

    q_latent_smem_layout = cute.select(q_latent_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        q_latent,
        q_latent_smem_layout,
        config.mma_qk_tiler,
        qk_tiled_mma,
        cta_layout_vmnk.shape,
    )
    q_rope_smem_layout = cute.select(q_rope_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q_rope, tma_tensor_q_rope = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        q_rope,
        q_rope_smem_layout,
        config.mma_qk_rope_tiler,
        qk_tiled_mma,
        cta_layout_vmnk.shape,
    )

    kc_smem_layout = cute.select(kc_smem_layout_for_tma, mode=[0])
    tma_atom_c_latent, tma_tensor_c_latent = make_paged_tiled_tma_atom(
        tma_load_op,
        c_latent,
        kc_smem_layout,
        (config.mma_qk_tiler[1], config.mma_qk_tiler[2]),
        qk_tiled_mma,
        config.page_size,
        is_k_load=True,
    )
    tma_atom_c_rope, tma_tensor_c_rope = make_paged_tiled_tma_atom(
        tma_load_op,
        c_rope,
        kc_smem_layout,
        (config.mma_qk_tiler[1], config.mma_qk_tiler[2]),
        qk_tiled_mma,
        config.page_size,
        is_k_load=True,
    )

    vc_smem_layout = cute.select(vc_smem_layout_for_tma, mode=[0])
    tma_atom_c_latent_transpose, tma_tensor_c_latent_transpose = (
        make_paged_tiled_tma_atom(
            tma_load_op,
            c_latent_transpose,
            vc_smem_layout,
            (config.mma_pv_tiler[1], config.mma_pv_tiler[2]),
            pv_tiled_mma,
            config.page_size,
            is_k_load=False,
        )
    )

    # Copy sizes
    q_latent_copy_size = (
        cute.size_in_bytes(q_dtype, q_latent_smem_layout)
        * cute.size(qk_tiled_mma.thr_id.shape)
        * config.iterations_qk_latent
    )
    q_rope_copy_size = (
        cute.size_in_bytes(q_dtype, q_rope_smem_layout)
        * cute.size(qk_tiled_mma.thr_id.shape)
        * config.iterations_qk_rope
    )
    tma_copy_q_bytes = q_latent_copy_size + q_rope_copy_size
    tma_copy_kc_bytes = cute.size_in_bytes(
        k_dtype, cute.select(kc_smem_layout_staged, mode=[0, 1, 2])
    ) * cute.size(qk_tiled_mma.thr_id.shape)

    # SharedStorage struct
    align = mainloop.buffer_align_bytes
    threads_per_warp = schedule.threads_per_warp
    num_compute_warps = config.num_compute_warps

    @cute.struct
    class SplitKVKernelSharedStorage:
        load_q_mbar_ptr: cute.struct.MemRange[Int64, config.load_q_stage * 2]
        load_kv_mbar_ptr: cute.struct.MemRange[Int64, config.load_kv_stage * 2]
        mma_s_mbar_ptr: cute.struct.MemRange[Int64, config.mma_s_stage * 2]
        p_mma_mbar_ptr: cute.struct.MemRange[Int64, config.p_mma_stage * 2]
        p_cor_mbar_ptr: cute.struct.MemRange[Int64, config.p_cor_stage * 2]
        mma_o_mbar_ptr: cute.struct.MemRange[Int64, config.mma_o_stage * 2]
        load_pt_mbar_ptr: cute.struct.MemRange[Int64, config.load_pt_stage * 2]
        tmem_dealloc_mbar_ptr: Int64
        tmem_holding_buf: cutlass.Int32
        softmax_smem_exchange: cute.struct.MemRange[
            config.acc_dtype, num_compute_warps * threads_per_warp
        ]
        epilogue_smem_exchange: cute.struct.MemRange[
            config.acc_dtype, num_compute_warps * threads_per_warp
        ]
        smem_q_latent: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(q_latent_smem_layout_staged)],
            align,
        ]
        smem_q_rope: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(q_rope_smem_layout_staged)],
            align,
        ]
        smem_kc: cute.struct.Align[
            cute.struct.MemRange[k_dtype, cute.cosize(kc_smem_layout_staged)],
            align,
        ]
        smem_p: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(p_smem_layout_staged)],
            align,
        ]
        smem_page_table: cute.struct.MemRange[
            cutlass.Int32, config.load_pt_stage * config.mma_qk_tiler[1] // 2
        ]

        @classmethod
        def size_in_bytes(cls) -> int: ...  # noqa: F811

    return SimpleNamespace(
        qk_tiled_mma=qk_tiled_mma,
        pv_tiled_mma=pv_tiled_mma,
        q_latent_smem_layout_staged=q_latent_smem_layout_staged,
        q_rope_smem_layout_staged=q_rope_smem_layout_staged,
        kc_smem_layout_staged=kc_smem_layout_staged,
        p_smem_layout_staged=p_smem_layout_staged,
        vc_smem_layout_staged=vc_smem_layout_staged,
        kc_smem_layout_for_tma=kc_smem_layout_for_tma,
        vc_smem_layout_for_tma=vc_smem_layout_for_tma,
        tma_atom_q_latent=tma_atom_q_latent,
        tma_tensor_q_latent=tma_tensor_q_latent,
        tma_atom_q_rope=tma_atom_q_rope,
        tma_tensor_q_rope=tma_tensor_q_rope,
        tma_atom_c_latent=tma_atom_c_latent,
        tma_tensor_c_latent=tma_tensor_c_latent,
        tma_atom_c_rope=tma_atom_c_rope,
        tma_tensor_c_rope=tma_tensor_c_rope,
        tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
        tma_tensor_c_latent_transpose=tma_tensor_c_latent_transpose,
        kc_page_tile_size=kc_page_tile_size,
        vc_page_tile_size=vc_page_tile_size,
        tma_copy_q_bytes=tma_copy_q_bytes,
        tma_copy_kc_bytes=tma_copy_kc_bytes,
        SharedStorage=SplitKVKernelSharedStorage,
        cta_layout_vmnk=cta_layout_vmnk,
        epi_tile=epi_tile,
        cluster_shape_mnk=config.cluster_shape_mnk,
    )


def build_mla_fp8_launch_params(
    mainloop: MLAMainloopSpec,
    schedule: MLAWarpScheduleFP8,
    q_latent: cute.Tensor,
    q_rope: cute.Tensor,
    c_latent: cute.Tensor,
    c_rope: cute.Tensor,
    c_latent_transpose: cute.Tensor,
    page_table: cute.Tensor,
    o: cute.Tensor,
    lse: cute.Tensor,
    acc_o: cute.Tensor,
    acc_lse: cute.Tensor,
    q_dtype,
    k_dtype,
    v_dtype,
    o_dtype,
) -> SimpleNamespace:
    """Build MMA atoms, SMEM layouts, TMA atoms, and SharedStorage for FP8 MLA decode.

    FP8 differs from FP16 in:
    - Separate KC-latent, KC-rope, and VC SMEM buffers (no aliasing)
    - KC-latent stages use logical_divide for iterations_qk_latent
    - VC stages use nested logical_divide for iterations_pv_k * iterations_pv_n
    - No page-table SMEM buffer or load_pt barriers
    - Separate tma_copy_kc_bytes and tma_copy_vc_bytes
    """
    config = mainloop.config

    cta_group = tcgen05.CtaGroup.TWO
    q_major_mode = OperandMajorMode.K
    k_major_mode = OperandMajorMode.K
    v_major_mode = OperandMajorMode.MN
    p_major_mode = OperandMajorMode.K

    qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        q_dtype,
        q_major_mode,
        k_major_mode,
        config.acc_dtype,
        cta_group,
        config.mma_qk_tiler[:2],
    )
    pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        v_dtype,
        p_major_mode,
        v_major_mode,
        config.acc_dtype,
        cta_group,
        config.mma_pv_tiler[:2],
    )

    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(config.cluster_shape_mnk),
        (qk_tiled_mma.thr_id.shape,),
    )
    epi_tile = config.mma_pv_tiler[:2]

    # --- Q SMEM layouts (same structure as FP16) ---
    q_latent_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_tiled_mma,
        config.mma_qk_tiler,
        q_dtype,
        (config.iterations_qk_latent * config.load_q_stage),
    )
    q_latent_smem_layout_staged = cute.logical_divide(
        q_latent_smem_layout_staged,
        (None, None, None, config.iterations_qk_latent),
    )
    q_rope_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_tiled_mma,
        config.mma_qk_rope_tiler,
        q_dtype,
        config.load_q_stage,
    )

    # --- KC-latent SMEM: separate buffer with logical_divide for latent iterations ---
    kc_latent_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma,
        config.mma_qk_tiler,
        k_dtype,
        (config.iterations_qk_latent * config.load_k_stage),
    )
    kc_page_tile_size = min(
        config.page_size,
        qk_tiled_mma.op.shape_mnk[0] // qk_tiled_mma.thr_id.shape,
    )
    kc_latent_smem_layout_staged = cute.logical_divide(
        kc_latent_smem_layout_staged,
        (None, None, None, config.iterations_qk_latent),
    )

    kc_latent_smem_layout_for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.K,
        (config.mma_qk_tiler[0] // qk_tiled_mma.thr_id.shape, config.mma_qk_tiler[2]),
        k_dtype,
        (config.iterations_qk_latent * config.load_k_stage),
    )
    kc_latent_smem_layout_for_tma = cute.tiled_divide(
        kc_latent_smem_layout_for_tma,
        (kc_page_tile_size, config.mma_qk_tiler[2]),
    )
    kc_latent_smem_layout_for_tma = cute.logical_divide(
        kc_latent_smem_layout_for_tma,
        (None, None, None, config.iterations_qk_latent),
    )

    # --- KC-rope SMEM: separate buffer ---
    kc_rope_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma,
        config.mma_qk_rope_tiler,
        k_dtype,
        config.load_k_stage,
    )
    kc_rope_smem_layout_for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.K,
        (
            config.mma_qk_rope_tiler[0] // qk_tiled_mma.thr_id.shape,
            config.mma_qk_rope_tiler[2],
        ),
        k_dtype,
        (config.iterations_qk_rope * config.load_k_stage),
    )
    kc_rope_smem_layout_for_tma = cute.tiled_divide(
        kc_rope_smem_layout_for_tma,
        (kc_page_tile_size, config.mma_qk_rope_tiler[2]),
    )

    # --- P SMEM layout ---
    p_smem_layout_staged = sm100_utils.make_smem_layout_a(
        pv_tiled_mma,
        config.mma_pv_tiler,
        q_dtype,
        (config.iterations_pv_k * config.p_mma_stage),
    )
    p_smem_layout_staged = cute.logical_divide(
        p_smem_layout_staged,
        (None, None, None, config.iterations_pv_k),
    )

    # --- VC SMEM: separate buffer with nested logical_divide ---
    vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma,
        config.mma_pv_tiler,
        v_dtype,
        (config.iterations_pv_k * config.iterations_pv_n * config.load_v_stage),
    )
    vc_smem_layout_staged = cute.logical_divide(
        cute.logical_divide(
            vc_smem_layout_staged,
            (None, None, None, config.iterations_pv_k * config.iterations_pv_n),
        ),
        (None, None, None, (config.iterations_pv_n, None)),
    )
    vc_page_tile_size = min(config.page_size, config.mma_pv_tiler[2])
    vc_smem_layout_for_tma = sm100_utils.make_smem_layout(
        OperandMajorMode.MN,
        (config.mma_pv_tiler[1] // pv_tiled_mma.thr_id.shape, config.mma_pv_tiler[2]),
        v_dtype,
        (config.iterations_pv_k * config.iterations_pv_n * config.load_v_stage),
    )
    vc_smem_layout_for_tma = cute.tiled_divide(
        vc_smem_layout_for_tma,
        (
            pv_tiled_mma.op.shape_mnk[1] // pv_tiled_mma.thr_id.shape,
            vc_page_tile_size,
        ),
    )
    vc_smem_layout_for_tma = cute.logical_divide(
        cute.logical_divide(
            vc_smem_layout_for_tma,
            (None, None, None, config.iterations_pv_k * config.iterations_pv_n),
        ),
        (None, None, None, (config.iterations_pv_n, None)),
    )

    # --- TMA atoms ---
    tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

    q_smem_layout = cute.select(q_latent_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        q_latent,
        q_smem_layout,
        config.mma_qk_tiler,
        qk_tiled_mma,
        cta_layout_vmnk.shape,
    )
    q_rope_smem_layout = cute.select(q_rope_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q_rope, tma_tensor_q_rope = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        q_rope,
        q_rope_smem_layout,
        config.mma_qk_rope_tiler,
        qk_tiled_mma,
        cta_layout_vmnk.shape,
    )

    kc_smem_layout = cute.select(kc_latent_smem_layout_for_tma, mode=[0])
    tma_atom_c_latent, tma_tensor_c_latent = make_paged_tiled_tma_atom(
        tma_load_op,
        c_latent,
        kc_smem_layout,
        (config.mma_qk_tiler[1], config.mma_qk_tiler[2]),
        qk_tiled_mma,
        config.page_size,
        is_k_load=True,
    )
    kc_rope_smem_layout = cute.select(kc_rope_smem_layout_for_tma, mode=[0])
    tma_atom_c_rope, tma_tensor_c_rope = make_paged_tiled_tma_atom(
        tma_load_op,
        c_rope,
        kc_rope_smem_layout,
        (config.mma_qk_rope_tiler[1], config.mma_qk_rope_tiler[2]),
        qk_tiled_mma,
        config.page_size,
        is_k_load=True,
    )

    vc_smem_layout = cute.select(vc_smem_layout_for_tma, mode=[0])
    tma_atom_c_latent_transpose, tma_tensor_c_latent_transpose = (
        make_paged_tiled_tma_atom(
            tma_load_op,
            c_latent_transpose,
            vc_smem_layout,
            (config.mma_pv_tiler[1], config.mma_pv_tiler[2]),
            pv_tiled_mma,
            config.page_size,
            is_k_load=False,
        )
    )

    # --- Copy sizes ---
    q_latent_copy_size = (
        cute.size_in_bytes(q_dtype, q_smem_layout)
        * cute.size(qk_tiled_mma.thr_id.shape)
        * config.iterations_qk_latent
    )
    q_rope_copy_size = (
        cute.size_in_bytes(q_dtype, q_rope_smem_layout)
        * cute.size(qk_tiled_mma.thr_id.shape)
        * config.iterations_qk_rope
    )
    tma_copy_q_bytes = q_latent_copy_size + q_rope_copy_size

    kc_latent_copy_size = (
        cute.size_in_bytes(
            k_dtype,
            cute.select(kc_latent_smem_layout_staged, mode=[0, 1, 2]),
        )
        * cute.size(qk_tiled_mma.thr_id.shape)
        * config.iterations_qk_latent
    )
    kc_rope_copy_size = (
        cute.size_in_bytes(
            k_dtype,
            cute.select(kc_rope_smem_layout_staged, mode=[0, 1, 2]),
        )
        * cute.size(qk_tiled_mma.thr_id.shape)
        * config.iterations_qk_rope
    )
    tma_copy_kc_bytes = kc_latent_copy_size + kc_rope_copy_size

    tma_copy_vc_bytes = (
        cute.size_in_bytes(
            v_dtype,
            cute.select(vc_smem_layout_staged, mode=[0, 1, 2]),
        )
        * cute.size(pv_tiled_mma.thr_id.shape)
        * config.iterations_pv_n
        * config.iterations_pv_k
    )

    # --- SharedStorage struct (no page-table buffer) ---
    align = mainloop.buffer_align_bytes
    threads_per_warp = schedule.threads_per_warp
    num_compute_warps = config.num_compute_warps

    @cute.struct
    class FP8SplitKVKernelSharedStorage:
        load_q_mbar_ptr: cute.struct.MemRange[Int64, config.load_q_stage * 2]
        load_k_mbar_ptr: cute.struct.MemRange[Int64, config.load_k_stage * 2]
        load_v_mbar_ptr: cute.struct.MemRange[Int64, config.load_v_stage * 2]
        mma_s_mbar_ptr: cute.struct.MemRange[Int64, config.mma_s_stage * 2]
        p_mma_mbar_ptr: cute.struct.MemRange[Int64, config.p_mma_stage * 2]
        p_cor_mbar_ptr: cute.struct.MemRange[Int64, config.p_cor_stage * 2]
        mma_o_mbar_ptr: cute.struct.MemRange[Int64, config.mma_o_stage * 2]

        smem_p: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(p_smem_layout_staged)],
            align,
        ]
        smem_kc_latent: cute.struct.Align[
            cute.struct.MemRange[k_dtype, cute.cosize(kc_latent_smem_layout_staged)],
            align,
        ]
        smem_kc_rope: cute.struct.Align[
            cute.struct.MemRange[k_dtype, cute.cosize(kc_rope_smem_layout_staged)],
            align,
        ]
        smem_q_latent: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(q_latent_smem_layout_staged)],
            align,
        ]
        smem_q_rope: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(q_rope_smem_layout_staged)],
            align,
        ]
        smem_vc: cute.struct.Align[
            cute.struct.MemRange[v_dtype, cute.cosize(vc_smem_layout_staged)],
            align,
        ]
        softmax_smem_exchange: cute.struct.MemRange[
            config.acc_dtype, num_compute_warps * threads_per_warp
        ]
        epilogue_smem_exchange: cute.struct.MemRange[
            config.acc_dtype, num_compute_warps * threads_per_warp
        ]
        tmem_dealloc_mbar_ptr: Int64
        tmem_holding_buf: cutlass.Int32

        @classmethod
        def size_in_bytes(cls) -> int: ...  # noqa: F811

    return SimpleNamespace(
        qk_tiled_mma=qk_tiled_mma,
        pv_tiled_mma=pv_tiled_mma,
        q_latent_smem_layout_staged=q_latent_smem_layout_staged,
        q_rope_smem_layout_staged=q_rope_smem_layout_staged,
        kc_latent_smem_layout_staged=kc_latent_smem_layout_staged,
        kc_rope_smem_layout_staged=kc_rope_smem_layout_staged,
        p_smem_layout_staged=p_smem_layout_staged,
        vc_smem_layout_staged=vc_smem_layout_staged,
        kc_latent_smem_layout_for_tma=kc_latent_smem_layout_for_tma,
        kc_rope_smem_layout_for_tma=kc_rope_smem_layout_for_tma,
        vc_smem_layout_for_tma=vc_smem_layout_for_tma,
        tma_atom_q_latent=tma_atom_q_latent,
        tma_tensor_q_latent=tma_tensor_q_latent,
        tma_atom_q_rope=tma_atom_q_rope,
        tma_tensor_q_rope=tma_tensor_q_rope,
        tma_atom_c_latent=tma_atom_c_latent,
        tma_tensor_c_latent=tma_tensor_c_latent,
        tma_atom_c_rope=tma_atom_c_rope,
        tma_tensor_c_rope=tma_tensor_c_rope,
        tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
        tma_tensor_c_latent_transpose=tma_tensor_c_latent_transpose,
        kc_page_tile_size=kc_page_tile_size,
        vc_page_tile_size=vc_page_tile_size,
        tma_copy_q_bytes=tma_copy_q_bytes,
        tma_copy_kc_bytes=tma_copy_kc_bytes,
        tma_copy_vc_bytes=tma_copy_vc_bytes,
        SharedStorage=FP8SplitKVKernelSharedStorage,
        cta_layout_vmnk=cta_layout_vmnk,
        epi_tile=epi_tile,
        cluster_shape_mnk=config.cluster_shape_mnk,
    )
