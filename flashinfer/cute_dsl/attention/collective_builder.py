# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .mainloop_spec import MainloopSpec, MLAMainloopSpec


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
        qk_tiled_mma, config.qk_mma_tiler, q_dtype, mainloop.q_stages,
    )
    k_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma, config.qk_mma_tiler, k_dtype, mainloop.kv_stages,
    )
    p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
        pv_tiled_mma, config.pv_mma_tiler, q_dtype, mainloop.acc_stage,
    )
    v_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma, config.pv_mma_tiler, v_dtype, mainloop.kv_stages,
    )
    o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
        o_dtype, o_layout, epi_tile, mainloop.epi_stage,
    )

    # TMA atoms
    tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
    tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

    q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op, q, q_smem_layout,
        config.qk_mma_tiler, qk_tiled_mma, cluster_layout_vmnk.shape,
    )
    k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op, k, k_smem_layout,
        config.qk_mma_tiler, qk_tiled_mma, cluster_layout_vmnk.shape,
    )
    v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op, v, v_smem_layout,
        config.pv_mma_tiler, pv_tiled_mma, cluster_layout_vmnk.shape,
    )
    o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
    tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
        tma_store_op, o, o_smem_layout, epi_tile,
    )

    tma_copy_q_bytes = cute.size_in_bytes(q_dtype, q_smem_layout)
    tma_copy_kv_bytes = cute.size_in_bytes(k_dtype, k_smem_layout)

    # SharedStorage struct
    align = mainloop.buffer_align_bytes
    sched = mainloop.warp_schedule

    @cute.struct
    class SharedStorage:
        load_q_mbar_ptr: cute.struct.MemRange[Int64, mainloop.q_stages * 2]
        load_kv_mbar_ptr: cute.struct.MemRange[Int64, mainloop.kv_stages * 2]
        mma_s0_mbar_ptr: cute.struct.MemRange[Int64, mainloop.mma_softmax_stage * 2]
        mma_s1_mbar_ptr: cute.struct.MemRange[Int64, mainloop.mma_softmax_stage * 2]
        s0_corr_mbar_ptr: cute.struct.MemRange[Int64, mainloop.softmax_corr_stage * 2]
        s1_corr_mbar_ptr: cute.struct.MemRange[Int64, mainloop.softmax_corr_stage * 2]
        s0_s1_sequence_mbar_ptr: cute.struct.MemRange[Int64, sched.softmax_warpgroup_count]
        corr_epi_mbar_ptr: cute.struct.MemRange[Int64, mainloop.epi_stage * 2]
        mma_corr_mbar_ptr: cute.struct.MemRange[Int64, mainloop.mma_corr_stage * 2]
        tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
        tmem_holding_buf: cutlass.Int32
        sO: cute.struct.Align[
            cute.struct.MemRange[o_dtype, cute.cosize(o_smem_layout_staged)], align,
        ]
        sQ: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(q_smem_layout_staged)], align,
        ]
        sK: cute.struct.Align[
            cute.struct.MemRange[k_dtype, cute.cosize(k_smem_layout_staged)], align,
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


def build_mla_launch_params(
    mainloop: MLAMainloopSpec,
    schedule,
    q_latent: cute.Tensor,
    q_rope: cute.Tensor,
    c_latent: cute.Tensor,
    c_rope: cute.Tensor,
    c_latent_transpose: cute.Tensor,
    q_dtype,
    k_dtype,
    v_dtype,
) -> SimpleNamespace:
    """Build all MMA atoms, SMEM layouts, TMA atoms, and SharedStorage for MLA decode.

    :param mainloop: Resolved MLAMainloopSpec (resolve() must have been called).
    :param schedule: MLAWarpSchedule.
    :param q_latent: Query latent tensor.
    :param q_rope: Query RoPE tensor.
    :param c_latent: Compressed latent tensor.
    :param c_rope: Compressed RoPE tensor.
    :param c_latent_transpose: Transposed compressed latent for PV MMA.
    :param q_dtype: Element type of Q.
    :param k_dtype: Element type of K.
    :param v_dtype: Element type of V.
    :returns: SimpleNamespace with all derived objects needed for kernel launch.
    """
    config = mainloop.config

    cta_group = tcgen05.CtaGroup.TWO
    p_major_mode = tcgen05.OperandMajorMode.K
    q_major_mode = tcgen05.OperandMajorMode.K
    k_major_mode = tcgen05.OperandMajorMode.K
    v_major_mode = tcgen05.OperandMajorMode.MN

    qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        q_dtype, q_major_mode, k_major_mode,
        config.acc_dtype, cta_group, config.mma_qk_tiler[:2],
    )
    pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        v_dtype, p_major_mode, v_major_mode,
        config.acc_dtype, cta_group, config.mma_pv_tiler[:2],
    )

    cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout(config.cluster_shape_mnk),
        (qk_tiled_mma.thr_id.shape,),
    )
    epi_tile = config.mma_pv_tiler[:2]

    # SMEM layouts
    q_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_tiled_mma, config.mma_qk_tiler, q_dtype, mainloop.load_q_stages,
    )
    kc_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma, config.mma_qk_tiler, k_dtype, mainloop.load_kv_stages,
    )
    p_smem_layout_staged = sm100_utils.make_smem_layout_a(
        pv_tiled_mma, config.mma_pv_tiler, q_dtype,
        (config.iterations_pv_k * mainloop.p_mma_stages),
    )
    p_smem_layout_staged = cute.logical_divide(
        p_smem_layout_staged, (None, None, None, config.iterations_pv_k),
    )
    vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma, config.mma_pv_tiler, v_dtype, mainloop.load_kv_stages,
    )

    # TMA atoms
    tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

    q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op, q_latent, q_smem_layout,
        config.mma_qk_tiler, qk_tiled_mma, cta_layout_vmnk.shape,
    )
    tma_atom_q_rope, tma_tensor_q_rope = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op, q_rope, q_smem_layout,
        config.mma_qk_tiler, qk_tiled_mma, cta_layout_vmnk.shape,
    )
    kc_smem_layout = cute.select(kc_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_c_latent, tma_tensor_c_latent = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op, c_latent, kc_smem_layout,
        config.mma_qk_tiler, qk_tiled_mma, cta_layout_vmnk.shape,
    )
    tma_atom_c_rope, tma_tensor_c_rope = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op, c_rope, kc_smem_layout,
        config.mma_qk_tiler, qk_tiled_mma, cta_layout_vmnk.shape,
    )
    vc_smem_layout = cute.select(vc_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_c_latent_transpose, tma_tensor_c_latent_transpose = (
        cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, c_latent_transpose, vc_smem_layout,
            config.mma_pv_tiler, pv_tiled_mma, cta_layout_vmnk.shape,
        )
    )

    q_copy_size = cute.size_in_bytes(q_dtype, q_smem_layout) * cute.size(
        qk_tiled_mma.thr_id.shape
    )
    kc_copy_size = cute.size_in_bytes(k_dtype, kc_smem_layout) * cute.size(
        qk_tiled_mma.thr_id.shape
    )
    vc_copy_size = cute.size_in_bytes(v_dtype, vc_smem_layout) * cute.size(
        pv_tiled_mma.thr_id.shape
    )
    assert kc_copy_size == vc_copy_size, "kc_copy_size and vc_copy_size must be the same"

    # SharedStorage struct
    @cute.struct
    class SplitKVKernelSharedStorage:
        load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, mainloop.load_q_stages * 2]
        load_kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, mainloop.load_kv_stages * 2]
        mma_s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, mainloop.mma_s_stages * 2]
        p_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, mainloop.p_mma_stages * 2]
        mma_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, mainloop.mma_o_stages * 2]
        load_pt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, mainloop.load_pt_stages * 2]
        smem_exchange: cute.struct.MemRange[
            config.acc_dtype, schedule.num_compute_warps * schedule.threads_per_warp
        ]
        smem_page_table: cute.struct.MemRange[
            cutlass.Int32, mainloop.load_pt_stages * config.mma_qk_tiler[1]
        ]
        smem_q: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(q_smem_layout_staged)], 1024,
        ]
        smem_kc: cute.struct.Align[
            cute.struct.MemRange[k_dtype, cute.cosize(kc_smem_layout_staged)], 1024,
        ]
        smem_p: cute.struct.Align[
            cute.struct.MemRange[q_dtype, cute.cosize(p_smem_layout_staged)], 1024,
        ]
        tmem_dealloc_mbar_ptr: cutlass.Int64
        tmem_holding_buf: cutlass.Int32

        @classmethod
        def size_in_bytes(cls) -> int: ...  # noqa: F811

    return SimpleNamespace(
        qk_tiled_mma=qk_tiled_mma,
        pv_tiled_mma=pv_tiled_mma,
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
        q_smem_layout_staged=q_smem_layout_staged,
        kc_smem_layout_staged=kc_smem_layout_staged,
        p_smem_layout_staged=p_smem_layout_staged,
        vc_smem_layout_staged=vc_smem_layout_staged,
        SharedStorage=SplitKVKernelSharedStorage,
        tma_copy_q_bytes=q_copy_size,
        tma_copy_kc_bytes=kc_copy_size,
        cta_layout_vmnk=cta_layout_vmnk,
        epi_tile=epi_tile,
    )
