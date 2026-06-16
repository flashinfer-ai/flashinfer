import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from .helpers import SM80, select_tensor_10


# ─── CollectiveInverse ───────────────────────────────────────────────────────
# Translates flat::collective::CollectiveInverse<Element, GarbageFilledDiagonal,
# GarbageFilledUpperTriangular>.
# Inverts a 64×64 lower-triangular smem matrix in-place (I + lower_tril(K)).
# Warp group: 128 threads (4 warps). Layout of sT: (64,64) col-major.


class CollectiveInverse:
    def __init__(
        self,
        garbage_filled_diagonal: bool = True,
        garbage_filled_upper_triangular: bool = False,
    ):
        self.garbage_filled_diagonal = garbage_filled_diagonal
        self.garbage_filled_upper_triangular = garbage_filled_upper_triangular

    # ── Level 1: NxN Gauss elimination on diagonal blocks ───────────────────────

    @cute.jit
    def compute_diagonal_inverse_NxN(
        self,
        mat: cute.Tensor,
        tid_in_group: cutlass.Int32,
        N: cutlass.Constexpr,
    ):
        """Invert one NxN lower-triangular block in-place.
        N threads participate; each owns its own row."""
        MASK_N = (N - 1) | ((32 - N) << 8)

        # Match C++ CollectiveInverse: vectorize the row load/store. Scalar U16
        # indexing here creates excessive shared-memory wavefronts on the KK buffer.
        row = cute.make_rmem_tensor(N, cutlass.Float16)
        cute.autovec_copy(mat[tid_in_group, None], row)

        # Apply I + lower_tril masking in fp32.
        frag = cute.make_rmem_tensor(N, cutlass.Float32)
        for j in cutlass.range_constexpr(N):
            raw = cutlass.Float32(row[j])
            if cutlass.const_expr(
                self.garbage_filled_diagonal or self.garbage_filled_upper_triangular
            ):
                if tid_in_group == cutlass.Int32(j):
                    frag[j] = cutlass.Float32(1.0)
                elif tid_in_group < cutlass.Int32(j):
                    frag[j] = cutlass.Float32(0.0)
                else:
                    frag[j] = raw
            else:
                frag[j] = raw

        # Gaussian elimination: row-reduce to produce inv(I + lower_tril(K))
        for src_row in cutlass.range_constexpr(N - 1):
            row_scale = -frag[src_row]
            for i in cutlass.range_constexpr(src_row):
                src_val = cute.arch.shuffle_sync(
                    frag[i], cutlass.Int32(src_row), mask_and_clamp=MASK_N
                )
                if tid_in_group > cutlass.Int32(src_row):
                    frag[i] = frag[i] + row_scale * src_val
            if tid_in_group > cutlass.Int32(src_row):
                frag[src_row] = row_scale

        row_out = cute.make_rmem_tensor(N, cutlass.Float16)
        for j in cutlass.range_constexpr(N):
            row_out[j] = cutlass.Float16(frag[j])
        cute.autovec_copy(row_out, mat[tid_in_group, None])

    # ── Level 2: 8×8 blocks → 16×16 blockwise inverse ───────────────────────────
    # Translates blockwise_diagonal_inversed_8x8_to_16x16 (col-major path).
    # Called by all 32 threads of one warp.
    # mat: (16, 16) smem slice (one 16×16 diagonal block).

    @cute.jit
    def blockwise_8x8_to_16x16(self, mat: cute.Tensor):
        lane_id = cute.arch.lane_idx()

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 8)),
            (1, 1, 1),
            permutation_mnk=(16, 8, 8),
        )

        mat_2x2 = cute.flat_divide(mat, (8, 8))
        sDinv = mat_2x2[None, None, 1, 1]
        sC = select_tensor_10(mat_2x2[None, None, 1, 0])
        sAinv = select_tensor_10(mat_2x2[None, None, 0, 0])
        sO = mat_2x2[None, None, 1, 0]

        # Broadcast sDinv (8,8) → (16,8) by stride-0 on the extra M dimension
        sDinv_bcast = cute.make_tensor(
            sDinv.iterator,
            cute.make_layout(
                ((cute.size(sDinv, mode=[0]), 2), cute.size(sDinv, mode=[1])),
                stride=((sDinv.layout.stride[0], 0), sDinv.layout.stride[1]),
            ),
        )
        sO_bcast = cute.make_tensor(
            sO.iterator,
            cute.make_layout(
                ((cute.size(sO, mode=[0]), 2), cute.size(sO, mode=[1])),
                stride=((sO.layout.stride[0], 0), sO.layout.stride[1]),
            ),
        )

        thr_mma = tiled_mma.get_slice(lane_id)
        tOrDinv = thr_mma.make_fragment_A(thr_mma.partition_A(sDinv_bcast))
        tOrC = thr_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = thr_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((16, 8)), cutlass.Float32
        )
        tOrO = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((16, 8)), cutlass.Float32
        )

        # Row-major copy atoms: A→LdMatrix_N, B→LdMatrix_T, C→StMatrix_N
        dinv_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=1), cutlass.Float16
        )
        b_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=1), cutlass.Float16
        )
        o_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=1), cutlass.Float16
        )

        D_tiled_copy = cute.make_tiled_copy_A(dinv_atom, tiled_mma)
        C_tiled_copy = cute.make_tiled_copy_B(b_atom, tiled_mma)
        A_tiled_copy = cute.make_tiled_copy_B(b_atom, tiled_mma)
        O_tiled_copy = cute.make_tiled_copy_C(o_atom, tiled_mma)

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        tOsDinv = D_thr_copy.partition_S(sDinv_bcast)
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO_bcast)
        tOrO_f16 = cute.make_fragment_like(tOrO, cutlass.Float16)
        tOrO_cv = O_thr_copy.retile(tOrO_f16)

        # ── Step 1: tDCrDC = -inv(D) @ C ─────────────────────────────────────────
        # Load only the first MMA_M slice of D_inv (rows 0-7 of the 16-row broadcast)
        cute.copy(D_tiled_copy, tOsDinv[None, None, 0], tOrDinv_cv[None, None, 0])
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)
        for i in cutlass.range_constexpr(cute.size(tDCrDC)):
            tDCrDC[i] = -tDCrDC[i]

        # ── Step 2: tOrO = tDCrDC @ inv(A) ───────────────────────────────────────
        tOrDC = SM80.make_acc_into_op(tDCrDC, tiled_mma, cutlass.Float16)

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        # ── Write output (first MMA_M slice → sO rows 0-7) ───────────────────────
        tOrO_f16.store(tOrO.load().to(cutlass.Float16))
        cute.copy(O_tiled_copy, tOrO_cv[None, None, 0], tOsO[None, None, 0])

    # ── Level 3: 16×16 blocks → 32×32 blockwise inverse ─────────────────────────
    # Called by one warp (thread_idx 0-31 or 32-63, i.e. thread_idx<64 in the WG).
    # mat: (32, 32) smem slice.

    @cute.jit
    def blockwise_16x16_to_32x32(self, mat: cute.Tensor):
        lane_id = cute.arch.lane_idx()

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )

        mat_2x2 = cute.flat_divide(mat, (16, 16))
        sDinv = mat_2x2[None, None, 1, 1]
        sC = select_tensor_10(mat_2x2[None, None, 1, 0])
        sAinv = select_tensor_10(mat_2x2[None, None, 0, 0])
        sO = mat_2x2[None, None, 1, 0]

        thr_mma = tiled_mma.get_slice(lane_id)
        tOrDinv = thr_mma.make_fragment_A(thr_mma.partition_A(sDinv))
        tOrC = thr_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = thr_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((16, 16)), cutlass.Float32
        )
        tOrO = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((16, 16)), cutlass.Float32
        )

        dinv_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2), cutlass.Float16
        )
        b_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=2), cutlass.Float16
        )
        o_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2), cutlass.Float16
        )

        D_tiled_copy = cute.make_tiled_copy_A(dinv_atom, tiled_mma)
        C_tiled_copy = cute.make_tiled_copy_B(b_atom, tiled_mma)
        A_tiled_copy = cute.make_tiled_copy_B(b_atom, tiled_mma)
        O_tiled_copy = cute.make_tiled_copy_C(o_atom, tiled_mma)

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        tOsDinv = D_thr_copy.partition_S(sDinv)
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO)
        tOrO_f16 = cute.make_fragment_like(tOrO, cutlass.Float16)
        tOrO_cv = O_thr_copy.retile(tOrO_f16)

        # ── Step 1: tDCrDC = -inv(D) @ C ─────────────────────────────────────────
        cute.copy(D_tiled_copy, tOsDinv, tOrDinv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)
        for i in cutlass.range_constexpr(cute.size(tDCrDC)):
            tDCrDC[i] = -tDCrDC[i]

        # ── Step 2: tOrO = tDCrDC @ inv(A) ───────────────────────────────────────
        tOrDC = SM80.make_acc_into_op(tDCrDC, tiled_mma, cutlass.Float16)

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_f16.store(tOrO.load().to(cutlass.Float16))
        cute.copy(O_tiled_copy, tOrO_cv, tOsO)

    # ── Level 4: 32×32 blocks → 64×64 blockwise inverse ─────────────────────────
    # Called by all 4 warps (128 threads).
    # sT: (64, 64) smem tensor (the full matrix).

    @cute.jit
    def blockwise_32x32_to_64x64(self, sT: cute.Tensor, barrier_id: cutlass.Int32):
        lane_id = cute.arch.lane_idx()
        warp_id = cute.arch.warp_idx() % 4  # WG-local warp ID 0..3
        x = warp_id // 2  # 0 or 1
        y = warp_id % 2  # 0 or 1

        # TiledMMA1: 16×16×32  (for -inv(D)@C, 1 K-pass of K=32)
        tiled_mma1 = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 32),
        )
        # TiledMMA2: 16×32×16  (for (-inv(D)@C)@inv(A), N=32 output)
        tiled_mma2 = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 32, 16),
        )

        mat_2x2 = cute.flat_divide(sT, (32, 32))
        mat_16x2_2x2 = cute.logical_divide(mat_2x2, (16, 16))

        # Per-warp tile slices (each warp handles one 16×32 or 32×16 sub-tile)
        sDinv = mat_16x2_2x2[(None, y), None, 1, 1]
        sC = select_tensor_10(mat_16x2_2x2[None, (None, x), 1, 0])
        sAinv = select_tensor_10(mat_16x2_2x2[(None, x), None, 0, 0])
        sO = mat_16x2_2x2[(None, y), None, 1, 0]

        thr_mma1 = tiled_mma1.get_slice(lane_id)
        thr_mma2 = tiled_mma2.get_slice(lane_id)

        tOrDinv = thr_mma1.make_fragment_A(thr_mma1.partition_A(sDinv))
        tOrC = thr_mma1.make_fragment_B(thr_mma1.partition_B(sC))
        tOrAinv = thr_mma2.make_fragment_B(thr_mma2.partition_B(sAinv))

        tDCrDC = cute.make_rmem_tensor(
            thr_mma1.partition_shape_C((16, 16)), cutlass.Float32
        )
        tOrO = cute.make_rmem_tensor(
            thr_mma2.partition_shape_C((16, 32)), cutlass.Float32
        )

        dinv_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.Float16
        )
        c_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cutlass.Float16
        )
        ainv_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=2), cutlass.Float16
        )
        O_atom_s2r = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.Float16
        )
        O_atom_r2s = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.Float16
        )

        D_tiled_copy = cute.make_tiled_copy_A(dinv_atom, tiled_mma1)
        C_tiled_copy = cute.make_tiled_copy_B(c_atom, tiled_mma1)
        A_tiled_copy = cute.make_tiled_copy_B(ainv_atom, tiled_mma2)
        O_tiled_s2r = cute.make_tiled_copy_C(O_atom_s2r, tiled_mma2)
        O_tiled_r2s = cute.make_tiled_copy_C(O_atom_r2s, tiled_mma2)

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_s2r = O_tiled_s2r.get_slice(lane_id)
        O_thr_r2s = O_tiled_r2s.get_slice(lane_id)

        tOsDinv = D_thr_copy.partition_S(sDinv)
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)

        # ── Step 1: tDCrDC = -inv(D) @ C ─────────────────────────────────────────
        cute.copy(D_tiled_copy, tOsDinv, tOrDinv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma1, tDCrDC, tOrDinv, tOrC, tDCrDC)
        for i in cutlass.range_constexpr(cute.size(tDCrDC)):
            tDCrDC[i] = -tDCrDC[i]

        # ── Step 2: tOrO = tDCrDC @ inv(A) ───────────────────────────────────────
        tOrDC = SM80.make_acc_into_op(tDCrDC, tiled_mma2, cutlass.Float16)

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma2, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_f16 = cute.make_fragment_like(tOrO, cutlass.Float16)
        tOrO_f16.store(tOrO.load().to(cutlass.Float16))

        # ── Cross-warp reduction: warps with x=0 write first, x=1 reads+adds+writes
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

        tOsO = O_thr_r2s.partition_D(sO)
        tOrO_cv = O_thr_r2s.retile(tOrO_f16)
        if x == 0:
            cute.copy(O_tiled_r2s, tOrO_cv, tOsO)

        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

        if x == 1:
            tOrO_red = cute.make_fragment_like(tOrO_f16)
            tOsO_s = O_thr_s2r.partition_S(sO)
            tOrO_red_cv = O_thr_s2r.retile(tOrO_red)
            cute.copy(O_tiled_s2r, tOsO_s, tOrO_red_cv)
            for i in cutlass.range_constexpr(cute.size(tOrO_f16)):
                tOrO_f16[i] = tOrO_f16[i] + tOrO_red[i]
            cute.copy(O_tiled_r2s, tOrO_cv, tOsO)

    @cute.jit
    def run(self, sT: cute.Tensor, barrier_id: cutlass.Int32):
        """Invert a 64×64 col-major smem tensor in-place.
        Must be called by 128 threads (1 warp group) with barrier_id free for use."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % 128

        # ── Level 1: 8×8 Gauss on diagonal 8×8 blocks (threads 0-63) ────────────
        t8x8 = cute.flat_divide(sT, (8, 8))
        if thread_idx < 64:
            blk = thread_idx // 8
            self.compute_diagonal_inverse_NxN(
                t8x8[None, None, blk, blk], thread_idx % 8, 8
            )

        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

        # ── Level 2: 16×16 blockwise inverse (all 4 warps → 4 diagonal blocks) ──
        t16x16 = cute.flat_divide(sT, (16, 16))
        blk2 = thread_idx // 32
        self.blockwise_8x8_to_16x16(t16x16[None, None, blk2, blk2])

        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

        # ── Level 3: 32×32 blockwise inverse (threads 0-63 → 2 diagonal blocks) ─
        t32x32 = cute.flat_divide(sT, (32, 32))
        if thread_idx < 64:
            blk3 = thread_idx // 32
            self.blockwise_16x16_to_32x32(t32x32[None, None, blk3, blk3])

        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

        # ── Level 4: 64×64 blockwise inverse (all 4 warps) ───────────────────────
        self.blockwise_32x32_to_64x64(sT, barrier_id)
