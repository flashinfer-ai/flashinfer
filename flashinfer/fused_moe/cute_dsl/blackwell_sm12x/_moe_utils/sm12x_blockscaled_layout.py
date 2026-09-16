# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SM120 per-variant scale-factor configs for FP8, MXFP8 x MXFP4, and NVFP4."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils

from .....utils import ceil_div
from .....utils import round_up as align

TMA_ALIGN_BYTES = 16
SF_ELEM_BYTES = 4
SF_M_ALIGN = TMA_ALIGN_BYTES // SF_ELEM_BYTES
UE8M0_PACK_NUM = 4


def compute_padded_offset(offset, expert_idx, alignment):
    return (offset + expert_idx * (alignment - 1)) // alignment * alignment


@cute.jit
def _make_nvfp4_smem_layout_sfa(tiledmma, tile, sf_vec, stages):
    block_m, block_sf = 128, 4
    block_elems = block_m * block_sf
    mma_sf = tiledmma.shape_mnk[2] // sf_vec
    sfa_tile_m = max(block_m, ceil_div(tile[0], block_m) * block_m)
    shape_m = ((32, 4), sfa_tile_m // block_m)
    stride_m = ((16, 4), block_elems)
    shape_k = ((sf_vec, mma_sf), block_sf // mma_sf, tile[2] // sf_vec // block_sf)
    stride_k = ((0, 1), mma_sf, sfa_tile_m // block_m * block_elems)
    layout = cute.make_layout((shape_m, shape_k), stride=(stride_m, stride_k))
    return cute.append(
        layout, cute.make_layout(stages, stride=cute.cosize(cute.filter_zeros(layout)))
    )


@cute.jit
def _make_nvfp4_smem_layout_sfb(tiledmma, tile, sf_vec, stages):
    block_n, block_sf = 128, 4
    block_elems = block_n * block_sf
    mma_sf = tiledmma.shape_mnk[2] // sf_vec
    sfb_tile_n = max(block_n, ceil_div(tile[1], block_n) * block_n)
    shape_n = ((32, 4), sfb_tile_n // block_n)
    stride_n = ((16, 4), block_elems)
    shape_k = ((sf_vec, mma_sf), block_sf // mma_sf, tile[2] // sf_vec // block_sf)
    stride_k = ((0, 1), mma_sf, sfb_tile_n // block_n * block_elems)
    layout = cute.make_layout((shape_n, shape_k), stride=(stride_n, stride_k))
    return cute.append(
        layout, cute.make_layout(stages, stride=cute.cosize(cute.filter_zeros(layout)))
    )


class Sm120SfConfigNvfp4:
    SF_VEC = 16

    def __init__(self, sf_dtype=cutlass.Float8E4M3FN):
        self.sf_dtype = sf_dtype
        self.use_ue8m0 = False
        self.sf_load_warp = 1

    def ab_stages_contract(self, ab_stage):
        return ab_stage > 0

    def k_tiles_per_pack_a(self, tile_k):
        assert tile_k % self.SF_VEC == 0
        return 1

    def k_tiles_per_pack_b(self, tile_k):
        assert tile_k % self.SF_VEC == 0
        return 1

    def sfa_stages(self, ab_stage, tile_k):
        self.k_tiles_per_pack_a(tile_k)
        return ab_stage

    def sfb_stages(self, ab_stage, tile_k):
        self.k_tiles_per_pack_b(tile_k)
        return ab_stage

    def smem_bytes(self, tile, ab_stage):
        bm, bn, bk = tile
        return (max(128, bm) + max(128, bn)) * ceil_div(bk, self.SF_VEC) * ab_stage

    def gate_smem_bytes(self, tile, ab_stage, swap_ab):
        bm, bn, bk = tile
        mn = bm if swap_ab else bn
        return max(128, mn) * ceil_div(bk, self.SF_VEC) * ab_stage

    def sfa_tile_m(self, tile_m):
        return max(128, ceil_div(tile_m, 128) * 128)

    def sfa_tiles_per_block(self, tile_m):
        return self.sfa_tile_m(tile_m) // tile_m

    def sfb_tile_n(self, tile_n):
        return max(128, ceil_div(tile_n, 128) * 128)

    def sfb_tiles_per_block(self, tile_n):
        return self.sfb_tile_n(tile_n) // tile_n

    def deduce_sfa_layout(self, mn, K, L):
        return blockscaled_utils.tile_atom_to_shape_SF((mn, K, L), self.SF_VEC)

    def deduce_sfb_layout(self, mn, K, L):
        return blockscaled_utils.tile_atom_to_shape_SF((mn, K, L), self.SF_VEC)

    def make_smem_layout_sfa(self, tiledmma, tile, ab_stage):
        return _make_nvfp4_smem_layout_sfa(tiledmma, tile, self.SF_VEC, ab_stage)

    def make_smem_layout_sfb(self, tiledmma, tile, ab_stage):
        return _make_nvfp4_smem_layout_sfb(tiledmma, tile, self.SF_VEC, ab_stage)

    def thrfrg_SFA(self, sf_layout, tiled_mma):
        atom_shape_mnk = tiled_mma.shape_mnk
        perm = tiled_mma.permutation_mnk
        thr_vmnk = tiled_mma.thr_layout_vmnk
        atom_sf_layout = cute.make_layout(((2, 2, 8), 64), stride=((8, 0, 1), 16))
        t_tensor = cute.logical_divide(sf_layout, (perm[0], perm[2]))
        a_tensor = cute.zipped_divide(
            t_tensor,
            (cute.make_layout(atom_shape_mnk[0]), cute.make_layout(atom_shape_mnk[2])),
        )
        tv_tensor = cute.composition(a_tensor, (atom_sf_layout, None))
        return cute.zipped_divide(
            tv_tensor,
            (
                None,
                (
                    cute.make_layout(cute.size(thr_vmnk[1])),
                    cute.make_layout(cute.size(thr_vmnk[3])),
                ),
            ),
        )

    def thrfrg_SFB(self, sf_layout, tiled_mma):
        atom_shape_mnk = tiled_mma.shape_mnk
        perm = tiled_mma.permutation_mnk
        thr_vmnk = tiled_mma.thr_layout_vmnk
        atom_sf_layout = cute.make_layout(((4, 8), 64), stride=((0, 1), 8))
        t_tensor = cute.logical_divide(sf_layout, (perm[1], perm[2]))
        a_tensor = cute.zipped_divide(
            t_tensor,
            (cute.make_layout(atom_shape_mnk[1]), cute.make_layout(atom_shape_mnk[2])),
        )
        tv_tensor = cute.composition(a_tensor, (atom_sf_layout, None))
        return cute.zipped_divide(
            tv_tensor,
            (
                None,
                (
                    cute.make_layout(cute.size(thr_vmnk[2])),
                    cute.make_layout(cute.size(thr_vmnk[3])),
                ),
            ),
        )

    @staticmethod
    def _thr_partition(thr_tensor, thr_coord):
        rest = (None, tuple([None] * cute.rank(thr_tensor.layout, mode=[1, 1])))
        return thr_tensor[thr_coord, rest]

    def partition_fragment_SFA(self, sf_tensor, thr_mma, tidx):
        thr_tensor = cute.make_tensor(
            sf_tensor.iterator, self.thrfrg_SFA(sf_tensor.layout, thr_mma)
        )
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vmk = (thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3]))
        return cute.make_fragment_like(self._thr_partition(thr_tensor, thr_vmk))

    def partition_fragment_SFB(self, sf_tensor, thr_mma, tidx):
        thr_tensor = cute.make_tensor(
            sf_tensor.iterator, self.thrfrg_SFB(sf_tensor.layout, thr_mma)
        )
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
        return cute.make_fragment_like(self._thr_partition(thr_tensor, thr_vnk))

    def get_layoutSFA_TV(self, tiledmma):
        perm = tiledmma.permutation_mnk
        thr_vmnk = tiledmma.thr_layout_vmnk
        ref = cute.make_layout((cute.size(perm[0]), cute.size(perm[2])))
        atile = (
            None,
            (
                cute.make_layout(
                    shape=(cute.size(thr_vmnk[1]), cute.size(thr_vmnk[2])),
                    stride=(1, 0),
                ),
                None,
            ),
        )
        tv = cute.composition(self.thrfrg_SFA(ref, tiledmma), (atile, None))
        return cute.composition(tv, (cute.right_inverse(thr_vmnk), None))

    def get_layoutSFB_TV(self, tiledmma):
        perm = tiledmma.permutation_mnk
        thr_vmnk = tiledmma.thr_layout_vmnk
        ref = cute.make_layout((cute.size(perm[1]), cute.size(perm[2])))
        atile = (
            None,
            (
                cute.make_layout(
                    shape=(cute.size(thr_vmnk[1]), cute.size(thr_vmnk[2])),
                    stride=(0, 1),
                ),
                None,
            ),
        )
        tv = cute.composition(self.thrfrg_SFB(ref, tiledmma), (atile, None))
        return cute.composition(tv, (cute.right_inverse(thr_vmnk), None))

    def make_s2r_sf(self, tv_layout, tiler):
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
        return cute.make_tiled_copy(atom, tv_layout, tiler)


class Sm120SfConfigMxfp8Mxfp4:
    KTILE_SF = 1
    PACK_NSF = 4
    SF_VEC = 32

    def __init__(self, grank_a, grank_b):
        assert grank_a % grank_b == 0 or grank_b % grank_a == 0, (
            f"SF granularities must nest: neither {grank_a} nor {grank_b} divides the other, so the "
            f"coarser side's pack does not hold a whole number of the finer side's"
        )
        self.sf_dtype = cutlass.Int32
        self.use_ue8m0 = True
        self.grank_a = grank_a
        self.grank_b = grank_b
        self.sf_load_warp = 1

    def _k_tiles_per_pack(self, grank, tile_k):
        pack_nk = grank * self.PACK_NSF
        assert pack_nk % tile_k == 0, (
            "grank*PACK_NSF must be a multiple of tile_k (int32 SF pack aligns to tile-K)"
        )
        return pack_nk // tile_k

    def k_tiles_per_pack_a(self, tile_k):
        return self._k_tiles_per_pack(self.grank_a, tile_k)

    def k_tiles_per_pack_b(self, tile_k):
        return self._k_tiles_per_pack(self.grank_b, tile_k)

    def _sf_stages(self, ab_stage, k_tiles_per_pack):
        return max(1, ceil_div(ab_stage, k_tiles_per_pack))

    def ab_stages_contract(self, ab_stage):
        return ab_stage & (ab_stage - 1) == 0

    def sfa_stages(self, ab_stage, tile_k):
        return self._sf_stages(ab_stage, self.k_tiles_per_pack_a(tile_k))

    def sfb_stages(self, ab_stage, tile_k):
        return self._sf_stages(ab_stage, self.k_tiles_per_pack_b(tile_k))

    def _stage_stride(self, mn):
        return max(mn, 128 * 8 // self.sf_dtype.width)

    def _smem_layout(self, mn, sf_stages):
        return cute.make_layout(
            (mn, self.KTILE_SF, sf_stages), stride=(1, mn, self._stage_stride(mn))
        )

    def _cosize(self, mn, sf_stages):
        return (sf_stages - 1) * self._stage_stride(mn) + mn

    def smem_bytes(self, tile, ab_stage):
        bm, bn, bk = tile
        packs = self._cosize(bm, self.sfa_stages(ab_stage, bk)) + self._cosize(
            bn, self.sfb_stages(ab_stage, bk)
        )
        return packs * (self.sf_dtype.width // 8)

    def gate_smem_bytes(self, tile, ab_stage, swap_ab):
        bm, bn, bk = tile
        mn, stages = (
            (bm, self.sfa_stages(ab_stage, bk))
            if swap_ab
            else (bn, self.sfb_stages(ab_stage, bk))
        )
        return self._cosize(mn, stages) * (self.sf_dtype.width // 8)

    def make_smem_layout_sfa(self, tile_m, sf_stages):
        return self._smem_layout(tile_m, sf_stages)

    def make_smem_layout_sfb(self, tile_n, sf_stages):
        return self._smem_layout(tile_n, sf_stages)

    def make_s2r_sf(self, tv_layout, tiler):
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
        return cute.make_tiled_copy(atom, tv_layout, tiler)

    def _deduce_layout(self, mn, K, L, grank):
        pack_nk = grank * self.PACK_NSF
        scale_k = ceil_div(K, pack_nk)
        mn = align(mn, SF_M_ALIGN)
        return cute.make_layout((mn, scale_k, L), stride=(1, mn, mn * scale_k))

    def deduce_sfa_layout(self, mn, K, L):
        return self._deduce_layout(mn, K, L, self.grank_a)

    def deduce_sfb_layout(self, mn, K, L):
        return self._deduce_layout(mn, K, L, self.grank_b)

    def thrfrg_SFA(self, sf_layout, tiled_mma):
        atom_shape_mnk = tiled_mma.shape_mnk
        perm = tiled_mma.permutation_mnk
        thr_vmnk = tiled_mma.thr_layout_vmnk
        atom_sf_layout = cute.make_layout(
            ((2, 2, 8), self.KTILE_SF), stride=((8, 0, 1), 16)
        )
        t_tile = (perm[0], cute.make_layout(self.KTILE_SF))
        t_tensor = cute.logical_divide(sf_layout, t_tile)
        a_tile = (cute.make_layout(atom_shape_mnk[0]), cute.make_layout(self.KTILE_SF))
        a_tensor = cute.zipped_divide(t_tensor, a_tile)
        tv_tensor = cute.composition(a_tensor, (atom_sf_layout, None))
        thr_tile = (
            None,
            (
                cute.make_layout(cute.size(thr_vmnk[1])),
                cute.make_layout(cute.size(thr_vmnk[3])),
            ),
        )
        return cute.zipped_divide(tv_tensor, thr_tile)

    @staticmethod
    def _thr_partition(thr_tensor, thr_coord):
        rest = (None, tuple([None] * cute.rank(thr_tensor.layout, mode=[1, 1])))
        return thr_tensor[thr_coord, rest]

    def partition_fragment_SFA(self, sf_tensor, thr_mma, tidx):
        thr_tensor = cute.make_tensor(
            sf_tensor.iterator, self.thrfrg_SFA(sf_tensor.layout, thr_mma)
        )
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vmk = (thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3]))
        return cute.make_rmem_tensor_like(self._thr_partition(thr_tensor, thr_vmk))

    def get_layoutSFA_TV(self, tiled_mma):
        perm = tiled_mma.permutation_mnk
        thr_vmnk = tiled_mma.thr_layout_vmnk
        ref = cute.make_layout((cute.size(perm[0]), self.KTILE_SF))
        atile = (
            None,
            (
                cute.make_layout(
                    shape=(cute.size(thr_vmnk[1]), cute.size(thr_vmnk[2])),
                    stride=(1, 0),
                ),
                None,
            ),
        )
        thrfrg = self.thrfrg_SFA(ref, tiled_mma)
        tv = cute.composition(thrfrg, (atile, None))
        return cute.composition(tv, (cute.right_inverse(thr_vmnk), None))

    def make_sf_ue8m0_view(self, frg_int32, grank, tile_k):
        ue8m0_ptr = cute.recast_ptr(frg_int32.iterator, dtype=cutlass.Float8E8M0FNU)
        num_mn = cute.size(frg_int32.layout, mode=[1])
        pack_nk = grank * self.PACK_NSF
        atoms_per_sf = min(grank, tile_k) // self.SF_VEC
        num_sf_per_tk = tile_k // grank if tile_k >= grank else 1
        num_tk_per_sf = grank // tile_k if grank > tile_k else 1
        num_tk_groups = (pack_nk // tile_k) // num_tk_per_sf
        view = cute.make_layout(
            (
                self.SF_VEC,
                num_mn,
                (atoms_per_sf, num_sf_per_tk),
                (num_tk_per_sf, num_tk_groups),
            ),
            stride=(0, self.PACK_NSF, (0, 1), (0, num_sf_per_tk)),
        )
        return cute.make_tensor(ue8m0_ptr, view)

    def make_sfa_ue8m0_view(self, frg_int32, tile_k):
        return self.make_sf_ue8m0_view(frg_int32, self.grank_a, tile_k)

    def thrfrg_SFB(self, sf_layout, tiled_mma):
        atom_shape_mnk = tiled_mma.shape_mnk
        perm = tiled_mma.permutation_mnk
        thr_vmnk = tiled_mma.thr_layout_vmnk
        atom_sf_layout = cute.make_layout(((4, 8), self.KTILE_SF), stride=((0, 1), 8))
        t_tile = (perm[1], cute.make_layout(self.KTILE_SF))
        t_tensor = cute.logical_divide(sf_layout, t_tile)
        a_tile = (cute.make_layout(atom_shape_mnk[1]), cute.make_layout(self.KTILE_SF))
        a_tensor = cute.zipped_divide(t_tensor, a_tile)
        tv_tensor = cute.composition(a_tensor, (atom_sf_layout, None))
        thr_tile = (
            None,
            (
                cute.make_layout(cute.size(thr_vmnk[2])),
                cute.make_layout(cute.size(thr_vmnk[3])),
            ),
        )
        return cute.zipped_divide(tv_tensor, thr_tile)

    def partition_fragment_SFB(self, sf_tensor, thr_mma, tidx):
        thr_tensor = cute.make_tensor(
            sf_tensor.iterator, self.thrfrg_SFB(sf_tensor.layout, thr_mma)
        )
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
        return cute.make_rmem_tensor_like(self._thr_partition(thr_tensor, thr_vnk))

    def get_layoutSFB_TV(self, tiled_mma):
        perm = tiled_mma.permutation_mnk
        thr_vmnk = tiled_mma.thr_layout_vmnk
        ref = cute.make_layout((cute.size(perm[1]), self.KTILE_SF))
        btile = (
            None,
            (
                cute.make_layout(
                    shape=(cute.size(thr_vmnk[1]), cute.size(thr_vmnk[2])),
                    stride=(0, 1),
                ),
                None,
            ),
        )
        thrfrg = self.thrfrg_SFB(ref, tiled_mma)
        tv = cute.composition(thrfrg, (btile, None))
        return cute.composition(tv, (cute.right_inverse(thr_vmnk), None))

    def make_sfb_ue8m0_view(self, frg_int32, tile_k):
        return self.make_sf_ue8m0_view(frg_int32, self.grank_b, tile_k)


class Sm120SfConfigFp8:
    NUM_SCALE_COPY_THREADS = 32

    def __init__(self, gran_m, gran_n, gran_k, tile_n=None):
        self.sf_dtype = cutlass.Float32
        self.use_ue8m0 = False
        self.gran_m = gran_m
        self.gran_n = gran_n
        self.gran_k = gran_k
        self.tile_n = tile_n
        self.sf_load_warp = 1

    def k_tiles_per_pack(self, tile_k):
        assert self.gran_k % tile_k == 0, (
            f"gran_k ({self.gran_k}) must be a multiple of tile_k ({tile_k}): an SF entry has to "
            f"cover a whole number of k-tiles"
        )
        return self.gran_k // tile_k

    def sf_stages(self, ab_stage, tile_k):
        return max(1, ceil_div(ab_stage, self.k_tiles_per_pack(tile_k)))

    def ab_stages_contract(self, ab_stage):
        return True

    def assert_k_invariant(self, tile_k):
        assert tile_k == self.gran_k, (
            f"tile_k ({tile_k}) must equal gran_k ({self.gran_k}): the 2D staged SF layout has no in-tile "
            f"K dim (kTileScaleK must be 1); use tile_k == gran_k or extend to a 3D K-scale layout."
        )

    def assert_mn_invariant(self, tile_m, tile_n):
        assert tile_m % self.gran_m == 0, (
            f"tile_m {tile_m} must cover whole granules of {self.gran_m}"
        )
        assert tile_n % self.gran_n == 0 or self.gran_n % tile_n == 0, (
            f"tile_n {tile_n} and gran_n {self.gran_n} must nest one way or the other"
        )

    def deduce_sfa_layout(self, M, K, L):
        scale_m = ceil_div(M, self.gran_m)
        scale_k = ceil_div(K, self.gran_k)
        return cute.make_layout(
            (scale_m, scale_k, L), stride=(1, scale_m, scale_m * scale_k)
        )

    def deduce_sfb_layout(self, N, K, L):
        scale_n = ceil_div(N, self.gran_n)
        scale_k = ceil_div(K, self.gran_k)
        return cute.make_layout(
            (scale_n, scale_k, L), stride=(1, scale_n, scale_n * scale_k)
        )

    def make_smem_layout_sfa(self, tile_m, stages):
        tscale_m = ceil_div(tile_m, self.gran_m)
        return cute.make_layout((tscale_m, stages), stride=(1, tscale_m))

    def make_smem_layout_sfb(self, tile_n, stages):
        tscale_n = ceil_div(tile_n, self.gran_n)
        return cute.make_layout((tscale_n, stages), stride=(1, tscale_n))

    def smem_bytes(self, tile, ab_stage):
        bm, bn, _ = tile
        entries = (ceil_div(bm, self.gran_m) + ceil_div(bn, self.gran_n)) * ab_stage
        return entries * (self.sf_dtype.width // 8)

    def gate_smem_bytes(self, tile, ab_stage, swap_ab):
        bm, bn, _ = tile
        mn, gran = (bm, self.gran_m) if swap_ab else (bn, self.gran_n)
        return ceil_div(mn, gran) * ab_stage * (self.sf_dtype.width // 8)

    def make_tma_smem_layout_sfa(self, tile_m, stages):
        tscale_m = ceil_div(tile_m, self.gran_m)
        return cute.make_layout((tscale_m, 1, stages), stride=(1, tscale_m, tscale_m))

    def partition_scale_as_c(self, sSFA, sSFB, thr_mma):
        tscale_m, stages = cute.size(sSFA, mode=[0]), cute.size(sSFA, mode=[1])
        tscale_n = cute.size(sSFB, mode=[0])
        tile_m = self.gran_m * tscale_m
        tile_n = self.gran_n * tscale_n if self.tile_n is None else self.tile_n
        per_m, per_n = tile_m // tscale_m, tile_n // tscale_n
        sSFA_vc = cute.make_tensor(
            sSFA.iterator,
            cute.make_layout(
                ((per_m, tscale_m), tile_n, stages), stride=((0, 1), 0, tscale_m)
            ),
        )
        sSFB_vc = cute.make_tensor(
            sSFB.iterator,
            cute.make_layout(
                (tile_m, (per_n, tscale_n), stages), stride=(0, (0, 1), tscale_n)
            ),
        )
        tCsSFA = thr_mma.partition_C(sSFA_vc)
        tCsSFB = thr_mma.partition_C(sSFB_vc)
        tCrSFA = cute.make_rmem_tensor_like(tCsSFA[None, None, None, 0])
        tCrSFB = cute.make_rmem_tensor_like(tCsSFB[None, None, None, 0])
        return tCsSFA, tCsSFB, tCrSFA, tCrSFB


@cute.jit
def copy_scale_s2r(stage, tCsSFA, tCsSFB, tCrSFA, tCrSFB, tscale_mn):
    tscale_m, tscale_n = tscale_mn
    cute.autovec_copy(tCsSFA[None, None, None, stage], tCrSFA)
    cute.autovec_copy(tCsSFB[None, None, None, stage], tCrSFB)
    if tscale_m == 1 and tscale_n == 1:
        tCrSFA[0] = tCrSFA[0] * tCrSFB[0]
    elif tscale_m > 1 and tscale_n == 1:
        tCrSFA.store(tCrSFA.load() * tCrSFB.load()[0])
    elif tscale_m == 1 and tscale_n > 1:
        tCrSFB.store(tCrSFB.load() * tCrSFA.load()[0])


@cute.jit
def rescale(acc, tmp, tCrSFA, tCrSFB, tscale_mn):
    tscale_m, tscale_n = tscale_mn
    if tscale_m == 1 and tscale_n == 1:
        acc.store(acc.load() + tmp.load() * tCrSFA.load()[0])
    elif tscale_m > 1 and tscale_n == 1:
        acc.store(acc.load() + tmp.load() * tCrSFA.load())
    elif tscale_m == 1 and tscale_n > 1:
        acc.store(acc.load() + tmp.load() * tCrSFB.load())
    else:
        acc.store(acc.load() + tmp.load() * tCrSFA.load() * tCrSFB.load())
    tmp.fill(0.0)
