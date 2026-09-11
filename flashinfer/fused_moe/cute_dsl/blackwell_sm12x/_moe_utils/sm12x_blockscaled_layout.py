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
"""SM120 per-variant scale-factor configs (fp8 float-scale, mxfp8 x mxfp4 UE8M0), each self-contained."""

import cutlass
import cutlass.cute as cute

from .....utils import round_up as align, ceil_div

TMA_ALIGN_BYTES = 16
SF_ELEM_BYTES = 4
SF_M_ALIGN = TMA_ALIGN_BYTES // SF_ELEM_BYTES
UE8M0_PACK_NUM = 4


def compute_padded_offset(offset, expert_idx, alignment):
    return (offset + expert_idx * (alignment - 1)) // alignment * alignment


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
