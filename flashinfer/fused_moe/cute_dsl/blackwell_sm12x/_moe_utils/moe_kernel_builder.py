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
"""SM120 block-scaled GEMM builder for the warp-specialized kernel family: composable per-phase configs + top builder."""

import cutlass
import cutlass.utils
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp.mma as warp_mma
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync

from .moe_scheduler import MoeSchedStages, MoeWorkTile

_SM12X_ARCHS = ("sm_120a", "sm_120f", "sm_121a", "sm_121f")


def dsl_targets_sm12x() -> bool:
    try:
        from cutlass.cutlass_dsl import CuTeDSL

        target = str(CuTeDSL._get_dsl().envar.arch)
    except Exception:
        return True
    return target in _SM12X_ARCHS


class MmaConfig:
    def __init__(self, op, mma_tile_mn, num_math_warps, swap_ab=False):
        self.op = op
        self.mma_tile_mn = tuple(mma_tile_mn)
        self.num_math_warps = num_math_warps
        self.swap_ab = swap_ab
        self.use_mxf8f6f4 = not isinstance(op, warp_mma.MmaMXF4NVF4Op)
        self.plain_fp8 = isinstance(op, warp_mma.MmaFP8Op)
        assert self.mma_tile_mn[1] // self.num_warp_n >= op.shape_mnk[1], (
            f"tile_n {self.mma_tile_mn[1]} over {self.num_warp_n} N-warps leaves each warp less "
            f"than the atom's N extent {op.shape_mnk[1]}"
        )

    @property
    def num_warp_m(self):
        if self.swap_ab:
            bn, atom_n = self.mma_tile_mn[1], self.op.shape_mnk[1]
            return self.num_math_warps // (4 if bn >= 32 else bn // atom_n)
        return 4 if self.mma_tile_mn[0] >= 64 else 2

    @property
    def num_warp_n(self):
        return self.num_math_warps // self.num_warp_m

    def mma_warp_layout(self):
        warp_m = self.num_warp_m
        return cute.make_ordered_layout(
            (warp_m, self.num_math_warps // warp_m, 1), order=(0, 1, 2)
        )

    @property
    def sf_vec(self):
        return self.op.shape_mnk[2] if self.plain_fp8 else self.op.sf_vec_size

    def make_tiled_mma(self, tile):
        if self.plain_fp8 or self.swap_ab:
            perm = (tile[0], tile[1], self.sf_vec)
        else:
            perm = sm120_utils.get_permutation_mnk(tile, self.sf_vec, self.use_mxf8f6f4)
        return cute.make_tiled_mma(
            self.op, self.mma_warp_layout(), permutation_mnk=perm
        )

    def make_s2r_a(self, tiledmma, smem_dtype, transpose, unpack_bits=None):
        if unpack_bits is None:
            ld = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
                smem_dtype,
            )
        else:
            ld = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x16x8bOp(
                    transpose=False, num_matrices=4, unpack_bits=unpack_bits
                ),
                smem_dtype,
            )
        return cute.make_tiled_copy_A(ld, tiledmma)

    @property
    def ldsm_matrices_b(self):
        per_warp_n = self.mma_tile_mn[1] // self.num_warp_n
        return 4 if per_warp_n >= 16 else (2 if per_warp_n >= 8 else 1)

    def make_s2r_b(self, tiledmma, smem_dtype, transpose, unpack_bits=None):
        num_matrices = self.ldsm_matrices_b
        if unpack_bits is None:
            ld = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    transpose=transpose, num_matrices=num_matrices
                ),
                smem_dtype,
            )
        else:
            ld = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x16x8bOp(
                    transpose=False, num_matrices=num_matrices, unpack_bits=unpack_bits
                ),
                smem_dtype,
            )
        return cute.make_tiled_copy_B(ld, tiledmma)

    def make_s2r_sf(self, sf_dtype, tv_layout, tiler):
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), sf_dtype)
        return cute.make_tiled_copy(atom, tv_layout, tiler)


class LoadABConfig:
    def __init__(
        self,
        tile,
        ab_stage,
        a_dtype,
        b_dtype,
        a_smem_dtype=None,
        b_smem_dtype=None,
        a_tma_internal=None,
        b_tma_internal=None,
        a_unpack_bits=None,
        b_unpack_bits=None,
    ):
        self.tile, self.ab_stage = tuple(tile), ab_stage
        self.a_dtype, self.b_dtype = a_dtype, b_dtype
        self.a_smem_dtype = a_smem_dtype or a_dtype
        self.b_smem_dtype = b_smem_dtype or b_dtype
        self.a_tma_internal, self.b_tma_internal = a_tma_internal, b_tma_internal
        self.a_unpack_bits, self.b_unpack_bits = a_unpack_bits, b_unpack_bits

        bm, bn, bk = self.tile
        self.tma_box_a, self.tma_box_b = (bm, bk), (bn, bk)
        self.tma_bytes_a = bm * bk * a_dtype.width // 8
        self.tma_bytes_b = bn * bk * b_dtype.width // 8
        self.tma_bytes_ab = self.tma_bytes_a + self.tma_bytes_b

    def _smem_layout(self, box, smem_dtype):
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, smem_dtype, box[1]
            ),
            smem_dtype,
        )
        return cute.tile_to_shape(atom, (*box, self.ab_stage), order=(0, 1, 2))

    def smem_bytes_a(self):
        bm, _, bk = self.tile
        return bm * bk * self.a_smem_dtype.width * self.ab_stage // 8

    def smem_bytes_b(self):
        _, bn, bk = self.tile
        return bn * bk * self.b_smem_dtype.width * self.ab_stage // 8

    def smem_bytes(self):
        return self.smem_bytes_a() + self.smem_bytes_b()

    def make_smem_layout_a(self):
        return self._smem_layout(self.tma_box_a, self.a_smem_dtype)

    def make_smem_layout_b(self):
        return self._smem_layout(self.tma_box_b, self.b_smem_dtype)

    def _tma_atom(self, g, smem_layout, box, internal):
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            g,
            cute.slice_(smem_layout, (None, None, 0)),
            box,
            num_multicast=1,
            internal_type=internal,
        )

    def make_tma_atom_a(self, gA, smem_layout_a):
        return self._tma_atom(gA, smem_layout_a, self.tma_box_a, self.a_tma_internal)

    def make_tma_atom_b(self, gB, smem_layout_b):
        return self._tma_atom(gB, smem_layout_b, self.tma_box_b, self.b_tma_internal)


class SmemUnionTooSmall(AssertionError):
    pass


class Sm120GemmBuilder:
    ACC = cutlass.Float32
    I64, I32, I16, I8 = cutlass.Int64, cutlass.Int32, cutlass.Int16, cutlass.Int8

    gated = False

    def __init__(
        self,
        mma,
        load_ab,
        load_sf,
        epi,
        ab_stage,
        tile,
        epi_bar_id=2,
        union_smem=False,
        reg_prod=None,
        enable_pdl=False,
    ):
        self.mma, self.load_ab, self.load_sf, self.epi = mma, load_ab, load_sf, epi
        self.enable_pdl = enable_pdl
        self.union_smem = union_smem
        assert not union_smem or epi.DRAINS_SC_IN_WG, (
            f"union_smem overlays sC on A/B, which only a warpgroup-drained sC allows; "
            f"{epi.METHOD} would leave store_empty without a counterpart"
        )
        union_bytes = epi.smem_bytes(tile) + epi.aux_smem_bytes(tile)
        if union_smem and union_bytes > load_ab.smem_bytes():
            raise SmemUnionTooSmall(
                f"sC plus its aux buffers are {union_bytes} B over {load_ab.smem_bytes()} B of A/B "
                f"staging, so they reach the SF ring, which a different warp refills without waiting "
                f"on store_empty"
            )
        assert not epi.DRAINS_SC_IN_WG or union_smem, (
            f"{epi.METHOD} writes sC in place, and store_wg's entry barrier exists only to keep "
            f"one warp's sC write from racing another's still-pending read of the sA it overlays"
        )
        self.TILE = tuple(tile)
        self.ab_stage = ab_stage
        self.epi_bar_id = epi_bar_id
        num_math_warps = mma.num_math_warps
        self.num_math_warps = num_math_warps
        self.mma_threads = num_math_warps * 32
        assert epi.mma_threads == self.mma_threads, (
            f"EpiConfig was built for {epi.mma_threads} math threads but MmaConfig gives "
            f"{self.mma_threads}; the S2G thread layout is derived from that count"
        )
        self.store_threads = epi.num_store_threads
        self.sched_warp = num_math_warps
        self.ab_warp = num_math_warps + 1
        self.sf_warp = num_math_warps + 2
        self.store_warp = num_math_warps + 3
        num_producer_warps = (
            2 + self.load_sf.sf_load_warp + (1 if epi.HAS_STORE_WARP else 0)
        )
        assert num_producer_warps <= 4, (
            f"sched, ab, {self.load_sf.sf_load_warp} SF and "
            f"{'a store' if epi.HAS_STORE_WARP else 'no store'} warp need "
            f"{num_producer_warps} producer slots, but the layout has four"
        )
        assert num_math_warps % self.WARPS_PER_WG == 0, (
            f"{num_math_warps} math warps is not a whole number of warpgroups"
        )
        self.num_wg = num_math_warps // self.WARPS_PER_WG + 1
        self.threads = self.num_wg * 128
        self.epi_threads = self.mma_threads + (32 if epi.HAS_STORE_WARP else 0)
        self.reg_prod = self.REG_PROD if reg_prod is None else reg_prod
        self.reg_math = self._reg_math(num_math_warps, self.WARPS_PER_WG, self.reg_prod)
        self.sched_stages = MoeSchedStages
        self.store_stages = epi.epi_stage
        self.fields = MoeWorkTile.FIELDS
        self.num_sched_consumers = num_math_warps + num_producer_warps - 1

    MBAR_RESERVE = 1024
    MAX_AB_STAGE = 4

    WARPS_PER_WG = 4
    REGS_PER_SM = 65536
    REG_GRANULARITY = 8
    REG_MAX = 256
    REG_PROD = 40

    @classmethod
    def _reg_math(cls, num_math_warps, num_producer_warps, reg_prod):
        left = cls.REGS_PER_SM - num_producer_warps * 32 * reg_prod
        per_thread = left // (num_math_warps * 32)
        reg_math = min(cls.REG_MAX, per_thread - per_thread % cls.REG_GRANULARITY)
        if (
            num_producer_warps * 32 * reg_prod + num_math_warps * 32 * reg_math
            >= cls.REGS_PER_SM
        ):
            reg_math -= cls.REG_GRANULARITY
        assert reg_math >= reg_prod, (
            f"{num_math_warps} math and {num_producer_warps} producer warps leave {reg_math} registers "
            f"per math thread, below the {reg_prod} the producers get"
        )
        return reg_math

    @property
    def smem_bytes(self):
        total = self.load_ab.smem_bytes() + self.load_sf.smem_bytes(
            self.TILE, self.ab_stage
        )
        if self.gated:
            total += (
                self.load_ab.smem_bytes_a()
                if self.mma.swap_ab
                else self.load_ab.smem_bytes_b()
            )
            total += self.load_sf.gate_smem_bytes(
                self.TILE, self.ab_stage, self.mma.swap_ab
            )
        if not self.union_smem:
            total += self.epi.smem_bytes(self.TILE) + self.epi.aux_smem_bytes(self.TILE)
        return total

    @classmethod
    def max_ab_stage(cls, make_cfg, tile):
        capacity = cutlass.utils.SmemAllocator.capacity_in_bytes() - cls.MBAR_RESERVE
        for stage in range(cls.MAX_AB_STAGE, 0, -1):
            try:
                cfg = make_cfg(tile, stage)
            except SmemUnionTooSmall:
                continue
            if cfg.load_sf.ab_stages_contract(stage) and cfg.smem_bytes <= capacity:
                return stage
        raise ValueError(f"no ab_stage fits {capacity} B of smem for tile {tile}")

    def is_prod_wg(self, warp_idx):
        return warp_idx >= self.num_math_warps


class FC1ActBuilder(Sm120GemmBuilder):
    gated = True

    def __init__(self, *args, activation, fastmath=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation, self.fastmath = activation, fastmath
