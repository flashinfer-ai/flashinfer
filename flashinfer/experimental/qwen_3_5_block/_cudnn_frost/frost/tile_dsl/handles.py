# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MmaDesc:
    M: int
    N: int
    K: int
    bpe_a: int
    bpe_b: int
    tile_k_hw: int = 64
    btranspose: bool = False
    atranspose: bool = False
    k_subtile: int = -1
    cta_group: int = 1
    idesc: object = None
    kind: object = None
    is_block_scale: bool = False
    sf_blocks_per_step: int = 0
    sf_cycle: int = 0
    scale_vec_size: object = None

    def __post_init__(self):
        if self.k_subtile < 0:
            ks = self.K if self.btranspose else (128 // self.bpe_a if (self.K * self.bpe_a) % 128 == 0 else 64 // self.bpe_a)
            object.__setattr__(self, "k_subtile", ks)

    @property
    def m_per_cta(self):
        return self.M // self.cta_group

    @property
    def n_per_cta(self):
        return self.N // self.cta_group

    @property
    def num_k_steps(self):
        return self.K // self.tile_k_hw

    @property
    def steps_per_subtile(self):
        return self.k_subtile // self.tile_k_hw

    @property
    def num_subtiles(self):
        return self.num_k_steps // self.steps_per_subtile

    @staticmethod
    def _swz_from_inner(inner_bytes: int) -> int:
        if inner_bytes % 128 == 0:
            return 128
        if inner_bytes % 64 == 0:
            return 64
        return 32

    @property
    def swz_a_bytes(self):
        inner = self.m_per_cta if self.atranspose else self.K
        return MmaDesc._swz_from_inner(inner * self.bpe_a)

    @property
    def swz_b_bytes(self):
        inner = self.n_per_cta if self.btranspose else self.K
        return MmaDesc._swz_from_inner(inner * self.bpe_b)

    @property
    def smem_advance_A_intra(self):
        if self.atranspose:
            return self.tile_k_hw * self.swz_a_bytes
        return self.tile_k_hw * self.bpe_a

    @property
    def smem_advance_B_intra(self):
        if self.btranspose:
            return self.tile_k_hw * self.swz_b_bytes
        return self.tile_k_hw * self.bpe_b

    @property
    def smem_subtile_A(self):
        if self.atranspose:
            return self.K * self.swz_a_bytes
        return self.swz_a_bytes * self.m_per_cta

    @property
    def smem_subtile_B(self):
        if self.btranspose:
            return self.K * self.swz_b_bytes
        return self.swz_b_bytes * self.n_per_cta

    @property
    def sps_A(self):
        if self.atranspose:
            return self.K // self.tile_k_hw
        return (self.swz_a_bytes // self.bpe_a) // self.tile_k_hw

    @property
    def sps_B(self):
        if self.btranspose:
            return self.K // self.tile_k_hw
        return (self.swz_b_bytes // self.bpe_b) // self.tile_k_hw

    @property
    def num_subtiles_A(self):
        return self.num_k_steps // self.sps_A

    @property
    def num_subtiles_B(self):
        return self.num_k_steps // self.sps_B

    @property
    def tmem_advance_A(self):
        return self.tile_k_hw * self.bpe_a // 4


@dataclass(frozen=True)
class SmemTile:
    base: object
    elems_per_stage: int
    leading_byte_offset: int
    stride_byte_offset: int
    layout: int
    tma_loads_per_tile: int = 1
    tma_granu_elems: int = 0
    tma_subtile_stride_elems: int = 0
    stages: int = 1
    # tcgen05 SMEM-descriptor version for :meth:`desc`.
    #
    # 0 = the public ``Tcgen05SmemDesc.build`` path (SM100 format).  Its
    # ``start_address`` field is bits [0:15) with bit [14] *reserved on SM100*,
    # i.e. only 14 usable bits = a 256 KiB addressable window.
    #
    # 1 = the extended format, which is what the C++ ``SmemTile::make_desc``
    # always emitted.  REQUIRED on Rubin (SM107) for any MMA operand whose SMEM
    # buffer sits at or above 256 KiB — the arch raises the per-CTA SMEM cap to
    # 327 KiB, but a version-0 descriptor still truncates the address to 14 bits
    # and silently wraps it to the bottom of SMEM.
    #
    # Symptom of getting this wrong: the MMA runs, both operands are provably
    # correct in SMEM, the accumulator TMEM is the one the epilogue reads — and
    # the accumulator is EXACTLY zero, because the wrapped address landed on an
    # untouched buffer.  No crash, no launch error.  (Found 2026-09-04 on the
    # d512 f16 SM107 port: sP_xfer_raw began at exactly 262144.)
    desc_version: int = 0

    def __getitem__(self, stage):
        off = stage * self.elems_per_stage
        base = self.base.subview(off) if hasattr(self.base, "subview") else self.base + off
        return replace(self, base=base, stages=1)

    def shifted(self, off_elems):
        base = self.base.subview(off_elems) if hasattr(self.base, "subview") else self.base + off_elems
        return replace(self, base=base)

    def desc(self):
        from cutlass.experimental import primitives as prims

        if self.desc_version not in (0, 1):
            raise ValueError(f"SmemTile.desc_version must be 0 or 1 (got {self.desc_version}); there is no other tcgen05 SMEM-descriptor format")

        if self.desc_version == 0:
            return prims.Tcgen05SmemDesc.build(
                self.base,
                leading_byte_offset=self.leading_byte_offset,
                stride_byte_offset=self.stride_byte_offset,
                layout=self.layout,
            )

        # Hand-rolled because the public ``Tcgen05SmemDesc.build`` takes no
        # ``version`` argument — it calls the non-versioned intrinsic.  The
        # versioned one exists in the same wheel, just module-private, and the
        # argument list below mirrors ``build``'s defaults exactly.
        import cutlass
        from cutlass._mlir.dialects import llvm
        from cutlass.experimental.primitives import nvvm_wrapper as _nvvm_wrap

        if not hasattr(_nvvm_wrap, "_tcgen05_mma_smem_desc_v2"):
            raise RuntimeError(
                "desc_version=1 needs the versioned tcgen05 SMEM-descriptor intrinsic "
                "(cutlass.experimental.primitives.nvvm_wrapper._tcgen05_mma_smem_desc_v2), "
                "which this cutlass-dsl build does not expose.  It is required on Rubin "
                "(SM107) for any MMA operand at or above 256 KiB; there is no kernel-side "
                "workaround -- upgrade the DSL."
            )

        addr = cutlass.Int32(llvm.ptrtoint(cutlass.Int32.mlir_type, self.base.ir_value()))
        return prims.Tcgen05SmemDesc(
            _nvvm_wrap._tcgen05_mma_smem_desc_v2(
                addr >> cutlass.Int32(4),
                self.leading_byte_offset >> 4,
                self.stride_byte_offset >> 4,
                self.desc_version,
                0,  # base_offset
                0,  # leading_dim_mode
                0,  # k_segment_offset
                self.layout,
            )
        )


@dataclass(frozen=True)
class GmemTileTma:
    tma_desc: object

    def __call__(self, *coords, coord_0=None):
        if not coords:
            raise ValueError("GmemTileTma needs at least the innermost coord")
        if coord_0 is not None:
            if len(coords) != 4:
                raise ValueError(f"GmemTileTma 5-D form (coord_0=…) expects 4 positional " f"coords; got {len(coords)}")
            return GmemTileTmaSlice(
                tma_desc=self.tma_desc,
                coords=(coord_0,) + tuple(coords),
            )
        if not 2 <= len(coords) <= 5:
            raise ValueError(f"GmemTileTma supports rank 2..5; got {len(coords)} coords")
        return GmemTileTmaSlice(
            tma_desc=self.tma_desc,
            coords=tuple(coords),
        )


@dataclass(frozen=True)
class GmemTileTmaSlice:
    tma_desc: object
    coords: tuple
    desc_ptr: object = None

    @property
    def rank(self):
        return len(self.coords)

    @property
    def coord_d(self):
        return self.coords[0]

    def with_coord_d(self, new_d):
        return GmemTileTmaSlice(
            tma_desc=self.tma_desc,
            coords=(new_d,) + tuple(self.coords[1:]),
            desc_ptr=self.desc_ptr,
        )


def tma_slice_runtime_desc(desc_ptr, *coords):
    return GmemTileTmaSlice(tma_desc=None, coords=tuple(coords), desc_ptr=desc_ptr)


@dataclass(frozen=True)
class GmemTileLinear:
    base: object
    stride_b: int
    stride_h: int
    stride_tile: int

    def addr(self, batch, head, tile_idx):
        return self.base + batch * self.stride_b + head * self.stride_h + tile_idx * self.stride_tile
