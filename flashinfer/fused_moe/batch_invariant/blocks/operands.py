# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The operand blocks of the grouped MoE bodies: what a body needs to know about its operands' format, resolved at trace
time, so one grouped body (:mod:`.kernel`) and one swapped body (:mod:`.swap`) serve every class of the backend.

* :class:`DenseOperands` -- the dense formats (``bf16`` / ``e4m3``): the format's dtype and alignment, the K-steps per smem
  stage of each body, the mainloop block whose lowering ``mma_loop`` executes and the static plan; the TMA ring is the dense
  GEMM's (one A and one B box per stage).
* :class:`BlockScaledOperands` -- the block-scaled formats (the 4-bit ``nvfp4`` / ``mxfp4`` and the 8-bit ``mxfp8`` element
  formats, and the mixed ``mxfp4w8a`` pair of an e4m3 token operand against an e2m1 weight operand, each with the sm_100 /
  sm_103 and the sm_107 instruction class), parametrised by the plan of the instruction class
  (:data:`flashinfer.gemm.batch_invariant.blocks.blockscaled.PLANS`, the mixed plans of
  :mod:`flashinfer.gemm.batch_invariant.blocks.blockscaled_mx_mixed`; the swapped body runs a mixed class on the twin plan
  with the operand formats exchanged): the element dtype per operand, the scale-factor dtype (``e4m3`` per
  16 elements for NVFP4, ``e8m0`` per 32 for the MX formats) and the scale block, the plan with the grouped bodies' own stage
  rule (the operand and scale-factor rings fill the shared memory after the barriers and the body's own tables and staging
  buffers), the views of the byte-packed operands and of the flat 128x4 swizzled scale-factor buffers, the four-operand TMA
  setup, and the pieces of the swapped body's token-side scale ring (a ``cp.async`` producer warp gathering the token tile's
  scale-factor rows from the padded positions). The scale factors of a group start at a 128-row scale-factor atom
  (``group_align``): the swizzled scale layout is affine only inside an atom, so a TMA box of SFA starts at an atom
  boundary. Two packings of the A rows are placements of the same per-row fold: ``dense`` (the groups' rows back to back,
  the scale factors and the per-row scales at the padded positions through a second table of group offsets) and ``padded``
  (every group's rows start at an atom; the padding rows, zero-filled by the caller, are computed and never stored).

The interface a body consumes (stable; additions only): ``block_scaled``, ``mixed``, ``in_fmt``, ``packing`` / ``dense_a``,
``dtype``, ``align``, ``epi_warpgroups``, ``table_bytes``, ``k_blocks(body)``, ``mainloop(body)``, ``plan(...)``,
``plan_cls_for(body)``, ``view_tokens(t)`` / ``view_weights(t)``, ``smem_dtypes(st)``, ``tmem_alloc_kwargs(st)``; for the
block-scaled block also ``sf_dtype``, ``sf_vec``, ``elem_bits``, ``group_align``, ``view_elems(t)``, ``sf_view_rows(...)``,
``sf_view_experts(...)``, ``tma(...)``, ``sf_groups(st)``, ``sf_stage_bytes(st)``, and the module-level swapped-body pieces
:class:`PipelineCpAsyncUmma`, :func:`gather_sf_row_padded`, :func:`swap_sf_producer_loop`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as bs_layout
from cutlass import Boolean, Int32, Uint8
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineOp,
    agent_sync,
)
from cutlass.pipeline.helpers import MbarrierArray
from cutlass.tensor_utils import LayoutEnum

from ....gemm.batch_invariant.blocks.blockscaled import (
    A_RING_MAX_BYTES,
    PLANS,
    SF_ATOM_BYTES,
    fp4_view,
    sf_view,
)
from ....gemm.batch_invariant.blocks.blockscaled_mx_mixed import (
    PLANS as MIXED_PLANS,
)
from ....gemm.batch_invariant.blocks.blockscaled_mx_mixed import (
    SWAPPED_PLANS as MIXED_SWAPPED_PLANS,
)
from ....gemm.batch_invariant.blocks.operands import (
    ACC_DTYPE,
    ALIGN,
    IN_DT,
    MBAR_RESERVE_BYTES,
    NUM_C_STAGE_MIN,
    StaticPlan,
)
from ....gemm.batch_invariant.generator import Knobs, MainloopBlock
from ....gemm.batch_invariant.spec import (
    A_ELEM_BITS_OF,
    B_ELEM_BITS_OF,
    BLOCK_SCALED,
    ELEM_BITS_OF,
    MX_MIXED_FMTS,
    SF_ATOM_KBLOCKS,
    SF_ATOM_ROWS,
)
from .scheduler import EPI_WARPGROUPS_DENSE, MAX_E

# instruction K-steps per smem stage of the grouped records: two 128-byte swizzle atoms of K per row (the dense class's default
# is one), halving the pipeline round trips per weight byte of a decode tile; a placement of the same fold (the lowering's inner
# radix; Lean's ``kblockLoopG_eq_kfoldG`` holds for every radix)
GROUPED_K_BLOCKS = 8
#: Instruction K-steps per smem stage by body and operand format (``None``: the format's default, one swizzle atom of K):
#: the bf16 grouped body and both formats' swapped box bodies stream two atoms per stage, the e4m3 grouped body the
#: format's default, the gather form one atom (the gather4 box is the atom).
K_BLOCKS_OF: Dict[str, Dict[str, Optional[int]]] = {
    "grouped": {"bf16": GROUPED_K_BLOCKS, "e4m3": None},
    "swapped": {"bf16": GROUPED_K_BLOCKS, "e4m3": GROUPED_K_BLOCKS},
    "gather": {"bf16": None},
}
#: The two packings of the block-scaled bodies' A rows.
PACKINGS = ("dense", "padded")
#: Epilogue warpgroups of the block-scaled grouped body: the 4-bit MMA leaves the declared fp32 chain as the tile's longest
#: leg, so four warpgroups split a tile's epilogue subtiles (the dense bodies take two). A placement.
EPI_WARPGROUPS_BLOCKSCALED = 4
#: Scale-factor bytes per 4-byte ``cp.async`` of the swapped body's token-side scale gather (one 4-block column group of the atom).
SF_GROUP = 4


@dataclass(frozen=True)
class DenseOperands:
    """The operand block of the dense formats (``bf16`` / ``e4m3``): the format's operand dtype and alignment, the K-steps
    per smem stage of each body (:data:`K_BLOCKS_OF`), the mainloop block whose lowering ``mma_loop`` executes and the
    static plan (tiled MMA, tilers, stages, smem layouts); the TMA ring is the dense GEMM's (one A and one B box per
    stage, ``TmaSetup``), the smem beyond the ring is the group tables' and the epilogue's."""

    in_fmt: str = "bf16"
    block_scaled = False
    mixed = False  # one element format for both operands
    packing: str = "dense"

    def __post_init__(self):
        assert self.in_fmt in K_BLOCKS_OF["grouped"], self.in_fmt
        assert self.packing == "dense", self.packing

    @property
    def dense_a(self) -> bool:
        return True

    @property
    def dtype(self):
        return IN_DT[self.in_fmt]

    def view_tokens(self, t: cute.Tensor) -> cute.Tensor:
        """The token operand as the TMA reads it (the dense formats: as given)."""
        return t

    def view_weights(self, t: cute.Tensor) -> cute.Tensor:
        """The weight operand as the TMA reads it (the dense formats: as given)."""
        return t

    @staticmethod
    def smem_dtypes(st):
        """The element types of the A and B smem rings: the operands' own (the dense plan has no container)."""
        return st.a_dtype, st.b_dtype

    @property
    def align(self) -> int:
        """Elements per 16-byte row: the TMA alignment of every operand."""
        return ALIGN[self.in_fmt]

    @property
    def epi_warpgroups(self) -> int:
        return EPI_WARPGROUPS_DENSE

    @property
    def table_bytes(self) -> int:
        """The group tables the bodies keep in shared memory (the group offsets and the cluster-row-block prefix)."""
        return 2 * (MAX_E + 1) * 4

    def k_blocks(self, body: str) -> Optional[int]:
        return K_BLOCKS_OF[body][self.in_fmt]

    def mainloop(self, body: str) -> MainloopBlock:
        """The generator's mainloop block for the format and body: its lowering (blocked issue order, accumulator reset) is
        what ``mma_loop`` executes (the override by name: the block's overrides are siblings, never positional)."""
        return MainloopBlock(self.in_fmt, inner_steps=self.k_blocks(body))

    def plan(self, knobs: Knobs, a, b, extra_smem_bytes: int, body: str) -> StaticPlan:
        """The static plan of a body for the operands ``a`` / ``b`` (their dtypes and majors); the accumulator-side "C" is
        fp32 row-major and only sizes the epilogue tile; ``extra_smem_bytes`` is the caller's own shared storage."""
        assert a.element_type is self.dtype and b.element_type is self.dtype, (
            a.element_type,
            b.element_type,
            self.in_fmt,
        )
        return StaticPlan(
            knobs,
            a.element_type,
            b.element_type,
            ACC_DTYPE,
            LayoutEnum.from_tensor(a).mma_major_mode(),
            LayoutEnum.from_tensor(b).mma_major_mode(),
            LayoutEnum.ROW_MAJOR,
            c_in_smem=False,
            extra_smem_bytes=extra_smem_bytes,
            in_fmt=self.in_fmt,
            k_blocks=self.k_blocks(body),
        )

    def tmem_alloc_kwargs(self, st) -> dict:
        """Further keyword arguments of the TMEM allocator (none for the dense formats: the sm_100 allocation rule)."""
        return {}


def sf_view_experts(sf: cute.Tensor, rows, k, E, vec: int, sf_dtype):
    """The ``E`` consecutive 128x4 swizzled scale-factor buffers of ``(rows, k)`` operands as the ``(rows, k, E)`` scale tensor
    the block-scaled TMA reads: tiles of 128 rows x 4 K-blocks (512 B), K-tiles consecutive, row-tiles strided by the
    K-tiles, experts strided by one whole buffer (``sf_bytes(rows, k)``)."""
    mt = cute.ceil_div(rows, SF_ATOM_ROWS)
    kt = cute.ceil_div(k, vec * SF_ATOM_KBLOCKS)
    ptr = cute.recast_ptr(sf.iterator, dtype=sf_dtype)
    layout = cute.make_layout(
        (((32, 4), mt), ((vec, SF_ATOM_KBLOCKS), kt), E),
        stride=(
            ((16, 4), SF_ATOM_BYTES * kt),
            ((0, 1), SF_ATOM_BYTES),
            SF_ATOM_BYTES * kt * mt,
        ),
    )
    return cute.make_tensor(ptr, layout)


class GroupedBlockScaledTma:
    """TMA atoms and coordinate tensors for A, SFA (re-partitioned per tile in the producer), B and SFB (the expert as
    the batch coordinate); the transaction bytes of one stage of all four operands. The dense ``BlockScaledTmaSetup``
    without the output store atom (the grouped epilogue stores from registers)."""

    def __init__(
        self, st, a: cute.Tensor, b: cute.Tensor, sfa: cute.Tensor, sfb: cute.Tensor
    ):
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        a_one = cute.slice_(st.a_smem_layout_staged, (None, None, None, 0))
        # `internal_type`: the TMA's unpacking format of a sub-byte global operand into an 8-bit smem container (the mixed
        # sm_100 class's e2m1 weight, the plan's container hooks); None keeps the operand's own format
        self.atom_a, self.tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_one,
            st.mma_tiler,
            st.tiled_mma,
            st.cluster_layout_vmnk.shape,
            internal_type=st.A_TMA_INTERNAL,
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        b_one = cute.slice_(st.b_smem_layout_staged, (None, None, None, 0))
        self.atom_b, self.tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_one,
            st.mma_tiler,
            st.tiled_mma,
            st.cluster_layout_vmnk.shape,
            internal_type=st.B_TMA_INTERNAL,
        )
        sfa_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        sfa_one = cute.slice_(st.sfa_smem_layout_staged, (None, None, None, 0))
        self.atom_sfa, self.tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_one,
            st.mma_tiler,
            st.tiled_mma,
            st.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = utils.sm100.cluster_shape_to_tma_atom_SFB(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        sfb_one = cute.slice_(st.sfb_smem_layout_staged, (None, None, None, 0))
        self.atom_sfb, tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_one,
            st.mma_tiler_sfb,
            st.tiled_mma_sfb,
            st.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if st.cta_tile_shape_mnk[1] == 192:
            # a 192-wide N tile spans one and a half scale-factor atoms: N tile n (a 256-row box) starts at atom 3(n//2) + n%2
            # (the dense block's re-addressing)
            x = tensor_sfb.stride[0][1]
            y = cute.ceil_div(tensor_sfb.shape[0][1], 4)
            new_shape = (
                (tensor_sfb.shape[0][0], ((2, 2), y)),
                tensor_sfb.shape[1],
                tensor_sfb.shape[2],
            )
            new_stride = (
                (tensor_sfb.stride[0][0], ((x, x), 3 * x)),
                tensor_sfb.stride[1],
                tensor_sfb.stride[2],
            )
            tensor_sfb = cute.make_tensor(
                tensor_sfb.iterator, cute.make_layout(new_shape, stride=new_stride)
            )
        self.tensor_sfb = tensor_sfb
        self.tx_bytes = (
            cute.size_in_bytes(st.a_dtype, a_one)
            + cute.size_in_bytes(st.b_dtype, b_one)
            + cute.size_in_bytes(st.sf_dtype, sfa_one)
            + cute.size_in_bytes(st.sf_dtype, sfb_one)
        ) * st.atom_thr_size


def xch_smem_bytes(layout) -> int:
    """Bytes of an fp32 shared-memory exchange buffer with ``layout`` (a chain's ``smem_layout``; None: no buffer)."""
    return 0 if layout is None else 4 * cute.cosize(layout)


def grouped_plan(
    plan_cls,
    knobs: Knobs,
    own_bytes: Callable,
    umma_addr_max: Optional[int] = None,
    k_blocks: Optional[int] = None,
):
    """The block-scaled plan with the grouped bodies' own stage rule: the operand ring (A, B, SFA, SFB per stage) fills the
    CTA's shared memory after the barrier reserve and the body's own buffers (``own_bytes(st)``: the group tables, the
    epilogue staging buffers), under the dense plan's A-ring cap. The dense plan's output stages are not reserved (the
    grouped epilogue stores from its own staging buffers). ``umma_addr_max``: the operand rings a UMMA reads (A and B,
    behind the barriers and the body's own buffers) end below this many bytes of the CTA's shared memory (the plan's
    ``UMMA_SMEM_ADDR_MAX`` on the sm_100 instruction plans: their descriptors' address window). ``k_blocks``: the
    instruction K steps per stage (the swapped body's two-atom stage; None: the format's one swizzle atom of K). A
    placement: the ring depth moves no bits."""

    class _Plan(plan_cls):
        def _stages(self):
            # the rings at the smem container widths (a sub-byte operand the TMA unpacks occupies a byte per element: the
            # mixed sm_100 class's e2m1 weight; the one-format plans' containers are their operands' own types)
            a_one = utils.sm100.make_smem_layout_a(
                self.tiled_mma, self.mma_tiler, self.a_smem_dtype, 1
            )
            b_one = utils.sm100.make_smem_layout_b(
                self.tiled_mma, self.mma_tiler, self.b_smem_dtype, 1
            )
            sfa_one = bs_layout.make_smem_layout_sfa(
                self.tiled_mma, self.mma_tiler, self.sf_vec, 1
            )
            sfb_one = bs_layout.make_smem_layout_sfb(
                self.tiled_mma, self.mma_tiler, self.sf_vec, 1
            )
            a_bytes = cute.size_in_bytes(self.a_smem_dtype, a_one)
            b_bytes = cute.size_in_bytes(self.b_smem_dtype, b_one)
            ab_bytes = (
                a_bytes
                + b_bytes
                + cute.size_in_bytes(self.sf_dtype, sfa_one)
                + cute.size_in_bytes(self.sf_dtype, sfb_one)
            )
            num_ab = (
                self.smem_capacity - MBAR_RESERVE_BYTES - own_bytes(self)
            ) // ab_bytes
            num_ab = min(num_ab, (A_RING_MAX_BYTES - 1) // a_bytes)
            if umma_addr_max is not None:
                num_ab = min(
                    num_ab,
                    (umma_addr_max - MBAR_RESERVE_BYTES - own_bytes(self))
                    // (a_bytes + b_bytes),
                )
            assert num_ab >= 2, (
                "the grouped kernel's operand ring does not fit the shared memory",
                self.knobs,
                num_ab,
            )
            return num_ab, NUM_C_STAGE_MIN

    return _Plan(knobs, ACC_DTYPE, LayoutEnum.ROW_MAJOR, k_blocks=k_blocks)


@dataclass(frozen=True)
class BlockScaledOperands:
    """The operand block of the block-scaled formats, keyed by the instruction class (``in_fmt``: a key of ``PLANS``) and the
    packing of the A rows (:data:`PACKINGS`). The plan class carries the element dtype, the scale-factor dtype and block and
    the instruction; this block adds the grouped bodies' rules on top of it."""

    in_fmt: str = "nvfp4"
    packing: str = "dense"
    block_scaled = True

    def __post_init__(self):
        assert self.in_fmt in BLOCK_SCALED and (
            self.in_fmt in PLANS or self.in_fmt in MIXED_PLANS
        ), self.in_fmt
        assert self.packing in PACKINGS, self.packing

    @property
    def mixed(self) -> bool:
        """The mixed classes: the token operand e4m3, the weight operand e2m1 (the swapped body's plan is the twin with the
        formats exchanged); the one-format classes read one element type on both operands."""
        return self.in_fmt in MX_MIXED_FMTS

    @property
    def plan_cls(self):
        """The grouped body's plan class (the tokens as A, the weights as B)."""
        return PLANS[self.in_fmt] if self.in_fmt in PLANS else MIXED_PLANS[self.in_fmt]

    def plan_cls_for(self, body: str):
        """The plan class of a body: the swapped body of a mixed class runs the twin plan with the operand formats exchanged
        (the weight is the kernel's A); every other case is the class's plan."""
        if self.mixed and body == "swapped":
            return MIXED_SWAPPED_PLANS[self.in_fmt]
        return self.plan_cls

    @property
    def dense_a(self) -> bool:
        """The dense packing: the A rows back to back, the scale factors and the per-row scales at the padded positions."""
        return self.packing == "dense"

    @property
    def dtype(self):
        """The token operand's element type (the one-format classes': both operands')."""
        return self.plan_cls.A_DT

    @property
    def sf_dtype(self):
        return self.plan_cls.SF_DT

    @property
    def sf_vec(self) -> int:
        """Elements per block scale (16 for the NVFP4 formats, 32 for the MX formats)."""
        return self.plan_cls.SF_VEC

    @property
    def elem_bits(self) -> int:
        """The token operand's element width (the one-format classes': both operands')."""
        return ELEM_BITS_OF[self.in_fmt]

    @property
    def token_elem_bits(self) -> int:
        return A_ELEM_BITS_OF[self.in_fmt]

    @property
    def weight_elem_bits(self) -> int:
        return B_ELEM_BITS_OF[self.in_fmt]

    @property
    def align(self) -> int:
        """Elements per 16-byte row: the TMA alignment of every operand."""
        return ALIGN[self.in_fmt]

    @property
    def group_align(self) -> int:
        """The group alignment of the scale factors (rows): one scale-factor atom."""
        return SF_ATOM_ROWS

    @property
    def epi_warpgroups(self) -> int:
        return EPI_WARPGROUPS_BLOCKSCALED

    @property
    def table_bytes(self) -> int:
        """The group tables in shared memory: the dense offsets and the cluster-row-block prefix, plus the padded offsets and
        prefix of the dense packing."""
        return (4 if self.dense_a else 2) * (MAX_E + 1) * 4

    def k_blocks(self, body: str) -> Optional[int]:
        """The block-scaled plans fix their inner radix (``None``: the plan's)."""
        return None

    def mainloop(self, body: str) -> MainloopBlock:
        return MainloopBlock(self.in_fmt)

    def plan(
        self,
        knobs: Knobs,
        own_bytes: Callable,
        body: str,
        umma_addr_max: Optional[int] = None,
        k_blocks: Optional[int] = None,
    ):
        """The block-scaled plan with the grouped bodies' stage rule (:func:`grouped_plan`); ``own_bytes(st)`` the body's own
        shared storage beyond the barriers, ``umma_addr_max`` the UMMA descriptors' address window the bodies' A ring must
        end inside (``None``: the plan's A-ring cap alone), ``k_blocks`` the instruction K steps per stage (None: the
        format's)."""
        return grouped_plan(
            self.plan_cls_for(body), knobs, own_bytes, umma_addr_max, k_blocks
        )

    def view_elems(self, t: cute.Tensor) -> cute.Tensor:
        """The API's operand as the element tensor the TMA reads: the byte-packed 4-bit operand viewed as fp4 of the logical
        K, an 8-bit element operand as given (the token operand's width; the one-format classes' both operands')."""
        return fp4_view(t) if self.elem_bits == 4 else t

    def view_tokens(self, t: cute.Tensor) -> cute.Tensor:
        """The token operand as the TMA reads it (its own element width)."""
        return fp4_view(t) if self.token_elem_bits == 4 else t

    def view_weights(self, t: cute.Tensor) -> cute.Tensor:
        """The weight operand as the TMA reads it (its own element width: the mixed classes' e2m1 weight is byte-packed)."""
        return fp4_view(t) if self.weight_elem_bits == 4 else t

    @staticmethod
    def smem_dtypes(st):
        """The element types of the A and B smem rings: the plan's containers (a sub-byte operand the TMA unpacks occupies a
        byte per element; the one-format plans' containers are their operands' own types)."""
        return st.a_smem_dtype, st.b_smem_dtype

    def sf_view_rows(self, sf: cute.Tensor, rows, k) -> cute.Tensor:
        """The flat 128x4 swizzled scale-factor buffer of a ``(rows, k)`` A operand as the scale tensor the TMA reads."""
        return sf_view(sf, rows, k, 1, self.sf_vec, self.sf_dtype)

    def sf_view_experts(self, sf: cute.Tensor, n, k, E) -> cute.Tensor:
        """The ``E`` consecutive 128x4 buffers of the ``(n, k)`` expert weights as the ``(n, k, E)`` scale tensor."""
        return sf_view_experts(sf, n, k, E, self.sf_vec, self.sf_dtype)

    def tma(self, st, a, b, sfa_t, sfb_t) -> GroupedBlockScaledTma:
        return GroupedBlockScaledTma(st, a, b, sfa_t, sfb_t)

    def tmem_alloc_kwargs(self, st) -> dict:
        """The allocator's architecture: the sm_107 plans allocate the whole 576-column TMEM when the need exceeds 512."""
        return dict(arch=st.TMEM_ALLOC_ARCH)

    def sf_groups(self, st) -> int:
        """4-byte block groups per row and k-tile of the token tile's scale-factor gather (the swapped body)."""
        return (st.cta_tile_shape_mnk[2] // st.sf_vec) // SF_GROUP

    def sf_stage_bytes(self, st) -> int:
        """Bytes of one stage of the token tile's scale-factor atom (the SFB ring of the swapped body)."""
        return cute.size_in_bytes(
            st.sf_dtype, cute.slice_(st.sfb_smem_layout_staged, (None, None, None, 0))
        )


def operands_for(in_fmt: str, packing: str = "dense"):
    """The operand block of a chain's format: the dense block for ``bf16`` / ``e4m3``, the block-scaled one otherwise."""
    if in_fmt in K_BLOCKS_OF["grouped"]:
        assert packing == "dense", (in_fmt, packing)
        return DenseOperands(in_fmt)
    return BlockScaledOperands(in_fmt, packing)


# ----------------------------------------------------------------------- the swapped body's token-side scale ring


@dataclass(frozen=True)
class PipelineCpAsyncUmma(PipelineAsync):
    """``cp.async`` producers (the scale-factor warp: each thread's arrival is its ``cp.async.mbarrier.arrive.noinc``, so the
    full barrier flips when every thread's copies have landed) and the UMMA consumer (its release is the ``tcgen05`` commit of
    the MMA that read the stage). The DSL's own ``PipelineCpAsync`` has an async-thread consumer; this is its UMMA-consumer
    sibling (the 1-CTA form: no peer mask)."""

    cta_group: tcgen05.CtaGroup = tcgen05.CtaGroup.ONE

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        barrier_storage: cute.Pointer,
        defer_sync: bool = False,
    ):
        producer = (PipelineOp.AsyncLoad, producer_group)
        consumer = (PipelineOp.TCGen05Mma, consumer_group)
        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer
        )
        sync_object_empty = MbarrierArray(
            barrier_storage=barrier_storage.align(min_align=8) + num_stages,
            num_stages=num_stages,
            agent=consumer,
        )
        if not defer_sync:
            agent_sync(Agent.ThreadBlock)
        return PipelineCpAsyncUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            None,
            None,
            tcgen05.CtaGroup.ONE,
        )

    def consumer_release(self, state):
        self.sync_object_empty.arrive(state.index, self.consumer_mask, self.cta_group)


@cute.jit
def gather_sf_row_padded(
    sf_atom,
    mSFA_flat: cute.Tensor,
    sSFA: cute.Tensor,
    r: Int32,
    tk: Int32,
    ok: Boolean,
    kt: Int32,
    stage: Int32,
    n_blocks: Int32,
    pcols: Int32,
    sf_stage_bytes: cutlass.Constexpr,
    sf_groups: cutlass.Constexpr,
    pred1,
    linear: cutlass.Constexpr = False,
    row_sf_bytes: Int32 = None,
):
    """The scale factors of tile row ``r`` (padded position ``tk``; with ``linear`` the token's row of the row-major ``[T, K /
    sf_vec]`` bytes, ``row_sf_bytes`` per row) for k-tile ``kt``: its ``sf_groups`` 4-byte block groups from
    the 128x4 swizzled buffer into the stage's atom by 4-byte ``cp.async``; a block group past the operand's ``n_blocks`` (the tail
    of a partial k-tile) is the same ``cp.async`` with a false predicate, a zero-fill (ignore-src) tracked by the stage's cp.async
    arrive -- never a generic store, whose visibility to the MMA's async-proxy reads that arrive does not order; a row that is not
    ``ok`` copies nothing (only the owning lane may touch a row's slot). The byte positions are the layout's: row ``r`` at
    ``(r % 32) * 16 + (r // 32) * 4``, block group ``g`` at ``g * 512``, atoms at ``128 * pcols``."""
    if cutlass.const_expr(linear):
        g_src_row = tk * row_sf_bytes
    else:
        g_src_row = (
            (tk // Int32(128)) * (Int32(128) * pcols)
            + (tk % Int32(32)) * Int32(16)
            + ((tk % Int32(128)) // Int32(32)) * Int32(4)
        )
    g_dst_row = (
        stage * Int32(sf_stage_bytes)
        + (r % Int32(32)) * Int32(16)
        + (r // Int32(32)) * Int32(4)
    )
    for g in cutlass.range_constexpr(sf_groups):
        blk0 = (kt * Int32(sf_groups) + Int32(g)) * Int32(SF_GROUP)
        if cutlass.const_expr(linear):
            src_off = g_src_row + (kt * Int32(sf_groups) + Int32(g)) * Int32(SF_GROUP)
        else:
            src_off = g_src_row + (kt * Int32(sf_groups) + Int32(g)) * Int32(
                SF_ATOM_BYTES
            )
        src_off = cute.assume(src_off, divby=4)
        dst_off = g_dst_row + Int32(g * SF_ATOM_BYTES)
        dst_off = cute.assume(dst_off, divby=4)
        src = cute.make_tensor(
            mSFA_flat.iterator + src_off, cute.make_layout((SF_GROUP,))
        )
        dst = cute.make_tensor(
            cute.recast_ptr(sSFA.iterator, dtype=Uint8) + dst_off,
            cute.make_layout((SF_GROUP,)),
        )
        pred1[0] = ok & (blk0 < n_blocks)
        if ok:
            cute.copy_atom_call(sf_atom, src, dst, pred=pred1)


@cute.jit
def swap_sf_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    sf_pipeline,
    mSFA_flat: cute.Tensor,
    sSFA: cute.Tensor,
    lane: Int32,
    Tk: Int32,
    K: Int32,
    pcols: Int32,
    s_off: cute.Tensor,
    s_off_pad: cute.Tensor,
    tn: cutlass.Constexpr,
    sf_groups: cutlass.Constexpr,
    sf_stage_bytes: cutlass.Constexpr,
    mSrc: cute.Tensor = None,
    k_div: Int32 = None,
    sf_linear: cutlass.Constexpr = False,
):
    """The LDGSTS producer of the token tile's scale factors (one warp): lane ``j < tn`` owns tile row ``j`` = dense row
    ``row0 + j``, whose scale factors sit at the padded position ``row0 + j + (off_pad[e] - off[e])`` -- or, in the gather form
    (``mSrc`` given), in the token's own row ``src[row0 + j] // k_div`` of the token rows' buffer (the 128x4 layout, or with
    ``sf_linear`` the row-major bytes); rows past the group's
    real end are skipped (their accumulator columns are never stored). Every lane arrives on the stage's barrier (the pipeline
    counts the warp)."""
    sf_atom = cute.make_copy_atom(cpasync.CopyG2SOp(), Uint8, num_bits_per_copy=32)
    pred1 = cute.make_rmem_tensor(cute.make_layout((1,)), Boolean)
    n_blocks = K // Int32(st.sf_vec)
    row_sf_bytes = K // Int32(st.sf_vec)
    state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, st.num_ab_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0 = work_tile.row0
        row_end = cutlass.min(work_tile.row_end, Tk)
        e = work_tile.e
        p = row0 + lane
        ok = (lane < Int32(tn)) & (p < row_end)
        if cutlass.const_expr(mSrc is not None):
            # the gather form: the token's own scale row
            pc = cutlass.min(cutlass.max(p, Int32(0)), mSrc.shape[0] - Int32(1))
            tk = cutlass.max(mSrc[pc], Int32(0)) // k_div
        else:
            shift = s_off_pad[e] - s_off[e]
            tk = cutlass.max(p + shift, Int32(0))
        state.reset_count()
        peek = Boolean(1)
        if state.count < k_tile_cnt:
            peek = sf_pipeline.producer_try_acquire(state)
        for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            sf_pipeline.producer_acquire(state, peek)
            gather_sf_row_padded(
                sf_atom,
                mSFA_flat,
                sSFA,
                lane,
                tk,
                ok,
                state.count,
                state.index,
                n_blocks,
                pcols,
                sf_stage_bytes,
                sf_groups,
                pred1,
                sf_linear,
                row_sf_bytes,
            )
            sf_pipeline.producer_commit(state)
            state.advance()
            peek = Boolean(1)
            if state.count < k_tile_cnt:
                peek = sf_pipeline.producer_try_acquire(state)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    sf_pipeline.producer_tail(state)
