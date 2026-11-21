# Copyright (c) 2026 by FlashInfer team.
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

"""Host and device topology helpers for wider/flexible CGAs.

``umma_cta_count`` is the logical UMMA instruction atom width; the preferred and
fallback shapes are complete physical CUDA cluster shapes.  Supported wider
shapes use an even X dimension, so adjacent physical ranks are independent
2-CTA UMMA atoms.  For example, a 4x2 CGA is laid out as::

    physical Y
       1       [4 5] [6 7]       rank = x + dim_x * y
       0       [0 1] [2 3]       [a b] = one 2-CTA UMMA atom
                 0     1         logical atom X

There are three different masks for each TMA-to-UMMA operand.  They must not
be conflated:

``payload``
    CTAs whose SMEM receives the TMA data.
``full owners``
    Even ranks whose UMMA full barriers are armed.  Every payload rank maps to
    the leader of its adjacent atom: ``owner = rank & ~1``.
``empty release``
    Ranks signaled by UMMA consumers when a stage can be reused.

The following 4x2 examples use issuer rank 0. ``E`` is the number of consumer
arrivals expected by the producer's empty barrier::

    operand   payload P       full owners O   empty release R   E
    A         {0,4}           {0,4}           {0,1}             dim_y = 2
    B         {0,2}           {0,2}           {0,1}             dim_x / 2 = 2
    SFB       {0,1,2,3}       {0,2}           {0}               dim_x / 2 = 2

Mask reductions for the supported shapes make the local-versus-multicast
distinction explicit::

    shape   A(P/O/R/E)              B(P/O/R/E)              SFB(P/O/R/E)
    2x1     {0}/{0}/{0,1}/1         {0}/{0}/{0,1}/1         {0,1}/{0}/{0}/1
    4x1     {0}/{0}/{0,1}/1         {0,2}/{0,2}/{0,1}/2     {0..3}/{0,2}/{0}/2
    2x2     {0,2}/{0,2}/{0,1}/2     {0}/{0}/{0,1}/1         {0,1}/{0}/{0}/1
    4x2     {0,4}/{0,4}/{0,1}/2     {0,2}/{0,2}/{0,1}/2     {0..3}/{0,2}/{0}/2
    4x4     {0,4,8,12}/same/{0,1}/4 {0,2}/same/{0,1}/2      {0..3}/{0,2}/{0}/2

Rows other than Y=0 use the same formulas with their row base added.  A 2x1
SFB entry is relevant to the active fallback of a flexible wider kernel: that
compiled kernel still has one elected issuer.  A standalone legacy 2x1 kernel
does not compile the extended SFB path.
These tables describe unpartitioned SFB. Partitioned SFB has issuers X=0
and X=1 and releases both CTAs' empty barriers with the same arrival count.

Only a flexible launch queries its active physical dimensions on the device;
fixed wider shapes use compile-time dimensions.  Keeping all rank, election,
and mask math here prevents task ownership, TMA issue, and pipeline barriers
from silently drifting apart.
"""

from dataclasses import dataclass, replace
from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

from .batched_gemm_config import SfLayout, SfSmemToTmemCopy


class CgaPolicy(IntEnum):
    """CTA participation policy for one task/resource.

    Selected ranks for the diagrammed 4x2 CGA and its 2x1 fallback are::

        policy                    4x2 selected ranks   2x1 selected ranks
        ALL                       0,1,2,3,4,5,6,7      0,1
        ATOM_LEADER               0,2,4,6              0
        ONE_Y_ROW                 0,1,2,3              0,1
        FIRST_X_ATOM_PER_Y        0,1,4,5              0,1
        ONE_X_COLUMN              0,4                  0
        CTA_ZERO                  0                    0
        MCAST_A                   0,1,2,3              0,1
        MCAST_B                   0,1,4,5              0,1
        MCAST_SFB                 0,4                  0
        TMA_B_OR_FULL_OWNER       0,1,2,4,5,6          0,1

    The ``MCAST_*`` policies adapt to the active shape.  A and B become
    all-CTA policies when their multicast dimension collapses.  SFB remains
    one-X-column on a flexible 2x1 fallback because rank 0 multicasts the
    payload to rank 1.  Rank-only policies avoid querying active dimensions.
    """

    ALL = 0
    ATOM_LEADER = 1
    ONE_Y_ROW = 2
    FIRST_X_ATOM_PER_Y = 3
    ONE_X_COLUMN = 4
    CTA_ZERO = 5
    MCAST_A = 6
    MCAST_B = 7
    MCAST_SFB = 8
    # Combined B/SFB: both first-atom CTAs issue payloads, while every
    # even-X CTA arms its local full barrier. Count their union only once.
    TMA_B_OR_FULL_OWNER = 9


def _cga_policy_predicate(policy, dim_x, dim_y, x, y):
    """One ownership expression for host integers and device Int32 values.

    ``policy`` is always a compile-time value. Bitwise boolean operations
    keep the same expression valid for both Python and traced DSL scalars.
    Task selection, release-mask filtering, and WorkId counts all use it.
    """
    if isinstance(policy, tuple):
        result = False
        for member in policy:
            result = result | _cga_policy_predicate(member, dim_x, dim_y, x, y)
        return result
    if policy == CgaPolicy.ALL:
        return True
    if policy == CgaPolicy.ATOM_LEADER:
        return (x & 1) == 0
    if policy in (CgaPolicy.ONE_Y_ROW, CgaPolicy.MCAST_A):
        return (y == 0) | ((dim_y <= 1) if policy == CgaPolicy.MCAST_A else False)
    if policy in (CgaPolicy.FIRST_X_ATOM_PER_Y, CgaPolicy.MCAST_B):
        return x < 2
    if policy in (CgaPolicy.ONE_X_COLUMN, CgaPolicy.MCAST_SFB):
        return x == 0
    if policy == CgaPolicy.CTA_ZERO:
        return (x == 0) & (y == 0)
    if policy == CgaPolicy.TMA_B_OR_FULL_OWNER:
        return (x < 2) | ((x & 1) == 0)
    raise ValueError(f"Unknown CGA policy: {policy}")


def union_cga_policies(*policies):
    """Union task owners without counting a CTA once per logical operand."""
    members = set()
    for policy in policies:
        members.update(policy if isinstance(policy, tuple) else (policy,))
    if CgaPolicy.ALL in members:
        return CgaPolicy.ALL
    # These subsets are independent of the active preferred/fallback shape.
    if CgaPolicy.MCAST_B in members:
        members.discard(CgaPolicy.MCAST_SFB)
    if CgaPolicy.MCAST_B in members and CgaPolicy.ATOM_LEADER in members:
        members.difference_update((CgaPolicy.MCAST_B, CgaPolicy.ATOM_LEADER))
        members.add(CgaPolicy.TMA_B_OR_FULL_OWNER)
    if not members:
        return ()
    if len(members) == 1:
        return next(iter(members))
    return tuple(sorted(members))


class CgaMcastDirection(IntEnum):
    """Logical relation used to release a multicast pipeline's empty barrier.

    This is the FlashInfer equivalent of TRT-LLM/CUTLASS ``McastDirection``::

        FlashInfer        TRT-LLM name   operands   expected empty arrivals
        ALONG_Y           Row            A, SFA     NY = dim_y
        ACROSS_X_ATOMS    Col            B, SFB     NX = dim_x / atom_x

    TRT-LLM's diagrams call Y-varying communication ``Row`` because they draw
    X vertically and Y horizontally.  The FlashInfer names state the physical
    coordinate directly to avoid that orientation ambiguity.

    These directions describe separate operand pipelines. The optional shared
    A/B/SFA/SFB pipeline instead uses :func:`shared_tma_release`: a row/column
    union with ``NY + NX - 1`` arrivals, including both CTAs of each atom.
    """

    ALONG_Y = 0
    ACROSS_X_ATOMS = 1


def tma_mcast_direction(operand: str) -> CgaMcastDirection:
    """Map an operand to its empty-barrier multicast relation."""
    if operand == "a" or operand == "sfa":
        return CgaMcastDirection.ALONG_Y
    if operand == "b" or operand == "sfb":
        return CgaMcastDirection.ACROSS_X_ATOMS
    raise ValueError(f"Unknown CGA operand: {operand}")


def is_flexible_cga(cfg) -> bool:
    """Return whether one launch may use either configured physical shape."""
    return cfg.fallback_cluster_shape is not None


def has_multiple_umma_atoms(cfg) -> bool:
    """Return whether the preferred physical CGA contains multiple atoms."""
    dim_x, dim_y, dim_z = cfg.preferred_cluster_shape
    return dim_x * dim_y * dim_z > cfg.umma_cta_count


def _routed_operand(cfg, *, scale_factor: bool) -> str | None:
    """Return the logical operand supplied by the configured routed path."""
    has_route = cfg.has_routed_sfs if scale_factor else cfg.has_routed_act
    if not has_route:
        return None
    if scale_factor:
        return "sfb" if cfg.is_swap_ab else "sfa"
    return "b" if cfg.is_swap_ab else "a"


def _uses_compact_sfb_copy(cfg) -> bool:
    """Return whether SFB uses the per-CTA compact R8c4 LDS+STTM path."""
    return cfg.sfb_smem_to_tmem_copy == int(
        SfSmemToTmemCopy.LDS_STTM
    ) and cfg.smem_sfb_layout == int(SfLayout.R8c4)


def sfb_copy_uses_atom_leader(cfg) -> bool:
    """Return whether CopySfB is owned by one CTA per 2-CTA UMMA atom."""
    return cfg.uses_two_cta_mma and cfg.sfb_smem_to_tmem_copy == int(
        SfSmemToTmemCopy.UTCCP
    )


def needs_extended_tma_multicast(cfg, operand: str) -> bool:
    """Return whether one operand needs topology-aware TMA multicast.

    This compile-time boolean controls payload multicast and load-task
    election, not TMA-to-UMMA barrier routing.  Every multi-atom
    ``PipelineTmaUmma`` still needs atom-relative barriers even when this
    function returns ``False``.

    The decision table is::

        operand  topology trigger       additionally requires / excludes
        A        dim_y > 1              no CastA; no routed-LDGSTS A
        B        dim_x > 2              no routed-LDGSTS B
        SFA      dim_y > 1              SFA present; no CastA/routed SFA/
                                        local LDS+STTM copy
        SFB      dim_x > 1, with        SFs present; no routed SFB/compact
                 multi-atom or shared  R8c4 LDS+STTM copy
                 TMA pipeline

    Validation guarantees that a fallback divides the preferred shape.  An
    enabled flexible kernel therefore compiles one extended path whose masks
    and policies adapt to either active shape.  An unaffected operand retains
    its local payload/task path even when another operand multicasts.
    """
    dim_x, dim_y, _ = cfg.preferred_cluster_shape
    routed_operand = _routed_operand(cfg, scale_factor=operand.startswith("sf"))
    if operand == "a":
        return (
            dim_y > 1
            and not cfg.has_cast_a
            and not (routed_operand == "a" and not cfg.has_tma_route)
        )
    if operand == "b":
        return dim_x > 2 and not (routed_operand == "b" and not cfg.has_tma_route)
    if operand == "sfa":
        return (
            cfg.has_scale_factor_a
            and dim_y > 1
            and not cfg.has_cast_a
            and routed_operand != "sfa"
            and cfg.sfa_smem_to_tmem_copy != int(SfSmemToTmemCopy.LDS_STTM)
        )
    if operand == "sfb":
        # SFB is shared by both CTAs even in one 2x1 atom. The dense shared
        # pipeline multicasts to the pair. DKG distributes the slices across X.
        # Keep legacy independent-pipeline behavior unchanged.
        return (
            cfg.has_input_block_scaling
            and (
                has_multiple_umma_atoms(cfg)
                or cfg.dense_fuse_tma_pipeline
                or cfg.uses_routed_weight_pipeline
                or (cfg.fuse_operand_sf_loads and cfg.dense_partition_tma_loads)
            )
            and dim_x > 1
            and routed_operand != "sfb"
            and not _uses_compact_sfb_copy(cfg)
        )
    raise ValueError(f"Unknown CGA operand: {operand}")


def tma_payload_is_local(cfg, operand: str) -> bool | None:
    """Classify payload locality, or return None when the active CGA decides.

    Locality concerns payload receivers, not full-barrier owners. In
    particular, extended SFB still targets both CTAs in a single 2x1 atom.
    Non-extended loads already have a single, issuer-local payload bit.
    """
    if not needs_extended_tma_multicast(cfg, operand):
        return True
    if operand == "sfb":
        return False
    shapes = (cfg.preferred_cluster_shape,)
    if cfg._active_cga_shape is not None:
        shapes = (cfg._active_cga_shape,)
    elif cfg.fallback_cluster_shape is not None:
        shapes += (cfg.fallback_cluster_shape,)
    local = [
        shape[1] == 1 if operand in ("a", "sfa") else shape[0] == 2 for shape in shapes
    ]
    if all(local):
        return True
    if not any(local):
        return False
    return None


def input_tma_instruction_kind(cfg, operand: str, *, gather=False) -> str:
    """Select the existing CTA, CTA-local cluster, or two-CTA instruction."""
    if not gather and (
        (cfg.has_cast_a and operand in ("a", "sfa"))
        or (cfg.uses_two_cta_mma and operand == "sfb" and _uses_compact_sfb_copy(cfg))
    ):
        return "cluster"
    return "umma" if cfg.uses_two_cta_mma else "cta"


def input_tma_maps_barrier(cfg, operand: str, *, gather=False) -> bool:
    """Gather must explicitly map completion even with dense_local_barrier."""
    return input_tma_instruction_kind(cfg, operand, gather=gather) == "umma" and (
        gather
        or (not cfg.dense_local_barrier and (operand != "b" or cfg.split_b_across_ctas))
    )


def input_tma_umma_group(cfg):
    """Pipeline's tcgen05 enum differs from the TMA primitive's group enum."""
    return (
        cute.nvgpu.tcgen05.CtaGroup.TWO
        if cfg.uses_two_cta_mma
        else cute.nvgpu.tcgen05.CtaGroup.ONE
    )


@cute.jit
def input_tma_barrier_address(
    cfg: cutlass.Constexpr[object],
    operand: cutlass.Constexpr[str],
    barrier,
    gather: cutlass.Constexpr[bool] = False,
):
    if cutlass.const_expr(input_tma_maps_barrier(cfg, operand, gather=gather)):
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # Retain each emitter's existing arithmetic during centralization.
        # These all select the same adjacent-atom leader; rewriting signed
        # division as a bit mask would also change generated instructions.
        if cutlass.const_expr(gather):
            owner = rank & Int32(~1)
        elif cutlass.const_expr(
            operand != "b" and needs_extended_tma_multicast(cfg, operand)
        ):
            owner = (rank // Int32(2)) * Int32(2)
        else:
            owner = (rank >> Int32(1)) * Int32(2)
        return prims.cvta_to(prims.mapa(barrier, owner), prims.CvtaSpace.SHARED)
    return barrier


@cute.jit
def input_tma_full_armer_cta(cfg: cutlass.Constexpr[object]):
    """Use the same block-X expression as the existing UMMA pipeline."""
    if cutlass.const_expr(not cfg.uses_two_cta_mma):
        return True
    bidx, _, _ = cute.arch.block_idx()
    return bidx % cfg.umma_cta_count == 0


@cute.jit
def load_input_tma(
    cfg: cutlass.Constexpr[object],
    operand: cutlass.Constexpr[str],
    smem_dst,
    tma_desc,
    coords,
    barrier,
    l2_cache_hint=None,
):
    """Emit an input tile, preserving an optional per-expert weight cache hint."""
    if cutlass.const_expr(input_tma_instruction_kind(cfg, operand) == "cluster"):
        prims.cp_async_bulk_tensor_shared_cluster_global(
            smem_dst, tma_desc, coords, barrier, [], l2_cache_hint=l2_cache_hint
        )
    elif cutlass.const_expr(input_tma_instruction_kind(cfg, operand) == "umma"):
        mask = input_tma_payload_mask(cfg, operand)
        barrier = input_tma_barrier_address(cfg, operand, barrier)
        issue_tma_load(
            cfg,
            operand,
            smem_dst,
            tma_desc,
            coords,
            barrier,
            mask,
            l2_cache_hint=l2_cache_hint,
        )
    else:
        prims.cp_async_bulk_tensor_shared_cta_global(
            smem_dst, tma_desc, coords, barrier, l2_cache_hint=l2_cache_hint
        )


@cute.jit
def issue_tma_load(
    cfg: cutlass.Constexpr[object],
    operand: cutlass.Constexpr[str],
    smem_dst,
    tma_desc,
    coords,
    barrier,
    multicast_mask,
    l2_cache_hint=None,
):
    """Payload election is distinct from a pipeline's local full arming."""
    control = input_tma_control(cfg, operand)
    if cga_policy_selected(cfg, control.issuers):
        _issue_tma_load_impl(
            cfg,
            operand,
            smem_dst,
            tma_desc,
            coords,
            barrier,
            multicast_mask,
            l2_cache_hint=l2_cache_hint,
        )


@dsl_user_op
def _retained_a_cache_policy(*, loc=None, ip=None):
    """Prioritize activation reuse across neighboring N tiles."""
    return cutlass.Int64(
        llvm.inline_asm(
            cutlass.Int64.mlir_type,
            [],
            "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _issue_tma_load_impl(
    cfg: cutlass.Constexpr[object],
    operand: cutlass.Constexpr[str],
    smem_dst,
    tma_desc,
    coords,
    barrier,
    multicast_mask,
    l2_cache_hint=None,
):
    """Use unicast for issuer-local data without changing barrier routing."""
    # A per-expert weight policy takes precedence over activation retention.
    # Without an explicit policy, retain the native path's existing behavior.
    cache_hint = l2_cache_hint
    if cutlass.const_expr(
        l2_cache_hint is None and cfg.dense_retain_a and operand in ("a", "sfa")
    ):
        cache_hint = _retained_a_cache_policy()
    if cutlass.const_expr(tma_payload_is_local(cfg, operand) is not None):
        prims.cp_async_bulk_tensor_shared_cluster_global(
            smem_dst,
            tma_desc,
            coords,
            barrier,
            [],
            multicast_mask=(
                None
                if cutlass.const_expr(tma_payload_is_local(cfg, operand))
                else multicast_mask
            ),
            l2_cache_hint=cache_hint,
            group=cfg.cta_group,
        )
    else:
        # A flexible unspecialized body can multicast in the preferred CGA
        # and use unicast in its fallback. Reuse the actual payload mask.
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if multicast_mask == (Int32(1) << rank):
            prims.cp_async_bulk_tensor_shared_cluster_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
                [],
                l2_cache_hint=cache_hint,
                group=cfg.cta_group,
            )
        else:
            prims.cp_async_bulk_tensor_shared_cluster_global(
                smem_dst,
                tma_desc,
                coords,
                barrier,
                [],
                multicast_mask=multicast_mask,
                l2_cache_hint=cache_hint,
                group=cfg.cta_group,
            )


@dataclass(frozen=True)
class HostCgaTopology:
    """Pure-Python mirror of device rank, policy, and mask formulas.

    Operand masks are built from rank-bit, atom, row, column, and X-parity
    primitives. Tests use this model as an executable version of the module
    diagrams rather than duplicating mask arithmetic.
    """

    dim_x: int
    dim_y: int
    dim_z: int = 1

    def __post_init__(self) -> None:
        if self.dim_x <= 0 or self.dim_y <= 0 or self.dim_z != 1:
            raise ValueError(
                "CGA dimensions must have positive X/Y and Z=1, got "
                f"({self.dim_x}, {self.dim_y}, {self.dim_z})"
            )
        if self.dim_x * self.dim_y > 16:
            raise ValueError("CGA masks support at most 16 CTAs")

    def rank(self, x: int, y: int, z: int = 0) -> int:
        return x + self.dim_x * (y + self.dim_y * z)

    def rank_bit(self, x: int, y: int) -> int:
        """Return the mask bit for physical coordinate ``(x, y)``."""
        return 1 << self.rank(x, y)

    def atom_mask(self, x: int, y: int) -> int:
        """Return both ranks in the adjacent 2-CTA atom containing ``x``."""
        atom_x = x & ~1
        return self.rank_bit(atom_x, y) | self.rank_bit(atom_x + 1, y)

    def row_mask(self, y: int) -> int:
        """Return every CTA in physical Y row ``y``."""
        return sum(self.rank_bit(x, y) for x in range(self.dim_x))

    def column_mask(self, x: int) -> int:
        """Return every CTA at physical X coordinate ``x``."""
        return sum(self.rank_bit(x, y) for y in range(self.dim_y))

    def parity_mask(self, x: int, y: int) -> int:
        """Return one X-parity lane from every atom in physical row ``y``."""
        parity = x & 1
        return sum(
            self.rank_bit(atom_x + parity, y) for atom_x in range(0, self.dim_x, 2)
        )

    def selected(self, policy: CgaPolicy, x: int, y: int) -> bool:
        """Evaluate the documented policy table for one physical coordinate."""
        return _cga_policy_predicate(policy, self.dim_x, self.dim_y, x, y)

    def participant_count(self, policy: CgaPolicy) -> int:
        return sum(
            self.selected(policy, x, y)
            for y in range(self.dim_y)
            for x in range(self.dim_x)
        )

    def payload_mask_a(self, x: int) -> int:
        return self.column_mask(x)

    def payload_mask_b(self, x: int, y: int) -> int:
        return self.parity_mask(x, y)

    def payload_mask_sfb(self, y: int) -> int:
        return self.row_mask(y)

    @staticmethod
    def full_barrier_mask(payload_mask: int) -> int:
        # Fold each odd-rank payload bit onto its preceding even-rank owner,
        # then discard odd positions. This is the host form of the device
        # projection below.
        return (payload_mask | (payload_mask >> 1)) & 0x5555

    def release_mask_a(self, x: int) -> int:
        return self.atom_mask(x, 0)

    def release_mask_b(self, y: int) -> int:
        return self.atom_mask(0, y)

    def release_mask_sfb(self, y: int) -> int:
        return self.rank_bit(0, y)


@cute.jit
def runtime_cga_coords():
    """Return runtime physical CGA dimensions, rank, and `(x, y)` coordinates."""
    dim_x, dim_y, dim_z = cute.arch.block_in_cluster_dim()
    rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    x = rank % dim_x
    y = (rank // dim_x) % dim_y
    return dim_x, dim_y, dim_z, rank, x, y


@cute.jit
def tma_y_partition_offset(
    cfg: cutlass.Constexpr[object], extent: cutlass.Constexpr[int]
):
    """Partition a complete CTA slice among the active Y issuers."""
    if cutlass.const_expr(cfg.dense_partition_tma_loads):
        _, dim_y, _, _, _, y = active_cga_coords(cfg)
        return y * (Int32(extent) // dim_y)
    return Int32(0)


@cute.jit
def active_cga_coords(cfg: cutlass.Constexpr[object]):
    """Return active physical dimensions and CTA coordinates.

    Flexible launches select preferred/fallback dimensions at device runtime.
    Fixed launches, including legacy 1x1/2x1, materialize their configured
    dimensions as constants.
    """
    if cutlass.const_expr(cfg._active_cga_shape is not None):
        host_x, host_y, host_z = cfg._active_cga_shape
        dim_x, dim_y, dim_z = Int32(host_x), Int32(host_y), Int32(host_z)
    elif cutlass.const_expr(is_flexible_cga(cfg)):
        dim_x, dim_y, dim_z = cute.arch.block_in_cluster_dim()
    else:
        host_x, host_y, host_z = cfg.preferred_cluster_shape
        dim_x, dim_y, dim_z = Int32(host_x), Int32(host_y), Int32(host_z)
    rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    x = rank % dim_x
    y = (rank // dim_x) % dim_y
    return dim_x, dim_y, dim_z, rank, x, y


@cute.jit
def cga_policy_selected(
    cfg: cutlass.Constexpr[object], policy: cutlass.Constexpr[object]
) -> Boolean:
    """Return whether the current CTA owns a task with ``policy``.

    ``policy`` is a compile-time constant stored on :class:`CgaTask`; only the
    active dimensions of a flexible launch are runtime values.  The function
    is deliberately split into three groups:

    * ``ALL`` has no topology dependency.
    * ``ATOM_LEADER`` and ``CTA_ZERO`` need only physical rank.  Avoiding
      :func:`active_cga_coords` here also avoids a runtime dimension query.
    * Row/column and ``MCAST_*`` policies need active X/Y.  Their primitive
      booleans are computed once and reused below.

    ``MCAST_A`` and ``MCAST_B`` expand to all CTAs when their multicast
    dimension collapses on a fallback. ``MCAST_SFB`` keeps one issuer column
    whenever X > 1 because its fallback rank 0 still multicasts to rank 1.
    See :class:`CgaPolicy` for the complete 4x2/2x1 truth table.
    """
    if cutlass.const_expr(policy == int(CgaPolicy.ALL)):
        return Boolean(_cga_policy_predicate(policy, 1, 1, 0, 0))
    if cutlass.const_expr(
        policy in (int(CgaPolicy.ATOM_LEADER), int(CgaPolicy.CTA_ZERO))
    ):
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        return Boolean(_cga_policy_predicate(policy, 1, 1, rank, 0))
    dim_x, dim_y, _, _, x, y = active_cga_coords(cfg)
    return Boolean(_cga_policy_predicate(policy, dim_x, dim_y, x, y))


@cute.jit
def cga_policy_mask(cfg: cutlass.Constexpr[object], policy: cutlass.Constexpr[object]):
    """Active producer/consumer ranks, derived from the task-selection rule."""
    dim_x, dim_y, _, _, _, _ = active_cga_coords(cfg)
    mask = Int32(0)
    # Validation limits physical CGAs to X,Y<=4. Constant coordinates let
    # fixed/specialized launches fold this mask entirely at compile time.
    for yy in cutlass.range_constexpr(4):
        for xx in cutlass.range_constexpr(4):
            selected = (
                _cga_policy_predicate(policy, dim_x, dim_y, Int32(xx), Int32(yy))
                & (Int32(xx) < dim_x)
                & (Int32(yy) < dim_y)
            )
            mask = mask | (Int32(selected) << (Int32(xx) + dim_x * Int32(yy)))
    return mask


@cute.jit
def _rank_bit(rank: Int32) -> Int32:
    """Return one 16-bit CGA mask bit."""
    return Int32(1) << rank


@cute.jit
def _atom_mask(dim_x: Int32, x: Int32, y: Int32) -> Int32:
    """Return the adjacent 2-CTA atom containing physical ``(x, y)``."""
    atom_leader = (x & Int32(~1)) + dim_x * y
    return _rank_bit(atom_leader) | _rank_bit(atom_leader + Int32(1))


@cute.jit
def _column_mask(dim_x: Int32, dim_y: Int32, x: Int32) -> Int32:
    """Return the active physical X column used by A/SFA multicast."""
    result = Int32(0)
    for yy in cutlass.range_constexpr(4):
        if Int32(yy) < dim_y:
            result = result | _rank_bit(x + dim_x * Int32(yy))
    return result


@cute.jit
def _row_mask(dim_x: Int32, y: Int32) -> Int32:
    """Return the active physical Y row used by SFB multicast."""
    result = Int32(0)
    for xx in cutlass.range_constexpr(4):
        if Int32(xx) < dim_x:
            result = result | _rank_bit(Int32(xx) + dim_x * y)
    return result


@cute.jit
def _x_parity_mask(dim_x: Int32, x: Int32, y: Int32) -> Int32:
    """Return the issuer's X parity from every atom in one row for B."""
    result = Int32(0)
    parity = x & Int32(1)
    for atom_x in cutlass.range_constexpr(0, 4, 2):
        if Int32(atom_x) < dim_x:
            result = result | _rank_bit(Int32(atom_x) + parity + dim_x * y)
    return result


@cute.jit
def _project_payload_to_barrier_owners(payload_mask: Int32) -> Int32:
    """Project every payload bit to its adjacent even-rank barrier owner."""
    return (payload_mask | (payload_mask >> Int32(1))) & Int32(0x5555)


def input_tma_arrival_count(scope, dim_x, dim_y, atom_size=2):
    """Shared host/device dependency count, including one-atom degeneracies."""
    count = 1
    if scope in ("y", "xy"):
        count = dim_y
    if scope in ("x", "xy"):
        count += dim_x // atom_size - 1
    return count


@cute.jit
def input_tma_release(
    cfg: cutlass.Constexpr[object],
    scope: cutlass.Constexpr[str],
    participants: cutlass.Constexpr[object],
):
    """Consumer dependency relation intersected with actual empty waiters."""
    dim_x, dim_y, _, rank, x, y = active_cga_coords(cfg)
    if cutlass.const_expr(scope == "cta"):
        return _rank_bit(rank), Int32(1)
    atom_x = x & Int32(~1)
    relation = _atom_mask(dim_x, x, y)
    if cutlass.const_expr(scope in ("y", "xy")):
        relation = _column_mask(dim_x, dim_y, atom_x) | _column_mask(
            dim_x, dim_y, atom_x + Int32(1)
        )
    if cutlass.const_expr(scope in ("x", "xy")):
        relation = (
            relation if cutlass.const_expr(scope == "xy") else Int32(0)
        ) | _row_mask(dim_x, y)
    count = Int32(input_tma_arrival_count(scope, dim_x, dim_y, cfg.umma_cta_count))
    return relation & cga_policy_mask(cfg, participants), count


@cute.jit
def shared_tma_release(cfg: cutlass.Constexpr[object]):
    control = input_tma_group_control(cfg)
    return input_tma_release(cfg, control.release_scope, control.participants)


@cute.jit
def split_tma_release(cfg: cutlass.Constexpr[object], weights: cutlass.Constexpr[bool]):
    control = input_tma_group_control(cfg, "b" if weights else "a")
    return input_tma_release(cfg, control.release_scope, control.participants)


@cute.jit
def input_tma_payload_mask(
    cfg: cutlass.Constexpr[object], operand: cutlass.Constexpr[str]
):
    if cutlass.const_expr(not needs_extended_tma_multicast(cfg, operand)):
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        return _rank_bit(rank)
    dim_x, dim_y, _, rank, x, y = active_cga_coords(cfg)
    if cutlass.const_expr(operand in ("a", "sfa")):
        return _column_mask(dim_x, dim_y, x)
    elif cutlass.const_expr(operand == "b"):
        return _x_parity_mask(dim_x, x, y)
    else:
        return _row_mask(dim_x, y)


@cute.jit
def cga_tma_masks(cfg: cutlass.Constexpr[object], operand: cutlass.Constexpr[str]):
    """Payload destinations, full-barrier recipients, empty targets/count.

    Full-barrier recipients do not imply local task participation: elected
    remote-arming producers execute those arrivals on their behalf. Byte
    credits remain the complete received stage, independent of issuer count.
    """
    control = input_tma_control(cfg, operand)
    payload = input_tma_payload_mask(cfg, operand)
    participants = control.empty_participants
    release, count = input_tma_release(cfg, control.release_scope, participants)
    owners = payload
    if cutlass.const_expr(control.full_arming in ("local_atom", "remote_atom")):
        owners = _project_payload_to_barrier_owners(payload)
    return payload, owners, release, count


@cute.jit
def complete_input_tma_bytes(
    cfg: cutlass.Constexpr[object],
    operand: cutlass.Constexpr[str],
    barrier,
    tx_bytes: Int32,
):
    """Complete omitted payload bytes at every receiving full-barrier owner.

    Call once per elected issuer for bytes it did not copy. This only credits
    transactions: producer arrivals, expected bytes and empty releases stay
    unchanged, including when operands share a ring. The caller must ensure
    the omitted SMEM payload is unobserved; this does not zero-fill it.
    """
    _, owners, _, _ = cga_tma_masks(cfg, operand)
    max_ctas = cfg.preferred_cluster_shape[0] * cfg.preferred_cluster_shape[1]
    if cutlass.const_expr(cfg.fallback_cluster_shape is not None):
        max_ctas = max(
            max_ctas, cfg.fallback_cluster_shape[0] * cfg.fallback_cluster_shape[1]
        )
    for owner in cutlass.range_constexpr(max_ctas):
        if (owners & (Int32(1) << Int32(owner))) != Int32(0):
            remote = prims.cvta_to(
                prims.mapa(barrier, Int32(owner)), prims.CvtaSpace.SHARED
            )
            prims.inline_ptx_hl(
                "mbarrier.complete_tx.relaxed.cluster.shared::cluster.b64 [{$r0}], {$r1};",
                read_only_args=[remote.data_ptr(), tx_bytes],
            )


@dataclass(frozen=True)
class TaskParticipation:
    """One real task's contribution to the persistent work queue."""

    name: str
    warps: int
    policy: CgaPolicy


@dataclass(frozen=True)
class InputTmaControl:
    """Resolved input ownership, independent of transaction byte accounting.

    ``issuers`` execute payload instructions. ``participants`` also includes
    CTAs which *locally* arm a full barrier; remotely armed recipients are not
    producers. ``release_scope`` describes consumer-to-producer dependencies.
    A shared physical ring replaces member dependencies with their union.

    ``local_async`` retains the stock CTA-local thread-arrival protocol (CastA,
    LDS/STTM scales and routed scale gather). Its bytes and per-warp full
    arrivals remain owned by PipelineConfig, rather than the UMMA mask model.
    """

    operand: str
    issuers: object
    participants: object
    release_scope: str
    full_arming: str
    group: str = ""
    ring_participants: object = None

    @property
    def empty_participants(self):
        return (
            self.participants
            if self.ring_participants is None
            else self.ring_participants
        )


def _independent_input_tma_control(cfg, operand: str) -> InputTmaControl:
    if operand not in ("a", "b", "sfa", "sfb"):
        raise ValueError(f"Unknown input operand: {operand}")
    routed = _routed_operand(cfg, scale_factor=operand.startswith("sf")) == operand
    local_async = (
        (cfg.has_cast_a and operand in ("a", "sfa"))
        or (
            operand == "sfa"
            and cfg.sfa_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
        )
        or (operand == "sfb" and (_uses_compact_sfb_copy(cfg) or routed))
    )
    non_tma = routed and (
        not cfg.uses_tma_routed_sfs
        if operand.startswith("sf")
        else not cfg.has_tma_route
    )
    if non_tma or local_async:
        return InputTmaControl(
            operand,
            CgaPolicy.ALL,
            CgaPolicy.ALL,
            "cta",
            "non_tma" if non_tma else "local_async",
        )

    extended = needs_extended_tma_multicast(cfg, operand)
    partitioned = cfg.dense_partition_tma_loads and (
        operand == "sfb" or (operand in ("a", "sfa") and not cfg.has_routed_act)
    )
    if cfg.partitions_routed_tma_a and operand == "a":
        partitioned = True
    issuers = CgaPolicy.ALL
    scope = "atom" if cfg.uses_two_cta_mma else "cta"
    if extended:
        scope = "y" if operand in ("a", "sfa") else "x"
        issuers = {
            "a": CgaPolicy.ALL if partitioned else CgaPolicy.MCAST_A,
            "sfa": CgaPolicy.ALL if partitioned else CgaPolicy.MCAST_A,
            "b": CgaPolicy.MCAST_B,
            "sfb": CgaPolicy.MCAST_B if partitioned else CgaPolicy.MCAST_SFB,
        }[operand]

    if partitioned:
        # Partitioned A gather and dense A/SFA slices are supplied by all Y
        # CTAs, with the complete byte credit armed once in each local atom.
        if operand in ("a", "sfa"):
            return InputTmaControl(operand, issuers, CgaPolicy.ALL, scope, "local_atom")
    return InputTmaControl(
        operand,
        issuers,
        issuers,
        scope,
        "remote_atom"
        if extended
        else ("local_atom" if cfg.uses_two_cta_mma else "local_cta"),
    )


@dataclass(frozen=True)
class InputTmaGroupControl:
    """One physical shared ring, regardless of member/task alias count."""

    operands: tuple[str, ...]
    participants: object
    release_scope: str
    primary_operand: str
    single_full_arrival: bool
    shared_loader: bool


def input_tma_group_control(cfg, split_operand: str = "") -> InputTmaGroupControl:
    if split_operand not in ("", "a", "b"):
        raise ValueError(f"Unknown input TMA group: {split_operand}")
    operands = {"": ("a", "b", "sfa", "sfb"), "a": ("a", "sfa"), "b": ("b", "sfb")}[
        split_operand
    ]
    single_full_arrival = bool(
        has_multiple_umma_atoms(cfg)
        or cfg.dense_fuse_tma_loads
        or cfg.uses_routed_weight_pipeline
    )
    members = tuple(_independent_input_tma_control(cfg, op) for op in operands)
    # Remote recipients never enter this union. Shared groups explicitly arm
    # locally, so add the local armers once alongside member payload issuers.
    participants = union_cga_policies(
        *(m.issuers for m in members),
        CgaPolicy.ATOM_LEADER if single_full_arrival else (),
    )
    return InputTmaGroupControl(
        operands,
        participants,
        {"": "xy", "a": "y", "b": "x"}[split_operand],
        "a" if split_operand == "a" else "b",
        single_full_arrival,
        bool(cfg.dense_fuse_tma_loads or cfg.uses_routed_weight_pipeline),
    )


def input_tma_control(cfg, operand: str) -> InputTmaControl:
    member = _independent_input_tma_control(cfg, operand)
    group = None
    if cfg.dense_fuse_tma_pipeline:
        group = (
            ("a" if operand in ("a", "sfa") else "b")
            if cfg.dense_split_tma_pipeline
            else ""
        )
    elif cfg.uses_routed_weight_pipeline and operand in ("b", "sfb"):
        group = "b"
    if group is None:
        return member
    physical = input_tma_group_control(cfg, group)
    participants = member.issuers
    if physical.shared_loader:
        participants = physical.participants
    elif physical.single_full_arrival and operand == physical.primary_operand:
        participants = union_cga_policies(member.issuers, CgaPolicy.ATOM_LEADER)
    return replace(
        member,
        participants=participants,
        release_scope=physical.release_scope,
        full_arming="local_atom",
        group=group or "ab",
        ring_participants=physical.participants,
    )


def operand_cga_policy(cfg, operand: str):
    """The same participant set drives task selection, releases and WorkId."""
    return input_tma_control(cfg, operand).participants


def operand_task_policy(cfg, *operands):
    return union_cga_policies(*(operand_cga_policy(cfg, op) for op in operands))


def clc_task_participation(cfg) -> tuple[TaskParticipation, ...]:
    """Resolve actual tasks once; fused resource aliases add no participants."""
    w = cfg.warp_layout
    tasks = []

    def add(name, warps, policy=CgaPolicy.ALL):
        if warps:
            tasks.append(TaskParticipation(name, warps, policy))

    if cfg.uses_routed_weight_pipeline:
        add(
            "load_a",
            w.num_load_a_warps,
            operand_task_policy(cfg, "a", "sfa")
            if cfg.fuse_operand_sf_loads
            else operand_cga_policy(cfg, "a"),
        )
        if not cfg.fuse_operand_sf_loads:
            add("load_sfa", w.num_load_sfa_warps, operand_cga_policy(cfg, "sfa"))
        add(
            "load_b_sf",
            w.num_load_b_warps,
            input_tma_group_control(cfg, "b").participants,
        )
    elif cfg.dense_fuse_tma_loads:
        add(
            "load_ab_sf",
            w.num_load_a_warps,
            input_tma_group_control(
                cfg, "a" if cfg.dense_split_tma_pipeline else ""
            ).participants,
        )
        if cfg.dense_split_tma_pipeline:
            add(
                "load_b_sf",
                w.num_load_b_warps,
                input_tma_group_control(cfg, "b").participants,
            )
    else:
        add(
            "load_a",
            w.num_load_a_warps,
            operand_task_policy(cfg, "a", "sfa")
            if cfg.fuse_operand_sf_loads
            else operand_cga_policy(cfg, "a"),
        )
        add(
            "load_b",
            w.num_load_b_warps,
            operand_task_policy(cfg, "b", "sfb")
            if cfg.fuse_operand_sf_loads
            else operand_cga_policy(cfg, "b"),
        )
    add("epilogue", w.num_epilogue_warps)
    add("padding", w.num_padding_warps)
    add("sync", w.num_sync_warps)
    add("cast_a", w.num_cast_a_warps)
    add("mma", w.num_mma_warps, CgaPolicy.ATOM_LEADER)
    add("workid", w.num_workid_warps, CgaPolicy.CTA_ZERO)
    if not cfg.fuse_operand_sf_loads and not cfg.dense_fuse_tma_loads:
        if cfg.has_scale_factor_a:
            add("load_sfa", w.num_load_sfa_warps, operand_cga_policy(cfg, "sfa"))
        if cfg.has_input_block_scaling:
            add("load_sfb", w.num_load_sfb_warps, operand_cga_policy(cfg, "sfb"))
    if cfg.use_deepseek_fp8:
        add("load_sfab", w.num_load_sfab_warps)
    if cfg.has_input_block_scaling and not cfg.fuse_sf_copy_to_mma:
        if cfg.use_combined_sfab_copy:
            add("copy_sfab", w.num_copy_sfa_warps, CgaPolicy.ATOM_LEADER)
        else:
            add(
                "copy_sfa",
                w.num_copy_sfa_warps,
                CgaPolicy.ATOM_LEADER
                if cfg.sfa_smem_to_tmem_copy == int(SfSmemToTmemCopy.UTCCP)
                else CgaPolicy.ALL,
            )
            add(
                "copy_sfb",
                w.num_copy_sfb_warps,
                CgaPolicy.ATOM_LEADER
                if sfb_copy_uses_atom_leader(cfg)
                else CgaPolicy.ALL,
            )
    return tuple(tasks)


def host_clc_consumer_threads(cfg, dim_x: int, dim_y: int) -> int:
    """Exact arrivals for the physical CGA, from resolved task ownership."""
    topo = HostCgaTopology(dim_x, dim_y)
    return 32 * sum(
        t.warps * topo.participant_count(t.policy) for t in clc_task_participation(cfg)
    )


def static_queue_consumer_threads(cfg, cta_rank: int) -> int:
    """Local empty-barrier arrivals, excluding the static producer warp.

    MMA consumes work in every atom leader. Multicast loader ownership may
    also vary by X/Y. Count each physical CTA's actual consumers; a missing
    arrival would leave its local queue permanently waiting.
    """
    dim_x, dim_y, _ = cfg.preferred_cluster_shape
    topo = HostCgaTopology(dim_x, dim_y)
    x, y = cta_rank % dim_x, cta_rank // dim_x
    tasks = [t for t in clc_task_participation(cfg) if t.name != "workid"]
    return 32 * sum(t.warps for t in tasks if topo.selected(t.policy, x, y))


def verifier_clc_consumer_threads(cfg) -> int:
    """Project actual tasks onto the stock verifier's atom-only CTA model."""
    if not has_multiple_umma_atoms(cfg):
        return host_clc_consumer_threads(cfg, cfg.umma_cta_count, 1)
    return 32 * sum(
        t.warps * (1 if t.name == "workid" else cfg.umma_cta_count)
        for t in clc_task_participation(cfg)
    )


@cute.jit
def active_clc_consumer_threads(cfg: cutlass.Constexpr[object]) -> Int32:
    """Device evaluation of the same task list for the active physical CGA."""
    tasks = clc_task_participation(cfg)
    total = Int32(0)
    for i in cutlass.range_constexpr(len(tasks)):
        participants = cute.arch.popc(cga_policy_mask(cfg, tasks[i].policy))
        total += Int32(tasks[i].warps) * participants
    return total * Int32(32)
