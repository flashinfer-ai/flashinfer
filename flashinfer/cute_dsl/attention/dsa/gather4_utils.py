# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility helpers for CuTe DSL 2D Gather4 TMA loads.

CuTe DSL 4.6 exposes the Gather4 dialect operation and lowering, but its
``cpasync`` module does not expose a corresponding high-level copy operation.
Keep the low-level dependency isolated here until that wrapper is public.
"""

import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.nvgpu import tcgen05


_REQUIRED_GATHER4_IR_SYMBOLS = (
    "atom_make_non_exec_2d_gather4_tma_load",
    "GatherScatterTmaLoadEnum",
    "CopyAtomNonExec2DGather4TmaLoadType",
)


def _ensure_gather4_ir_support() -> None:
    """Require the low-level Gather4 API shipped by CuTe DSL 4.6 or newer."""
    missing = [
        name
        for name in _REQUIRED_GATHER4_IR_SYMBOLS
        if not hasattr(_cute_nvgpu_ir, name)
    ]
    if missing:
        raise RuntimeError(
            "HCA Gather4 requires nvidia-cutlass-dsl>=4.6.0; missing "
            + ", ".join(missing)
        )


class _CopyBulkTensor2DGather4G2SOp(cpasync.CopyBulkTensorTileG2SOp):
    """Local copy-op adapter for the public low-level Gather4 IR API."""

    def _get_description(self) -> str:
        return "cp.async GMEM -> SMEM bulk tensor Gather4 copy operation"

    def _to_ir(self):
        if self.cta_group == tcgen05.CtaGroup.ONE:
            return _cute_nvgpu_ir.GatherScatterTmaLoadEnum.sm_100
        if self.cta_group == tcgen05.CtaGroup.TWO:
            return _cute_nvgpu_ir.GatherScatterTmaLoadEnum.sm_100_2sm
        raise ValueError(f"unsupported CTA group for Gather4: {self.cta_group}")


@cute.jit
def make_gather4_2sm_tma_atom(
    gmem: cute.Tensor,
    smem_layout: cute.Layout,
    mma_tiler,
    tiled_mma: cute.TiledMma,
    gmem_coord_tensor: cute.Tensor,
):
    """Build a two-CTA Gather4 TMA atom for token-level absolute indices."""
    _ensure_gather4_ir_support()
    gather4_op = _CopyBulkTensor2DGather4G2SOp(cta_group=tcgen05.CtaGroup.TWO)
    ident = cute.make_identity_layout(gmem.shape)
    g_tile = cute.composition(ident, mma_tiler)
    cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
    cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
    cta_v_map = cute.select(cta_v_map, mode=[0, 2])
    cta_v_map = cute.zipped_divide(cta_v_map, (cta_mn, mma_tiler[1]))
    cta_v_map = cute.select(cta_v_map, mode=[0])

    smem_layout_ir = smem_layout.value if hasattr(smem_layout, "value") else smem_layout
    res = _cute_nvgpu_ir.atom_make_non_exec_2d_gather4_tma_load(
        gmem.value,
        gmem_coord_tensor.layout,
        smem_layout_ir,
        cta_v_map,
        gather4_op._to_ir(),
        num_multicast=1,
    )
    return cute.CopyAtom(
        gather4_op, cpasync.CopyBulkTensorTileG2SNonExecTrait(res[0])
    ), res[1]
