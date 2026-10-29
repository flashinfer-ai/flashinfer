# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel-template registry — six-dimension support funnel.

Whether a kernel runs depends on: gpu arch · kernel template · tile config
(geometry) · graph type · mma type · other graph info. Tile configs are PURE
GEOMETRY (including cta_group and the MMA mode); the template supplies
pipeline, graph type, mainloop), so one geometry runs on several templates.

:meth:`KernelTemplate.accepts` funnels cheapest-first:
  1. graph type + pipeline-family + mainloop axis
  2. mma type per TEMPLATE FAMILY (:data:`MMA_TYPE_SUPPORT` existence sets) +
     the rare family×combo×GPU exceptions (:data:`MMA_GPU_ARCH_SPECIAL_CASES`)
  3. tile config — accepted geometries by PREDICATE over CATALOG (cta_group
     constraints), never a hand-maintained list
  4. other graph info — mainloop op scope + TMA alignment

Capability checks are reused from ``compiler._check_*`` (lazy import; compiler
doesn't import this module). :func:`select_template` drives the compiler's
template-file selection."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path

from . import arch_family
from .fusion_ir import BINARY_OPS, UNARY_OPS, FusionChain, MoeSwapAbSpec, swap_ab as swap_ab_graph
from .tile_config import CATALOG, TileConfig, as_mma_tile_k, as_pipeline, config_class_for_pipeline


def _pipeline_from_file(template_file: str) -> str:
    """Leading ``sm<NNN>`` pipeline-family token of a template filename — the
    template pairs with the config family of the same pipeline (config_sm<NNN>)
    and runs on that family's SM ranges (:data:`PIPELINE_ARCH_RANGES`)."""
    m = re.match(r"(sm\d+)_", template_file)
    if m is None:
        raise ValueError(f"cannot determine pipeline family from template {template_file!r} " f"(expected a leading 'sm<NNN>_' token)")
    return m.group(1)


# Active-GPU SM ranges per template pipeline — half-open [lo, hi) segments,
# SM = major*10 + minor. A family may support several DISJOINT segments
PIPELINE_ARCH_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "sm100": ((100, 120),),
    "sm103": ((103, 110),),
    "sm120": ((100, 130),),
}

# The per-arch SOURCE TREE (``cudnn/gemm/frost/<family>/``, see ``arch_family``)
# a pipeline family's templates ship in. sm103 is a tcgen05 pipeline and rides
# the sm100 tree. A new pipeline needs an entry here AND in PIPELINE_ARCH_RANGES.
PIPELINE_FAMILY: dict[str, str] = {
    "sm100": "sm100",
    "sm103": "sm100",
    "sm120": "sm120",
}


def template_path(template_file: str) -> Path:
    """Where ``template_file`` ships: the ``kernel_templates/`` of the arch tree
    its pipeline prefix names (:data:`PIPELINE_FAMILY`). Resolved off the
    TEMPLATE, not off whichever compiler copy is running, because one family may
    render another's template where both run (an sm120 config on an SM 10.x
    part, or a render-only host with no GPU at all)."""
    return arch_family.template_dir(PIPELINE_FAMILY[_pipeline_from_file(template_file)]) / template_file


# SM ranges whose dense FP8 / block-scale MMA issues a 64-byte K (half the
# instruction count of sm100's 32). SILICON, not a pipeline -- an sm100-pipeline
# kernel on a 10.7 part gets it, exactly like the B collector and the 576-column
# TMEM. Shared by the dense and block-scale config gates and auto selection.
MMA_INST_K64_ARCH_RANGES: tuple[tuple[int, int], ...] = ((107, 110),)

# Pointwise ops a mainloop-fusion template can transform in SMEM.
_SUPPORTED_MAINLOOP_OPS: frozenset[str] = frozenset(UNARY_OPS) | frozenset(BINARY_OPS)


# Dimension 1: graph type


class GraphType(Enum):
    """The kind of computation the graph expresses; each template supports
    exactly one. ``CONVOLUTION`` is a placeholder (no template yet)."""

    MATMUL = "matmul"
    BLOCK_SCALE_MATMUL = "block_scale_matmul"
    MOE = "moe"
    MOE_SWAP_AB = "moe_grouped_matmul_swap_ab"
    MOE_BLOCK_SCALE = "moe_block_scale"
    MOE_BLOCK_SCALE_SWAP_AB = "moe_grouped_block_scale_matmul_swap_ab"
    CONVOLUTION = "convolution"  # placeholder — no template yet


def classify_graph_type(chain: FusionChain) -> GraphType:
    """Stage-1 classifier: which graph type this chain is."""
    if isinstance(chain.moe, MoeSwapAbSpec):
        return GraphType.MOE_BLOCK_SCALE_SWAP_AB if chain.has_block_scale else GraphType.MOE_SWAP_AB
    if chain.has_moe and chain.has_block_scale:
        return GraphType.MOE_BLOCK_SCALE
    if chain.has_block_scale:
        return GraphType.BLOCK_SCALE_MATMUL
    if chain.has_moe:
        return GraphType.MOE
    return GraphType.MATMUL


# Dimension 2: mma type × arch (graph-type-level, config/template independent).
# Unified MMA-type × arch support — SINGLE source of truth (merges the old
# compiler matmul table + block-scale per-side case list). Indexed by graph
# type; value = {mma_type_key -> supported arch ranges}. Key shape differs per
# graph type (matmul = (a, b, accum); block-scale = full per-side spec) but the
# lookup is one code path.


def _matmul_mma_type(chain: FusionChain) -> tuple:
    mm = chain.matmul
    return (mm.a_dtype, mm.b_dtype, mm.accum_dtype)


def _block_scale_mma_type(chain: FusionChain) -> tuple:
    mm = chain.matmul
    bs = chain.block_scale
    assert bs is not None
    return (
        mm.a_dtype,
        bs.sf_dtype_a,
        bs.block_size_a,
        bs.sfa_reorder,
        bs.dequant_compute_a,
        bs.dequant_out_a,
        mm.b_dtype,
        bs.sf_dtype_b,
        bs.block_size_b,
        bs.sfb_reorder,
        bs.dequant_compute_b,
        bs.dequant_out_b,
        mm.accum_dtype,
    )


def _bs_key(a: str, sfa: str, b: str, sfb: str, kblk: int) -> tuple:
    """Block-scale mma-type key from the varying parts (data + SF dtypes,
    K-block). All cases share reorder F8_128x4, fp32 dequant compute/out, fp32
    accumulate, A block=(1,kblk) / B block=(kblk,1)."""
    return (
        a,
        sfa,
        (1, kblk),
        "F8_128x4",
        "fp32",
        "fp32",
        b,
        sfb,
        (kblk, 1),
        "F8_128x4",
        "fp32",
        "fp32",
        "fp32",
    )


_BLOCK_SCALE_CASES = frozenset(
    {
        _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 16),
        _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32),
        _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 16),
        _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32),
        _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 16),
        _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 32),
        _bs_key("fp8_e4m3", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32),
        _bs_key("fp8_e4m3", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32),
        _bs_key("fp8_e5m2", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32),
        _bs_key("fp8_e5m2", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32),
        _bs_key("fp4_e2m1", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32),
        _bs_key("fp4_e2m1", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32),
        _bs_key("fp8_e4m3", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32),
        _bs_key("fp8_e5m2", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32),
    }
)

_MATMUL_CASES = frozenset(
    {
        ("bf16", "bf16", "fp32"),
        ("fp16", "fp16", "fp32"),
        ("int8", "int8", "int32"),
        ("fp8_e4m3", "fp8_e4m3", "fp32"),
        ("fp8_e4m3", "fp8_e5m2", "fp32"),
        ("fp8_e5m2", "fp8_e4m3", "fp32"),
        ("fp8_e5m2", "fp8_e5m2", "fp32"),
    }
)

# MoE grouped matmul reuses its base pipeline's MMA machinery verbatim — combo
# lookups fold to the base graph type, so MMA_TYPE_SUPPORT never carries MoE rows.
_MMA_BASE_GRAPH_TYPE: dict[GraphType, GraphType] = {
    GraphType.MOE: GraphType.MATMUL,
    GraphType.MOE_SWAP_AB: GraphType.MATMUL,
    GraphType.MOE_BLOCK_SCALE: GraphType.BLOCK_SCALE_MATMUL,
    GraphType.MOE_BLOCK_SCALE_SWAP_AB: GraphType.BLOCK_SCALE_MATMUL,
}

# Per-graph-type mma-type key extractors (key SHAPES differ per graph type, so
# the key spaces never collide).
_MMA_KEY_FNS: dict[GraphType, object] = {
    GraphType.MATMUL: _matmul_mma_type,
    GraphType.BLOCK_SCALE_MATMUL: _block_scale_mma_type,
}

# {template_pipeline: {GraphType: frozenset of supported mma-type keys}} — which
# dtype combos each TEMPLATE FAMILY implements, arch-FREE (whether the active
# GPU can execute a supported combo is the family's stage-0 SM gate, except
# the rare MMA_GPU_ARCH_SPECIAL_CASES below). A new family (e.g. an sm120
# mxfp8 pipeline) adds its own entry; it never inherits another family's set.
MMA_TYPE_SUPPORT: dict[str, dict[GraphType, frozenset]] = {
    "sm100": {
        GraphType.MATMUL: _MATMUL_CASES,
        GraphType.BLOCK_SCALE_MATMUL: _BLOCK_SCALE_CASES,
    },
    "sm103": {
        GraphType.BLOCK_SCALE_MATMUL: frozenset(
            {
                _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 16),
                _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32),
                _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 16),
                _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32),
                _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 16),
                _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 32),
            }
        ),
    },
    "sm120": {
        GraphType.MATMUL: _MATMUL_CASES,
        GraphType.BLOCK_SCALE_MATMUL: frozenset(
            {
                _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 16),
                _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32),
                _bs_key("fp8_e4m3", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32),
                _bs_key("fp8_e4m3", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32),
                _bs_key("fp8_e5m2", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32),
                _bs_key("fp8_e5m2", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32),
                _bs_key("fp4_e2m1", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32),
                _bs_key("fp4_e2m1", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32),
                _bs_key("fp8_e4m3", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32),
                _bs_key("fp8_e5m2", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32),
                _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 16),
                _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32),
            }
        ),
    },
}

# The ONE home for checks that need template SM family × mma dtype × ACTUAL
# GPU arch together: (family, combo) pairs whose MMA instruction exists only
# on specific family members — NARROWER than the family's SM ranges. Most
# combos never appear here (stage-0 family gate + the existence sets decide).
# Values are half-open [lo, hi) segments, same shape as PIPELINE_ARCH_RANGES.
MMA_GPU_ARCH_SPECIAL_CASES: dict[tuple[str, tuple], tuple[tuple[int, int], ...]] = {
    ("sm100", ("int8", "int8", "int32")): ((100, 101), (110, 111)),
    ("sm100", _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 16)): ((107, 110),),
    ("sm100", _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 32)): ((107, 110),),
    ("sm100", _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32)): ((107, 110),),
    ("sm103", _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 16)): ((107, 110),),
    ("sm103", _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 32)): ((107, 110),),
    ("sm103", _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32)): ((107, 110),),
    ("sm120", _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 16)): ((120, 130),),
    ("sm120", _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32)): ((120, 130),),
    ("sm120", _bs_key("fp8_e4m3", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32)): ((120, 130),),
    ("sm120", _bs_key("fp8_e4m3", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32)): ((120, 130),),
    ("sm120", _bs_key("fp8_e5m2", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32)): ((120, 130),),
    ("sm120", _bs_key("fp8_e5m2", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32)): ((120, 130),),
    ("sm120", _bs_key("fp4_e2m1", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32)): ((120, 130),),
    ("sm120", _bs_key("fp4_e2m1", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32)): ((120, 130),),
    ("sm120", _bs_key("fp8_e4m3", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32)): ((120, 130),),
    ("sm120", _bs_key("fp8_e5m2", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32)): ((120, 130),),
    ("sm120", _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 16)): ((120, 130),),
    ("sm120", _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32)): ((120, 130),),
}


def linear_token_sf_reject(dim, stride, rows, scale_k) -> str | None:
    """GATHER SF layout support, shared by graph checks and runtime binding."""
    if dim is None or stride is None or tuple(dim) != (1, rows, scale_k):
        return f"MoE GATHER token SF must have shape [1, {rows}, {scale_k}]"
    if len(stride) != 3 or stride[2] != 1 or stride[1] < ((scale_k + 15) // 16) * 16 or stride[1] % 16:
        return "MoE GATHER token SF needs contiguous K scales and a row stride padded to a multiple of 16 bytes"
    return None


def moe_gather_sf_reject(chain: FusionChain) -> str | None:
    """Token SF layouts supported by the block-scale GATHER producer."""
    if not (chain.has_moe and chain.has_block_scale and chain.moe.mode == "gather"):
        return None
    bs = chain.block_scale
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    fake = bs.fake_dequant_b if swapped else bs.fake_dequant_a
    reorder = bs.sfb_reorder if swapped else bs.sfa_reorder
    if not fake:
        if reorder is not None:
            return "MoE block-scale GATHER requires token SF reorder=NONE; F8_128x4 is not supported"
        dim, stride = (bs.sfb_dim, bs.sfb_stride) if swapped else (bs.sfa_dim, bs.sfa_stride)
        if dim is None or stride is None or len(dim) != 3 or len(stride) != 3:
            return "MoE GATHER token SF must have rank-3 shape and strides"
        if swapped:
            dim, stride = (dim[0], dim[2], dim[1]), (stride[0], stride[2], stride[1])
        return linear_token_sf_reject(dim, stride, dim[1], chain.matmul.K // bs.block_size)
    return None


def mma_arch_reject(chain: FusionChain, graph_type: GraphType, template_pipeline: str) -> str | None:
    """Stage 2: does the ``template_pipeline`` family's pipeline support the
    graph's MMA type, and — for the rare :data:`MMA_GPU_ARCH_SPECIAL_CASES`
    (family, combo) pairs — does the ACTUAL GPU have the instruction?
    ``None`` = yes. Independent of tile config / cta_group."""
    from . import compiler as C

    base_type = _MMA_BASE_GRAPH_TYPE.get(graph_type, graph_type)
    key_fn = _MMA_KEY_FNS.get(base_type)
    if key_fn is None:
        return f"graph type {graph_type.value!r} has no kernel pipeline yet"
    cases = MMA_TYPE_SUPPORT.get(template_pipeline, {}).get(base_type)
    if cases is None:
        return f"the {template_pipeline} family has no {graph_type.value} pipeline"
    key = key_fn(chain)
    if chain.has_moe and chain.has_block_scale and chain.moe.mode == "gather":
        reason = moe_gather_sf_reject(chain)
        if reason is not None:
            return reason
        # The gather producer lowers the linear token SF to the same MMA atom.
        # Preserve the instruction/architecture table, including special cases.
        key = list(key)
        bs = chain.block_scale
        key[9 if isinstance(chain.moe, MoeSwapAbSpec) else 3] = "F8_128x4"
        if bs.fake_dequant_a:
            key[3] = "F8_128x4"
        if bs.fake_dequant_b:
            key[9] = "F8_128x4"
        key = tuple(key)
    if key not in cases:
        if base_type is GraphType.MATMUL:
            mm = chain.matmul
            return f"the {template_pipeline} {graph_type.value} pipeline does not support " f"input/acc dtype combo {mm.a_dtype}x{mm.b_dtype}->{mm.accum_dtype}"
        return f"the {template_pipeline} {graph_type.value} pipeline does not support " f"this configuration: mma type {key}"

    if template_pipeline == "sm120" and base_type is GraphType.BLOCK_SCALE_MATMUL:
        for tmpl in TEMPLATES:
            if tmpl.pipeline == template_pipeline and tmpl.graph_type is graph_type and isinstance(tmpl, Sm120KernelTemplate):
                reason = tmpl.block_scale_form_reject(chain) or tmpl.block_scale_toolchain_reject(chain)
                if reason is not None:
                    return reason
    special = MMA_GPU_ARCH_SPECIAL_CASES.get((template_pipeline, key))
    if special is not None:
        arch = C._current_arch()
        if arch is not None and not any(lo <= arch < hi for lo, hi in special):
            spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in special)
            return f"the {template_pipeline} {key[0]}x{key[1]} MMA instruction exists only " f"on {spans}, but the active GPU is sm_{arch}"
    return None


# Kernel template — owns the pipeline family, graph type and mainloop axes


@dataclass(frozen=True)
class KernelTemplate:
    """One kernel template. Carries the execution-strategy axes the pure-geometry
    config does NOT (pipeline, graph_type, mainloop). The MMA mode is NOT here:
    it is a TileConfig axis, and which modes a pipeline issues is a fact of the
    config family (``_CTA_GROUPS_BY_PIPELINE``), not of the template."""

    file: str  # template filename; ships under <family>/kernel_templates/ (see template_path)
    pipeline: str  # pipeline family from the filename; pairs with config_<pipeline>
    graph_type: GraphType  # the single graph type this template supports
    # ``None`` = take the config's. The warp count is a config axis; a template
    # overrides it only where its own specialization differs from what the
    # geometry asks for -- the mainloop variant adds 4 warps to the SAME
    # geometry, and which of the two runs is decided by the CHAIN.
    warps_per_cta: int | None = None
    # Per-CTA SMEM this template holds back off the top of the ab pipeline: the
    # scheduler ring, every smem barrier, the TMEM base address and -- on the
    # MoE templates -- the per-CTA TMA tensormap scratch. A template fact, not a
    # geometry one, so it lives here rather than on the config.
    smem_fixed_reserve: int = 2048
    supports_mainloop_fusion: bool = False
    supports_multi_gemm: bool = True

    @property
    def block_scale(self) -> bool:
        """True iff this template consumes block-scaled (FP4/FP8 + SF) inputs."""
        return self.graph_type in (
            GraphType.BLOCK_SCALE_MATMUL,
            GraphType.MOE_BLOCK_SCALE,
            GraphType.MOE_BLOCK_SCALE_SWAP_AB,
        )

    @property
    def family(self) -> str:
        """The arch tree (``cudnn/gemm/frost/<family>/``) whose compiler renders
        this template (:data:`PIPELINE_FAMILY`); every other tree declines it."""
        return PIPELINE_FAMILY[self.pipeline]

    @property
    def path(self) -> Path:
        """The template source on disk (:func:`template_path`)."""
        return template_path(self.file)

    # stage 0: active-GPU SM ranges (from the template's pipeline-family prefix)

    def arch_active_reject(self) -> str | None:
        """``None`` if the active GPU's SM is in one of the template pipeline
        family's ranges (:data:`PIPELINE_ARCH_RANGES`), or no GPU is visible
        (render-only / CI). Per-ARCH-FAMILY gate (vs :func:`mma_arch_reject`,
        which gates the graph's MMA type)."""
        from . import compiler as C

        arch = C._current_arch()
        ranges = PIPELINE_ARCH_RANGES[self.pipeline]
        if arch is not None and not any(lo <= arch < hi for lo, hi in ranges):
            spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in ranges)
            return f"template {self.file} runs only on {spans}, " f"but the active GPU is sm_{arch}"
        return None

    # stage 1: pipeline / graph-type / mainloop axes

    def _axis_reject(self, chain: FusionChain, config: TileConfig, graph_type: GraphType) -> str | None:
        cfg_cls = config_class_for_pipeline(self.pipeline)
        if not isinstance(config, cfg_cls):
            return f"config {config.name} is not a {cfg_cls.__name__} " f"(template {self.file} pairs with config_{self.pipeline})"
        if graph_type is not self.graph_type:
            return f"graph_type {graph_type.value} != " f"template graph_type {self.graph_type.value}"
        if chain.has_mainloop_fusion != self.supports_mainloop_fusion:
            return f"graph mainloop_fusion={chain.has_mainloop_fusion} != " f"template supports_mainloop_fusion={self.supports_mainloop_fusion}"
        if chain.is_multi_gemm and not self.supports_multi_gemm:
            return f"template {self.file} does not support multi-GEMM " f"({chain.num_gemms} parallel GEMMs); only the 1ctamma CLC " "template does this pass"
        return None

    # stage 3: tile-config gates (the geometry, including its cta_group)

    def _config_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        from . import compiler as C

        # cta_group constraints (geometry, so they read the config).
        try:
            C._check_cta_group_geometry(config)
            C._check_mma_n_dim(chain, config)
        except NotImplementedError as e:
            return str(e)
        try:
            if self.block_scale:
                from .tile_config import validate_block_scale_config

                bs = chain.block_scale
                assert bs is not None
                data_elem_bits = max(C.DTYPE_BITS[bs.a_dtype], C.DTYPE_BITS[bs.b_dtype])
                cta_k_elems = config.cta_tile_k_bytes * 8 // data_elem_bits
                validate_block_scale_config(config, bs.block_size, cta_k_elems)
            else:
                C._check_dtype_config_compat(chain, config)
        except (ValueError, NotImplementedError) as e:
            return str(e)
        return None

    # stage 4a: per-template capability hook

    def _extra_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        """Template-SPECIFIC constraints beyond the shared gates (base = none;
        mainloop overrides). Future variable-MMA / per-template caps attach here."""
        return None

    # stage 4: other graph info

    def _other_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        extra = self._extra_reject(chain, config)
        if extra is not None:
            return extra
        from . import compiler as C

        try:
            C._check_input_alignment(chain)
            if not self.block_scale:
                # Block-scale skips only the OUTPUT vec-bytes gate here (its
                # jit path applies it with block-scale-specific handling); the
                # TMA input-alignment rule above is universal.
                C._compute_output_vec_bytes(chain)
        except (ValueError, NotImplementedError) as e:
            return str(e)
        return None

    # full accept/reject: the four-stage funnel

    def accepts(self, chain: FusionChain, config: TileConfig) -> str | None:
        """``None`` if this template can compile (chain, config); else the first
        stage's rejection reason. Cheapest-first (short-circuits):
        pipeline/graph-type/mainloop → mma-type×arch → tile-config → other."""
        if chain.has_moe and chain.moe.mode == "combine" and self.pipeline not in ("sm100", "sm120"):
            return "MoE COMBINE is supported only by the SM100 and SM120 pipelines"
        gt = classify_graph_type(chain)
        return (
            self.arch_active_reject()
            or self._axis_reject(chain, config, gt)
            or mma_arch_reject(chain, gt, self.pipeline)
            or self._config_reject(chain, config)
            or self._other_reject(chain, config)
        )

    def active_reject(self, config: TileConfig, chain: FusionChain | None = None) -> str | None:
        """The gates a JIT path applies once it has picked this template: the
        active GPU's SM range, then capabilities a pure-geometry config can ask
        for that this template does not implement, and — when the caller passes
        the ``chain`` — the template-specific scope (:meth:`_extra_reject`)."""
        reason = self.arch_active_reject()
        if reason is None and chain is not None:
            reason = self._extra_reject(chain, config)
        return reason

    def candidate_configs(self, chain: FusionChain) -> tuple[TileConfig, ...]:
        """Catalog geometries this template accepts for ``chain`` — by
        predicate filter, never hand-maintained."""
        return tuple(c for c in CATALOG if self.accepts(chain, c) is None)


class MainloopKernelTemplate(KernelTemplate):
    """A mainloop-fusion template. Every pre-MMA op must be in
    :data:`_SUPPORTED_MAINLOOP_OPS` (can't be tripped by constructible input
    today, but keeps the contract on the template)."""

    def _extra_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        for side, ops in (("A", chain.mainloop_a_ops), ("B", chain.mainloop_b_ops)):
            for op in ops:
                if op.op not in _SUPPORTED_MAINLOOP_OPS:
                    return f"{self.file} cannot fuse mainloop op {op.op!r} on " f"operand {side} (supported: unary + scalar-aux binary)"
        return None


def sm120_padded_e2m1_sides(chain: FusionChain) -> tuple[bool, bool]:
    """Which side (A, B) of a mixed MXFP4 x MXFP8 block-scale pair the sm120 MoE
    block-scale templates load through TMA's *padded* e2m1 form (``.b4x16_p64``).

    OPT-IN: ``(False, False)`` unless the chain carries ``block_scale.padded_e2m1``
    (``jit_from_cudnn_graph(..., padded_e2m1=True)``). The default is the packed
    form (``.b4x16``, 64-byte rows, fragment registers built from the packed
    words), which measures level with the padded one on the kernel and ahead of
    it end to end once the TMA descriptors carry the 128-byte L2 promotion
    (#1880): the padded SMEM tile is twice its GMEM payload, so the ring loses
    stages.

    ``kind::mxf8f6f4`` reads e2m1 from bits [5:2] of an 8-bit container. The
    padded form lands every 16 values in a 16-byte SMEM chunk (8 B payload + 8 B
    zero), so the row is the fp8 side's 128 bytes and
    ``ldmatrix.m8n16.x4.shared.b8x16.b4x16_p64`` expands the fragment in hardware
    (one shift per register follows) instead of the LDS + PRMT/shift/mask chain
    that builds it from packed words. Its TMA box must be exactly 64 payload
    bytes: a 128-element K tile of a K that is a multiple of 128, so a shorter K
    (any multiple of 32, e.g. gpt-oss's 2880) stays packed. The gathered token
    (``mode == "gather"``, TMA ``tile::gather4``) also stays packed on every K.
    A request no side can honor is a decline (``sm120_padded_e2m1_reject``),
    never a silent fallback.
    """
    bs = chain.block_scale
    if bs is None or not bs.padded_e2m1 or not bs.mixed_width or chain.matmul.K % 128:
        return False, False
    gather = chain.has_moe and chain.moe.mode == "gather"
    token_is_b = chain.has_moe and isinstance(chain.moe, MoeSwapAbSpec)
    padded_a = bs.a_dtype == "fp4_e2m1" and not (gather and not token_is_b)
    padded_b = bs.b_dtype == "fp4_e2m1" and not (gather and token_is_b)
    return padded_a, padded_b


def sm120_padded_e2m1_reject(chain: FusionChain) -> str | None:
    """Why a ``padded_e2m1`` request cannot be honored on ``chain``, or ``None``
    (no request, or at least one e2m1 side takes the padded form -- a request is
    honored on every side the hardware form allows)."""
    bs = chain.block_scale
    if bs is None or not bs.padded_e2m1:
        return None
    if not bs.mixed_width:
        return f"padded_e2m1 applies to the e2m1 side of a mixed MXFP4 x MXFP8 block-scale pair only; got {bs.a_dtype} x {bs.b_dtype}"
    if chain.matmul.K % 128:
        return (
            "padded_e2m1 needs K a multiple of 128 (a 128-element K tile is one 64-byte TMA .b4x16_p64 box); "
            f"got K={chain.matmul.K}, which only the packed form runs"
        )
    if not any(sm120_padded_e2m1_sides(chain)):
        return "padded_e2m1: the only e2m1 operand is the gathered token (TMA tile::gather4), which stays packed on every K"
    return None


def _sm120_block_scale_row_bytes(chain: FusionChain, kb: int) -> tuple[int, int]:
    """SMEM bytes of one K row per side of an sm120 block-scale stage: one
    ``kb`` (128 B) span, except the packed e2m1 side of a mixed pair, which holds
    the fp8 K tile in half the bytes (its padded form, and every other operand,
    holds ``kb``)."""
    bs = chain.block_scale
    padded_a, padded_b = sm120_padded_e2m1_sides(chain)
    a_kb = kb // 2 if bs.mixed_width and bs.a_dtype == "fp4_e2m1" and not padded_a else kb
    b_kb = kb // 2 if bs.mixed_width and bs.b_dtype == "fp4_e2m1" and not padded_b else kb
    return a_kb, b_kb


class Sm120KernelTemplate(KernelTemplate):
    COMPUTE_WARPS = 8  # WARPS_M * WARPS_N in every sm120 template

    def multi_gemm_ab_stages(self, chain: FusionChain, config: TileConfig) -> tuple[int, int, int]:
        from .dtypes import DTYPE_BYTES
        from .tile_config import _SM120_STG_STAGE_ELEMS, _sm_smem_budget_bytes, smem_ab_stages

        staging = 4 * _SM120_STG_STAGE_ELEMS * self.COMPUTE_WARPS
        if self.block_scale:
            bs = chain.block_scale
            # The K tile counts fp8 elements unless both operands are e2m1, so an
            # SMEM K row is one cta_tile_k_bytes span -- except the packed e2m1
            # side of a mixed pair, whose row holds the fp8 K-tile in half the
            # bytes (its padded form holds the whole span: sm120_padded_e2m1_sides).
            kb = config.cta_tile_k_bytes
            a_kb, b_kb = _sm120_block_scale_row_bytes(chain, kb)
            sf_k = kb * 8 // (4 if bs.is_fp4 else 8) // bs.block_size
            # A one-sided graph's raw-FP8 side reads an identity scale register:
            # no SF box in the stage.
            sfa_bytes = 0 if bs.fake_dequant_a else -(-config.cta_tile_m // 128) * 128 * sf_k
            sfb_bytes = 0 if bs.fake_dequant_b else -(-config.cta_tile_n // 128) * 128 * sf_k
            per_stage = chain.num_a_operands * (config.cta_tile_m * a_kb + sfa_bytes) + chain.num_b_operands * (config.cta_tile_n * b_kb + sfb_bytes) + 16
            swapped = isinstance(chain.moe, MoeSwapAbSpec)
            if chain.has_moe and chain.moe.mode == "gather" and not (bs.fake_dequant_b if swapped else bs.fake_dequant_a):
                # 32 B per routed token row of the tile, per token operand (the
                # token is B after the swap-AB lowering) -- the linear token
                # scales' scratch, absent when the token is the raw-FP8 side.
                staging += (config.cta_tile_n if swapped else config.cta_tile_m) * 32 * (chain.num_b_operands if swapped else chain.num_a_operands)
        else:
            _m, cta_smem_n, _k = config.cta_smem_tile_mnk(DTYPE_BYTES[chain.matmul.a_dtype])
            per_stage = (chain.num_a_operands * config.cta_tile_m + chain.num_b_operands * cta_smem_n) * config.cta_tile_k_bytes + 16
        stages = smem_ab_stages(per_stage, smem_fixed_reserve=self.smem_fixed_reserve, extra_smem_bytes=staging)
        budget = _sm_smem_budget_bytes()
        while stages > 0 and self.multi_gemm_smem_bytes(chain, config, stages) > budget:
            stages -= 1
        return stages, per_stage, staging

    def multi_gemm_smem_bytes(self, chain: FusionChain, config: TileConfig, stages: int) -> int:
        from .dtypes import DTYPE_BYTES
        from .tile_config import _SM120_STG_STAGE_ELEMS

        def up(offset: int, align: int) -> int:
            return -(-offset // align) * align

        offset = 1024
        kb = config.cta_tile_k_bytes
        if self.block_scale:
            bs = chain.block_scale
            sf_k = kb * 8 // (4 if bs.is_fp4 else 8) // bs.block_size
            # The same per-side rows as multi_gemm_ab_stages: the packed e2m1 side
            # of a mixed pair holds the fp8 K-tile in half the bytes (its opt-in
            # padded form the whole span: sm120_padded_e2m1_sides), and the ring
            # this sizes (sA_elems * ab_stages) is laid out in those bytes.
            a_kb, b_kb = _sm120_block_scale_row_bytes(chain, kb)
            a_tile, b_tile = config.cta_tile_m * a_kb, config.cta_tile_n * b_kb
            sf_tiles = (-(-config.cta_tile_m // 128) * 128 * sf_k, -(-config.cta_tile_n // 128) * 128 * sf_k)
        else:
            cta_smem_m, cta_smem_n, _k = config.cta_smem_tile_mnk(DTYPE_BYTES[chain.matmul.a_dtype])
            a_tile, b_tile = cta_smem_m * kb, cta_smem_n * kb
            sf_tiles = None
        for count, tile in ((chain.num_a_operands, a_tile), (chain.num_b_operands, b_tile)):
            for _ in range(count):
                offset = up(offset, 1024) + stages * tile
        if sf_tiles is not None:
            for count, tile in zip((chain.num_a_operands, chain.num_b_operands), sf_tiles):
                for _ in range(count):
                    offset = up(offset, 512) + stages * tile
        offset = up(offset, 1024) + 4 * _SM120_STG_STAGE_ELEMS * self.COMPUTE_WARPS
        if self.block_scale and chain.has_moe and chain.moe.mode == "gather":
            swapped = isinstance(chain.moe, MoeSwapAbSpec)
            rows, tokens = (config.cta_tile_n, chain.num_b_operands) if swapped else (config.cta_tile_m, chain.num_a_operands)
            for _ in range(tokens):
                offset = up(offset, 128) + rows * 32
            for _ in range(tokens):
                offset = up(offset, 8) + 8
        return offset

    def multi_gemm_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        if not chain.is_multi_gemm:
            return None
        stages, per_stage, staging = self.multi_gemm_ab_stages(chain, config)
        if stages < 1:
            tiles = f"{chain.num_a_operands} A + {chain.num_b_operands} B tiles" + (" with their scale-factor boxes" if self.block_scale else "")
            # The block-scale K tile is pinned at 128 bytes (validate_block_scale_config_sm120).
            advice = "pick a smaller CTA tile" if self.block_scale else "pick a smaller CTA tile or K"
            return (
                f"one AB stage of {per_stage} B ({tiles}) plus the "
                f"{staging}-B epilogue staging does not fit the SMEM budget of the sm120 multi-GEMM kernel; "
                f"{advice}"
            )
        return None

    def gather_sf_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        if not (self.block_scale and chain.has_moe and chain.moe.mode == "gather"):
            return None
        bs = chain.block_scale
        sf_k4 = config.cta_tile_k_bytes * 8 // (4 if bs.is_fp4 else 8) // bs.block_size // 4
        swapped = isinstance(chain.moe, MoeSwapAbSpec)
        if bs.fake_dequant_b if swapped else bs.fake_dequant_a:
            return None  # a raw-FP8 token has no scales to repack
        token_rows, axis = (config.cta_tile_n, "N") if swapped else (config.cta_tile_m, "M")
        if (token_rows * sf_k4) % 32:
            return (
                f"GATHER repacks the token scales in passes of 32 SF words, but the {token_rows}-row token tile "
                f"(CTA {axis}) holds {token_rows * sf_k4} per K-tile ({sf_k4} per row); pick a wider token tile"
            )
        return None

    DENSE_BLOCK_SCALE_FILE = "sm120_block_scale_matmul.py"
    DENSE_UNISSUED_FP4_SCALES = {
        (16, "fp8_e8m0"): "kind::mxf4nvf4 scale_vec::4X .ue8m0 as inline PTX",
        (32, "fp8_e4m3"): "kind::mxf4nvf4 scale_vec::4X .ue4m3 with each block-32 scale byte in both of its block-16 slots",
    }

    def block_scale_form_reject(self, chain: FusionChain) -> str | None:
        bs = chain.block_scale
        if bs is None or self.file != self.DENSE_BLOCK_SCALE_FILE:
            return None
        if bs.fake_dequant_a or bs.fake_dequant_b:
            return (
                "the sm120 block_scale_matmul pipeline does not support one-sided dequantization: fake scale-factor "
                f"identity handling is not implemented in {self.file} (the sm120 MoE block-scale templates feed the "
                "raw FP8 operand an identity scale)"
            )
        if bs.mixed_width:
            return (
                f"{self.file} does not support mixed-width block-scale operands ({bs.a_dtype} x {bs.b_dtype}): it "
                "loads A and B in one packed width (the sm120 MoE block-scale templates give each side its own)"
            )
        issue = self.DENSE_UNISSUED_FP4_SCALES.get((bs.block_size, bs.sf_dtype)) if bs.both_fp4 else None
        if issue is not None:
            return (
                f"{self.file} does not support fp4_e2m1 data with {bs.sf_dtype} scales at block {bs.block_size}: its "
                f"warp MMA has no such form yet (the sm120 MoE block-scale templates issue {issue})"
            )
        return None

    PTX_ISA_FLOORS = {
        (16, "fp8_e8m0"): ("kind::mxf4nvf4 scale_vec::4X .ue8m0", (9, 1), (13, 1)),
    }

    @staticmethod
    @lru_cache(maxsize=1)
    def dsl_cuda_version() -> tuple[int, int] | None:
        try:
            from cutlass.base_dsl.version_info import CUDA_VERSION
        except Exception:  # noqa: BLE001 -- an absent / unusable DSL is reported by probe_cutedsl
            return None
        return (CUDA_VERSION.major, CUDA_VERSION.minor)

    def block_scale_toolchain_reject(self, chain: FusionChain) -> str | None:
        bs = chain.block_scale
        if bs is None or not self.block_scale or not bs.both_fp4:
            return None
        floor = self.PTX_ISA_FLOORS.get((bs.block_size, bs.sf_dtype))
        cuda = self.dsl_cuda_version() if floor is not None else None
        if cuda is None or cuda >= floor[2]:
            return None
        form, (isa_major, isa_minor), (need_major, need_minor) = floor
        return (
            f"the sm120 block-scale templates issue fp4_e2m1 data with {bs.sf_dtype} scales at block {bs.block_size} "
            f"as mma.sync {form} (PTX ISA {isa_major}.{isa_minor}), which needs a CUDA {need_major}.{need_minor}+ "
            f"toolchain, but the installed CuTe DSL compiles with CUDA {cuda[0]}.{cuda[1]} (nvidia-cutlass-dsl uses "
            f"its cu13 toolchain only under a CUDA 13.x driver)"
        )

    def warp_layout_reject(self, config: TileConfig) -> str | None:
        compute_warps = (config.cta_tile_m // config.warp_tile_m) * (config.cta_tile_n // config.warp_tile_n)
        warps = self.warps_per_cta or config.warps_per_cta
        if compute_warps % 4 or compute_warps + 4 > warps:
            return (
                f"{self.file}: {compute_warps} compute warps do not fill whole warpgroups beside a producer "
                f"warpgroup (warps_per_cta={warps}); the register hand-off (setmaxnreg) needs a multiple of "
                f"4 compute warps, at most warps_per_cta - 4"
            )
        return None

    def block_quant_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        if not chain.quants:
            return None
        from .sm120 import compiler as C120

        if self.block_scale and not chain.has_moe:
            return C120._DENSE_BLOCK_SCALE_QUANT_DECLINE
        try:
            C120._check_block_quant_supported(chain, C120._epi_vec_bytes(chain, config), config)
        except (NotImplementedError, ValueError) as e:
            return str(e)
        return None

    def active_reject(self, config: TileConfig, chain: FusionChain | None = None) -> str | None:
        return super().active_reject(config, chain) or self.warp_layout_reject(config)

    @staticmethod
    def moe_combine_reject(chain: FusionChain) -> str | None:
        if not (chain.has_moe and chain.moe.mode == "combine"):
            return None
        if chain.num_gemms != 1 or chain.ops or chain.quants or chain.reductions or len(chain.output_specs) != 1:
            return "MoE COMBINE requires one GEMM and one final output without post-combine operations"
        if chain.matmul.accum_dtype != "fp32" or chain.output_dtype not in ("fp32", "fp16", "bf16"):
            return "MoE COMBINE requires FP32 MMA accumulation and FP32/FP16/BF16 output"
        return None

    def _other_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        # Last, after the shared alignment / output-width gates -- where the
        # sm120 prechecks run the quant gates too.
        return super()._other_reject(chain, config) or self.block_quant_reject(chain, config)

    def output_dtype_reject(self, chain: FusionChain) -> str | None:
        if chain.output_dtype == "fp4_e2m1" or any(spec.dtype == "fp4_e2m1" for spec in chain.output_specs):
            return f"{self.file} does not support fp4 output"
        return None

    def _extra_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        from .dtypes import DTYPE_BITS, DTYPE_BYTES

        _mc = self.moe_combine_reject(chain)
        if _mc is not None:
            return f"{self.file}: {_mc}"

        _wl = self.warp_layout_reject(config)
        if _wl is not None:
            return _wl
        mm = chain.matmul
        if mm.a_major == "m" and DTYPE_BITS[mm.a_dtype] not in (16, 8):
            return f"{self.file} loads MN-major operands with ldmatrix.trans (b16) / " f"ldmatrix.m16n16.trans.b8; {mm.a_dtype} A must be K-major"
        if mm.b_major == "n" and DTYPE_BITS[mm.b_dtype] not in (16, 8):
            return f"{self.file} loads MN-major operands with ldmatrix.trans (b16) / " f"ldmatrix.m16n16.trans.b8; {mm.b_dtype} B must be K-major"
        if mm.a_major == "m" or mm.b_major == "n":
            elem_bytes = DTYPE_BYTES[mm.a_dtype]
            group = config.cta_tile_k_bytes // elem_bytes
            cta_smem_m, cta_smem_n, _ = config.cta_smem_tile_mnk(elem_bytes)
            if mm.a_major == "m" and (cta_smem_m < group or cta_smem_m % group):
                return f"{self.file}: M-major A SMEM extent {cta_smem_m} is not a " f"whole number of {group}-element swizzle groups"
            if mm.b_major == "n" and (cta_smem_n < group or cta_smem_n % group):
                return f"{self.file}: N-major B SMEM extent {cta_smem_n} is not a " f"whole number of {group}-element swizzle groups"
        _od = self.output_dtype_reject(chain)
        if _od is not None:
            return _od
        try:
            _mg = self.multi_gemm_reject(chain, config)
        except NotImplementedError as e:  # no device to size the SMEM ring against (render-only / CI host)
            return f"{self.file}: {e}"
        if _mg is not None:
            return f"{self.file}: {_mg}"
        _gs = self.gather_sf_reject(chain, config)
        if _gs is not None:
            return f"{self.file}: {_gs}"
        return None


# Registry — one entry per template file (20 today). A geometry config expands
# across these via `candidates`. mainloop lives HERE; cta_group is geometry.


def _mm(
    file: str,
    *,
    graph_type: GraphType = GraphType.MATMUL,
    warps_per_cta: int | None = None,
    smem_fixed_reserve: int = 2048,
    supports_mainloop_fusion: bool = False,
    supports_multi_gemm: bool = True,
    template_cls: "type[KernelTemplate] | None" = None,
) -> KernelTemplate:
    pipeline = _pipeline_from_file(file)
    if pipeline not in PIPELINE_ARCH_RANGES:
        raise KeyError(
            f"template {file!r}: pipeline family {pipeline!r} has no SM-range entry in " f"PIPELINE_ARCH_RANGES — add one when introducing a new family"
        )
    if pipeline not in PIPELINE_FAMILY:
        raise KeyError(
            f"template {file!r}: pipeline family {pipeline!r} has no arch-tree entry in " f"PIPELINE_FAMILY — say which cudnn/gemm/frost/<family>/ it ships in"
        )
    cls = template_cls or (MainloopKernelTemplate if supports_mainloop_fusion else KernelTemplate)
    return cls(
        file=file,
        pipeline=pipeline,
        graph_type=graph_type,
        warps_per_cta=warps_per_cta,
        smem_fixed_reserve=smem_fixed_reserve,
        supports_mainloop_fusion=supports_mainloop_fusion,
        supports_multi_gemm=supports_multi_gemm,
    )


TEMPLATES: tuple[KernelTemplate, ...] = (
    _mm("sm100_matmul.py"),
    _mm("sm100_matmul_mainloop.py", warps_per_cta=12, supports_mainloop_fusion=True, supports_multi_gemm=False),
    _mm(
        "sm100_block_scale_matmul.py",
        graph_type=GraphType.BLOCK_SCALE_MATMUL,
    ),
    _mm(
        "sm100_moe_grouped_matmul_fwd.py",
        # + the per-CTA TMA tensormap scratch the per-group descriptor patch needs.
        smem_fixed_reserve=4096,
        graph_type=GraphType.MOE,
    ),
    _mm(
        "sm100_moe_grouped_block_scale_matmul_fwd.py",
        # + the per-CTA TMA tensormap scratch the per-group descriptor patch needs.
        smem_fixed_reserve=4096,
        graph_type=GraphType.MOE_BLOCK_SCALE,
    ),
    _mm(
        "sm100_moe_grouped_matmul_fwd_swap_ab.py",
        smem_fixed_reserve=4096,
        graph_type=GraphType.MOE_SWAP_AB,
    ),
    _mm(
        "sm100_moe_grouped_block_scale_matmul_fwd_swap_ab.py",
        smem_fixed_reserve=4096,
        graph_type=GraphType.MOE_BLOCK_SCALE_SWAP_AB,
    ),
    _mm(
        "sm103_block_scale_matmul.py",
        graph_type=GraphType.BLOCK_SCALE_MATMUL,
        supports_multi_gemm=False,
    ),
    _mm(
        "sm120_matmul.py",
        supports_multi_gemm=True,
        template_cls=Sm120KernelTemplate,
    ),
    _mm(
        "sm120_block_scale_matmul.py",
        graph_type=GraphType.BLOCK_SCALE_MATMUL,
        supports_multi_gemm=False,
        template_cls=Sm120KernelTemplate,
    ),
    _mm(
        "sm120_moe_grouped_matmul_fwd.py",
        graph_type=GraphType.MOE,
        supports_multi_gemm=True,
        template_cls=Sm120KernelTemplate,
    ),
    _mm(
        "sm120_moe_grouped_block_scale_matmul_fwd.py",
        graph_type=GraphType.MOE_BLOCK_SCALE,
        supports_multi_gemm=True,
        template_cls=Sm120KernelTemplate,
    ),
    _mm(
        "sm120_moe_grouped_matmul_fwd_swap_ab.py",
        graph_type=GraphType.MOE_SWAP_AB,
        supports_multi_gemm=True,
        template_cls=Sm120KernelTemplate,
    ),
    _mm(
        "sm120_moe_grouped_block_scale_matmul_fwd_swap_ab.py",
        graph_type=GraphType.MOE_BLOCK_SCALE_SWAP_AB,
        supports_multi_gemm=True,
        template_cls=Sm120KernelTemplate,
    ),
)
# [vendor.py P6] the vendored tree carries the sm100 arch family only: the sm120 registrations above are dropped (their
# template files are not vendored, and their block-quant gate imports the sm120 compiler).
TEMPLATES = tuple(t for t in TEMPLATES if t.family in arch_family.FAMILIES)


# Pipeline families the AUTO path (``tile_config.select_config``) may build
# with, best first. sm103 is deliberately absent: its 384-byte K-tile is outside
# select_config's geometry ladder, so it stays an explicit-config pipeline.
# sm120 is last: it serves the GPUs (SM 12.x) the tcgen05 families cannot, and
# never outranks them where both run.
_AUTO_PIPELINE_ORDER: tuple[str, ...] = ("sm100", "sm120")


def preferred_pipeline(chain: FusionChain) -> str:
    """Pipeline family the auto path should build ``chain`` with: the first
    :data:`_AUTO_PIPELINE_ORDER` entry that has a template for this graph type,
    whose SM range covers the active GPU, and whose arch tree is the one a
    process on that GPU runs (:func:`arch_family.serving_family` -- each tree's
    compiler renders only its own family). A graph type the newer family does
    not implement (plain matmul, MoE) falls through to sm100 by itself."""
    from . import compiler as C

    gt = classify_graph_type(chain)
    family = arch_family.serving_family(C._current_arch())
    for pipeline in _AUTO_PIPELINE_ORDER:
        if any(t.pipeline == pipeline and t.graph_type is gt and t.family == family and t.arch_active_reject() is None for t in TEMPLATES):
            return pipeline
    return _AUTO_PIPELINE_ORDER[-1]


def preferred_mma_tile_k_bytes(chain: FusionChain) -> int:
    """MMA-inst K width the auto path should build ``chain`` with. The 64-byte
    block-scale MMA halves the instruction count at a wide tile, so take it
    whenever the ACTIVE GPU issues it — it is silicon, not a pipeline, so this
    asks the arch and not the config family. Everything else stays at 32."""
    mm = chain.matmul
    dense_fp8 = mm.a_dtype.startswith("fp8_") and mm.b_dtype.startswith("fp8_")
    if not dense_fp8 and classify_graph_type(chain) not in (GraphType.BLOCK_SCALE_MATMUL, GraphType.MOE_BLOCK_SCALE, GraphType.MOE_BLOCK_SCALE_SWAP_AB):
        return 32
    from . import compiler as C

    arch = C._current_arch()
    return 64 if arch is not None and any(lo <= arch < hi for lo, hi in MMA_INST_K64_ARCH_RANGES) else 32


def dense_k64_envelope(config: TileConfig) -> bool:
    """Whether ``config`` is a geometry the DENSE 64-byte-K MMA is rendered for:
    a 128-tall CTA in a 2-CTA pair -- the envelope :func:`tile_config.select_config`
    emits it at and the one the sweep validated. The compiler rejects every other
    geometry (``compiler._check_mma_k_dim``: dense K64 needs ``mma_tile_m == 128``
    per CTA), so a shorter or 1-CTA pick must keep the 32-byte form. Block-scale
    is a different instruction with its own rules and does not go through here."""
    return config.cta_tile_m >= 128 and config.mma_tile_m == 128 and getattr(config, "cta_group", 1) == 2


def preferred_strategy(chain: FusionChain, config: TileConfig) -> TileConfig:
    """Re-target an auto pick at the family :func:`preferred_pipeline` chooses and
    the MMA-inst K width :func:`preferred_mma_tile_k_bytes` wants. ``cta_group``
    rides the geometry, and family-fixed axes (sm120's warp-MMA pair, cluster and
    default warp grid) are snapped by :func:`as_pipeline`, so nothing is clamped
    here.

    The dense-FP8 64-byte-K width is taken only where the kernel exists for it
    (:func:`dense_k64_envelope`). Re-stamping it onto a 64-tall or 1-CTA pick
    produced a config the compiler rejects, and because the auto path probes its
    ONE pick, that rejection declined the whole graph: every gated-attention-block
    fp8 projection GEMM on sm_107 lost its ``frost_gemm`` plan at small M
    (``CONFIG_sm100_64x32x128_64x32x64_cluster2x4_2ctamma`` ->
    ``plain FP8 mma_tile_k_bytes=64 requires mma_tile_m=128 per CTA``)."""
    pipeline = preferred_pipeline(chain)
    if config.pipeline == "sm120" and pipeline != "sm120":
        # sm120 is warp-scoped MMA: its geometries (16x16 warp pairs) have no
        # cluster-family equivalent, and the sm120 compiler only plans for
        # sm120 silicon in production. A cross-family preference here can only
        # mean this module is being driven off-device (unit tests on another
        # runner), where the conversion is unrepresentable -- keep the family.
        pipeline = config.pipeline
    k_bytes = preferred_mma_tile_k_bytes(chain)
    if k_bytes == 64 and not chain.has_block_scale and not dense_k64_envelope(config):
        k_bytes = 32
    config = as_mma_tile_k(config, k_bytes)
    if pipeline != config.pipeline:
        config = as_pipeline(config, pipeline)
    return config


def select_template(
    chain: FusionChain,
    config: TileConfig,
) -> KernelTemplate:
    """The single template that renders (chain, config). Everything that selects
    it -- pipeline, MMA mode, graph type, mainloop -- is on the config or the
    chain. Capability gates are NOT applied: this renders even unsupported
    configs for deliberate single-point probing."""
    gt = classify_graph_type(chain)
    matches = [
        t
        for t in TEMPLATES
        if isinstance(config, config_class_for_pipeline(t.pipeline)) and t.graph_type is gt and t.supports_mainloop_fusion == chain.has_mainloop_fusion
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"no kernel template for graph_type={gt.value}, "
            f"mainloop={chain.has_mainloop_fusion}, config={config.name!r}, "
            f"pipeline={config.pipeline}. "
            "E.g. mainloop fusion has no block-scale template variant yet."
        )
    raise ValueError(f"ambiguous template match (registry bug): {[t.file for t in matches]}")


def candidates(chain: FusionChain, *, sweep_swap_ab: bool = False) -> list[tuple[KernelTemplate, TileConfig]]:
    """Traversal-mode candidate set for ``chain`` via the funnel. Each accepted
    (template, geometry) is a JIT-able point; one geometry expands across the
    templates that accept it ({1,2}ctamma, etc.)."""
    if sweep_swap_ab:
        out = candidates(chain)
        try:
            swapped = swap_ab_graph(chain)
        except (ValueError, NotImplementedError):
            return out
        return out + [(template, replace(config, swap_ab=True)) for template, config in candidates(swapped)]
    gt = classify_graph_type(chain)
    tmpls = [t for t in TEMPLATES if t.graph_type is gt]  # stage 1
    if not tmpls:
        return []
    out: list[tuple[KernelTemplate, TileConfig]] = []
    for tmpl in tmpls:
        for cfg in tmpl.candidate_configs(chain):  # stages 0 + 2 + 3 + 4
            out.append((tmpl, cfg))
    return out
