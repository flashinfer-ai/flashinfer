# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fusion IR for GEMM operands, pointwise operations, and outputs."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

Dtype = Literal[
    "bf16",
    "fp16",
    "fp32",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_e8m0",
    "fp8_e5m3",
    "fp4_e2m1",
    "int8",
    "uint8",
    "int32",
    "int64",
]
SUPPORTED_DTYPES: tuple[Dtype, ...] = (
    "bf16",
    "fp16",
    "fp32",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_e8m0",
    "fp8_e5m3",
    "fp4_e2m1",
    "int8",
    "uint8",
    "int32",
)
COMPUTE_DTYPES: tuple[Dtype, ...] = ("fp32", "int32")
AMajor = Literal["k", "m"]
BMajor = Literal["k", "n"]
OutMajor = Literal["n", "m"]

BroadcastMode = Literal["scalar", "per_row", "per_col", "per_elem", "per_k"]


class Axis(Enum):
    """Logical GEMM extents, independent of codegen's symbol names."""

    M = auto()
    N = auto()
    K = auto()
    BATCH = auto()
    GROUP = auto()


ReductionMode = Literal["add", "amax", "max", "min", "avg", "mul", "mul_no_zeros", "norm1", "norm2"]
REDUCTION_DTYPES: tuple[Dtype, ...] = ("fp32", "int32")

# Unary pointwise ops (take only the running accumulator).
UNARY_OPS: tuple[str, ...] = (
    "identity",
    "relu",
    "gelu",  # exact gelu via erf
    "gelu_tanh",  # gelu approx-tanh
    "swish",
    "sigmoid",
    "tanh",
    "exp",
    "abs",
    "neg",
    "cos",
    "sin",
    "ceil",
    "floor",
    "erf",
    "log",
    "reciprocal",
    "rsqrt",
    "sqrt",
    "elu",
    "softplus",
    "tan",
    "logical_not",
    "leaky_relu",
    "gen_index",
)

BINARY_OPS: tuple[str, ...] = (
    "add",
    "mul",
    "sub",
    "div",
    "max",
    "min",
    "pow",
    "add_square",
    "mod",
    "logical_and",
    "logical_or",
    "cmp_eq",
    "cmp_neq",
    "cmp_gt",
    "cmp_ge",
    "cmp_lt",
    "cmp_le",
    "relu_backward",
    "leaky_relu_backward",
    "swish_backward",
    "sigmoid_backward",
    "tanh_backward",
    "elu_backward",
    "gelu_backward",
    "gelu_tanh_backward",
    "softplus_backward",
)

TERNARY_OPS: tuple[str, ...] = ("binary_select",)

ALL_OPS: tuple[str, ...] = UNARY_OPS + BINARY_OPS + TERNARY_OPS + ("aux_load",)

ZERO_PRESERVING_OPS: frozenset[str] = frozenset(
    {
        "identity",
        "relu",
        "gelu",
        "gelu_tanh",
        "swish",
        "tanh",
        "abs",
        "neg",
        "sin",
        "ceil",
        "floor",
        "erf",
        "sqrt",
        "mul",
        "elu",
        "tan",
    }
)


def out_major_of(stride: "tuple[int, ...] | None") -> "OutMajor":
    """A dense output's layout, read off its STRIDE alone."""
    if not stride:
        return "n"
    if stride[-1] == 1:
        return "n"
    if stride[-2] == 1:
        return "m"
    raise ValueError(f"output must be N-major or M-major in the inner (M,N) plane; got stride={stride!r}")


@dataclass(frozen=True)
class ChainOutput:
    """One materialized GMEM output."""

    source: str
    dtype: Dtype
    dim: "tuple[int, int, int] | None" = None
    stride: "tuple[int, int, int] | None" = None
    is_reduction: bool = False
    is_quant_scale: bool = False
    quant_block_size: int | None = None

    @property
    def major(self) -> "OutMajor":
        """Derived, not stored -- see `out_major_of`."""
        return out_major_of(self.stride)


@dataclass(frozen=True)
class OutputSpec:
    """One dense (per-element) GMEM output: a producer's value cast to ``dtype``."""

    source_ref: int
    dtype: Dtype
    quant_idx: int | None = None
    dim: "tuple[int, int, int] | None" = None
    stride: "tuple[int, int, int] | None" = None

    def __post_init__(self) -> None:
        for _f, _v in (("dim", self.dim), ("stride", self.stride)):
            if _v is not None and len(_v) != 3:
                raise ValueError(f"output {_f} must be rank-3 (got {_v!r})")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported output dtype {self.dtype!r}")
        self.major  # noqa: B018 -- reject an unsupported layout at construction

    @property
    def major(self) -> OutMajor:
        """Derived, not stored -- see `out_major_of`."""
        return out_major_of(self.stride)


@dataclass(frozen=True)
class ReductionSpec:
    """A materialized reduction side-output."""

    mode: ReductionMode
    source_ref: int
    dim: tuple[int, int, int]
    dtype: Dtype = "fp32"
    compute_dtype: Dtype = "fp32"
    grouped_by_moe: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("add", "amax", "max", "min", "avg", "mul", "mul_no_zeros", "norm1", "norm2"):
            raise ValueError(f"unsupported reduction mode {self.mode!r}")
        if self.mode in ("avg", "mul", "mul_no_zeros", "norm2") and self.compute_dtype != "fp32":
            raise ValueError(f"reduction mode {self.mode!r} requires fp32 compute; got {self.compute_dtype!r}")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported reduction output dtype {self.dtype!r}; " f"expected one of {SUPPORTED_DTYPES}")
        if self.dtype not in REDUCTION_DTYPES:
            raise ValueError(f"reduction output dtype {self.dtype!r} is not supported; " f"expected one of {REDUCTION_DTYPES}")
        if self.compute_dtype not in REDUCTION_DTYPES:
            raise ValueError(f"reduction compute_dtype {self.compute_dtype!r} is not supported; " f"expected one of {REDUCTION_DTYPES}")
        if self.dtype != self.compute_dtype:
            raise ValueError(f"reduction output dtype {self.dtype!r} must match compute_dtype " f"{self.compute_dtype!r} for direct atomic reduction")


QUANT_DATA_DTYPES = ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1")

M_MAJOR_OUT_DTYPES = ("bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2", "fp8_e8m0", "fp8_e5m3", "int8", "uint8", "int32")


def segmented_row_scale_capacity_rows(total_rows: int, num_groups: int) -> int:
    """Static rows needed by independently 128-row-padded group segments."""
    if total_rows < 0:
        raise ValueError(f"segmented row scale total_rows must be non-negative; got {total_rows}")
    if num_groups < 1:
        raise ValueError(f"segmented row scale num_groups must be positive; got {num_groups}")
    active = min(total_rows, num_groups)
    max_atoms = active + (total_rows - active) // 128
    return 128 * max_atoms


@dataclass(frozen=True)
class BlockQuantizeSpec:
    """Describe one block quantizer and its scale output."""

    source_ref: int
    block_size: int
    axis: int = -1
    transpose: bool = False
    scale_dtype: Dtype = "fp8_e8m0"
    scale_dim: tuple[int, int, int] | None = None
    scale_reorder: str | None = None
    compute_dtype: Dtype = "fp32"
    grouped_by_moe: bool = False

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block quantize block_size must be positive; got {self.block_size}")
        if self.axis not in (-1, 1, 2):
            raise NotImplementedError(f"block quantize supports the N axis (-1/2) or the M axis (1) in " f"cudnn.gemm.frost; got axis={self.axis}")
        if self.transpose and self.axis != 1:
            raise ValueError("block quantize transpose=True requires the M axis (axis=1)")
        if self.scale_dtype not in ("fp8_e8m0", "fp8_e4m3", "fp8_e5m3"):
            raise ValueError(f"block quantize scale dtype {self.scale_dtype!r} is not supported; " "expected fp8_e8m0, fp8_e4m3 or fp8_e5m3")
        if self.scale_reorder not in (None, "F8_128x4"):
            raise ValueError(f"block quantize scale reordering {self.scale_reorder!r} is not supported; " "expected None or F8_128x4")
        if self.compute_dtype != "fp32":
            raise ValueError(f"block quantize compute_dtype {self.compute_dtype!r} is not supported; " "expected fp32")
        if self.grouped_by_moe and self.scale_reorder != "F8_128x4":
            raise ValueError("grouped block quantize requires F8_128x4 scale reordering")


@dataclass(frozen=True)
class TensorRef:
    """An auxiliary fusion input (epilogue broadcasts or mainloop per-K scale)."""

    name: str
    dim: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Dtype
    bcast_mode: BroadcastMode
    grouped_by_moe: bool = False
    reduction_axis: int | None = None

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported aux dtype {self.dtype!r}; expected one of {SUPPORTED_DTYPES}")
        if len(self.dim) != len(self.stride):
            raise ValueError(f"dim {self.dim} and stride {self.stride} length mismatch")
        if self.bcast_mode == "per_k" and (len(self.dim) != 3 or self.dim[0] != 1 or (self.dim[1] != 1 and self.dim[2] != 1)):
            raise ValueError(f"per-K aux {self.name!r} must have shape [1,1,K] or [1,K,1]; got {self.dim}")
        if self.bcast_mode == "per_k":
            if self.reduction_axis not in (1, 2) or self.dim[3 - self.reduction_axis] != 1:
                raise ValueError(f"per-K aux {self.name!r} needs an explicit reduction axis consistent with dim {self.dim}")
        elif self.reduction_axis is not None:
            raise ValueError(f"aux {self.name!r}: reduction_axis is only valid for per-K broadcasts")
        if self.grouped_by_moe:
            if len(self.dim) != 3:
                raise ValueError(f"per-group aux {self.name!r} must be rank-3; got dim {self.dim}")
            if self.dim[1] != 1 and self.dim[2] != 1:
                raise ValueError(f"per-group aux {self.name!r} must broadcast an M/N axis; got dim {self.dim}")
            if self.bcast_mode not in ("scalar", "per_col", "per_row"):
                raise ValueError(f"per-group aux {self.name!r} supports scalar, per_col or per_row broadcast; " f"got {self.bcast_mode!r}")

    @property
    def k_axis(self) -> int:
        """Logical reduction axis of a rank-3 per-K broadcast (A: 2, B: 1)."""
        if self.bcast_mode != "per_k":
            raise ValueError(f"aux {self.name!r} is not a per-K broadcast")
        assert self.reduction_axis is not None  # Validated at construction.
        return self.reduction_axis

    @property
    def symbolic_shape(self) -> tuple[int | Axis, ...]:
        """Logical dynamic extents, with the same rank as declared ``dim``.

        Kernel ABI rank promotion belongs to lowering, not this record.
        Per-K orientation comes from the operand, even when K equals one.
        """
        if self.bcast_mode == "per_k":
            shape = [1, 1, 1]
            shape[self.k_axis] = Axis.K
            return tuple(shape)
        axes = {
            "scalar": (1, 1),
            "per_row": (Axis.M, 1),
            "per_col": (1, Axis.N),
            "per_elem": (Axis.M, Axis.N),
        }[self.bcast_mode]
        if len(self.dim) == 3:
            batch = Axis.GROUP if self.grouped_by_moe else (1 if self.dim[0] == 1 else Axis.BATCH)
            return (batch, *axes)
        return axes[2 - len(self.dim) :]


def gemm_source(g: int) -> int:
    """Encode 'this input is GEMM ``g``'s output' as a parent reference."""
    return -1 - g


def is_gemm_source(ref: int) -> bool:
    """True iff a parent reference names a GEMM output (vs a pointwise op)."""
    return ref < 0


def gemm_index(ref: int) -> int:
    """The GEMM index named by a (gemm) parent reference (inverse of gemm_source)."""
    return -1 - ref


@dataclass(frozen=True)
class FusionOp:
    """A single pointwise op in the epilogue chain."""

    op: str
    aux: str | None = None  # TensorRef.name, or None for unary / fan-in
    aux_on_rhs: bool = True  # for binary ops; True = op(parent_idx, other)
    parent_idx: int | None = None  # producing-op ref (None = auto)
    parent_idx_b: int | None = None  # fan-in: second producing-op ref (None = absent)
    parent_idx_c: int | None = None  # ternary (binary_select): third producing-op ref
    attrs: tuple[tuple[str, float], ...] = ()  # scalar attrs (negative_slope/clips/swish_beta/axis)
    compute_dtype: Dtype = "fp32"
    out_dtype: Dtype = "fp32"

    def __post_init__(self) -> None:
        if self.op not in ALL_OPS:
            raise ValueError(f"unknown op {self.op!r}; expected one of {ALL_OPS}")
        if self.compute_dtype not in COMPUTE_DTYPES:
            raise ValueError(f"op {self.op!r}: compute_data_type {self.compute_dtype!r} is not " f"supported — expected one of {COMPUTE_DTYPES}.")
        if self.out_dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"op {self.op!r}: unsupported out_dtype {self.out_dtype!r}; " f"expected one of {SUPPORTED_DTYPES}")
        if self.op == "aux_load":
            if self.aux is None:
                raise ValueError("aux_load requires an aux tensor name")
            if self.parent_idx is not None or self.parent_idx_b is not None:
                raise ValueError("aux_load takes no producing-operation inputs")
        if self.op in TERNARY_OPS:
            if self.parent_idx is None or self.parent_idx_b is None or self.parent_idx_c is None:
                raise ValueError(f"ternary op {self.op!r} requires parent_idx, parent_idx_b, and parent_idx_c")
            if self.aux is not None:
                raise ValueError(f"ternary op {self.op!r} cannot have an aux input")
        elif self.parent_idx_c is not None:
            raise ValueError(f"op {self.op!r} cannot have parent_idx_c (ternary ops only)")
        if self.op in UNARY_OPS and self.aux is not None:
            raise ValueError(f"unary op {self.op!r} cannot have aux input")
        if self.op in UNARY_OPS and self.parent_idx_b is not None:
            raise ValueError(f"unary op {self.op!r} cannot have parent_idx_b")
        if self.op in BINARY_OPS and self.aux is None and self.parent_idx_b is None:
            raise ValueError(f"binary op {self.op!r} requires either an aux tensor name or " f"a parent_idx_b pointing at another producing operation")
        if self.aux is not None and self.parent_idx_b is not None:
            raise ValueError(f"binary op {self.op!r}: aux and parent_idx_b are mutually exclusive")

    def resolved_parent_idx(self, position: int) -> int:
        """Resolve an auto (``None``) ``parent_idx`` given the op's position:
        previous op for position > 0, else GEMM 0 (-1). Non-None returned as-is."""
        if self.parent_idx is not None:
            return self.parent_idx
        return position - 1 if position > 0 else gemm_source(0)


@dataclass(frozen=True)
class MatmulSpec:
    """Shape + dtype of the anchor matmul."""

    M: int
    N: int
    K: int
    batch: int = 1
    a_batch: int = 1
    b_batch: int = 1
    a_major: AMajor = "k"
    b_major: BMajor = "k"
    a_dtype: Dtype = "bf16"
    b_dtype: Dtype = "bf16"
    accum_dtype: Dtype = "fp32"
    out_dtype: Dtype = "bf16"

    def __post_init__(self) -> None:
        if self.a_major not in ("k", "m"):
            raise ValueError(f"a_major must be 'k' or 'm' (got {self.a_major!r})")
        if self.b_major not in ("k", "n"):
            raise ValueError(f"b_major must be 'k' or 'n' (got {self.b_major!r})")
        if self.batch < 1 or self.a_batch < 1 or self.b_batch < 1:
            raise ValueError("batch dimensions must be positive")
        if self.a_batch not in (1, self.batch) or self.b_batch not in (1, self.batch):
            raise ValueError(
                f"matmul batches must either match output batch={self.batch} "
                f"or be broadcast batch=1; got a_batch={self.a_batch}, "
                f"b_batch={self.b_batch}"
            )
        if max(self.a_batch, self.b_batch) != self.batch:
            raise ValueError(f"output batch must be max(a_batch, b_batch); got " f"batch={self.batch}, a_batch={self.a_batch}, " f"b_batch={self.b_batch}")
        if self.out_dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported matmul out_dtype {self.out_dtype!r}; " f"expected one of {SUPPORTED_DTYPES}")


@dataclass(frozen=True)
class BlockScaleSpec:
    """Describe the data and scale inputs of a block-scaled GEMM."""

    a_dtype: Dtype  # packed data dtype of A (mirror of matmul.a_dtype)
    b_dtype: Dtype  # packed data dtype of B
    block_size_a: "tuple[int, ...] | None" = None
    block_size_b: "tuple[int, ...] | None" = None
    sf_dtype_a: "Dtype | None" = None  # A's scale-factor dtype (None if A not scaled)
    sf_dtype_b: "Dtype | None" = None
    # SF reorder layout per side (cuDNN name, e.g. "F8_128x4"; None = NONE).
    sfa_reorder: "str | None" = None
    sfb_reorder: "str | None" = None
    sfa_dim: tuple[int, ...] | None = None
    sfa_stride: tuple[int, ...] | None = None
    sfb_dim: tuple[int, ...] | None = None
    sfb_stride: tuple[int, ...] | None = None
    dequant_compute_a: "Dtype | None" = None
    dequant_compute_b: "Dtype | None" = None
    dequant_out_a: "Dtype | None" = None  # = MMA input type for A
    dequant_out_b: "Dtype | None" = None  # = MMA input type for B
    fake_dequant_a: bool = False
    fake_dequant_b: bool = False
    padded_e2m1: bool = False

    def __post_init__(self) -> None:
        if self.sf_dtype_a is None and self.sf_dtype_b is None:
            raise ValueError("block_scale spec has neither operand dequantized " "(at least one of A / B must be block-scaled)")

    @property
    def both_sided(self) -> bool:
        """True iff both operands have logical scale metadata."""
        return self.sf_dtype_a is not None and self.sf_dtype_b is not None

    @property
    def block_size(self) -> int:
        """K-block size as a single int. A is [non_K, K] (K = last dim); B is
        [K, non_K] (K = first dim)."""
        if self.block_size_a is not None:
            return int(self.block_size_a[-1])
        if self.block_size_b is not None:
            return int(self.block_size_b[0])
        return 1

    @property
    def sf_dtype(self) -> "Dtype | None":
        return self.sf_dtype_a if self.sf_dtype_a is not None else self.sf_dtype_b

    @property
    def is_fp4(self) -> bool:
        return self.both_fp4

    @property
    def both_fp4(self) -> bool:
        return self.a_dtype == "fp4_e2m1" and self.b_dtype == "fp4_e2m1"

    @property
    def mixed_width(self) -> bool:
        return (self.a_dtype == "fp4_e2m1") != (self.b_dtype == "fp4_e2m1")

    @property
    def mma_block_scale_kind(self) -> str:
        """GEMM ``nvvm.MMABlockScaleKind`` member name."""
        return "MXF4NVF4" if self.both_fp4 else "MXF8F6F4"

    @property
    def scale_vec_size(self) -> str:
        """GEMM ``nvvm.Tcgen05MMABlockScale`` member name. block16→X4 scale-vec,
        block32→X2 (the MMA reads 1 SF per ``block_size`` K-elements)."""
        return "BLOCK16" if self.block_size == 16 else "BLOCK32"

    @property
    def sf_scale_format(self) -> int:
        """``Tcgen05MxInstrDesc`` ``scale_format`` field (idesc bits 24-23); E5M3 (2) needs SM 10.7+ silicon."""
        fmt = {"fp8_e4m3": 0, "fp8_e8m0": 1, "fp8_e5m3": 2}.get(self.sf_dtype)
        if fmt is None:
            raise NotImplementedError(f"no MMA scale_format encoding for block-scale SF dtype {self.sf_dtype!r}")
        return fmt


@dataclass(frozen=True)
class MoeSpec:
    """A MoE grouped matmul forward pass: per routed group g, ``out[fto[g]:fto[g+1]] = token[range] @ weight[g % num_experts].T``."""

    num_experts: int  # E — the weight batch; routed group g uses expert g % E
    mode: str = "none"
    offset_dtype: Dtype = "int32"
    num_groups: int = 0
    offset_multiple: int = 1
    top_k: int = 1

    def __post_init__(self) -> None:
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be positive; got {self.num_experts}")
        if self.num_groups < 1:
            object.__setattr__(self, "num_groups", self.num_experts)
        if self.mode not in ("none", "gather", "scatter", "combine"):
            raise ValueError(f"MoE grouped matmul mode {self.mode!r} is unsupported")
        if self.top_k < 1:
            raise ValueError("MoE top_k must be positive")
        if self.offset_dtype not in ("int32", "int64"):
            raise ValueError(f"first_token_offset dtype must be int32 or int64; " f"got {self.offset_dtype!r}")
        if self.offset_multiple < 1:
            raise ValueError(f"first_token_offset alignment_value must be >= 1; " f"got {self.offset_multiple}")


@dataclass(frozen=True)
class MoeSwapAbSpec(MoeSpec):
    """Internal weight-by-token grouped matmul, with groups partitioning N."""


def _walk_dtype_fields(obj: object, found: "set[Dtype]", *, in_dtype_field: bool = False) -> None:
    """Recursive half of :meth:`FusionChain.dtypes_used`. A field counts when its
    declared type mentions ``Dtype``; containers inherit that from their field."""
    if isinstance(obj, str):
        if in_dtype_field:
            found.add(obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _walk_dtype_fields(getattr(obj, f.name), found, in_dtype_field="Dtype" in str(f.type))
    elif isinstance(obj, dict):
        for item in obj.values():
            _walk_dtype_fields(item, found, in_dtype_field=in_dtype_field)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            _walk_dtype_fields(item, found, in_dtype_field=in_dtype_field)


@dataclass
class FusionChain:
    """Full description of a matmul + pointwise-epilogue fusion (linear chain
    or fan-out DAG)."""

    matmul: MatmulSpec
    aux_tensors: list[TensorRef] = field(default_factory=list)
    ops: list[FusionOp] = field(default_factory=list)
    output_specs: list[OutputSpec] = field(default_factory=list)
    num_a_operands: int = 1
    num_b_operands: int = 1
    gemm_operands: list[tuple[int, int]] = field(default_factory=lambda: [(0, 0)])
    mainloop_a_ops: list[FusionOp] = field(default_factory=list)
    # Same for B (``C = A @ op(B)``); each CTA transforms its own B tile in place.
    mainloop_b_ops: list[FusionOp] = field(default_factory=list)
    mainloop_a_load_dtype: Dtype | None = None
    mainloop_b_load_dtype: Dtype | None = None
    block_scale: "BlockScaleSpec | None" = None
    moe: "MoeSpec | None" = None
    reductions: list[ReductionSpec] = field(default_factory=list)
    quants: list["BlockQuantizeSpec"] = field(default_factory=list)
    gemm_out_dtypes: tuple[Dtype, ...] = ()

    def __post_init__(self) -> None:
        for i, spec in enumerate(self.output_specs):
            if is_gemm_source(spec.source_ref):
                g = gemm_index(spec.source_ref)
                if g >= self.num_gemms:
                    raise ValueError(f"output_specs[{i}] source GEMM {g} out of range for " f"{self.num_gemms} GEMM(s)")
            elif not (0 <= spec.source_ref < len(self.ops)):
                raise ValueError(f"output_specs[{i}] source op index {spec.source_ref} out of " f"range for {len(self.ops)} op(s)")
        # Multi-GEMM structural invariants.
        if self.num_a_operands < 1 or self.num_b_operands < 1:
            raise ValueError("num_a_operands / num_b_operands must be positive")
        if not self.gemm_operands:
            raise ValueError("gemm_operands must list at least one GEMM")
        if self.gemm_out_dtypes and len(self.gemm_out_dtypes) != self.num_gemms:
            raise ValueError("gemm_out_dtypes must describe every GEMM")
        if any(dtype not in SUPPORTED_DTYPES for dtype in self.gemm_out_dtypes):
            raise ValueError(f"unsupported GEMM output dtypes: {self.gemm_out_dtypes}")
        for g, (ai, bi) in enumerate(self.gemm_operands):
            if not (0 <= ai < self.num_a_operands) or not (0 <= bi < self.num_b_operands):
                raise ValueError(f"gemm_operands[{g}]=({ai},{bi}) out of range for " f"{self.num_a_operands} A / {self.num_b_operands} B operands")
        token_axis = 2 if isinstance(self.moe, MoeSwapAbSpec) else 1
        for aux in self.aux_tensors:
            if aux.grouped_by_moe and (self.moe is None or aux.dim[token_axis] != 1):
                raise ValueError(f"per-group aux {aux.name!r} must broadcast the MoE token axis {token_axis}; got dim {aux.dim}")
        names = {t.name for t in self.aux_tensors}
        if len(names) != len(self.aux_tensors):
            raise ValueError("aux_tensors contain duplicate names")
        for op in self.ops:
            if op.aux is not None and op.aux not in names:
                raise ValueError(f"op {op.op!r} references unknown aux {op.aux!r}")
            if op.aux is not None and self.aux_by_name(op.aux).bcast_mode == "per_k":
                raise ValueError("per-K scale is a mainloop-only aux, not an epilogue broadcast")
        for red in self.reductions:
            if is_gemm_source(red.source_ref):
                g = gemm_index(red.source_ref)
                if g >= self.num_gemms:
                    raise ValueError(f"reduction source GEMM {g} out of range for " f"{self.num_gemms} GEMM(s)")
            elif not (0 <= red.source_ref < len(self.ops)):
                raise ValueError(f"reduction source op index {red.source_ref} out of range " f"for {len(self.ops)} op(s)")
        for qi, q in enumerate(self.quants):
            if is_gemm_source(q.source_ref):
                g = gemm_index(q.source_ref)
                if g >= self.num_gemms:
                    raise ValueError(f"quants[{qi}] source GEMM {g} out of range for " f"{self.num_gemms} GEMM(s)")
            elif not (0 <= q.source_ref < len(self.ops)):
                raise ValueError(f"quants[{qi}] source op index {q.source_ref} out of range " f"for {len(self.ops)} op(s)")
        for i, spec in enumerate(self.output_specs):
            if spec.major == "m" and spec.dtype not in M_MAJOR_OUT_DTYPES:
                raise ValueError(f"output_specs[{i}] is M-major, so its dtype must be one of " f"{M_MAJOR_OUT_DTYPES}; got {spec.dtype!r}.")
        quant_refs = []
        for i, spec in enumerate(self.output_specs):
            if spec.quant_idx is None:
                continue
            if not (0 <= spec.quant_idx < len(self.quants)):
                raise ValueError(f"output_specs[{i}] quant_idx {spec.quant_idx} out of range " f"for {len(self.quants)} quant(s)")
            if spec.source_ref != self.quants[spec.quant_idx].source_ref:
                raise ValueError(
                    f"output_specs[{i}] source ref {spec.source_ref} does not match "
                    f"quants[{spec.quant_idx}] source ref {self.quants[spec.quant_idx].source_ref}"
                )
            if spec.dtype not in QUANT_DATA_DTYPES:
                raise ValueError(
                    f"output_specs[{i}] carries quants[{spec.quant_idx}], so its value is that "
                    f"node's quantized DATA and its dtype must be one of {QUANT_DATA_DTYPES}; got "
                    f"{spec.dtype!r}. The SCALE half is a separate output and its dtype is "
                    f"`BlockQuantizeSpec.scale_dtype`."
                )
            quant_refs.append(spec.quant_idx)
        if sorted(quant_refs) != list(range(len(self.quants))):
            raise ValueError(f"each quant needs exactly one dense data output; quants={len(self.quants)}, " f"referenced={sorted(quant_refs)}")
        for label, oplist in (
            ("mainloop_a_ops", self.mainloop_a_ops),
            ("mainloop_b_ops", self.mainloop_b_ops),
        ):
            for op in oplist:
                if op.parent_idx_b is not None:
                    raise ValueError(f"{label} op {op.op!r} cannot have parent_idx_b (no fan-in)")
                if op.op in BINARY_OPS and op.aux is None:
                    raise ValueError(f"{label} binary op {op.op!r} needs an aux tensor")
                if op.aux is not None:
                    if op.aux not in names:
                        raise ValueError(f"{label} op {op.op!r} references unknown aux {op.aux!r}")
                    ref = self.aux_by_name(op.aux)
                    if ref.bcast_mode == "per_k":
                        expected = (1, 1, self.matmul.K) if label == "mainloop_a_ops" else (1, self.matmul.K, 1)
                        axis = 2 if label == "mainloop_a_ops" else 1
                        if ref.dim != expected or ref.k_axis != axis:
                            raise ValueError(
                                f"{label} per-K aux {ref.name!r}: expected {expected} with K axis {axis}, " f"got {ref.dim} with K axis {ref.k_axis}"
                            )
                    elif ref.bcast_mode != "scalar":
                        raise ValueError(
                            f"{label} op {op.op!r} aux {op.aux!r} must broadcast as a "
                            f"scalar (got {ref.bcast_mode!r}); per-row/col/elem mainloop "
                            "aux is out of POC scope"
                        )

    @property
    def num_gemms(self) -> int:
        """Number of parallel GEMMs feeding the shared epilogue (1 = legacy)."""
        return len(self.gemm_operands)

    @property
    def is_multi_gemm(self) -> bool:
        """True iff more than one parallel GEMM feeds the epilogue."""
        return self.num_gemms > 1

    @property
    def has_block_scale(self) -> bool:
        """True iff this is a block-scaled (FP4/FP8 + per-block SF) matmul."""
        return self.block_scale is not None

    @property
    def has_moe(self) -> bool:
        """True iff this is a MoE grouped matmul."""
        return self.moe is not None

    @property
    def has_mainloop_fusion_a(self) -> bool:
        """True iff a pointwise op is applied to A before the MMA."""
        return bool(self.mainloop_a_ops)

    @property
    def has_mainloop_fusion_b(self) -> bool:
        """True iff a pointwise op is applied to B before the MMA."""
        return bool(self.mainloop_b_ops)

    @property
    def has_mainloop_fusion(self) -> bool:
        """True iff any operand has a mainloop (pre-MMA) pointwise op."""
        return self.has_mainloop_fusion_a or self.has_mainloop_fusion_b

    @property
    def mainloop_a_cast(self) -> bool:
        """True iff A is loaded narrow and widened to the MMA dtype in the mainloop."""
        return self.mainloop_a_load_dtype is not None

    @property
    def mainloop_b_cast(self) -> bool:
        """True iff B is loaded narrow and widened to the MMA dtype in the mainloop."""
        return self.mainloop_b_load_dtype is not None

    @property
    def output_dtype(self) -> Dtype:
        """Dtype of the primary-ABI-slot output (``output_specs[0]``)."""
        return self.output_specs[0].dtype if self.output_specs else "bf16"

    @property
    def widen_integer_accumulators(self) -> bool:
        """Select the register format entering the epilogue."""
        return self.matmul.accum_dtype == "int32" and self.gemm_out_dtype(0) != "int32" and not self.is_multi_gemm

    @property
    def out_major(self) -> "OutMajor":
        """Layout of the primary-ABI-slot output."""
        return self.output_specs[0].major if self.output_specs else "n"

    def gemm_out_dtype(self, index: int) -> Dtype:
        """Return one GEMM's declared output dtype."""
        if self.gemm_out_dtypes:
            return self.gemm_out_dtypes[index]
        if self.is_multi_gemm and not self.ops:
            for spec in self.output_specs:
                if spec.source_ref == gemm_source(index) and spec.quant_idx is None:
                    return spec.dtype
        return self.matmul.out_dtype

    def aux_by_name(self, name: str) -> TensorRef:
        for t in self.aux_tensors:
            if t.name == name:
                return t
        raise KeyError(name)

    def _output_label(self, spec: "OutputSpec") -> str:
        if spec.quant_idx is not None:
            return f"quant_{spec.quant_idx}"
        if is_gemm_source(spec.source_ref):
            g = gemm_index(spec.source_ref)
            return "matmul" if g == 0 else f"gemm_{g}"
        return f"op_{spec.source_ref}"

    @property
    def outputs(self) -> list["ChainOutput"]:
        """Return materialized outputs in kernel argument order."""
        outs: list[ChainOutput] = [
            ChainOutput(source=self._output_label(spec), dtype=spec.dtype, dim=spec.dim, stride=spec.stride) for spec in self.output_specs
        ]
        for i, red in enumerate(self.reductions):
            outs.append(
                ChainOutput(
                    source=f"reduction_{i}",
                    dtype=red.dtype,
                    dim=red.dim,
                    is_reduction=True,
                )
            )
        for i, q in enumerate(self.quants):
            outs.append(
                ChainOutput(
                    source=f"quant_scale_{i}",
                    dtype=q.scale_dtype,
                    dim=q.scale_dim,
                    is_quant_scale=True,
                    quant_block_size=q.block_size,
                )
            )
        return outs

    @property
    def taps(self) -> list["ChainOutput"]:
        """Every output beyond the primary ABI slot (all of them when no dense
        output exists — reductions/scale still ride the tap plumbing)."""
        return self.outputs[1:] if self.output_specs else list(self.outputs)

    def dtypes_used(self) -> frozenset[Dtype]:
        """Every dtype this chain names, anywhere."""
        found: set[Dtype] = set()
        _walk_dtype_fields(self, found)
        return frozenset(found)

    def summary(self) -> str:
        """One-line human-readable summary for logs / error messages."""
        m = self.matmul
        batch = f"batch={m.batch}"
        if m.a_batch != m.batch or m.b_batch != m.batch:
            batch += f" A_batch={m.a_batch} B_batch={m.b_batch}"
        chain = " -> ".join(op.op if op.aux is None else f"{op.op}({op.aux})" for op in self.ops) or "identity"
        reductions = ""
        if self.reductions:
            red = ", ".join(f"{r.mode}{r.dim}" for r in self.reductions)
            reductions = f" | reductions: {red}"
        quant = ""
        if self.quants:
            qs = ", ".join(f"block{q.block_size}->{q.scale_dtype}" for q in self.quants)
            quant = f" | quants: {qs}"
        mainloop = ""
        if self.mainloop_a_ops:
            ml = " -> ".join(op.op for op in self.mainloop_a_ops)
            cast = f" [{self.mainloop_a_load_dtype}->{self.matmul.a_dtype}]" if self.mainloop_a_cast else ""
            mainloop += f"mainloop A: {ml}(A){cast} | "
        if self.mainloop_b_ops:
            ml = " -> ".join(op.op for op in self.mainloop_b_ops)
            cast = f" [{self.mainloop_b_load_dtype}->{self.matmul.b_dtype}]" if self.mainloop_b_cast else ""
            mainloop += f"mainloop B: {ml}(B){cast} | "
        return (
            f"FusionChain[matmul {m.M}x{m.N}x{m.K} {batch} "
            f"A{m.a_major}/B{m.b_major} "
            f"{m.a_dtype}*{m.b_dtype}->{m.accum_dtype} | {mainloop}"
            f"epilogue: {chain} -> {self.output_dtype}{reductions}{quant}]"
        )


def swap_ab(chain: FusionChain) -> FusionChain:
    """Return the logical-transpose view ``C.T = B.T @ A.T`` of ``chain``."""

    def _mn(values):
        if values is None or len(values) < 2:
            return values
        return (*values[:-2], values[-1], values[-2])

    def _bcast(mode):
        return {"per_row": "per_col", "per_col": "per_row"}.get(mode, mode)

    def _op(op):
        attrs = tuple((key, 3 - value if key == "axis" and value in (1, 2) else value) for key, value in op.attrs)
        return dataclasses.replace(op, attrs=attrs) if attrs != op.attrs else op

    mm = chain.matmul
    swapped_mm = dataclasses.replace(
        mm,
        M=mm.N,
        N=mm.M,
        a_batch=mm.b_batch,
        b_batch=mm.a_batch,
        a_major="k" if mm.b_major == "k" else "m",
        b_major="k" if mm.a_major == "k" else "n",
        a_dtype=mm.b_dtype,
        b_dtype=mm.a_dtype,
    )
    swapped_aux = []
    for t in chain.aux_tensors:
        dim, stride = t.dim, t.stride
        if len(dim) == 1:
            dim, stride = (1, dim[0]), (stride[0], stride[0])
        swapped_aux.append(
            dataclasses.replace(
                t,
                dim=_mn(dim),
                stride=_mn(stride),
                bcast_mode=_bcast(t.bcast_mode),
                reduction_axis=None if t.reduction_axis is None else 3 - t.reduction_axis,
            )
        )
    swapped_outputs = [
        dataclasses.replace(
            o,
            dim=_mn(o.dim or (mm.batch, mm.M, mm.N)),
            stride=_mn(o.stride or (mm.M * mm.N, mm.N, 1)),
        )
        for o in chain.output_specs
    ]
    swapped_reductions = [dataclasses.replace(r, dim=_mn(r.dim)) for r in chain.reductions]
    swapped_quants = [
        dataclasses.replace(
            q,
            axis=2 if q.axis == 1 else 1,
            transpose=False,
            scale_dim=q.scale_dim if q.scale_reorder == "F8_128x4" else _mn(q.scale_dim),
        )
        for q in chain.quants
    ]

    block_scale = chain.block_scale
    if block_scale is not None:
        block_scale = dataclasses.replace(
            block_scale,
            a_dtype=block_scale.b_dtype,
            b_dtype=block_scale.a_dtype,
            block_size_a=_mn(block_scale.block_size_b),
            block_size_b=_mn(block_scale.block_size_a),
            sf_dtype_a=block_scale.sf_dtype_b,
            sf_dtype_b=block_scale.sf_dtype_a,
            sfa_reorder=block_scale.sfb_reorder,
            sfb_reorder=block_scale.sfa_reorder,
            sfa_dim=_mn(block_scale.sfb_dim),
            sfa_stride=_mn(block_scale.sfb_stride),
            sfb_dim=_mn(block_scale.sfa_dim),
            sfb_stride=_mn(block_scale.sfa_stride),
            dequant_compute_a=block_scale.dequant_compute_b,
            dequant_compute_b=block_scale.dequant_compute_a,
            dequant_out_a=block_scale.dequant_out_b,
            dequant_out_b=block_scale.dequant_out_a,
            fake_dequant_a=block_scale.fake_dequant_b,
            fake_dequant_b=block_scale.fake_dequant_a,
        )

    return dataclasses.replace(
        chain,
        matmul=swapped_mm,
        aux_tensors=swapped_aux,
        ops=[_op(op) for op in chain.ops],
        output_specs=swapped_outputs,
        reductions=swapped_reductions,
        quants=swapped_quants,
        num_a_operands=chain.num_b_operands,
        num_b_operands=chain.num_a_operands,
        gemm_operands=[(b, a) for a, b in chain.gemm_operands],
        mainloop_a_ops=list(chain.mainloop_b_ops),
        mainloop_b_ops=list(chain.mainloop_a_ops),
        mainloop_a_load_dtype=chain.mainloop_b_load_dtype,
        mainloop_b_load_dtype=chain.mainloop_a_load_dtype,
        block_scale=block_scale,
        moe=(MoeSpec if isinstance(chain.moe, MoeSwapAbSpec) else MoeSwapAbSpec)(**dataclasses.asdict(chain.moe)) if chain.has_moe else None,
    )
