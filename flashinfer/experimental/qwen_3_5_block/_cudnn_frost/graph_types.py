# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python data types for cuDNN Frontend graph representation.

This module provides Python dataclasses for tensor attributes and node types,
enabling native Python access to graph structure.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union


class NodeType(Enum):
    """Operation node types. Maps to INode::Type in node_interface.h."""

    # Only the op types exercised by this version are listed. Add more as
    # needed, following the block-scale / MoE / reduction examples (enum entry
    # here + a builder in graph_native + inference in nodes + lowering).
    COMPOSITE = auto()
    CONV_FPROP = auto()
    CONV_DGRAD = auto()
    CONV_WGRAD = auto()
    GENSTATS = auto()
    RESHAPE = auto()
    SLICE = auto()
    CONCATENATE = auto()
    TRANSPOSE = auto()
    ROPE = auto()
    ROPE_BWD = auto()
    MOE_GROUPED_MATMUL_BWD = auto()
    MATMUL = auto()
    MATMUL_FP8 = auto()
    POINTWISE = auto()
    REDUCTION = auto()
    RMSNORM = auto()
    RMSNORM_BWD = auto()
    LAYERNORM = auto()
    LAYERNORM_BWD = auto()
    ADALAYERNORM = auto()
    ADALAYERNORM_BWD = auto()
    INSTANCENORM = auto()
    INSTANCENORM_BWD = auto()
    BATCHNORM = auto()
    BATCHNORM_INFERENCE = auto()
    BATCHNORM_BWD = auto()
    SDPA = auto()
    SDPA_BWD = auto()
    SDPA_FP8 = auto()
    SDPA_FP8_BWD = auto()
    SDPA_MXFP8 = auto()
    SDPA_MXFP8_BWD = auto()
    GDN = auto()
    GDN_BWD = auto()
    KDA = auto()
    KDA_BWD = auto()
    GDN2 = auto()
    GDN2_BWD = auto()
    MOE_GROUPED_MATMUL = auto()
    BLOCK_SCALE_QUANTIZE = auto()
    BLOCK_SCALE_DEQUANTIZE = auto()
    GDP = auto()
    GDP_BWD = auto()
    GDN_SUMMARY = auto()
    KDA_SUMMARY = auto()
    GDN2_SUMMARY = auto()
    GDP_SUMMARY = auto()
    GDN_SUMMARY_BWD = auto()
    KDA_SUMMARY_BWD = auto()
    GDN2_SUMMARY_BWD = auto()
    GDP_SUMMARY_BWD = auto()
    MAMBA2 = auto()
    MAMBA2_BWD = auto()
    MAMBA2_SUMMARY = auto()
    MAMBA2_SUMMARY_BWD = auto()


@dataclass(eq=False)  # identity-based hash/eq: uid/name are mutable
class Tensor:
    """Pure Python representation of tensor attributes.

    Mirrors cudnn_frontend::graph::Tensor_attributes from graph_properties.h.

    Attributes:
        name: Tensor identifier
        data_type: Data type (uses cudnn.data_type values)
        dim: Dimensions of the tensor
        stride: Memory strides
        is_virtual: True if tensor is an intermediate (not I/O)
        is_pass_by_value: True if tensor is a scalar passed at execution
        pass_by_value: Embedded constant value (for fused scalars)
        uid: Unique identifier for backend mapping
        uid_assigned: True if UID was explicitly assigned
        reordering_type: Memory layout transformation type
        ragged_offset: Tensor for variable-length tensor offsets
        alignment_value: caller's promise that every VALUE this tensor holds is a
            multiple of it (1 = no promise). Unlike every other attribute here it
            constrains the CONTENTS, not the layout, and it is not validated --
            violating it is undefined behaviour. Only the MoE
            first_token_offset tensor reads it.
    """

    name: str = ""
    data_type: Any = None
    dim: List[int] = field(default_factory=list)
    stride: List[int] = field(default_factory=list)
    is_virtual: bool = False
    is_pass_by_value: bool = False
    pass_by_value: Optional[Union[int, float]] = None
    uid: int = 0
    uid_assigned: bool = False
    # user-assigned vs IR-inferred layout: only USER-assigned dim/stride are
    # pushed to the lowered C++ output tensors — inferred values are
    # provisional (row-major) and the backend applies its own classic
    # per-op layout inference (e.g. channels-last conv). Internal inference
    # writes the attributes directly and leaves these False.
    dim_assigned: bool = False
    stride_assigned: bool = False
    reordering_type: Any = None
    ragged_offset: Optional["Tensor"] = None
    ragged_offset_multiplier: int = 1
    alignment_value: int = 1
    scalar_type: Any = None  # cudnn.scalar_type for tensor_scalar-created scalars
    # weakref to the owning graph (set at registration): identity mutations
    # (set_name / set_uid) delegate to the graph so its indexes stay coherent.
    owner: Any = field(default=None, repr=False)

    def _guard(self, what: str = "mutate a tensor attribute") -> None:
        g = self.owner() if self.owner is not None else None
        if g is not None:
            g._check_mutable(what)

    def set_output(self, value: bool) -> "Tensor":
        """Mark this tensor as an output (non-virtual) or intermediate (virtual)."""
        self._guard()
        self.is_virtual = not value
        return self

    def set_data_type(self, dtype: Any) -> "Tensor":
        """Set the data type."""
        self._guard()
        self.data_type = dtype
        return self

    def set_alignment_value(self, value: int) -> "Tensor":
        """Promise every value this tensor holds is a multiple of `value`."""
        self._guard()
        if value < 1:
            raise ValueError(f"alignment_value must be >= 1, got {value}")
        self.alignment_value = value
        return self

    def set_name(self, name: str) -> "Tensor":
        """Set the tensor name (graph-owned tensors re-index atomically)."""
        g = self.owner() if self.owner is not None else None
        if g is not None:
            g._rename_tensor(self, name)
        else:
            self.name = name
        return self

    def set_dim(self, dim: List[int]) -> "Tensor":
        """Set the tensor dimensions (user-assigned: pushed at lowering)."""
        self._guard()
        self.dim = list(dim)
        self.dim_assigned = True
        return self

    def set_stride(self, stride: List[int]) -> "Tensor":
        """Set the tensor strides (user-assigned: pushed at lowering)."""
        self._guard()
        self.stride = list(stride)
        self.stride_assigned = True
        return self

    def set_uid(self, uid: int) -> "Tensor":
        """Set the tensor UID (graph-owned tensors re-index atomically; a
        colliding auto-assigned uid is renumbered, user-user conflicts raise)."""
        g = self.owner() if self.owner is not None else None
        if g is not None:
            g._reuid_tensor(self, uid)
        else:
            self.uid = uid
            self.uid_assigned = True
        return self

    def set_ragged_offset(self, ragged_offset: "Tensor") -> "Tensor":
        """Set the ragged-offset tensor (variable-length layouts)."""
        self._guard()
        self.ragged_offset = ragged_offset
        return self

    def set_ragged_offset_multiplier(self, multiplier: int) -> "Tensor":
        """Set the ragged-offset unit size in tensor elements."""
        self._guard()
        self.ragged_offset_multiplier = multiplier
        return self

    def set_reordering_type(self, reordering_type: Any) -> "Tensor":
        """Set the memory reordering layout (e.g. F8_128x4)."""
        self._guard()
        self.reordering_type = reordering_type
        return self

    def set_is_pass_by_value(self, value: bool) -> "Tensor":
        """Mark the tensor as a host pass-by-value scalar."""
        self._guard()
        self.is_pass_by_value = value
        return self

    def set_is_virtual(self, value: bool) -> "Tensor":
        """Set virtualness directly (classic parity; inverse of set_output)."""
        self._guard()
        self.is_virtual = value
        return self

    def get_uid(self) -> int:
        return self.uid

    def get_name(self) -> str:
        return self.name

    def get_dim(self) -> List[int]:
        return list(self.dim)  # a copy, like the classic pybind getter

    def get_stride(self) -> List[int]:
        return list(self.stride)  # a copy, like the classic pybind getter

    def get_data_type(self) -> Any:
        # classic parity: the pybind getter returns the cudnn enum even when
        # the user set a torch dtype (classic converts at set time)
        dt = self.data_type
        if dt is not None and type(dt).__module__ == "torch":
            from .datatypes import _torch_to_cudnn_data_type

            return _torch_to_cudnn_data_type(dt)
        return self.data_type

    def get_is_virtual(self) -> bool:
        return self.is_virtual

    def get_is_pass_by_value(self) -> bool:
        return self.is_pass_by_value

    def get_reordering_type(self) -> Any:
        return self.reordering_type

    def get_ragged_offset_multiplier(self) -> int:
        return self.ragged_offset_multiplier

    def validate(self) -> None:
        """Validate tensor configuration."""
        if not self.dim:
            raise ValueError(f"Tensor '{self.name}' dims not set.")
        if not self.stride:
            raise ValueError(f"Tensor '{self.name}' strides not set.")
        if len(self.dim) != len(self.stride):
            raise ValueError(f"Tensor '{self.name}' dim/stride length mismatch: " f"{len(self.dim)} vs {len(self.stride)}")
        if self.is_virtual and self.is_pass_by_value:
            raise ValueError(f"Tensor '{self.name}' can't be both virtual and pass_by_value.")

    # NOTE: hash/eq are object identity (dataclass eq=False). uid and name are
    # mutable, so value-based hashing would violate the dict-key invariant.


def describing_tensor(uid: int, dim, stride, data_type) -> Tensor:
    """A Tensor describing a caller's buffer, built without the dataclass
    ``__init__``.

    ``execute()`` builds one of these per operand per call; the generated
    ``__init__`` would set seventeen attributes and run two default factories
    where only four fields are known. Every field
    left unset resolves to the class attribute the dataclass already installed
    for its default, so the result is indistinguishable from ``Tensor(...)`` --
    ``test_describing_tensor_matches_the_dataclass`` compares them field by
    field, and fails loudly if a new field arrives with a ``default_factory``
    (those get no class attribute, so reading one would raise).
    """
    tensor = object.__new__(Tensor)
    attributes = tensor.__dict__
    attributes["uid"] = uid
    attributes["dim"] = dim
    attributes["stride"] = stride
    attributes["data_type"] = data_type
    return tensor


def storage_geometry(dim, stride, data_type):
    """A cuDNN (element) geometry as the STORAGE-slot geometry a buffer reports.

    Every dtype but fp4 stores one element per slot, so the geometry is its
    own. fp4 packs two elements per slot along the unit-stride axis (torch's
    ``float4_e2m1fn_x2``, or a uint8 view): that extent halves and every other
    stride halves with it. None when the extent is odd -- no slot geometry
    spells it. Shared by the variant pack (which stores slots) and the engines
    that compare a slot against a declaration.
    """
    dim = tuple(int(d) for d in dim)
    if stride:
        stride = tuple(int(x) for x in stride)
    else:
        acc, dense = 1, []
        for d in reversed(dim):
            dense.insert(0, acc)
            acc *= d
        stride = tuple(dense)
    if data_type != _fp4_enum():
        return dim, stride
    # The packed axis is a unit-stride axis with an even extent above one -- a
    # singleton axis may also carry stride 1 and must not be the one picked.
    for c, (extent, step) in enumerate(zip(dim, stride)):
        if step != 1 or extent <= 1 or extent % 2:
            continue
        if any(x % 2 for j, x in enumerate(stride) if j != c and x != 1):
            continue
        # a singleton axis keeps its (never stepped) unit stride
        return tuple(d // 2 if j == c else d for j, d in enumerate(dim)), tuple(x if (j == c or x == 1) else x // 2 for j, x in enumerate(stride))
    return None


_FP4_ENUM = None


def _fp4_enum():
    # cudnn imports this module, so the enum is fetched on first use, once:
    # storage_geometry runs per overridden operand per execute.
    global _FP4_ENUM
    if _FP4_ENUM is None:
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

        _FP4_ENUM = cudnn.data_type.FP4_E2M1
    return _FP4_ENUM


def storage_slot_bytes(data_type) -> "int | None":
    """Bytes per STORAGE slot of a declared dtype: 1 for fp4 (two elements per
    slot), the element width otherwise, None when the width is unknown."""
    if data_type == _fp4_enum():
        return 1
    from .datatypes import _CUDNN_TO_FROST_DTYPE_NAME
    from .frost.buffers import DTYPE_ITEMSIZE

    name = _CUDNN_TO_FROST_DTYPE_NAME.get(data_type)
    return None if name is None else int(DTYPE_ITEMSIZE[name])


def byte_size(tensor: Tensor) -> int:
    """Bytes a dense tensor of this dim and dtype occupies, or 0 when the dtype
    has no known width (a bare address describes neither)."""
    from .datatypes import _CUDNN_TO_FROST_DTYPE_NAME
    from .frost.buffers import DTYPE_ITEMSIZE

    name = _CUDNN_TO_FROST_DTYPE_NAME.get(tensor.data_type)
    if name is None or not tensor.dim:
        return 0
    total = DTYPE_ITEMSIZE[name]
    for extent in tensor.dim:
        total *= int(extent)
    return total
