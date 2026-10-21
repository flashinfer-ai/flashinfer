# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python node class for cuDNN Frontend graph representation.

Simple, Pythonic design - everything stored directly on the Node.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Any

from .graph_types import NodeType, Tensor

if TYPE_CHECKING:
    from ._pygraph import GraphContext


class Node:
    """A single operation node in the computation graph.

    All operation-specific parameters are stored in the `params` dict,
    keeping the design simple and flexible.

    Attributes:
        name: Operation name
        node_type: Type of operation (MATMUL, POINTWISE, SDPA, etc.)
        inputs: Dict mapping port names to input Tensor
        outputs: Dict mapping port names to output Tensor
        params: Dict of operation-specific parameters (padding, stride, mode, etc.)
        compute_data_type: Data type for computation
    """

    def __init__(
        self,
        name: str,
        node_type: NodeType,
        compute_data_type: Any = None,
    ):
        self.name = name
        self.node_type = node_type
        self.compute_data_type = compute_data_type
        self.inputs: Dict[str, Tensor] = {}
        self.outputs: Dict[str, Tensor] = {}
        self.params: Dict[str, Any] = {}

    def validate(self) -> None:
        """Validate node configuration."""
        for port_name, tensor in self.inputs.items():
            if tensor is None:
                raise ValueError(f"Node '{self.name}': Input '{port_name}' is None")
        for port_name, tensor in self.outputs.items():
            if tensor is None:
                raise ValueError(f"Node '{self.name}': Output '{port_name}' is None")

        if self.node_type == NodeType.MATMUL:
            self._validate_matmul()
        elif self.node_type == NodeType.CONV_FPROP:
            self._validate_conv_fprop()

    def infer_properties(self, context: "GraphContext") -> None:
        """Infer unset tensor properties from context and inputs."""
        # SDPA Stats is always computed and stored in FP32, so an unset dtype must
        # not inherit a narrower io dtype: the kernel would write 4-byte rows past
        # the end of a buffer the caller sized from the declared dtype.
        if self.node_type in (NodeType.SDPA, NodeType.SDPA_FP8, NodeType.SDPA_MXFP8):
            stats = self.outputs.get("Stats")
            if stats is not None and stats.data_type is None:
                from ._compiled_module import data_type as _cudnn_data_type

                stats.data_type = _cudnn_data_type.FLOAT

        # Fill data types from context
        for tensor in self.inputs.values():
            if tensor and tensor.data_type is None:
                tensor.data_type = context.intermediate_data_type if tensor.is_virtual else context.io_data_type

        for tensor in self.outputs.values():
            if tensor and tensor.data_type is None:
                tensor.data_type = context.intermediate_data_type if tensor.is_virtual else context.io_data_type

        # Operation-specific inference
        if self.node_type == NodeType.MATMUL:
            self._infer_matmul()
        elif self.node_type == NodeType.POINTWISE:
            self._infer_pointwise()
        # structured/captured ops (norms/conv/moe/sdpa/...) infer at build time
        # via their tables' per-output lambdas

    def _validate_matmul(self) -> None:
        """Validate matmul dimensions: C = A @ B."""
        a = self.inputs.get("A")
        b = self.inputs.get("B")
        if not (a and b and a.dim and b.dim):
            return
        if a.dim[-1] != b.dim[-2]:
            raise ValueError(f"Node '{self.name}': Inner dimensions must match for matmul: " f"A{a.dim} @ B{b.dim}")

    def _validate_conv_fprop(self) -> None:
        """Validate the restricted NDHWC 3D convolution path."""
        x = self.inputs.get("image")
        w = self.inputs.get("weight")
        if not (x and w and x.dim and w.dim):
            return

        if any(extent <= 0 for extent in x.dim):
            raise ValueError(f"Node '{self.name}': X dimensions must be positive, got X{x.dim}")
        if any(extent <= 0 for extent in w.dim):
            raise ValueError(f"Node '{self.name}': W dimensions must be positive, got W{w.dim}")

    def _infer_matmul(self) -> None:
        """Infer output dims for matmul: C = A @ B."""
        a = self.inputs.get("A")
        b = self.inputs.get("B")
        c = self.outputs.get("C")

        if not (a and b and c):
            return

        if not c.dim and a.dim and b.dim:
            # Output shape: [..., M, N] where M=A[-2], N=B[-1]
            ndim = max(len(a.dim), len(b.dim))
            c_dim = [1] * ndim

            # Last two dims: M from A, N from B
            if len(a.dim) >= 2:
                c_dim[-2] = a.dim[-2]
            if len(b.dim) >= 2:
                c_dim[-1] = b.dim[-1]

            # Broadcast batch dims (incompatible extents raise, matching numpy
            # rules: equal, or one side is 1)
            for i in range(ndim - 2):
                a_idx = i - (ndim - len(a.dim))
                b_idx = i - (ndim - len(b.dim))
                a_val = a.dim[a_idx] if 0 <= a_idx < len(a.dim) - 2 else 1
                b_val = b.dim[b_idx] if 0 <= b_idx < len(b.dim) - 2 else 1
                if a_val != b_val and 1 not in (a_val, b_val):
                    raise ValueError(f"Node '{self.name}': batch dims not broadcastable: A{a.dim} vs B{b.dim}")
                c_dim[i] = max(a_val, b_val)

            c.dim = c_dim

        if not c.stride and c.dim:
            c.stride = _row_major_stride(c.dim)

    def _infer_pointwise(self) -> None:
        """Infer output dims for pointwise (broadcast inputs)."""
        out = self.outputs.get("OUT_0")
        if not out:
            return

        if not out.dim:
            # Right-aligned elementwise broadcast across all inputs (numpy
            # rules); lower-rank operands contribute to the trailing dims.
            max_dim: list = []
            for tensor in self.inputs.values():
                if not (tensor and tensor.dim):
                    continue
                d = list(tensor.dim)
                if len(d) > len(max_dim):
                    d, max_dim = max_dim, d  # keep max_dim the longer one
                for i in range(1, len(d) + 1):  # merge right-aligned
                    a, b = max_dim[-i], d[-i]
                    if a != b and 1 not in (a, b):
                        raise ValueError(f"Node '{self.name}': pointwise inputs not broadcastable")
                    max_dim[-i] = max(a, b)
            if max_dim:
                out.dim = max_dim

        if not out.stride and out.dim:
            out.stride = _row_major_stride(out.dim)

    def __repr__(self) -> str:
        return f"Node({self.name!r}, {self.node_type.name})"


def _row_major_stride(dim: List[int]) -> List[int]:
    """Compute row-major (C-contiguous) strides."""
    if not dim:
        return []
    stride = [1] * len(dim)
    for i in range(len(dim) - 2, -1, -1):
        stride[i] = stride[i + 1] * dim[i + 1]
    return stride
