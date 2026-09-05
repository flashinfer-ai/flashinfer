# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compile-time FC1 activation geometry for the NVFP4 swap-AB kernels.

This module intentionally has no CUDA/CuTe imports.  Besides keeping the host
contract easy to test, it gives framework adapters one source of truth for the
difference between the physical FC1 projection width and the post-activation
width consumed by FC2.
"""

from typing import Literal, Sequence, Tuple


Fc1Activation = Literal["swiglu", "relu2"]
_SUPPORTED_FC1_ACTIVATIONS = ("swiglu", "relu2")


def validate_fc1_activation(activation: str) -> Fc1Activation:
    """Return a normalized activation or reject an unsupported codegen key."""
    if activation not in _SUPPORTED_FC1_ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {_SUPPORTED_FC1_ACTIVATIONS}, "
            f"got {activation!r}."
        )
    return activation  # type: ignore[return-value]


def fc1_projection_planes(activation: str) -> int:
    """Number of physical FC1 planes per post-activation output value."""
    activation = validate_fc1_activation(activation)
    return 2 if activation == "swiglu" else 1


def post_activation_width(physical_fc1_width: int, activation: str) -> int:
    """Derive FC2 K from the physical FC1 output width."""
    if physical_fc1_width <= 0:
        raise ValueError(
            f"physical_fc1_width must be positive, got {physical_fc1_width}."
        )
    planes = fc1_projection_planes(activation)
    if physical_fc1_width % planes != 0:
        raise ValueError(
            f"physical_fc1_width ({physical_fc1_width}) must be divisible by "
            f"the {planes} projection plane(s) required by {activation}."
        )
    return physical_fc1_width // planes


def physical_fc1_width(semantic_intermediate: int, activation: str) -> int:
    """Derive the FC1 GEMM width from the semantic post-activation width."""
    if semantic_intermediate <= 0:
        raise ValueError(
            f"semantic_intermediate must be positive, got {semantic_intermediate}."
        )
    return semantic_intermediate * fc1_projection_planes(activation)


def validate_fc1_fc2_widths(
    physical_width: int,
    fc2_k: int,
    activation: str,
) -> int:
    """Validate the FC1/FC2 hand-off and return its semantic width."""
    expected_fc2_k = post_activation_width(physical_width, activation)
    if fc2_k != expected_fc2_k:
        raise ValueError(
            f"fc2 K ({fc2_k}) does not match {activation} post-activation "
            f"width ({expected_fc2_k}) derived from physical FC1 width "
            f"({physical_width})."
        )
    return expected_fc2_k


def nvfp4_logical_shape(
    storage_shape: Sequence[int], packed_dim: int,
) -> Tuple[int, ...]:
    """Expand a torch ``float4_e2m1fn_x2`` storage shape to logical FP4.

    Torch exposes one byte (two FP4 values) as one element of its ``x2``
    dtype.  CuTe sees the corresponding logical FP4 extent.  Host-side
    launch validation must therefore expand the actual tensor shape before
    comparing it with the kernel contract; reconstructing the shape from a
    problem descriptor would miss a padded or mispacked tensor.
    """
    shape = tuple(map(int, storage_shape))
    if not shape:
        raise ValueError("NVFP4 storage shape must have at least one dimension.")
    packed_dim %= len(shape)
    logical = list(shape)
    logical[packed_dim] *= 2
    return tuple(logical)
