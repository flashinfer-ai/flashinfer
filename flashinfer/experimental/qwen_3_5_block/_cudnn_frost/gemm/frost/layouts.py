# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure matrix storage geometry, independent of dtypes and runtime buffers."""

from math import prod


def fold_batch_layout(shape, stride, *, writable=False, allow_identity_broadcast=False):
    """Return the equivalent (batch, row, column) layout, or reject a non-affine fold.

    Matrix axes remain unchanged. Unit batch axes never contribute an address;
    every other batch axis must follow one common linear batch pitch. Gaps
    between matrices and strided/transposed matrix axes are allowed.
    ``allow_identity_broadcast`` permits zero batch pitch only when no rank
    folding is needed; writable storage must still be non-overlapping.
    """
    shape, stride = tuple(map(int, shape)), tuple(map(int, stride))
    if len(shape) < 2 or len(stride) != len(shape) or any(d <= 0 for d in shape) or any(s < 0 for s in stride):
        raise ValueError(f"matrix batch view requires rank >= 2, positive extents and nonnegative strides; got {shape}, {stride}")
    prefix = shape[:-2]
    batch = prod(prefix)
    active = [i for i, dim in enumerate(prefix) if dim != 1]
    pitch = stride[active[-1]] if active else stride[len(prefix) - 1] if prefix else max(shape[-2] * stride[-2], shape[-1] * stride[-1])
    zero_pitch_allowed = allow_identity_broadcast and len(shape) == 3
    if active and ((pitch == 0 and not zero_pitch_allowed) or any(stride[i] != pitch * prod(prefix[i + 1 :]) for i in active)):
        raise ValueError(f"matrix batch axes are not affine and cannot be folded without copying: shape={shape}, stride={stride}")
    span = 1 + sum((d - 1) * s for d, s in zip(shape, stride))
    if max(batch, pitch, span, *shape, *stride) > (1 << 63) - 1:
        raise ValueError("matrix batch view exceeds signed 64-bit geometry")
    if writable:
        occupied = 1
        for s, d in sorted((s, d) for d, s in zip(shape, stride) if d > 1):
            if s < occupied:
                raise ValueError("matrix output view requires non-overlapping storage")
            occupied += (d - 1) * s
    return (batch, *shape[-2:]), (pitch, *stride[-2:])
