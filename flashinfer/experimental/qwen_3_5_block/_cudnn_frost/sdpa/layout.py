# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pure SDPA layout contracts shared by analysis and kernel configuration."""


def dense_layout_ok(dim: tuple, stride: tuple) -> bool:
    """Soundness of a rank-4 (B, H, S, D) dense layout.

    B/H/S physical nesting and padding are free. Kernel configuration adds
    its TMA pitch alignment requirements to this shared address-set check.
    What must hold:

      * head dim (axis 3) innermost-contiguous: stride 1 (size-1 wildcarded);
      * no zero stride on a size>1 dim — a broadcast view aliases distinct
        logical elements onto one address, which is ill-defined for the O
        write-back and rejected uniformly;
      * non-overlapping: visiting dims by ascending stride, each stride must
        cover the span of every dim below it (dense == equality; padded is
        larger; SMALLER than dense would alias, so it is rejected).

    Size-1 dims are wildcards: any stride, 0 and garbage included (cuDNN's
    own harnesses corrupt them deliberately).
    """
    if dim[3] != 1 and stride[3] != 1:
        return False
    active = sorted((int(st), int(sz)) for st, sz in zip(stride, dim) if sz != 1)
    span = 1
    for st, sz in active:
        if st < span:  # st == 0 (broadcast) or overlapping / sub-dense
            return False
        span = st * sz
    return True
