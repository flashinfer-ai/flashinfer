# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import cutlass.cute as cute


def dumb_output_transform(x: cute.Tensor, scale: float) -> cute.Tensor:
    return x * scale * 2.0
