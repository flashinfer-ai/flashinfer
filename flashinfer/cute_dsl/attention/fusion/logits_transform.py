# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import cutlass.cute as cute


def sigmoid_logits_transform(x: cute.Tensor) -> cute.Tensor:
    scale = 1.0 * math.log2(math.exp(1.0))
    bias = 0.0
    return 1 / (1 + cute.arch.exp2(-(x * scale + bias)))
