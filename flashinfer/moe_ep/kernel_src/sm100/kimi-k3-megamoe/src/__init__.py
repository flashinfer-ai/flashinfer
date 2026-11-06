# Exported for FlashInfer; local changes are recorded in VENDOR.md.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel import (
    BlockScaledSwapAbMegaMoeKernel as BlackwellInferenceMegaMoE,
)

__all__ = ["BlackwellInferenceMegaMoE"]
