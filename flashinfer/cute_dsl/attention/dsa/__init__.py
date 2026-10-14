# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""DeepSeek sparse-attention kernels implemented with CuTe DSL."""

from .hca_fp8 import BlackwellHeavilyCompressedAttentionForwardFP8

__all__ = ["BlackwellHeavilyCompressedAttentionForwardFP8"]
