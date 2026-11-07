# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
# Ported from ResInfer 51b906836a69fa43915040314df57bc9f23030ad.
# Original source: csrc/resattn/kv_layout.py
"""Views of the stock NVFP4 KV cache consumed by ResAttn decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

SF_VEC = 16
PAGE_SIZE = 128


@dataclass
class NvFp4PagedKVPool:
    """Expose packed data and scale views of a vLLM NVFP4 KV cache."""

    num_kv_heads: int
    head_dim: int
    k_data: torch.Tensor
    v_data: torch.Tensor
    k_sf: torch.Tensor
    v_sf: torch.Tensor
    page_size: int = PAGE_SIZE
    k_global_scale: float = 1.0
    v_global_scale: float = 1.0
    layout_profile: ClassVar[str] = "trtllm"
