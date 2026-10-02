"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Configuration for the SM90 push BF16 mega-MoE backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig:
    """Static dimensions and protocol choices for the Hopper BF16 backend."""

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_bf16_bf16_bf16_push_cuda"
    capacity_factor: float = 1.0
    dedup_dispatch: bool = True
    grouped_combine: bool = False
    fuse_fc1_epilogue: bool = False
    allow_unverified_p2p: bool = False
    init_timeout_s: float = 600.0
    wave_schedule: Literal["mono", "serial2", "pipe2"] = "mono"


__all__ = ["Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig"]
