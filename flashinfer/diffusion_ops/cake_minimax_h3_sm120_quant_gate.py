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
"""

import torch

from ..utils import get_compute_capability


def require_sm120(device: torch.device, op_name: str) -> None:
    major, minor = get_compute_capability(device)
    if (major, minor) != (12, 0):
        raise NotImplementedError(
            f"MiniMax-H3 SM120 quantized {op_name} is built for compute capability 12.0 "
            f"(RTX 5090 / RTX PRO 6000 Blackwell) only, got {major}.{minor}"
        )
