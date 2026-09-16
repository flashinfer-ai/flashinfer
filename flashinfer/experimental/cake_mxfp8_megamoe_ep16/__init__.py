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

# Experimental MXFP8 MegaMoE EP16 implementation.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

    from ...moe_ep.cake_mxfp8_megamoe_ep16 import CakeMxfp8MegaMoeEp16Weights


def preprocess_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> CakeMxfp8MegaMoeEp16Weights:
    from .backend import preprocess_cake_mxfp8_megamoe_ep16_weights

    return preprocess_cake_mxfp8_megamoe_ep16_weights(w13, w2)


def create_session(
    weights: CakeMxfp8MegaMoeEp16Weights,
    topk_ids: torch.Tensor,
    *,
    process_group: dist.ProcessGroup | None = None,
) -> Any:
    from .backend import CakeMxfp8MegaMoeEp16

    return CakeMxfp8MegaMoeEp16(
        weights,
        topk_ids,
        process_group=process_group,
    )


__all__ = ["create_session", "preprocess_weights"]
