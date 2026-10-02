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

SM90 push BF16 mega-MoE backend.
"""

from .backend import Sm90PushBf16MegaKernelBackend
from .config import Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig
from .weights import preprocess_mega_weights, validate_transformed_mega_weights


def __getattr__(name: str) -> object:
    if name == "TransformedMegaWeights":
        from .weights import TransformedMegaWeights

        return TransformedMegaWeights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Sm90PushBf16MegaKernelBackend",
    "Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
