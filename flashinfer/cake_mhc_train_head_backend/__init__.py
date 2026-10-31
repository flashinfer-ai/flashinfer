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

Compatibility alias of :mod:`flashinfer.cake_mhc_det_train_head_backend` under
its former name ``flashinfer.cake_mhc_train_head_backend``: every name below is
the same object as the ``flashinfer.cake_mhc_det_train_head_backend`` name it
is imported from.
"""

from ..cake_mhc_det_train_head_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeMHCDetTrainHeadFunction as CakeMHCTrainHeadFunction,
    cake_mhc_det_train_head as cake_mhc_train_head,
    cake_mhc_det_train_head_backward as cake_mhc_train_head_backward,
    cake_mhc_det_train_head_forward as cake_mhc_train_head_forward,
    is_cake_mhc_det_train_head_supported as is_cake_mhc_train_head_supported,
)

__all__ = [
    "ARCHES",
    "CakeMHCTrainHeadFunction",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "cake_mhc_train_head",
    "cake_mhc_train_head_backward",
    "cake_mhc_train_head_forward",
    "is_cake_mhc_train_head_supported",
]
