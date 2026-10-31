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

Compatibility alias of :mod:`flashinfer.cake_mhc_det_train_backend` under its
former name ``flashinfer.cake_mhc_train_backend``: every name below is the same
object as the ``flashinfer.cake_mhc_det_train_backend`` name it is imported
from.
"""

from ..cake_mhc_det_train_backend import (  # noqa: F401
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    TENSOR_ALIGNMENT_BYTES,
    WORKSPACE_ALIGNMENT_BYTES,
    CakeMHCDetTrainFinishFunction as CakeMHCTrainFinishFunction,
    CakeMHCDetTrainPrepareFunction as CakeMHCTrainPrepareFunction,
    cake_mhc_det_train_backward_finish as cake_mhc_train_backward_finish,
    cake_mhc_det_train_backward_prepare as cake_mhc_train_backward_prepare,
    cake_mhc_det_train_backward_workspace as cake_mhc_train_backward_workspace,
    cake_mhc_det_train_backward_workspace_bytes as cake_mhc_train_backward_workspace_bytes,
    cake_mhc_det_train_finish as cake_mhc_train_finish,
    cake_mhc_det_train_forward_finish as cake_mhc_train_forward_finish,
    cake_mhc_det_train_forward_prepare as cake_mhc_train_forward_prepare,
    cake_mhc_det_train_forward_stats as cake_mhc_train_forward_stats,
    cake_mhc_det_train_forward_workspace as cake_mhc_train_forward_workspace,
    cake_mhc_det_train_forward_workspace_bytes as cake_mhc_train_forward_workspace_bytes,
    cake_mhc_det_train_prepare as cake_mhc_train_prepare,
    cake_mhc_det_train_reset_workspace as cake_mhc_train_reset_workspace,
    is_cake_mhc_det_train_supported as is_cake_mhc_train_supported,
    route_key,
    route_modules,
)

__all__ = [
    "ARCHES",
    "CakeMHCTrainFinishFunction",
    "CakeMHCTrainPrepareFunction",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "TENSOR_ALIGNMENT_BYTES",
    "WORKSPACE_ALIGNMENT_BYTES",
    "cake_mhc_train_backward_finish",
    "cake_mhc_train_backward_prepare",
    "cake_mhc_train_backward_workspace",
    "cake_mhc_train_backward_workspace_bytes",
    "cake_mhc_train_finish",
    "cake_mhc_train_forward_finish",
    "cake_mhc_train_forward_prepare",
    "cake_mhc_train_forward_stats",
    "cake_mhc_train_forward_workspace",
    "cake_mhc_train_forward_workspace_bytes",
    "cake_mhc_train_prepare",
    "cake_mhc_train_reset_workspace",
    "is_cake_mhc_train_supported",
    "route_key",
    "route_modules",
]
