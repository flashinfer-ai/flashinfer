#
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""``backend="cake"``: the Cake MXFP4 x MXFP8 SiTU routed MoE (Kimi K3 W4A8) lowered to CUDA C++ (tvm_ffi_utils bindings, pointer TMA ABI).

Generated package (see the manifest ``csrc/fused_moe/cake_mxfp4_situ_moe/cake_mxfp4_situ_moe_manifest.json`` for the pinned Cake
revision and the per-file inventory).  The public surface is the host plan of
:mod:`.cake_mxfp4_situ_moe_plan` and the kernel loader of
:mod:`.cake_mxfp4_situ_moe_kernels`.  Do not edit.
"""

from .cake_mxfp4_situ_moe_kernels import (
    ARCH,
    BACKEND,
    Kernel,
    cake_revision,
    compile_all,
    is_available,
    load,
    manifest,
    modules,
    select_gemm1,
    select_gemm2,
    select_routing,
    unsupported_forms,
)
from .cake_mxfp4_situ_moe_plan import (
    PACKAGE_BACKEND,
    CakeMxfp4MoEWrapper,
    CakeSwapAbPlan,
    CakeSwapAbPolicy,
    GemmForm,
    Launch,
    PlanDecision,
    RankLayout,
    RoutingConfig,
    WorkspaceField,
    kernel_stages,
    plan_decision,
    prepare_cake_mxfp4_weights,
    resolve_layout,
    routing_config,
)

__all__ = [
    "ARCH",
    "BACKEND",
    "PACKAGE_BACKEND",
    "CakeMxfp4MoEWrapper",
    "CakeSwapAbPlan",
    "CakeSwapAbPolicy",
    "GemmForm",
    "Kernel",
    "Launch",
    "PlanDecision",
    "RankLayout",
    "RoutingConfig",
    "WorkspaceField",
    "cake_revision",
    "compile_all",
    "is_available",
    "kernel_stages",
    "load",
    "manifest",
    "modules",
    "plan_decision",
    "prepare_cake_mxfp4_weights",
    "resolve_layout",
    "routing_config",
    "select_gemm1",
    "select_gemm2",
    "select_routing",
    "unsupported_forms",
]
