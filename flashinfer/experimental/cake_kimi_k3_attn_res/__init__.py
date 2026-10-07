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

# Experimental Cake backend: Kimi-K3 attention-residual mixing (AttnRes) on
# SM100 / SM103 (tracker flashinfer-ai/flashinfer#4254).  The public entry
# points are ``flashinfer.kimi_k3_attn_res.kimi_k3_attn_res`` and
# ``prepare_kimi_k3_attn_res``; the host planner, launch binding, JIT
# registration and the generated sources live in this package.

from .cake_backend import (
    KimiK3AttnResRunner,
    RoutePlan,
    common_path_eligible,
    generated_program_available,
    kimi_k3_attn_res,
    plan_route,
    prepare_kimi_k3_attn_res,
    reference_kimi_k3_attn_res,
    validate_kimi_k3_attn_res_inputs,
)
from .cake_jit import (
    KERNELS,
    MODULES,
    gen_cake_kimi_k3_attn_res_module,
    load_cake_kimi_k3_attn_res_module,
    registered_kernel_keys,
)

__all__ = [
    "KERNELS",
    "MODULES",
    "KimiK3AttnResRunner",
    "RoutePlan",
    "common_path_eligible",
    "gen_cake_kimi_k3_attn_res_module",
    "generated_program_available",
    "kimi_k3_attn_res",
    "load_cake_kimi_k3_attn_res_module",
    "plan_route",
    "prepare_kimi_k3_attn_res",
    "reference_kimi_k3_attn_res",
    "registered_kernel_keys",
    "validate_kimi_k3_attn_res_inputs",
]
