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

# Experimental fused mHC post -> pre (+ input RMSNorm) with specialized SM103 kernels.
#
# The public entry points are flashinfer.mhc.mhc_fused_post_pre,
# mhc_fused_post_pre_supported and mhc_fused_post_pre_warmup (marked
# @flashinfer_experimental_api); they hand off to this package, which holds the support
# checks, the JIT module, the kernels and the composable fallback. Nothing here is
# compiled at import time.

from .backend import SUPPORTED_SURFACE, VARIANTS, run, stats, supported, warmup
from .jit import gen_mhc_fused_post_pre_module
from .reference import mhc_fused_post_pre_reference

__all__ = [
    "SUPPORTED_SURFACE",
    "VARIANTS",
    "gen_mhc_fused_post_pre_module",
    "mhc_fused_post_pre_reference",
    "run",
    "stats",
    "supported",
    "warmup",
]
