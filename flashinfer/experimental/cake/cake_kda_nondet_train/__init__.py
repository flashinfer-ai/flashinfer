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

# Experimental Cake backend: packed KDA (Kimi Delta Attention) chunked
# training forward and backward in the GLM-5.3-Flash geometry (head dimension
# 128, chunk 64, per-document zero initial state) on SM100 / SM103 / SM107, the
# non-deterministic counterpart of the deterministic package
# ``flashinfer.experimental.cake.cake_kda_det_train``.  The public entry points
# are ``flashinfer.cake_kda_nondet_train.cake_kda_nondet_train_forward``,
# ``cake_kda_nondet_train_backward`` and ``cake_kda_nondet``; the host plan,
# workspace, launch sequences, autograd Function, registry and generated sources
# live here.  The entries take no mode argument and refuse a workspace, saved
# set or arena option of the deterministic package with ``ValueError``.  The
# package imports nothing from the deterministic package.

from .cake_backend import (
    SAVE_MODES,
    SAVED_VERSION,
    SAVED_VERSION_TAPE,
    SCALE,
    SEG_SAVE_MIN_TOKENS_PER_SEG,
    CakeKDANondetArena,
    CakeKDANondetChunkSaved,
    CakeKDANondetFunction,
    CakeKDANondetWorkspace,
    cake_kda_nondet,
    cake_kda_nondet_train_backward,
    cake_kda_nondet_train_forward,
)
from .cake_jit import ARCH_BY_CAPABILITY, ARCHES, S1_RECORDS

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "S1_RECORDS",
    "SAVE_MODES",
    "SAVED_VERSION",
    "SAVED_VERSION_TAPE",
    "SCALE",
    "SEG_SAVE_MIN_TOKENS_PER_SEG",
    "CakeKDANondetArena",
    "CakeKDANondetChunkSaved",
    "CakeKDANondetFunction",
    "CakeKDANondetWorkspace",
    "cake_kda_nondet",
    "cake_kda_nondet_train_backward",
    "cake_kda_nondet_train_forward",
]
