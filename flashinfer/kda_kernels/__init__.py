"""
Copyright (c) 2025 by FlashInfer team.

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

"""
KDA (Key-Driven Attention) Kernels - CuTe DSL Implementations
==============================================================

Per-K-dimension gating variant of GDN. Gate g[B,T,HV,K] applied per-lane
instead of GDN's scalar broadcast.

Exported:
- recurrent_kda: Recurrent KDA decode kernel (T=1)
"""

try:
    from .recurrent_kda import recurrent_kda

    _has_cute_dsl = True
except ImportError:
    _has_cute_dsl = False
    recurrent_kda = None  # type: ignore

__all__ = [
    "recurrent_kda",
]
