# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CuTe DSL MLA Decode Kernels for Blackwell SM100.

Backward-compat shim: re-exports from the modular attention framework.
"""

from flashinfer.cute_dsl.utils import is_cute_dsl_available

if is_cute_dsl_available():
    from flashinfer.cute_dsl.attention.wrappers.batch_mla import cute_dsl_mla_decode

__all__ = [
    "is_cute_dsl_available",
]

if is_cute_dsl_available():
    __all__ += [
        "cute_dsl_mla_decode",
    ]
