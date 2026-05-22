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
"""Monolithic CuTe DSL MLA decode kernels for Blackwell SM100.

This subpackage hosts the original single-file MLA decode kernels
(``BlackwellMultiHeadLatentAttentionForwardFP16`` / ``…FP8``) that were
introduced in #2743 and #2901. They were removed in #2805 in favor of the
modular kernel under ``flashinfer.cute_dsl.attention`` and restored here as
an alternate implementation under the same ``backend="cute-dsl"`` user
surface.

Selection between the modular and monolithic implementations is handled by
``flashinfer.cute_dsl.attention.cute_dsl_mla_decode``; this module is not
intended to be imported directly by users. To force the monolithic path,
pass ``cute_dsl_impl="monolithic"`` to the public API call.
"""

from flashinfer.cute_dsl.utils import is_cute_dsl_available

if is_cute_dsl_available():
    from .mla_decode import cute_dsl_mla_decode

__all__ = [
    "is_cute_dsl_available",
]

if is_cute_dsl_available():
    __all__ += [
        "cute_dsl_mla_decode",
    ]
