# Copyright (c) 2025 by FlashInfer team.
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
flashinfer.trace — TraceTemplate system for fi_trace.

Usage::

    from flashinfer.trace import TraceTemplate, Var, Const, Tensor, Scalar
"""

from .solution import (
    BuildSpec,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedLanguages,
)
from .template import (
    Const,
    Scalar,
    Tensor,
    TraceTemplate,
    Var,
    _TRACE_DUMP_DIR,
    default_check,
    default_tolerances,
    standard_check,
)

__all__ = [
    "_TRACE_DUMP_DIR",
    "BuildSpec",
    "Const",
    "Scalar",
    "Solution",
    "SourceFile",
    "SupportedBindings",
    "SupportedLanguages",
    "Tensor",
    "TraceTemplate",
    "Var",
    "default_check",
    "default_tolerances",
    "standard_check",
]
