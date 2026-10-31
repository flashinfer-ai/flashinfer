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
"""Host-only validation for the immutable FP8 scoring math policy."""

from typing import Literal, cast

_MathMode = Literal["relu_fast", "relu_strict"]


def _validate_math_mode(math_mode: str) -> _MathMode:
    if not isinstance(math_mode, str) or math_mode not in (
        "relu_fast",
        "relu_strict",
    ):
        raise ValueError("math_mode must be 'relu_fast' or 'relu_strict'")
    return cast(_MathMode, math_mode)
