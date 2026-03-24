"""
Copyright (c) 2024 by FlashInfer team.

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

from .topk import (
    can_implement_filtered_topk,
    get_topk_module,
    top_k,
    top_k_page_table_transform,
    top_k_ragged_transform,
    topk,
)

__all__ = [
    "top_k",
    "topk",
    "top_k_page_table_transform",
    "top_k_ragged_transform",
    "can_implement_filtered_topk",
    "get_topk_module",
]

_cute_dsl_available = False
try:
    from ..cute_dsl import is_cute_dsl_available

    if is_cute_dsl_available():
        from .kernels.top_k_cute_dsl import top_k_cute_dsl  # noqa: F401

        _cute_dsl_available = True
except ImportError:
    pass

if _cute_dsl_available:
    __all__ += [
        "top_k_cute_dsl",
    ]
