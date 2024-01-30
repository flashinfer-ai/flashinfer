"""
Copyright (c) 2023 by FlashInfer team.

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
import torch

try:
    from . import _kernels
except ImportError:
    _kernels = None
from .utils import TensorLayout, check_kv_layout


def append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    append_indptr: torch.Tensor,
    kv_data: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: str = "NHD",
):
    check_kv_layout(kv_layout)
    _kernels.append_paged_kv_cache(
        append_key,
        append_value,
        append_indptr,
        kv_data,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        getattr(TensorLayout, kv_layout),
    )
