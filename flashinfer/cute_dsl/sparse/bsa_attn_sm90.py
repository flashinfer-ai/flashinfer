# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch

from flashinfer.api_logging import flashinfer_api

from .bsa_attn_sm120 import _bsa_attn_fp16_blk64_fwd
from .bsa_utils.cache_utils import get_jit_cache

_sm90_compile_cache = get_jit_cache("bsa_fwd_sm90")


@flashinfer_api
def bsa_attn_sm90_blk64_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run FP16/BF16 block-sparse attention on SM90.

    ``q2k_block_index`` has shape ``(batch, num_heads, num_q_blocks,
    max_kv_blocks)``. Active entries must be valid KV-block indices and
    ``q2k_block_nums`` values must be in ``[0, max_kv_blocks]``. This low-level
    performance API treats those values as trusted metadata.
    """
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (9, 0):
        arch = capability[0] * 10 + capability[1]
        raise RuntimeError(
            f"bsa_attn_sm90_blk64_fwd only supports SM90, current device is SM{arch}"
        )
    with torch.cuda.device(q.device):
        return _bsa_attn_fp16_blk64_fwd(
            q,
            k,
            v,
            q2k_block_index,
            block_sparse_num,
            block_sizes,
            q2k_block_nums,
            softmax_scale,
            return_lse,
            out,
            lse,
            backend_name="sm90",
            arch_major=9,
            compile_cache=_sm90_compile_cache,
        )
