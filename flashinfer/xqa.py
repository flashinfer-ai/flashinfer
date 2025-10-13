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

import functools
from types import SimpleNamespace
from typing import Optional

import torch

from .jit.xqa import gen_xqa_module
from .utils import (
    register_custom_op,
    register_fake_op,
    get_compute_capability,
)


@functools.cache
def get_xqa_module(
    use_fp16: bool,
    token_per_page: int,
    head_size: int,
    head_grp_size: int,
    use_sliding_window: bool,
):
    module = gen_xqa_module(
        use_fp16, token_per_page, head_size, head_grp_size, use_sliding_window
    ).build_and_load()

    @register_custom_op(
        f"flashinfer::xqa_use_fp16_{use_fp16}_token_per_page_{token_per_page}_head_size_{head_size}_head_grp_size_{head_grp_size}_use_sliding_window_{use_sliding_window}",
        mutates_args=("output", "scratch"),
    )
    def xqa(
        multiProcessorCount: int,
        nbKHeads: int,
        slidingWinSize: int,
        qScale: float,
        output: torch.Tensor,
        q: torch.Tensor,
        attentionSinks: Optional[torch.Tensor],
        pool: torch.Tensor,
        kvCachePageList: torch.Tensor,
        maxSeqLen: int,
        seqLen: torch.Tensor,
        batchSize: int,
        kvCacheScale: torch.Tensor,
        semaphores: torch.Tensor,
        scratch: torch.Tensor,
    ) -> None:
        module.xqa_wrapper(
            multiProcessorCount,
            nbKHeads,
            slidingWinSize,
            qScale,
            output,
            q,
            attentionSinks,
            pool,
            kvCachePageList,
            maxSeqLen,
            seqLen,
            batchSize,
            kvCacheScale,
            semaphores,
            scratch,
        )

    @register_fake_op(
        f"flashinfer::xqa_use_fp16_{use_fp16}_token_per_page_{token_per_page}_head_size_{head_size}_head_grp_size_{head_grp_size}_use_sliding_window_{use_sliding_window}"
    )
    def _fake_xqa(
        multiProcessorCount: int,
        nbKHeads: int,
        slidingWinSize: int,
        qScale: float,
        output: torch.Tensor,
        q: torch.Tensor,
        attentionSinks: Optional[torch.Tensor],
        pool: torch.Tensor,
        kvCachePageList: torch.Tensor,
        maxSeqLen: int,
        seqLen: torch.Tensor,
        batchSize: int,
        kvCacheScale: torch.Tensor,
        semaphores: torch.Tensor,
        scratch: torch.Tensor,
    ) -> None:
        pass

    return SimpleNamespace(
        xqa=xqa,
    )


def xqa(
    use_fp16: bool,
    token_per_page: int,
    head_size: int,
    head_grp_size: int,
    use_sliding_window: bool,
    sliding_win_size: int,
    multiProcessorCount: int,
    nbKHeads: int,
    qScale: float,
    output: torch.Tensor,
    q: torch.Tensor,
    attentionSinks: Optional[torch.Tensor],
    pool: torch.Tensor,
    kvCachePageList: torch.Tensor,
    maxSeqLen: int,
    seqLen: torch.Tensor,
    batchSize: int,
    kvCacheScale: torch.Tensor,
    semaphores: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    if get_compute_capability(torch.device(device="cuda"))[0] != 9:
        raise RuntimeError("XQA is only supported on SM90 GPUs")
    xqa_module = get_xqa_module(
        use_fp16, token_per_page, head_size, head_grp_size, use_sliding_window
    )
    xqa_module.xqa(
        multiProcessorCount,
        nbKHeads,
        sliding_win_size if use_sliding_window else 0,
        qScale,
        output,
        q,
        attentionSinks,
        pool,
        kvCachePageList,
        maxSeqLen,
        seqLen,
        batchSize,
        kvCacheScale,
        semaphores,
        scratch,
    )
