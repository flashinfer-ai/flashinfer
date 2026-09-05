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

import functools
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit.dsv3_indexer import gen_dsv3_indexer_histogram_module
from flashinfer.utils import (
    backend_requirement,
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)


@functools.cache
def get_dsv3_indexer_histogram_module():
    module = gen_dsv3_indexer_histogram_module().build_and_load()

    @register_custom_op(
        "flashinfer::dsv3_topk_indexer", mutates_args=("histogram", "logits", "indices")
    )
    def _dsv3_topk_indexer(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        histogram: torch.Tensor,
        sm_map: torch.Tensor,
        logits: torch.Tensor,
        indices: torch.Tensor,
        pdl_enabled: bool,
        sm_multiple: int,
        num_cached: int,
        num_clusters: int,
        global_topk_overflow: int,
    ) -> None:
        module.dsv3_topk_indexer(
            q,
            k_cache,
            weights,
            seq_lens,
            block_table,
            histogram,
            sm_map,
            logits,
            indices,
            pdl_enabled,
            sm_multiple,
            num_cached,
            num_clusters,
            global_topk_overflow,
        )

    @register_fake_op("flashinfer::dsv3_topk_indexer")
    def _fake_dsv3_topk_indexer(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        histogram: torch.Tensor,
        sm_map: torch.Tensor,
        logits: torch.Tensor,
        indices: torch.Tensor,
        pdl_enabled: bool,
        sm_multiple: int,
        num_cached: int,
        num_clusters: int,
        global_topk_overflow: int,
    ) -> None:
        pass

    @register_custom_op("flashinfer::get_indexer_metadata", mutates_args=())
    def _get_indexer_metadata(
        seq_lens: torch.Tensor,
        num_physical_sms: int,
    ) -> torch.Tensor:
        return module.get_indexer_metadata(seq_lens, num_physical_sms)

    @register_fake_op("flashinfer::get_indexer_metadata")
    def _fake_get_indexer_metadata(
        seq_lens: torch.Tensor,
        num_physical_sms: int,
    ) -> torch.Tensor:
        batch_size = seq_lens.size(0)
        num_logical_sms = (
            (batch_size + num_physical_sms - 1) // num_physical_sms * num_physical_sms
        )
        return torch.empty(
            num_logical_sms, 4, dtype=torch.int32, device=seq_lens.device
        )

    return SimpleNamespace(
        dsv3_topk_indexer=_dsv3_topk_indexer,
        get_indexer_metadata=_get_indexer_metadata,
    )


def _dsv3_indexer_histogram_num_clusters(batch_size: int) -> int:
    # low batch size, allocate more clusters to get more parallelism
    # high batch size, more parallelism available per row
    if batch_size <= 32:
        return 8
    elif batch_size < 128:
        return 4
    else:
        return 2


@supported_compute_capability([100, 103])
def _check_dsv3_indexer_histogram_supported(
    *args,
    **kwargs,
) -> bool:
    return True


@backend_requirement({}, common_check=_check_dsv3_indexer_histogram_supported)
@flashinfer_api
def get_indexer_metadata(
    seq_lens: torch.Tensor, num_sms: Optional[int] = None
) -> torch.Tensor:
    """Compute SM mapping metadata for MQA load balancing.

    Args:
        seq_lens: [batch] int32 CUDA tensor of sequence lengths.
        num_sms: Number of physical SMs. If None, auto-detected from device.

    Returns:
        sm_map: [num_logical_sms, 4] int32 CUDA tensor.
    """
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(
            seq_lens.device
        ).multi_processor_count
    return get_dsv3_indexer_histogram_module().get_indexer_metadata(seq_lens, num_sms)


@backend_requirement({}, common_check=_check_dsv3_indexer_histogram_supported)
@flashinfer_api
def dsv3_topk_indexer(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    sm_map: Optional[torch.Tensor] = None,
    max_model_len: int = 163840,
    exact_topk=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused DSV3.2 indexer, performs the following for each batch bi:
        - dequant_kvcache = dequant(K, seq_lens[bi]) # [seq_len, 128]
        - logits = (relu(q[bi] @ dequant_kvcache) * weights[bi][:, None]).sum(0) # [seq_len]
        - topk(logits, k = 2048)

    Args:
        q:           [batch, 64, 128] fp8/uint8 CUDA tensor.
        k_cache:     [num_pages, 64, 1, 132] fp8/uint8 CUDA tensor.
        weights:     [batch, 64] float32 CUDA tensor.
        seq_lens:    [batch] int32 CUDA tensor.
        block_table: [batch, max_num_pages] int32 CUDA tensor.
        sm_map:      Optional [num_sms, 4] int32 tensor from get_indexer_metadata().
                     Auto-computed if None. Note that sm_map depends on the sequence length only, so it's cost
                     can be amortized per forward pass if you compute this once at the beginning
        max_model_len: Maximum sequence length to allocate logits buffer for.
        exact_topk:  if true then the exact topk over the entire sequence length is computed,
                     otherwise, a faster approximate algorithm is used that prefers elements at the start
                     of the sequence. The speed and accuracy of topk is dependent on the logit distribution,
                     but for distribution with outliers the approximate algorithm can pick those up.

    Returns:
        (indices, logits): indices [batch, 2048] int32, logits [batch, max_model_len] float32.
    """
    batch_size = q.shape[0]
    histogram = torch.zeros(batch_size, 256, device=q.device, dtype=torch.int32)
    max_model_len = (
        (max_model_len + 3) // 4
    ) * 4  # must be aligned to 4 for the kernel
    logits = torch.empty(
        batch_size, max_model_len, device=q.device, dtype=torch.float32
    )
    indices = torch.empty(batch_size, 2048, device=q.device, dtype=torch.int32)
    if sm_map is None:
        sm_map = get_indexer_metadata(seq_lens)

    num_clusters = _dsv3_indexer_histogram_num_clusters(batch_size)

    if exact_topk:
        topk_global_overflow = max_model_len // num_clusters
    else:
        topk_global_overflow = 0

    get_dsv3_indexer_histogram_module().dsv3_topk_indexer(
        q,
        k_cache,
        weights,
        seq_lens,
        block_table,
        histogram,
        sm_map,
        logits,
        indices,
        True,  # PDL enabled
        1,  # sm multiple
        4096,  # num_cached, found via sweep, larger seems to decrease occupancy, it's a balance between global overflow and occupancy
        num_clusters,
        topk_global_overflow,
    )
    return indices, logits
