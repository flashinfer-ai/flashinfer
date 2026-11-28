# Copyright (c) 2026 by FlashInfer team.
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

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from flashinfer.mla import _cudnn_sparse


pytestmark = pytest.mark.cpu_only
_MISSING = object()


@pytest.fixture
def cudnn_module(monkeypatch):
    module = ModuleType("cudnn")
    monkeypatch.setitem(sys.modules, "cudnn", module)
    _cudnn_sparse._get_sparse_attention_forward.cache_clear()
    yield module
    _cudnn_sparse._get_sparse_attention_forward.cache_clear()


@pytest.mark.parametrize(
    "version",
    [
        pytest.param(_MISSING, id="missing"),
        pytest.param("not-a-version", id="malformed"),
        pytest.param(None, id="non-string"),
        pytest.param("1.31.0", id="too-old"),
    ],
)
def test_unusable_cudnn_version_declines_auto_and_rejects_explicit(
    monkeypatch, cudnn_module, version
):
    if version is not _MISSING:
        cudnn_module.__version__ = version
    cudnn_module.DSA = SimpleNamespace(SparseAttentionForward=object())

    # Simulate a compatible GPU while exercising the real dependency probe.
    monkeypatch.setattr(_cudnn_sparse, "_incompatibility_reason", lambda *a, **kw: None)
    args = dict(
        query=torch.empty((128, 1, 64, 576), dtype=torch.bfloat16),
        kv_cache=torch.empty((1, 1, 32, 576), dtype=torch.bfloat16),
        workspace_buffer=torch.empty(128 * 64 * 8, dtype=torch.uint8),
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=torch.zeros((128, 1, 1), dtype=torch.int32),
        seq_lens=torch.full((128,), 32, dtype=torch.int32),
        sparse_mla_top_k=1,
        out=None,
        bmm1_scale=1 / 16,
        bmm2_scale=1,
        sinks=None,
        skip_softmax_threshold_scale_factor=None,
        enable_pdl=False,
        uses_shared_paged_kv_idx=True,
        lse=None,
        return_lse=False,
        return_lse_base=None,
        cum_seq_lens_q=None,
        max_q_len=None,
        multi_ctas_kv_counter_buffer=None,
        sparse_mla_top_k_lens=None,
        use_fp16_softmax=False,
    )
    assert _cudnn_sparse.try_cudnn_sparse_mla(**args) is None
    with pytest.raises(ImportError, match="nvidia-cudnn-frontend>=1.32.0"):
        _cudnn_sparse.try_cudnn_sparse_mla(**args, required=True)


def test_supported_cudnn_version_returns_dsa_operation(cudnn_module):
    operation = object()
    cudnn_module.__version__ = "1.32.0"
    cudnn_module.DSA = SimpleNamespace(SparseAttentionForward=operation)

    assert _cudnn_sparse._get_sparse_attention_forward() is operation
