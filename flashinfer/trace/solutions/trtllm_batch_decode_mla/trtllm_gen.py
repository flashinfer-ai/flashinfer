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

"""FlashInfer trtllm-gen solution for trtllm_batch_decode_mla."""

from flashinfer.mla._core import trtllm_batch_decode_with_kv_cache_mla as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "trtllm_batch_decode_mla"
api = "flashinfer.mla._core.trtllm_batch_decode_with_kv_cache_mla"
backend = "trtllm-gen"
inputs = (
    "query",
    "kv_cache",
    "workspace_buffer",
    "qk_nope_head_dim",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "block_tables",
    "seq_lens",
    "max_seq_len",
    "bmm1_scale",
    "bmm2_scale",
    "skip_softmax_threshold_scale_factor",
)
outputs = ("output",)
api_kwargs = {
    "query": "query",
    "kv_cache": "kv_cache",
    "workspace_buffer": "workspace_buffer",
    "qk_nope_head_dim": "qk_nope_head_dim",
    "kv_lora_rank": "kv_lora_rank",
    "qk_rope_head_dim": "qk_rope_head_dim",
    "block_tables": "block_tables",
    "seq_lens": "seq_lens",
    "max_seq_len": "max_seq_len",
    "bmm1_scale": "bmm1_scale",
    "bmm2_scale": "bmm2_scale",
    "skip_softmax_threshold_scale_factor": "skip_softmax_threshold_scale_factor",
}


def run(
    query,
    kv_cache,
    workspace_buffer,
    qk_nope_head_dim,
    kv_lora_rank,
    qk_rope_head_dim,
    block_tables,
    seq_lens,
    max_seq_len,
    bmm1_scale,
    bmm2_scale,
    skip_softmax_threshold_scale_factor,
):
    with solution_autotune(
        definition,
        backend,
        query,
        kv_cache,
        workspace_buffer,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        block_tables,
        seq_lens,
        max_seq_len,
        bmm1_scale,
        bmm2_scale,
        skip_softmax_threshold_scale_factor,
    ):
        result = _api(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            backend=backend,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "trtllm_batch_decode_mla"
            + " returned None without mutating declared outputs"
        )
