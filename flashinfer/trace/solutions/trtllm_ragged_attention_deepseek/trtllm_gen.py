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

"""FlashInfer trtllm-gen solution for trtllm_ragged_attention_deepseek."""

from flashinfer.prefill import trtllm_ragged_attention_deepseek as _api
from flashinfer.trace.solutions._helpers import first_output, solution_autotune

definition = "trtllm_ragged_attention_deepseek"
api = "flashinfer.prefill.trtllm_ragged_attention_deepseek"
backend = "trtllm-gen"
inputs = (
    "query",
    "key",
    "value",
    "workspace_buffer",
    "seq_lens",
    "max_q_len",
    "max_kv_len",
    "bmm1_scale",
    "bmm2_scale",
    "o_sf_scale",
    "batch_size",
    "window_left",
    "cum_seq_lens_q",
    "cum_seq_lens_kv",
    "is_causal",
    "return_lse",
    "enable_pdl",
    "skip_softmax_threshold_scale_factor",
)
outputs = ("output",)
api_kwargs = {
    "query": "query",
    "key": "key",
    "value": "value",
    "workspace_buffer": "workspace_buffer",
    "seq_lens": "seq_lens",
    "max_q_len": "max_q_len",
    "max_kv_len": "max_kv_len",
    "bmm1_scale": "bmm1_scale",
    "bmm2_scale": "bmm2_scale",
    "o_sf_scale": "o_sf_scale",
    "batch_size": "batch_size",
    "window_left": "window_left",
    "cum_seq_lens_q": "cum_seq_lens_q",
    "cum_seq_lens_kv": "cum_seq_lens_kv",
    "is_causal": "is_causal",
    "return_lse": "return_lse",
    "enable_pdl": "enable_pdl",
    "skip_softmax_threshold_scale_factor": "skip_softmax_threshold_scale_factor",
}


def run(
    query,
    key,
    value,
    workspace_buffer,
    seq_lens,
    max_q_len,
    max_kv_len,
    bmm1_scale,
    bmm2_scale,
    o_sf_scale,
    batch_size,
    window_left,
    cum_seq_lens_q,
    cum_seq_lens_kv,
    is_causal,
    return_lse,
    enable_pdl,
    skip_softmax_threshold_scale_factor,
):
    with solution_autotune(
        definition,
        backend,
        query,
        key,
        value,
        workspace_buffer,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        o_sf_scale,
        batch_size,
        window_left,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        is_causal,
        return_lse,
        enable_pdl,
        skip_softmax_threshold_scale_factor,
    ):
        result = _api(
            query=query,
            key=key,
            value=value,
            workspace_buffer=workspace_buffer,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            o_sf_scale=o_sf_scale,
            batch_size=batch_size,
            window_left=window_left,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            enable_pdl=enable_pdl,
            is_causal=is_causal,
            return_lse=return_lse,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            backend=backend,
        )
        return first_output(result, definition)
