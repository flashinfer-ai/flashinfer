# Copyright (c) 2025-2026 by FlashInfer team.
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

"""CPU trace coverage for FP8-Q/NVFP4-KV output and head dimensions."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize(
    "output_dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn]
)
@pytest.mark.parametrize("output_source", ["out_dtype", "out"])
@pytest.mark.parametrize("combined", [False, True])
def test_attention_ts_nvfp4_trace_output_dtype(
    head_dim, output_dtype, output_source, combined
):
    """FP8-Q/NVFP4-KV traces preserve H512 and the actual output dtype."""
    from flashinfer.trace.templates.attention import attention_ts_decode_trace_dispatch

    q = torch.empty(2, 8, head_dim, dtype=torch.float8_e4m3fn)
    k = torch.empty(4, 2, 32, head_dim // 2, dtype=torch.uint8)
    v = torch.empty_like(k)
    sf = torch.empty(4, 2, 32, head_dim // 16, dtype=torch.float8_e4m3fn)
    kwargs = dict(
        q=q,
        paged_kv_cache=torch.stack((k, v), dim=1) if combined else (k, v),
        kv_scale_factors=(sf, sf),
        block_tables=torch.arange(4, dtype=torch.int32).reshape(2, 2),
        seq_lens_kv=torch.full((2,), 32, dtype=torch.int32),
        seq_len_q=1,
    )
    if output_source == "out":
        kwargs["out"] = torch.empty(q.shape, dtype=output_dtype)
        # The supplied buffer takes precedence over the requested dtype.
        kwargs["out_dtype"] = torch.float16
    else:
        kwargs["out_dtype"] = output_dtype
    template = attention_ts_decode_trace_dispatch(**kwargs)
    definition = template.build_fi_trace_fn("batch_decode_with_paged_kv_cache")(
        **kwargs
    )
    assert definition["outputs"]["output"]["dtype"] == str(output_dtype).removeprefix(
        "torch."
    )
    assert "head_dim in (64, 128, 256, 512)" in definition["constraints"]


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize(
    "output_dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn]
)
@pytest.mark.parametrize("output_source", ["plan", "out"])
def test_prims_ts_nvfp4_wrapper_trace_output_dtype(
    head_dim, output_dtype, output_source
):
    """Wrapper traces use the plan dtype unless an explicit buffer overrides it."""
    from flashinfer.trace.templates.attention import (
        prims_ts_decode_wrapper_trace_dispatch,
    )

    q = torch.empty(2, 8, head_dim, dtype=torch.float8_e4m3fn)
    k = torch.empty(4, 2, 32, head_dim // 2, dtype=torch.uint8)
    sf = torch.empty(4, 2, 32, head_dim // 16, dtype=torch.float8_e4m3fn)
    state = SimpleNamespace(
        page_size=32,
        storage_page_size=32,
        use_packed_q=False,
        seq_len_q=1,
        output_dtype=output_dtype if output_source == "plan" else torch.float16,
        mask_type="dense",
        window_left=-1,
        max_kv_len=32,
        kv_prefix_mode="dynamic",
        kv_lengths_mode="dynamic",
        planned_seq_lens_device=None,
    )
    kwargs = dict(
        self=SimpleNamespace(_plan_state=state),
        q=q,
        paged_kv_cache=(k, k),
        kv_scale_factors=(sf, sf),
        block_tables=torch.arange(4, dtype=torch.int32).reshape(2, 2),
        seq_lens=torch.full((2,), 32, dtype=torch.int32),
    )
    if output_source == "out":
        kwargs["out"] = torch.empty(q.shape, dtype=output_dtype)
    template = prims_ts_decode_wrapper_trace_dispatch(**kwargs)
    definition = template.build_fi_trace_fn("BatchDecodePagedTSWrapper.run")(**kwargs)
    assert definition["outputs"]["output"]["dtype"] == str(output_dtype).removeprefix(
        "torch."
    )
    assert "head_dim in (64, 128, 256, 512)" in definition["constraints"]
