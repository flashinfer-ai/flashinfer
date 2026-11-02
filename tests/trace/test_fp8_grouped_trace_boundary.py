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
"""The ordinary trace must never serialize grouped physical-row semantics."""

import pytest
import torch


def test_grouped_trace_rejects_before_export_and_ordinary_remains_supported(
    tmp_path, monkeypatch
):
    import flashinfer
    from flashinfer.api_logging import flashinfer_api
    from flashinfer.attn_scores.fp8_kernel_config import _FP8MQAKernelConfig
    from flashinfer.trace.templates.attn_scores import fp8_paged_mqa_logits_trace

    inputs = dict(
        q=torch.empty((1, 1, 8, 128), dtype=torch.float8_e4m3fn),
        kv_fused=torch.empty((1, 128, 1, 132), dtype=torch.uint8),
        weights=torch.ones((1, 8)),
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.ones((1,), dtype=torch.int32),
        max_seq_len=128,
    )
    for grouped in (
        dict(query_metadata=torch.zeros((1, 3), dtype=torch.int32)),
        dict(output_rows=1),
        dict(kernel_config=_FP8MQAKernelConfig(128, 1, 160, 16)),
    ):
        with pytest.raises(TypeError, match="grouped scorer options are private"):
            flashinfer.fp8_paged_mqa_logits.fi_trace(
                save_dir=tmp_path / "rejected", **inputs, **grouped
            )
    assert not list((tmp_path / "rejected").rglob("*.json"))

    result = flashinfer.fp8_paged_mqa_logits.fi_trace(
        save_dir=tmp_path / "ordinary", **inputs
    )
    assert result
    assert list((tmp_path / "ordinary").rglob("*.json"))

    # Exercise the actual pre-call auto-dump wrapper without a numerical kernel.
    calls = []

    @flashinfer_api(trace=fp8_paged_mqa_logits_trace)
    def numerical_body(query_metadata=None, output_rows=None, kernel_config=None):
        calls.append(True)
        return "executed"

    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path / "automatic"))
    with pytest.warns(UserWarning, match="fi_trace auto-dump failed"):
        assert numerical_body(query_metadata=object()) == "executed"
    assert calls == [True]
    assert not list((tmp_path / "automatic").rglob("*.json"))


def test_fp8_grouped_options_remain_private():
    import inspect
    import flashinfer.attn_scores as api
    from flashinfer.attn_scores.fp8_kernel_config import _FP8MQAKernelConfig

    assert not hasattr(api, "FP8MQAKernelConfig")
    assert not hasattr(api, "FP8QueryGroups")
    parameters = inspect.signature(api.fp8_paged_mqa_logits).parameters
    assert not {"kernel_config", "query_metadata", "output_rows"} & parameters.keys()
    for tile in (64, 256):
        with pytest.raises(ValueError, match="compute_block_kv must be 128"):
            _FP8MQAKernelConfig(tile, 1, 160)
