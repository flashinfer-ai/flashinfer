# SPDX-License-Identifier: Apache-2.0
import torch

from flashinfer.mla import batch_decode_with_kv_cache_mla_swa as run
from tests.trace.template_registry import collect_registered_trace_templates


def test_mla_swa_trace_registration():
    entries = collect_registered_trace_templates()
    assert any(
        func.__module__ == run.__module__
        and func.__qualname__ == run.__qualname__
        and label == "mla_swa_decode"
        for func, _, label in entries
    )


def test_mla_swa_trace(tmp_path):
    definition = run.fi_trace(
        save_dir=tmp_path,
        query=torch.empty(1, 1, 64, 1088, dtype=torch.bfloat16, device="meta"),
        kv_cache=torch.empty(9, 64, 1088, dtype=torch.bfloat16, device="meta"),
        workspace_buffer=torch.empty(2361600, dtype=torch.uint8, device="meta"),
        block_tables=torch.empty(1, 9, dtype=torch.int32, device="meta"),
        seq_lens=torch.empty(1, dtype=torch.int32, device="meta"),
        window_left=512,
        sm_scale=0.0625,
    )
    assert definition["op_type"] == "mla_swa"
    assert definition["axes"]["head_dim_v"]["value"] == 1024
    assert list(tmp_path.glob("*.json"))
