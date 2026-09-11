# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch
from flashinfer.deepseek_v41 import deepseek_v41_quantize_cache
from .test_deepseek_v41 import gate


def test_mixed_decode_known_constant_sink_and_plan_contract():
    gate()
    pytest.importorskip(
        "flash_mla", reason="explicit FlashMLA decode provider required"
    )
    from flashinfer.deepseek_v41 import deepseek_v41_decode

    q = torch.zeros(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.ones(128, 512, device="cuda", dtype=torch.bfloat16)
    global_kv = torch.full((512, 512), 2.0, device="cuda", dtype=torch.bfloat16)
    swa_cache = deepseek_v41_quantize_cache(swa, format="swa_mxfp8")
    global_cache = deepseek_v41_quantize_cache(global_kv, format="main_kv_fp4")
    swa_ids = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, 128)
    global_ids = torch.arange(512, device="cuda", dtype=torch.int32).view(1, 1, 512)
    sink = torch.zeros(64, device="cuda")
    out, lse, plan = deepseek_v41_decode(
        q, swa_cache, global_cache, swa_ids, global_ids, sink
    )
    # SWA=1 exactly; global amax/6 rounds to E4M3 0.34375 and code6 -> 2.0625.
    expected = torch.full_like(out, (128 + 512 * 2.0625) / 641)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        lse, torch.full_like(lse, 640.0).log(), rtol=1e-6, atol=1e-6
    )
    with pytest.raises(ValueError, match="mismatch"):
        deepseek_v41_decode(
            q.expand(2, 1, 64, 512).contiguous(),
            swa_cache,
            global_cache,
            swa_ids.expand(2, 1, 128).contiguous(),
            global_ids.expand(2, 1, 512).contiguous(),
            sink,
            plan=plan,
        )
    swa_ids.fill_(-1)
    global_ids.fill_(-1)
    empty, empty_lse, _ = deepseek_v41_decode(
        q, swa_cache, global_cache, swa_ids, global_ids, sink, plan=plan
    )
    torch.testing.assert_close(empty, torch.zeros_like(empty), rtol=0, atol=0)
    assert torch.isinf(empty_lse).all()
