"""Check whether the separate SM120 XQA MLA backend accepts GLM's 20 heads."""

import json
from pathlib import Path

import torch
from flashinfer.mla import xqa_batch_decode_with_kv_cache_mla

workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
results = []
for dtype in (torch.bfloat16, torch.float8_e4m3fn):
    q = torch.randn(1, 1, 20, 576, device="cuda").to(dtype)
    kv = torch.randn(64, 16, 576, device="cuda").to(dtype)
    row = {"dtype": str(dtype), "heads": 20, "backend": "xqa"}
    try:
        xqa_batch_decode_with_kv_cache_mla(
            q,
            kv,
            workspace,
            192,
            512,
            64,
            torch.arange(64, dtype=torch.int32, device="cuda")[None],
            torch.tensor([1024], dtype=torch.int32, device="cuda"),
            1024,
            bmm1_scale=1 / 16,
            bmm2_scale=1.0,
        )
        torch.cuda.synchronize()
        row["unexpected_success"] = True
    except Exception as exc:
        row.update(error_type=type(exc).__name__, error=str(exc))
    results.append(row)
result = json.dumps(results, indent=2) + "\n"
Path(__file__).with_name("xqa_results.json").write_text(result)
print(result)
