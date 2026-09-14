# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import torch
from flashinfer import deepseek_v41 as ds

x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
main = ds.deepseek_v41_quantize_cache(x, format="main_kv_fp4")
swa = ds.deepseek_v41_quantize_cache(x, format="swa_mxfp8")
index = ds.deepseek_v41_quantize_index_cache(x[:, :128].contiguous())
print("main/SWA/index page shapes:", main.shape, swa.shape, index.shape)
