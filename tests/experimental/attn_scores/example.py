# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Run selective TopK prefill with continuous keys."""

import torch
from flashinfer import FP8MQATopKWrapper


def main():
    rows, keys, heads = 64, 1024, 64
    q = torch.randn(rows, heads, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.ones(rows, heads, device="cuda")
    scale = torch.ones(keys, device="cuda")
    qo = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    kv = torch.tensor([0, keys], dtype=torch.int32, device="cuda")
    wrapper = FP8MQATopKWrapper(
        num_heads=heads,
        max_context_length=keys,
        max_chunk_size=rows,
        max_batch_size=1,
        device="cuda",
    )
    wrapper.plan(qo, kv, num_query_rows=rows)
    wrapper.prepare_pool(k, k_scale=scale)
    indices = wrapper.run(q, weights, k, k_scale=scale)
    torch.cuda.synchronize()
    assert not wrapper.error.item() and not wrapper.repair_error.item()
    print(indices.shape)


if __name__ == "__main__":
    main()
