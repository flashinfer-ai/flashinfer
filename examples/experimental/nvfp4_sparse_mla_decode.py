"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Run the experimental NVFP4 sparse MLA decode kernel (SM100/SM103) on a random
# nvfp4_ds_mla cache and compare it with a PyTorch reference.

import math

import torch

import flashinfer

NUM_ROWS, NUM_TOKENS, TOPK = 64 * 1024, 8, 2048
NUM_HEADS, HEAD_DIM, V_HEAD_DIM, ROW_BYTES = 16, 576, 512, 352
# e2m1 code -> value; the sign is bit 3.
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    device="cuda",
)


def make_cache(num_rows: int) -> torch.Tensor:
    """Random nvfp4_ds_mla rows: 256 B of e2m1 pairs, 64 B of e4m3 RoPE, 32 B of e4m3 block scales."""
    kv = torch.randint(0, 256, (num_rows, ROW_BYTES), dtype=torch.uint8, device="cuda")
    rope_and_scales = torch.rand(num_rows, 96, device="cuda")
    kv[:, 256:] = rope_and_scales.to(torch.float8_e4m3fn).view(torch.uint8)
    return kv


def dequantize(kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """The rows of ``kv`` at ``indices`` as float32 ``[..., 576]``."""
    raw = kv[indices.clamp_min(0).long()]
    nope = torch.stack(
        [E2M1[(raw[..., :256] & 15).long()], E2M1[(raw[..., :256] >> 4).long()]], -1
    ).flatten(-2)
    block = torch.arange(32, device=kv.device)
    scale = raw[..., 320 + 8 * (block % 4) + block // 4].contiguous()
    scale = scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    rope = raw[..., 256:320].contiguous().view(torch.float8_e4m3fn).float()
    return torch.cat([nope * scale, rope], -1)


def main() -> None:
    kv_cache = make_cache(NUM_ROWS)
    query = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, device="cuda") * 0.5
    query = query.to(torch.float8_e4m3fn)
    indices = torch.stack(
        [torch.randperm(NUM_ROWS, device="cuda")[:TOPK] for _ in range(NUM_TOKENS)]
    ).to(torch.int32)
    indices[:, 1500:] = -1  # a short context leaves the tail of the list empty
    sm_scale = 1 / math.sqrt(HEAD_DIM)

    out = flashinfer.mla.nvfp4_sparse_mla_decode(
        query, kv_cache, indices, bmm1_scale=sm_scale
    )

    keys = dequantize(kv_cache, indices)  # [tokens, topk, 576]
    scores = torch.einsum("thd,tkd->thk", query.float(), keys) * sm_scale
    scores = scores.masked_fill((indices < 0)[:, None, :], float("-inf"))
    reference = torch.einsum("thk,tkd->thd", scores.softmax(-1), keys[..., :V_HEAD_DIM])
    error = ((out.float() - reference).abs().max() / reference.abs().max()).item()
    print(
        f"output shape={tuple(out.shape)}, dtype={out.dtype}, "
        f"max error relative to the reference: {error:.2e}"
    )
    assert error < 1e-2


if __name__ == "__main__":
    main()
