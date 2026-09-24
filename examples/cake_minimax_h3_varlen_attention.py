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

"""MiniMax-H3 packed-varlen attention (experimental Cake backend, SM100/SM103).

Runs the BF16 kernel and both NVFP4 variants on one packed THD problem and
checks them against a per-segment FP32 reference.
"""

import math

import torch

from flashinfer.prefill import (
    minimax_h3_varlen_attention,
    minimax_h3_varlen_nvfp4_attention,
)


def reference(q, k, v, cu, scale):
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for a, b in zip(cu, cu[1:], strict=False):
        if b <= a:
            continue
        logits = torch.einsum("qhd,khd->hqk", q[a:b].float(), k[a:b].float()) * scale
        out[a:b] = torch.einsum(
            "hqk,khd->qhd", torch.softmax(logits, dim=-1), v[a:b].float()
        )
    return out


def main():
    torch.manual_seed(42)
    cu = [0, 133, 300, 900]  # three segments: 133, 167 and 600 tokens
    heads = 7  # 56 global heads / Ulysses degree 8
    T = cu[-1]
    q = torch.randn((T, heads, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device="cuda")
    expected = reference(q, k, v, cu, 1.0 / math.sqrt(128))

    out = minimax_h3_varlen_attention(q, k, v, cu_seqlens)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)
    print("BF16 attention output:", tuple(out.shape), out.dtype)

    for pv_mode in ("fp8", "fp4"):
        out = minimax_h3_varlen_nvfp4_attention(q, k, v, cu_seqlens, pv_mode=pv_mode)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, atol=1.0, rtol=0.1)
        err = (out.float() - expected).abs().max().item()
        print(
            f"NVFP4 QK + {pv_mode.upper()} PV attention output:",
            tuple(out.shape),
            f"max abs err {err:.4f}",
        )


if __name__ == "__main__":
    main()
