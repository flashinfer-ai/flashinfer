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

import torch

from flashinfer.prefill import fp8_attention_workspace_bytes, prepare_fp8_attention


def main():
    torch.manual_seed(42)
    batch, heads, seqlen = 1, 40, 4800
    # An [B, S, H, 128] activation buffer, passed as its logical [B, H, S, 128] view and read in place.
    storage = torch.randn(
        (batch, seqlen, heads, 128), dtype=torch.bfloat16, device="cuda"
    )
    q = storage.transpose(1, 2)
    k, v = (
        torch.randn_like(storage).transpose(1, 2),
        torch.randn_like(storage).transpose(1, 2),
    )
    output = torch.empty_like(storage).transpose(1, 2)
    # One workspace per problem size; a caller may pool one of the largest size.
    workspace = torch.empty(
        fp8_attention_workspace_bytes(batch, heads, seqlen, q.device),
        dtype=torch.uint8,
        device="cuda",
    )
    attention = prepare_fp8_attention(
        q, k, v, output, backend="cake", workspace=workspace
    )
    attention()  # statistics, quantization and attention; values are re-read on every launch
    torch.cuda.synchronize()
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(output, reference, atol=1.0, rtol=0.1)
    print(
        "FP8 attention output:",
        tuple(output.shape),
        output.dtype,
        attention.route_metadata["arch"],
    )


if __name__ == "__main__":
    main()
