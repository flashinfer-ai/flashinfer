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

from flashinfer.cake_mla_value_gate import prepare_mla_value_gate


def main():
    torch.manual_seed(42)
    heads, rows, max_rows = 12, 8, 64  # one Kimi-K3 TP8 rank, eight decode tokens
    # The absorbed value weight: the [H, 512, 128] view of a contiguous [H, 128, 512] tensor.
    w_uv = (
        (torch.randn(heads, 128, 512, device="cuda") * 0.05)
        .to(torch.bfloat16)
        .transpose(1, 2)
    )
    # Persistent buffers of the largest decode width; a call uses their first rows.
    latent_buf = torch.randn(max_rows, heads, 512, device="cuda").to(torch.bfloat16)
    # The gate logits are the last 1536 columns of the fused [B, 3648] projection output.
    projected = torch.randn(max_rows, 3648, device="cuda").to(torch.bfloat16)
    out_buf = torch.empty(max_rows, heads * 128, dtype=torch.bfloat16, device="cuda")
    # One length per request; a length <= 0 marks a padded request, whose row is
    # written as zero.
    valid_lengths = torch.tensor(
        [17, 5, 0, 9, 33, -1, 1, 2], dtype=torch.int32, device="cuda"
    )

    # Prepare once, outside CUDA graph capture: binds w_uv, decides the program of every B
    # (16-row or 8-row tile; the same bits either way) and loads it.
    runner = prepare_mla_value_gate(w_uv, max_rows=max_rows, sigmoid_precision="fp32")
    latent, gate, out = latent_buf[:rows], projected[:rows, 2112:], out_buf[:rows]

    # The launch allocates nothing and runs on the current stream: capture it and replay it
    # with new contents in the same buffers.
    runner.launch(latent, gate, valid_lengths, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.launch(latent, gate, valid_lengths, out)
    latent.copy_(torch.randn_like(latent, dtype=torch.float32).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()

    # Unfused reference: the BF16 projection times the FP32 sigmoid, padded rows zero.
    value = torch.einsum("bhk,hkv->bhv", latent.float(), w_uv.float())
    value = value.to(torch.bfloat16).float().reshape(rows, heads * 128)
    reference = (value * torch.sigmoid(gate.float())).to(torch.bfloat16)
    reference[valid_lengths <= 0] = 0
    torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
    print(
        "MLA value gate:",
        tuple(out.shape),
        out.dtype,
        runner.arch,
        runner.program_for(rows),
        f"tile_rows={runner.tile_rows_for(rows)}",
    )


if __name__ == "__main__":
    main()
