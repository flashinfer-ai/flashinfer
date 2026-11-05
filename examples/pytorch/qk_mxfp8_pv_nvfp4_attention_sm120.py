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

import flashinfer


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (12, 0),
        (12, 1),
    ):
        raise RuntimeError("This example requires an SM120 or SM121 GPU")

    batch, num_qo_heads, num_kv_heads = 1, 8, 2
    qo_len, kv_len, head_dim = 193, 317, 128
    q = torch.randn(
        batch,
        num_qo_heads,
        qo_len,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        batch,
        num_kv_heads,
        kv_len,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)

    quantized_qkv = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    out = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized_qkv,
        causal=True,
        unpadded_q_len=qo_len,
        unpadded_k_len=kv_len,
    )

    # The low-level kernel writes the independently padded Q extent.
    out = out[:, :, :qo_len]
    print(out.shape, out.dtype)


if __name__ == "__main__":
    main()
