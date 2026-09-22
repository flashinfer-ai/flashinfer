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

from flashinfer.prefill import prepare_nvfp4_attention


def main():
    torch.manual_seed(42)
    q = torch.randn((4, 8, 4096, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    output = torch.empty_like(q)
    attention = prepare_nvfp4_attention(q, k, v, output, backend="cake")
    attention()
    torch.cuda.synchronize()
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(output, reference, atol=1.0, rtol=0.1)
    print("NVFP4 attention output:", tuple(output.shape), output.dtype)


if __name__ == "__main__":
    main()
