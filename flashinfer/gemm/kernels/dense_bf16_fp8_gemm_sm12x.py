# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Temporary PyTorch implementation for the dense W8A16 GEMM.

This function defines the kernel integration boundary and canonical tensor
contract. It is a correctness reference, not a production kernel. Replace its
body with an optimized implementation without changing the public API or
weight format.
"""

import torch


def mm_bf16_fp8_sm12x(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """PyTorch W8A16 placeholder; replace this body with the optimized kernel."""
    # TODO(kernel owner): replace this dequantize + matmul reference.
    # B remains the column-major logical [K, N] view used by bmm_fp8.
    weight = B.to(A.dtype)
    result = torch.mm(A.float(), weight.float()) * B_scale
    out.copy_(result.to(out.dtype))
    return out


__all__ = ["mm_bf16_fp8_sm12x"]
