# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal compute-sanitizer driver covering all dual-BF16 dispatch paths.

Run from the repository root with ``FLASHINFER_JIT_LINEINFO=1`` and one of::

    compute-sanitizer --tool memcheck --kernel-name kns=dual_bf16_weight \
      --error-exitcode 1 python tests/gemm/dual_bf16_weight_gemm_sanitizer.py
    compute-sanitizer --tool racecheck --kernel-name kns=dual_bf16_weight \
      --suppressions tests/gemm/dual_bf16_weight_gemm_racecheck.supp.xml \
      --error-exitcode 1 python tests/gemm/dual_bf16_weight_gemm_sanitizer.py
    compute-sanitizer --tool synccheck --kernel-name kns=dual_bf16_weight \
      --error-exitcode 1 python tests/gemm/dual_bf16_weight_gemm_sanitizer.py
"""

import torch

import flashinfer
from flashinfer.gemm.gemm_dual_bf16_weight import (
    _dual_bf16_weight_gemm_kernel_kind,
)


def main() -> None:
    cases = [
        (128, 65, 256, 0, "split-K 1SM"),
        (257, 64, 128, 1, "persistent 1SM"),
        (257, 65, 128, 2, "cluster 2SM"),
    ]
    for seed, (m, n, k, expected_kind, name) in enumerate(cases):
        torch.manual_seed(seed)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.float32)
        weight_high, weight_low = flashinfer.prepare_dual_bf16_weights(weight)
        workspace_size = flashinfer.dual_bf16_weight_gemm_workspace_size(
            m, n, k, a.device
        )
        workspace = torch.empty(
            max(workspace_size, 1), device=a.device, dtype=torch.uint8
        )
        out = torch.empty(m, n, device=a.device, dtype=torch.float32)
        actual_kind = _dual_bf16_weight_gemm_kernel_kind(m, n, k, a.device)
        if actual_kind != expected_kind:
            raise AssertionError(
                f"{name}: expected kernel kind {expected_kind}, got {actual_kind}"
            )
        flashinfer.mm_bf16_dual_weight(
            a,
            weight_high,
            weight_low,
            out_dtype=torch.float32,
            out=out,
            workspace_buffer=workspace,
        )
        torch.cuda.synchronize()
        print(f"PASS: {name} (M={m}, N={n}, K={k})")


if __name__ == "__main__":
    main()
