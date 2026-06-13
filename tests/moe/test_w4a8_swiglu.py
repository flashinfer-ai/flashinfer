# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness test: fused SwiGLU epilogue in the W4A8 MXFP4 GEMM1 (`swiglu=True`).

The GEMM1 weight columns are interleaved gate/up (row 2j = gate_j, row 2j+1 = up_j), so
the GEMM produces C[:, 2j] = gate_proj_j and C[:, 2j+1] = up_proj_j with each (gate, up)
pair in one thread's adjacent WGMMA registers. The fused epilogue then computes
silu(gate)*up and writes FP8 [M, I] directly -- no [M, 2I] FP16 round-trip, no separate
activation/requant kernel. Validated against torch silu(A@gate^T) * (A@up^T).
"""

import pytest
import torch
import torch.nn.functional as F

try:
    import cutlass

    from flashinfer.fused_moe import w4a8_mxfp4_grouped_gemm
    from flashinfer.utils import is_sm90a_supported

    _OK = True
except ImportError:
    _OK = False

_FP4_LUT = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
_SCALE_BASE = 127


def _make_weight(n, k, dev, seed):
    """Return (packed_u8 [n, k//2], scale_u8 [n, k//32], w_fp32 [n, k]) — interleaved
    gate/up rows (even = gate, odd = up). UE8M0 scale near the baseline so the FP8
    dequant is exact and the reference matches tightly."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)
    flat = codes.reshape(-1)
    packed = (flat[0::2] | (flat[1::2] << 4)).reshape(n, k // 2).contiguous()
    scale = torch.randint(
        _SCALE_BASE - 1, _SCALE_BASE + 3, (n, k // 32), generator=g, dtype=torch.uint8
    )
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32)
    vals = lut[codes.long()]
    se = (scale.to(torch.int32) - _SCALE_BASE).float()
    w_fp32 = vals * (2.0**se).repeat_interleave(32, dim=1)
    return packed.to(dev), scale.to(dev), w_fp32.to(dev)


# clamp variants: None = plain silu(gate)*up; else (alpha, beta, limit) = clamped
# SwiGLUBias. The small limit (0.5) clamps a meaningful fraction of the O(1) gate/up.
_CLAMPS = [None, (1.702, 1.0, 0.5)]


@pytest.mark.skipif(
    not _OK
    or not torch.cuda.is_available()
    or not is_sm90a_supported(torch.device("cuda")),
    reason="needs SM90a + cute_dsl",
)
@pytest.mark.parametrize("M,K,I", [(128, 256, 256), (256, 512, 512), (512, 1024, 1024)])
@pytest.mark.parametrize("clamp", _CLAMPS)
def test_w4a8_swiglu(M, K, I, clamp):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    N = 2 * I  # GEMM1 N = gate|up interleaved
    x_fp8 = (torch.randn(M, K, device=dev) / (K**0.5)).to(torch.float8_e4m3fn)
    packed, scale, w_fp32 = _make_weight(N, K, dev, seed=1)
    c_out = torch.empty(M, I, device=dev, dtype=torch.float8_e4m3fn)

    alpha, beta, limit = (None, None, None) if clamp is None else clamp
    w4a8_mxfp4_grouped_gemm(
        [x_fp8],
        [packed],
        [scale],
        [c_out],
        [(M, N, K, 1)],
        acc_dtype=cutlass.Float32,
        c_dtype=cutlass.Float8E4M3FN,
        swiglu=True,
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
    )
    torch.cuda.synchronize()

    # Reference: gate/up = even/odd weight rows. Plain: silu(gate)*up. Clamped (SwiGLUBias):
    # gate clamped to (-inf, L], up to [-L, L] then + beta, sigmoid arg scaled by alpha.
    a = x_fp8.float()
    gate_w, up_w = w_fp32[0::2], w_fp32[1::2]  # [I, K] each
    gate = a @ gate_w.t()  # [M, I]
    up = a @ up_w.t()
    if clamp is None:
        ref = F.silu(gate) * up
    else:
        gate_c = gate.clamp(max=limit)
        up_c = up.clamp(-limit, limit) + beta
        ref = gate_c * torch.sigmoid(alpha * gate_c) * up_c
        # the test is only meaningful if the limit actually clamps a chunk of values
        assert ((gate > limit) | (up.abs() > limit)).float().mean().item() > 0.1

    denom = ref.abs().clamp_min(0.5)  # FP8 e4m3 ~2^-3 rel precision
    frac_bad = ((c_out.float() - ref).abs() / denom > 0.18).float().mean().item()
    assert frac_bad < 0.02, f"frac_bad {frac_bad:.4f} too high (M={M} K={K} I={I})"


if __name__ == "__main__":
    for clamp in _CLAMPS:
        for shape in [(128, 256, 256), (256, 512, 512), (512, 1024, 1024)]:
            test_w4a8_swiglu(*shape, clamp)
            print(f"PASS {shape} clamp={clamp}")
    print("ALL PASS")
