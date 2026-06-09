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
"""Correctness test for the full W4A8 MXFP4 MoE wrapper (cutlass_fused_moe-aligned).

Builds per-expert MXFP4 weights, runs ``w4a8_mxfp4_moe`` (permute -> GEMM1+SwiGLU ->
GEMM2 -> finalize), and compares against a torch reference that mirrors the kernel's
intermediate precisions (activation -> FP8, SwiGLU output -> FP8) so the check isolates
kernel correctness from quantization error.
"""

import pytest
import torch
import torch.nn.functional as F

try:
    try:
        from flashinfer.fused_moe import w4a8_mxfp4_moe
    except ImportError:
        # The package-level re-export can lag in editable/dev installs; fall back to the
        # cute_dsl module directly. Non-ImportError failures still propagate and fail.
        from flashinfer.fused_moe.cute_dsl import w4a8_mxfp4_moe
    from flashinfer.utils import is_sm90a_supported

    _OK = True
except ImportError:
    _OK = False

_FP4_LUT = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
_SCALE_BASE = 127


def _make_weight(n, k, dev, seed):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)
    flat = codes.reshape(-1)
    packed = (flat[0::2] | (flat[1::2] << 4)).reshape(n, k // 2).contiguous()
    scale = torch.randint(
        _SCALE_BASE - 1, _SCALE_BASE + 3, (n, k // 32), generator=g, dtype=torch.uint8
    )
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32)
    w_fp32 = lut[codes.long()] * (
        2.0 ** (scale.to(torch.int32) - _SCALE_BASE).float()
    ).repeat_interleave(32, dim=1)
    return packed.to(dev), scale.to(dev), w_fp32.to(dev)


# clamp variants: None = plain silu(gate)*up; else (alpha, beta, limit) = clamped
# SwiGLUBias (GPT-OSS / DeepSeek-V4). The small limit clamps a chunk of the O(1) gate/up.
_CLAMPS = [None, (1.702, 1.0, 0.5)]


@pytest.mark.skipif(
    not _OK
    or not torch.cuda.is_available()
    or not is_sm90a_supported(torch.device("cuda")),
    reason="needs SM90a + cute_dsl + triton",
)
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("clamp", _CLAMPS)
def test_w4a8_mxfp4_moe(top_k, clamp):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    E, T, H, I = 8, 128, 256, 256

    # Per-expert weights: fc1 [2I, H] interleaved gate/up (even=gate, odd=up), fc2 [H, I].
    fc1_p, fc1_s, fc1_w = [], [], []
    fc2_p, fc2_s, fc2_w = [], [], []
    for e in range(E):
        p1, s1, w1 = _make_weight(2 * I, H, dev, seed=e + 1)
        p2, s2, w2 = _make_weight(H, I, dev, seed=1000 + e)
        fc1_p.append(p1)
        fc1_s.append(s1)
        fc1_w.append(w1)
        fc2_p.append(p2)
        fc2_s.append(s2)
        fc2_w.append(w2)
    fc1 = torch.stack(fc1_p)  # [E, 2I, H/2]
    fc1_scale = torch.stack(fc1_s)  # [E, 2I, H/32]
    fc2 = torch.stack(fc2_p)  # [E, H, I/2]
    fc2_scale = torch.stack(fc2_s)  # [E, H, I/32]

    x = (torch.randn(T, H, device=dev) / (H**0.5)).to(torch.bfloat16)
    # Balanced routing (static shapes): token t -> experts (t+s) % E.
    tok = torch.arange(T, device=dev)
    sel = torch.stack([(tok + s) % E for s in range(top_k)], dim=1).to(torch.int)
    rw = (torch.rand(T, top_k, device=dev) + 0.5).float()

    alpha, beta, limit = (None, None, None) if clamp is None else clamp
    swiglu_kw = {}
    if clamp is not None:
        swiglu_kw = dict(
            swiglu_alpha=torch.full((E,), alpha, device=dev),
            swiglu_beta=torch.full((E,), beta, device=dev),
            swiglu_limit=torch.full((E,), limit, device=dev),
        )
    out = w4a8_mxfp4_moe(
        x, sel, rw, fc1, fc2, torch.bfloat16, [fc1_scale, fc2_scale], **swiglu_kw
    ).float()

    # Reference (mirrors kernel intermediate precisions: x->FP8, swiglu out->FP8).
    xf = x.to(torch.float8_e4m3fn).float()
    ref = torch.zeros(T, H, dtype=torch.float32, device=dev)
    for e in range(E):
        gate = xf @ fc1_w[e][0::2].t()  # [T, I]
        up = xf @ fc1_w[e][1::2].t()
        if clamp is None:
            act = F.silu(gate) * up
        else:
            gate_c = gate.clamp(max=limit)
            up_c = up.clamp(-limit, limit) + beta
            act = gate_c * torch.sigmoid(alpha * gate_c) * up_c
        # Mirror the kernel's intermediate path: SwiGLU -> BF16 -> per-token (per-row)
        # FP8 requant with scale -> dequant (the scale is folded into the routing weight).
        act_bf16 = act.to(torch.bfloat16).float()
        sc = act_bf16.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
        a2 = (act_bf16 / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
        down = a2 @ fc2_w[e].t()  # [T, H]
        for s in range(top_k):
            m = sel[:, s] == e
            ref[m] += rw[m, s : s + 1] * down[m]

    denom = ref.abs().clamp_min(0.5)
    frac_bad = ((out - ref).abs() / denom > 0.20).float().mean().item()
    assert frac_bad < 0.03, (
        f"frac_bad {frac_bad:.4f} too high (top_k={top_k} clamp={clamp})"
    )


@pytest.mark.skipif(
    not _OK
    or not torch.cuda.is_available()
    or not is_sm90a_supported(torch.device("cuda")),
    reason="needs SM90a + cute_dsl + triton",
)
@pytest.mark.parametrize("top_k", [1, 2])
def test_w4a8_mxfp4_moe_empty_batch(top_k):
    # Empty batch (no tokens) must not crash: every downstream step (routing,
    # 0-group GEMM, scatter / build_reduce_index) assumes >= 1 active expert.
    dev = torch.device("cuda")
    E, H, I = 8, 256, 256
    fc1 = torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8, device=dev)
    fc1_scale = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=dev)
    fc2 = torch.zeros(E, H, I // 2, dtype=torch.uint8, device=dev)
    fc2_scale = torch.full((E, H, I // 32), 127, dtype=torch.uint8, device=dev)
    x = torch.zeros(0, H, device=dev, dtype=torch.bfloat16)
    sel = torch.zeros(0, top_k, device=dev, dtype=torch.int)
    rw = torch.zeros(0, top_k, device=dev, dtype=torch.float32)
    out = w4a8_mxfp4_moe(x, sel, rw, fc1, fc2, torch.bfloat16, [fc1_scale, fc2_scale])
    assert out.shape == (0, H)


if __name__ == "__main__":
    import sys

    # Dispatch through pytest so the @pytest.mark.skipif (CUDA / SM90a) guards apply
    # instead of being bypassed by direct calls.
    sys.exit(pytest.main([__file__, "-v"]))
