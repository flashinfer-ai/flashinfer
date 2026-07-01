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
"""Numerics-hardening tests for the W4A8 MXFP4 path, covering the blind spots found
during DeepSeek-V4 serving bring-up (each of these would have caught a real bug):

1. REALISTIC scale range (UE8M0 bytes ~118-124, i.e. 2^-9..2^-3): real checkpoints
   carry small block scales whose FP4*scale products fall into FP8 e4m3's SUBNORMAL
   band, where the exponent-add dequant mis-encodes/underflows. The fix is the
   ``dequant_exp_bias`` exponent re-centering; with it the dequant must be BIT-EXACT
   (FP4 mantissas are exactly representable in e4m3's normal range). Earlier tests
   used scales 126-130 (the normal-range sweet spot) and missed this entirely.
2. Decode shapes: tiny T with top_k routing over many experts => most expert groups
   EMPTY (M=0) and active groups M~1. Absolute correctness vs a torch reference.
3. Tile completeness (NaN canary): C prefilled with NaN; after the grouped GEMM no
   NaN may survive (every output tile written exactly), up to thousands of clusters.
4. Input outliers: hidden-state outliers beyond the e4m3 max (448) must stay finite
   (torch's bare fp8 cast maps |x|>448 to NaN; real deep-layer hiddens carry such
   outliers). The per-token input scale handles them exactly.
"""

import pytest
import torch
import torch.nn.functional as F

try:
    import cutlass

    from flashinfer.fused_moe import w4a8_mxfp4_grouped_gemm
    from flashinfer.fused_moe.cute_dsl import w4a8_mxfp4_moe
    from flashinfer.utils import is_sm90a_supported

    _OK = True
except ImportError:
    _OK = False

_SKIP = pytest.mark.skipif(
    not _OK
    or not torch.cuda.is_available()
    or not is_sm90a_supported(torch.device("cuda")),
    reason="needs SM90a + cute_dsl + triton",
)

_FP4_LUT = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
# Realistic checkpoint scale range (DSv4: 118-125). Products reach down to
# 0.5 * 2^(118-127) = 2^-10 -- deep in e4m3's subnormal/underflow zone.
_SCALE_LO, _SCALE_HI = 118, 125
# Exponent bias that lifts the whole range into e4m3 normals:
# feasible window is [122 - s_min, 133 - s_max] = [4, 8].
_BIAS = 6


def _make_weight(n, k, dev, seed, lo=_SCALE_LO, hi=_SCALE_HI):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)
    flat = codes.reshape(-1)
    packed = (flat[0::2] | (flat[1::2] << 4)).reshape(n, k // 2).contiguous()
    scale = torch.randint(lo, hi + 1, (n, k // 32), generator=g, dtype=torch.uint8)
    w = _FP4_LUT[codes.long()] * (
        2.0 ** (scale.to(torch.int32) - 127).float()
    ).repeat_interleave(32, dim=1)
    return packed.to(dev), scale.to(dev), w.to(dev)


@_SKIP
def test_dequant_subnormal_scales_bit_exact():
    # One-hot FP8 activations read back the kernel's effective dequantized weights
    # column by column; with the exponent bias the readout must be BIT-EXACT against
    # the reference dequant across the realistic (subnormal-triggering) scale range.
    dev = torch.device("cuda")
    N, K, M = 512, 2048, 128
    packed, scale, wref = _make_weight(N, K, dev, seed=7)
    x = torch.zeros(M, K, device=dev)
    for i in range(M):
        x[i, i] = 1.0
    x8 = x.to(torch.float8_e4m3fn)
    c = torch.full((M, N), float("nan"), device=dev, dtype=torch.float16)
    w4a8_mxfp4_grouped_gemm(
        [x8],
        [packed],
        [(scale + _BIAS)],
        [c],
        [(M, N, K, 1)],
        acc_dtype=cutlass.Float32,
        c_dtype=cutlass.Float16,
        dequant_exp_bias=_BIAS,
    )
    torch.cuda.synchronize()
    got = c.float().cpu()
    ref = wref.cpu()[:, :M].t().contiguous()
    exact = float((got == ref).float().mean())
    assert exact == 1.0, (
        f"dequant not bit-exact at realistic scales (exact-match {exact:.4f}); "
        "FP4*scale products in e4m3's subnormal band are being mis-encoded"
    )


@_SKIP
@pytest.mark.parametrize("T", [1, 4])
def test_small_T_decode_shapes(T):
    # Decode-shape forward: top_k=6 over E=64 experts -> nearly all groups M=0.
    # Absolute correctness vs an own-semantics torch reference at realistic scales.
    dev = torch.device("cuda")
    torch.manual_seed(0)
    E, H, I, top_k = 64, 512, 256, 6
    fc1p, fc1s, fc1w, fc2p, fc2s, fc2w = [], [], [], [], [], []
    for e in range(E):
        p, s, w = _make_weight(2 * I, H, dev, seed=e + 1)
        fc1p.append(p)
        fc1s.append(s)
        fc1w.append(w)
        p, s, w = _make_weight(H, I, dev, seed=1000 + e)
        fc2p.append(p)
        fc2s.append(s)
        fc2w.append(w)
    fc1 = torch.stack(fc1p)
    s1 = torch.stack(fc1s) + _BIAS
    fc2 = torch.stack(fc2p)
    s2 = torch.stack(fc2s) + _BIAS

    g = torch.Generator(device=dev).manual_seed(3)
    x = (torch.randn(T, H, device=dev, generator=g) / (H**0.5)).to(torch.bfloat16)
    sel = torch.stack(
        [torch.randperm(E, device=dev, generator=g)[:top_k] for _ in range(T)]
    ).to(torch.int)
    rw = torch.full((T, top_k), 1.0 / top_k, device=dev)

    out = w4a8_mxfp4_moe(
        x,
        sel,
        rw,
        fc1,
        fc2,
        torch.bfloat16,
        [s1, s2],
        dequant_exp_bias=_BIAS,
    ).float()

    sc_in = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    xf = (x.float() / sc_in).to(torch.float8_e4m3fn).float() * sc_in
    ref = torch.zeros(T, H, dtype=torch.float32, device=dev)
    for e in sorted(set(sel.reshape(-1).tolist())):
        act = F.silu(xf @ fc1w[e][0::2].t()) * (xf @ fc1w[e][1::2].t())
        act = act.to(torch.bfloat16).float()
        sc = act.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
        a2 = (act / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
        down = a2 @ fc2w[e].t()
        for s_ in range(top_k):
            m = sel[:, s_] == e
            if m.any():
                ref[m] += rw[m, s_ : s_ + 1] * down[m]

    cos = F.cosine_similarity(out.reshape(1, -1), ref.reshape(1, -1)).item()
    assert torch.isfinite(out).all()
    assert cos > 0.995, f"decode-shape T={T} cosine {cos:.5f}"


@_SKIP
def test_tile_completeness_nan_canary():
    # Every output tile must be written: prefill C with NaN, run a many-cluster
    # grouped GEMM (E=64 x 16 n-tiles), assert zero surviving NaN + values correct.
    dev = torch.device("cuda")
    E, M, N, K = 64, 6, 4096, 256
    a_list, b_list, s_list, c_list, wref, ps = [], [], [], [], [], []
    for e in range(E):
        a = (torch.randn(M, K, device=dev) / (K**0.5)).to(torch.float8_e4m3fn)
        p, s, w = _make_weight(N, K, dev, seed=e + 1)
        a_list.append(a)
        b_list.append(p)
        s_list.append(s + _BIAS)
        wref.append(w)
        c_list.append(torch.full((M, N), float("nan"), device=dev, dtype=torch.float16))
        ps.append((M, N, K, 1))
    w4a8_mxfp4_grouped_gemm(
        a_list,
        b_list,
        s_list,
        c_list,
        ps,
        acc_dtype=cutlass.Float32,
        c_dtype=cutlass.Float16,
        dequant_exp_bias=_BIAS,
    )
    torch.cuda.synchronize()
    n_nan, n_bad, total = 0, 0, 0
    for e in range(E):
        got = c_list[e].float()
        ref = a_list[e].float() @ wref[e].t()
        n_nan += int(torch.isnan(got).sum())
        rel = torch.nan_to_num((got - ref).abs() / ref.abs().clamp_min(0.05), nan=1e9)
        n_bad += int((rel > 0.25).sum())
        total += got.numel()
    assert n_nan == 0, f"{n_nan} NaN cells survived -> unwritten output tiles"
    assert n_bad / total < 0.005, f"bad fraction {n_bad / total:.4%}"


@_SKIP
def test_input_outlier_saturation():
    # Hidden states with outliers beyond 448 must not produce NaN: the wrapper's
    # per-token input scale maps each row's amax to the e4m3 max (no bare cast).
    dev = torch.device("cuda")
    torch.manual_seed(0)
    E, T, H, I, top_k = 8, 32, 512, 256, 2
    fc1 = torch.stack([_make_weight(2 * I, H, dev, seed=e + 1)[0] for e in range(E)])
    s1 = (
        torch.stack([_make_weight(2 * I, H, dev, seed=e + 1)[1] for e in range(E)])
        + _BIAS
    )
    fc2 = torch.stack([_make_weight(H, I, dev, seed=100 + e)[0] for e in range(E)])
    s2 = (
        torch.stack([_make_weight(H, I, dev, seed=100 + e)[1] for e in range(E)])
        + _BIAS
    )
    x = (torch.randn(T, H, device=dev) / (H**0.5)).to(torch.bfloat16)
    x[0, 0] = 1200.0  # outlier feature beyond the e4m3 max
    x[1, 5] = -3000.0
    tok = torch.arange(T, device=dev)
    sel = torch.stack([(tok + s) % E for s in range(top_k)], 1).to(torch.int)
    rw = torch.full((T, top_k), 1.0 / top_k, device=dev)
    out = w4a8_mxfp4_moe(
        x, sel, rw, fc1, fc2, torch.bfloat16, [s1, s2], dequant_exp_bias=_BIAS
    )
    assert torch.isfinite(out.float()).all(), "outlier input produced non-finite output"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
