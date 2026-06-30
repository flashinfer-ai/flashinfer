#!/usr/bin/env python
"""Minimal repro: trtllm_fp4_block_scale_routed_moe silently misreads float32
topk_weights in the unpacked (ids, weights) routing format.

The unpacked routing format documents `weights` as bfloat16. If a caller passes
float32 weights (a very easy mistake -- torch.topk / softmax routing produce
float32), the wrapper forwards them to the UnpackedPrecomputed kernel with no
dtype check or cast, and the kernel reinterprets the float32 bytes as bf16. The
result is garbage (values ~1e36 / NaN), not an error. The packed int32 format is
unaffected (the value is masked to its low 16 bits).

Expected: either accept float32 (cast) or raise a clear dtype error.
Actual: silent NaN/garbage.

Run on SM100 (B200): python tests/moe/repro_unpacked_fp32_weights.py
"""

import torch
from flashinfer import ActivationType, fp4_quantize
from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
from flashinfer.utils import device_support_pdl, get_compute_capability

GS = 448.0 * 6.0


def q(t, vec=16, swizzled=True):
    a, s = fp4_quantize(t, torch.tensor([GS], device=t.device),
                        sf_vec_size=vec, sf_use_ue8m0=False, is_sf_swizzled_layout=swizzled)
    return a, s.view(torch.float8_e4m3fn)


def main():
    assert get_compute_capability(torch.device("cuda"))[0] == 10, "SM100 only"
    dev = torch.device("cuda:0")
    pdl = device_support_pdl(dev)
    torch.manual_seed(0)
    T, H, I, E, k = 16, 512, 512, 8, 2

    hs, hs_s = q(torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 0.1, swizzled=False)
    hs_s = hs_s.reshape(T, -1)
    w13, w13_s = q(torch.randn(E, I * 2, H, device=dev, dtype=torch.bfloat16) * 0.1)
    w13_s = w13_s.reshape(E, I * 2, -1)
    w2, w2_s = q(torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.1)
    w2_s = w2_s.reshape(E, H, -1)
    sc = torch.full((E,), 1.0 / GS / GS, device=dev)

    scores = torch.softmax(torch.randn(T, E, device=dev), dim=-1)
    weights, ids = torch.topk(scores, k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True))  # float32, as torch.topk yields
    ids = ids.to(torch.int32)

    def run(w):
        return trtllm_fp4_block_scale_routed_moe(
            (ids, w), None, hs, hs_s, w13, w13_s, None, None, None, None,
            w2, w2_s, None, sc, sc, sc, E, k, 0, 0, I, 0, E, None,
            1, True, pdl, ActivationType.Swiglu.value, None,
        )[0].float()

    out_bf16 = run(weights.to(torch.bfloat16))
    out_fp32 = run(weights)  # <-- float32, the bug
    print(f"weights dtype passed: bf16 -> finite={torch.isfinite(out_bf16).all().item()}, "
          f"max|x|={out_bf16.abs().max().item():.3e}")
    print(f"weights dtype passed: fp32 -> finite={torch.isfinite(out_fp32).all().item()}, "
          f"max|x|={out_fp32.abs().max().item():.3e}")
    print(f"bf16-vs-fp32 max abs diff: {(out_bf16 - out_fp32).abs().max().item():.3e}")
    print("BUG reproduced" if not torch.isfinite(out_fp32).all()
          or (out_bf16 - out_fp32).abs().max() > 1e-1 else "no divergence")


if __name__ == "__main__":
    main()
