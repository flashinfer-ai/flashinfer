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

# Private grouped SM90 FP8 block-scale MoE GEMM vs a pure-torch oracle.
# Layout contract (kernel_src/sm90_push_megamoe/src/fp8_gemm/fp8_moe_binding.cu):
#     a:        (M, K) fp8, rows expert-contiguous per `offsets`
#     b:        (G, N, K) fp8 contiguous
#     offsets:  cumulative int64 of length G + 1
#     scales_a: grouped column-major (K/128, P), P = compute_padded_offset(M, G),
#               group g's rows start at compute_padded_offset(offsets[g], g)
#     scales_b: (G, N/128, K/128) row-major
# Probe cases (zero-token expert, non-128-aligned per-expert M) xfail rather
# than fail: they document capability, not a regression.

import pytest
import torch

pytestmark = pytest.mark.usefixtures("isolated_deep_gemm_cache")

CASES = [
    pytest.param(2, 256, 256, [128, 128], None, id="uniform-aligned"),
    pytest.param(2, 256, 256, [128, 256], None, id="nonuniform-M"),
    pytest.param(
        4, 256, 256, [128, 0, 256, 128], "zero-token expert", id="zero-token-expert"
    ),
    pytest.param(
        2, 256, 256, [132, 124], "per-expert M not 128-aligned", id="unaligned-M"
    ),
    pytest.param(2, 512, 1024, [128, 256], None, id="larger-NK"),
]


def padded_offset(off: int, g: int) -> int:
    # deep_gemm::compute_padded_offset (scheduler.cuh)
    return (off + g * 31) // 32 * 32


def quant_act_grouped(x_bf16: torch.Tensor, offsets: torch.Tensor):
    """1x128 activation quant + scale packing in the grouped layout moe_gemm
    reads. Returns (q (M, K) fp8, sfa (K/128, P) f32 for scales_a,
    sc (M, K/128) f32 dequant scales for the oracle)."""
    M, K = x_bf16.shape
    nkb = K // 128
    G = offsets.numel() - 1
    off = offsets.tolist()
    P = max(padded_offset(off[-1], G), 1)
    xb = x_bf16.float().reshape(M, nkb, 128)
    amax = xb.abs().amax(dim=-1)
    sc = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    q = (xb / sc.unsqueeze(-1)).clamp(-448, 448).reshape(M, K).to(torch.float8_e4m3fn)
    sfa = torch.zeros(nkb, P, dtype=torch.float32, device=x_bf16.device)
    for g in range(G):
        s, t = off[g], off[g + 1]
        if t > s:
            ps = padded_offset(s, g)
            sfa[:, ps : ps + (t - s)] = sc[s:t].T
    return q, sfa.contiguous(), sc


def dequant_weight_128x128(w_fp8: torch.Tensor, w_sf: torch.Tensor) -> torch.Tensor:
    n, k = w_fp8.shape
    return (
        w_fp8.float().reshape(n // 128, 128, k // 128, 128)
        * w_sf.reshape(n // 128, 1, k // 128, 1)
    ).reshape(n, k)


@pytest.fixture
def runner():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip("moe_gemm binding targets SM90 (Hopper)")
    from flashinfer.jit.cpp_ext import is_cuda_version_at_least

    if not is_cuda_version_at_least("12.8"):
        pytest.skip("SM90 push FP8 MoE GEMM requires CUDA Toolkit 12.8+")
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim.gemm import (
        create_sm90_push_fp8_moe_gemm_runner,
    )

    return create_sm90_push_fp8_moe_gemm_runner()


@pytest.mark.parametrize("G,N,K,m_per,probe", CASES)
def test_moe_gemm_grouped_fp8(runner, G, N, K, m_per, probe):
    from flashinfer.testing.utils import per_block_cast_to_fp8

    torch.manual_seed(0)
    M = sum(m_per)
    dev = "cuda"
    offsets = torch.tensor(
        [0, *torch.tensor(m_per).cumsum(0).tolist()], dtype=torch.int64, device=dev
    )

    x = torch.randn(max(M, 1), K, device=dev, dtype=torch.bfloat16)[:M]
    a_fp8, a_sfa, a_sc = quant_act_grouped(x, offsets)

    w_fp8 = torch.empty(G, N, K, device=dev, dtype=torch.float8_e4m3fn)
    w_sf = torch.empty(G, N // 128, K // 128, device=dev, dtype=torch.float32)
    for e in range(G):
        f, s = per_block_cast_to_fp8(
            torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        )
        w_fp8[e].copy_(f)
        w_sf[e].copy_(s)

    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    sz = runner.get_moe_workspace_size(max(M, 1), max(M, 1), N, K, G, True, True)
    runner.configure_workspace(
        torch.empty(max(int(sz), 1), device=dev, dtype=torch.uint8)
    )
    try:
        runner.moe_gemm(out, a_fp8, w_fp8, offsets, N, K, a_sfa, w_sf, False)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        if probe is not None:
            pytest.xfail(f"probe [{probe}] unsupported: {e!r}")
        raise

    nkb = K // 128
    a_dq = (a_fp8.float().reshape(M, nkb, 128) * a_sc.unsqueeze(-1)).reshape(M, K)
    oracle = torch.zeros(M, N, device=dev)
    for e in range(G):
        s0, s1 = offsets[e].item(), offsets[e + 1].item()
        if s1 > s0:
            oracle[s0:s1] = a_dq[s0:s1] @ dequant_weight_128x128(w_fp8[e], w_sf[e]).T

    err_rms = (out.float() - oracle).pow(2).mean().sqrt()
    nrm = oracle.pow(2).mean().sqrt().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), oracle.flatten(), dim=0
    )
    ok = float(err_rms / nrm) < 0.02 and float(cos) > 0.999
    if not ok and probe is not None:
        pytest.xfail(
            f"probe [{probe}] numerically off: err/rms={float(err_rms / nrm):.4f}"
        )
    assert ok, f"err_rms/rms={float(err_rms / nrm):.4f} cos={float(cos):.6f}"


if __name__ == "__main__":
    pytest.main([__file__])
