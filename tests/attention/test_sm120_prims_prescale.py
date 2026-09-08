"""Precision regression tests for the SM120 cute-dsl-prims FP8 prefill P
pre-scale.

The kernel quantizes the softmax matrix P to e4m3 before the PV MMA while
accumulating the softmax denominator ``row_sum`` from the unquantized fp32 P.
Without the SM100-style P*2^8 pre-scale, softmax weights below 2^-10 round to
zero in e4m3: the weight vanishes from the numerator but stays in the
denominator, so the output is silently biased toward the hot tokens.  The bug
is invisible to random-input tests (e4m3's normal-range 3-bit mantissa
dominates there) and to LSE checks (the denominator is exact); it needs
structured inputs - a few hot tokens plus a long, diffuse, value-correlated
tail - to show up.

Both tests below use exactly representable e4m3 logits (q is 64 in dim 0, k
is b_head/b_tail in dim 0) so the softmax distribution is analytic: a
tail-hot plateau with deficit delta = 64*(b_head - b_tail)/sqrt(d) = 7.07
puts every tail weight at exp(-delta) = 8.49e-4 < 2^-10, i.e. exactly on the
flush boundary that the pre-scale removes.
"""

import math
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
from packaging.version import Version

import flashinfer


def _has_required_cutlass_dsl() -> bool:
    try:
        installed_version = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        return False
    return Version(installed_version) >= Version("4.7.0")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not _has_required_cutlass_dsl(),
    reason="requires SM120 and nvidia-cutlass-dsl>=4.7.0",
)

FP8 = torch.float8_e4m3fn
DELTA = 64.0 * (1.375 - 0.125) / math.sqrt(128)  # = 7.071: p_tail = 8.49e-4


def _tail_hot_plateau(n_kv, q_len, hq, hkv, d, all_ones_v, head_count=4, seed=0):
    """Tail-hot plateau with exactly representable e4m3 logits.

    q = 64 in dim 0; k = 1.375 (last ``head_count`` tokens) / 0.125 (rest) in
    dim 0, so scaled logits are exactly 64*b/sqrt(d). Values are either all
    ones or a correlated tail (shared direction per KV head + noise), which
    makes flushed tail mass translate ~1:1 into output error.
    """
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    q = torch.zeros((q_len, hq, d), device="cuda")
    q[..., 0] = 64.0
    k = torch.zeros((n_kv, hkv, d), device="cuda")
    k[..., 0] = 0.125
    k[n_kv - head_count :, :, 0] = 1.375
    if all_ones_v:
        v = torch.ones((n_kv, hkv, d), device="cuda")
    else:
        u = torch.randn((1, hkv, d), generator=g, device="cuda")
        v = u + 0.25 * torch.randn((n_kv, hkv, d), generator=g, device="cuda")
        v[n_kv - head_count :] = torch.randn(
            (head_count, hkv, d), generator=g, device="cuda"
        )
    return q.to(FP8), k.to(FP8), v.to(FP8)


def _run_ragged(q, k, v, hq, hkv, d, out_dtype=torch.float16):
    q_len, n_kv = q.shape[0], k.shape[0]
    qo = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    kvi = torch.tensor([0, n_kv], dtype=torch.int32, device="cuda")
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cute-dsl-prims"
    )
    wrapper.plan(
        qo,
        kvi,
        hq,
        hkv,
        d,
        causal=False,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=out_dtype,
    )
    out, lse = wrapper.run_return_lse(q, k, v)
    return out.float(), lse.float()


def _reference(q, k, v, sm_scale):
    """fp32 softmax attention on the dequantized fp8 inputs."""
    hq, hkv = q.shape[1], k.shape[1]
    qf = q.float().transpose(0, 1)
    kf = k.float().transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
    vf = v.float().transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", qf, kf) * sm_scale
    out = torch.einsum("hqk,hkd->hqd", scores.softmax(-1), vf).transpose(0, 1)
    lse = torch.logsumexp(scores, -1) * math.log2(math.e)
    return out, lse.t()


def test_tail_hot_all_ones_values():
    """With every value equal to 1 the output must be exactly 1.

    Without the P pre-scale the diffuse tail (87% of the softmax mass at
    N=32768) rounds to zero in e4m3 and the output collapses to
    head_count / (head_count + (N - head_count)*exp(-delta)) ~= 0.126
    while the LSE stays exact - the silent form of the bug. With the
    pre-scale the tail survives at ~0.6% quantization error.
    """
    hq, hkv, d, n_kv = 16, 2, 128, 8192
    q, k, v = _tail_hot_plateau(n_kv, 128, hq, hkv, d, all_ones_v=True)
    out, lse = _run_ragged(q, k, v, hq, hkv, d)
    torch.testing.assert_close(out, torch.ones_like(out), atol=2e-2, rtol=2e-2)
    _, lse_ref = _reference(q, k, v, 1.0 / math.sqrt(d))
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


def test_tail_hot_correlated_values():
    """Correlated tail values: flushed mass becomes output error ~1:1.

    Without the pre-scale the RMS error of this config is ~97% of the
    reference magnitude; with it, ~0.6%.
    """
    hq, hkv, d, n_kv = 16, 2, 128, 8192
    q, k, v = _tail_hot_plateau(n_kv, 128, hq, hkv, d, all_ones_v=False)
    out, lse = _run_ragged(q, k, v, hq, hkv, d)
    ref, lse_ref = _reference(q, k, v, 1.0 / math.sqrt(d))
    rel_rms = (out - ref).square().mean().sqrt() / ref.square().mean().sqrt()
    assert rel_rms.item() < 0.03, f"tail-hot relative rms error {rel_rms.item():.4f}"
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
