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

Regression tests for GDN-H4 (issue #4214): mutable staging buffers and
workspaces must not be shared between CUDA streams.

The stress tests are probabilistic on broken code (they need real GPU-side
overlap, which requires the enqueued work to outlast the host launch gap) but
deterministic on fixed code. The buffer-ownership tests are deterministic in
both directions and are the primary regression gate.
"""

from __future__ import annotations

import math
import random

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm100a_supported


def _skip_if_not_sm90_or_later():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("GDN WY decode requires SM90 (Hopper) or later")


def _skip_if_not_sm100():
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100 (Blackwell)")
    cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    if cuda_major < 13:
        pytest.skip(f"SM100 GDN prefill requires CUDA 13+, got {torch.version.cuda}")


try:
    from flashinfer.gdn_kernels import gdn_decode_bf16_wy_output_only as wy

    WY_AVAILABLE = True
except (ImportError, RuntimeError):
    WY_AVAILABLE = False


# =============================================================================
# WY output-only decode: per-stream staging buffers (_STAGE)
# =============================================================================


def _make_wy_inputs(B, T, H, HK, HV, K, V, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, dtype=dtype, device=device)

    return dict(
        q=rnd(B, T, H, K),
        k=rnd(B, T, HK, K),
        v=rnd(B, T, HV, V),
        a=rnd(B, T, HV) * 0.1,
        b=rnd(B, T, HV),
        A_log=rnd(HV, dtype=torch.float32) * 0.1,
        dt_bias=rnd(HV, dtype=torch.float32) * 0.1,
        state=rnd(B, HV, V, K).to(torch.bfloat16),
        indices=torch.arange(B, dtype=torch.int32, device=device),
    )


def _call_wy(x, scale, output=None):
    return wy.gated_delta_rule_mtp(
        A_log=x["A_log"],
        a=x["a"],
        dt_bias=x["dt_bias"],
        q=x["q"],
        k=x["k"],
        v=x["v"],
        b=x["b"],
        initial_state_source=x["state"],
        initial_state_indices=x["indices"],
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
        output=output,
    )


@pytest.mark.parametrize(
    "seq_len, contiguous_ab",
    [
        (2, True),  # full staged path: q/k/v/a/b share one 5-tensor buffer set
        (4, False),  # native-T path with non-contiguous a/b: "ab" staging pair
    ],
)
def test_wy_staging_buffers_are_per_stream(seq_len, contiguous_ab):
    """Same-shape calls on two streams must not share a staging buffer set."""
    _skip_if_not_sm90_or_later()
    if not WY_AVAILABLE:
        pytest.skip("gdn_decode_bf16_wy_output_only kernel not available")

    device = torch.device("cuda")
    B, H, HK, HV, K, V = 2, 16, 16, 32, 128, 128
    scale = 1.0 / math.sqrt(K)
    x = _make_wy_inputs(B, seq_len, H, HK, HV, K, V, 0, device)
    if not contiguous_ab:
        # Non-contiguous a/b defeats the zero-copy native-a/b fast path and
        # forces the "ab" staging-pair branch.
        gen = torch.Generator(device=device).manual_seed(2)
        wide = (
            torch.randn(
                B, seq_len, HV * 2, generator=gen, dtype=torch.bfloat16, device=device
            )
            * 0.1
        )
        x["a"], x["b"] = wide[:, :, ::2], wide[:, :, 1::2]
        assert not x["a"].is_contiguous() and not x["b"].is_contiguous()

    keys_before = set(wy._STAGE.keys())
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(s1):
        _call_wy(x, scale)
    with torch.cuda.stream(s2):
        _call_wy(x, scale)
    torch.cuda.synchronize()

    new_keys = set(wy._STAGE.keys()) - keys_before
    if not new_keys:
        pytest.skip("staging path not taken for this configuration")
    assert len(new_keys) == 2, (
        f"expected one staging buffer set per stream, got keys: {new_keys}"
    )
    bufs = [wy._STAGE[k] for k in new_keys]
    ptrs_a = {t.data_ptr() for t in bufs[0]}
    ptrs_b = {t.data_ptr() for t in bufs[1]}
    assert ptrs_a.isdisjoint(ptrs_b), (
        "staging buffers are shared between streams; concurrent same-shape "
        "calls can overwrite each other's kernel inputs"
    )


def test_wy_multistream_overlap_stress():
    """Interleaved same-shape calls on two streams must match single-stream
    results.

    On the pre-fix code the shared staging buffer let stream B's copy-in land
    while stream A's kernel was still reading, corrupting outputs by ~0.37
    absmax (observed at these sizes on SM120, ~40/100 iterations affected).
    The shape is deliberately large: overlap only happens when enqueued GPU
    work outlasts the host launch gap.
    """
    _skip_if_not_sm90_or_later()
    if not WY_AVAILABLE:
        pytest.skip("gdn_decode_bf16_wy_output_only kernel not available")

    device = torch.device("cuda")
    B, T, H, HK, HV, K, V = 512, 2, 16, 16, 32, 128, 128
    iters = 100
    scale = 1.0 / math.sqrt(K)
    x1 = _make_wy_inputs(B, T, H, HK, HV, K, V, 0, device)
    x2 = _make_wy_inputs(B, T, H, HK, HV, K, V, 1, device)

    ref1 = _call_wy(x1, scale).float().clone()
    torch.cuda.synchronize()
    ref2 = _call_wy(x2, scale).float().clone()
    torch.cuda.synchronize()

    out1 = torch.empty(B, T, HV, V, dtype=torch.bfloat16, device=device)
    out2 = torch.empty_like(out1)
    err1 = torch.zeros((), dtype=torch.float32, device=device)
    err2 = torch.zeros((), dtype=torch.float32, device=device)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    random.seed(0)
    for _ in range(iters):
        with torch.cuda.stream(s1):
            _call_wy(x1, scale, output=out1)
            torch.maximum(err1, (out1.float() - ref1).abs().max(), out=err1)
        with torch.cuda.stream(s2):
            # Random phase jitter sweeps stream2's staging copy-in across
            # stream1's [copy-in .. kernel-read] window over the iterations.
            torch.cuda._sleep(random.randrange(0, 2_000_000))
            _call_wy(x2, scale, output=out2)
            torch.maximum(err2, (out2.float() - ref2).abs().max(), out=err2)
    torch.cuda.synchronize()

    assert err1.item() < 1e-3, (
        f"stream1 output diverged from single-stream result by {err1.item()}"
    )
    assert err2.item() < 1e-3, (
        f"stream2 output diverged from single-stream result by {err2.item()}"
    )


# =============================================================================
# Blackwell (SM100) chunked prefill: workspace must not live in the compile
# cache
# =============================================================================


def _make_prefill_inputs(seq_lens, HQ, HV, DK, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    total = sum(seq_lens)
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, dtype=dtype, device=device)

    def unit_rows(*shape):
        # Un-normalized rows (norm ~= sqrt(DK)) make the delta-rule triangular
        # inverse (I + tril(diag(beta) K K^T))^-1 overflow over a 512-token sequence.
        return (
            F.normalize(rnd(*shape, dtype=torch.float32), dim=-1)
            .to(torch.bfloat16)
            .contiguous()
        )

    return dict(
        q=unit_rows(total, HQ, DK),
        k=unit_rows(total, HQ, DK),
        v=rnd(total, HV, DK),
        # FlashInfer consumes linear-space alpha = exp(log_g) in (0, 1]; the
        # kernel takes log2 itself, so any non-positive entry becomes NaN.
        gate=torch.exp(
            -F.softplus(rnd(total, HV, dtype=torch.float32) * 0.5 - 2.0)
        ).contiguous(),
        beta=torch.sigmoid(rnd(total, HV, dtype=torch.float32)),
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device),
        initial_state=rnd(len(seq_lens), HV, DK, DK, dtype=torch.float32) * 0.01,
    )


def _call_prefill_sm100(x, HV, DK, device):
    from flashinfer.gdn_kernels.blackwell.gdn_prefill import (
        chunk_gated_delta_rule_sm100,
    )

    total = x["q"].shape[0]
    output = torch.empty(total, HV, DK, dtype=torch.bfloat16, device=device)
    output_state = torch.empty_like(x["initial_state"])
    chunk_gated_delta_rule_sm100(
        q=x["q"],
        k=x["k"],
        v=x["v"],
        gate=x["gate"],
        beta=x["beta"],
        output=output,
        cu_seqlens=x["cu_seqlens"],
        initial_state=x["initial_state"],
        output_state=output_state,
        scale=1.0 / math.sqrt(DK),
    )
    return output, output_state


def test_blackwell_prefill_workspace_not_in_compile_cache():
    """The per-specialization compile cache must hold only the compiled
    callable and static metadata; the launch workspace (which the kernel
    rewrites with TMA descriptors every call) must not be stored there."""
    _skip_if_not_sm100()
    from flashinfer.gdn_kernels.blackwell import gdn_prefill as prefill_mod

    device = torch.device("cuda")
    HQ = HV = 4
    DK = 128
    x1 = _make_prefill_inputs([512, 384], HQ, HV, DK, 0, device)
    x2 = _make_prefill_inputs([512, 384], HQ, HV, DK, 1, device)

    ref1, ref_state1 = _call_prefill_sm100(x1, HV, DK, device)
    torch.cuda.synchronize()
    ref2, ref_state2 = _call_prefill_sm100(x2, HV, DK, device)
    torch.cuda.synchronize()

    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    outs = []
    for _ in range(20):
        with torch.cuda.stream(s1):
            outs.append(_call_prefill_sm100(x1, HV, DK, device))
        with torch.cuda.stream(s2):
            outs.append(_call_prefill_sm100(x2, HV, DK, device))
    torch.cuda.synchronize()

    for i, (out, out_state) in enumerate(outs):
        ref, ref_state = (ref1, ref_state1) if i % 2 == 0 else (ref2, ref_state2)
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(out_state, ref_state, atol=1e-3, rtol=1e-3)

    # White-box: no workspace tensors may live in the compile-cache value.
    cache = prefill_mod._get_compiled_cache(
        str(x1["q"].dtype),
        str(x1["initial_state"].dtype),
        HQ,
        HV,
        HQ >= HV,
        True,
        True,
        False,
        False,
        str(x1["cu_seqlens"].dtype),
        "none",
        "none",
        None,
        None,
    )
    assert set(cache.keys()) <= {"compiled", "num_sm"}, (
        f"compile cache holds mutable execution state: {set(cache.keys())}"
    )
