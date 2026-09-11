# Copyright (c) 2026 by FlashInfer team.
# Tests for the output-only (frozen-state) KDA decode kernel.

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm100a_supported

# Public module under test: must always import — flashinfer.kda_decode is
# designed to import without the CuTe DSL (its internal imports are guarded
# and it exposes the availability flag), so any ImportError here is a real
# regression and should fail collection rather than skip.
from flashinfer.kda_decode import _KDA_OUTPUT_ONLY_AVAILABLE

if _KDA_OUTPUT_ONLY_AVAILABLE:
    # Internal dispatcher (module path, not public API): same signature the
    # public op had; used here to force individual backends. The public
    # surface is recurrent_kda(disable_state_update=True), tested below.
    # Gated on the availability flag instead of try/except so that an import
    # failure when the flag says "available" is also a loud failure.
    from flashinfer.kda_kernels.kda_decode_wy_output_only import (
        kda_wy_output_only as kda_output_only_decode,
    )
else:
    kda_output_only_decode = None


@pytest.fixture(autouse=True)
def _require_kda_output_only():
    """Skip unless CUDA + SM100a + the CuTe-DSL kernel are available."""
    if not torch.cuda.is_available():
        pytest.skip("KDA output-only decode requires CUDA")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("KDA output-only decode requires SM100a (Blackwell)")
    if not _KDA_OUTPUT_ONLY_AVAILABLE:
        pytest.skip("kda_output_only_decode unavailable (missing cutlass DSL deps)")


def naive_output_only_reference(q, k, v, g, beta, h0_pool, idx, scale):
    """fp32 recurrent reference matching the kernel contract.

    q/k [B,T,H,K] (caller pre-applies L2 norm), v [B,T,HV,V],
    g [B,T,HV,K] log-space fp32, beta [B,T,HV] fp32 (post-sigmoid),
    h0_pool [pool,HV,V,K] bf16, idx [B] int32. Returns o [B,T,HV,V] fp32.
    """
    B, T, H, _K = q.shape
    HV = v.shape[2]
    rep = HV // H
    q = q.float() * scale
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()
    o = torch.zeros(B, T, HV, v.shape[3], dtype=torch.float32, device=q.device)
    for b in range(B):
        S = h0_pool[idx[b]].float().clone()  # [HV, V, K]
        for t in range(T):
            a = g[b, t].exp()
            S = S * a[:, None, :]
            k_hv = k[b, t].repeat_interleave(rep, dim=0)
            u = torch.einsum("hvk,hk->hv", S, k_hv)
            w = beta[b, t][:, None] * (v[b, t] - u)
            S = S + torch.einsum("hv,hk->hvk", w, k_hv)
            q_hv = q[b, t].repeat_interleave(rep, dim=0)
            o[b, t] = torch.einsum("hvk,hk->hv", S, q_hv)
    return o


def l2n(x):
    """L2-normalize the last dim (same dtype in/out)."""
    return F.normalize(x.float(), p=2, dim=-1).to(x.dtype)


def _make_inputs(B, T, H, HV, seed=0):
    """Random bf16 q/k/v/gate/beta, a bf16 state pool, and gate params."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    K = V = 128
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    graw = torch.randn(B, T, HV, K, dtype=torch.bfloat16, device=dev)
    beta = (
        torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev)
        .sigmoid()
        .to(torch.bfloat16)
    )
    pool = B + 2
    h0 = torch.randn(pool, HV, V, K, dtype=torch.bfloat16, device=dev) * 0.1
    idx = torch.randperm(pool, device=dev)[:B].to(torch.int32)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.3
    dt_bias = torch.randn(H * K, dtype=torch.float32, device=dev) * 0.1
    return q, k, v, graw, beta, h0, idx, A_log, dt_bias


@pytest.mark.parametrize("backend", ["wy", "recurrent", "auto"])
@pytest.mark.parametrize(
    "B,T,H,HV",
    [
        (1, 1, 2, 2),
        (2, 4, 4, 4),
        (3, 8, 4, 4),
        (2, 5, 2, 4),  # non-power-of-two T + GQA
        (2, 16, 12, 12),  # Kimi K3 heads
        (17, 16, 4, 4),  # odd batch
    ],
)
def test_output_only_precomputed_gate(B, T, H, HV, backend):
    """Precomputed log-space gate matches the fp32 recurrent reference."""
    q, k, v, _, beta, h0, idx, _, _ = _make_inputs(B, T, H, HV)
    g_log = F.logsigmoid(
        torch.randn(B, T, HV, 128, dtype=torch.float32, device=q.device)
    ).to(torch.bfloat16)
    scale = 128**-0.5
    out = kda_output_only_decode(
        q, k, v, g_log, beta, h0, idx, scale=scale, backend=backend
    )
    ref = naive_output_only_reference(
        l2n(q), l2n(k), v, g_log.float(), beta.float(), h0, idx, scale
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("backend", ["wy", "recurrent"])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("beta_is_logit", [False, True])
def test_output_only_in_kernel_gate(backend, lower_bound, beta_is_logit):
    """In-kernel lower-bound (Kimi K3) and softplus gates match reference."""
    B, T, H, HV = 3, 8, 4, 4
    q, k, v, graw, beta, h0, idx, A_log, dt_bias = _make_inputs(B, T, H, HV, seed=1)
    beta_in = (
        torch.randn(B, T, HV, dtype=torch.bfloat16, device=q.device)
        if beta_is_logit
        else beta
    )
    scale = 128**-0.5
    out = kda_output_only_decode(
        q,
        k,
        v,
        graw,
        beta_in,
        h0,
        idx,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        beta_is_logit=beta_is_logit,
        backend=backend,
    )
    x = (
        graw.float()
        + dt_bias.view(H, 128).repeat_interleave(HV // H, dim=0)[None, None]
    )
    a_hv = A_log.exp().repeat_interleave(HV // H)[None, None, :, None]
    if lower_bound is not None:
        g_ref = lower_bound * torch.sigmoid(a_hv * x)
    else:
        g_ref = -a_hv * F.softplus(x)
    beta_ref = beta_in.float().sigmoid() if beta_is_logit else beta_in.float()
    ref = naive_output_only_reference(
        l2n(q), l2n(k), v, g_ref, beta_ref, h0, idx, scale
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1e-2)


def test_output_only_state_not_written():
    """The committed-state pool must be bit-identical after the call."""
    B, T, H, HV = 4, 8, 4, 4
    q, k, v, _, beta, h0, idx, _, _ = _make_inputs(B, T, H, HV, seed=2)
    g_log = F.logsigmoid(
        torch.randn(B, T, HV, 128, dtype=torch.float32, device=q.device)
    ).to(torch.bfloat16)
    h0_before = h0.clone()
    for backend in ["wy", "recurrent"]:
        kda_output_only_decode(q, k, v, g_log, beta, h0, idx, backend=backend)
        torch.cuda.synchronize()
        assert torch.equal(h0, h0_before), f"{backend}: state pool was modified"


def test_output_only_matches_recurrent_kda():
    """Outputs match flashinfer.recurrent_kda run token-by-token."""
    from flashinfer.kda_decode import recurrent_kda

    B, T, H, HV = 2, 8, 4, 4
    q, k, v, _, beta, h0, idx, _, _ = _make_inputs(B, T, H, HV, seed=3)
    g_log = F.logsigmoid(
        torch.randn(B, T, HV, 128, dtype=torch.float32, device=q.device)
    ).to(torch.bfloat16)
    scale = 128**-0.5

    out = kda_output_only_decode(q, k, v, g_log, beta, h0, idx, scale=scale)

    # Reference: recurrent_kda one token at a time from the same initial state
    state = h0[idx.long()].clone()
    outs = []
    for t in range(T):
        o_t, _ = recurrent_kda(
            q[:, t : t + 1].contiguous(),
            k[:, t : t + 1].contiguous(),
            v[:, t : t + 1].contiguous(),
            g_log[:, t : t + 1].contiguous(),
            beta[:, t : t + 1].contiguous(),
            scale=scale,
            initial_state=state,
            use_qk_l2norm_in_kernel=True,
        )
        outs.append(o_t)
    ref = torch.cat(outs, dim=1)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=1e-2)


def test_output_only_preallocated_output():
    """A caller-provided output tensor is written in place and returned."""
    B, T, H, HV = 2, 8, 4, 4
    q, k, v, _, beta, h0, idx, _, _ = _make_inputs(B, T, H, HV, seed=4)
    g_log = F.logsigmoid(
        torch.randn(B, T, HV, 128, dtype=torch.float32, device=q.device)
    ).to(torch.bfloat16)
    buf = torch.empty(B, T, HV, 128, dtype=torch.bfloat16, device=q.device)
    out = kda_output_only_decode(q, k, v, g_log, beta, h0, idx, output=buf)
    assert out.data_ptr() == buf.data_ptr()
    ref = kda_output_only_decode(q, k, v, g_log, beta, h0, idx)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.0, rtol=0.0)


def ref_corrections(q, k, v, g_log, beta, h0, idx, scale):
    """fp32 reference also returning per-token corrections U_t = beta*(v-u)."""
    B, T, H, _K = q.shape
    HV, V = v.shape[2], v.shape[3]
    rep = HV // H
    o = torch.zeros(B, T, HV, V, dtype=torch.float32, device=q.device)
    U = torch.zeros_like(o)
    for b in range(B):
        S = h0[idx[b]].float().clone()
        for t in range(T):
            S = S * g_log[b, t].exp()[:, None, :]
            k_hv = k[b, t].repeat_interleave(rep, dim=0)
            pred = torch.einsum("hvk,hk->hv", S, k_hv)
            w = beta[b, t][:, None] * (v[b, t].float() - pred)
            U[b, t] = w
            S = S + torch.einsum("hv,hk->hvk", w, k_hv)
            q_hv = (q[b, t] * scale).repeat_interleave(rep, dim=0)
            o[b, t] = torch.einsum("hvk,hk->hv", S, q_hv)
    return o, U


@pytest.mark.parametrize("backend", ["wy", "recurrent"])
@pytest.mark.parametrize("B,T,H,HV", [(2, 8, 4, 4), (3, 16, 2, 4), (2, 5, 4, 4)])
def test_emit_corrections(B, T, H, HV, backend):
    """emit_corrections returns U_t = beta*(v-u) and the kg cache (k_norm|g)."""
    q, k, v, _, beta, h0, idx, _, _ = _make_inputs(B, T, H, HV, seed=5)
    g_log = F.logsigmoid(
        torch.randn(B, T, HV, 128, dtype=torch.float32, device=q.device)
    ).to(torch.bfloat16)
    scale = 128**-0.5
    out, corr, kg = kda_output_only_decode(
        q,
        k,
        v,
        g_log,
        beta,
        h0,
        idx,
        scale=scale,
        emit_corrections=True,
        backend=backend,
    )
    o_ref, U_ref = ref_corrections(
        l2n(q).float(),
        l2n(k).float(),
        v,
        g_log.float(),
        beta.float(),
        h0,
        idx,
        scale,
    )
    torch.testing.assert_close(out.float(), o_ref, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(corr.float(), U_ref, atol=3e-2, rtol=2e-2)
    kn_ref = l2n(k).float().repeat_interleave(HV // H, dim=2)
    torch.testing.assert_close(kg[..., :128].float(), kn_ref, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(kg[..., 128:].float(), g_log.float())
    # emit mode must not change the primary output vs non-emit
    out_plain = kda_output_only_decode(
        q, k, v, g_log, beta, h0, idx, scale=scale, backend=backend
    )
    torch.testing.assert_close(out.float(), out_plain.float(), atol=1e-3, rtol=1e-3)


class _RecoverSSMRef:
    """fp32 reference for the vLLM RecoverSSM verify contract."""

    @staticmethod
    def run(q, k, v, g, beta, A_log, dt_bias, lower_bound, state, qsl, sidx, T):
        H, K = q.shape[2], q.shape[3]
        V = v.shape[3]
        B = sidx.shape[0]
        nblk = state.shape[0]
        out = torch.zeros_like(v, dtype=torch.float32)
        corr = torch.full(
            (nblk, H, T, V), float("nan"), dtype=torch.float32, device=q.device
        )
        kg = torch.full(
            (nblk, H, T, 2 * K), float("nan"), dtype=torch.float32, device=q.device
        )
        A = A_log.float().exp()
        dtb = dt_bias.float().view(H, K)
        for b in range(B):
            bos, eos = int(qsl[b]), int(qsl[b + 1])
            slot = int(sidx[b])
            if slot <= 0:
                out[0, bos:eos] = 0.0
                continue
            S = state[slot].float().clone()  # [H, V, K]
            for t in range(eos - bos):
                tok = bos + t
                kk = k[0, tok].float()
                kn = kk / (kk.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
                x = g[0, tok].float() + dtb
                if lower_bound is not None:
                    gate = lower_bound * torch.sigmoid(A[:, None] * x)
                else:
                    sp = torch.where(x > 20.0, x, torch.log1p(x.exp()))
                    gate = -A[:, None] * sp
                S = S * gate.exp()[:, None, :]
                pred = torch.einsum("hvk,hk->hv", S, kn)
                c = torch.sigmoid(beta[0, tok].float())[:, None] * (
                    v[0, tok].float() - pred
                )
                S = S + torch.einsum("hv,hk->hvk", c, kn)
                qq = q[0, tok].float()
                qn = qq / (qq.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt() * K**-0.5
                out[0, tok] = torch.einsum("hvk,hk->hv", S, qn)
                corr[slot, :, t] = c
                kg[slot, :, t, :K] = kk
                kg[slot, :, t, K:] = g[0, tok].float()
        return out, corr, kg


@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("ragged,with_null", [(False, False), (True, True)])
@pytest.mark.parametrize("padded_checkpoint", [False, True])
def test_recoverssm_dropin(lower_bound, ragged, with_null, padded_checkpoint):
    """kda_recoverssm_verify matches the vLLM RecoverSSM verify contract."""
    from flashinfer.kda_kernels.kda_decode_wy_output_only import (
        kda_recoverssm_verify,
    )

    torch.manual_seed(0)
    dev = "cuda"
    B, T, H = 5, 8, 4
    K = V = 128
    lens = torch.randint(1, T + 1, (B,)) if ragged else torch.full((B,), T)
    qsl = torch.zeros(B + 1, dtype=torch.int32)
    qsl[1:] = lens.cumsum(0)
    tot = int(qsl[-1])
    qsl = qsl.to(dev)
    q = torch.randn(1, tot, H, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn_like(q)
    v = torch.randn(1, tot, H, V, dtype=torch.bfloat16, device=dev)
    g = torch.randn(1, tot, H, K, dtype=torch.bfloat16, device=dev)
    beta = torch.randn(1, tot, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.3
    dt = torch.randn(H * K, dtype=torch.float32, device=dev) * 0.1
    nblk = B + 2
    if padded_checkpoint:
        state_storage = torch.randn(
            nblk,
            H * V * K + 128,
            dtype=torch.bfloat16,
            device=dev,
        )
        state_storage.mul_(0.1)
        state = state_storage[:, : H * V * K].view(nblk, H, V, K)
    else:
        state = torch.randn(nblk, H, V, K, dtype=torch.bfloat16, device=dev) * 0.1
    sidx = torch.arange(1, B + 1, dtype=torch.int32, device=dev)
    if with_null:
        sidx = sidx.clone()
        sidx[0] = 0
        sidx[2] = -3
    # Sentinel-filled caches: entries outside the written region (rows past
    # each sequence's query_len, unused slots, null slots) must survive
    # bit-identically — the vLLM contract leaves them untouched.
    corr = torch.full((nblk, H, T, V), 7.0, dtype=torch.float32, device=dev)
    kg = torch.full((nblk, H, T, 2 * K), 7.0, dtype=torch.bfloat16, device=dev)
    state_before = state.clone()

    out = kda_recoverssm_verify(
        q, k, v, g, beta, A_log, dt, lower_bound, state, corr, kg, qsl, sidx, T
    )
    torch.cuda.synchronize()
    assert torch.equal(state, state_before), "checkpoint pool was modified"

    o_ref, c_ref, kg_ref = _RecoverSSMRef.run(
        q, k, v, g, beta, A_log, dt, lower_bound, state, qsl, sidx, T
    )
    written = ~torch.isnan(c_ref)
    torch.testing.assert_close(out.float(), o_ref, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(corr[written], c_ref[written], atol=3e-2, rtol=2e-2)
    kgw = ~torch.isnan(kg_ref)
    torch.testing.assert_close(kg.float()[kgw], kg_ref[kgw], atol=1e-6, rtol=0.0)
    # untouched-region immutability (null slots, unused slots, ragged tails)
    assert torch.all(corr[~written] == 7.0), "corrections cache written OOB"
    assert torch.all(kg[~kgw] == 7.0), "kg cache written outside valid tokens"


def test_recoverssm_dropin_padding_is_runtime():
    """One compiled kernel must serve pools with different block paddings.

    The pool block stride is a runtime kernel argument (not baked into the
    compiled TMA descriptor or the compile-cache key); calling with
    unpadded, +128- and +256-element-padded pools in one process must give
    identical results for identical block contents.
    """
    from flashinfer.kda_kernels.kda_decode_wy_output_only import (
        kda_recoverssm_verify,
    )

    torch.manual_seed(11)
    dev = "cuda"
    B, T, H = 3, 8, 4
    K = V = 128
    qsl = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
    tot = B * T
    q = torch.randn(1, tot, H, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn_like(q)
    v = torch.randn(1, tot, H, V, dtype=torch.bfloat16, device=dev)
    g = torch.randn(1, tot, H, K, dtype=torch.bfloat16, device=dev)
    beta = torch.randn(1, tot, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.3
    dt = torch.randn(H * K, dtype=torch.float32, device=dev) * 0.1
    nblk = B + 2
    dense = torch.randn(nblk, H, V, K, dtype=torch.bfloat16, device=dev) * 0.1
    sidx = torch.arange(1, B + 1, dtype=torch.int32, device=dev)

    outs = []
    for pad in [0, 128, 256]:
        if pad:
            storage = torch.full(
                (nblk, H * V * K + pad), 9.0, dtype=torch.bfloat16, device=dev
            )
            state = storage[:, : H * V * K].view(nblk, H, V, K)
            state.copy_(dense)
        else:
            state = dense.clone()
        corr = torch.zeros(nblk, H, T, V, dtype=torch.float32, device=dev)
        kg = torch.zeros(nblk, H, T, 2 * K, dtype=torch.bfloat16, device=dev)
        out = kda_recoverssm_verify(
            q, k, v, g, beta, A_log, dt, -5.0, state, corr, kg, qsl, sidx, T
        )
        torch.cuda.synchronize()
        outs.append((out.clone(), corr.clone(), kg.clone()))

    for pad, (o, c, kgc) in zip([128, 256], outs[1:], strict=True):
        assert torch.equal(o, outs[0][0]), f"pad={pad}: outputs differ"
        assert torch.equal(c, outs[0][1]), f"pad={pad}: corrections differ"
        assert torch.equal(kgc, outs[0][2]), f"pad={pad}: kg differs"


# =============================================================================
# Public API: recurrent_kda(disable_state_update=True)
# =============================================================================


def test_frozen_mode_public_api_matches_internal():
    """recurrent_kda's frozen mode returns (out, None) and matches the
    internal dispatcher on identical inputs; the state pool is untouched."""
    from flashinfer import recurrent_kda

    B, T, H, HV = 3, 8, 4, 4
    q, k, v, _, beta, h0, idx, _, _ = _make_inputs(B, T, H, HV, seed=6)
    g_log = F.logsigmoid(
        torch.randn(B, T, HV, 128, dtype=torch.float32, device=q.device)
    ).to(torch.bfloat16)
    h0_before = h0.clone()
    out, fs = recurrent_kda(
        q,
        k,
        v,
        g_log,
        beta,
        initial_state_source=h0,
        initial_state_indices=idx,
        disable_state_update=True,
    )
    assert fs is None
    torch.cuda.synchronize()
    assert torch.equal(h0, h0_before), "frozen mode modified the state pool"
    ref = kda_output_only_decode(q, k, v, g_log, beta, h0, idx, backend="auto")
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)


def test_frozen_mode_public_api_caches():
    """Frozen mode fills the slot-indexed fp32 correction / bf16 kg caches
    (untouched rows keep their sentinel) for batched and packed inputs."""
    from flashinfer import recurrent_kda
    from flashinfer.kda_kernels.kda_decode_wy_output_only import (
        kda_recoverssm_verify,
    )

    B, T, H, HV = 3, 8, 4, 4
    q, k, v, graw, _, h0, idx, A_log, dt_bias = _make_inputs(B, T, H, HV, seed=7)
    beta = torch.randn(B, T, HV, dtype=torch.bfloat16, device=q.device)
    corr = torch.full((B + 2, HV, T, 128), 7.0, dtype=torch.float32, device=q.device)
    kg = torch.full((B + 2, HV, T, 256), 7.0, dtype=torch.bfloat16, device=q.device)
    out, fs = recurrent_kda(
        q,
        k,
        v,
        graw,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
        initial_state_source=h0,
        initial_state_indices=idx,
        disable_state_update=True,
        correction_cache=corr,
        kg_cache=kg,
    )
    assert fs is None
    # Bit-identical to the (private) verify path on the packed views.
    corr_r = torch.full_like(corr, 7.0)
    kg_r = torch.full_like(kg, 7.0)
    qsl = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
    # null_min=0: the recurrent_kda mode treats slot 0 as valid (only
    # negative slots are padding), unlike the vLLM drop-in default where
    # slot 0 is the reserved null block.
    out_r = kda_recoverssm_verify(
        q.view(1, B * T, H, 128),
        k.view(1, B * T, H, 128),
        v.view(1, B * T, HV, 128),
        graw.view(1, B * T, HV, 128),
        beta.view(1, B * T, HV),
        A_log,
        dt_bias,
        -5.0,
        h0,
        corr_r,
        kg_r,
        qsl,
        idx,
        T,
        null_min=0,
    )
    assert torch.equal(out.reshape(1, B * T, HV, 128), out_r)
    assert torch.equal(corr, corr_r) and torch.equal(kg, kg_r)
    # Unused slots keep their sentinel (slot-indexed, only active slots hit).
    used = set(idx.tolist())
    for s in range(B + 2):
        if s not in used:
            assert torch.all(corr[s] == 7.0) and torch.all(kg[s] == 7.0)


def test_frozen_mode_public_api_errors():
    """Invalid frozen-mode combinations raise."""
    from flashinfer import recurrent_kda

    B, T, H, HV = 2, 4, 2, 2
    q, k, v, _, beta, h0, idx, _, _ = _make_inputs(B, T, H, HV, seed=8)
    g_log = torch.zeros(B, T, HV, 128, dtype=torch.bfloat16, device=q.device)
    corr = torch.zeros(B, HV, T, 128, dtype=torch.float32, device=q.device)
    with pytest.raises(ValueError):  # caches without the flag
        recurrent_kda(
            q,
            k,
            v,
            g_log,
            beta,
            initial_state_source=h0,
            initial_state_indices=idx,
            correction_cache=corr,
        )
    with pytest.raises(ValueError):  # cake has no frozen kernels
        recurrent_kda(
            q,
            k,
            v,
            g_log,
            beta,
            initial_state_source=h0,
            initial_state_indices=idx,
            disable_state_update=True,
            backend="cake",
        )
    with pytest.raises(ValueError):  # no final state in frozen mode
        recurrent_kda(
            q,
            k,
            v,
            g_log,
            beta,
            initial_state_source=h0,
            initial_state_indices=idx,
            disable_state_update=True,
            output_final_state=True,
        )
    with pytest.raises(ValueError):  # a state pool is required
        recurrent_kda(q, k, v, g_log, beta, disable_state_update=True)


def test_frozen_mode_shape_validation():
    """Mis-shaped cross-tensor inputs raise instead of reading out of bounds."""
    from flashinfer import recurrent_kda

    B, T, H, HV = 2, 4, 2, 2
    q, k, v, _, beta, h0, idx, _, _ = _make_inputs(B, T, H, HV, seed=9)
    g = torch.zeros(B, T, HV, 128, dtype=torch.bfloat16, device=q.device)
    ok = dict(
        initial_state_source=h0, initial_state_indices=idx, disable_state_update=True
    )
    with pytest.raises(ValueError):  # k shape mismatch
        recurrent_kda(q, k[:, :, :, :64], v, g, beta, **ok)
    with pytest.raises(ValueError):  # gate head-count mismatch
        recurrent_kda(q, k, v, g[:, :, :1], beta, **ok)
    with pytest.raises(ValueError):  # beta token-count mismatch
        recurrent_kda(q, k, v, g, beta[:, :2], **ok)
    with pytest.raises(ValueError):  # wrong dtype
        recurrent_kda(q, k, v.float(), g, beta, **ok)
    with pytest.raises(ValueError):  # state pool inner dims
        recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state_source=h0[:, :, :64],
            initial_state_indices=idx,
            disable_state_update=True,
        )
    with pytest.raises(ValueError):  # slot indices wrong length
        recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state_source=h0,
            initial_state_indices=idx[:1],
            disable_state_update=True,
        )


def test_frozen_mode_caches_presigmoided_beta():
    """Regression: the packed verify path must honor beta_is_logit=False
    (a leftover vLLM-contract override used to force the in-kernel sigmoid,
    double-sigmoiding pre-sigmoided betas)."""
    from flashinfer import recurrent_kda

    B, T, H, HV = 2, 4, 2, 2
    q, k, v, _, _, h0, idx, _, _ = _make_inputs(B, T, H, HV, seed=11)
    g_log = F.logsigmoid(
        torch.randn(B, T, HV, 128, dtype=torch.float32, device=q.device)
    ).to(torch.bfloat16)
    beta = torch.rand(B, T, HV, device=q.device).to(torch.bfloat16)  # pre-sigmoided
    corr = torch.full((B + 2, HV, T, 128), 7.0, dtype=torch.float32, device=q.device)
    kg = torch.full((B + 2, HV, T, 256), 7.0, dtype=torch.bfloat16, device=q.device)
    out, _ = recurrent_kda(
        q,
        k,
        v,
        g_log,
        beta,
        initial_state_source=h0,
        initial_state_indices=idx,
        disable_state_update=True,
        correction_cache=corr,
        kg_cache=kg,
    )
    scale = 128**-0.5
    o_ref, U_ref = ref_corrections(
        l2n(q).float(),
        l2n(k).float(),
        v,
        g_log.float(),
        beta.float(),
        h0,
        idx,
        scale,
    )
    torch.testing.assert_close(out.float(), o_ref, atol=2e-2, rtol=1e-2)
    got_U = corr[idx.long()].permute(0, 2, 1, 3)
    torch.testing.assert_close(got_U, U_ref, atol=6e-2, rtol=2e-2)
    # verify contract stores the RAW key (vLLM convention)
    kraw = k.float().repeat_interleave(HV // H, dim=2)
    torch.testing.assert_close(
        kg[idx.long(), :, :, :128].permute(0, 2, 1, 3).float(), kraw
    )
