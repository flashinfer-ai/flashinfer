import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm100a_supported

try:
    from flashinfer import recurrent_kda
    from flashinfer.kda_decode import _RECURRENT_KDA_AVAILABLE

    _has_recurrent_kda = _RECURRENT_KDA_AVAILABLE
except ImportError:
    recurrent_kda = None
    _has_recurrent_kda = False


K = 128
POOL_SLOTS = 11


@pytest.fixture(autouse=True)
def _require_recurrent_kda():
    if not torch.cuda.is_available():
        pytest.skip("Recurrent KDA requires CUDA")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Recurrent KDA requires SM100a (Blackwell)")
    if not _has_recurrent_kda:
        pytest.skip("recurrent_kda kernel not available (missing cutlass DSL deps)")


def _reference_step(q, k, v, g, beta, state, scale):
    """One l2-normalised, pre-gated delta-rule step; state is [HV, V, K] fp32."""
    H, HV = q.shape[0], v.shape[0]
    q = F.normalize(q.float(), dim=-1) * scale
    k = F.normalize(k.float(), dim=-1)
    q, k = q.repeat_interleave(HV // H, 0), k.repeat_interleave(HV // H, 0)
    state = state * g.float().exp()[:, None, :]
    delta = (v.float() - torch.einsum("hvk,hk->hv", state, k)) * beta.float()[:, None]
    state = state + delta[:, :, None] * k[:, None, :]
    return torch.einsum("hvk,hk->hv", state, q), state


def _make(B, H, HV, layout, state_dtype, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, B) if layout == "packed" else (B, 1)

    def r(*s, scale=1.0):
        return (torch.randn(*s, generator=gen) * scale).to(torch.bfloat16).cuda()

    p = dict(
        q=r(*shape, H, K),
        k=r(*shape, H, K),
        v=r(*shape, HV, K),
        g=(-torch.rand(*shape, HV, K, generator=gen)).to(torch.bfloat16).cuda(),
        beta=torch.rand(*shape, HV, generator=gen).to(torch.bfloat16).cuda(),
    )
    pool32 = (torch.randn(POOL_SLOTS, HV, K, K, generator=gen) * 0.1).cuda()
    p["pool32"] = pool32
    p["pool"] = pool32.to(state_dtype).clone()
    p["idx"] = torch.randperm(POOL_SLOTS, generator=gen)[:B].to(torch.int32).cuda()
    return p


@pytest.mark.parametrize("layout", ["packed", "batched"])
@pytest.mark.parametrize(
    "B,H,HV", [(1, 64, 64), (2, 64, 64), (8, 64, 64), (3, 16, 16), (4, 16, 32)]
)
def test_fp32_state_decode_matches_reference(B, H, HV, layout):
    p = _make(B, H, HV, layout, torch.float32)
    scale = K**-0.5
    kwargs = dict(
        scale=scale,
        initial_state=p["pool"],
        ssm_state_indices=p["idx"],
        use_qk_l2norm_in_kernel=True,
    )
    if layout == "packed":
        kwargs["cu_seqlens"] = torch.arange(B + 1, dtype=torch.int32, device="cuda")
    out, _ = recurrent_kda(p["q"], p["k"], p["v"], p["g"], p["beta"], **kwargs)
    torch.cuda.synchronize()

    out = out.reshape(B, HV, K).float()
    flat = lambda t: t.reshape(B, *t.shape[2:])
    q, k, v, g, beta = (flat(p[n]) for n in ("q", "k", "v", "g", "beta"))
    untouched = torch.ones(POOL_SLOTS, dtype=torch.bool)
    for b in range(B):
        slot = int(p["idx"][b])
        untouched[slot] = False
        ref_o, ref_s = _reference_step(
            q[b], k[b], v[b], g[b], beta[b], p["pool32"][slot], scale
        )
        torch.testing.assert_close(out[b], ref_o, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(p["pool"][slot], ref_s, atol=1e-4, rtol=1e-4)
    assert p["pool"].dtype == torch.float32
    assert torch.equal(p["pool"][untouched.cuda()], p["pool32"][untouched.cuda()])


def test_fp32_state_is_not_rounded_to_bf16():
    p = _make(8, 64, 64, "packed", torch.float32)
    recurrent_kda(
        p["q"],
        p["k"],
        p["v"],
        p["g"],
        p["beta"],
        initial_state=p["pool"],
        ssm_state_indices=p["idx"],
        cu_seqlens=torch.arange(9, dtype=torch.int32, device="cuda"),
        use_qk_l2norm_in_kernel=True,
    )
    slot = p["pool"][p["idx"].long()]
    assert not torch.equal(slot, slot.bfloat16().float())


def test_float16_state_rejected():
    p = _make(2, 64, 64, "packed", torch.float16)
    with pytest.raises(TypeError, match="bfloat16 or float32 state"):
        recurrent_kda(
            p["q"],
            p["k"],
            p["v"],
            p["g"],
            p["beta"],
            initial_state=p["pool"],
            ssm_state_indices=p["idx"],
            cu_seqlens=torch.arange(3, dtype=torch.int32, device="cuda"),
        )


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("num_spec_tokens", [1, 3])
def test_fp32_state_spec_decode_matches_reference(num_spec_tokens, accepted):
    N, H, HV, T = 4, 16, 16, 1 + num_spec_tokens
    gen = torch.Generator(device="cpu").manual_seed(1)

    def r(*s):
        return torch.randn(*s, generator=gen).to(torch.bfloat16).cuda()

    q, k, v = r(1, N * T, H, K), r(1, N * T, H, K), r(1, N * T, HV, K)
    g = (-torch.rand(1, N * T, HV, K, generator=gen)).to(torch.bfloat16).cuda()
    beta = torch.rand(1, N * T, HV, generator=gen).to(torch.bfloat16).cuda()
    pool32 = (torch.randn(N * T + 5, HV, K, K, generator=gen) * 0.1).cuda()
    pool = pool32.clone()
    idx = torch.arange(N * T, dtype=torch.int32).reshape(N, T).cuda()
    cu = torch.arange(0, N * T + 1, T, dtype=torch.int32, device="cuda")
    nat = torch.tensor([1, 2, T, 1][:N], dtype=torch.int32).cuda() if accepted else None
    extra = {"num_accepted_tokens": nat} if accepted else {}
    scale = K**-0.5

    out, _ = recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=pool,
        cu_seqlens=cu,
        ssm_state_indices=idx,
        num_spec_tokens=num_spec_tokens,
        use_qk_l2norm_in_kernel=True,
        **extra,
    )
    torch.cuda.synchronize()
    assert pool.dtype == torch.float32

    for b in range(N):
        start = max(int(nat[b]) - 1, 0) if accepted else 0
        state = pool32[int(idx[b, start])]
        for t in range(T):
            n = b * T + t
            ref_o, state = _reference_step(
                q[0, n], k[0, n], v[0, n], g[0, n], beta[0, n], state, scale
            )
            torch.testing.assert_close(out[0, n].float(), ref_o, atol=3e-2, rtol=3e-2)
            if t >= start:
                torch.testing.assert_close(
                    pool[int(idx[b, t])], state, atol=1e-4, rtol=1e-4
                )
    assert torch.equal(pool[N * T :], pool32[N * T :])


@pytest.mark.parametrize("B", [1, 8])
def test_bf16_then_fp32_same_process_uses_separate_specializations(B):
    ref = {}
    for dt in (torch.bfloat16, torch.float32, torch.bfloat16):
        p = _make(B, 64, 64, "packed", dt, seed=3)
        p["idx"] = torch.arange(B, dtype=torch.int32).cuda()
        recurrent_kda(
            p["q"],
            p["k"],
            p["v"],
            p["g"],
            p["beta"],
            initial_state=p["pool"],
            ssm_state_indices=p["idx"],
            cu_seqlens=torch.arange(B + 1, dtype=torch.int32, device="cuda"),
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()
        assert p["pool"].dtype == dt
        ref.setdefault(dt, []).append(p["pool"][:B].float().clone())
    torch.testing.assert_close(ref[torch.bfloat16][0], ref[torch.bfloat16][1])
    torch.testing.assert_close(
        ref[torch.float32][0], ref[torch.bfloat16][0], atol=5e-2, rtol=5e-2
    )
