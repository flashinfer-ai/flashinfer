# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Adapted for Recurrent KDA kernel testing

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm100a_supported

try:
    from flashinfer.kda_kernels import recurrent_kda

    _has_recurrent_kda = True
except ImportError:
    _has_recurrent_kda = False

try:
    from fla.ops.kda import fused_recurrent_kda

    _has_fla = True
except ImportError:
    _has_fla = False


def _skip_if_not_sm100():
    """Skip test if not Blackwell (SM100+) architecture."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Recurrent KDA requires SM100a (Blackwell)")
    if not _has_recurrent_kda:
        pytest.skip("recurrent_kda kernel not available (missing cutlass DSL deps)")


# ==============================================================================
# Reference implementations (inlined to avoid external dependencies)
# ==============================================================================


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K**-0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "b h k, b h v -> b h k v",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def naive_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = (-A_log.view(H, 1).float().exp() * F.softplus(g.float())).to(output_dtype)
    return g


def naive_kda_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = lower_bound * F.sigmoid(A_log.view(H, 1).exp() * g)
    return g.to(output_dtype)


# ==============================================================================
# Test helpers
# ==============================================================================


def assert_close(prefix, ref, tri, atol=5e-3, rtol=5e-3):
    """Assert tensors are close with bf16-appropriate tolerances."""
    ref_f, tri_f = ref.float(), tri.float()
    abs_diff = (ref_f - tri_f).flatten().abs().max().item()
    assert not torch.isnan(ref).any(), f"{prefix}: NaN in ref"
    assert not torch.isnan(tri).any(), f"{prefix}: NaN in tri"
    torch.testing.assert_close(
        ref_f, tri_f, atol=atol, rtol=rtol, msg=f"{prefix} diff: {abs_diff:.6f}"
    )


# ==============================================================================
# Tests
# ==============================================================================


@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "D",
        "scale",
        "gate_logit_normalizer",
        "use_qk_l2norm_in_kernel",
        "dtype",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-norm{}-qk_l2{}-{}".format(*test),
        )
        for test in [
            (1, 1, 4, 64, 1, 1, True, torch.bfloat16),
            (4, 1, 4, 128, 0.1, 1, True, torch.bfloat16),
            (16, 1, 4, 128, 0.1, 1, True, torch.bfloat16),
        ]
    ],
)
def test_recurrent_kda_vs_naive(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    """Recurrent KDA kernel matches naive recurrent KDA reference."""
    _skip_if_not_sm100()
    torch.manual_seed(42)
    device = torch.device("cuda")

    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    g = (
        F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float, device=device))
        / gate_logit_normalizer
    ).to(dtype)
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    h0_bf16 = torch.randn(B, H, D, D, dtype=torch.bfloat16, device=device) * 0.01
    # Reference needs f32 [B, H, K, V] state
    h0_f32 = h0_bf16.transpose(-1, -2).float()

    # Reference: naive recurrent (float32, pre-normalized)
    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.float(), p=2, dim=-1),
        k=F.normalize(k.float(), p=2, dim=-1),
        v=v.float(),
        g=g.float(),
        beta=beta.float(),
        scale=scale,
        initial_state=h0_f32.clone(),
        output_final_state=True,
    )

    # Recurrent KDA kernel (bf16 state [B, H, V, K], in-kernel L2 norm)
    tri, tri_ht = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0_bf16.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    assert_close("o", ref, tri)
    # Convert Recurrent KDA bf16 [V,K] state to f32 [K,V] for comparison
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2).float())


@pytest.mark.skipif(not _has_fla, reason="fla package not installed")
@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "D",
        "scale",
        "gate_logit_normalizer",
        "use_qk_l2norm_in_kernel",
        "dtype",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-norm{}-qk_l2{}-{}".format(*test),
        )
        for test in [
            (1, 1, 4, 64, 1, 1, True, torch.bfloat16),
            (4, 1, 4, 128, 0.1, 1, True, torch.bfloat16),
            (16, 1, 4, 128, 0.1, 1, True, torch.bfloat16),
        ]
    ],
)
def test_recurrent_kda_vs_fla(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    """Recurrent KDA kernel matches fla fused_recurrent_kda."""
    _skip_if_not_sm100()
    torch.manual_seed(42)
    device = torch.device("cuda")

    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    g = (
        F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float, device=device))
        / gate_logit_normalizer
    ).to(dtype)
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    h0_bf16 = torch.randn(B, H, D, D, dtype=torch.bfloat16, device=device) * 0.01
    # fla needs f32 [B, H, K, V] state
    h0_f32 = h0_bf16.transpose(-1, -2).float()

    # fla reference (f32 state [K,V])
    ref, ref_ht = fused_recurrent_kda(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g.float(),
        beta=beta.float(),
        scale=scale,
        initial_state=h0_f32.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Recurrent KDA (bf16 state [B, H, V, K])
    tri, tri_ht = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0_bf16.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    assert_close("o", ref, tri)
    # Convert Recurrent KDA bf16 [V,K] state to f32 [K,V] for comparison
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2).float())


@pytest.mark.parametrize(
    (
        "B",
        "H",
        "D",
        "scale",
        "gate_logit_normalizer",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "safe_gate",
        "dtype",
    ),
    [
        pytest.param(
            *test,
            id="B{}-H{}-D{}-scale{}-norm{}-qk_l2{}-gate{}-safe{}-{}".format(*test),
        )
        for test in [
            (16, 16, 128, 0.1, 1.0, True, False, False, torch.bfloat16),
            (32, 8, 64, 1.0, 1.0, True, False, False, torch.bfloat16),
            (7, 32, 128, 0.5, 0.5, True, False, False, torch.bfloat16),
            (16, 16, 128, 0.1, 1.0, True, True, False, torch.bfloat16),
            (32, 8, 64, 1.0, 1.0, True, True, False, torch.bfloat16),
            (7, 32, 128, 0.5, 0.5, True, True, True, torch.bfloat16),
        ]
    ],
)
def test_vllm_decode(
    B: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    dtype: torch.dtype,
):
    """vLLM-style decoding: continuous batching with paged state, Recurrent KDA vs naive."""
    _skip_if_not_sm100()
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Setup cache pool (bf16 [V,K] for Recurrent KDA)
    max_cache_slots = B * 3
    state_pool_bf16 = torch.randn(
        max_cache_slots, H, D, D, dtype=torch.bfloat16, device=device
    )
    state_indices = torch.randperm(max_cache_slots, device=device)[:B].int()

    # Fill unaccessed slots with huge value to detect out-of-bound access
    HUGE_VALUE = 1e4  # bf16 max is ~65504, use 1e4 as sentinel
    unaccessed = torch.ones(max_cache_slots, dtype=torch.bool, device=device)
    unaccessed[state_indices.long()] = False
    state_pool_bf16[unaccessed] = HUGE_VALUE

    T = 1
    total_tokens = B * T

    q = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    k = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    v = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    g = torch.randn(
        1,
        total_tokens,
        H,
        D,
        dtype=torch.float if not use_gate_in_kernel else dtype,
        device=device,
    )

    if use_gate_in_kernel:
        A_log = torch.log(
            torch.randn(1, 1, H, 1, dtype=torch.float32, device=device).uniform_(1, 16)
        ).squeeze()
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
        lower_bound = -5.0 if safe_gate else None
        naive_gate_fn = naive_kda_lowerbound_gate if safe_gate else naive_kda_gate
    else:
        g = (F.logsigmoid(g) / gate_logit_normalizer).to(dtype)
        A_log = None
        dt_bias = None
        lower_bound = None
        naive_gate_fn = None

    beta = torch.randn(1, total_tokens, H, dtype=dtype, device=device).sigmoid()

    cu_seqlens = torch.arange(
        0, total_tokens + 1, step=T, device=device, dtype=torch.long
    )
    # Reference needs f32 [K,V] state; Recurrent KDA uses bf16 [V,K] directly
    ref_state_pool = state_pool_bf16.transpose(-1, -2).float()
    tri_state_pool_bf16 = state_pool_bf16.clone()

    # Reference: loop over batch with naive recurrent
    ref_outputs = []
    for i in range(B):
        start, end = i, i + 1
        slot_idx = state_indices[i].item()

        q_i = q[:, start:end]
        k_i = k[:, start:end]
        v_i = v[:, start:end]
        g_i = g[:, start:end]
        beta_i = beta[:, start:end]

        h_init = ref_state_pool[slot_idx].unsqueeze(0)
        gate_kwargs = dict(lower_bound=lower_bound) if safe_gate else {}
        ref_o_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q_i.float(), p=2, dim=-1),
            k=F.normalize(k_i.float(), p=2, dim=-1),
            v=v_i.float(),
            g=(
                naive_gate_fn(g_i, A_log, dt_bias, **gate_kwargs).float()
                if use_gate_in_kernel
                else g_i.float()
            ),
            beta=beta_i.float(),
            scale=scale,
            initial_state=h_init.clone(),
            output_final_state=True,
        )
        ref_outputs.append(ref_o_i)
        ref_state_pool[slot_idx] = ref_ht_i.squeeze(0)

    ref_out = torch.cat(ref_outputs, dim=1)

    # Recurrent KDA kernel with cu_seqlens + ssm_state_indices (bf16 state)
    tri_out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=tri_state_pool_bf16,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
    )

    assert_close("o", ref_out, tri_out)
    # Convert Recurrent KDA bf16 [V,K] state to f32 [K,V] for comparison
    tri_ht_f32 = tri_state_pool_bf16[state_indices.long()].transpose(-1, -2).float()
    assert_close("ht", ref_state_pool[state_indices.long()], tri_ht_f32)

    # Verify untouched slots (bf16 state pool)
    tri_untouched = tri_state_pool_bf16[unaccessed].transpose(-1, -2).float()
    ref_untouched = state_pool_bf16[unaccessed].transpose(-1, -2).float()
    # Untouched slots should not have been modified by the kernel
    assert_close("Untouched ht", ref_untouched, tri_untouched, atol=1e-2, rtol=1e-2)
