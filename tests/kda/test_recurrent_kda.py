# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Adapted for Recurrent KDA kernel testing

import importlib

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm100a_supported

try:
    from flashinfer.kda_decode import _RECURRENT_KDA_AVAILABLE, recurrent_kda

    _has_recurrent_kda = _RECURRENT_KDA_AVAILABLE
except ImportError:
    recurrent_kda = None
    _has_recurrent_kda = False

recurrent_kda_module = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")
TREE_REDUCTION = recurrent_kda_module.DOT_REDUCTION_TREE
DUAL_ACCUM_REDUCTION = recurrent_kda_module.DOT_REDUCTION_DUAL_ACCUM

try:
    from fla.ops.kda import fused_recurrent_kda

    _has_fla = True
except ImportError:
    _has_fla = False


@pytest.fixture(autouse=True)
def _require_recurrent_kda():
    if not torch.cuda.is_available():
        pytest.skip("Recurrent KDA requires CUDA")
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


def maybe_l2norm(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    x = x.float()
    return F.normalize(x, p=2, dim=-1) if enabled else x


# ==============================================================================
# Tests
# ==============================================================================

# Shared configs for vs_naive and vs_fla tests: (B, T, H, D, scale, norm, qk_l2, dtype)
_VS_REFERENCE_CONFIGS = [
    (1, 1, 4, 64, 1, 1, True, torch.bfloat16),
    (1, 1, 4, 64, 1, 1, False, torch.bfloat16),
    (4, 1, 4, 128, 0.1, 1, True, torch.bfloat16),
    (4, 1, 4, 128, 0.1, 1, False, torch.bfloat16),
    (16, 1, 4, 128, 0.1, 1, True, torch.bfloat16),
]

_VS_REFERENCE_PARAMS = [
    pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-norm{}-qk_l2{}-{}".format(*test))
    for test in _VS_REFERENCE_CONFIGS
]


def _make_vs_reference_tensors(B, T, H, D, gate_logit_normalizer, dtype):
    """Create shared tensors for vs_naive / vs_fla tests."""
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
    h0_f32 = h0_bf16.transpose(-1, -2).float()
    return q, k, v, g, beta, h0_bf16, h0_f32


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
    _VS_REFERENCE_PARAMS,
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
    q, k, v, g, beta, h0_bf16, h0_f32 = _make_vs_reference_tensors(
        B, T, H, D, gate_logit_normalizer, dtype
    )

    # Reference: naive recurrent (float32, pre-normalized)
    ref, ref_ht = naive_recurrent_kda(
        q=maybe_l2norm(q, use_qk_l2norm_in_kernel),
        k=maybe_l2norm(k, use_qk_l2norm_in_kernel),
        v=v.float(),
        g=g.float(),
        beta=beta.float(),
        scale=scale,
        initial_state=h0_f32.clone(),
        output_final_state=True,
    )

    # Recurrent KDA kernel (bf16 state [B, H, V, K], optional in-kernel L2 norm)
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
    _VS_REFERENCE_PARAMS,
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
    q, k, v, g, beta, h0_bf16, h0_f32 = _make_vs_reference_tensors(
        B, T, H, D, gate_logit_normalizer, dtype
    )

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
            (4, 16, 128, 0.1, 1.0, False, False, False, torch.bfloat16),
            (32, 8, 64, 1.0, 1.0, True, False, False, torch.bfloat16),
            (8, 8, 64, 1.0, 1.0, False, True, False, torch.bfloat16),
            (7, 32, 128, 0.5, 0.5, True, False, False, torch.bfloat16),
            (16, 16, 128, 0.1, 1.0, True, True, False, torch.bfloat16),
            (32, 8, 64, 1.0, 1.0, True, True, False, torch.bfloat16),
            (7, 32, 128, 0.5, 0.5, True, True, False, torch.bfloat16),
            (7, 32, 128, 0.5, 0.5, True, True, True, torch.bfloat16),
            (4, 32, 128, 0.5, 0.5, False, True, True, torch.bfloat16),
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
            q=maybe_l2norm(q_i, use_qk_l2norm_in_kernel),
            k=maybe_l2norm(k_i, use_qk_l2norm_in_kernel),
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


def test_standard_decode_state_indices_update_pool():
    """Standard decode with ssm_state_indices updates the caller's state pool."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B, H, D = 4, 8, 64
    HV = H
    n_slots = B * 3

    q = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    k = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    v = torch.rand(B, 1, HV, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, 1, HV, D, device=device)).to(dtype)
    beta = torch.rand(B, 1, HV, dtype=dtype, device=device).sigmoid()
    scale = D**-0.5

    state_pool = torch.randn(n_slots, HV, D, D, dtype=dtype, device=device) * 0.01
    state_indices = torch.randperm(n_slots, device=device)[:B].int()
    untouched = torch.ones(n_slots, dtype=torch.bool, device=device)
    untouched[state_indices.long()] = False

    compact_state = state_pool[state_indices].clone()
    ref_out, ref_state = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=compact_state,
        output_final_state=True,
    )

    indexed_pool = state_pool.clone()
    original_untouched = indexed_pool[untouched].clone()
    tri_out, tri_state = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=indexed_pool,
        output_final_state=True,
        ssm_state_indices=state_indices,
    )

    assert_close("indexed output", ref_out.float(), tri_out.float())
    assert_close("indexed returned state", ref_state.float(), tri_state.float())
    assert_close(
        "indexed state pool",
        ref_state.float(),
        indexed_pool[state_indices.long()].float(),
    )
    assert_close(
        "indexed untouched state",
        original_untouched.float(),
        indexed_pool[untouched].float(),
        atol=1e-3,
        rtol=1e-3,
    )


# ==============================================================================
# vLLM CUDA graph padding tests
# ==============================================================================


@pytest.mark.parametrize(
    ("H", "D", "use_gate_in_kernel", "use_qk_l2norm_in_kernel"),
    [
        pytest.param(8, 64, False, True, id="H8-D64-precomputed"),
        pytest.param(16, 128, False, True, id="H16-D128-precomputed"),
        pytest.param(8, 64, False, False, id="H8-D64-precomputed-no-qk-l2"),
        pytest.param(8, 64, True, True, id="H8-D64-in-kernel"),
        pytest.param(16, 128, True, True, id="H16-D128-in-kernel"),
    ],
)
def test_vllm_padded_cuda_graph(
    H: int, D: int, use_gate_in_kernel: bool, use_qk_l2norm_in_kernel: bool
):
    """vLLM CUDA graph padding: padded slots (PAD_SLOT_ID=-1, seq_len=0) must not
    corrupt any state, and active slots must produce correct output + state.

    This verifies three things:
    1. In-kernel clamp: PAD_SLOT_ID=-1 doesn't cause IMA (negative pointer offset)
    2. seq_len guard: state slot 0 is not corrupted by padded CTAs
    3. Active slots: output and state match naive reference
    """
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B_active = 4
    B_padded = 3  # padded slots appended to simulate CUDA graph fixed batch size
    B_total = B_active + B_padded

    # State pool: active slots randomly placed, slot 0 filled with sentinel
    max_slots = 32
    SENTINEL = 7.0  # visible value to detect corruption (bf16 max ~65504)
    state_pool = torch.zeros(max_slots, H, D, D, dtype=dtype, device=device)
    state_pool[0] = SENTINEL  # slot 0 is the clamp target for padded entries

    # Assign active slots to random non-zero indices
    active_indices = torch.randperm(max_slots - 1, device=device)[:B_active] + 1
    state_pool[active_indices.long()] = (
        torch.randn(B_active, H, D, D, dtype=dtype, device=device) * 0.01
    )
    state_pool_ref = state_pool.clone()

    # Inputs: [1, B_total, H, D] packed
    q = torch.rand(1, B_total, H, D, dtype=dtype, device=device)
    k = torch.rand(1, B_total, H, D, dtype=dtype, device=device)
    v = torch.rand(1, B_total, H, D, dtype=dtype, device=device)

    if use_gate_in_kernel:
        g = torch.randn(1, B_total, H, D, dtype=dtype, device=device)
        A_log = torch.log(
            torch.ones(H, dtype=torch.float32, device=device).uniform_(1, 16)
        )
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
        lower_bound = None
        naive_gate_fn = naive_kda_gate
    else:
        g = (F.logsigmoid(torch.randn(1, B_total, H, D, device=device)) / 1.0).to(dtype)
        A_log = None
        dt_bias = None
        lower_bound = None
        naive_gate_fn = None

    beta = torch.rand(1, B_total, H, dtype=dtype, device=device).sigmoid()

    # cu_seqlens: active tokens have 1 token each, padded tokens have 0
    # [0, 1, 2, 3, 4, 4, 4, 4] -- last B_padded entries are duplicates
    cu_seqlens = torch.cat(
        [
            torch.arange(B_active + 1, device=device, dtype=torch.int32),
            torch.full((B_padded,), B_active, device=device, dtype=torch.int32),
        ]
    )

    # ssm_state_indices: active -> real slots, padded -> PAD_SLOT_ID=-1
    ssm_state_indices = torch.cat(
        [
            active_indices.int(),
            torch.full((B_padded,), -1, device=device, dtype=torch.int32),
        ]
    )

    scale = 1.0 / D**0.5

    # Run kernel (in-place state update)
    tri_out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state_pool,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    # ---- Test A: state slot 0 must not be corrupted ----
    slot0_after = state_pool[0]
    assert torch.all(slot0_after == SENTINEL), (
        f"State slot 0 corrupted! Expected all {SENTINEL}, "
        f"got min={slot0_after.min().item():.3f} max={slot0_after.max().item():.3f}"
    )

    # ---- Test B: active slots match naive reference ----
    ref_state_pool = state_pool_ref.transpose(-1, -2).float()  # [N, H, K, V]
    ref_outputs = []
    for i in range(B_active):
        slot_idx = active_indices[i].item()
        q_i = q[:, i : i + 1]
        k_i = k[:, i : i + 1]
        v_i = v[:, i : i + 1]
        g_i = g[:, i : i + 1]
        beta_i = beta[:, i : i + 1]

        gate_kwargs = {}
        ref_o_i, ref_ht_i = naive_recurrent_kda(
            q=maybe_l2norm(q_i, use_qk_l2norm_in_kernel),
            k=maybe_l2norm(k_i, use_qk_l2norm_in_kernel),
            v=v_i.float(),
            g=(
                naive_gate_fn(g_i, A_log, dt_bias, **gate_kwargs).float()
                if use_gate_in_kernel
                else g_i.float()
            ),
            beta=beta_i.float(),
            scale=scale,
            initial_state=ref_state_pool[slot_idx].unsqueeze(0).clone(),
            output_final_state=True,
        )
        ref_outputs.append(ref_o_i)
        ref_state_pool[slot_idx] = ref_ht_i.squeeze(0)

    ref_out = torch.cat(ref_outputs, dim=1)  # [1, B_active, H, D]

    assert_close("active output", ref_out, tri_out[:, :B_active])
    tri_ht = state_pool[active_indices.long()].transpose(-1, -2).float()
    assert_close(
        "active state",
        ref_state_pool[active_indices.long()],
        tri_ht,
        atol=0.1,
        rtol=0.05,
    )


@pytest.mark.parametrize(
    ("H", "D"),
    [
        pytest.param(8, 64, id="H8-D64"),
        pytest.param(16, 128, id="H16-D128"),
    ],
)
def test_non_compact_state_stride(H: int, D: int):
    """State tensor with non-compact stride[0] (simulating vLLM block-based cache
    where page_size includes extra conv_state padding) produces correct results.

    The kernel's compiled fake tensor uses a free sym_int64 stride[0], so it must
    handle any stride divisible by 16, not just the compact HV*V*K stride.

    Must use cu_seqlens path: the non-cu_seqlens path always calls .contiguous()
    on the state tensor, so non-compact stride only matters for the cu_seqlens path
    where state is passed directly to the kernel without copying.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = 4
    HV = H  # GQA ratio 1
    PADDING = 32  # extra elements per row to simulate page padding

    # Create state pool with non-compact stride[0]: stride[0] = HV*D*D + PADDING
    compact_stride = HV * D * D
    padded_stride = compact_stride + PADDING
    backing = torch.zeros(B * padded_stride, dtype=dtype, device=device)
    state_noncompact = backing.as_strided(
        size=(B, HV, D, D),
        stride=(padded_stride, D * D, D, 1),
    )
    state_init = torch.randn(B, HV, D, D, dtype=dtype, device=device) * 0.01
    state_noncompact.copy_(state_init)

    # Compact reference (same logical values, standard [B, HV, D, D] layout)
    state_compact = state_init.clone()

    # Inputs: packed [1, B, H, D] with cu_seqlens (1 token per sequence)
    q = torch.rand(1, B, H, D, dtype=dtype, device=device)
    k = torch.rand(1, B, H, D, dtype=dtype, device=device)
    v = torch.rand(1, B, H, D, dtype=dtype, device=device)
    g = (F.logsigmoid(torch.randn(1, B, H, D, device=device)) / 1.0).to(dtype)
    beta = torch.rand(1, B, H, dtype=dtype, device=device).sigmoid()
    scale = 1.0 / D**0.5
    cu_seqlens = torch.arange(B + 1, device=device, dtype=torch.int32)
    ssm_state_indices = torch.arange(B, device=device, dtype=torch.int32)

    # Run kernel with non-compact state (state passed directly, no .contiguous())
    out_nc, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_noncompact,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    # Run kernel with compact state (reference)
    out_c, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_compact,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    assert_close("non-compact output", out_c, out_nc)
    # Verify state was written to the non-compact tensor at the correct logical positions
    nc_vals = state_noncompact.transpose(-1, -2).float()
    c_vals = state_compact.transpose(-1, -2).float()
    assert_close("non-compact state", c_vals, nc_vals, atol=0.1, rtol=0.05)


# ==============================================================================
# Non-contiguous gate stride tests
# ==============================================================================


@pytest.mark.parametrize(
    "H, D",
    [
        pytest.param(8, 64, id="H8-D64"),
        pytest.param(16, 128, id="H16-D128"),
    ],
)
def test_non_contiguous_gate_stride(H: int, D: int):
    """Gate tensor with non-contiguous strides (simulating split() on a fused
    projection output) produces correct results.

    In vLLM, a fused projection [B, T, total_proj_dim] is split along dim=-1,
    yielding views where stride[1] = total_proj_dim != HV*K. After reshape to
    [B, T, HV, K], the batch and token strides are non-compact while HV and K
    strides remain compact.

    Tests both the batched path and the cu_seqlens path.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = 4
    HV = H  # GQA ratio 1
    EXTRA = 256  # extra elements to simulate Q+K+V+beta in the fused projection

    # --- Batched path (no cu_seqlens) ---
    # Simulate fused projection: backing tensor has padded token stride.
    # Use as_strided to force non-compact strides (reshape won't preserve them
    # when T=1 since size-1 dims have ambiguous strides in PyTorch).
    gate_dim = HV * D
    total_proj_dim = gate_dim + EXTRA  # token stride = total_proj_dim
    batch_stride = 1 * total_proj_dim  # B stride = T * total_proj_dim

    # Allocate backing buffer large enough for all B batches with padded strides
    backing = torch.zeros(B * batch_stride + EXTRA, dtype=dtype, device=device)
    # Fill gate values at the correct offsets
    g_values = F.logsigmoid(torch.randn(B, 1, HV, D, device=device)).to(dtype)
    for b in range(B):
        offset = b * batch_stride + EXTRA
        backing[offset : offset + gate_dim] = g_values[b, 0].reshape(-1)

    # Create non-contiguous view with explicit strides
    g_noncontiguous = backing[EXTRA:].as_strided(
        size=(B, 1, HV, D),
        stride=(batch_stride, total_proj_dim, D, 1),
    )
    assert g_noncontiguous.stride()[1] == total_proj_dim, (
        f"Expected non-compact token stride {total_proj_dim}, "
        f"got {g_noncontiguous.stride()[1]}"
    )
    g_contiguous = g_noncontiguous.contiguous()

    q = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    k = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    v = torch.rand(B, 1, HV, D, dtype=dtype, device=device)
    beta = torch.rand(B, 1, HV, dtype=dtype, device=device).sigmoid()
    scale = 1.0 / D**0.5
    state_nc = torch.randn(B, HV, D, D, dtype=dtype, device=device) * 0.01
    state_c = state_nc.clone()

    out_nc, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g_noncontiguous,
        beta=beta,
        scale=scale,
        initial_state=state_nc,
    )
    out_c, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g_contiguous,
        beta=beta,
        scale=scale,
        initial_state=state_c,
    )

    assert_close("non-contiguous gate output (batched)", out_c, out_nc)
    assert_close(
        "non-contiguous gate state (batched)",
        state_c.transpose(-1, -2).float(),
        state_nc.transpose(-1, -2).float(),
        atol=0.1,
        rtol=0.05,
    )

    # --- cu_seqlens path ---
    # Packed layout: [1, B, H/HV, D] with cu_seqlens
    cu_seqlens = torch.arange(B + 1, device=device, dtype=torch.int32)
    ssm_state_indices = torch.arange(B, device=device, dtype=torch.int32)

    # Same as_strided approach for packed layout [1, B, HV, D]
    # Here dim0=1 (outer batch=1), dim1=B (sequences), non-compact stride on dim1
    token_stride_cu = total_proj_dim  # stride for the sequence dim
    batch_stride_cu = B * total_proj_dim  # stride for outer dim0
    backing_cu = torch.zeros(batch_stride_cu + EXTRA, dtype=dtype, device=device)
    g_vals_cu = F.logsigmoid(torch.randn(1, B, HV, D, device=device)).to(dtype)
    for s in range(B):
        offset = s * token_stride_cu + EXTRA
        backing_cu[offset : offset + gate_dim] = g_vals_cu[0, s].reshape(-1)
    g_nc_cu = backing_cu[EXTRA:].as_strided(
        size=(1, B, HV, D),
        stride=(batch_stride_cu, token_stride_cu, D, 1),
    )
    assert g_nc_cu.stride()[1] == total_proj_dim
    g_c_cu = g_nc_cu.contiguous()

    q_cu = torch.rand(1, B, H, D, dtype=dtype, device=device)
    k_cu = torch.rand(1, B, H, D, dtype=dtype, device=device)
    v_cu = torch.rand(1, B, HV, D, dtype=dtype, device=device)
    beta_cu = torch.rand(1, B, HV, dtype=dtype, device=device).sigmoid()
    state_nc_cu = torch.randn(B, HV, D, D, dtype=dtype, device=device) * 0.01
    state_c_cu = state_nc_cu.clone()

    out_nc_cu, _ = recurrent_kda(
        q=q_cu,
        k=k_cu,
        v=v_cu,
        g=g_nc_cu,
        beta=beta_cu,
        scale=scale,
        initial_state=state_nc_cu,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )
    out_c_cu, _ = recurrent_kda(
        q=q_cu,
        k=k_cu,
        v=v_cu,
        g=g_c_cu,
        beta=beta_cu,
        scale=scale,
        initial_state=state_c_cu,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    assert_close("non-contiguous gate output (cu_seqlens)", out_c_cu, out_nc_cu)
    assert_close(
        "non-contiguous gate state (cu_seqlens)",
        state_c_cu.transpose(-1, -2).float(),
        state_nc_cu.transpose(-1, -2).float(),
        atol=0.1,
        rtol=0.05,
    )


# ==============================================================================
# Speculative decoding tests (spec-decode Phase 1)
# ==============================================================================


def make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device, dtype=torch.bfloat16):
    """Create packed spec-decode inputs.

    Returns:
        q, k, v, g, beta: packed [1, N*T, H/HV, D] tensors
        cu_seqlens: [0, T, 2T, ..., N*T] int32
        ssm_state_indices: [N, T] int32 -- unique slots, no overlap
        state_pool: [N*T + 5, HV, D, D] bf16 -- extra slots as sentinels
        T: NUM_TOKENS
    """
    T = 1 + num_spec_tokens
    total_tokens = N * T

    # All slots unique: sequence i uses slots i*T .. i*T+T-1
    ssm_state_indices = torch.stack(
        [
            torch.arange(i * T, i * T + T, dtype=torch.int32, device=device)
            for i in range(N)
        ]
    )  # [N, T]

    # state pool: N*T active slots + 5 sentinel slots
    n_pool = N * T + 5
    state_pool = torch.randn(n_pool, HV, D, D, dtype=dtype, device=device) * 0.1

    cu_seqlens = torch.arange(
        0, total_tokens + 1, step=T, dtype=torch.int32, device=device
    )

    scale = D**-0.5
    q = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    k = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    v = torch.rand(1, total_tokens, HV, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(1, total_tokens, HV, D, device=device)).to(dtype)
    beta = torch.rand(1, total_tokens, HV, dtype=dtype, device=device).sigmoid()

    return q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, T


def spec_decode_naive_reference(
    q,
    k,
    v,
    g,
    beta,
    ssm_state_indices,
    state_pool,
    scale,
    N,
    T,
    H,
    HV,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=True,
):
    """Run spec decode reference: T sequential naive tokens per sequence.

    Args:
        num_accepted_tokens: Optional[list or tensor] of length N. When provided,
            initial state for sequence b is loaded from ssm_state_indices[b, nat[b]-1]
            instead of ssm_state_indices[b, 0]. This matches the FLA Triton kernel's
            num_accepted_tokens behavior.

    Returns:
        ref_out: [1, N*T, HV, D] float32 -- output
        ref_states: list of N lists of T tensors, each [HV, K, V] = [HV, D, D] float32
    """
    device = q.device
    D = q.shape[-1]
    ref_out = torch.zeros(1, N * T, HV, D, dtype=torch.float32, device=device)
    ref_states = []  # ref_states[b][t] = [HV, D, D] float32 (K-major)

    gqa_ratio = HV // H

    for b in range(N):
        seq_start = b * T
        # Initial state: use nat-based slot selection when num_accepted_tokens provided
        if num_accepted_tokens is not None:
            nat_b = int(num_accepted_tokens[b])
            init_col = max(nat_b - 1, 0)
        else:
            init_col = 0
        init_slot = int(ssm_state_indices[b, init_col].item())
        current_state = state_pool[init_slot].float().transpose(-1, -2)  # [HV, K, V]
        # Expand to [1, HV, K, V] for naive_recurrent_kda
        current_state_b = current_state.unsqueeze(0)

        b_states = []
        for t in range(T):
            tok_idx = seq_start + t
            q_t = q[:, tok_idx : tok_idx + 1, :H, :]  # [1, 1, H, D]
            k_t = k[:, tok_idx : tok_idx + 1, :H, :]
            v_t = v[:, tok_idx : tok_idx + 1, :HV, :]  # [1, 1, HV, D]
            g_t = g[:, tok_idx : tok_idx + 1, :HV, :]  # [1, 1, HV, D]
            beta_t = beta[:, tok_idx : tok_idx + 1, :HV]  # [1, 1, HV]

            # For GQA (HV > H): expand q and k so each value head has its own q/k copy.
            # naive_recurrent_kda treats all heads uniformly, so we replicate query heads.
            if gqa_ratio > 1:
                q_t_exp = q_t.repeat_interleave(gqa_ratio, dim=2)  # [1, 1, HV, D]
                k_t_exp = k_t.repeat_interleave(gqa_ratio, dim=2)  # [1, 1, HV, D]
            else:
                q_t_exp = q_t
                k_t_exp = k_t

            o_t, ht = naive_recurrent_kda(
                q=maybe_l2norm(q_t_exp, use_qk_l2norm_in_kernel),
                k=maybe_l2norm(k_t_exp, use_qk_l2norm_in_kernel),
                v=v_t.float(),
                g=g_t.float(),
                beta=beta_t.float(),
                scale=scale,
                initial_state=current_state_b.clone(),
                output_final_state=True,
            )
            ref_out[0, tok_idx] = o_t[0, 0]  # [HV, D]
            current_state_b = ht  # carry forward
            b_states.append(ht[0].clone())  # [HV, K, V]
        ref_states.append(b_states)

    return ref_out, ref_states


def assert_spec_states(
    state_pool,
    ssm_state_indices,
    ref_states,
    N,
    T,
    atol=1e-1,
    rtol=5e-2,
    num_accepted_tokens=None,
):
    """Assert each per-token checkpoint slot matches the reference state.

    When num_accepted_tokens is provided, only validate slots from nat[b]-1 onward.
    Earlier slots (0..nat-2) were computed from a different initial state and are
    valid-but-meaningless -- vLLM's postprocess_mamba only reads from nat-1 onward.
    """
    for b in range(N):
        start_t = 0
        if num_accepted_tokens is not None:
            start_t = max(int(num_accepted_tokens[b]) - 1, 0)
        for t in range(start_t, T):
            slot = int(ssm_state_indices[b, t].item())
            # kernel: [HV, V, K] bf16 -> [HV, K, V] float for comparison
            kernel_state = state_pool[slot].float().transpose(-1, -2)
            ref_state = ref_states[b][t]
            assert_close(
                f"spec_state[b={b}, t={t}]",
                ref_state,
                kernel_state,
                atol=atol,
                rtol=rtol,
            )


# ------------------------------------------------------------------------------
# 3a: test_spec_decode_basic
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "N",
        "H",
        "HV",
        "D",
        "num_spec_tokens",
        "use_qk_l2norm_in_kernel",
    ),
    [
        pytest.param(4, 8, 8, 64, 1, True, id="N4-H8-D64-S1"),
        pytest.param(4, 8, 8, 64, 3, True, id="N4-H8-D64-S3"),
        pytest.param(4, 8, 8, 64, 1, False, id="N4-H8-D64-S1-no-qk-l2"),
        pytest.param(8, 16, 16, 128, 2, True, id="N8-H16-D128-S2"),
        pytest.param(8, 16, 16, 128, 4, True, id="N8-H16-D128-S4"),
        pytest.param(8, 16, 16, 128, 2, False, id="N8-H16-D128-S2-no-qk-l2"),
        pytest.param(4, 4, 8, 64, 2, True, id="N4-H4-HV8-D64-S2-GQA"),
        pytest.param(4, 8, 8, 128, 3, True, id="N4-H8-D128-S3"),
    ],
)
def test_spec_decode_basic(
    N,
    H,
    HV,
    D,
    num_spec_tokens,
    use_qk_l2norm_in_kernel,
):
    """Single spec-decode call matches T sequential naive calls."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, T = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device)
    )

    # Reference
    ref_out, ref_states = spec_decode_naive_reference(
        q,
        k,
        v,
        g,
        beta,
        ssm_state_indices,
        state_pool,
        scale,
        N,
        T,
        H,
        HV,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Kernel
    tri_state_pool = state_pool.clone()
    tri_out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=tri_state_pool,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Output comparison
    assert_close(
        "spec_output", ref_out.bfloat16().float(), tri_out.float(), atol=1e-1, rtol=5e-2
    )

    # Per-token state comparison (EACH slot must match)
    assert_spec_states(
        tri_state_pool, ssm_state_indices, ref_states, N, T, atol=1e-1, rtol=5e-2
    )


@pytest.mark.parametrize("D", [64, 128])
def test_spec_decode_separate_source_and_beta_logits(D):
    """Committed state is read in-kernel and beta logits are activated there."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H, HV, num_spec_tokens = 4, 8, 8, 3

    q, k, v, g, _, cu_seqlens, ssm_state_indices, scratch, scale, T = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device)
    )
    beta_logits = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device=device)
    beta = beta_logits.sigmoid()
    source_values = torch.randn(N + 3, HV, D, D, dtype=torch.bfloat16, device=device)
    source_stride = HV * D * D + 32
    source_backing = torch.zeros(
        (N + 3) * source_stride, dtype=torch.bfloat16, device=device
    )
    source = source_backing.as_strided(
        size=(N + 3, HV, D, D),
        stride=(source_stride, D * D, D, 1),
    )
    source.copy_(source_values)
    source_backing_before = source_backing.clone()
    source_indices_storage = torch.tensor(
        [3, -1, 1, -1, 5, -1, 0, -1], dtype=torch.int32, device=device
    )
    source_indices = source_indices_storage[::2]
    assert not source_indices.is_contiguous()

    reference_pool = scratch.clone()
    for batch_idx in range(N):
        reference_pool[ssm_state_indices[batch_idx, 0]] = source[
            source_indices[batch_idx]
        ]
    ref_out, ref_states = spec_decode_naive_reference(
        q, k, v, g, beta, ssm_state_indices, reference_pool, scale, N, T, H, HV
    )

    output_pool = scratch.clone()
    out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta_logits,
        beta_is_logit=True,
        scale=scale,
        initial_state=output_pool,
        initial_state_source=source,
        initial_state_indices=source_indices,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
    )

    assert_close(
        "separate_source_output",
        ref_out.bfloat16().float(),
        out.float(),
        atol=1e-1,
        rtol=5e-2,
    )
    assert_spec_states(
        output_pool, ssm_state_indices, ref_states, N, T, atol=1e-1, rtol=5e-2
    )
    assert torch.equal(source_backing, source_backing_before)


@pytest.mark.parametrize(
    (
        "D",
        "N",
        "tokens",
        "use_gate",
        "expected_rows",
        "expected_reduction",
    ),
    [
        pytest.param(
            64,
            1,
            1,
            False,
            8,
            TREE_REDUCTION,
            id="D64-T1-low-grid-tree",
        ),
        pytest.param(
            64,
            128,
            1,
            False,
            16,
            DUAL_ACCUM_REDUCTION,
            id="D64-T1-large-grid-dual",
        ),
        pytest.param(
            128,
            4,
            1,
            True,
            16,
            DUAL_ACCUM_REDUCTION,
            id="D128-T1-smart-row16",
        ),
        pytest.param(
            128,
            8,
            8,
            True,
            8,
            TREE_REDUCTION,
            id="D128-low-grid-tree",
        ),
        pytest.param(
            128,
            14,
            8,
            True,
            16,
            DUAL_ACCUM_REDUCTION,
            id="D128-mid-grid-row16",
        ),
        pytest.param(
            128,
            45,
            2,
            True,
            16,
            DUAL_ACCUM_REDUCTION,
            id="D128-large-grid-T2",
        ),
        pytest.param(
            128,
            45,
            8,
            False,
            8,
            DUAL_ACCUM_REDUCTION,
            id="D128-large-grid-row8-dual",
        ),
        pytest.param(
            128,
            16,
            8,
            False,
            32,
            DUAL_ACCUM_REDUCTION,
            id="D128-mid-grid-row32-dual",
        ),
        pytest.param(
            64,
            8,
            3,
            False,
            8,
            TREE_REDUCTION,
            id="D64-T3-low-grid-tree",
        ),
        pytest.param(
            64,
            8,
            4,
            False,
            8,
            TREE_REDUCTION,
            id="D64-low-grid-tree",
        ),
        pytest.param(
            64,
            8,
            4,
            True,
            16,
            DUAL_ACCUM_REDUCTION,
            id="D64-fused-row16",
        ),
        pytest.param(
            64,
            256,
            4,
            False,
            16,
            DUAL_ACCUM_REDUCTION,
            id="D64-mid-grid-row16",
        ),
        pytest.param(
            64,
            256,
            6,
            False,
            32,
            DUAL_ACCUM_REDUCTION,
            id="D64-high-grid-row32",
        ),
    ],
)
def test_kernel_schedule(
    D,
    N,
    tokens,
    use_gate,
    expected_rows,
    expected_reduction,
):
    """The measured tile and reduction schedule is selected for each workload."""
    HV = 8 if D == 64 else 16
    selected = recurrent_kda_module._select_kernel_schedule(
        D,
        tokens,
        use_gate,
        N * HV,
    )
    assert selected == (expected_rows, expected_reduction)


# ------------------------------------------------------------------------------
# 3b: test_spec_decode_gate_modes
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gate_mode",
    [
        pytest.param("precomputed", id="precomputed"),
        pytest.param("softplus", id="softplus"),
        pytest.param("lower_bound", id="lower_bound"),
    ],
)
def test_spec_decode_gate_modes(gate_mode):
    """All 3 gate modes work correctly with spec decode."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H, HV, D, num_spec_tokens = 4, 16, 16, 128, 2
    dtype = torch.bfloat16

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, T = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device, dtype)
    )

    if gate_mode == "precomputed":
        use_gate_in_kernel = False
        lower_bound = None
        A_log = None
        dt_bias = None
        g_ref = g.float()
        g_kernel = g
    elif gate_mode == "softplus":
        use_gate_in_kernel = True
        lower_bound = None
        A_log = torch.log(
            torch.rand(H, device=device, dtype=torch.float32) + 1
        ).squeeze()
        dt_bias = None
        # g is the raw input; reference computes the gate from same raw g
        g_ref = naive_kda_gate(g, A_log).float()
        g_kernel = g  # same raw g for kernel (gate computed internally)
    else:  # lower_bound
        use_gate_in_kernel = True
        lower_bound = -5.0
        A_log = torch.log(
            torch.rand(H, device=device, dtype=torch.float32) + 1
        ).squeeze()
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
        # g is the raw input; reference computes the gate from same raw g
        g_ref = naive_kda_lowerbound_gate(g, A_log, dt_bias).float()
        g_kernel = g  # same raw g for kernel (gate computed internally)

    # Reference using the computed log-space gate
    ref_out, ref_states = spec_decode_naive_reference(
        q,
        k,
        v,
        g_ref.bfloat16(),
        beta,
        ssm_state_indices,
        state_pool,
        scale,
        N,
        T,
        H,
        HV,
    )

    tri_state_pool = state_pool.clone()
    tri_out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g_kernel,
        beta=beta,
        scale=scale,
        initial_state=tri_state_pool,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )

    assert_close(
        "spec_output_gate",
        ref_out.bfloat16().float(),
        tri_out.float(),
        atol=1e-1,
        rtol=5e-2,
    )
    assert_spec_states(
        tri_state_pool, ssm_state_indices, ref_states, N, T, atol=1e-1, rtol=5e-2
    )


# ------------------------------------------------------------------------------
# 3c: test_spec_decode_padded_cuda_graph
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("D", [64, 128], ids=["D64", "D128"])
def test_spec_decode_padded_cuda_graph(D):
    """Padded slots (PAD_SLOT_ID=-1) don't corrupt state; active slots correct; padded output=0.

    Slot 0 is reserved as a true PAD sentinel -- no active sequence uses it.
    Active slots start from 1 to ensure slot 0 is only touched by PAD clamp reads
    (which are harmless -- reads are allowed, only writes are guarded by is_active).
    D=64 exercises the register-carry path; D=128 exercises the GMEM round-trip path.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")
    N_active = 4
    N_padded = 3
    N = N_active + N_padded
    H, HV, num_spec_tokens = 8, 8, 2
    T = 1 + num_spec_tokens
    dtype = torch.bfloat16

    # Active slots start from 1 -- slot 0 is reserved as PAD sentinel
    # ssm_active[i] = [1+i*T, 1+i*T+1, ..., 1+i*T+T-1]
    ssm_active = torch.stack(
        [
            torch.arange(1 + i * T, 1 + i * T + T, dtype=torch.int32, device=device)
            for i in range(N_active)
        ]
    )  # [N_active, T], slots 1..N_active*T

    n_pool = N_active * T + 1 + 5  # slot 0 + active slots + sentinel slots
    state_pool = torch.randn(n_pool, HV, D, D, dtype=dtype, device=device) * 0.1
    scale = D**-0.5

    # Inputs for active sequences
    q = torch.rand(1, N_active * T, H, D, dtype=dtype, device=device)
    k = torch.rand(1, N_active * T, H, D, dtype=dtype, device=device)
    v = torch.rand(1, N_active * T, HV, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(1, N_active * T, HV, D, device=device)).to(dtype)
    beta = torch.rand(1, N_active * T, HV, dtype=dtype, device=device).sigmoid()

    # Extend inputs for padded sequences (dummy tokens -- read unconditionally but writes guarded)
    total_tokens = N * T
    q_full = torch.cat(
        [q, torch.zeros(1, N_padded * T, H, D, dtype=dtype, device=device)], dim=1
    )
    k_full = torch.cat(
        [k, torch.zeros(1, N_padded * T, H, D, dtype=dtype, device=device)], dim=1
    )
    v_full = torch.cat(
        [v, torch.zeros(1, N_padded * T, HV, D, dtype=dtype, device=device)], dim=1
    )
    g_full = torch.cat(
        [g, torch.zeros(1, N_padded * T, HV, D, dtype=dtype, device=device)], dim=1
    )
    beta_full = torch.cat(
        [beta, torch.zeros(1, N_padded * T, HV, dtype=dtype, device=device)], dim=1
    )

    cu_seqlens = torch.arange(
        0, total_tokens + 1, step=T, dtype=torch.int32, device=device
    )

    # Padded rows: all -1 (PAD_SLOT_ID)
    ssm_padded = torch.full((N_padded, T), -1, dtype=torch.int32, device=device)
    ssm_state_indices = torch.cat([ssm_active, ssm_padded], dim=0)

    # Reference: only for active sequences
    ref_out_active, ref_states = spec_decode_naive_reference(
        q, k, v, g, beta, ssm_active, state_pool, scale, N_active, T, H, HV
    )

    # Slot 0 = PAD sentinel -- record before kernel run
    state_pool_kernel = state_pool.clone()
    slot0_before = state_pool_kernel[0].clone()

    tri_out, _ = recurrent_kda(
        q=q_full,
        k=k_full,
        v=v_full,
        g=g_full,
        beta=beta_full,
        scale=scale,
        initial_state=state_pool_kernel,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
    )

    # Active outputs correct (first N_active*T output positions)
    assert_close(
        "padded_active_out",
        ref_out_active.bfloat16().float(),
        tri_out[:, : N_active * T].float(),
        atol=1e-1,
        rtol=5e-2,
    )

    # Active states correct (each per-token checkpoint slot)
    assert_spec_states(
        state_pool_kernel, ssm_active, ref_states, N_active, T, atol=1e-1, rtol=5e-2
    )

    # Padded output positions are zero (spec mode allocates zeros)
    padded_out = tri_out[:, N_active * T :]
    assert padded_out.abs().max().item() == 0.0, (
        f"Padded output should be zero, max={padded_out.abs().max().item()}"
    )

    # Slot 0 not corrupted by PAD clamp -- padded CTAs read slot 0 (is_active=False skips writes)
    assert_close(
        "slot0_not_corrupted",
        slot0_before.float().transpose(-1, -2),
        state_pool_kernel[0].float().transpose(-1, -2),
        atol=1e-3,
        rtol=1e-3,
    )


# ------------------------------------------------------------------------------
# 3d: test_spec_decode_shape_validation
# ------------------------------------------------------------------------------


def test_spec_decode_shape_validation():
    """ValueError raised for invalid inputs in spec mode."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H, HV, D, num_spec_tokens = 4, 8, 8, 64, 2
    T = 3
    dtype = torch.bfloat16

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, _ = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device, dtype)
    )

    def try_call(**overrides):
        kw = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=state_pool.clone(),
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_spec_tokens=num_spec_tokens,
        )
        kw.update(overrides)
        return recurrent_kda(**kw)

    # num_spec_tokens=0 raises
    with pytest.raises(ValueError, match="num_spec_tokens must be >= 1"):
        try_call(num_spec_tokens=0)

    # ssm_state_indices=None raises
    with pytest.raises(ValueError, match="ssm_state_indices is required"):
        try_call(ssm_state_indices=None)

    # 1D ssm_state_indices raises
    with pytest.raises(ValueError, match="2D"):
        try_call(ssm_state_indices=torch.arange(N, dtype=torch.int32, device=device))

    # Wrong column count raises
    with pytest.raises(ValueError, match="2D"):
        try_call(
            ssm_state_indices=torch.zeros(N, T + 1, dtype=torch.int32, device=device)
        )

    # Missing cu_seqlens activates batched spec shim -- T mismatch raises
    with pytest.raises(ValueError, match="must equal 1\\+num_spec_tokens"):
        try_call(cu_seqlens=None)

    # Wrong row count
    with pytest.raises(ValueError, match="shape\\[0\\]"):
        try_call(
            ssm_state_indices=torch.zeros(N + 1, T, dtype=torch.int32, device=device)
        )

    # Checks below (bounds, uniqueness, cu_seqlens deltas, shape mismatches) were
    # removed from the hot path to enable CUDA graph compatibility (no D2H transfers).
    # Use validate_spec_decode_inputs() for development-time validation.


# ------------------------------------------------------------------------------
# 3e-neg: test_spec_decode_cu_seqlens_validation
# ------------------------------------------------------------------------------


def test_spec_decode_cu_seqlens_validation():
    """Valid cu_seqlens still works; invalid delta checks were removed from the hot path."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H, HV, D, num_spec_tokens = 4, 8, 8, 64, 2
    dtype = torch.bfloat16

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, _ = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device, dtype)
    )

    out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_pool.clone(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
    )

    assert out.shape == v.shape


# ------------------------------------------------------------------------------
# 3e: test_spec_decode_num_accepted_tokens
# ------------------------------------------------------------------------------


def test_spec_decode_num_accepted_tokens():
    """num_accepted_tokens selects initial state checkpoint; kernel outputs differ by nat."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H, HV, D, num_spec_tokens = 4, 8, 8, 64, 2
    T = 1 + num_spec_tokens
    dtype = torch.bfloat16

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, _ = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device, dtype)
    )

    # nat=1: should match no-nat (start from slot 0)
    nat_1 = torch.ones(N, dtype=torch.int32, device=device)
    tri_pool_none = state_pool.clone()
    out_none, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=tri_pool_none,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=None,
    )
    tri_pool_1 = state_pool.clone()
    out_1, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=tri_pool_1,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=nat_1,
    )
    assert_close("nat1_vs_none_out", out_none, out_1, atol=0, rtol=0)

    # nat=T: start from last checkpoint of previous round
    nat_T = torch.full((N,), T, dtype=torch.int32, device=device)
    tri_pool_T = state_pool.clone()
    out_T, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=tri_pool_T,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=nat_T,
    )
    ref_out_T, ref_states_T = spec_decode_naive_reference(
        q,
        k,
        v,
        g,
        beta,
        ssm_state_indices,
        state_pool,
        scale,
        N,
        T,
        H,
        HV,
        num_accepted_tokens=nat_T,
    )
    # Only compare output from nat-1 onward (earlier tokens have wrong initial state)
    for b in range(N):
        start = T - 1  # nat=T -> start_t = T-1
        for t in range(start, T):
            tok_idx = b * T + t
            assert_close(
                f"nat_T_out[b={b},t={t}]",
                ref_out_T[0, tok_idx],
                out_T[0, tok_idx].float(),
                atol=1e-1,
                rtol=5e-2,
            )
    assert_spec_states(
        tri_pool_T, ssm_state_indices, ref_states_T, N, T, num_accepted_tokens=nat_T
    )

    # Values above T clamp to the final checkpoint instead of indexing the next row.
    nat_over = torch.full((N,), T + 2, dtype=torch.int32, device=device)
    tri_pool_over = state_pool.clone()
    out_over, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=tri_pool_over,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=nat_over,
    )
    assert_close("nat_over_vs_T_out", out_over, out_T, atol=0, rtol=0)
    assert_close("nat_over_vs_T_state", tri_pool_over, tri_pool_T, atol=0, rtol=0)


def test_spec_decode_nat_equals_one_matches_no_nat():
    """nat=1 for all sequences produces bit-identical output to nat=None."""
    torch.manual_seed(123)
    device = torch.device("cuda")
    for D in [64, 128]:
        H = HV = 8 if D == 64 else 16
        N, num_spec_tokens = 4, 2
        q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, _ = (
            make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device)
        )

        pool_a, pool_b = state_pool.clone(), state_pool.clone()
        out_none, _ = recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=pool_a,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_spec_tokens=num_spec_tokens,
        )
        nat_1 = torch.ones(N, dtype=torch.int32, device=device)
        out_1, _ = recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=pool_b,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens=nat_1,
        )
        assert_close(f"nat1_compat_D{D}_out", out_none, out_1, atol=0, rtol=0)
        # State pools should also be identical
        assert_close(f"nat1_compat_D{D}_state", pool_a, pool_b, atol=0, rtol=0)


@pytest.mark.parametrize("D", [64, 128], ids=["D64", "D128"])
def test_spec_decode_nat_heterogeneous(D):
    """Different nat values per sequence in the same batch."""
    torch.manual_seed(77)
    device = torch.device("cuda")
    H = HV = 8 if D == 64 else 16
    N, num_spec_tokens = 6, 3
    T = 1 + num_spec_tokens  # T=4

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, _ = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device)
    )

    # Heterogeneous nat: 1, 2, 3, 4, 1, 2
    nat_values = [1, 2, 3, T, 1, 2]
    nat = torch.tensor(nat_values, dtype=torch.int32, device=device)

    tri_pool = state_pool.clone()
    tri_out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=tri_pool,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=nat,
    )
    ref_out, ref_states = spec_decode_naive_reference(
        q,
        k,
        v,
        g,
        beta,
        ssm_state_indices,
        state_pool,
        scale,
        N,
        T,
        H,
        HV,
        num_accepted_tokens=nat,
    )
    # Compare output from nat-1 onward per sequence
    for b in range(N):
        start = nat_values[b] - 1
        for t in range(start, T):
            tok_idx = b * T + t
            assert_close(
                f"hetero_out[b={b},t={t}]",
                ref_out[0, tok_idx],
                tri_out[0, tok_idx].float(),
                atol=1e-1,
                rtol=5e-2,
            )
    assert_spec_states(
        tri_pool, ssm_state_indices, ref_states, N, T, num_accepted_tokens=nat
    )


def test_spec_decode_nat_with_padding():
    """Mixed active (varying nat) + padded (-1) sequences."""
    torch.manual_seed(99)
    device = torch.device("cuda")
    N_active, N_padded = 3, 2
    N = N_active + N_padded
    H, HV, D, num_spec_tokens = 8, 8, 64, 2
    T = 1 + num_spec_tokens
    dtype = torch.bfloat16

    # Active slots start from 1 (slot 0 reserved as PAD sentinel)
    ssm_active = torch.stack(
        [
            torch.arange(1 + i * T, 1 + i * T + T, dtype=torch.int32, device=device)
            for i in range(N_active)
        ]
    )
    ssm_padded = torch.full((N_padded, T), -1, dtype=torch.int32, device=device)
    ssm_state_indices = torch.cat([ssm_active, ssm_padded], dim=0)

    n_pool = N_active * T + 1 + 5
    state_pool = torch.randn(n_pool, HV, D, D, dtype=dtype, device=device) * 0.1
    scale = D**-0.5

    q = torch.rand(1, N_active * T, H, D, dtype=dtype, device=device)
    k = torch.rand(1, N_active * T, H, D, dtype=dtype, device=device)
    v = torch.rand(1, N_active * T, HV, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(1, N_active * T, HV, D, device=device)).to(dtype)
    beta = torch.rand(1, N_active * T, HV, dtype=dtype, device=device).sigmoid()

    total_tokens = N * T
    q_full = torch.cat(
        [q, torch.zeros(1, N_padded * T, H, D, dtype=dtype, device=device)], dim=1
    )
    k_full = torch.cat(
        [k, torch.zeros(1, N_padded * T, H, D, dtype=dtype, device=device)], dim=1
    )
    v_full = torch.cat(
        [v, torch.zeros(1, N_padded * T, HV, D, dtype=dtype, device=device)], dim=1
    )
    g_full = torch.cat(
        [g, torch.zeros(1, N_padded * T, HV, D, dtype=dtype, device=device)], dim=1
    )
    beta_full = torch.cat(
        [beta, torch.zeros(1, N_padded * T, HV, dtype=dtype, device=device)], dim=1
    )
    cu_seqlens = torch.arange(
        0, total_tokens + 1, step=T, dtype=torch.int32, device=device
    )

    # nat: active seqs have varying nat, padded seqs have nat=1 (doesn't matter, they're -1)
    nat_active = torch.tensor([1, 2, T], dtype=torch.int32, device=device)
    nat_padded = torch.ones(N_padded, dtype=torch.int32, device=device)
    nat = torch.cat([nat_active, nat_padded])

    # Reference for active sequences only
    ref_out, ref_states = spec_decode_naive_reference(
        q,
        k,
        v,
        g,
        beta,
        ssm_active,
        state_pool,
        scale,
        N_active,
        T,
        H,
        HV,
        num_accepted_tokens=nat_active,
    )

    state_pool_kernel = state_pool.clone()
    slot0_before = state_pool_kernel[0].clone()

    tri_out, _ = recurrent_kda(
        q=q_full,
        k=k_full,
        v=v_full,
        g=g_full,
        beta=beta_full,
        scale=scale,
        initial_state=state_pool_kernel,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=nat,
    )

    # Active states correct (from nat-1 onward)
    assert_spec_states(
        state_pool_kernel,
        ssm_active,
        ref_states,
        N_active,
        T,
        num_accepted_tokens=nat_active,
    )

    # Padded output zero
    padded_out = tri_out[:, N_active * T :]
    assert padded_out.abs().max().item() == 0.0, (
        f"Padded output should be zero, max={padded_out.abs().max().item()}"
    )

    # Slot 0 not corrupted
    assert_close(
        "slot0_not_corrupted",
        slot0_before.float().transpose(-1, -2),
        state_pool_kernel[0].float().transpose(-1, -2),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "gate_mode",
    ["precomputed", "softplus", "lower_bound"],
    ids=["precomputed", "softplus", "lower_bound"],
)
def test_spec_decode_nat_gate_modes(gate_mode):
    """All 3 gate modes with nat > 1."""
    torch.manual_seed(55)
    device = torch.device("cuda")
    N, H, HV, D, num_spec_tokens = 4, 8, 8, 64, 2
    T = 1 + num_spec_tokens
    dtype = torch.bfloat16

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, _ = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device, dtype)
    )

    nat = torch.full((N,), 2, dtype=torch.int32, device=device)  # nat=2 for all

    use_gate_in_kernel = gate_mode != "precomputed"
    lower_bound = -5.0 if gate_mode == "lower_bound" else None
    A_log = (
        torch.randn(H, device=device, dtype=torch.float32)
        if use_gate_in_kernel
        else None
    )
    dt_bias = (
        torch.randn(H * D, device=device, dtype=torch.float32)
        if use_gate_in_kernel
        else None
    )

    if not use_gate_in_kernel:
        g = F.logsigmoid(torch.randn(1, N * T, HV, D, device=device)).to(dtype)
    else:
        g = torch.randn(1, N * T, HV, D, device=device).to(dtype) * 0.1

    tri_pool = state_pool.clone()
    tri_out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=tri_pool,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=nat,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )

    # Verify no NaN/Inf
    assert not torch.isnan(tri_out).any(), f"NaN in output for gate_mode={gate_mode}"
    assert not torch.isinf(tri_out).any(), f"Inf in output for gate_mode={gate_mode}"

    # Verify nat=2 differs from nat=1 (proving nat actually works)
    nat_1 = torch.ones(N, dtype=torch.int32, device=device)
    tri_pool_1 = state_pool.clone()
    out_1, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=tri_pool_1,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=nat_1,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    # Outputs should differ (different initial state)
    diff = (tri_out.float() - out_1.float()).abs().max().item()
    assert diff > 1e-3, f"nat=2 should differ from nat=1 for {gate_mode}, diff={diff}"


# ------------------------------------------------------------------------------
# 3f: test_spec_decode_checkpoint_correctness (stress test)
# ------------------------------------------------------------------------------


def test_spec_decode_checkpoint_correctness():
    """Stress test: N=8, H=16, D=128, S=4 -- all T checkpoint slots correct."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H, HV, D, num_spec_tokens = 8, 16, 16, 128, 4
    dtype = torch.bfloat16

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, T = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device, dtype)
    )

    ref_out, ref_states = spec_decode_naive_reference(
        q, k, v, g, beta, ssm_state_indices, state_pool, scale, N, T, H, HV
    )

    tri_state_pool = state_pool.clone()
    tri_out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=tri_state_pool,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
    )

    assert_close(
        "checkpoint_out",
        ref_out.bfloat16().float(),
        tri_out.float(),
        atol=1e-1,
        rtol=5e-2,
    )
    assert_spec_states(
        tri_state_pool, ssm_state_indices, ref_states, N, T, atol=1e-1, rtol=5e-2
    )


# ------------------------------------------------------------------------------
# 3g: test_spec_decode_non_compact_state
# ------------------------------------------------------------------------------


def test_spec_decode_non_compact_state():
    """Non-compact stride[0] state pool works correctly with spec decode.

    Non-compact means stride[0] > HV*V*K -- slots are not contiguous in memory.
    Strides [1:] remain standard (D*D, D, 1) within each slot.
    This matches vLLM's paged cache where page_size includes padding.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H, HV, D, num_spec_tokens = 4, 8, 8, 64, 2
    dtype = torch.bfloat16

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, T = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device, dtype)
    )

    n_slots = state_pool.shape[0]

    # Create non-compact state: stride[0] is padded (like vLLM paged cache).
    # stride[1:] = (D*D, D, 1) must remain standard for the kernel to work correctly.
    compact_stride = HV * D * D  # standard stride[0]
    PADDING = 32  # extra elements between slots
    padded_stride = compact_stride + PADDING
    backing = torch.zeros(n_slots * padded_stride, dtype=dtype, device=device)
    state_nc = backing.as_strided(
        size=(n_slots, HV, D, D),
        stride=(padded_stride, D * D, D, 1),
    )
    # Copy initial state values into non-compact backing
    state_nc.copy_(state_pool)

    # Reference with compact state (same initial values)
    ref_out, ref_states = spec_decode_naive_reference(
        q, k, v, g, beta, ssm_state_indices, state_pool, scale, N, T, H, HV
    )

    # Kernel with non-compact state (stride[0] = padded_stride, strides[1:] = standard)
    tri_out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_nc,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
    )

    assert_close(
        "nc_output", ref_out.bfloat16().float(), tri_out.float(), atol=1e-1, rtol=5e-2
    )
    # Verify each per-token state checkpoint slot in the non-compact pool
    for b in range(N):
        for t in range(T):
            slot = int(ssm_state_indices[b, t].item())
            kernel_state = state_nc[slot].float().transpose(-1, -2)
            assert_close(
                f"nc_state[b={b},t={t}]",
                ref_states[b][t],
                kernel_state,
                atol=1e-1,
                rtol=5e-2,
            )


# ------------------------------------------------------------------------------
# 3h: test_spec_decode_all_padded
# ------------------------------------------------------------------------------


def test_spec_decode_all_padded():
    """All-padded spec batch: output=0, state pool slot 0 not corrupted."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H, HV, D, num_spec_tokens = 4, 8, 8, 64, 2
    T = 1 + num_spec_tokens
    dtype = torch.bfloat16

    total_tokens = N * T
    cu_seqlens = torch.arange(
        0, total_tokens + 1, step=T, dtype=torch.int32, device=device
    )
    ssm_state_indices = torch.full(
        (N, T), -1, dtype=torch.int32, device=device
    )  # all padded

    q = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    k = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    v = torch.rand(1, total_tokens, HV, D, dtype=dtype, device=device)
    g = torch.zeros(1, total_tokens, HV, D, dtype=dtype, device=device)
    beta = torch.ones(1, total_tokens, HV, dtype=dtype, device=device) * 0.1
    scale = D**-0.5

    # State pool: at least 1 slot required by the wrapper guard
    state_pool = torch.randn(1, HV, D, D, dtype=dtype, device=device) * 0.5
    slot0_before = state_pool[0].clone()

    tri_out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_pool,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
    )

    # Output should be all zeros (spec mode allocates zeros for padded positions)
    assert tri_out.abs().max().item() == 0.0, (
        f"All-padded output should be zero, max={tri_out.abs().max().item()}"
    )

    # Slot 0 not corrupted
    assert_close(
        "allpad_slot0",
        slot0_before.float().transpose(-1, -2),
        state_pool[0].float().transpose(-1, -2),
        atol=1e-3,
        rtol=1e-3,
    )


# ------------------------------------------------------------------------------
# 3i: test_spec_decode_validate_slots_collision
# ------------------------------------------------------------------------------


def test_spec_decode_validate_slots_collision():
    """Cross-row slot collisions no longer validate in the hot path, but execution still works."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H, HV, D, num_spec_tokens = 4, 8, 8, 64, 1
    dtype = torch.bfloat16

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, scale, _ = (
        make_spec_decode_inputs(N, H, HV, D, num_spec_tokens, device, dtype)
    )

    ssm_dup = ssm_state_indices.clone()
    ssm_dup[0, 0] = 0
    ssm_dup[1, 0] = 0

    out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_pool.clone(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_dup,
        num_spec_tokens=num_spec_tokens,
    )

    assert out.shape == v.shape


# ------------------------------------------------------------------------------
# 3j: test_t1_cu_seqlens_all_padded
# ------------------------------------------------------------------------------


def test_t1_cu_seqlens_all_padded():
    """All-padded T=1 cu_seqlens batch: correct (no-op) output, no corruption.
    Two sub-cases: (a) initial_state=None exercises min-one-slot alloc guard;
                   (b) initial_state provided exercises no-corruption guard.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H, D = 4, 8, 64
    dtype = torch.bfloat16

    q = torch.rand(1, N, H, D, dtype=dtype, device=device)
    k = torch.rand(1, N, H, D, dtype=dtype, device=device)
    v = torch.rand(1, N, H, D, dtype=dtype, device=device)
    g = torch.zeros(1, N, H, D, dtype=dtype, device=device)
    beta = torch.ones(1, N, H, dtype=dtype, device=device) * 0.1
    scale = D**-0.5

    cu_seqlens = torch.arange(N + 1, dtype=torch.int32, device=device)
    ssm_all_padded = torch.full((N,), -1, dtype=torch.int32, device=device)

    # Sub-case (a): initial_state=None -> auto-allocation with at least 1 slot
    out_a, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_all_padded,
    )
    # All-padded: output is pre-zeroed and all kernel writes are guarded.
    assert out_a.abs().max().item() == 0.0, "Output should be zero for all-padded T=1"

    # Sub-case (b): initial_state provided -- slot 0 must not be corrupted
    state_pool = torch.randn(1, H, D, D, dtype=dtype, device=device)
    slot0_before = state_pool[0].clone()

    out_b, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_pool,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_all_padded,
    )

    # Slot 0 not corrupted
    assert_close(
        "t1_allpad_slot0",
        slot0_before.float().transpose(-1, -2),
        state_pool[0].float().transpose(-1, -2),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("D", [64, 128])
def test_t1_cu_seqlens_zero_length_rows(D):
    """Middle and trailing empty rows neither read tokens nor update state."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N, H = 3, 8
    dtype = torch.bfloat16

    q = torch.rand(1, 1, H, D, dtype=dtype, device=device)
    k = torch.rand(1, 1, H, D, dtype=dtype, device=device)
    v = torch.rand(1, 1, H, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(1, 1, H, D, device=device)).to(dtype)
    beta = torch.rand(1, 1, H, dtype=dtype, device=device).sigmoid()
    scale = D**-0.5
    state_pool = torch.randn(N, H, D, D, dtype=dtype, device=device)
    state_before = state_pool.clone()

    ref_state = state_pool[1:2].clone()
    ref_out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=ref_state,
    )

    out, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_pool,
        cu_seqlens=torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device),
    )

    assert_close("zero_length_output", out, ref_out, atol=0, rtol=0)
    assert_close("zero_length_active_state", state_pool[1:2], ref_state, atol=0, rtol=0)
    assert_close(
        "zero_length_leading_state", state_pool[0], state_before[0], atol=0, rtol=0
    )
    assert_close(
        "zero_length_trailing_state", state_pool[2], state_before[2], atol=0, rtol=0
    )

    empty_state = state_before.clone()
    empty_state_before = empty_state.clone()
    empty_out, empty_final_state = recurrent_kda(
        q=q[:, :0],
        k=k[:, :0],
        v=v[:, :0],
        g=g[:, :0],
        beta=beta[:, :0],
        scale=scale,
        initial_state=empty_state,
        output_final_state=True,
        cu_seqlens=torch.zeros(N + 1, dtype=torch.int32, device=device),
    )
    assert empty_out.shape == (1, 0, H, D)
    assert empty_final_state is empty_state
    assert_close(
        "all_zero_length_state", empty_state, empty_state_before, atol=0, rtol=0
    )


# ==============================================================================
# 4: Batched spec decode (no cu_seqlens)
# ==============================================================================
# The batched API (q shape [B, T, H, D] without cu_seqlens) is a thin shim that
# reshapes inputs to the packed cu_seqlens form and reuses the same kernel.
# Coverage below is intentionally minimal:
#   - vs_cuseqlens: parity with the cu_seqlens path (proves the shim is a pass-through).
#   - auto_ssi: verifies ssm_state_indices=None auto-generates identity slots.
# The kernel-side correctness (gate modes, CUDA graph capture, various shapes) is
# fully covered by the cu_seqlens tests above.

# ------------------------------------------------------------------------------
# 4a: test_spec_decode_batched_vs_cuseqlens
# ------------------------------------------------------------------------------


def test_spec_decode_batched_vs_cuseqlens():
    """Batched shim and cu_seqlens path produce identical results."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, H, HV, D, S = 4, 8, 8, 64, 2
    T = 1 + S
    scale = D**-0.5

    q, k, v, g, beta, cu_seqlens, ssm_state_indices, state_pool, _, _ = (
        make_spec_decode_inputs(B, H, HV, D, S, device, dtype)
    )

    # cu_seqlens path: inputs are [1, B*T, H, D]
    state_pool_csl = state_pool.clone()
    out_csl, _ = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_pool_csl,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=S,
    )

    # Batched path: reshape inputs to [B, T, H, D]
    q_b = q.reshape(B, T, H, D)
    k_b = k.reshape(B, T, H, D)
    v_b = v.reshape(B, T, HV, D)
    g_b = g.reshape(B, T, HV, D)
    beta_b = beta.reshape(B, T, HV)

    state_pool_bat = state_pool.clone()
    out_bat, _ = recurrent_kda(
        q=q_b,
        k=k_b,
        v=v_b,
        g=g_b,
        beta=beta_b,
        scale=scale,
        initial_state=state_pool_bat,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=S,
    )

    # cu_seqlens output is [1, B*T, HV, D]; batched output is [B, T, HV, D]
    assert_close(
        "batched_vs_csl_out",
        out_csl.reshape(B, T, HV, D).float(),
        out_bat.float(),
        atol=0,
        rtol=0,
    )

    # States in both pools should be identical
    assert_close(
        "batched_vs_csl_state",
        state_pool_csl.float(),
        state_pool_bat.float(),
        atol=0,
        rtol=0,
    )


# ------------------------------------------------------------------------------
# 4b: test_spec_decode_batched_auto_ssi
# ------------------------------------------------------------------------------


def test_spec_decode_batched_auto_ssi():
    """ssm_state_indices=None: auto-generates sequential slots 0,1,...,B*T-1."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, H, HV, D, S = 4, 8, 8, 64, 1
    T = 1 + S
    scale = D**-0.5

    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, HV, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, HV, D, device=device)).to(dtype)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()

    n_pool = B * T + 5
    state_pool = torch.randn(n_pool, HV, D, D, dtype=dtype, device=device) * 0.1

    # Call with ssm_state_indices=None -- shim auto-generates arange(B*T).reshape(B, T)
    tri_state_pool = state_pool.clone()
    tri_out, final_state = recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=tri_state_pool,
        output_final_state=True,
        num_spec_tokens=S,
        ssm_state_indices=None,
    )

    # Output shape is [B, T, HV, D]
    assert tri_out.shape == (B, T, HV, D), (
        f"Expected [{B},{T},{HV},{D}], got {list(tri_out.shape)}"
    )

    # Auto-generated indices: slot b*T+t for sequence b, token t
    expected_ssi = torch.arange(B * T, dtype=torch.int32, device=device).reshape(B, T)

    # Per-token states are in slots 0, 1, ..., B*T-1
    q_packed = q.reshape(1, B * T, H, D)
    k_packed = k.reshape(1, B * T, H, D)
    v_packed = v.reshape(1, B * T, HV, D)
    g_packed = g.reshape(1, B * T, HV, D)
    beta_packed = beta.reshape(1, B * T, HV)

    ref_out, ref_states = spec_decode_naive_reference(
        q_packed,
        k_packed,
        v_packed,
        g_packed,
        beta_packed,
        expected_ssi,
        state_pool,
        scale,
        B,
        T,
        H,
        HV,
    )

    ref_out_batched = ref_out.reshape(B, T, HV, D)
    assert_close(
        "auto_ssi_out",
        ref_out_batched.bfloat16().float(),
        tri_out.float(),
        atol=1e-1,
        rtol=5e-2,
    )

    # Verify states are in slots 0..B*T-1
    assert_spec_states(
        tri_state_pool, expected_ssi, ref_states, B, T, atol=1e-1, rtol=5e-2
    )
    assert final_state is tri_state_pool
    assert final_state.shape == (n_pool, HV, D, D)
