import pytest
import torch
from flashinfer import shuffle_matrix_a
from flashinfer.fused_moe.core import convert_to_block_layout
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    WeightLayout,
    trtllm_fp8_block_scale_moe,
)
from .utils import skip_checks, QuantMode
from flashinfer import ActivationType


def dequant_fp8_block_scaled(
    intermediate_size: int,
    hidden_size: int,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    num_experts_global: int,
    num_local_experts: int,
):
    # FP8 block-scale dequantization: float ≈ fp8 * scale
    H = hidden_size
    I = intermediate_size  # deepseek v3: 2048
    E_local = gemm1_weights.shape[0]

    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert E_global == num_experts_global, "num_experts_global shape mismatch"
    assert E_local == num_local_experts, "num_local_experts shape mismatch"

    # Block counts
    num_hidden_blocks = H // BLOCK  # 56
    num_intermediate_blocks = I // BLOCK  # 16
    num_gemm1_out_blocks = (2 * I) // BLOCK  # 32

    # Shape checks
    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (num_hidden_blocks, T)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_weights_scale.shape == (
        E_local,
        num_gemm1_out_blocks,
        num_hidden_blocks,
    )
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_weights_scale.shape == (
        E_local,
        num_hidden_blocks,
        num_intermediate_blocks,
    )
    assert routing_bias.shape[-1] == E_global

    # hidden_states: [T, H], scale: [H/128, T] (transposed layout)
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)  # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()  # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)  # [T, H/128, 128]
        .reshape(T, H)  # [T, H]
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded  # [T, H] float32

    # W13: [E_local, 2I, H], scale: [E_local, (2I)/128, H/128]
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)  # [E, 2I, H/128]
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)  # [E, 2I, H]
    W13 = W13_fp32 * S13_expanded  # [E, 2I, H] float32

    # W2: [E_local, H, I], scale: [E_local, H/128, I/128]
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)  # [E, H, I/128]
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)  # [E, H, I]
    W2 = W2_fp32 * S2_expanded  # [E, H, I] float32

    return A, W13, W2


def _deepseek_moe_core(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    intermediate_size: int,
    num_experts_global: int,
    num_local_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    hidden_size: int,
    A: torch.Tensor,
    W13: torch.Tensor,
    W2: torch.Tensor,
):
    """
    - DeepSeek-V3 no-aux routing:
        s = sigmoid(logits)
        s_with_bias = s + bias
        group by n_group=8; per group take top-2 sum → pick topk_group=4 groups
        on the kept groups, take global top_k=8 experts
        combine with weights derived from s (without bias), normalized and
        scaled by routed_scaling_factor
    - Local computation:
        only experts in [local_expert_offset, local_expert_offset + E_local) are
        computed on this rank (GEMM1 → SwiGLU → GEMM2), then per-token weighted
        accumulation.
    """

    # Routing constants
    TOP_K = top_k  # deepseek v3: 8
    N_GROUP = n_group  # deepseek v3: 8
    TOPK_GROUP = topk_group  # deepseek v3: 4

    I = intermediate_size  # deepseek v3: 2048
    H = hidden_size  # deepseek v3: 7168
    E_local = num_local_experts
    E_global = num_experts_global
    T = routing_logits.shape[0]

    device = A.device

    # 2) No-aux routing
    logits = routing_logits.to(torch.float32)  # [T, E_global]
    bias = routing_bias.to(torch.float32).reshape(-1)  # [E_global]

    # Sigmoid
    s = 1.0 / (1.0 + torch.exp(-logits))  # [T, E]
    s_with_bias = s + bias  # [T, E] (broadcast)

    # Grouping
    group_size = E_global // N_GROUP  # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)  # [T, 8, 32]

    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(
        s_wb_grouped, k=2, dim=2, largest=True, sorted=False
    )  # [T, 8, 2]
    group_scores = top2_vals.sum(dim=2)  # [T, 8]

    # Select topk_group groups → group mask
    _, group_idx = torch.topk(
        group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False
    )  # [T, 4]
    group_mask = torch.zeros_like(group_scores)  # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)
    )  # [T, E]

    # Global top-k (within kept groups), based on s_with_bias
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)  # [T, E]
    _, topk_idx = torch.topk(
        scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False
    )  # [T, 8]

    # Combination weights: use s (without bias) for normalization
    M = torch.zeros_like(s)  # [T, E]
    M.scatter_(1, topk_idx, 1.0)  # 0/1 mask
    weights = s * M  # [T, E]
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor  # [T, E]

    # 3) Local expert compute and accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    local_start = int(local_expert_offset)

    # For each local expert: find selected tokens, run GEMM1→SwiGLU→GEMM2, accumulate by weights
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        # Tokens that selected this global expert ge in their top-k
        sel_mask_per_token = (topk_idx == ge).any(dim=1)  # [T] bool
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)  # [Tk]

        # Gather inputs and weights for this expert
        A_e = A.index_select(0, token_idx)  # [Tk, H]
        W13_e = W13[le]  # [2I, H]
        W2_e = W2[le]  # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())  # [Tk, 2I]

        # SwiGLU: split and apply silu(x) = x / (1 + exp(-x))
        X1 = G1[:, :I]  # [Tk, I]
        X2 = G1[:, I:]  # [Tk, I]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))  # [Tk, I]
        C = silu_X2 * X1  # [Tk, I]

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t())  # [Tk, H]

        # Accumulate with per-token routing weights for this expert
        w_tok = weights.index_select(0, token_idx)[:, ge]  # [Tk]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))  # [Tk,H] * [Tk,1]

    return output.to(torch.bfloat16)


def run_fp8_block_scale_moe_reference(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    intermediate_size: int,
    num_experts_global: int,
    num_local_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    hidden_size: int,
):
    I = intermediate_size  # deepseek v3: 2048
    E_local = gemm1_weights.shape[0]
    H = hidden_size  # deepseek v3: 7168
    assert E_local == num_local_experts, "num_local_experts shape mismatch"

    E_global = routing_logits.shape[1]
    assert E_global == num_experts_global, "num_experts_global shape mismatch"

    # FP8 block-scale dequantization
    A, W13, W2 = dequant_fp8_block_scaled(
        hidden_size=H,
        intermediate_size=I,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        num_experts_global=E_global,
        num_local_experts=E_local,
    )

    # DeepSeek-V3 no-aux routing
    output = _deepseek_moe_core(
        A=A,
        W13=W13,
        W2=W2,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        num_experts_global=E_global,
        num_local_experts=E_local,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        hidden_size=H,
        intermediate_size=I,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=routed_scaling_factor,
    )

    return output


# -----------------------------
# Helpers: FP8 block quantization (dequant scale semantics)
# -----------------------------
def _fp8_block_quant_1d(x_bf16: torch.Tensor, block: int = 128):
    """
    Quantize [T, H] activations into FP8 with per-(token, 128-col) block scales.
    Returns:
      x_fp8: [T, H] (float8_e4m3fn)
      scales_TxNb: [T, H/128] (float32)  -- dequant scales (float ≈ fp8 * scale)
    """
    assert x_bf16.dim() == 2
    T, H = x_bf16.shape
    assert H % block == 0
    nb = H // block

    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max

    x_f32 = x_bf16.to(torch.float32)
    x_fp8 = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=x_bf16.device)
    scales = torch.empty((T, nb), dtype=torch.float32, device=x_bf16.device)

    for j in range(nb):
        sl = slice(j * block, (j + 1) * block)
        blk = x_f32[:, sl]  # [T, 128]
        amax = torch.amax(torch.abs(blk), dim=1)  # [T]
        # dequant scale s = amax / max_fp8  (float ≈ fp8 * s)
        s = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
        q = (blk / s.unsqueeze(1)).to(torch.float8_e4m3fn)  # quantization
        x_fp8[:, sl] = q
        scales[:, j] = s
    return x_fp8, scales  # scales in [T, H/128]


def _fp8_block_quant_2d(w_bf16: torch.Tensor, block: int = 128):
    """
    Quantize weights with 2D block scales over the last two dims.
      w_bf16: [*, R, C]  (R and C are multiples of 128)
    Returns:
      w_fp8: [*, R, C] (float8_e4m3fn)
      scales: [*, R/128, C/128] (float32) -- dequant scales
    """
    assert w_bf16.dim() >= 2
    *prefix, R, C = w_bf16.shape
    assert R % block == 0 and C % block == 0
    nb_r = R // block
    nb_c = C // block

    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max

    w_f32 = w_bf16.to(torch.float32).contiguous()
    prefix_ndim = len(prefix)

    # Reshape weights into 128x128 blocks and move block dims to the tail:
    # [..., nb_r, block, nb_c, block] -> [..., nb_r, nb_c, block, block]
    reshaped = w_f32.reshape(*prefix, nb_r, block, nb_c, block)
    permute_dims = tuple(range(prefix_ndim)) + (
        prefix_ndim,
        prefix_ndim + 2,
        prefix_ndim + 1,
        prefix_ndim + 3,
    )
    blocks = reshaped.permute(permute_dims).contiguous()

    # Compute per-block scales
    amax = torch.amax(torch.abs(blocks), dim=(-1, -2))
    scales = torch.where(
        amax > 0,
        amax / max_fp8,
        torch.ones_like(amax, dtype=torch.float32),
    )

    # Quantize blocks in parallel
    q_blocks = (blocks / scales.unsqueeze(-1).unsqueeze(-1)).to(torch.float8_e4m3fn)

    # Restore original layout
    inv_permute = [0] * (prefix_ndim + 4)
    for i, d in enumerate(permute_dims):
        inv_permute[d] = i
    w_fp8 = q_blocks.permute(*inv_permute).reshape(*prefix, R, C)

    return w_fp8, scales


# -----------------------------
# Random input generator for MoE DS-V3
# -----------------------------
def generate_random_inputs_moe(
    seq_len: int,
    *,
    num_experts_global: int = 256,
    num_local_experts: int = 32,
    hidden_size: int = 7168,
    intermediate_size: int = 2048,
    use_bias: bool = True,
    local_expert_offset: int = 0,
    routed_scaling_factor: float = 2.5,
    device: str = "cuda",
):
    assert hidden_size % 128 == 0 and intermediate_size % 128 == 0
    T, H, I = seq_len, hidden_size, intermediate_size
    E_global, E_local = num_experts_global, num_local_experts

    # Inputs for routing
    routing_logits = torch.randn(T, E_global, dtype=torch.float32, device=device)
    if use_bias:
        routing_bias = torch.randn(E_global, dtype=torch.bfloat16, device=device)
    else:
        routing_bias = torch.zeros(E_global, dtype=torch.bfloat16, device=device)

    # Activations: start from bf16, then FP8 block-quant with dequant scales
    a_bf16 = 2.0 * torch.randn(T, H, dtype=torch.bfloat16, device=device)
    a_fp8, a_scales_TxNb = _fp8_block_quant_1d(a_bf16, block=128)  # scales: [T, H/128]
    hidden_states = a_fp8
    hidden_states_scale = a_scales_TxNb.transpose(0, 1).contiguous()  # [H/128, T]

    # Weights per local expert
    # W13: [E_local, 2I, H], W2: [E_local, H, I]
    w13_bf16 = torch.randn(E_local, 2 * I, H, dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.randn(E_local, H, I, dtype=torch.bfloat16, device=device)

    w13_fp8, w13_scales = _fp8_block_quant_2d(
        w13_bf16, block=128
    )  # scales: [E, (2I)/128, H/128]
    w2_fp8, w2_scales = _fp8_block_quant_2d(
        w2_bf16, block=128
    )  # scales: [E, H/128, I/128]

    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": w13_fp8,
        "gemm1_weights_scale": w13_scales,
        "gemm2_weights": w2_fp8,
        "gemm2_weights_scale": w2_scales,
        "local_expert_offset": int(local_expert_offset),
        "local_num_experts": E_local,
        "routed_scaling_factor": float(routed_scaling_factor),
    }


def stats_accuracy(
    ref_out: torch.Tensor,
    fi_out: torch.Tensor,
    atol: float = 1e-1,
    rtol: float = 2e-1,
    percent: float = 0.85,
):
    H = ref_out.shape[1]
    assert H == 7168

    # Compare
    ref_f32 = ref_out.float()
    fi_f32 = fi_out.float()

    abs_diff = (ref_f32 - fi_f32).abs()
    rel_diff = abs_diff / (fi_f32.abs() + 1e-8)

    print("\nComparison stats:")
    print(f"Max abs diff:  {abs_diff.max().item():.6e}")
    print(f"Mean abs diff: {abs_diff.mean().item():.6e}")
    print(f"Max rel diff:  {rel_diff.max().item():.6e}")
    print(f"Mean rel diff: {rel_diff.mean().item():.6e}")

    # Cosine similarity and MSE
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), fi_f32.flatten(), dim=0
    ).item()
    mse = torch.mean((ref_f32 - fi_f32) ** 2).item()
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"MSE: {mse:.6e}")

    # Strict allclose
    allclose = torch.allclose(ref_f32, fi_f32, atol=atol, rtol=rtol)
    print(f"\nAllclose(atol={atol}, rtol={rtol}): {allclose}")

    if not allclose:
        # Show top-5 largest absolute errors
        flat = abs_diff.flatten()
        k = min(5, flat.numel())
        topv, topi = torch.topk(flat, k)
        print("\nTop-5 absolute error locations:")
        for rank in range(k):
            idx = topi[rank].item()
            t = idx // H
            h = idx % H
            print(
                f"  [t={t}, h={h}]: ref={ref_f32.flatten()[idx].item():.6e}, "
                f"fi={fi_f32.flatten()[idx].item():.6e}, diff={topv[rank].item():.6e}"
            )

    left = (ref_f32 - fi_f32).abs()
    right = atol + rtol * fi_f32.abs()
    ok = left <= right
    hit_ratio = ok.float().mean().item()
    print(f"\nHit ratio: {hit_ratio * 100:.2f}%  (need >= {percent * 100:.2f}%)")

    assert hit_ratio >= percent, (
        f"Hit ratio {hit_ratio * 100:.2f}% is less than required {percent * 100:.2f}%"
    )


# Max num tokens to tune for trtllm-gen fused moe
TUNE_MAX_NUM_TOKENS = 4096


# -----------------------------
# Test Entry
# -----------------------------
@pytest.mark.parametrize(
    "seq_len, local_expert_offset, use_bias",
    [
        (1, 0, False),
        (4, 0, True),
        (8, 64, True),
        (16, 32, True),
        (64, 128, True),
        (256, 64, True),
        (1024, 32, True),
    ],
)
@pytest.mark.parametrize("intermediate_size", [2048, 1024, 768, 512, 384])
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 384,
                "top_k": 8,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "compatible_intermediate_size": [1024, 2048],
                "enable_autotune": True,
            },
            id="kimi_k2",
        ),
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 8,
                "padding": 8,
                "n_groups": 8,
                "top_k_groups": 4,
                "routed_scaling": 2.5,
                "compatible_intermediate_size": [512, 1024, 2048],
                "enable_autotune": True,
            },
            id="DSv3",
        ),
        pytest.param(
            {
                "num_experts": 72,
                "top_k": 6,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "compatible_intermediate_size": [384, 768],
                "enable_autotune": False,
            },
            id="DSLite",
        ),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": False,
                "layout": WeightLayout.MajorK,
            },
            id="NoShuffle_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
            },
            id="Shuffled_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.BlockMajorK,
            },
            id="Shuffled_BlockMajorK",
        ),
    ],
)
def test_correctness_dpsk_fp8_fused_moe(
    seq_len: int,
    local_expert_offset: int,
    use_bias: bool,
    intermediate_size: int,
    routing_config: dict,
    enable_pdl: bool,
    weight_processing: dict,
    atol: float = 1e-1,
    rtol: float = 2e-1,
    percent: float = 0.85,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if trtllm_fp8_block_scale_moe is None:
        pytest.skip("flashinfer fused_moe kernel not available")

    # Create a mock MoE implementation for skip_checks
    class FP8BlockScaleMoe:
        def __init__(self):
            self.name = "FP8BlockScale"
            self.quant_mode = QuantMode.FP8_BLOCK_SCALE_DEEPSEEK

    moe_impl = FP8BlockScaleMoe()

    # Make copies of config dicts to avoid modifying the original parametrize values
    routing_config = dict(routing_config)
    weight_processing = dict(weight_processing)

    # Ensure they have compatible_moe_impls
    if "compatible_moe_impls" not in routing_config:
        routing_config["compatible_moe_impls"] = [type(moe_impl)]
    if "compatible_moe_impls" not in weight_processing:
        weight_processing["compatible_moe_impls"] = [type(moe_impl)]

    # Use the complete skip_checks function from utils
    skip_checks(
        moe_impl=moe_impl,
        routing_config=routing_config,
        weight_processing=weight_processing,
        activation_type=ActivationType.Swiglu,
        num_tokens=seq_len,
        hidden_size=7168,  # DeepSeek-V3 hidden size
        intermediate_size=intermediate_size,
    )

    device = "cuda"
    torch.manual_seed(42)

    # Constants (DeepSeek-V3)
    E_GLOBAL = routing_config["num_experts"]  # deepseek v3: 256
    E_LOCAL = 32  # todo(yingyi): default to tp8 for now, update later
    H = 7168
    I = intermediate_size  # deepseek v3: 2048
    TOP_K = routing_config["top_k"]  # deepseek v3: 8
    N_GROUP = routing_config["n_groups"]  # deepseek v3: 8
    TOPK_GROUP = routing_config["top_k_groups"]  # deepseek v3: 4

    if local_expert_offset + E_LOCAL > E_GLOBAL:
        pytest.skip(
            f"Local expert offset {local_expert_offset} + {E_LOCAL} is greater than number of experts {E_GLOBAL}"
        )

    # Generate random but consistent inputs
    inputs = generate_random_inputs_moe(
        seq_len,
        num_experts_global=E_GLOBAL,
        num_local_experts=E_LOCAL,
        hidden_size=H,
        intermediate_size=I,
        use_bias=use_bias,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=routing_config["routed_scaling"],
        device=device,
    )

    # Run reference (returns bf16)
    ref_out = run_fp8_block_scale_moe_reference(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        local_expert_offset=inputs["local_expert_offset"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
        hidden_size=H,
        intermediate_size=I,
        num_experts_global=E_GLOBAL,
        num_local_experts=E_LOCAL,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
    )

    # Prepare weights based on weight_processing configuration
    use_shuffled_weight = weight_processing["use_shuffled_weight"]
    weight_layout = weight_processing["layout"]

    gemm1_weights = inputs["gemm1_weights"]
    gemm2_weights = inputs["gemm2_weights"]

    if use_shuffled_weight:
        # Apply weight shuffling similar to the trtllm_gen_fused_moe test
        epilogue_tile_m = (
            64  # todo(yingyi): FIXME: this depends on the kernel internals
        )

        gemm1_weights_shuffled = []
        gemm2_weights_shuffled = []

        for i in range(E_LOCAL):
            # Shuffle weights for better performance
            tmp_weights1 = shuffle_matrix_a(
                gemm1_weights[i].view(torch.uint8), epilogue_tile_m
            )
            tmp_weights2 = shuffle_matrix_a(
                gemm2_weights[i].view(torch.uint8), epilogue_tile_m
            )

            if weight_layout == WeightLayout.BlockMajorK:
                block_k = 128
                tmp_weights1 = convert_to_block_layout(tmp_weights1, block_k)
                tmp_weights2 = convert_to_block_layout(tmp_weights2, block_k)

            gemm1_weights_shuffled.append(tmp_weights1)
            gemm2_weights_shuffled.append(tmp_weights2)

        gemm1_weights = torch.stack(gemm1_weights_shuffled).view(torch.float8_e4m3fn)
        gemm2_weights = torch.stack(gemm2_weights_shuffled).view(torch.float8_e4m3fn)

    # Run FlashInfer fused kernel
    with autotune(routing_config["enable_autotune"]):
        fi_out = trtllm_fp8_block_scale_moe(
            inputs["routing_logits"].to(torch.float32),
            inputs["routing_bias"],  # bf16
            inputs["hidden_states"],  # fp8
            inputs["hidden_states_scale"],  # [H/128, T]
            gemm1_weights,  # fp8 (potentially shuffled)
            inputs["gemm1_weights_scale"].to(torch.float32),
            gemm2_weights,  # fp8 (potentially shuffled)
            inputs["gemm2_weights_scale"].to(torch.float32),
            E_GLOBAL,
            TOP_K,
            N_GROUP,
            TOPK_GROUP,
            I,
            inputs["local_expert_offset"],
            inputs["local_num_experts"],
            inputs["routed_scaling_factor"],
            routing_method_type=2,  # DeepSeek-styled
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            enable_pdl=enable_pdl,
            tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
        )

    stats_accuracy(ref_out, fi_out, atol=atol, rtol=rtol, percent=percent)


if __name__ == "__main__":
    test_correctness_dpsk_fp8_fused_moe(
        seq_len=1,
        local_expert_offset=0,
        use_bias=False,
        intermediate_size=2048,
        routing_config={
            "num_experts": 256,
            "top_k": 8,
            "padding": 8,
            "n_groups": 8,
            "top_k_groups": 4,
            "routed_scaling": 2.5,
            "compatible_intermediate_size": [512, 1024, 2048],
            "enable_autotune": True,
        },
        enable_pdl=True,
        weight_processing={
            "use_shuffled_weight": False,
            "layout": WeightLayout.MajorK,
        },
    )
