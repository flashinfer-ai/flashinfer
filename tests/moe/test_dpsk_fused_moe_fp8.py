import pytest
import torch
import numpy as np
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe, WeightLayout
from flashinfer.autotuner import autotune


@torch.no_grad()
def run(
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
):
    """
    • FP8 block-scale dequantization: float ≈ fp8 * scale
    • DeepSeek-V3 no-aux routing:
        s = sigmoid(logits)
        s_with_bias = s + bias
        group by n_group=8; per group take top-2 sum → pick topk_group=4 groups
        on the kept groups, take global top_k=8 experts
        combine with weights derived from s (without bias), normalized and
        scaled by routed_scaling_factor
    • Local computation:
        only experts in [local_expert_offset, local_expert_offset + E_local) are
        computed on this rank (GEMM1 → SwiGLU → GEMM2), then per-token weighted
        accumulation.
    """

    # Fixed DeepSeek-V3/R1 geometry
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert H == 7168, "hidden_size must be 7168"
    assert I == 2048, "intermediate_size must be 2048"
    assert E_global == 256, "num_experts must be 256"
    assert E_local == 32, "num_local_experts must be 32"

    # Routing constants
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

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

    device = hidden_states.device

    # 1) FP8 block-scale dequantization
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
    w_fp8 = torch.empty_like(w_f32, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (*prefix, nb_r, nb_c), dtype=torch.float32, device=w_bf16.device
    )

    it = np.ndindex(*prefix) if prefix else [()]
    for idx in it:
        sel = idx if isinstance(idx, tuple) else (idx,)
        for i in range(nb_r):
            rs = slice(i * block, (i + 1) * block)
            for j in range(nb_c):
                cs = slice(j * block, (j + 1) * block)
                blk = w_f32[(*sel, rs, cs)]  # [128, 128]
                amax = torch.amax(torch.abs(blk))
                s = (
                    (amax / max_fp8)
                    if amax > 0
                    else torch.tensor(1.0, device=w_bf16.device)
                )
                q = (blk / s).to(torch.float8_e4m3fn)
                w_fp8[(*sel, rs, cs)] = q
                scales[(*sel, i, j)] = s
    return w_fp8, scales


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


def get_tile_tokens_dim(num_tokens, top_k, num_experts):
    # Guess tokens per expert assuming perfect expert distribution first.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    return tile_tokens_dim


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
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("enable_autotune", [True, False])
def test_correctness_dpsk_fp8_fused_moe(
    seq_len,
    *,
    local_expert_offset,
    use_bias,
    enable_pdl,
    enable_autotune,
    atol: float = 1e-1,
    rtol: float = 2e-1,
    percent: float = 0.85,
):
    print("\n" + "=" * 70)
    print(
        f"Testing MoE FP8 Block-Scale: seq_len={seq_len}, offset={local_expert_offset}, use_bias={use_bias}"
    )
    print("=" * 70)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping test.")
        return True

    if trtllm_fp8_block_scale_moe is None:
        print("WARNING: flashinfer fused_moe kernel not available.")
        return False

    device = "cuda"
    torch.manual_seed(42)

    # Constants (DeepSeek-V3)
    E_GLOBAL = 256
    E_LOCAL = 32
    H = 7168
    I = 2048
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    # Generate random but consistent inputs
    print("Generating random inputs")
    inputs = generate_random_inputs_moe(
        seq_len,
        num_experts_global=E_GLOBAL,
        num_local_experts=E_LOCAL,
        hidden_size=H,
        intermediate_size=I,
        use_bias=use_bias,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=2.5,
        device=device,
    )

    # Run reference (returns bf16)
    print("Running reference")
    ref_out = run(
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
    )

    # Run FlashInfer fused kernel
    print("Running FlashInfer fused kernel")
    tile_tokens_dim = get_tile_tokens_dim(seq_len, TOP_K, E_GLOBAL)
    with autotune(enable_autotune):
        fi_out = trtllm_fp8_block_scale_moe(
            inputs["routing_logits"].to(torch.float32),
            inputs["routing_bias"],  # bf16
            inputs["hidden_states"],  # fp8
            inputs["hidden_states_scale"],  # [H/128, T]
            inputs["gemm1_weights"],  # fp8
            inputs["gemm1_weights_scale"].to(torch.float32),
            inputs["gemm2_weights"],  # fp8
            inputs["gemm2_weights_scale"].to(torch.float32),
            E_GLOBAL,
            TOP_K,
            N_GROUP,
            TOPK_GROUP,
            I,
            inputs["local_expert_offset"],
            inputs["local_num_experts"],
            inputs["routed_scaling_factor"],
            tile_tokens_dim=tile_tokens_dim,
            routing_method_type=2,  # DeepSeek-styled
            use_shuffled_weight=False,
            weight_layout=WeightLayout.BlockMajorK.value,
            enable_pdl=enable_pdl,
            tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
        )

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


if __name__ == "__main__":
    test_correctness_dpsk_fp8_fused_moe(
        seq_len=1,
        local_expert_offset=0,
        use_bias=False,
        enable_pdl=True,
        enable_autotune=True,
    )
