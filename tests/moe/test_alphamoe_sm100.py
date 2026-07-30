"""Tests for alphamoe_sm100 — the generated SM100/SM103 fused W8A8 MoE kernel.

The frozen device TU in ``csrc/alphamoe_sm100.cu`` is a generated Loom
schedule of the Alpha-MoE up+SwiGLU+down megakernel. The torch reference
reproduced below rounds each routed per-128-block down contribution to BF16
before accumulating, exactly like the kernel's
``cp.reduce.async.bulk .add.noftz.bf16`` output path — the two computations
differ ONLY in accumulation order: the reference adds in a fixed
(expert, intermediate-block) order while the kernel's cross-CTA reduce-adds
land in hardware scheduling order, which is nondeterministic run to run.
BF16 addition is commutative but not associative, so:

- cases where every output element receives at most TWO contributions into a
  zero accumulator are order-insensitive and are asserted with
  ``torch.equal`` (any hardware order gives the same bits);
- all other cases are asserted at the FP8-tier contract tolerance
  (``atol=0.1, rtol=0.1``) widened per element by an accumulation-order
  bound of 2 ulp of the accumulated |contribution| mass (see
  ``_assert_order_tolerant``). The widening matters only at near-zero
  outputs produced by catastrophic cancellation of O(10) partials, where a
  legitimate order permutation moves the result by ~1 ulp of the PARTIAL
  magnitude (~0.11 at partial mass 16-32) and grazes the plain contract
  bound with ~10-20% probability per full run — reproduced by a pure-torch
  reorder of the reference's own contributions, i.e. by any order-legal
  implementation. Do not tighten back to ``torch.equal``: it only passes
  when the CTAs happen to complete in launch order.

Contributions per output element = ``top_k * (N // 256)`` (one per routed
expert per 128-wide intermediate block).
"""

import pytest
import torch

from flashinfer.utils import is_sm100a_supported

_FP8_MAX = 448.0
_GROUP = 128


def _skip_if_not_sm100_family():
    if not torch.cuda.is_available():
        pytest.skip("alphamoe_sm100 tests require a CUDA device")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("alphamoe_sm100 requires SM100/SM103")


def _quantize_per_row_group(values):
    """Per-row FP8 E4M3 quantization in groups of 128 columns."""
    rows, columns = values.shape
    groups = values.float().reshape(rows, columns // _GROUP, _GROUP)
    scales = groups.abs().amax(dim=-1).clamp_min(1.0e-8) / _FP8_MAX
    quantized = (groups / scales.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    return (
        quantized.to(torch.float8_e4m3fn).reshape(rows, columns).contiguous(),
        scales.contiguous(),
    )


def _quantize_block_2d(values):
    """Per-expert 128x128 block FP8 E4M3 quantization."""
    experts, rows, columns = values.shape
    groups = values.float().reshape(
        experts, rows // _GROUP, _GROUP, columns // _GROUP, _GROUP
    )
    scales = groups.abs().amax(dim=(2, 4)).clamp_min(1.0e-8) / _FP8_MAX
    quantized = (groups / scales[:, :, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX)
    return (
        quantized.to(torch.float8_e4m3fn).reshape(experts, rows, columns).contiguous(),
        scales.contiguous(),
    )


def _expand_block_scales(scales):
    return scales.repeat_interleave(_GROUP, dim=1).repeat_interleave(_GROUP, dim=2)


def _make_aligned_routing_plan(topk_ids, *, block_m, num_experts):
    """vLLM/SGLang ``moe_align_block_size`` semantics, in plain torch.

    Returns worst-case-sized buffers. Slots past the valid plan extent are
    filled with pair 0 / expert 0 — *valid-looking* garbage, so a kernel that
    ignores ``num_tokens_post_padded`` corrupts token 0 visibly instead of
    faulting; the parity checks then catch it.
    """
    m, top_k = topk_ids.shape
    sentinel = m * top_k
    flat = topk_ids.reshape(-1).to(device="cpu", dtype=torch.int64)
    sorted_positions, block_experts = [], []
    for expert in range(num_experts):
        positions = torch.nonzero(flat == expert, as_tuple=False).flatten().tolist()
        if not positions:
            continue
        padded = ((len(positions) + block_m - 1) // block_m) * block_m
        sorted_positions.extend(positions + [sentinel] * (padded - len(positions)))
        block_experts.extend([expert] * (padded // block_m))
    max_blocks = (sentinel + num_experts * (block_m - 1) + block_m - 1) // block_m
    max_blocks = max(max_blocks, len(block_experts))
    device = topk_ids.device
    sorted_token_ids = torch.zeros(
        max_blocks * block_m, dtype=torch.int32, device=device
    )
    sorted_token_ids[: len(sorted_positions)] = torch.tensor(
        sorted_positions, dtype=torch.int32, device=device
    )
    expert_ids = torch.zeros(max_blocks, dtype=torch.int32, device=device)
    expert_ids[: len(block_experts)] = torch.tensor(
        block_experts, dtype=torch.int32, device=device
    )
    num_tokens_post_padded = torch.tensor(
        [len(sorted_positions)], dtype=torch.int32, device=device
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# Six of the twelve internal contract rows: M=1 decode, the DeepSeek-R1
# coordinate (E=257, top_k=9, shared expert), the Qwen serving coordinate
# (E=512, top_k=10, scaling 1.0), an unbalanced hot-expert row, a ragged
# M=17 tail, and the N=1024 wide-intermediate row.
# (label, M, N, K, E, top_k, block_m, shared_expert, balancedness, scaling, seed)
CONTRACT_CASES = [
    ("decode_m1_e4_k256_top2", 1, 256, 256, 4, 2, 8, False, 1.0, 2.5, 28007),
    ("tail_m17_e8_k512_top3", 17, 256, 512, 8, 3, 8, False, 0.8, 2.5, 28002),
    (
        "source_decode_m8_e257_k7168_top9",
        8,
        256,
        7168,
        257,
        9,
        8,
        True,
        0.8,
        2.5,
        28003,
    ),
    (
        "qwen_serving_m8_e512_k2048_top10_scale1",
        8,
        256,
        2048,
        512,
        10,
        8,
        False,
        0.8,
        1.0,
        28009,
    ),
    ("hot_expert_m64_e8_k512_top3_bal0", 64, 256, 512, 8, 3, 8, False, 0.0, 2.5, 28010),
    ("width_n1024_m8_e4_k256_top2", 8, 1024, 256, 4, 2, 8, False, 1.0, 2.5, 28011),
]


def _make_case(
    m,
    n,
    k,
    num_experts,
    top_k,
    block_m,
    shared_expert,
    balancedness,
    scaling_factor,
    seed,
):
    from flashinfer.fused_moe import alphamoe_interleave_gated_weights

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden = (
        torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=generator)
        * 0.25
    )
    w1_logical = (
        torch.randn(
            (num_experts, n, k),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.125
    )
    w2_logical = (
        torch.randn(
            (num_experts, k, n // 2),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.125
    )
    x, x_scale = _quantize_per_row_group(hidden)
    w1, w1_scale = _quantize_block_2d(w1_logical)
    w2, w2_scale = _quantize_block_2d(w2_logical)
    w1_dev, w1_scale_dev = alphamoe_interleave_gated_weights(w1, w1_scale)

    routed_top_k = top_k - int(shared_expert)
    routed_experts = num_experts - int(shared_expert)
    scores = torch.randn(
        (m, routed_experts), dtype=torch.float32, device=device, generator=generator
    )
    scores[:, 0] += (1.0 - balancedness) * 6.0
    topk_ids = torch.topk(scores, routed_top_k, dim=-1).indices.to(torch.int32)
    if shared_expert:
        shared = torch.full((m, 1), num_experts - 1, dtype=torch.int32, device=device)
        topk_ids = torch.cat((topk_ids, shared), dim=-1)
    topk_weights = torch.softmax(
        torch.randn(
            (m, top_k), dtype=torch.float32, device=device, generator=generator
        ),
        dim=-1,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = _make_aligned_routing_plan(
        topk_ids, block_m=block_m, num_experts=num_experts
    )
    return {
        "M": m,
        "N": n,
        "K": k,
        "E": num_experts,
        "top_k": top_k,
        "block_m": block_m,
        "scaling_factor": scaling_factor,
        "x": x,
        "x_scale": x_scale,
        "w1": w1,  # logical (non-interleaved) quantized weights for the reference
        "w1_scale": w1_scale,
        "w2": w2,
        "w2_scale": w2_scale,
        "w1_dev": w1_dev,  # interleaved device layout for the kernel
        "w1_scale_dev": w1_scale_dev,
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_post_padded": num_tokens_post_padded,
        "topk_weights": topk_weights,
    }


def _reference(case, *, out_init=None, plan_extent=None, return_abs_sum=False):
    """Independent torch oracle (ported from the internal contract harness).

    Dequantized routed expert math with the kernel's source-visible rounding:
    the intermediate is requantized to FP8 per token in groups of 128, each
    128-wide intermediate block's down contribution rounds to BF16 before the
    accumulator add, and every accumulator update rounds to BF16.
    """
    m, k, top_k = case["M"], case["K"], case["top_k"]
    intermediate = case["N"] // 2
    block_m = case["block_m"]

    x = case["x"].float() * case["x_scale"].repeat_interleave(_GROUP, dim=1)
    w1 = case["w1"].float() * _expand_block_scales(case["w1_scale"])
    w2 = case["w2"].float() * _expand_block_scales(case["w2_scale"])

    # Pair -> expert map from the plan, honoring the valid extent; pairs not
    # covered by the (possibly truncated) plan contribute nothing.
    pair_expert = torch.full((m * top_k,), -1, dtype=torch.int64, device=x.device)
    extent = (
        int(case["num_tokens_post_padded"].item())
        if plan_extent is None
        else plan_extent
    )
    sorted_ids = case["sorted_token_ids"][:extent].to(torch.int64)
    block_experts = case["expert_ids"].to(torch.int64)
    for block in range(extent // block_m):
        pairs = sorted_ids[block * block_m : (block + 1) * block_m]
        valid = pairs < m * top_k
        pair_expert[pairs[valid]] = block_experts[block]

    output = (
        torch.zeros((m, k), dtype=torch.bfloat16, device=x.device)
        if out_init is None
        else out_init.clone()
    )
    abs_sum = torch.zeros((m, k), dtype=torch.float32, device=x.device)
    flat_weights = case["topk_weights"].reshape(-1).float()
    for expert in range(case["E"]):
        pair_indices = torch.nonzero(pair_expert == expert, as_tuple=False).flatten()
        if pair_indices.numel() == 0:
            continue
        token_indices = torch.div(pair_indices, top_k, rounding_mode="floor")
        gate_up = x[token_indices] @ w1[expert].transpose(0, 1)
        gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
        activated = torch.nn.functional.silu(gate) * up
        act_q, act_scale = _quantize_per_row_group(activated)
        activated_dequant = act_q.float() * act_scale.repeat_interleave(_GROUP, dim=1)
        for base in range(0, intermediate, _GROUP):
            down = activated_dequant[:, base : base + _GROUP] @ w2[
                expert, :, base : base + _GROUP
            ].transpose(0, 1)
            down *= flat_weights[pair_indices, None] * case["scaling_factor"]
            routed_bf16 = down.to(torch.bfloat16)
            output[token_indices] = (
                output[token_indices].float() + routed_bf16.float()
            ).to(torch.bfloat16)
            abs_sum[token_indices] += routed_bf16.float().abs()
    return (output, abs_sum) if return_abs_sum else output


def _assert_order_tolerant(out, expected, abs_sum, label):
    """Contract tolerance widened by an accumulation-order bound near zero.

    The kernel and the reference differ only in BF16 accumulation order, so a
    near-zero output element (catastrophic cancellation of O(abs_sum)
    partials) legitimately moves by a few ulp of the PARTIAL-SUM magnitude —
    which can exceed ``atol + rtol*|ref|`` when |ref| is tiny. Bound each
    element by the larger of the FP8-tier contract tolerance and 2 ulp of the
    accumulated |contribution| mass (ulp_bf16(x) <= x * 2**-7).
    """
    diff = (out.float() - expected.float()).abs()
    tol = torch.maximum(
        0.1 + 0.1 * expected.float().abs(),
        2.0 * abs_sum * 2.0**-7,
    )
    bad = diff > tol
    assert not bool(bad.any()), (
        f"{label}: {int(bad.sum())} elements exceed the order-aware bound; "
        f"worst diff={diff[bad].max().item():.6f} vs tol={tol[bad].min().item():.6f}"
    )


def _launch(case, *, out=None, num_tokens_post_padded=None):
    from flashinfer.fused_moe import alphamoe_fp8_block_scale_aligned_moe

    return alphamoe_fp8_block_scale_aligned_moe(
        case["x"],
        case["x_scale"],
        case["w1_dev"],
        case["w1_scale_dev"],
        case["w2"],
        case["w2_scale"],
        case["sorted_token_ids"],
        case["expert_ids"],
        (
            case["num_tokens_post_padded"]
            if num_tokens_post_padded is None
            else num_tokens_post_padded
        ),
        case["topk_weights"],
        top_k=case["top_k"],
        block_m=case["block_m"],
        routed_scaling_factor=case["scaling_factor"],
        out=out,
    )


@pytest.mark.parametrize(
    "label,m,n,k,num_experts,top_k,block_m,shared_expert,balancedness,scaling_factor,seed",
    CONTRACT_CASES,
    ids=[case[0] for case in CONTRACT_CASES],
)
def test_alphamoe_sm100_matches_reference(
    label,
    m,
    n,
    k,
    num_experts,
    top_k,
    block_m,
    shared_expert,
    balancedness,
    scaling_factor,
    seed,
):
    """End-to-end parity against the independent torch oracle (out=None path)."""
    _skip_if_not_sm100_family()
    case = _make_case(
        m,
        n,
        k,
        num_experts,
        top_k,
        block_m,
        shared_expert,
        balancedness,
        scaling_factor,
        seed,
    )
    out = _launch(case)
    torch.cuda.synchronize()
    expected, abs_sum = _reference(case, return_abs_sum=True)
    contributions_per_element = top_k * (n // 256)
    if contributions_per_element <= 2:
        # Two BF16 addends into a zero accumulator are order-insensitive.
        assert torch.equal(out, expected), (
            f"mismatch vs torch reference at {label}: "
            f"{(out != expected).sum().item()} differing elements, "
            f"max |diff|={(out.float() - expected.float()).abs().max().item()}"
        )
    else:
        # 3+ addends: kernel and reference differ only by accumulation order;
        # assert the contract tolerance widened by the near-zero order bound.
        _assert_order_tolerant(out, expected, abs_sum, label)


def test_alphamoe_sm100_guard_skips_blocks_past_plan_extent():
    """Blocks past num_tokens_post_padded must not touch out.

    The plan buffers are worst-case sized and garbage-filled past the valid
    extent (pair 0 / expert 0), so any block ignoring the guard adds a
    visible spurious contribution to token 0.
    """
    _skip_if_not_sm100_family()
    label, m, n, k, num_experts, top_k, block_m, shared, bal, scaling, seed = (
        CONTRACT_CASES[1]
    )
    case = _make_case(m, n, k, num_experts, top_k, block_m, shared, bal, scaling, seed)

    # Extent 0: with every block guarded off, a non-zero seeded out is
    # untouched even though the full worst-case grid was launched.
    zero_extent = torch.zeros((1,), dtype=torch.int32, device="cuda")
    seeded = torch.full((m, k), 3.0, dtype=torch.bfloat16, device="cuda")
    out = _launch(case, out=seeded.clone(), num_tokens_post_padded=zero_extent)
    torch.cuda.synchronize()
    assert torch.equal(out, seeded), "blocks past the plan extent wrote to out"

    # Truncated extent: only the first block's pairs may contribute. Block 0
    # holds one expert's pairs, so each token contributes at most once and
    # N=256 has a single intermediate block — order-insensitive, exact.
    truncated = torch.tensor([block_m], dtype=torch.int32, device="cuda")
    out = _launch(case, num_tokens_post_padded=truncated)
    torch.cuda.synchronize()
    expected = _reference(case, plan_extent=block_m)
    assert torch.equal(out, expected), "truncated plan extent was not honored"


def test_alphamoe_sm100_accumulates_into_out():
    """out is a caller-owned accumulator: result = initial value + contributions."""
    _skip_if_not_sm100_family()
    label, m, n, k, num_experts, top_k, block_m, shared, bal, scaling, seed = (
        CONTRACT_CASES[0]
    )
    case = _make_case(m, n, k, num_experts, top_k, block_m, shared, bal, scaling, seed)

    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    init = torch.randn(
        (m, k), dtype=torch.float32, device="cuda", generator=generator
    ).to(torch.bfloat16)
    out = _launch(case, out=init.clone())
    torch.cuda.synchronize()
    expected, abs_sum = _reference(case, out_init=init, return_abs_sum=True)
    # Reduce-adds into a NON-zero base round sequentially, so the result is
    # order-sensitive — assert the order-aware bound, and separately prove
    # the initial value was accumulated (not overwritten) by checking the
    # contributions-only result is NOT within tolerance.
    _assert_order_tolerant(out, expected, abs_sum, "accumulate")
    contributions_only = _reference(case)
    assert not torch.allclose(
        out.float(), contributions_only.float(), atol=0.1, rtol=0.1
    ), "out looks overwritten: initial accumulator value is missing"
