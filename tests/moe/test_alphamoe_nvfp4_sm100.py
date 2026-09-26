"""Correctness and contract tests for the frozen AlphaMoE NVFP4 kernel."""

import pytest
import torch
from flashinfer.fused_moe.alphamoe_nvfp4_sm100 import prepare_nvfp4_w2_data


_SF_VEC = 16
_INTERMEDIATE_TILE = 128
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _skip_if_not_supported() -> None:
    if not torch.cuda.is_available():
        pytest.skip("AlphaMoE NVFP4 tests require CUDA")
    capability = torch.cuda.get_device_capability()
    if capability not in {(10, 0), (10, 3)}:
        pytest.skip(f"AlphaMoE NVFP4 requires exact SM100/SM103, got {capability}")


def _make_aligned_routing_plan(
    topk_ids: torch.Tensor,
    *,
    block_m: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build worst-case-capacity vLLM/SGLang aligned-plan buffers."""

    m, top_k = topk_ids.shape
    sentinel = m * top_k
    flat_ids = topk_ids.reshape(-1).to(device="cpu", dtype=torch.int64)
    sorted_positions: list[int] = []
    block_experts: list[int] = []
    for expert in range(num_experts):
        positions = torch.nonzero(flat_ids == expert, as_tuple=False).flatten().tolist()
        if not positions:
            continue
        padded = ((len(positions) + block_m - 1) // block_m) * block_m
        sorted_positions.extend(positions)
        sorted_positions.extend([sentinel] * (padded - len(positions)))
        block_experts.extend([expert] * (padded // block_m))

    nonempty_experts = min(num_experts, sentinel)
    max_blocks = nonempty_experts + (sentinel - nonempty_experts) // block_m
    max_blocks = max(max_blocks, len(block_experts))
    device = topk_ids.device
    # Fill the unused capacity with valid-looking garbage. The device-side
    # extent guard, rather than an invalid index fault, must make it inert.
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


def _random_nvfp4(
    leading_shape: tuple[int, ...],
    columns: int,
    *,
    generator: torch.Generator,
    min_scale_exp: float,
    max_scale_exp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = torch.randint(
        0,
        256,
        (*leading_shape, columns // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    scale_exp = torch.rand(
        (*leading_shape, columns // _SF_VEC),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    scale = torch.exp2(min_scale_exp + scale_exp * (max_scale_exp - min_scale_exp)).to(
        torch.float8_e4m3fn
    )
    return packed, scale.contiguous()


def _make_case(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    top_k: int,
    block_m: int,
    balancedness: float,
    scaling_factor: float,
    seed: int,
) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x, x_scale = _random_nvfp4(
        (m,),
        k,
        generator=generator,
        min_scale_exp=-3.0,
        max_scale_exp=-2.0,
    )
    w1, w1_scale = _random_nvfp4(
        (num_experts, n),
        k,
        generator=generator,
        min_scale_exp=-5.0,
        max_scale_exp=-4.0,
    )
    w2, w2_scale = _random_nvfp4(
        (num_experts, k),
        n // 2,
        generator=generator,
        min_scale_exp=-5.0,
        max_scale_exp=-4.0,
    )
    scores = torch.randn(
        (m, num_experts),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    scores[:, 0] += (1.0 - balancedness) * 6.0
    topk_ids = torch.topk(scores, top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(
            (m, top_k),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        ),
        dim=-1,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = _make_aligned_routing_plan(
        topk_ids,
        block_m=block_m,
        num_experts=num_experts,
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
        "w1": w1,
        "w1_scale": w1_scale,
        "w2": w2,
        "w2_scale": w2_scale,
        "output1_scale_gate_scalar": torch.ones(
            num_experts, dtype=torch.float32, device="cuda"
        ),
        "output1_scale_scalar": torch.ones(
            num_experts, dtype=torch.float32, device="cuda"
        ),
        "output2_scale_scalar": torch.ones(
            num_experts, dtype=torch.float32, device="cuda"
        ),
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_post_padded": num_tokens_post_padded,
        "topk_weights": topk_weights,
        "out": torch.zeros((m, k), dtype=torch.bfloat16, device="cuda"),
    }


def _decode_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    columns: int,
) -> torch.Tensor:
    packed_u8 = packed.view(torch.uint8)
    codes = torch.empty(
        (*packed_u8.shape[:-1], columns), dtype=torch.uint8, device=packed.device
    )
    codes[..., 0::2] = packed_u8 & 0x0F
    codes[..., 1::2] = (packed_u8 >> 4) & 0x0F
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    values = lut[codes.to(torch.int64)]
    expanded_scale = scale.float().repeat_interleave(_SF_VEC, dim=-1)[..., :columns]
    return values * expanded_scale


def _quantize_nvfp4_fused_reference(values: torch.Tensor) -> torch.Tensor:
    """Model the fused FP32 SwiGLU -> E4M3 scale -> E2M1 epilogue."""

    rows, columns = values.shape
    blocks = values.reshape(rows, columns // _SF_VEC, _SF_VEC)
    scale = (blocks.abs().amax(dim=-1) * (1.0 / 6.0)).to(torch.float8_e4m3fn).float()
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    normalized = blocks / safe_scale[..., None]
    levels = torch.tensor(
        (
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ),
        dtype=torch.float32,
        device=values.device,
    )
    nearest = (normalized[..., None] - levels).abs().argmin(dim=-1)
    return (levels[nearest] * scale[..., None]).reshape(rows, columns)


def _pair_experts(case: dict, *, plan_extent: int | None = None) -> torch.Tensor:
    m = case["M"]
    top_k = case["top_k"]
    block_m = case["block_m"]
    pair_expert = torch.full(
        (m * top_k,), -1, dtype=torch.int64, device=case["x"].device
    )
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
    return pair_expert


def _reference(
    case: dict,
    *,
    out_init: torch.Tensor | None = None,
    plan_extent: int | None = None,
) -> torch.Tensor:
    """Independent source-ordered NVFP4 oracle for the exported schedule."""

    m, n, k = case["M"], case["N"], case["K"]
    top_k = case["top_k"]
    intermediate = n // 2
    x = _decode_nvfp4(case["x"], case["x_scale"], columns=k)
    pair_expert = _pair_experts(case, plan_extent=plan_extent)
    flat_weights = case["topk_weights"].reshape(-1).float()
    output = (
        torch.zeros((m, k), dtype=torch.bfloat16, device=x.device)
        if out_init is None
        else out_init.clone()
    )

    for expert in range(case["E"]):
        pair_indices = torch.nonzero(pair_expert == expert, as_tuple=False).flatten()
        if pair_indices.numel() == 0:
            continue
        token_indices = torch.div(pair_indices, top_k, rounding_mode="floor")
        w1 = _decode_nvfp4(case["w1"][expert], case["w1_scale"][expert], columns=k)
        gate_up = torch.zeros(
            (token_indices.numel(), n), dtype=torch.float32, device=x.device
        )
        for k_base in range(0, k, 64):
            gate_up += x[token_indices, k_base : k_base + 64] @ w1[
                :, k_base : k_base + 64
            ].transpose(0, 1)
        gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
        activated = torch.nn.functional.silu(gate) * up
        activated_dequant = _quantize_nvfp4_fused_reference(activated)
        w2 = _decode_nvfp4(
            case["w2"][expert], case["w2_scale"][expert], columns=intermediate
        )

        for intermediate_base in range(0, intermediate, _INTERMEDIATE_TILE):
            down = torch.zeros(
                (token_indices.numel(), k), dtype=torch.float32, device=x.device
            )
            for k_base in range(
                intermediate_base, intermediate_base + _INTERMEDIATE_TILE, 64
            ):
                down += activated_dequant[:, k_base : k_base + 64] @ w2[
                    :, k_base : k_base + 64
                ].transpose(0, 1)
            down *= flat_weights[pair_indices, None] * case["scaling_factor"]
            routed_bf16 = down.to(torch.bfloat16)
            output[token_indices] = (
                output[token_indices].float() + routed_bf16.float()
            ).to(torch.bfloat16)
    return output


def _assert_nvfp4_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.1)
    except AssertionError as error:
        raise AssertionError(f"{label}: AlphaMoE NVFP4 contract mismatch") from error


def _launch(
    case: dict,
    *,
    out: torch.Tensor | None = None,
    num_tokens_post_padded: torch.Tensor | None = None,
) -> torch.Tensor:
    from flashinfer.fused_moe import alphamoe_nvfp4_aligned_moe

    output = case["out"] if out is None else out
    result = alphamoe_nvfp4_aligned_moe(
        case["x"],
        case["x_scale"],
        case["w1"],
        case["w1_scale"],
        case["w2"],
        case["w2_scale"],
        case["output1_scale_gate_scalar"],
        case["output1_scale_scalar"],
        case["output2_scale_scalar"],
        case["sorted_token_ids"],
        case["expert_ids"],
        (
            case["num_tokens_post_padded"]
            if num_tokens_post_padded is None
            else num_tokens_post_padded
        ),
        case["topk_weights"],
        output,
        case["top_k"],
        case["block_m"],
        case["scaling_factor"],
    )
    assert result is None
    return output


# The five correctness-bearing rows in the originating Loom contract cover
# smoke, N tiling, route tails, and the K=7168 serving coordinate.
_CONTRACT_CASES = [
    ("smoke_m8_n256_e4_k256_top2", 8, 256, 256, 4, 2, 8, 1.0, 2.5, 28101),
    ("width_m8_n512_e4_k256_top2", 8, 512, 256, 4, 2, 8, 1.0, 2.5, 28102),
    ("width_m8_n1024_e4_k256_top2", 8, 1024, 256, 4, 2, 8, 1.0, 2.5, 28107),
    ("tail_m17_n256_e8_k512_top3", 17, 256, 512, 8, 3, 8, 0.8, 2.5, 28103),
    ("decode_m8_n256_e256_k7168_top8", 8, 256, 7168, 256, 8, 8, 0.8, 2.5, 28104),
]


@pytest.mark.parametrize(
    "label,m,n,k,num_experts,top_k,block_m,balancedness,scaling_factor,seed",
    _CONTRACT_CASES,
    ids=[case[0] for case in _CONTRACT_CASES],
)
def test_alphamoe_nvfp4_matches_reference(
    label,
    m,
    n,
    k,
    num_experts,
    top_k,
    block_m,
    balancedness,
    scaling_factor,
    seed,
):
    _skip_if_not_supported()
    case = _make_case(
        m,
        n,
        k,
        num_experts,
        top_k,
        block_m,
        balancedness,
        scaling_factor,
        seed,
    )
    actual = _launch(case)
    torch.cuda.synchronize()
    expected = _reference(case)
    _assert_nvfp4_close(actual, expected, label)


def test_alphamoe_nvfp4_plan_extent_and_preallocated_accumulator():
    _skip_if_not_supported()
    case = _make_case(17, 256, 512, 8, 3, 8, 0.8, 2.5, 28103)

    zero_extent = torch.zeros((1,), dtype=torch.int32, device="cuda")
    seeded = torch.full((17, 512), 3.0, dtype=torch.bfloat16, device="cuda")
    actual = _launch(
        case,
        out=seeded.clone(),
        num_tokens_post_padded=zero_extent,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, seeded), "guarded capacity blocks changed out"

    truncated = torch.tensor([case["block_m"]], dtype=torch.int32, device="cuda")
    truncated_out = torch.zeros_like(case["out"])
    actual = _launch(
        case,
        out=truncated_out,
        num_tokens_post_padded=truncated,
    )
    torch.cuda.synchronize()
    expected = _reference(case, plan_extent=case["block_m"])
    assert torch.equal(actual, expected), "truncated plan extent was not honored"

    generator = torch.Generator(device="cuda").manual_seed(28130)
    initial = torch.randn(
        (17, 512), dtype=torch.float32, device="cuda", generator=generator
    ).to(torch.bfloat16)
    actual = _launch(case, out=initial.clone())
    torch.cuda.synchronize()
    expected = _reference(case, out_init=initial)
    _assert_nvfp4_close(actual, expected, "seeded accumulator")


def test_alphamoe_nvfp4_accepts_nonoverlapping_row_strided_hidden_states():
    _skip_if_not_supported()
    case = _make_case(8, 256, 256, 4, 2, 8, 1.0, 2.5, 28131)
    packed_k = case["x"].shape[1]
    storage = torch.empty((case["M"], packed_k + 16), dtype=torch.uint8, device="cuda")
    storage[:, :packed_k].copy_(case["x"])
    case["x"] = storage[:, :packed_k]
    assert not case["x"].is_contiguous()
    assert case["x"].stride(0) % 16 == 0

    actual = _launch(case)
    torch.cuda.synchronize()
    expected = _reference(case)
    _assert_nvfp4_close(actual, expected, "row-strided hidden")


def test_alphamoe_nvfp4_rejects_invalid_host_contracts():
    _skip_if_not_supported()
    case = _make_case(8, 256, 256, 4, 2, 8, 1.0, 2.5, 28132)
    from flashinfer.fused_moe import alphamoe_nvfp4_aligned_moe

    args = [
        case["x"],
        case["x_scale"],
        case["w1"],
        case["w1_scale"],
        case["w2"],
        case["w2_scale"],
        case["output1_scale_gate_scalar"],
        case["output1_scale_scalar"],
        case["output2_scale_scalar"],
        case["sorted_token_ids"],
        case["expert_ids"],
        case["num_tokens_post_padded"],
        case["topk_weights"],
        case["out"],
        case["top_k"],
        case["block_m"],
        case["scaling_factor"],
    ]

    wrong_scale_dtype = list(args)
    wrong_scale_dtype[1] = case["x_scale"].float()
    with pytest.raises(ValueError, match="hidden_states_scale must have dtype"):
        alphamoe_nvfp4_aligned_moe(*wrong_scale_dtype)

    too_short_plan = list(args)
    too_short_plan[9] = case["sorted_token_ids"][:-1]
    with pytest.raises(ValueError, match="capacity must be at least"):
        alphamoe_nvfp4_aligned_moe(*too_short_plan)

    wrong_scale_layout = list(args)
    wrong_scale_layout[1] = case["x_scale"].reshape(-1)
    with pytest.raises(ValueError, match="must be 2D"):
        alphamoe_nvfp4_aligned_moe(*wrong_scale_layout)

    overlapping_storage = torch.empty(256, dtype=torch.uint8, device="cuda")
    overlapping = torch.as_strided(overlapping_storage, (8, 128), (16, 1))
    overlapping_args = list(args)
    overlapping_args[0] = overlapping
    with pytest.raises(ValueError, match="non-overlapping"):
        alphamoe_nvfp4_aligned_moe(*overlapping_args)

    misaligned_storage = torch.empty(
        case["M"] * case["K"] + 1, dtype=torch.bfloat16, device="cuda"
    )
    misaligned_out = misaligned_storage[1:].view(case["M"], case["K"])
    assert misaligned_out.is_contiguous()
    assert misaligned_out.data_ptr() % 16 != 0
    misaligned_args = list(args)
    misaligned_args[13] = misaligned_out
    with pytest.raises(ValueError, match="16-byte aligned"):
        alphamoe_nvfp4_aligned_moe(*misaligned_args)

    misaligned_w1_storage = torch.empty(
        case["w1"].numel() + 1, dtype=torch.uint8, device="cuda"
    )
    misaligned_w1 = misaligned_w1_storage[1:].view_as(case["w1"])
    assert misaligned_w1.is_contiguous()
    assert misaligned_w1.data_ptr() % 16 != 0
    misaligned_w1_args = list(args)
    misaligned_w1_args[2] = misaligned_w1
    with pytest.raises(ValueError, match=r"gemm1_weights.*16-byte aligned"):
        alphamoe_nvfp4_aligned_moe(*misaligned_w1_args)

    aliased_storage = torch.empty(
        case["M"] * case["K"] * 2, dtype=torch.uint8, device="cuda"
    )
    aliased_x = aliased_storage[: case["M"] * (case["K"] // 2)].view(
        case["M"], case["K"] // 2
    )
    aliased_out = aliased_storage.view(torch.bfloat16).view(case["M"], case["K"])
    alias_args = list(args)
    alias_args[0] = aliased_x
    alias_args[13] = aliased_out
    with pytest.raises(RuntimeError, match="out must not overlap hidden_states"):
        alphamoe_nvfp4_aligned_moe(*alias_args)

    k128 = _make_case(8, 256, 128, 4, 2, 8, 1.0, 2.5, 28133)
    with pytest.raises(ValueError, match="must be at least 256"):
        _launch(k128)


# ---------------------------------------------------------------------------
# Deferred finalize (complete routes): identical arithmetic, BF16 caller seed.
# ---------------------------------------------------------------------------

_COMPLETE_E, _COMPLETE_N, _COMPLETE_K, _COMPLETE_TOP_K, _COMPLETE_BLOCK_M = 256, 1024, 6144, 8, 8


def _complete_route_case(m: int, seed: int, *, prepared_data: bool):
    from flashinfer.fused_moe import alphamoe_nvfp4_sm100 as api

    generator = torch.Generator(device="cuda").manual_seed(seed)
    device = torch.device("cuda")

    def u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=device, generator=generator)

    def e4m3(*shape):
        return (torch.rand(shape, device=device, generator=generator) * 0.75 + 0.25).to(torch.float8_e4m3fn)

    x = u8(m, _COMPLETE_K // 2)
    x_scale = e4m3(m, _COMPLETE_K // 16)
    w1 = u8(_COMPLETE_E, _COMPLETE_N, _COMPLETE_K // 2)
    w1_scale = e4m3(_COMPLETE_E, _COMPLETE_N, _COMPLETE_K // 16)
    w2 = u8(_COMPLETE_E, _COMPLETE_K, _COMPLETE_N // 4)
    w2_scale = e4m3(_COMPLETE_E, _COMPLETE_K, _COMPLETE_N // 32)
    scalar = lambda: (torch.rand((_COMPLETE_E,), device=device, generator=generator) * 0.5 + 0.5)
    logits = torch.rand((m, _COMPLETE_E), device=device, generator=generator)
    topk_weights, topk_ids = torch.topk(logits, _COMPLETE_TOP_K, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).float().contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    pairs = m * _COMPLETE_TOP_K
    raw_capacity = pairs * _COMPLETE_BLOCK_M if pairs < _COMPLETE_E + 1 else pairs + (_COMPLETE_E + 1) * (_COMPLETE_BLOCK_M - 1)
    expert_capacity = (raw_capacity + _COMPLETE_BLOCK_M - 1) // _COMPLETE_BLOCK_M
    case = dict(
        hidden_states=x, hidden_states_scale=x_scale, gemm1_weights=w1, gemm1_weights_scale=w1_scale,
        gemm2_weights=w2, gemm2_weights_scale=w2_scale, output1_scale_gate_scalar=scalar(),
        output1_scale_scalar=scalar(), output2_scale_scalar=scalar(), topk_weights=topk_weights, topk_ids=topk_ids,
        top_k=_COMPLETE_TOP_K, block_m=_COMPLETE_BLOCK_M, routed_scaling_factor=1.0,
        w1_scale_prepared=api.prepare_nvfp4_w1_scales(w1_scale), w2_scale_prepared=api.prepare_nvfp4_w2_scales(w2_scale),
        w1_data_prepared=api.prepare_nvfp4_w1_data(w1) if prepared_data else None,
        w2_data_prepared=api.prepare_nvfp4_w2_data(w2) if prepared_data else None,
        w1_scale_prepared_interleaved=api.prepare_nvfp4_w1_scales_interleaved(w1_scale) if prepared_data else None,
    )
    seed_out = torch.randn((m, _COMPLETE_K), device=device, generator=generator).to(torch.bfloat16)

    def workspaces():
        return dict(
            sorted_token_ids=torch.empty((expert_capacity * _COMPLETE_BLOCK_M,), dtype=torch.int32, device=device),
            expert_ids=torch.empty((expert_capacity,), dtype=torch.int32, device=device),
            num_tokens_post_padded=torch.empty((1,), dtype=torch.int32, device=device),
            cumsum_buffer=torch.empty((_COMPLETE_E + 2,), dtype=torch.int32, device=device),
        )

    return case, seed_out, workspaces


@pytest.mark.parametrize(
    "m,prepared_data,expected_route",
    [(1, False, 1), (4, False, 8), (8, False, 2), (8, True, 9), (16, False, 7), (16, True, 9), (128, False, 3)],
    ids=["m1_route1", "m4_route8", "m8_route2", "m8_route9", "m16_route7", "m16_route9", "m128_route3_not_deferrable"],
)
def test_alphamoe_nvfp4_deferred_finalize_matches_routed(m, prepared_data, expected_route):
    _skip_if_not_supported()
    from flashinfer.fused_moe import alphamoe_nvfp4_sm100 as api

    case, seed_out, workspaces = _complete_route_case(m, 4020 + m, prepared_data=prepared_data)
    route_id = api._alphamoe_complete_route_id(
        case["hidden_states"], case["hidden_states_scale"], case["gemm1_weights"], case["gemm1_weights_scale"],
        case["gemm2_weights_scale"], case["topk_ids"], seed_out, case["top_k"], case["block_m"],
        case["w1_scale_prepared"], case["w1_data_prepared"],
        w1_scale_prepared_interleaved=case.get("w1_scale_prepared_interleaved"), w2_scale_prepared=case["w2_scale_prepared"],
        w2_data_prepared=case["w2_data_prepared"],
    )
    assert route_id == expected_route, route_id

    if expected_route in (1, 2, 7, 8, 9):
        # Deferrable routes: the BF16-seed finalize equals the seeded routed entry bitwise.
        reference = api.alphamoe_nvfp4_routed_moe(out=seed_out.clone(), **workspaces(), **case)
    else:
        # Other routes keep the zero-seeded routed result and add it to the seed afterwards
        # (the same arithmetic the separate model-side add performs today).
        reference = seed_out.clone()
        reference.add_(api.alphamoe_nvfp4_routed_moe(out=torch.zeros_like(seed_out), **workspaces(), **case))
    torch.cuda.synchronize()

    work = torch.full((m, _COMPLETE_K), float("nan"), dtype=torch.bfloat16, device="cuda")
    deferred = api.alphamoe_nvfp4_routed_moe_deferred(out=work, **workspaces(), **case)
    assert deferred.route_id == expected_route
    assert deferred.deferred == (expected_route in (1, 2, 7, 8, 9))
    in_place = seed_out.clone()
    returned = api.alphamoe_nvfp4_finalize_deferred(deferred, in_place)
    torch.cuda.synchronize()
    assert returned.data_ptr() == in_place.data_ptr()
    assert torch.equal(in_place, reference), "in-place deferred finalize differs from the routed entry"
    if deferred.deferred:
        assert torch.isnan(work.float()).all(), "deferred routes must not touch the output workspace"

    deferred = api.alphamoe_nvfp4_routed_moe_deferred(out=work, **workspaces(), **case)
    separate = torch.empty_like(seed_out)
    api.alphamoe_nvfp4_finalize_deferred(deferred, seed_out, separate)
    torch.cuda.synchronize()
    assert torch.equal(separate, reference), "separate-output deferred finalize differs from the routed entry"

    with pytest.raises(ValueError):
        api.alphamoe_nvfp4_finalize_deferred(deferred, seed_out.float())


def test_prepare_nvfp4_w2_data_is_a_panel_permutation():
    experts, k, packed_n = 3, 256, 128
    w2 = torch.randint(0, 256, (experts, k, packed_n), dtype=torch.uint8)
    prepared = prepare_nvfp4_w2_data(w2)
    blocks = packed_n // 64
    assert prepared.shape == (experts * (k // 128) * blocks, 128, 64)
    for e in range(experts):
        for ob in range(k // 128):
            for j in range(blocks):
                panel = prepared[(e * (k // 128) + ob) * blocks + j]
                assert torch.equal(panel, w2[e, ob * 128:(ob + 1) * 128, j * 64:(j + 1) * 64])
    with pytest.raises(ValueError):
        prepare_nvfp4_w2_data(w2[:, :100].contiguous())
    with pytest.raises(TypeError):
        prepare_nvfp4_w2_data(w2.view(torch.int8))
