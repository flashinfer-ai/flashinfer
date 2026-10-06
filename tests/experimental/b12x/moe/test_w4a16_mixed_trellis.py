from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mixed_trellis_module
from b12x.moe._shared.kernels.w4a16.host import (
    make_w4a16_packed_buffers,
    max_packed_route_slots,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    W4A16FusedMoeKernel,
    W4A16TopKSumKernel,
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    MixedTrellisRotations,
    W4A16MixedTrellis3Kernel,
    W4A16MixedTrellisKernel,
    _check_descriptor_projection_counts,
    _mixed_route_num_experts,
    _require_capture_safe_descriptor_metadata,
    _validate_mixed_trellis_tier_storage,
    bind_mixed_trellis,
    bind_mixed_trellis3,
    build_ordered_maps,
    build_projection_tiered_maps,
    build_tiered_maps,
    combine_trellis_rotations,
    compile_mixed_trellis,
    compile_mixed_trellis3,
    make_mixed_trellis_buffers,
    make_mixed_trellis3_buffers,
    run_bound_mixed_trellis,
    run_bound_mixed_trellis3,
    warmup_mixed_trellis_route_pack,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)


def _mixed_cache_key(tier0_experts: int, tier1_experts: int) -> tuple[object, ...]:
    """Build the key without constructing CUDA-backed kernels.

    The tier subkeys are stubbed, so this proves only that the outer mixed key
    does not read expert counts directly. The GPU ABBA test covers real child
    keys and verifies that they resolve to one compiled object.
    """

    kernel = object.__new__(W4A16MixedTrellisKernel)
    kernel.driver = SimpleNamespace(__cache_key__=("driver",))
    kernel.tier0 = SimpleNamespace(
        __cache_key__=("dynamic-k3",), num_experts=tier0_experts
    )
    kernel.tier1 = SimpleNamespace(
        __cache_key__=("dynamic-k4",), num_experts=tier1_experts
    )
    kernel.blocks_per_sm = 1
    kernel.shared_words = 1
    return kernel.__cache_key__


def test_mixed_kernel_cache_key_is_tier_partition_agnostic() -> None:
    # The exact K3/K4 partition is checkpoint data, not launch geometry. One
    # compiled object must serve every 256-expert GLM-5.2 mixed layout.
    assert _mixed_cache_key(206, 50) == _mixed_cache_key(160, 96)
    assert _mixed_cache_key(192, 64) == _mixed_cache_key(206, 50)
    assert _mixed_cache_key(96, 32) == _mixed_cache_key(80, 16)


def test_mixed_kernel_uses_runtime_expert_bounds() -> None:
    emit_source = textwrap.dedent(
        inspect.getsource(W4A16MixedTrellisKernel._emit_tier_tile)
    )
    kernel_source = textwrap.dedent(inspect.getsource(W4A16MixedTrellisKernel.kernel))
    call_parameters = inspect.signature(W4A16MixedTrellisKernel.__call__).parameters

    assert "tier0_num_experts" in call_parameters
    assert "tier1_num_experts" in call_parameters
    for projection in ("fc2", "gate", "up"):
        assert f"tier0_{projection}_experts" in call_parameters
        assert f"tier1_{projection}_experts" in call_parameters
    assert "self.tier0.num_experts" not in emit_source
    assert "self.tier1.num_experts" not in emit_source
    assert "self.total_experts" not in emit_source
    assert "self.total_experts" not in kernel_source
    for projection in ("fc2", "gate", "up"):
        assert f"local_expert < tier0_{projection}_experts" in emit_source
        assert f"local_expert < tier1_{projection}_experts" in emit_source


def test_three_tier_kernel_uses_projection_specific_k3_k4_k5_bounds() -> None:
    emit_source = textwrap.dedent(
        inspect.getsource(W4A16MixedTrellis3Kernel._emit_tier_tile3)
    )
    call_parameters = inspect.signature(
        W4A16MixedTrellis3Kernel.__call__
    ).parameters

    for tier in range(3):
        assert f"tier{tier}_num_experts" in call_parameters
        for projection in ("fc2", "gate", "up"):
            name = f"tier{tier}_{projection}_experts"
            assert name in call_parameters
            assert f"local_expert < {name}" in emit_source


def test_mixed_runtime_rejects_invalid_raw_tier_storage() -> None:
    device = torch.device("cpu")
    experts, hidden, intermediate, bits = 2, 128, 128, 3
    tier = SimpleNamespace(
        num_experts=experts,
        w13=torch.empty(
            experts * (hidden // 16) * ((2 * intermediate) // 16) * (8 * bits),
            dtype=torch.int32,
        ),
        w2=torch.empty(
            experts * (intermediate // 16) * (hidden // 16) * (8 * bits),
            dtype=torch.int32,
        ),
        w13_scale=torch.empty(4, dtype=torch.uint8),
        w2_scale=torch.empty(4, dtype=torch.uint8),
        w13_global_scale=torch.empty(experts, dtype=torch.float32),
        w2_global_scale=torch.empty(experts, dtype=torch.float32),
    )

    def validate(candidate) -> None:
        _validate_mixed_trellis_tier_storage(
            name="tier0",
            tier=candidate,
            expected_experts=experts,
            bits=bits,
            hidden_size=hidden,
            intermediate_size=intermediate,
            device=device,
        )

    validate(tier)
    for field, replacement in (
        ("w13", tier.w13[:-1]),
        ("w2", tier.w2.to(torch.int16)),
        ("w13_scale", tier.w13_scale[:3]),
        ("w2_global_scale", tier.w2_global_scale[:1]),
    ):
        candidate = SimpleNamespace(**{**vars(tier), field: replacement})
        with pytest.raises(ValueError, match=rf"tier0\.{field}"):
            validate(candidate)


def test_projection_tight_storage_requires_exact_gate_and_up_counts() -> None:
    device = torch.device("cpu")
    experts, hidden, intermediate, bits = 4, 128, 128, 3
    stride = (hidden // 16) * (intermediate // 16) * (8 * bits)

    def tier(planes: int) -> SimpleNamespace:
        return SimpleNamespace(
            num_experts=experts,
            w13=torch.empty(planes * stride, dtype=torch.int32),
            w2=torch.empty(experts * stride, dtype=torch.int32),
            w13_scale=torch.empty(4, dtype=torch.uint8),
            w2_scale=torch.empty(4, dtype=torch.uint8),
            w13_global_scale=torch.empty(experts, dtype=torch.float32),
            w2_global_scale=torch.empty(experts, dtype=torch.float32),
        )

    def validate(candidate, gate=None, up=None) -> None:
        _validate_mixed_trellis_tier_storage(
            name="tier0",
            tier=candidate,
            expected_experts=experts,
            bits=bits,
            hidden_size=hidden,
            intermediate_size=intermediate,
            device=device,
            gate_experts=gate,
            up_experts=up,
        )

    validate(tier(4), gate=1, up=3)
    with pytest.raises(ValueError, match="exactly 4 projection planes"):
        validate(tier(5), gate=1, up=3)
    with pytest.raises(ValueError, match="paired gate_experts/up_experts"):
        validate(tier(4), gate=1)
    with pytest.raises(ValueError, match="projection counts"):
        validate(tier(5), gate=5, up=0)
    with pytest.raises(ValueError, match="exactly 8 projection planes"):
        validate(tier(4))
    validate(tier(8))


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major == 12 and minor in (0, 1)


@pytest.mark.parametrize(
    ("kernel_type", "entry_name"),
    [
        (W4A16MixedTrellisKernel, "kernel"),
        (W4A16MixedTrellis3Kernel, "kernel3"),
    ],
)
def test_mixed_kernel_tracks_shared_moe_body_contract(
    kernel_type: type, entry_name: str
) -> None:
    """Keep the direct CuTe call aligned with the shared driver's ABI."""

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(getattr(kernel_type, entry_name)))
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_moe_body"
    ]
    assert len(calls) == 1

    driver_parameters = inspect.signature(W4A16FusedMoeKernel._moe_body).parameters
    assert len(calls[0].args) + len(calls[0].keywords) == len(driver_parameters) - 1
    assert [ast.unparse(arg) for arg in calls[0].args[-12:]] == [
        "descriptor_map",
        "trellis_lut_addr",
        "trellis_lut_addr",
        "total_experts",
        "total_experts",
        "smem_base",
        "tid",
        "cta",
        "grid_x",
        "active_m",
        "fc1_emit",
        "fc2_emit",
    ]


def test_mixed_runtime_tracks_topk_sum_contract() -> None:
    """Keep the direct compiled top-k launch aligned with its runtime ABI."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(run_bound_mixed_trellis)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compiled"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "topk_sum"
    ]
    assert len(calls) == 1

    sum_parameters = inspect.signature(W4A16TopKSumKernel.__call__).parameters
    assert len(calls[0].args) + len(calls[0].keywords) == len(sum_parameters) - 1
    assert [ast.unparse(arg) for arg in calls[0].args[-4:]] == [
        "Int32(launch.topk_sum.num_experts)",
        "Int32(launch.topk_sum.route_num_experts)",
        "m",
        "stream",
    ]


@pytest.mark.parametrize(
    "runner", [run_bound_mixed_trellis, run_bound_mixed_trellis3]
)
def test_bound_mixed_runtime_does_not_revalidate_fixed_artifacts(runner) -> None:
    """Serving validates only request tensors and preallocated capacity."""

    source = textwrap.dedent(inspect.getsource(runner))
    assert "_validate_mixed_trellis_tier_storage" not in source
    assert "_check_descriptor_projection_counts" not in source
    assert ".w13" not in source
    assert ".rotations" not in source


def test_mixed_dispatch_calls_shared_tile_primitive() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(W4A16MixedTrellisKernel._dispatch_tier_gemm))
    )
    tile_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("_run_tile")
    ]

    assert [node.func.attr for node in tile_calls] == ["_run_tile"]
    assert [ast.unparse(arg) for arg in tile_calls[0].args[10:13]] == [
        "trellis_lut_addr",
        "smem_base",
        "tid",
    ]


def _prepared(
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    bits: int,
    seed: int,
    device: torch.device,
    tile_config: tuple[int, int, int, int] = (128, 128, 128, 128),
    shared_h: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    codebook: str = "mcg",
):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)
        ).to(torch.float16)

    if shared_h is None:
        gate_suh = scales((experts, hidden))
        up_suh = scales((experts, hidden))
        intermediate_rotations = scales((experts, 3 * intermediate))
        down_svh = scales((experts, hidden))
    else:
        gate_suh, up_suh, down_svh = shared_h
        intermediate_rotations = scales((experts, 3 * intermediate))
    return prepare_trellis256_moe_weights(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=tile_config[1],
        fc2_tile_n=tile_config[3],
        device=device,
        seed=seed,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        codebook=codebook,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=tile_config,
    )


def _serial_tier(
    x: torch.Tensor,
    prepared,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    block_size_m: int = 8,
    activation: str = "silu",
) -> torch.Tensor:
    m, topk = int(topk_ids.shape[0]), int(topk_ids.shape[1])
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=x.device,
        route_num_experts=int(expert_map.numel()),
        full_rotation=True,
        block_size_m=block_size_m,
    )
    assert buffers.rotation_a_gate is not None
    assert buffers.rotation_a_up is not None
    return run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        expert_map=expert_map,
        output_expert_map=expert_map,
        route_block_size_m=block_size_m,
        intermediate_rotation_scales=prepared.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=prepared.gate_suh,
        suh_up_table=prepared.up_suh,
        svh_table=prepared.down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("route_ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("codebook", ["mcg", "lut_e4m3"])
def test_mixed_k3_k4_matches_serial_and_captures(
    route_ids_dtype: torch.dtype,
    codebook: str,
) -> None:
    torch.manual_seed(20260730)
    device = torch.device("cuda", torch.cuda.current_device())
    m, hidden, intermediate, topk = 2, 128, 128, 2
    tier0 = _prepared(
        experts=2,
        hidden=hidden,
        intermediate=intermediate,
        bits=3,
        seed=301,
        device=device,
        codebook=codebook,
    )
    tier1 = _prepared(
        experts=2,
        hidden=hidden,
        intermediate=intermediate,
        bits=4,
        seed=401,
        device=device,
        codebook=codebook,
    )
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    # Global expert ids deliberately interleave K3 and K4 tiers. The combined
    # namespace remains tier ordered so weight and rotation tables stay dense.
    topk_ids = torch.tensor([[0, 1], [3, 2]], dtype=route_ids_dtype, device=device)
    topk_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )
    map0 = torch.tensor([1, -1, 0, -1], dtype=torch.int32, device=device)
    map1 = torch.tensor([-1, 1, -1, 0], dtype=torch.int32, device=device)
    serial = _serial_tier(x, tier0, topk_weights, topk_ids, map0)
    serial = serial + _serial_tier(x, tier1, topk_weights, topk_ids, map1)
    torch.cuda.synchronize(device)

    props = torch.cuda.get_device_properties(device)
    launch = compile_mixed_trellis(
        size_m=m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=2,
        tier1_num_experts=2,
        top_k=topk,
        max_m_blocks=8,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=(128, 128, 128, 128),
        route_ids_dtype=route_ids_dtype,
        trellis_codebook=codebook,
    )
    global_to_combined, descriptor = build_tiered_maps((2, 0), (3, 1), device=device)
    rotations = combine_trellis_rotations(tier0, tier1)
    buffers = make_mixed_trellis_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )
    binding = bind_mixed_trellis(
        tier0,
        tier1,
        global_to_combined,
        descriptor,
        rotations,
        launch,
    )
    assert buffers.fc2.data_ptr() == buffers.rotation_gate.data_ptr()

    misaligned_x = torch.empty(m * hidden + 1, dtype=torch.bfloat16, device=device)[
        1:
    ].view(m, hidden)
    assert misaligned_x.is_contiguous()
    assert misaligned_x.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match=r"input.*16-byte alignment"):
        run_bound_mixed_trellis(
            misaligned_x,
            topk_weights,
            topk_ids,
            binding,
            buffers,
        )

    misaligned_intermediate = torch.empty(
        rotations.intermediate.numel() + 1, dtype=torch.float16, device=device
    )[1:]
    assert misaligned_intermediate.is_contiguous()
    assert misaligned_intermediate.data_ptr() % 16 != 0
    misaligned_rotations = type(rotations)(
        intermediate=misaligned_intermediate,
        gate_suh=rotations.gate_suh,
        up_suh=rotations.up_suh,
        down_svh=rotations.down_svh,
    )
    with pytest.raises(ValueError, match=r"intermediate rotations.*16-byte alignment"):
        bind_mixed_trellis(
            tier0,
            tier1,
            global_to_combined,
            descriptor,
            misaligned_rotations,
            launch,
        )

    eager = run_bound_mixed_trellis(
        x,
        topk_weights,
        topk_ids,
        binding,
        buffers,
    )
    torch.cuda.synchronize(device)
    eager = eager.clone()
    assert not torch.isnan(eager).any()
    relative = (eager - serial).norm() / serial.norm().clamp_min(1.0e-12)
    assert float(relative) < 4.0e-3

    repeat = run_bound_mixed_trellis(
        x,
        topk_weights,
        topk_ids,
        binding,
        buffers,
    )
    torch.cuda.synchronize(device)
    assert torch.equal(repeat, eager)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_bound_mixed_trellis(
            x,
            topk_weights,
            topk_ids,
            binding,
            buffers,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, eager)

    # An unmapped global expert must contribute zero without changing any
    # mapped route. This is the contract needed by expert-parallel placement.
    skipped_map1 = map1.clone()
    skipped_map1[3] = -1
    skipped_serial = _serial_tier(
        x, tier0, topk_weights, topk_ids, map0
    ) + _serial_tier(x, tier1, topk_weights, topk_ids, skipped_map1)
    skipped_global_to_combined = global_to_combined.clone()
    skipped_global_to_combined[3] = -1
    skipped_binding = bind_mixed_trellis(
        tier0,
        tier1,
        skipped_global_to_combined,
        descriptor,
        rotations,
        launch,
    )
    skipped = run_bound_mixed_trellis(
        x,
        topk_weights,
        topk_ids,
        skipped_binding,
        buffers,
    )
    torch.cuda.synchronize(device)
    skipped_relative = (
        skipped - skipped_serial
    ).norm() / skipped_serial.norm().clamp_min(1.0e-12)
    assert float(skipped_relative) < 4.0e-3


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(
    (
        "hidden",
        "intermediate",
        "m",
        "topk",
        "experts_per_tier",
        "route_num_experts",
    ),
    [
        pytest.param(128, 128, 2, 3, 2, 6, id="small-single-k-fc2"),
        pytest.param(128, 256, 2, 3, 2, 6, id="small-multi-k-fc2"),
        pytest.param(
            6144,
            256,
            4,
            8,
            3,
            256,
            id="qwen38-flash-next-decode",
        ),
    ],
)
def test_mixed_k3_k4_k5_matches_serial_and_captures(
    hidden: int,
    intermediate: int,
    m: int,
    topk: int,
    experts_per_tier: int,
    route_num_experts: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate MCG K5 dispatch, multi-K FC2, and graph replay."""

    torch.manual_seed(20260809)
    device = torch.device("cuda", torch.cuda.current_device())
    tiers = tuple(
        _prepared(
            experts=experts_per_tier,
            hidden=hidden,
            intermediate=intermediate,
            bits=bits,
            seed=300 + bits,
            device=device,
            codebook="mcg",
        )
        for bits in (3, 4, 5)
    )
    total_experts = 3 * experts_per_tier
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = (
        torch.arange(m * topk, dtype=torch.int32, device=device)
        .remainder(total_experts)
        .reshape(m, topk)
    )
    topk_weights = torch.softmax(
        torch.randn((m, topk), dtype=torch.float32, device=device), dim=-1
    )
    tier_maps = tuple(
        torch.cat(
            (
                torch.full(
                    (tier * experts_per_tier,),
                    -1,
                    dtype=torch.int32,
                    device=device,
                ),
                torch.arange(
                    experts_per_tier, dtype=torch.int32, device=device
                ),
                torch.full(
                    ((2 - tier) * experts_per_tier,),
                    -1,
                    dtype=torch.int32,
                    device=device,
                ),
            )
        )
        for tier in range(3)
    )
    serial = sum(
        (
            _serial_tier(x, tier, topk_weights, topk_ids, expert_map)
            for tier, expert_map in zip(tiers, tier_maps, strict=True)
        ),
        torch.zeros((m, hidden), dtype=torch.float32, device=device),
    )
    props = torch.cuda.get_device_properties(device)
    route_slots = max_packed_route_slots(m * topk, 8, route_num_experts)
    launch = compile_mixed_trellis3(
        size_m=m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=experts_per_tier,
        tier1_num_experts=experts_per_tier,
        tier2_num_experts=experts_per_tier,
        route_num_experts=route_num_experts,
        top_k=topk,
        max_m_blocks=(route_slots + 7) // 8,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=(128, 128, 128, 128),
        trellis_codebook="mcg",
    )
    projection_tiers = tuple(
        tier for tier in range(3) for _ in range(experts_per_tier)
    )
    global_to_combined, descriptor = build_projection_tiered_maps(
        projection_tiers,
        projection_tiers,
        projection_tiers,
        tier_slots=(experts_per_tier,) * 3,
        device=device,
    )
    if route_num_experts > total_experts:
        global_to_combined = torch.cat(
            (
                global_to_combined,
                torch.full(
                    (route_num_experts - total_experts,),
                    -1,
                    dtype=torch.int32,
                    device=device,
                ),
            )
        )
    rotations = combine_trellis_rotations(*tiers)
    buffers = make_mixed_trellis3_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )

    assert tiers[0].trellis is not None
    mismatched_tier0 = replace(
        tiers[0], trellis=replace(tiers[0].trellis, codebook="lut_e4m3")
    )
    with pytest.raises(ValueError, match="launch-plan codebook"):
        bind_mixed_trellis3(
            mismatched_tier0,
            tiers[1],
            tiers[2],
            global_to_combined,
            descriptor,
            rotations,
            launch,
        )

    binding = bind_mixed_trellis3(
        *tiers,
        global_to_combined,
        descriptor,
        rotations,
        launch,
    )

    def reject_revalidation(**_kwargs) -> None:
        raise AssertionError("bound execution repeated fixed-artifact validation")

    monkeypatch.setattr(
        mixed_trellis_module,
        "_validate_mixed_trellis_tier_storage",
        reject_revalidation,
    )
    eager = run_bound_mixed_trellis3(
        x,
        topk_weights,
        topk_ids,
        binding,
        buffers,
    ).clone()
    torch.cuda.synchronize(device)
    assert not torch.isnan(eager).any()
    relative = (eager - serial).norm() / serial.norm().clamp_min(1.0e-12)
    assert float(relative) < 4.0e-3

    repeated = run_bound_mixed_trellis3(
        x,
        topk_weights,
        topk_ids,
        binding,
        buffers,
    )
    torch.cuda.synchronize(device)
    assert torch.equal(repeated, eager)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_bound_mixed_trellis3(
            x,
            topk_weights,
            topk_ids,
            binding,
            buffers,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, eager)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_mixed_k3_k4_k5_partition_reuses_one_compiled_object() -> None:
    """Three-tier checkpoint counts are runtime data, not JIT geometry."""

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)

    def compile_partition(counts: tuple[int, int, int]):
        return compile_mixed_trellis3(
            size_m=2,
            hidden_size=128,
            intermediate_size=128,
            tier0_num_experts=counts[0],
            tier1_num_experts=counts[1],
            tier2_num_experts=counts[2],
            top_k=3,
            max_m_blocks=8,
            sms=int(props.multi_processor_count),
            max_shared_mem=int(props.shared_memory_per_block_optin),
            force_tile_config=(128, 128, 128, 128),
        )

    counts_a = (2, 2, 2)
    counts_b = (3, 2, 1)
    launch_a = compile_partition(counts_a)
    launch_b = compile_partition(counts_b)

    assert launch_a.compiled is launch_b.compiled
    assert (
        launch_a.tier0_num_experts,
        launch_a.tier1_num_experts,
        launch_a.tier2_num_experts,
    ) == counts_a
    assert (
        launch_b.tier0_num_experts,
        launch_b.tier1_num_experts,
        launch_b.tier2_num_experts,
    ) == counts_b


@pytest.mark.parametrize(
    ("codebook", "bits", "message"),
    [
        ("lut_e4m3", (3, 4, 5), "only the MCG codebook"),
        ("mcg", (3, 4, 6), "one K3, one K4, and one K5 tier"),
        ("mcg", (3, 3, 5), "one K3, one K4, and one K5 tier"),
    ],
)
def test_compile_mixed_trellis3_rejects_unqualified_formats(
    codebook: str,
    bits: tuple[int, int, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_mixed_trellis3(
            size_m=1,
            hidden_size=128,
            intermediate_size=128,
            tier0_num_experts=1,
            tier1_num_experts=1,
            tier2_num_experts=1,
            top_k=1,
            max_m_blocks=3,
            sms=1,
            max_shared_mem=1,
            force_tile_config=(128, 128, 128, 128),
            tier0_bits=bits[0],
            tier1_bits=bits[1],
            tier2_bits=bits[2],
            trellis_codebook=codebook,
        )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_projection_padded_launch_uses_exact_route_namespace() -> None:
    """Projection padding must not enlarge routing or its JIT specialization."""

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    launch = compile_mixed_trellis3(
        size_m=2,
        hidden_size=128,
        intermediate_size=128,
        tier0_num_experts=3,
        tier1_num_experts=3,
        tier2_num_experts=1,
        route_num_experts=6,
        top_k=3,
        max_m_blocks=8,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=(128, 128, 128, 128),
    )
    buffers = make_mixed_trellis3_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )
    route_map = torch.arange(6, dtype=torch.int32, device=device)

    assert launch.topk_sum.num_experts == 7
    assert launch.topk_sum.route_num_experts == 6
    assert warmup_mixed_trellis_route_pack(
        launch, buffers, expert_map=route_map
    ) > 0


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_projection_padded_k345_runner_matches_serial_and_captures() -> None:
    """Exercise the complete runner when descriptor slots exceed routes."""

    torch.manual_seed(20260810)
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    m, hidden, intermediate, topk = 2, 128, 128, 3
    tiers = tuple(
        _prepared(
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
            bits=bits,
            seed=500 + bits,
            device=device,
            codebook="mcg",
        )
        for bits, experts in ((3, 3), (4, 3), (5, 1))
    )
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 3, 5], [4, 2, 1]], dtype=torch.int32, device=device
    )
    topk_weights = torch.tensor(
        [[0.5, 0.3, 0.2], [0.25, 0.25, 0.5]],
        dtype=torch.float32,
        device=device,
    )
    tier_maps = (
        torch.tensor([0, 1, 2, -1, -1, -1], dtype=torch.int32, device=device),
        torch.tensor([-1, -1, -1, 0, 1, -1], dtype=torch.int32, device=device),
        torch.tensor([-1, -1, -1, -1, -1, 0], dtype=torch.int32, device=device),
    )
    serial = sum(
        (
            _serial_tier(x, tier, topk_weights, topk_ids, expert_map)
            for tier, expert_map in zip(tiers, tier_maps, strict=True)
        ),
        torch.zeros((m, hidden), dtype=torch.float32, device=device),
    )
    tier1_w13_stride = int(tiers[1].w13.numel()) // 6
    tier1_w13 = tiers[1].w13.reshape(6, tier1_w13_stride)
    runner_tiers = (
        tiers[0],
        replace(
            tiers[1],
            w13=torch.cat((tier1_w13[:2], tier1_w13[3:5])).contiguous(),
        ),
        tiers[2],
    )
    global_to_combined, descriptor = build_projection_tiered_maps(
        [0, 0, 0, 1, 1, 2],
        [0, 0, 0, 1, 1, 2],
        [0, 0, 0, 1, 1, 2],
        tier_slots=(3, 3, 1),
        device=device,
    )

    def padded_rows(name: str) -> torch.Tensor:
        rows = [getattr(tiers[0], name), getattr(tiers[1], name)[:2]]
        rows.extend((getattr(tiers[2], name), getattr(tiers[1], name)[2:]))
        return torch.cat(rows).contiguous()

    rotations = MixedTrellisRotations(
        intermediate=padded_rows("intermediate_rotations"),
        gate_suh=padded_rows("gate_suh"),
        up_suh=padded_rows("up_suh"),
        down_svh=padded_rows("down_svh"),
    )
    launch = compile_mixed_trellis3(
        size_m=m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=3,
        tier1_num_experts=3,
        tier2_num_experts=1,
        route_num_experts=6,
        top_k=topk,
        max_m_blocks=8,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=(128, 128, 128, 128),
        trellis_codebook="mcg",
    )
    buffers = make_mixed_trellis3_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )
    binding = bind_mixed_trellis3(
        *runner_tiers,
        global_to_combined,
        descriptor,
        rotations,
        launch,
        gate_experts=(3, 2, 1),
        up_experts=(3, 2, 1),
    )

    def run() -> torch.Tensor:
        return run_bound_mixed_trellis3(
            x,
            topk_weights,
            topk_ids,
            binding,
            buffers,
        )

    eager = run().clone()
    torch.cuda.synchronize(device)
    relative = (eager - serial).norm() / serial.norm().clamp_min(1.0e-12)
    assert float(relative) < 4.0e-3

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, eager)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_mixed_k3_k4_shared_h_matches_expanded_and_captures() -> None:
    """A physical one-row H rotation must stay broadcast through mixed K3/K4."""

    torch.manual_seed(20260804)
    device = torch.device("cuda", torch.cuda.current_device())
    m, hidden, intermediate, topk = 2, 128, 128, 2
    generator = torch.Generator(device=device).manual_seed(20260804)

    def shared_row() -> torch.Tensor:
        return (
            0.875
            + 0.25 * torch.rand((1, hidden), generator=generator, device=device)
        ).to(torch.float16)

    shared_h = (shared_row(), shared_row(), shared_row())
    tier0 = _prepared(
        experts=2,
        hidden=hidden,
        intermediate=intermediate,
        bits=3,
        seed=301,
        device=device,
        shared_h=shared_h,
    )
    tier1 = _prepared(
        experts=2,
        hidden=hidden,
        intermediate=intermediate,
        bits=4,
        seed=401,
        device=device,
        shared_h=shared_h,
    )
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [3, 2]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )
    map0 = torch.tensor([1, -1, 0, -1], dtype=torch.int32, device=device)
    map1 = torch.tensor([-1, 1, -1, 0], dtype=torch.int32, device=device)
    serial = _serial_tier(x, tier0, topk_weights, topk_ids, map0)
    serial.add_(_serial_tier(x, tier1, topk_weights, topk_ids, map1))

    intermediate_rotations = torch.cat(
        (tier0.intermediate_rotations, tier1.intermediate_rotations), dim=0
    ).contiguous()
    broadcast_rotations = MixedTrellisRotations(
        intermediate=intermediate_rotations,
        gate_suh=shared_h[0],
        up_suh=shared_h[1],
        down_svh=shared_h[2],
    )
    total_experts = int(tier0.num_experts + tier1.num_experts)
    expanded_rotations = MixedTrellisRotations(
        intermediate=intermediate_rotations,
        gate_suh=shared_h[0].expand(total_experts, -1).contiguous(),
        up_suh=shared_h[1].expand(total_experts, -1).contiguous(),
        down_svh=shared_h[2].expand(total_experts, -1).contiguous(),
    )

    props = torch.cuda.get_device_properties(device)

    def compile_launch(*, broadcast: bool):
        return compile_mixed_trellis(
            size_m=m,
            hidden_size=hidden,
            intermediate_size=intermediate,
            tier0_num_experts=2,
            tier1_num_experts=2,
            top_k=topk,
            max_m_blocks=8,
            sms=int(props.multi_processor_count),
            max_shared_mem=int(props.shared_memory_per_block_optin),
            force_tile_config=(128, 128, 128, 128),
            broadcast_suh=broadcast,
            broadcast_svh=broadcast,
        )

    broadcast_launch = compile_launch(broadcast=True)
    expanded_launch = compile_launch(broadcast=False)
    assert broadcast_launch.broadcast_suh is True
    assert broadcast_launch.broadcast_svh is True
    assert expanded_launch.broadcast_suh is False
    assert expanded_launch.broadcast_svh is False
    assert broadcast_launch.compiled is not expanded_launch.compiled

    global_to_combined, descriptor = build_tiered_maps((2, 0), (3, 1), device=device)
    broadcast_buffers = make_mixed_trellis_buffers(
        broadcast_launch, device=device, sms=int(props.multi_processor_count)
    )
    expanded_buffers = make_mixed_trellis_buffers(
        expanded_launch, device=device, sms=int(props.multi_processor_count)
    )
    with pytest.raises(ValueError, match=r"gate SUH.*512 elements"):
        bind_mixed_trellis(
            tier0,
            tier1,
            global_to_combined,
            descriptor,
            broadcast_rotations,
            expanded_launch,
        )
    with pytest.raises(ValueError, match=r"gate SUH.*128 elements"):
        bind_mixed_trellis(
            tier0,
            tier1,
            global_to_combined,
            descriptor,
            expanded_rotations,
            broadcast_launch,
        )

    broadcast_binding = bind_mixed_trellis(
        tier0,
        tier1,
        global_to_combined,
        descriptor,
        broadcast_rotations,
        broadcast_launch,
    )
    expanded_binding = bind_mixed_trellis(
        tier0,
        tier1,
        global_to_combined,
        descriptor,
        expanded_rotations,
        expanded_launch,
    )
    broadcast = run_bound_mixed_trellis(
        x,
        topk_weights,
        topk_ids,
        broadcast_binding,
        broadcast_buffers,
    ).clone()
    expanded = run_bound_mixed_trellis(
        x,
        topk_weights,
        topk_ids,
        expanded_binding,
        expanded_buffers,
    ).clone()
    torch.cuda.synchronize(device)
    assert torch.equal(broadcast, expanded)
    relative = (broadcast - serial).norm() / serial.norm().clamp_min(1.0e-12)
    assert float(relative) < 4.0e-3

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_bound_mixed_trellis(
            x,
            topk_weights,
            topk_ids,
            broadcast_binding,
            broadcast_buffers,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, broadcast)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(
    ("layouts", "max_m_blocks"),
    (
        (((206, 50), (160, 96)), 8),
        (((160, 96), (206, 50)), 9),
        (((80, 16), (96, 32)), 10),
    ),
    ids=("206-50_then_160-96", "160-96_then_206-50", "total-96_then_total-128"),
)
def test_mixed_runtime_partition_reuses_one_compiled_object_abba(
    layouts: tuple[tuple[int, int], tuple[int, int]], max_m_blocks: int
) -> None:
    """One compiled kernel must safely serve both production split families."""

    torch.manual_seed(20260803)
    device = torch.device("cuda", torch.cuda.current_device())
    m, hidden, intermediate, topk = 2, 256, 128, 2
    props = torch.cuda.get_device_properties(device)

    def prepare_partition(
        tier0_experts: int, tier1_experts: int, seed: int
    ) -> tuple[object, object]:
        return (
            _prepared(
                experts=tier0_experts,
                hidden=hidden,
                intermediate=intermediate,
                bits=3,
                seed=seed,
                device=device,
            ),
            _prepared(
                experts=tier1_experts,
                hidden=hidden,
                intermediate=intermediate,
                bits=4,
                seed=seed + 100,
                device=device,
            ),
        )

    def serial_partition(
        x: torch.Tensor,
        tiers: tuple[object, object],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        tier0_experts = int(tiers[0].num_experts)
        tier1_experts = int(tiers[1].num_experts)
        map0 = torch.cat(
            (
                torch.arange(tier0_experts, dtype=torch.int32, device=device),
                torch.full((tier1_experts,), -1, dtype=torch.int32, device=device),
            )
        )
        map1 = torch.cat(
            (
                torch.full((tier0_experts,), -1, dtype=torch.int32, device=device),
                torch.arange(tier1_experts, dtype=torch.int32, device=device),
            )
        )
        return _serial_tier(x, tiers[0], topk_weights, topk_ids, map0) + _serial_tier(
            x, tiers[1], topk_weights, topk_ids, map1
        )

    tiers_a = prepare_partition(*layouts[0], seed=301)
    tiers_b = prepare_partition(*layouts[1], seed=501)
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )

    # Exercise the highest local id in both tiers. Stale compile-time bounds
    # would either skip these routes or read beyond the compact tier tensor.
    def boundary_routes(layout: tuple[int, int]) -> torch.Tensor:
        tier0_experts, tier1_experts = layout
        total_experts = tier0_experts + tier1_experts
        return torch.tensor(
            [[tier0_experts - 1, total_experts - 1], [0, tier0_experts]],
            dtype=torch.int32,
            device=device,
        )

    topk_ids_a = boundary_routes(layouts[0])
    topk_ids_b = boundary_routes(layouts[1])

    def compile_partition(
        tier0_experts: int, tier1_experts: int, *, sms: int | None = None
    ):
        return compile_mixed_trellis(
            size_m=m,
            hidden_size=hidden,
            intermediate_size=intermediate,
            tier0_num_experts=tier0_experts,
            tier1_num_experts=tier1_experts,
            top_k=topk,
            max_m_blocks=max_m_blocks,
            sms=int(props.multi_processor_count if sms is None else sms),
            max_shared_mem=int(props.shared_memory_per_block_optin),
            force_tile_config=(128, 128, 128, 128),
        )

    launch_a = compile_partition(*layouts[0])
    launch_b = compile_partition(*layouts[1])
    assert launch_a.compiled is launch_b.compiled
    assert (launch_a.tier0_num_experts, launch_a.tier1_num_experts) == layouts[0]
    assert (launch_b.tier0_num_experts, launch_b.tier1_num_experts) == layouts[1]
    alternate_sms = max(int(props.multi_processor_count) - 1, 1)
    launch_alt_sms = compile_partition(*layouts[1], sms=alternate_sms)
    launch_restored_sms = compile_partition(
        *layouts[0], sms=int(props.multi_processor_count)
    )
    assert launch_alt_sms.compiled is launch_a.compiled
    assert launch_restored_sms.compiled is launch_a.compiled
    assert launch_alt_sms.sms == alternate_sms
    assert launch_restored_sms.sms == int(props.multi_processor_count)

    def run_partition(
        topk_ids: torch.Tensor,
        binding,
        buffers,
    ) -> torch.Tensor:
        return run_bound_mixed_trellis(
            x,
            topk_weights,
            topk_ids,
            binding,
            buffers,
        )

    maps_a = build_ordered_maps(*layouts[0], device=device)
    maps_b = build_ordered_maps(*layouts[1], device=device)
    rotations_a = combine_trellis_rotations(*tiers_a)
    rotations_b = combine_trellis_rotations(*tiers_b)
    buffers_a = make_mixed_trellis_buffers(
        launch_a, device=device, sms=int(props.multi_processor_count)
    )
    buffers_b = make_mixed_trellis_buffers(
        launch_b, device=device, sms=int(props.multi_processor_count)
    )
    binding_a = bind_mixed_trellis(
        *tiers_a,
        *maps_a,
        rotations_a,
        launch_a,
    )
    binding_b = bind_mixed_trellis(
        *tiers_b,
        *maps_b,
        rotations_b,
        launch_b,
    )
    serial_a = serial_partition(x, tiers_a, topk_weights, topk_ids_a)
    serial_b = serial_partition(x, tiers_b, topk_weights, topk_ids_b)

    output_a1 = run_partition(topk_ids_a, binding_a, buffers_a).clone()
    output_b1 = run_partition(topk_ids_b, binding_b, buffers_b).clone()
    output_b2 = run_partition(topk_ids_b, binding_b, buffers_b).clone()
    output_a2 = run_partition(topk_ids_a, binding_a, buffers_a).clone()
    torch.cuda.synchronize(device)

    for actual, expected in (
        (output_a1, serial_a),
        (output_b1, serial_b),
        (output_b2, serial_b),
        (output_a2, serial_a),
    ):
        assert not torch.isnan(actual).any()
        relative = (actual - expected).norm() / expected.norm().clamp_min(1.0e-12)
        assert float(relative) < 4.0e-3
    assert torch.equal(output_a1, output_a2)
    assert torch.equal(output_b1, output_b2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_b = run_partition(topk_ids_b, binding_b, buffers_b)
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured_b, output_b1)


def test_build_tiered_maps_rejects_invalid_partitions() -> None:
    with pytest.raises(ValueError, match="disjoint partition"):
        build_tiered_maps((0, 1), (1, 2), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="disjoint partition"):
        build_tiered_maps((0, 4), (1, 2), device=torch.device("cpu"))


def test_build_tiered_maps_repeats_projection_independent_descriptor_row() -> None:
    route, descriptor = build_tiered_maps(
        (2, 0), (3, 1), device=torch.device("cpu")
    )
    assert route.tolist() == [1, 3, 0, 2]
    rows = descriptor.reshape(3, 4)
    assert rows[0].tolist() == [0, 1, 1 << 8, (1 << 8) | 1]
    assert torch.equal(rows[0], rows[1])
    assert torch.equal(rows[0], rows[2])


def test_projection_tiered_maps_pad_each_row_to_slot_stride() -> None:
    gate = [0] * 129 + [1] * 127
    up = [0] * 128 + [1] * 128
    down = [0] * 77 + [1] * 179
    route, descriptor = build_projection_tiered_maps(
        gate,
        up,
        down,
        tier_slots=(129, 128),
        device=torch.device("cpu"),
    )

    assert route.dtype == torch.int32
    assert route.tolist() == list(range(256))
    assert _mixed_route_num_experts(route, 256) == 256
    rows = descriptor.reshape(3, 257)
    assert torch.equal(rows[:, -1], torch.full((3,), -1, dtype=torch.int32))
    assert rows[0, 128].item() == 128
    assert rows[0, 129].item() == 1 << 8
    assert rows[1, 127].item() == 127
    assert rows[1, 128].item() == 1 << 8
    assert rows[2, 76].item() == 76
    assert rows[2, 77].item() == 1 << 8


def test_projection_route_namespace_must_match_the_map() -> None:
    route = torch.arange(4, dtype=torch.int32)
    with pytest.raises(ValueError, match="compiled route namespace"):
        _mixed_route_num_experts(route, 5)


def test_descriptor_projection_count_check_fails_closed() -> None:
    gate = [0] * 129 + [1] * 127
    up = [0] * 128 + [1] * 128
    down = [0] * 77 + [1] * 179
    _, descriptor = build_projection_tiered_maps(
        gate,
        up,
        down,
        tier_slots=(129, 128),
        device=torch.device("cpu"),
    )
    total = 257
    assert descriptor._mt_projection_counts == ((129, 127), (128, 128))
    _check_descriptor_projection_counts(
        descriptor,
        total,
        gate_counts=(129, 127),
        up_counts=(128, 128),
    )
    with pytest.raises(ValueError, match="disagree with the descriptor"):
        _check_descriptor_projection_counts(
            descriptor,
            total,
            gate_counts=(1, 3),
            up_counts=(128, 128),
        )

    stripped = descriptor.clone()
    assert not hasattr(stripped, "_mt_projection_counts")
    with pytest.raises(ValueError, match="disagree with the descriptor"):
        _check_descriptor_projection_counts(
            stripped,
            total,
            gate_counts=(129, 128),
            up_counts=(128, 128),
        )
    assert stripped._mt_projection_counts == ((129, 127), (128, 128))
    _check_descriptor_projection_counts(
        stripped,
        total,
        gate_counts=(129, 127),
        up_counts=(128, 128),
    )


def test_projection_tiered_maps_support_an_empty_tier() -> None:
    tiers = [0] * 256
    route, descriptor = build_projection_tiered_maps(
        tiers,
        tiers,
        tiers,
        tier_slots=(256, 0),
        device=torch.device("cpu"),
    )
    assert route.tolist() == list(range(256))
    rows = descriptor.reshape(3, 256)
    assert rows[:, 255].tolist() == [255, 255, 255]


@pytest.mark.parametrize("slots", [(256,), (256, 0, 0, 0)])
def test_projection_tiered_maps_reject_wrong_slot_arity(slots) -> None:
    with pytest.raises(ValueError, match="exactly two or three"):
        build_projection_tiered_maps(
            [0], [0], [0], tier_slots=slots, device=torch.device("cpu")
        )


@pytest.mark.parametrize("slots", [(-1, 2), (300, -44)])
def test_projection_tiered_maps_reject_invalid_slots(slots) -> None:
    with pytest.raises(ValueError, match=r"\[0, 256\]"):
        build_projection_tiered_maps(
            [0], [0], [0], tier_slots=slots, device=torch.device("cpu")
        )


def test_projection_tiered_maps_encode_gate_up_and_down_independently() -> None:
    route, descriptor = build_projection_tiered_maps(
        gate_tiers=(0, 1, 0, 2),
        up_tiers=(1, 1, 2, 0),
        down_tiers=(2, 0, 2, 1),
        tier_slots=(2, 2, 2),
        device=torch.device("cpu"),
    )

    assert route.tolist() == [0, 1, 2, 3]
    assert descriptor.view(3, 6).tolist() == [
        [0, 1 << 8, 1, 2 << 8, -1, -1],
        [1 << 8, (1 << 8) | 1, 2 << 8, 0, -1, -1],
        [2 << 8, 0, (2 << 8) | 1, 1 << 8, -1, -1],
    ]
    assert descriptor._mt_projection_counts == ((2, 1, 1), (1, 2, 1))

    with pytest.raises(ValueError, match="cannot address all gate/up locals"):
        build_projection_tiered_maps(
            gate_tiers=(0, 0, 0, 1),
            up_tiers=(0, 1, 2, 2),
            down_tiers=(0, 1, 2, 0),
            tier_slots=(2, 1, 2),
            device=torch.device("cpu"),
        )


def test_capture_binding_requires_host_prepared_projection_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _, descriptor = build_projection_tiered_maps(
        gate_tiers=(0, 1),
        up_tiers=(1, 0),
        down_tiers=(0, 1),
        tier_slots=(1, 1),
        device=torch.device("cpu"),
    )

    _require_capture_safe_descriptor_metadata(descriptor)
    with pytest.raises(RuntimeError, match="prepared on the host"):
        _require_capture_safe_descriptor_metadata(descriptor.clone())


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("candidate_block_size", [32, 64])
def test_one_grid_large_blocks_avoid_serial_prefill_drift(
    candidate_block_size: int,
) -> None:
    """Large route blocks with M8 FC2 subtiles preserve one-grid arithmetic."""

    torch.manual_seed(20260801)
    device = torch.device("cuda", torch.cuda.current_device())
    m, hidden, intermediate, topk = 64, 512, 256, 8
    tile_config = (128, 128, 32, 512)
    tier0_experts, tier1_experts = 6, 2

    prepared_tiers = tuple(
        _prepared(
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
            bits=bits,
            seed=seed,
            device=device,
            tile_config=tile_config,
        )
        for experts, bits, seed in (
            (tier0_experts, 3, 301),
            (tier1_experts, 4, 401),
        )
    )

    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = (
        torch.tensor([0, 6, 1, 7, 2, 3, 4, 5], dtype=torch.int32, device=device)
        .expand(m, -1)
        .contiguous()
    )
    topk_weights = torch.softmax(
        torch.randn((m, topk), dtype=torch.float32, device=device), dim=-1
    )
    props = torch.cuda.get_device_properties(device)
    global_to_combined, descriptor = build_tiered_maps(
        range(tier0_experts), range(tier0_experts, 8), device=device
    )

    def one_grid(
        block_size_m: int,
    ) -> tuple[torch.Tensor, object, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        route_slots = max_packed_route_slots(m * topk, block_size_m, 8)
        launch = compile_mixed_trellis(
            size_m=m,
            hidden_size=hidden,
            intermediate_size=intermediate,
            tier0_num_experts=tier0_experts,
            tier1_num_experts=tier1_experts,
            top_k=topk,
            max_m_blocks=(route_slots + block_size_m - 1) // block_size_m,
            moe_block_size=block_size_m,
            sms=int(props.multi_processor_count),
            max_shared_mem=int(props.shared_memory_per_block_optin),
            force_tile_config=tile_config,
        )
        buffers = make_mixed_trellis_buffers(
            launch, device=device, sms=int(props.multi_processor_count)
        )
        binding = bind_mixed_trellis(
            prepared_tiers[0],
            prepared_tiers[1],
            global_to_combined,
            descriptor,
            combine_trellis_rotations(*prepared_tiers),
            launch,
        )
        output = run_bound_mixed_trellis(
            x,
            topk_weights,
            topk_ids,
            binding,
            buffers,
        ).clone()
        phase_outputs = (
            buffers.fc1.clone(),
            buffers.activated.clone(),
            buffers.fc2.clone(),
        )
        return output, launch, phase_outputs

    reference, reference_launch, reference_phases = one_grid(8)
    candidate, candidate_launch, candidate_phases = one_grid(candidate_block_size)
    torch.cuda.synchronize(device)

    phase_equal = tuple(
        torch.equal(candidate_phase, reference_phase)
        for candidate_phase, reference_phase in zip(
            candidate_phases, reference_phases, strict=True
        )
    )
    geometry = (
        f"packed_block={candidate_launch.moe_block_size} "
        f"fc2_subtile={candidate_launch.fc2_moe_block_size} "
        f"fc2_schedule_factor={candidate_launch.fc2_schedule_route_block_factor} "
        f"smem={candidate_launch.shared_memory_bytes}"
    )
    assert reference_launch.moe_block_size == 8
    assert reference_launch.fc2_moe_block_size == 8
    assert reference_launch.fc2_schedule_route_block_factor == 1
    assert candidate_launch.moe_block_size == candidate_block_size
    assert candidate_launch.fc2_moe_block_size == 8
    assert candidate_launch.fc2_schedule_route_block_factor == 2
    assert phase_equal == (True, True, True), geometry
    assert torch.equal(candidate, reference), geometry
    assert candidate_launch.shared_memory_bytes <= int(
        props.shared_memory_per_block_optin
    )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_glm52_large_m_mixed_k3_k4_matches_serial() -> None:
    """Cover the production prefill shape that exposed lost FC1 reductions."""

    torch.manual_seed(20260730)
    device = torch.device("cuda", torch.cuda.current_device())
    m, hidden, intermediate, topk = 3072, 6144, 512, 8
    tile_config = (128, 128, 64, 256)
    tier0_experts, tier1_experts = 192, 64
    tier0 = _prepared(
        experts=tier0_experts,
        hidden=hidden,
        intermediate=intermediate,
        bits=3,
        seed=301,
        device=device,
        tile_config=tile_config,
    )
    tier1 = _prepared(
        experts=tier1_experts,
        hidden=hidden,
        intermediate=intermediate,
        bits=4,
        seed=401,
        device=device,
        tile_config=tile_config,
    )
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    route_row = torch.tensor(
        [0, 1, 2, 3, 4, 5, tier0_experts, tier0_experts + 1],
        dtype=torch.int32,
        device=device,
    )
    topk_ids = route_row.expand(m, -1).contiguous()
    topk_weights = torch.softmax(
        torch.randn((m, topk), dtype=torch.float32, device=device), dim=-1
    )
    map0 = torch.cat(
        (
            torch.arange(tier0_experts, dtype=torch.int32, device=device),
            torch.full((tier1_experts,), -1, dtype=torch.int32, device=device),
        )
    )
    map1 = torch.cat(
        (
            torch.full((tier0_experts,), -1, dtype=torch.int32, device=device),
            torch.arange(tier1_experts, dtype=torch.int32, device=device),
        )
    )
    serial = _serial_tier(x, tier0, topk_weights, topk_ids, map0).clone()
    serial.add_(_serial_tier(x, tier1, topk_weights, topk_ids, map1))

    props = torch.cuda.get_device_properties(device)
    route_slots = max_packed_route_slots(m * topk, 8, 256)
    launch = compile_mixed_trellis(
        size_m=m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=tier0_experts,
        tier1_num_experts=tier1_experts,
        top_k=topk,
        max_m_blocks=(route_slots + 7) // 8,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=tile_config,
    )
    assert launch.moe_block_size == 8
    global_to_combined, descriptor = build_tiered_maps(
        range(tier0_experts), range(tier0_experts, 256), device=device
    )
    buffers = make_mixed_trellis_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )
    binding = bind_mixed_trellis(
        tier0,
        tier1,
        global_to_combined,
        descriptor,
        combine_trellis_rotations(tier0, tier1),
        launch,
    )
    mixed = run_bound_mixed_trellis(
        x,
        topk_weights,
        topk_ids,
        binding,
        buffers,
    )
    torch.cuda.synchronize(device)

    relative = (serial - mixed).norm() / serial.norm().clamp_min(1.0e-12)
    # The serial oracle sums two independently reduced tier outputs, while the
    # mixed grid reduces them in one schedule. Normal rounding stays below
    # 1e-4; the unsafe 64x256 FC1 geometry was three orders larger (~8e-3).
    assert float(relative) < 1.0e-4


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("route_ids_dtype,broadcast,intermediate_hadamard", [
    (torch.int32, False, False),
    (torch.int64, True, False),
    (torch.int64, False, True),
    (torch.int32, True, True),
])
def test_full_rotation_prefill_capacity_reuses_native_launchers(tmp_path, route_ids_dtype, broadcast, intermediate_hadamard):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe.fused_moe import _impl as impl
    from b12x.moe._shared.kernels.w4a16.exl3 import read_exl3_layer
    from b12x.moe._shared.kernels.w4a16.exl3_synth import Exl3SynthConfig, write_exl3_checkpoint

    device = torch.device("cuda", torch.cuda.current_device())
    hidden, intermediate, experts, capacity, topk = 512 if intermediate_hadamard else 256, 256, 3, 128, 2
    codebook = "lut_e4m3" if intermediate_hadamard else "mcg"
    manifest = write_exl3_checkpoint(tmp_path, Exl3SynthConfig(
        codebook=codebook, num_experts=experts, hidden_size=hidden,
        intermediate_size=intermediate, moe_layer_indices=(0,), bits=3,
        per_expert_input_rotations=not broadcast, intermediate_hadamard=intermediate_hadamard,
        pre_block=512 if intermediate_hadamard else None, post_block=128 if intermediate_hadamard else None,
        extent_alignment_slots=4, seed=5,
    ))
    layer = read_exl3_layer(tmp_path, manifest, 0, first_slot=0, slot_count=intermediate // 32)
    weight_plan = impl.plan_b12x_fp4_moe_weights(
        quant_modes="w4a16", source_format="exl3", trellis_codebook=codebook,
        activation="silu", params_dtype=torch.float16, num_experts=experts,
        hidden_size=hidden, intermediate_size=intermediate, trellis_bits=3,
        trellis_tile_config=(64, 256, 64, 256), intermediate_hadamard=intermediate_hadamard,
    )
    weights = impl.prepare_b12x_fp4_moe_weights(
        plan=weight_plan, params_dtype=torch.float16, exl3_layer=layer, exl3_device=device,
    )
    plan = impl.plan_tp_moe_scratch(impl.TPMoEScratchCaps(
        max_tokens=capacity, core_token_counts=(capacity,), num_topk=topk,
        route_num_experts=experts, device=device, weight_plan=weight_plan,
        quant_mode="w4a16", w4a16_block_size_m=64,
        decode_config=impl.MoeDecodeConfig(backend="w4a16", route_planner="internal", max_active_clusters=None, w4a16_route_mode="packed"),
    ))
    assert {rows for rows, _ in plan._prewarmed_fused_launches} == {capacity}
    scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in plan.scratch_specs())
    torch.manual_seed(614)
    source = (torch.randn(capacity, hidden, device=device) * 0.125).half()
    route_weights = torch.softmax(torch.randn(capacity, topk, device=device), dim=1)
    route_ids = torch.randint(experts, (capacity, topk), device=device, dtype=route_ids_dtype)
    expert_map = torch.arange(experts, device=device, dtype=torch.int32)
    output = torch.empty_like(source)
    def bind(rows):
        return plan.bind(
            scratch=scratch, a=source[:rows], experts=weights,
            topk_weights=route_weights[:rows], topk_ids=route_ids[:rows],
            output=output[:rows], route_expert_map=expert_map, output_expert_map=expert_map,
        )
    def reference(rows):
        return _serial_tier(source[:rows], weights.representation.value, route_weights[:rows], route_ids[:rows], expert_map, block_size_m=64).to(output.dtype)
    bind(capacity).run()
    addresses = tuple(tensor.data_ptr() for tensor in (*scratch, output))
    for rows in (3, 11, 125, capacity):
        with kernel_resolution_guard("Trellis prefill capacity"):
            binding = bind(rows)
            assert any(binding.fused_launch is launch for _, launch in plan._prewarmed_fused_launches)
            assert binding.fused_launch.broadcast_suh == broadcast
            assert binding.topk_sum_launch.broadcast_svh == broadcast
            actual = binding.run().clone()
        expected = reference(rows)
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    binding = bind(11)
    graph = torch.cuda.CUDAGraph()
    try:
        with kernel_resolution_guard("Trellis capacity replay"), torch.cuda.graph(graph):
            binding.run()
        source.mul_(-0.5)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output[:11], reference(11), rtol=2e-3, atol=2e-3)
        assert tuple(tensor.data_ptr() for tensor in (*scratch, output)) == addresses
    finally:
        graph.reset()
