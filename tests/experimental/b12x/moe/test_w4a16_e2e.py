from __future__ import annotations

from contextlib import ExitStack
import pytest
import torch

from b12x._lib.intrinsics import (
    FLOAT4_E2M1_MAX,
    fp4_quantize_values_torch,
    pack_grouped_fp4_values,
    swizzle_block_scale,
)
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe.fused_moe._impl import (
    plan_b12x_fp4_moe_weights,
    prewarm_w4a16_fc2_e8m0,
    prepare_w4a16_fc2_e8m0,
    prepare_b12x_fp4_moe_weights,
    run_w4a16_fc2_e8m0,
)
from b12x.moe._shared.execution import PreparedWeightLayout
from b12x.moe._shared.kernels.reference import (
    moe_reference_nvfp4,
    moe_reference_w4a16_f32,
    moe_reference_w4a16_fp4_e8m0_k32,
)
from b12x.moe._shared.kernels.w4a16.host import (
    max_packed_route_slots,
    route_pack_capacity,
    select_route_block_size_m,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _DEFAULT_MAX_SHARED_MEM,
    _W4A16_SMALL_M_DIRECT_MAX_M,
    MoEMicroKernelW4A16SmallMDirect,
    _small_m_direct_host_barrier_reset_enabled,
    _w4a16_stream_is_capturing,
    _small_m_direct_supported,
    compile_w4a16_fused_moe,
    compile_w4a16_topk_sum,
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    make_w4a16_packed_buffers as make_w4a16_buffers,
    prepare_w4a16_e8m0_native_weights,
    prepare_w4a16_modelopt_native_weights,
    prepare_w4a16_modelopt_nvfp4_weights as prepare_w4a16_weights,
    prepare_w4a16_packed_weights,
)
from b12x.testing.reference.helpers import (
    make_tp_moe_fp4_binding,
    prepare_tp_moe_fp4_experts,
    run_tp_moe_fp4,
)
from b12x.testing.reference.w4a16_reference import compare_to_reference, moe_reference_w4a16


def test_w4a16_small_m_host_barrier_reset_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("B12X_W4A16_SMALL_M_HOST_BARRIER_RESET", raising=False)
    assert _small_m_direct_host_barrier_reset_enabled()
    monkeypatch.setenv("B12X_W4A16_SMALL_M_HOST_BARRIER_RESET", "0")
    assert not _small_m_direct_host_barrier_reset_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("source_format", ["modelopt_nvfp4", "fp4_e8m0_k32"])
@pytest.mark.parametrize("fused_sum", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("capacity", [14, 129])
def test_w4a16_prefill_reduction_prepared_capacity_and_graph(
    source_format, fused_sum, mapped, capacity, monkeypatch
):
    """One declared capacity serves changed inputs and live lengths without JIT."""
    from b12x.moe import fused_moe
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared

    monkeypatch.setenv("B12X_W4A16_PREFILL_FUSED_SUM", str(int(fused_sum)))
    torch.manual_seed(81281)
    experts, hidden, intermediate, topk = 8, 256, 192, 4
    x = (torch.randn(capacity, hidden, device="cuda") * 0.125).to(torch.bfloat16)
    ids = torch.randint(experts, (capacity, topk), device="cuda", dtype=torch.int32)
    expert_map = None
    if mapped:
        expert_map = torch.arange(experts - 1, -1, -1, dtype=torch.int32, device="cuda")
        expert_map[-1] = -1
    weights = torch.softmax(torch.randn(capacity, topk, device="cuda"), dim=-1)
    if source_format == "modelopt_nvfp4":
        raw = _make_weights(
            experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation="silu",
        )
    else:
        raw = (
            torch.randint(
                256,
                (experts, intermediate * 2, hidden // 2),
                dtype=torch.uint8,
                device="cuda",
            ),
            _pattern_e8m0((experts, intermediate * 2, hidden // 32)),
            torch.ones(experts, device="cuda"),
            torch.randint(
                256,
                (experts, hidden, intermediate // 2),
                dtype=torch.uint8,
                device="cuda",
            ),
            _pattern_e8m0((experts, hidden, intermediate // 32), offset=1),
            torch.ones(experts, device="cuda"),
        )
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat(source_format)
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A16,
            nonlinearity="silu",
            io_dtype=x.dtype,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
        ),
    )
    prepared = fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.PackedWeights(
            w13=raw[0].clone(),
            w13_block_scales=raw[1].clone(),
            w13_global_scales=raw[2].clone(),
            w2=raw[3].clone(),
            w2_block_scales=raw[4].clone(),
            w2_global_scales=raw[5].clone(),
        ),
    )
    plan = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=topk),
        routing=fused_moe.RoutingSpec(),
    )
    # Numerical controls must come from the declaration, not from bind-time env.
    monkeypatch.setenv("B12X_W4A16_PREFILL_FUSED_SUM", "0")
    output = torch.empty_like(x)

    def allocate(state):
        return tuple(
            torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in state.scratch.scratch_specs()
        )

    def primer(state):
        scratch = allocate(state)
        binding = state.bind(
            scratch=scratch,
            a=x,
            experts=prepared,
            topk_weights=weights,
            topk_ids=ids,
            output=output,
            route_expert_map=expert_map,
        )
        return PreparedCall(run=lambda: state.run(binding), owners=scratch)

    def expected(rows):
        oracle_ids, oracle_weights = ids[:rows], weights[:rows]
        if expert_map is not None:
            oracle_ids = expert_map[oracle_ids.long()]
            oracle_weights = oracle_weights.masked_fill(oracle_ids < 0, 0.0)
            oracle_ids = oracle_ids.clamp_min(0)
        if source_format == "fp4_e8m0_k32":
            return moe_reference_w4a16_fp4_e8m0_k32(
                x[:rows],
                *raw,
                oracle_ids,
                oracle_weights,
                experts,
                hidden,
                intermediate,
                activation="silu",
                swiglu_limit=None,
                w13_layout="w13",
            )
        return _reference_w4a16(
            x[:rows], *raw, oracle_ids, oracle_weights, activation="silu"
        )

    with PreparationSession(device=x.device, autotune=False) as session:
        request = plan.request(name="prefill-reduction-capacity", prepare_call=primer)
        session.prepare((request,))
        state = require_prepared(request.plan, "moe.decode")
        assert state.scratch._core_workspace_plan.prefill_fused_sum_fp32 == fused_sum
        scratch = allocate(state)
        pointers = tuple(t.data_ptr() for t in scratch)
        launchers = set()
        for rows in (capacity, min(17, capacity), 1, capacity - 1):
            oracle = expected(rows)
            with kernel_resolution_guard():
                binding = fused_moe.bind(
                    request.plan,
                    scratch=scratch,
                    a=x[:rows],
                    experts=prepared,
                    topk_weights=weights[:rows],
                    topk_ids=ids[:rows],
                    output=output[:rows],
                    route_expert_map=expert_map,
                )
                actual = binding.run()
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
            _assert_matches_oracle(actual, oracle, activation="silu")
            launchers.add(id(binding.fused_launch))
        assert len(launchers) == 1
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            captured = binding.run()
        for factor in (-0.5, 2.0):
            x.mul_(factor)
            oracle = expected(capacity - 1)
            for tensor in scratch:
                tensor.fill_(255)
            before = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            assert pointers == tuple(t.data_ptr() for t in scratch)
            assert torch.isfinite(captured).all() and torch.count_nonzero(captured)
            _assert_matches_oracle(captured, oracle, activation="silu")
        del graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [70, 300])
def test_w4a16_skipped_empty_row_blocks_are_bit_identical(tokens, monkeypatch):
    """Large-M route blocks skip the MMAs of 16-row blocks without live routes.

    Skewed routes end expert blocks at varied fills (block 32 at 70 tokens,
    block 64 at 300); the output must equal the full-block schedule exactly.
    """
    from b12x.moe import fused_moe
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared

    monkeypatch.setenv("B12X_W4A16_PREFILL_FUSED_SUM", "0")
    torch.manual_seed(5150 + tokens)
    experts, hidden, intermediate, topk = 8, 256, 192, 2
    x = (torch.randn(tokens, hidden, device="cuda") * 0.125).to(torch.bfloat16)
    shares = torch.arange(1, experts + 1, dtype=torch.float32, device="cuda")
    ids = torch.multinomial(shares.expand(tokens, -1), topk).to(torch.int32)
    weights = torch.softmax(torch.randn(tokens, topk, device="cuda"), dim=-1)
    raw = _make_weights(
        experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation="silu",
    )
    outputs = {}
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16GemmKernel
    original_init = W4A16GemmKernel.__init__
    compiled_controls = []

    def record_control(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        compiled_controls.append(self.skip_empty_m_blocks)

    monkeypatch.setattr(W4A16GemmKernel, "__init__", record_control)
    for skip in ("0", "1"):
        monkeypatch.setenv("B12X_W4A16_SKIP_EMPTY_M_BLOCKS", skip)
        weight_plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("modelopt_nvfp4")),
            activation=fused_moe.ActivationSpec(
                mode=fused_moe.ActivationMode.A16,
                nonlinearity="silu",
                io_dtype=x.dtype,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=experts,
                hidden_size=hidden,
                intermediate_size=intermediate,
            ),
        )
        prepared = fused_moe.prepare_weights(
            plan=weight_plan,
            weights=fused_moe.PackedWeights(
                w13=raw[0].clone(),
                w13_block_scales=raw[1].clone(),
                w13_global_scales=raw[2].clone(),
                w2=raw[3].clone(),
                w2_block_scales=raw[4].clone(),
                w2_global_scales=raw[5].clone(),
            ),
        )
        plan = fused_moe.plan_execution(
            experts=prepared,
            capacity=fused_moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
            routing=fused_moe.RoutingSpec(),
        )
        monkeypatch.setenv("B12X_W4A16_SKIP_EMPTY_M_BLOCKS", str(1 - int(skip)))
        compiled_controls.clear()
        output = torch.empty_like(x)

        def allocate(state):
            return tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                for spec in state.scratch.scratch_specs()
            )

        def primer(state, output=output, prepared=prepared):
            scratch = allocate(state)
            binding = state.bind(
                scratch=scratch,
                a=x,
                experts=prepared,
                topk_weights=weights,
                topk_ids=ids,
                output=output,
            )
            return PreparedCall(run=lambda: state.run(binding), owners=scratch)

        with PreparationSession(device=x.device, autotune=False, compile_workers=0) as session:
            request = plan.request(name=f"skip-empty-rows-{skip}", prepare_call=primer)
            session.prepare((request,))
            state = require_prepared(request.plan, "moe.decode")
            binding = fused_moe.bind(
                request.plan,
                scratch=allocate(state),
                a=x,
                experts=prepared,
                topk_weights=weights,
                topk_ids=ids,
                output=output,
            )
            assert binding.fused_launch.moe_block_size > 16
            outputs[skip] = binding.run().clone()
            assert compiled_controls and set(compiled_controls) == {skip == "1"}
            session.freeze()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                binding.run()
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(output, outputs[skip], rtol=0, atol=0)
            graph.reset()
    assert torch.isfinite(outputs["1"]).all() and torch.count_nonzero(outputs["1"])
    assert torch.equal(outputs["0"], outputs["1"])
    _assert_matches_oracle(
        outputs["1"],
        _reference_w4a16(x, *raw, ids, weights, activation="silu"),
        activation="silu",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("host_barrier_reset", [False, True])
def test_w4a16_small_m_direct_barrier_modes_eager_and_graph(
    host_barrier_reset: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    monkeypatch.setenv("B12X_W4A16_SMALL_M_DIRECT", "1")
    monkeypatch.setenv(
        "B12X_W4A16_SMALL_M_HOST_BARRIER_RESET",
        "1" if host_barrier_reset else "0",
    )
    torch.manual_seed(20260805 + int(host_barrier_reset))
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m, activation = 2, 1, "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale = weights
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(
        0,
        experts,
        (m, topk),
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    a_gscale = torch.ones(experts, dtype=torch.float32, device="cuda")
    expert_weights = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a_gscale,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_alphas=w13_global_scale,
        a2_gscale=a_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_global_scale,
        activation=activation,
        quant_mode="w4a16",
    )
    prepared = expert_weights._impl.representation_for("w4a16")
    assert prepared.weight_layout == "modelopt"
    assert _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout=prepared.weight_layout,
        w13_layout="w13",
        scale_format="e4m3_k16",
    )

    direct_launches = 0
    real_direct_launch = w4a16_kernel._w4a16_small_m_direct_launch_flat

    def spy_direct_launch(*args, **kwargs) -> None:
        nonlocal direct_launches
        direct_launches += 1
        real_direct_launch(*args, **kwargs)

    monkeypatch.setattr(
        w4a16_kernel,
        "_w4a16_small_m_direct_launch_flat",
        spy_direct_launch,
    )
    bindings = ExitStack()
    binding = bindings.enter_context(make_tp_moe_fp4_binding(
        a=x,
        experts=expert_weights,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_mode="w4a16",
        output=torch.empty_like(x),
    ))
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    barrier_count = prepared.workspace[-2:-1]

    eager = binding.run().clone()
    torch.cuda.synchronize()
    _assert_matches_oracle(eager, expected, activation=activation)
    assert int(barrier_count.item()) == 0

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output = binding.run()
    for _ in range(16):
        graph.replay()
    torch.cuda.synchronize()
    _assert_matches_oracle(graph_output, expected, activation=activation)
    assert int(barrier_count.item()) == 0
    assert direct_launches == 2
    del graph
    bindings.close()


def test_w4a16_fused_compile_rejects_unresolved_capture_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    monkeypatch.setattr(w4a16_kernel, "_FUSED_CACHE", {})
    monkeypatch.setattr(w4a16_kernel.torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        RuntimeError,
        match="W4A16 fused MoE launch is not resolved for CUDA graph capture",
    ):
        compile_w4a16_fused_moe(
            size_m=16,
            hidden_size=128,
            intermediate_size=128,
            num_experts=8,
            top_k=2,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            max_m_blocks=16,
            sms=188,
            max_shared_mem=_DEFAULT_MAX_SHARED_MEM,
            _require_cached=True,
        )


def test_w4a16_capture_guard_checks_selected_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    selected_stream = 22
    queried: list[object] = []
    monkeypatch.setattr(
        w4a16_kernel.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        w4a16_kernel.cuda,
        "cuStreamIsCapturing",
        lambda stream: (
            queried.append(stream) or w4a16_kernel.cuda.CUresult.CUDA_SUCCESS,
            1,
        ),
    )

    assert _w4a16_stream_is_capturing(selected_stream, current_stream=11)
    assert queried == [selected_stream]


def test_w4a16_capture_guard_preserves_current_stream_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    monkeypatch.setattr(
        w4a16_kernel.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        w4a16_kernel.cuda,
        "cuStreamIsCapturing",
        lambda _stream: (_ for _ in ()).throw(
            AssertionError("current-stream capture queried the selected stream")
        ),
    )

    assert _w4a16_stream_is_capturing(22, current_stream=11)


@pytest.mark.parametrize(
    ("tokens", "bucket_tokens"),
    (
        (1, 1),
        (8, 8),
        (9, 16),
        (3072, 4096),
        (4096, 4096),
        (4097, 8192),
    ),
)
def test_generic_w4a16_planner_runtime_key_agrees_at_bucket_boundaries(
    tokens: int,
    bucket_tokens: int,
) -> None:
    topk = 8
    block_size = 64
    num_experts = 160
    routed_capacity, packed_routes, max_m_blocks = route_pack_capacity(
        tokens * topk,
        block_size,
        num_experts,
        topk=topk,
    )
    expected_routes = bucket_tokens * topk
    expected_packed = max_packed_route_slots(
        expected_routes,
        block_size,
        num_experts,
    )

    assert routed_capacity == expected_routes
    assert packed_routes == expected_packed
    assert max_m_blocks == (expected_packed + block_size - 1) // block_size
    if tokens == 3072:
        assert max_m_blocks == 670


def test_trellis_w4a16_planner_runtime_key_preserves_exact_capacity() -> None:
    topk = 8
    block_size = 64
    num_experts = 160
    routed_capacity, packed_routes, max_m_blocks = route_pack_capacity(
        3072 * topk,
        block_size,
        num_experts,
        topk=topk,
        bucket_tokens=False,
    )

    assert routed_capacity == 3072 * topk
    assert packed_routes == max_packed_route_slots(
        3072 * topk,
        block_size,
        num_experts,
    )
    assert max_m_blocks == 542


def test_trellis_w4a16_capture_prewarm_uses_exact_runtime_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    from b12x.moe.fused_moe import _impl as tp_moe_impl
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    workspace = SimpleNamespace(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        full_rotation=True,
        route_block_size_m=64,
        num_topk=8,
        route_E=160,
        weight_E=160,
        k=6144,
        n=2048,
        activation="silu",
        trellis_bits=3,
        trellis_codebook="mcg",
        trellis_pair_kinds=None,
        trellis_tile_config=None,
        intermediate_hadamard=False,
    )
    fused_calls: list[dict[str, object]] = []
    resolved_fused = object()
    monkeypatch.setattr(tp_moe_impl.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        tp_moe_impl.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        tp_moe_impl.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            multi_processor_count=188,
            shared_memory_per_block_optin=_DEFAULT_MAX_SHARED_MEM,
        ),
    )
    monkeypatch.setattr(w4a16_kernel, "_rot_scales_dummy", lambda _device: None)
    monkeypatch.setattr(
        w4a16_kernel,
        "compile_w4a16_fused_moe",
        lambda **kwargs: fused_calls.append(kwargs) or resolved_fused,
    )
    monkeypatch.setattr(
        w4a16_kernel,
        "compile_w4a16_topk_sum",
        lambda **_kwargs: object(),
    )

    tp_moe_impl._prewarm_w4a16_planned_launches(
        workspace,
        token_counts=(3072,),
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=1.0,
        scale_format="e4m3_k32",
        weight_layout="trellis_t256",
        w13_layout="trellis_t256_proj",
        collect_activation_amax=False,
    )

    main_calls = [call for call in fused_calls if call["size_m"] == 3072]
    direct_calls = [call for call in fused_calls if call["size_m"] != 3072]
    assert len(main_calls) == 2
    assert {call["max_m_blocks"] for call in main_calls} == {542}
    assert {call["size_m"] for call in direct_calls} == set(
        range(1, _W4A16_SMALL_M_DIRECT_MAX_M + 1)
    )
    assert all(
        sum(call["size_m"] == direct_m for call in direct_calls) == 2
        for direct_m in range(1, _W4A16_SMALL_M_DIRECT_MAX_M + 1)
    )
    assert (
        workspace.planned_fused_moe_launches[("trellis_t256", "e4m3_k32", 3072, False)]
        is resolved_fused
    )


def _positive_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.rand(shape, device="cuda") * 0.25 + 0.03125).to(torch.float8_e4m3fn)


def _constant_e8m0(shape: tuple[int, ...], byte: int) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("requires torch.float8_e8m0fnu")
    return torch.full(shape, byte, dtype=torch.uint8, device="cuda").view(e8m0_dtype)


def _pattern_e8m0(shape: tuple[int, ...], *, offset: int = 0) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("requires torch.float8_e8m0fnu")
    numel = 1
    for dim in shape:
        numel *= int(dim)
    storage = (torch.arange(numel, device="cuda", dtype=torch.int64) + offset) % 4 + 119
    return storage.to(torch.uint8).reshape(shape).view(e8m0_dtype)


def _quantize_dense_moe_weight_storage(
    input_tensor: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_groups, rows, cols = input_tensor.shape
    quantized = torch.zeros(
        (num_groups, rows, cols),
        dtype=torch.float32,
        device=input_tensor.device,
    )
    scales = torch.zeros(
        (num_groups, rows, cols // 16),
        dtype=torch.float32,
        device=input_tensor.device,
    )
    for group_idx in range(num_groups):
        x = input_tensor[group_idx].float()
        sliced = x.view(rows, cols // 16, 16)
        block_max = sliced.abs().amax(dim=-1, keepdim=True)
        scale = (global_scale[group_idx] * (block_max / FLOAT4_E2M1_MAX)).to(
            torch.float8_e4m3fn
        )
        scale = scale.to(torch.float32)
        output_scale = 1.0 / (scale * (1.0 / global_scale[group_idx])).clamp(min=1e-30)
        clipped = torch.clamp(
            sliced * output_scale,
            -FLOAT4_E2M1_MAX,
            FLOAT4_E2M1_MAX,
        ).view(rows, cols)
        quantized[group_idx] = fp4_quantize_values_torch(clipped)
        scales[group_idx] = scale.squeeze(-1)

    packed = pack_grouped_fp4_values(quantized).permute(2, 0, 1).contiguous()
    swizzled = swizzle_block_scale(scales.to(torch.float8_e4m3fn))
    return packed, swizzled


def _make_weights(
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    is_gated = activation in {"silu", "situ"}
    w13_rows = intermediate_size * (2 if is_gated else 1)
    w13 = torch.randint(
        0,
        256,
        (experts, w13_rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_blockscale = swizzle_block_scale(
        _positive_fp8((experts, w13_rows, hidden_size // 16))
    )
    w2_blockscale = swizzle_block_scale(
        _positive_fp8((experts, hidden_size, intermediate_size // 16))
    )
    w13_global_scale = (torch.rand(experts, device="cuda") * 0.1 + 0.05).to(
        torch.float32
    )
    w2_global_scale = (torch.rand(experts, device="cuda") * 0.1 + 0.05).to(
        torch.float32
    )
    return w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale


def _run_w4a16(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
    expert_map: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    prepared = prepare_w4a16_weights(
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=x.shape[0],
        topk=topk_ids.shape[1],
        dtype=x.dtype,
        device=x.device,
        route_num_experts=None if expert_map is None else int(expert_map.numel()),
    )
    return run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        expert_map=expert_map,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        swiglu_limit=swiglu_limit,
    )


def _reference_w4a16(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
    expert_map: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    reference_topk_ids = topk_ids
    reference_topk_weights = topk_weights
    if expert_map is not None:
        reference_topk_ids = expert_map[topk_ids.long()].to(torch.int32)
        active = reference_topk_ids >= 0
        reference_topk_weights = topk_weights.masked_fill(~active, 0.0)
        reference_topk_ids = reference_topk_ids.clamp_min(0)
    return moe_reference_w4a16(
        x,
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        reference_topk_ids,
        reference_topk_weights,
        w13.shape[0],
        w2.shape[1],
        w2.shape[2] * 2,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )


def _assert_matches_oracle(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    activation: str,
) -> None:
    metrics = compare_to_reference(actual, expected)
    min_cos = 0.9975 if activation == "silu" else 0.9900
    assert metrics.cos >= min_cos, metrics


@pytest.mark.parametrize("scale_format", ["e4m3_k16", "e8m0_k32"])
@pytest.mark.parametrize("m", [1, 3, 8])
def test_w4a16_packed_weights_do_not_route_to_small_m_direct(
    m: int,
    scale_format: str,
) -> None:
    assert not _small_m_direct_supported(
        m=m,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        topk=2,
        activation="silu",
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="packed",
        w13_layout="packed",
        scale_format=scale_format,
    )


@pytest.mark.parametrize(
    "intermediate_size",
    [16, 144, 352, 464, 480, 496, 512],
)
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("m", [1, 3])
def test_w4a16_modelopt_direct_keeps_non64_intermediate_contract(
    m: int,
    activation: str,
    intermediate_size: int,
) -> None:
    """Native ModelOpt decode supports every 16-aligned intermediate shape."""
    assert _small_m_direct_supported(
        m=m,
        hidden_size=128,
        intermediate_size=intermediate_size,
        num_experts=1,
        topk=1,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="modelopt",
        w13_layout="w13",
        scale_format="e4m3_k16",
    )
    kernel = MoEMicroKernelW4A16SmallMDirect(
        activation=activation,
        fast_math=True,
        share_input_across_experts=True,
        share_expert_scales=True,
        single_token=m == 1,
    )
    kernel.configure(
        m=m,
        k=128,
        n=intermediate_size,
        num_topk=1,
        weight_E=1,
        max_active_ctas=1,
    )
    assert kernel._cfg is not None
    assert kernel._cfg.n % kernel._cfg.fc1_chunks == 0
    assert kernel._cfg.i_chunk % 16 == 0
    assert kernel._cfg.inter_blocks * 16 == kernel._cfg.i_chunk
    if m > 1:
        assert kernel._cfg.i_chunk <= 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_fp4_e8m0_k32_kernel_matches_raw_e8m0_oracle(
    activation: str,
) -> None:
    experts, hidden_size, intermediate_size = 4, 128, 128
    rows = intermediate_size * (2 if activation == "silu" else 1)
    m, topk = 8, 2
    torch.manual_seed(20260526 + (1000 if activation == "silu" else 0))
    w13 = torch.randint(
        0,
        256,
        (experts, rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0((experts, hidden_size, intermediate_size // 32), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    prepared = prepare_w4a16_packed_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        source_format="fp4_e8m0_k32",
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    x = torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [1, 0], [3, 2], [0, 2], [1, 3], [2, 0], [3, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.rand(m, topk, dtype=torch.float32, device="cuda")

    def launch() -> torch.Tensor:
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
            swiglu_limit=10.0 if activation == "silu" else None,
        )

    actual = launch()
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        swiglu_limit=10.0 if activation == "silu" else None,
        w13_layout="w13",
    )
    torch.cuda.synchronize()

    assert bool((actual != 0).any().item())
    _assert_matches_oracle(actual, expected, activation=activation)
    if activation == "relu2":
        eager = actual.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        buffers.output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(buffers.output).all().item())
        assert torch.equal(buffers.output, eager)
        _assert_matches_oracle(buffers.output, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tc_decode", [False, True], ids=["route-packed", "tc-decode"])
def test_w4a16_packed_runtime_expert_count_reuses_compiled_kernel(
    tc_decode: bool,
) -> None:
    """Packed tier size is runtime data, while the launch geometry stays static."""
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    torch.manual_seed(20260730)
    device = torch.device("cuda", torch.cuda.current_device())
    hidden_size = intermediate_size = 128
    activation = "situ"
    rows = 2 * intermediate_size
    m, topk = 4, 2
    props = torch.cuda.get_device_properties(device)
    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )
    w4a16_kernel.clear_w4a16_kernel_cache()

    compiled_after_first: dict[tuple, int] | None = None
    for experts in (2, 3):
        w13 = torch.randint(
            0,
            256,
            (experts, rows, hidden_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        w2 = torch.randint(
            0,
            256,
            (experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32), offset=experts)
        w2_scale = _pattern_e8m0(
            (experts, hidden_size, intermediate_size // 32),
            offset=experts + 1,
        )
        w13_global_scale = torch.ones(experts, dtype=torch.float32, device=device)
        w2_global_scale = torch.ones(experts, dtype=torch.float32, device=device)
        prepared = prepare_w4a16_packed_weights(
            w13,
            w13_scale,
            w13_global_scale,
            w2,
            w2_scale,
            w2_global_scale,
            activation=activation,
            params_dtype=torch.bfloat16,
            source_format="fp4_e8m0_k32",
        )
        buffers = make_w4a16_buffers(
            prepared,
            m=m,
            topk=topk,
            dtype=torch.bfloat16,
            device=device,
        )
        x = (torch.randn(m, hidden_size, device=device) * 0.125).to(torch.bfloat16)
        topk_ids = torch.tensor(
            [
                [0, experts - 1],
                [experts - 1, 0],
                [experts - 1, experts - 1],
                [0, experts - 1],
            ],
            dtype=torch.int32,
            device=device,
        )
        topk_weights = torch.softmax(
            torch.randn(m, topk, dtype=torch.float32, device=device), dim=-1
        )
        fused_launch = compile_w4a16_fused_moe(
            size_m=m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=experts,
            top_k=topk,
            activation=activation,
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            # Capacity metadata legitimately changes with the tier's expert
            # count, but it is not device-code geometry either.
            max_m_blocks=m * topk + experts,
            element_dtype="bf16",
            sms=int(props.multi_processor_count),
            max_shared_mem=max_shared_mem,
            weight_layout="packed",
            scale_format="e8m0_k32",
            w13_layout="packed",
            direct_topk_routes=tc_decode,
            tc_decode_fused_sum=tc_decode,
            force_tile_config=(64, 128, 64, 128),
        )
        assert fused_launch.num_experts == experts
        topk_sum_launch = (
            None
            if tc_decode
            else compile_w4a16_topk_sum(
                m=m,
                topk=topk,
                hidden_size=hidden_size,
                element_dtype="bf16",
            )
        )
        actual = run_w4a16_moe(
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
            fused_launch=fused_launch,
            topk_sum_launch=topk_sum_launch,
        ).clone()
        expected = moe_reference_w4a16_fp4_e8m0_k32(
            x,
            w13,
            w13_scale,
            w13_global_scale,
            w2,
            w2_scale,
            w2_global_scale,
            topk_ids,
            topk_weights,
            experts,
            hidden_size,
            intermediate_size,
            activation=activation,
            w13_layout="w13",
        )
        torch.cuda.synchronize(device)
        _assert_matches_oracle(actual, expected, activation=activation)

        compiled_now = {
            key: id(value.compiled) for key, value in w4a16_kernel._FUSED_CACHE.items()
        }
        if compiled_after_first is None:
            compiled_after_first = compiled_now
            assert len(compiled_after_first) == 1
        else:
            assert compiled_now == compiled_after_first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("intermediate_size", [128, 224])
@pytest.mark.parametrize("w13_layout", ["w13", "w31"])
# 0.5 is small enough to actually engage the SwiGLU clamp at these scales.
@pytest.mark.parametrize("swiglu_limit", [None, 0.5])
@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_e8m0_native_micro_matches_raw_e8m0_oracle(
    activation: str,
    m: int,
    swiglu_limit: float | None,
    w13_layout: str,
    intermediate_size: int,
) -> None:
    """Small-M micro decode path with native MXFP4 (E8M0 K/32) scales."""
    if swiglu_limit is not None and activation != "silu":
        pytest.skip("swiglu_limit only applies to gated (silu) activation")
    if (activation, m, swiglu_limit, w13_layout) == ("silu", 1, None, "w13"):
        common = dict(
            activation="silu",
            fast_math=True,
            share_input_across_experts=True,
            share_expert_scales=True,
            single_token=True,
        )
        e4m3 = MoEMicroKernelW4A16SmallMDirect(scale_format="e4m3_k16", **common)
        e8m0 = MoEMicroKernelW4A16SmallMDirect(scale_format="e8m0_k32", **common)
        assert e4m3.__cache_key__ != e8m0.__cache_key__
    experts, hidden_size = 4, 128
    rows = intermediate_size * (2 if activation == "silu" else 1)
    topk = 2
    torch.manual_seed(20260601 + (1000 if activation == "silu" else 0) + m)
    w13 = torch.randint(
        0, 256, (experts, rows, hidden_size // 2), dtype=torch.uint8, device="cuda"
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0((experts, hidden_size, intermediate_size // 32), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")

    # Confirm the dispatch routes this native E8M0 case to the micro kernel.
    assert _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="modelopt",
        w13_layout=w13_layout,
        scale_format="e8m0_k32",
    )

    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        w13_layout=w13_layout,
    )
    expected_w13_scale_rows = ((rows + 63) // 64) * 64
    assert tuple(prepared.w13_scale.shape) == (
        experts,
        hidden_size // 32,
        expected_w13_scale_rows,
    )
    assert prepared.micro_w13_scale is not None
    assert int(prepared.micro_w13_scale.numel()) == (
        experts * (hidden_size // 32) * expected_w13_scale_rows
    )
    buffers = make_w4a16_buffers(
        prepared, m=m, topk=topk, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    # The micro decode path needs m * fc2_n_chunks * 128 * topk u32 of FC1-output
    # scratch, which is larger than make_w4a16_buffers' generic intermediate_cache2.
    fc2_n_chunks = ((intermediate_size // 2) + 127) // 128
    intermediate_cache2 = torch.zeros(
        2 * m * fc2_n_chunks * 128 * topk, dtype=torch.bfloat16, device="cuda"
    )
    x = torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, experts, (m, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(m, topk, dtype=torch.float32, device="cuda")

    def launch() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            swiglu_limit=swiglu_limit,
        )

    actual = launch()
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        swiglu_limit=swiglu_limit,
        w13_layout=w13_layout,
    )
    torch.cuda.synchronize()

    assert bool((actual != 0).any().item())
    _assert_matches_oracle(actual, expected, activation=activation)
    if (activation, m, swiglu_limit, w13_layout) == ("silu", 1, None, "w13"):
        eager = actual.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        buffers.output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(buffers.output).all().item())
        assert torch.equal(buffers.output, eager)
        _assert_matches_oracle(buffers.output, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("m", "route_ids_dtype"),
    [
        (1, torch.int32),
        (2, torch.int32),
        (4, torch.int32),
        (8, torch.int32),
        (1, torch.int64),
    ],
)
def test_w4a16_e8m0_native_micro_ignores_inactive_routes_during_graph_replay(
    m: int,
    route_ids_dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native small-M execution must not address weights for inactive routes."""
    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    monkeypatch.setenv("B12X_W4A16_SMALL_M_DIRECT", "1")
    direct_launches = 0
    real_direct_launch = w4a16_kernel._w4a16_small_m_direct_launch_flat

    def spy_direct_launch(*args, **kwargs) -> None:
        nonlocal direct_launches
        direct_launches += 1
        real_direct_launch(*args, **kwargs)

    monkeypatch.setattr(
        w4a16_kernel,
        "_w4a16_small_m_direct_launch_flat",
        spy_direct_launch,
    )
    experts, hidden_size, intermediate_size = 4, 128, 192
    topk, activation = 2, "situ"
    rows = 2 * intermediate_size
    torch.manual_seed(20260817 + m)
    w13 = torch.randint(
        0,
        256,
        (experts, rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0((experts, hidden_size, intermediate_size // 32), offset=1)
    global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        global_scale,
        w2,
        w2_scale,
        global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        w13_layout="w31",
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    fc2_n_chunks = ((intermediate_size // 2) + 127) // 128
    intermediate_cache2 = torch.zeros(
        2 * m * fc2_n_chunks * 128 * topk,
        dtype=torch.bfloat16,
        device="cuda",
    )
    inputs = torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(
        0, experts, (m, topk), dtype=route_ids_dtype, device="cuda"
    )
    topk_weights = torch.rand(m, topk, dtype=torch.float32, device="cuda")

    def launch() -> torch.Tensor:
        return run_w4a16_moe(
            inputs,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
        )

    valid_eager = launch().clone()
    torch.cuda.synchronize()
    assert direct_launches == 1
    assert bool(torch.isfinite(valid_eager).all().item())
    assert bool((valid_eager.abs().sum(dim=1) > 0).all().item())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch()
    assert direct_launches == 2

    inactive_ids = topk_ids.clone()
    invalid_upper = 1 << 32 if route_ids_dtype == torch.int64 else experts
    inactive_ids[-1] = torch.tensor(
        [-1, invalid_upper], dtype=route_ids_dtype, device="cuda"
    )
    if m > 1:
        inactive_ids[0, 0] = -1
    topk_ids.copy_(inactive_ids)
    original_weights = topk_weights.clone()
    active = (inactive_ids >= 0) & (inactive_ids < experts)
    reference_ids = torch.where(active, inactive_ids, torch.zeros_like(inactive_ids))
    reference_weights = torch.where(
        active, topk_weights, torch.zeros_like(topk_weights)
    )
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        inputs,
        w13,
        w13_scale,
        global_scale,
        w2,
        w2_scale,
        global_scale,
        reference_ids,
        reference_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        w13_layout="w31",
    )

    eager = launch().clone()
    torch.cuda.synchronize()
    _assert_matches_oracle(eager, expected, activation=activation)
    torch.testing.assert_close(eager[-1], torch.zeros_like(eager[-1]), rtol=0, atol=0)

    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(captured).all().item())
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    torch.testing.assert_close(topk_ids, inactive_ids, rtol=0, atol=0)
    torch.testing.assert_close(topk_weights, original_weights, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_e8m0_native_aligned_generic_fc1_uses_k32_scale_stride() -> None:
    """K/32 scales keep their native expert stride in aligned generic FC1."""
    experts, hidden_size, intermediate_size = 8, 512, 128
    rows = 2 * intermediate_size
    torch.manual_seed(20260803)
    w13 = torch.randint(
        0,
        256,
        (experts, rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0((experts, hidden_size, intermediate_size // 32), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation="silu",
        params_dtype=torch.bfloat16,
        w13_layout="w31",
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=1,
        topk=2,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    intermediate_cache2 = torch.zeros(2 * 128 * 2, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(1, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[6, 7]], dtype=torch.int32, device="cuda")
    topk_weights = torch.tensor([[0.4, 0.6]], dtype=torch.float32, device="cuda")

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
    )
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation="silu",
        w13_layout="w31",
    )
    torch.cuda.synchronize()

    assert bool((actual != 0).any().item())
    _assert_matches_oracle(actual, expected, activation="silu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("w13_layout", ["w13", "w31"])
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_e8m0_native_large_m_uses_main_gemm(
    activation: str,
    w13_layout: str,
) -> None:
    """The native E8M0 object routes med/large M to the main W4A16 GEMM."""
    experts, hidden_size, intermediate_size = 8, 256, 256
    rows = intermediate_size * (2 if activation == "silu" else 1)
    topk, m = 2, 24  # m > _W4A16_SMALL_M_DIRECT_MAX_M -> main W4A16 kernel
    torch.manual_seed(20260607 + (1000 if activation == "silu" else 0))
    w13 = torch.randint(
        0, 256, (experts, rows, hidden_size // 2), dtype=torch.uint8, device="cuda"
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0((experts, hidden_size, intermediate_size // 32), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")

    # m=24 must NOT take the small-M micro path.
    assert not _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="modelopt",
        w13_layout=w13_layout,
        scale_format="e8m0_k32",
    )

    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        w13_layout=w13_layout,
    )
    buffers = make_w4a16_buffers(
        prepared, m=m, topk=topk, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    x = torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, experts, (m, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(m, topk, dtype=torch.float32, device="cuda")

    actual = run_w4a16_moe(
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
    )
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        swiglu_limit=None,
        w13_layout=w13_layout,
    )
    torch.cuda.synchronize()
    assert bool((actual != 0).any().item())
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("activation", "intermediate_size"),
    [
        ("silu", 312),
        ("relu2", 112),
        # 32-aligned but not 128-aligned: no dividing tile config, so the
        # same logical-tail path must engage (2048/TP6 = 352, 3072/TP16 = 192).
        ("silu", 352),
        ("relu2", 192),
        ("situ", 192),
    ],
)
def test_w4a16_e8m0_native_compact_tail_uses_ceil_scale_grid(
    activation: str,
    intermediate_size: int,
) -> None:
    """Native E8M0 supports compact FC2 K tails without padding I to 32."""
    experts, hidden_size = 4, 128
    rows = intermediate_size * (2 if activation in {"silu", "situ"} else 1)
    topk, m = 2, 24
    assert intermediate_size % 8 == 0
    assert intermediate_size % 128 != 0
    torch.manual_seed(20260610)
    w13 = torch.randint(
        0, 256, (experts, rows, hidden_size // 2), dtype=torch.uint8, device="cuda"
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale_cols = (intermediate_size + 31) // 32
    w2_scale = _pattern_e8m0((experts, hidden_size, w2_scale_cols), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")

    assert not _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="modelopt",
        w13_layout="w31",
        scale_format="e8m0_k32",
    )

    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        w13_layout="w31",
    )
    assert prepared.weight_layout == "modelopt"
    assert prepared.scale_format == "e8m0_k32"
    padded_w13_scale_rows = ((rows + 63) // 64) * 64
    assert tuple(prepared.w13_scale.shape) == (
        experts,
        hidden_size // 32,
        padded_w13_scale_rows,
    )
    assert tuple(prepared.w2_scale.shape) == (experts, w2_scale_cols, hidden_size)

    buffers = make_w4a16_buffers(
        prepared, m=m, topk=topk, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    x = torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, experts, (m, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(m, topk, dtype=torch.float32, device="cuda")

    actual = run_w4a16_moe(
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
    )
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        swiglu_limit=None,
        w13_layout="w31",
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual).all().item())
    assert bool((actual != 0).any().item())
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_beats_nvfp4_against_true_fp32_oracle_for_odd_shapes(
    activation: str,
) -> None:
    experts, hidden_size, intermediate_size = 8, 128, 128
    cases = [
        (1, 3, 2, torch.int32, 0.50),
        (5, 7, 3, torch.int64, 0.75),
    ]
    for batch_size, seq_len, topk, ids_dtype, input_scale in cases:
        m = batch_size * seq_len
        torch.manual_seed(
            20260525 + m * 31 + topk + (1000 if activation == "silu" else 0)
        )
        rows = intermediate_size * (2 if activation == "silu" else 1)
        w13_dense = torch.randn(experts, rows, hidden_size, device="cuda") * (
            0.18 if activation == "silu" else 0.08
        )
        w2_dense = torch.randn(
            experts, hidden_size, intermediate_size, device="cuda"
        ) * (0.18 if activation == "silu" else 0.08)
        w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
        w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
        w13, w13_blockscale = _quantize_dense_moe_weight_storage(
            w13_dense,
            w13_global_scale,
        )
        w2, w2_blockscale = _quantize_dense_moe_weight_storage(
            w2_dense,
            w2_global_scale,
        )

        x = (torch.randn(m, hidden_size, device="cuda") * input_scale).to(
            torch.bfloat16
        )
        topk_ids = torch.randint(
            0,
            experts,
            (m, topk),
            device="cuda",
            dtype=ids_dtype,
        )
        topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
        a_gscale = torch.ones(experts, dtype=torch.float32, device="cuda")
        nvfp4_experts = prepare_tp_moe_fp4_experts(
            a=x,
            a1_gscale=a_gscale,
            w1_fp4=w13,
            w1_blockscale=w13_blockscale,
            w1_alphas=w13_global_scale,
            a2_gscale=a_gscale,
            w2_fp4=w2,
            w2_blockscale=w2_blockscale,
            w2_alphas=w2_global_scale,
            activation=activation,
            quant_mode="nvfp4",
        )

        nvfp4 = run_tp_moe_fp4(
            a=x,
            experts=nvfp4_experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_mode="nvfp4",
        )
        w4a16_experts = prepare_tp_moe_fp4_experts(
            a=x,
            a1_gscale=a_gscale,
            w1_fp4=w13,
            w1_blockscale=w13_blockscale,
            w1_alphas=w13_global_scale,
            a2_gscale=a_gscale,
            w2_fp4=w2,
            w2_blockscale=w2_blockscale,
            w2_alphas=w2_global_scale,
            activation=activation,
            quant_mode="w4a16",
        )
        w4a16 = run_tp_moe_fp4(
            a=x,
            experts=w4a16_experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_mode="w4a16",
        )
        nvfp4_reference = moe_reference_nvfp4(
            x,
            w13,
            w13_blockscale,
            w13_global_scale,
            w2,
            w2_blockscale,
            w2_global_scale,
            a_gscale,
            a_gscale,
            topk_ids,
            topk_weights,
            experts,
            hidden_size,
            intermediate_size,
            activation=activation,
        )
        true_fp32 = moe_reference_w4a16_f32(
            x,
            w13,
            w13_blockscale,
            w13_global_scale,
            w2,
            w2_blockscale,
            w2_global_scale,
            topk_ids,
            topk_weights,
            experts,
            hidden_size,
            intermediate_size,
            activation=activation,
        )
        torch.cuda.synchronize()

        nvfp4_reference_metrics = compare_to_reference(nvfp4, nvfp4_reference)
        nvfp4_true_metrics = compare_to_reference(nvfp4, true_fp32)
        w4a16_true_metrics = compare_to_reference(w4a16, true_fp32)
        assert nvfp4_reference_metrics.cos > 0.98, nvfp4_reference_metrics
        assert w4a16_true_metrics.cos > 0.999, w4a16_true_metrics
        assert w4a16_true_metrics.cos > nvfp4_true_metrics.cos + 0.003, (
            nvfp4_true_metrics,
            w4a16_true_metrics,
            batch_size,
            seq_len,
            topk,
            ids_dtype,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("m", [1, 3])
@pytest.mark.parametrize(
    "intermediate_size",
    [32, 128, 144, 224, 352, 464, 480, 496, 512],
)
def test_w4a16_modelopt_direct_replay_ignores_stale_swizzle_tail(
    activation: str,
    m: int,
    intermediate_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    experts, hidden_size = 8, 128
    topk = 2
    monkeypatch.setenv("B12X_W4A16_SMALL_M_DIRECT", "1")
    torch.manual_seed(
        20260730 + (1000 if activation == "silu" else 0) + m + intermediate_size
    )
    real_direct_launch = w4a16_kernel._w4a16_small_m_direct_launch_flat
    direct_launches: list[tuple[int, int]] = []

    def spy_direct_launch(*args, **kwargs) -> None:
        direct_launches.append((int(kwargs["m"]), int(kwargs["intermediate_size"])))
        real_direct_launch(*args, **kwargs)

    monkeypatch.setattr(
        w4a16_kernel,
        "_w4a16_small_m_direct_launch_flat",
        spy_direct_launch,
    )
    rows = intermediate_size * (2 if activation == "silu" else 1)
    w13_dense = torch.randn(experts, rows, hidden_size, device="cuda") * 0.12
    w2_dense = (
        torch.randn(experts, hidden_size, intermediate_size, device="cuda") * 0.12
    )
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w13, w13_blockscale = _quantize_dense_moe_weight_storage(
        w13_dense,
        w13_global_scale,
    )
    w2, w2_blockscale = _quantize_dense_moe_weight_storage(
        w2_dense,
        w2_global_scale,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.5).to(torch.bfloat16)
    topk_ids = torch.randint(
        0,
        experts,
        (m, topk),
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    a_gscale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w4a16_experts = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a_gscale,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_alphas=w13_global_scale,
        a2_gscale=a_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_global_scale,
        activation=activation,
        quant_mode="w4a16",
    )
    prepared_w4a16 = w4a16_experts._impl.representation_for("w4a16")
    assert prepared_w4a16.weight_layout == "modelopt"
    micro_w13_scale = prepared_w4a16.micro_w13_scale
    assert micro_w13_scale is not None
    assert micro_w13_scale.data_ptr() == w13_blockscale.data_ptr()
    assert prepared_w4a16.micro_w2_scale.data_ptr() == w2_blockscale.data_ptr()
    assert _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout=prepared_w4a16.weight_layout,
        w13_layout="w13",
        scale_format="e4m3_k16",
    )
    bindings = ExitStack()
    binding = bindings.enter_context(make_tp_moe_fp4_binding(
        a=x,
        experts=w4a16_experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=torch.empty(
            (m, hidden_size),
            dtype=x.dtype,
            device=x.device,
        ),
        quant_mode="w4a16",
    ))
    expected = moe_reference_w4a16_f32(
        x,
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
    )

    first = binding.run().clone()
    torch.cuda.synchronize()
    first_metrics = compare_to_reference(first, expected)
    assert bool(torch.isfinite(first).all().item())
    assert first_metrics.cos > 0.999, first_metrics

    # The arena legitimately preserves the padded half of each 128-u32
    # swizzle block. Poison it all, then prove FC1 rewrites every logical lane
    # and FC2 ignores every inactive lane on the same-binding replay.
    binding.intermediate_cache2.fill_(float("nan"))
    replay = binding.run().clone()
    torch.cuda.synchronize()
    replay_metrics = compare_to_reference(replay, expected)
    assert bool(torch.isfinite(replay).all().item())
    assert replay_metrics.cos > 0.999, replay_metrics
    torch.testing.assert_close(replay, first, atol=0.0, rtol=0.0)

    # Capture the same direct path once, then repeatedly replay it after
    # poisoning the arena tail. Graph replay does not re-enter the Python spy,
    # so three launches prove two eager calls plus one captured custom-op node.
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output = binding.run()
    for _ in range(16):
        binding.intermediate_cache2.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    graph_metrics = compare_to_reference(graph_output, expected)
    assert bool(torch.isfinite(graph_output).all().item())
    assert graph_metrics.cos > 0.999, graph_metrics
    torch.testing.assert_close(graph_output, first, atol=0.0, rtol=0.0)
    assert direct_launches == [(m, intermediate_size)] * 3
    del graph
    bindings.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "intermediate_size",
    [144, 352, 464, 480, 496, 512],
)
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize("m", [1, 3])
def test_w4a16_modelopt_direct_non64_intermediate_is_bounds_safe(
    m: int,
    activation: str,
    intermediate_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the exact native row tail under CUDA memcheck.

    I=144 exercises the narrow one-chunk kernel: it has 18 logical packed-u32
    values per W2 row while the ModelOpt scale grid pads its control geometry
    to 24. I=352 exercises a partial scale/control tail in the wide kernel.
    I=496 is the boundary case where that padded control grid looks completely
    full even though the final two activation lanes are unwritten. Selecting
    the only expert makes an unmasked W2 tail load cross the tensor allocation
    instead of merely reading the next expert. Run this test under
    compute-sanitizer with ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` to audit
    m==1/m>=2 and ungated/gated direct-FC2 implementations.
    """
    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    experts, hidden_size, topk = 1, 128, 1
    monkeypatch.setenv("B12X_W4A16_SMALL_M_DIRECT", "1")
    real_direct_launch = w4a16_kernel._w4a16_small_m_direct_launch_flat
    direct_launches: list[tuple[int, int]] = []

    def spy_direct_launch(*args, **kwargs) -> None:
        direct_launches.append((int(kwargs["m"]), int(kwargs["intermediate_size"])))
        real_direct_launch(*args, **kwargs)

    monkeypatch.setattr(
        w4a16_kernel,
        "_w4a16_small_m_direct_launch_flat",
        spy_direct_launch,
    )
    torch.manual_seed(
        20260731 + m + intermediate_size + (1000 if activation == "silu" else 0)
    )

    rows = intermediate_size * (2 if activation == "silu" else 1)
    w13_dense = (
        torch.randn(
            experts,
            rows,
            hidden_size,
            device="cuda",
        )
        * 0.12
    )
    w2_dense = (
        torch.randn(
            experts,
            hidden_size,
            intermediate_size,
            device="cuda",
        )
        * 0.12
    )
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w13, w13_blockscale = _quantize_dense_moe_weight_storage(
        w13_dense,
        w13_global_scale,
    )
    w2, w2_blockscale = _quantize_dense_moe_weight_storage(
        w2_dense,
        w2_global_scale,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.5).to(torch.bfloat16)
    topk_ids = torch.zeros((m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.ones((m, topk), device="cuda", dtype=torch.float32)
    a_gscale = torch.ones(experts, dtype=torch.float32, device="cuda")

    experts_w4a16 = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a_gscale,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_alphas=w13_global_scale,
        a2_gscale=a_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_global_scale,
        activation=activation,
        quant_mode="w4a16",
    )
    prepared = experts_w4a16._impl.representation_for("w4a16")
    assert prepared.weight_layout == "modelopt"
    assert _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout=prepared.weight_layout,
        w13_layout="w13",
        scale_format="e4m3_k16",
    )

    bindings = ExitStack()
    binding = bindings.enter_context(make_tp_moe_fp4_binding(
        a=x,
        experts=experts_w4a16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_mode="w4a16",
    ))
    expected = moe_reference_w4a16_f32(
        x,
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
    )

    first = binding.run().clone()
    binding.intermediate_cache2.fill_(float("nan"))
    replay = binding.run().clone()
    torch.cuda.synchronize()
    for actual in (first, replay):
        metrics = compare_to_reference(actual, expected)
        assert bool(torch.isfinite(actual).all().item())
        assert metrics.cos > 0.999, metrics
    assert direct_launches == [(m, intermediate_size), (m, intermediate_size)]
    torch.testing.assert_close(first, replay, rtol=0, atol=0)
    bindings.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
def test_w4a16_tc_decode_fused_sum_matches_oracle(
    m: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TC-decode folds the top-k sum into the FC2 store via atomic accumulate.
    Validate the epilogue across the whole small-M range, not just powers of
    two, since 3/5/6/7 were never exercised through it before. This tiny-expert
    geometry remains below the planner's route-packing crossover."""
    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    # Spy on the fused compile so we can assert the fused-sum path actually engaged
    # (a silent fallback to the packed GEMM would also pass the cosine gate).
    real_compile = w4a16_kernel.compile_w4a16_fused_moe
    saw_tc_decode: list[bool] = []

    def _spy_compile(*args, **kwargs):
        saw_tc_decode.append(bool(kwargs.get("tc_decode_fused_sum", False)))
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(w4a16_kernel, "compile_w4a16_fused_moe", _spy_compile)

    torch.manual_seed(20260529 + m)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, activation = 2, "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    assert prepared.weight_layout == "packed"
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
    )
    tiny_route_workspace = torch.empty((1,), dtype=torch.int32, device=x.device)

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=tiny_route_workspace,
        block_expert_ids=tiny_route_workspace,
        packed_route_count=tiny_route_workspace,
        expert_offsets=tiny_route_workspace,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert any(saw_tc_decode), (
        f"TC-decode fused-sum path did not engage for m={m}; "
        f"compile calls: {saw_tc_decode}"
    )
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 2, 3, 6, 8])
def test_w4a16_tc_decode_preplanned_launch_matches_oracle(m: int) -> None:
    """The vLLM binding path passes a *preplanned* fused launch. Validate that a
    preplanned TC-decode launch (direct_topk_routes + tc_decode_fused_sum) is
    consumed correctly by run_w4a16_moe (contract validation + guard + epilogue)."""
    torch.manual_seed(20260529 + 100 + m)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, activation = 2, "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights, activation=activation, params_dtype=x.dtype
    )
    buffers = make_w4a16_buffers(
        prepared, m=m, topk=topk, dtype=x.dtype, device=x.device
    )
    tiny = torch.empty((1,), dtype=torch.int32, device=x.device)

    # Build the preplanned TC-decode launch exactly as the prewarm does.
    props = torch.cuda.get_device_properties(x.device)
    block_size_m = select_route_block_size_m(m, topk, experts)
    tc_launch = compile_w4a16_fused_moe(
        size_m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        top_k=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block_size_m,
        max_m_blocks=m * topk,
        element_dtype="bf16",
        sms=int(props.multi_processor_count),
        max_shared_mem=int(
            getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
        ),
        weight_layout="packed",
        scale_format="e4m3_k16",
        w13_layout="packed",
        direct_topk_routes=True,
        tc_decode_fused_sum=True,
    )
    assert bool(tc_launch.tc_decode_fused_sum)

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=tiny,
        block_expert_ids=tiny,
        packed_route_count=tiny,
        expert_offsets=tiny,
        fused_launch=tc_launch,
    )
    expected = _reference_w4a16(
        x, *weights, topk_ids, topk_weights, activation=activation
    )
    torch.cuda.synchronize()
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 4, 6])
def test_w4a16_small_m_packed_direct_topk_routes_matches_oracle(m: int) -> None:
    torch.manual_seed(20260524 + m)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, activation = 2, "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
    )
    tiny_route_workspace = torch.empty((1,), dtype=torch.int32, device=x.device)

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=tiny_route_workspace,
        block_expert_ids=tiny_route_workspace,
        packed_route_count=tiny_route_workspace,
        expert_offsets=tiny_route_workspace,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 8])
def test_w4a16_mapped_decode_consumes_global_map_without_route_pack(
    m: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    torch.manual_seed(20260730 + m)
    global_experts, local_experts = 12, 8
    hidden_size, intermediate_size = 128, 128
    topk, activation = 2, "silu"
    weights = _make_weights(
        experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
    mapped_global_ids = torch.tensor(
        [0, 2, 3, 5, 7, 8, 10, 11], dtype=torch.int32, device="cuda"
    )
    expert_map[mapped_global_ids] = torch.arange(
        local_experts, dtype=torch.int32, device="cuda"
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(
        0,
        global_experts,
        (m, topk),
        dtype=torch.int32,
        device="cuda",
    )
    # Exercise both a fully inactive token and mixed valid/inactive routes.
    topk_ids[0] = torch.tensor([1, 4], dtype=torch.int32, device="cuda")
    if m > 1:
        topk_ids[1] = torch.tensor([2, 6], dtype=torch.int32, device="cuda")
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
        route_num_experts=global_experts,
    )

    def run() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            expert_map=expert_map,
            fast_math=True,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
        )

    def route_pack_must_not_run(*args, **kwargs):
        raise AssertionError("mapped decode unexpectedly invoked route packing")

    compiled_launches = []
    compile_fused = w4a16_kernel.compile_w4a16_fused_moe

    def record_compile(*args, **kwargs):
        launch = compile_fused(*args, **kwargs)
        if kwargs.get("use_expert_map"):
            compiled_launches.append(launch)
        return launch

    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        expert_map=expert_map,
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            w4a16_kernel,
            "pack_topk_routes_by_expert",
            route_pack_must_not_run,
        )
        patch.setattr(w4a16_kernel, "compile_w4a16_fused_moe", record_compile)
        eager = run().clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
    torch.cuda.synchronize()

    _assert_matches_oracle(eager, expected, activation=activation)
    assert torch.equal(captured, eager)
    assert any(
        launch.use_expert_map
        and launch.direct_topk_routes
        and launch.tc_decode_fused_sum
        for launch in compiled_launches
    )




@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_activation_amax_calibration_tracks_routed_inputs_and_fc2_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(20260616)
    experts, hidden_size, intermediate_size = 4, 128, 128
    m, topk, activation = 3, 2, "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [0, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
    )
    activation_amax = torch.zeros((2, experts, 2), dtype=torch.float32, device="cuda")
    activation_amax[0].fill_(123.0)
    compile_calls: list[dict[str, object]] = []

    def spy_compile_w4a16_fused_moe(**kwargs):
        compile_calls.append(dict(kwargs))
        return compile_w4a16_fused_moe(**kwargs)

    monkeypatch.setattr(
        "b12x.moe._shared.kernels.w4a16.kernel.compile_w4a16_fused_moe",
        spy_compile_w4a16_fused_moe,
    )

    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        activation_amax=activation_amax,
        layer_idx=1,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert compile_calls
    assert any(bool(call["collect_activation_amax"]) for call in compile_calls)
    assert not any(bool(call["direct_topk_routes"]) for call in compile_calls)
    assert not any(bool(call["tc_decode_fused_sum"]) for call in compile_calls)
    _assert_matches_oracle(actual, expected, activation=activation)

    expected_amax = torch.zeros((experts, 2), dtype=torch.float32, device="cuda")
    activated_rows = buffers.intermediate_cache2.view(-1, intermediate_size)
    flat_ids = topk_ids.view(-1)
    route_tokens = torch.arange(m, device="cuda").repeat_interleave(topk)
    route_indices = torch.arange(m * topk, device="cuda")
    for expert in range(experts):
        mask = flat_ids == expert
        if bool(mask.any().item()):
            expected_amax[expert, 0] = x[route_tokens[mask]].float().abs().max()
            expected_amax[expert, 1] = (
                activated_rows[route_indices[mask]].float().abs().max()
            )

    torch.testing.assert_close(activation_amax[1], expected_amax, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        activation_amax[0],
        torch.full_like(activation_amax[0], 123.0),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_activation_amax_forces_main_kernel_over_native_small_m_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(20260617)
    experts, hidden_size, intermediate_size = 4, 128, 128
    rows = intermediate_size * 2
    m, topk, activation = 2, 2, "silu"
    w13 = torch.randint(
        0,
        256,
        (experts, rows, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _pattern_e8m0((experts, rows, hidden_size // 32))
    w2_scale = _pattern_e8m0((experts, hidden_size, intermediate_size // 32), offset=1)
    w13_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    w2_global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    assert _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        element_dtype="bf16",
        weight_layout="modelopt",
        w13_layout="w13",
        scale_format="e8m0_k32",
    )
    prepared = prepare_w4a16_e8m0_native_weights(
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    activation_amax = torch.zeros((1, experts, 2), dtype=torch.float32, device="cuda")
    compile_calls: list[dict[str, object]] = []

    def spy_compile_w4a16_fused_moe(**kwargs):
        compile_calls.append(dict(kwargs))
        return compile_w4a16_fused_moe(**kwargs)

    monkeypatch.setattr(
        "b12x.moe._shared.kernels.w4a16.kernel.compile_w4a16_fused_moe",
        spy_compile_w4a16_fused_moe,
    )

    actual = run_w4a16_moe(
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
        activation_amax=activation_amax,
        layer_idx=0,
    )
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        intermediate_size,
        activation=activation,
        w13_layout="w13",
    )
    torch.cuda.synchronize()

    assert compile_calls
    assert any(bool(call["collect_activation_amax"]) for call in compile_calls)
    assert not any(bool(call["direct_topk_routes"]) for call in compile_calls)
    assert not any(bool(call["tc_decode_fused_sum"]) for call in compile_calls)
    assert bool((activation_amax > 0).any().item())
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
@pytest.mark.parametrize(
    ("routed_size", "m"),
    [(8, 16), (16, 32), (32, 64), (48, 128), (64, 192)],
)
def test_w4a16_moe_matches_oracle(
    activation: str,
    routed_size: int,
    m: int,
) -> None:
    torch.manual_seed(20260515 + routed_size + (1000 if activation == "silu" else 0))
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk = 2
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    actual = _run_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_situ_moe_matches_oracle() -> None:
    torch.manual_seed(20260726)
    experts, hidden_size, intermediate_size = 8, 128, 128
    m, topk = 16, 2
    activation = "situ"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(
        0,
        experts,
        (m, topk),
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    actual = _run_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert actual.abs().sum().item() > 0
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_modelopt_nvfp4_prepare_moe_matches_oracle(
    activation: str,
) -> None:
    torch.manual_seed(20260523 + (1000 if activation == "silu" else 0))
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    actual = _run_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prepare_native", [False, True])
@pytest.mark.parametrize("w13_layout", ["up_gate", "gate_up"])
@pytest.mark.parametrize("parameter_scales", [False, True])
def test_w4a16_modelopt_nvfp4_explicit_w13_layout_matches_oracle(
    prepare_native: bool,
    w13_layout: str,
    parameter_scales: bool,
) -> None:
    """W13 physical order is an explicit input, independent of source_format."""
    torch.manual_seed(20260525 + (1000 if prepare_native else 0))
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    activation = "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale = weights
    if parameter_scales:
        w13_global_scale = torch.nn.Parameter(w13_global_scale, requires_grad=False)
        w2_global_scale = torch.nn.Parameter(w2_global_scale, requires_grad=False)
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    source_w13 = w13
    source_w13_blockscale = w13_blockscale
    if w13_layout == "gate_up":
        half = intermediate_size
        source_w13 = torch.cat([w13[:, half:], w13[:, :half]], dim=1).contiguous()
        source_w13_blockscale = torch.cat(
            [w13_blockscale[:, half:], w13_blockscale[:, :half]], dim=1
        ).contiguous()

    prepare = (
        prepare_w4a16_modelopt_native_weights
        if prepare_native
        else prepare_w4a16_weights
    )
    kwargs = {"source_format": "modelopt_nvfp4"} if prepare_native else {}
    prepared = prepare(
        source_w13,
        source_w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=x.dtype,
        w13_layout=w13_layout,
        **kwargs,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
    )
    actual = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_tp_moe_w4a16_modelopt_nvfp4_uses_normal_nvfp4_scale_contract(
    activation: str,
) -> None:
    torch.manual_seed(20260524 + (1000 if activation == "silu" else 0))
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale = weights
    w13_input_scale = (torch.rand(experts, device="cuda") * 2.0 + 1.5).to(torch.float32)
    w2_input_scale = (torch.rand(experts, device="cuda") * 2.0 + 1.5).to(torch.float32)
    a1_gscale = (1.0 / w13_input_scale).contiguous()
    a2_gscale = (1.0 / w2_input_scale).contiguous()

    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    output = torch.empty_like(x)
    prepared = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a1_gscale,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_alphas=w13_global_scale,
        a2_gscale=a2_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_global_scale,
        activation=activation,
        quant_mode="w4a16",
    )
    actual = run_tp_moe_fp4(
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        quant_mode="w4a16",
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert actual is output
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tp_moe_w4a16_prepared_reuse_path_is_deterministic_under_odd_shape_stress() -> (
    None
):
    torch.manual_seed(20260526)
    experts, hidden_size, intermediate_size = 8, 128, 128
    activation = "silu"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    reference_weights = tuple(t.clone() for t in weights)
    w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale = weights
    a_gscale = torch.ones(experts, dtype=torch.float32, device="cuda")
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="modelopt_nvfp4",
        activation=activation,
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=weight_plan,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_global_scale=w13_global_scale,
        a1_gscale=a_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_global_scale=w2_global_scale,
        a2_gscale=a_gscale,
        params_dtype=torch.bfloat16,
    )
    assert prepared.representation_for("w4a16") is not None

    cases = [
        (1, 1, 1, torch.int32, 0.25),
        (3, 5, 2, torch.int32, 0.50),
        (5, 7, 3, torch.int64, 0.75),
        (9, 11, 4, torch.int32, 1.00),
    ]
    for case_idx, (batch_size, seq_len, topk, ids_dtype, input_scale) in enumerate(
        cases
    ):
        m = batch_size * seq_len
        torch.manual_seed(2026052600 + case_idx)
        x = (torch.randn(m, hidden_size, device="cuda") * input_scale).to(
            torch.bfloat16
        )
        topk_ids = torch.randint(
            0,
            experts,
            (m, topk),
            device="cuda",
            dtype=ids_dtype,
        )
        topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
        expected = moe_reference_w4a16_f32(
            x,
            *reference_weights,
            topk_ids,
            topk_weights,
            experts,
            hidden_size,
            intermediate_size,
            activation=activation,
        )

        baseline = None
        for repeat in range(6):
            output = torch.empty_like(x)
            actual = run_tp_moe_fp4(
                a=x,
                experts=prepared,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=output,
                quant_mode="w4a16",
            )
            torch.cuda.synchronize()
            assert actual is output
            if baseline is None:
                baseline = output.detach().clone()
                continue
            if not torch.equal(output, baseline):
                max_abs = (output.float() - baseline.float()).abs().max().item()
                raise AssertionError(
                    "W4A16 prepared reuse path changed output for "
                    f"case={case_idx}, repeat={repeat}, m={m}, topk={topk}, "
                    f"ids_dtype={ids_dtype}, max_abs={max_abs}"
                )

        assert baseline is not None
        metrics = compare_to_reference(baseline, expected)
        assert metrics.cos > 0.999, (
            metrics,
            batch_size,
            seq_len,
            topk,
            ids_dtype,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_w4a16_moe_matches_oracle_with_expert_map(
    activation: str,
) -> None:
    torch.manual_seed(20260516 + (1000 if activation == "silu" else 0))
    global_experts, local_experts = 8, 4
    hidden_size, intermediate_size = 128, 128
    topk, m = 2, 24
    weights = _make_weights(
        experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
    expert_map[::2] = torch.arange(local_experts, dtype=torch.int32, device="cuda")

    valid_global_ids = torch.arange(
        0, global_experts, 2, dtype=torch.int32, device="cuda"
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = valid_global_ids[
        torch.randint(0, local_experts, (m, topk), device="cuda")
    ].to(torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    actual = _run_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        expert_map=expert_map,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        expert_map=expert_map,
    )
    torch.cuda.synchronize()

    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_moe_swiglu_limit_matches_oracle_under_cuda_graph() -> None:
    torch.manual_seed(20260519)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    activation = "silu"
    swiglu_limit = 10.0
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, _, w2, w2_blockscale, w2_global_scale = weights
    w13_global_scale = torch.full((experts,), 8.0, dtype=torch.float32, device="cuda")
    weights = (
        w13,
        w13_blockscale,
        w13_global_scale,
        w2,
        w2_blockscale,
        w2_global_scale,
    )
    x = (torch.randn(m, hidden_size, device="cuda") * 2.0).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=x.shape[0],
        topk=topk_ids.shape[1],
        dtype=x.dtype,
        device=x.device,
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )

    eager = run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        fast_math=True,
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
        swiglu_limit=swiglu_limit,
    )
    torch.cuda.synchronize()
    _assert_matches_oracle(eager, expected, activation=activation)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            fast_math=True,
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
            swiglu_limit=swiglu_limit,
        )
    graph.replay()
    torch.cuda.synchronize()

    _assert_matches_oracle(buffers.output, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_preplanned_capacity_launch_accepts_smaller_live_m() -> None:
    torch.manual_seed(20260522)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, live_m, capacity_m = 2, 24, 32
    activation = "relu2"
    assert select_route_block_size_m(
        live_m, topk, experts
    ) != select_route_block_size_m(capacity_m, topk, experts)
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    x = (torch.randn(live_m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(
        0, experts, (live_m, topk), device="cuda", dtype=torch.int32
    )
    topk_weights = torch.softmax(torch.randn(live_m, topk, device="cuda"), dim=-1)
    prepared = prepare_w4a16_weights(
        *weights,
        activation=activation,
        params_dtype=x.dtype,
    )
    buffers = make_w4a16_buffers(
        prepared,
        m=capacity_m,
        topk=topk,
        dtype=x.dtype,
        device=x.device,
    )
    props = torch.cuda.get_device_properties(x.device)
    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )
    block_size_m = select_route_block_size_m(capacity_m, topk, experts)
    route_slots = max_packed_route_slots(capacity_m * topk, block_size_m, experts)
    max_m_blocks = (route_slots + block_size_m - 1) // block_size_m
    fused_launch = compile_w4a16_fused_moe(
        size_m=capacity_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        top_k=topk,
        activation=activation,
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block_size_m,
        max_m_blocks=max_m_blocks,
        element_dtype="bf16",
        sms=int(props.multi_processor_count),
        max_shared_mem=max_shared_mem,
    )
    topk_sum_launch = compile_w4a16_topk_sum(
        m=capacity_m,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype="bf16",
    )
    output = torch.empty_like(x)

    def _run(output_buffer: torch.Tensor) -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            fast_math=True,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=output_buffer,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            fused_launch=fused_launch,
            topk_sum_launch=topk_sum_launch,
        )

    actual = _run(output)
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert actual is output
    _assert_matches_oracle(actual, expected, activation=activation)

    graph_output = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        _run(graph_output)
    graph.replay()
    torch.cuda.synchronize()

    _assert_matches_oracle(graph_output, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tp_moe_w4a16_dispatch_uses_native_path() -> None:
    torch.manual_seed(20260518)
    experts, hidden_size, intermediate_size = 8, 128, 128
    topk, m = 2, 24
    activation = "relu2"
    weights = _make_weights(
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    w13, w13_blockscale, w13_global_scale, w2, w2_blockscale, w2_global_scale = weights

    x = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    topk_ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    output = torch.empty_like(x)
    a_gscale = torch.ones((), dtype=torch.float32, device="cuda")
    prepared = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a_gscale,
        w1_fp4=w13,
        w1_blockscale=w13_blockscale,
        w1_alphas=w13_global_scale,
        a2_gscale=a_gscale,
        w2_fp4=w2,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_global_scale,
        activation=activation,
        quant_mode="w4a16",
    )
    actual = run_tp_moe_fp4(
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        quant_mode="w4a16",
    )
    expected = _reference_w4a16(
        x,
        *weights,
        topk_ids,
        topk_weights,
        activation=activation,
    )
    torch.cuda.synchronize()

    assert actual is output
    _assert_matches_oracle(actual, expected, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_fc2_only_consumes_contiguous_bf16_and_native_mxfp4() -> None:
    experts, hidden_size, intermediate_size, routes = 2, 128, 256, 3
    # MXFP4 nibbles 1 and 2 decode to 0.5 and 1.0 at unit UE8M0 scale.
    w2 = torch.empty(
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2[0].fill_(0x11)
    w2[1].fill_(0x22)
    scales = torch.full(
        (experts, hidden_size, intermediate_size // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    intermediate = torch.ones(
        (routes, intermediate_size), dtype=torch.bfloat16, device="cuda"
    )
    route_ids = torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda")
    route_weights = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32, device="cuda")

    prepared = prepare_w4a16_fc2_e8m0(w2, scales)
    actual = run_w4a16_fc2_e8m0(
        intermediate,
        prepared,
        route_ids,
        route_weights,
    )
    expected_values = torch.tensor(
        [32.0, 128.0, 128.0], dtype=torch.bfloat16, device="cuda"
    )
    expected = expected_values[:, None].expand_as(actual)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("intermediate_size", [256, 512])
@pytest.mark.parametrize("route_ids_dtype", [torch.int32, torch.int64])
def test_w4a16_fc2_only_zeroes_invalid_routes_at_runtime_m3(
    intermediate_size: int,
    route_ids_dtype: torch.dtype,
) -> None:
    experts, hidden_size, routes = 2, 128, 3
    w2 = torch.empty(
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2[0].fill_(0x11)
    w2[1].fill_(0x22)
    scales = torch.full(
        (experts, hidden_size, intermediate_size // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    intermediate = torch.ones(
        (routes, intermediate_size), dtype=torch.bfloat16, device="cuda"
    )
    invalid_upper = 1 << 32 if route_ids_dtype == torch.int64 else experts
    route_ids = torch.tensor(
        [0, -1, invalid_upper], dtype=route_ids_dtype, device="cuda"
    )
    route_weights = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32, device="cuda")
    original_ids = route_ids.clone()
    original_weights = route_weights.clone()

    prepared = prepare_w4a16_fc2_e8m0(w2, scales)
    actual = run_w4a16_fc2_e8m0(
        intermediate,
        prepared,
        route_ids,
        route_weights,
    )
    expected_values = torch.tensor(
        [intermediate_size / 8.0, 0.0, 0.0],
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = expected_values[:, None].expand_as(actual)

    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(route_ids, original_ids, rtol=0, atol=0)
    torch.testing.assert_close(route_weights, original_weights, rtol=0, atol=0)

    graph_output = torch.empty_like(actual)
    prewarm_w4a16_fc2_e8m0(prepared, route_ids_dtype=route_ids_dtype)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_w4a16_fc2_e8m0(
            intermediate,
            prepared,
            route_ids,
            route_weights,
            output=graph_output,
        )
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert captured is graph_output
    assert bool(torch.isfinite(captured).all().item())
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    torch.testing.assert_close(route_ids, original_ids, rtol=0, atol=0)
    torch.testing.assert_close(route_weights, original_weights, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("intermediate_size", [256, 512])
@pytest.mark.parametrize("route_ids_dtype", [torch.int32, torch.int64])
def test_w4a16_fc2_resident_bounds_reuse_frozen_kernel(
    intermediate_size: int,
    route_ids_dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import b12x
    import b12x.moe._shared.kernels.w4a16.kernel as w4a16_kernel

    hidden_size, capacity_m = 128, 9
    prepared = []
    for num_experts in (2, 5):
        w2 = torch.full(
            (num_experts, hidden_size, intermediate_size // 2),
            0x11,
            dtype=torch.uint8,
            device="cuda",
        )
        scales = torch.full(
            (num_experts, hidden_size, intermediate_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        )
        prepared.append(prepare_w4a16_fc2_e8m0(w2, scales))

    monkeypatch.setattr(w4a16_kernel, "_FC2_DIRECT_CACHE", {})
    prewarm_w4a16_fc2_e8m0(prepared[0], route_ids_dtype=route_ids_dtype)
    (launch,) = w4a16_kernel._FC2_DIRECT_CACHE.values()
    intermediate = torch.empty(
        (capacity_m, intermediate_size), dtype=torch.bfloat16, device="cuda"
    )
    route_ids = torch.empty(capacity_m, dtype=route_ids_dtype, device="cuda")
    route_weights = torch.empty(capacity_m, dtype=torch.float32, device="cuda")
    output = torch.empty(
        (capacity_m, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    invalid_upper = 1 << 32 if route_ids_dtype == torch.int64 else 5
    mutated_ids = [0, 2, -1, 4, 5, invalid_upper, 1, 0, 4]
    ids_source = torch.tensor(mutated_ids, dtype=route_ids_dtype, device="cuda")
    buffers = (intermediate, route_ids, route_weights, output)
    addresses = tuple(buffer.data_ptr() for buffer in buffers)
    workspace_contracts = [
        (item.workspace.data_ptr(), item.workspace.numel()) for item in prepared
    ]

    with kernel_resolution_guard('FC2 resident bounds and live row reuse'):
        for experts in prepared:
            prewarm_w4a16_fc2_e8m0(experts, route_ids_dtype=route_ids_dtype)
            for m in (1, 3, capacity_m):
                intermediate.fill_(1)
                route_ids.zero_()
                route_weights.fill_(0.25)
                args = (intermediate[:m], experts, route_ids[:m], route_weights[:m])
                live_output = output[:m]
                valid_expected = torch.full_like(live_output, intermediate_size / 8)
                values = [
                    intermediate_size / 2 if 0 <= i < experts.num_experts else 0
                    for i in mutated_ids[:m]
                ]
                mutated_expected = torch.tensor(
                    values, dtype=torch.bfloat16, device="cuda"
                )[:, None].expand_as(live_output)
                eager = run_w4a16_fc2_e8m0(*args, output=live_output)
                assert eager is live_output
                torch.testing.assert_close(eager, valid_expected, rtol=0, atol=0)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run_w4a16_fc2_e8m0(*args, output=live_output)
                intermediate.fill_(2)
                route_weights.fill_(0.5)
                route_ids.copy_(ids_source)
                output.fill_(float("nan"))
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                allocated_bytes = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                assert torch.cuda.memory_allocated() == allocated_bytes
                assert captured is live_output
                torch.testing.assert_close(captured, mutated_expected, rtol=0, atol=0)
                torch.testing.assert_close(route_ids, ids_source, rtol=0, atol=0)
                assert bool((route_weights == 0.5).all().item())
                assert tuple(buffer.data_ptr() for buffer in buffers) == addresses
                graph.reset()
            (cached,) = w4a16_kernel._FC2_DIRECT_CACHE.values()
            assert cached.compiled is launch.compiled
        assert [
            (item.workspace.data_ptr(), item.workspace.numel()) for item in prepared
        ] == workspace_contracts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_fc2_only_is_cuda_graph_safe_with_preallocated_output() -> None:
    experts, hidden_size, intermediate_size, routes = 5, 128, 256, 7
    w2 = torch.empty(
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    for expert in range(experts):
        w2[expert].fill_((expert + 1) | ((expert + 1) << 4))
    scales = torch.full(
        (experts, hidden_size, intermediate_size // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    intermediate = torch.ones(
        (routes, intermediate_size), dtype=torch.bfloat16, device="cuda"
    )
    route_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.int32, device="cuda")
    route_weights = torch.linspace(
        0.125, 0.875, routes, dtype=torch.float32, device="cuda"
    )
    output = torch.empty((routes, hidden_size), dtype=torch.bfloat16, device="cuda")

    prepared = prepare_w4a16_fc2_e8m0(w2, scales)
    prewarm_w4a16_fc2_e8m0(prepared, route_ids_dtype=torch.int32)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_w4a16_fc2_e8m0(
            intermediate,
            prepared,
            route_ids,
            route_weights,
            output=output,
        )
    graph.replay()
    torch.cuda.synchronize()

    valid_values = (
        256.0
        * route_weights
        * torch.where(
            route_ids == 0,
            torch.tensor(0.5, device="cuda"),
            torch.tensor(1.0, device="cuda"),
        )
    )
    valid_expected = valid_values.to(torch.bfloat16)[:, None].expand_as(output)
    torch.testing.assert_close(captured, valid_expected, rtol=0, atol=0)

    invalid_ids = torch.tensor(
        [0, -1, 1, experts, 0, -9, 1], dtype=torch.int32, device="cuda"
    )
    route_ids.copy_(invalid_ids)
    original_weights = route_weights.clone()
    invalid_values = torch.tensor(
        [16.0, 0.0, 96.0, 0.0, 80.0, 0.0, 224.0],
        dtype=torch.bfloat16,
        device="cuda",
    )
    invalid_expected = invalid_values[:, None].expand_as(output)

    eager = torch.empty_like(output)
    run_w4a16_fc2_e8m0(
        intermediate,
        prepared,
        route_ids,
        route_weights,
        output=eager,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(eager, invalid_expected, rtol=0, atol=0)

    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert captured is output
    assert bool(torch.isfinite(captured).all().item())
    torch.testing.assert_close(captured, invalid_expected, rtol=0, atol=0)
    torch.testing.assert_close(route_ids, invalid_ids, rtol=0, atol=0)
    torch.testing.assert_close(route_weights, original_weights, rtol=0, atol=0)
