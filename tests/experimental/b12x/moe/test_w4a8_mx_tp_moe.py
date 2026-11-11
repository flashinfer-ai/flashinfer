"""End-to-end w4a8_mx dispatch through b12x_moe_fp4 on synthetic MXFP4 weights.

([E, rows, K//32] bytes) feed the w4a8_mx kernels directly — no vec16 scale
stack, no residual grids. Tiny decode uses compile-time direct regimes inside
the unified dynamic kernel and consumes only its prepared weight layout.
"""

from __future__ import annotations

from contextlib import ExitStack

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.experimental.b12x.benchmark_ds4_moe import make_synthetic_mxfp4_moe

_E = 16
_K = 4096
_N = 1024
_TOPK = 4


@pytest.mark.parametrize(
    ("routed_rows", "expected_tile_m"),
    [
        (16 * _E, 16),
        (16 * _E + 1, 32),
        (36 * _E - 1, 32),
        (36 * _E, 64),
    ],
)
def test_w4a8_mx_dynamic_tile_density_boundaries(
    monkeypatch, routed_rows: int, expected_tile_m: int
) -> None:
    import b12x.moe.fused_moe._impl as tp_moe

    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    assert tp_moe._select_dynamic_tile_mn(
        routed_rows,
        _N,
        "w4a8_mx",
        num_experts=_E,
        activation="silu",
        compute_capability=(12, 0),
    ) == (expected_tile_m, 128)


@pytest.mark.parametrize("routed_rows", [384, 768, 1536, 2304])
def test_w4a8_mx_ds4_tp2_batch_m_uses_m32(monkeypatch, routed_rows: int) -> None:
    import b12x.moe.fused_moe._impl as tp_moe

    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    assert tp_moe._select_dynamic_tile_mn(
        routed_rows,
        1024,
        "w4a8_mx",
        num_experts=256,
        activation="silu",
    ) == (32, 128)


@pytest.mark.parametrize("routed_rows", [383, 2305])
def test_w4a8_mx_ds4_tp2_batch_m_tactic_is_band_limited(
    monkeypatch, routed_rows: int
) -> None:
    import b12x.moe.fused_moe._impl as tp_moe

    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    assert tp_moe._select_dynamic_tile_mn(
        routed_rows,
        1024,
        "w4a8_mx",
        num_experts=256,
        activation="silu",
        compute_capability=(12, 0),
    ) == (16, 128)


@pytest.mark.parametrize("m", [1024, 4096, 8192, 16384])
def test_w4a8_mx_ds4_tp2_sm121_prefill_uses_fused_m32(monkeypatch, m: int) -> None:
    import b12x.moe.fused_moe._impl as tp_moe

    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    assert tp_moe._select_dynamic_tile_mn(
        m * 6,
        1024,
        "w4a8_mx",
        num_experts=256,
        activation="silu",
        compute_capability=(12, 1),
    ) == (32, 128)


@pytest.mark.parametrize("m", [4096, 8192, 16384])
def test_w4a8_mx_ds4_tp2_sm120_prefill_keeps_coarse_tactic(monkeypatch, m: int) -> None:
    import b12x.moe.fused_moe._impl as tp_moe

    monkeypatch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
    assert tp_moe._select_dynamic_tile_mn(
        m * 6,
        1024,
        "w4a8_mx",
        num_experts=256,
        activation="silu",
        compute_capability=(12, 0),
    ) == (64, 128)


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")


def _weights(*, n: int = _N, seed: int = 21):
    """Create one checkpoint allocation whose ownership may be transferred."""

    return make_synthetic_mxfp4_moe(_E, _K, n, seed=seed, device=torch.device("cuda"))


def _prepare(
    weights: dict,
    *,
    n: int = _N,
    w13_layout: str = "w13",
    activation: str = "silu",
):
    """Destructively turn checkpoint storage into the sole runtime layout."""

    from b12x.moe.fused_moe._impl import (
        plan_b12x_fp4_moe_weights,
        prepare_b12x_fp4_moe_weights,
    )

    plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation=activation,
        params_dtype=torch.bfloat16,
        num_experts=_E,
        hidden_size=_K,
        intermediate_size=n,
        w13_layout=w13_layout,
    )
    source_ptrs = tuple(
        weights[name].untyped_storage().data_ptr()
        for name in ("w13_fp4", "w13_mx", "w2_fp4", "w2_mx")
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=plan,
        w1_fp4=weights["w13_fp4"],
        w1_blockscale=weights["w13_mx"],
        w1_global_scale=weights["alphas"],
        a1_gscale=weights["input_scale"],
        w2_fp4=weights["w2_fp4"],
        w2_blockscale=weights["w2_mx"],
        w2_global_scale=weights["alphas"],
        a2_gscale=weights["input_scale"],
        params_dtype=torch.bfloat16,
    )
    runtime = prepared.representation_for("w4a8_mx")
    runtime_ptrs = tuple(
        tensor.untyped_storage().data_ptr()
        for tensor in (
            runtime.w13_rp,
            runtime.w13_sfb,
            runtime.w2_rp,
            runtime.w2_sfb,
        )
    )
    # Exact N64 repacks preserve the source storage and specialize the runtime
    # layout in place. Legacy ceil-tiled tails require larger allocations.
    has_legacy_tail = (n % 128 != 0 or (2 * n) % 256 != 0) and not runtime.n64_repack
    if has_legacy_tail:
        assert runtime_ptrs != source_ptrs
    else:
        assert runtime_ptrs == source_ptrs
    return prepared


def _routed_inputs(m: int, seed: int):
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = (torch.randn(m, _K, generator=gen, device=device) * 2.0).to(torch.bfloat16)
    logits = torch.randn(m, _E, generator=gen, device=device)
    topk_logits, topk_ids = torch.topk(logits, _TOPK, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).float()
    return x, topk_ids.to(torch.int32), topk_weights


def _run(
    m: int,
    experts,
    *,
    seed: int = 33,
) -> torch.Tensor:
    from b12x.moe.fused_moe._impl import clear_tp_moe_caches
    from b12x.testing.reference.helpers import run_tp_moe_fp4

    clear_tp_moe_caches()
    x, topk_ids, topk_weights = _routed_inputs(m, seed)
    out = run_tp_moe_fp4(
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("n", [_N, 384, 352, 192, 320])
def test_w4a8_mx_dynamic_matches_oracle(n: int) -> None:
    _skip_if_unavailable()
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx

    weights = _weights(n=n)
    m = 16
    x, topk_ids, topk_weights = _routed_inputs(m, 33)
    ref = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        n,
        activation="silu",
    )
    prepared = _prepare(weights, n=n)
    out = _run(m, prepared)
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"w4a8_mx output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)


@pytest.mark.parametrize("m", [8, 144])
def test_w4a8_mx_situ_public_path_matches_oracle(m: int) -> None:
    _skip_if_unavailable()
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx

    n = 128
    weights = _weights(n=n, seed=20260726)
    x, topk_ids, topk_weights = _routed_inputs(m, 33)
    ref = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        n,
        activation="situ",
    )
    prepared = _prepare(weights, n=n, activation="situ")
    out = _run(m, prepared)

    assert out.abs().sum().item() > 0
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(),
        ref.float().flatten(),
        dim=0,
    ).item()
    assert cos > 0.998, cos


@pytest.mark.parametrize("m", [1, 144])
def test_w4a8_mx_situ_cuda_graph_replay_matches_oracle(m: int) -> None:
    _skip_if_unavailable()
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
    from b12x.moe.fused_moe._impl import b12x_moe_fp4
    from b12x.testing.reference.helpers import make_tp_moe_fp4_binding

    n = 128
    weights = _weights(n=n, seed=20260728)
    x, topk_ids, topk_weights = _routed_inputs(m, 20260729)
    new_x, new_ids, new_weights = _routed_inputs(m, 20260730)
    expected = moe_reference_w4a8_mx(
        new_x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        new_ids,
        new_weights,
        _E,
        _K,
        n,
        activation="situ",
    )
    prepared = _prepare(weights, n=n, activation="situ")
    output = torch.zeros(m, _K, dtype=torch.bfloat16, device="cuda")
    bindings = ExitStack()
    binding = bindings.enter_context(
        make_tp_moe_fp4_binding(
            a=x,
            experts=prepared,
            topk_weights=topk_weights.contiguous(),
            topk_ids=topk_ids.contiguous(),
            output=output,
            input_scales_static=True,
            quant_mode="w4a8_mx",
        )
    )

    b12x_moe_fp4(binding=binding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        b12x_moe_fp4(binding=binding)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    x.copy_(new_x)
    topk_ids.copy_(new_ids)
    topk_weights.copy_(new_weights)
    graph.replay()
    torch.cuda.synchronize()

    assert output.abs().sum().item() > 0
    cos = torch.nn.functional.cosine_similarity(
        output.float().flatten(),
        expected.float().flatten(),
        dim=0,
    ).item()
    assert cos > 0.998, cos
    del graph
    bindings.close()


def test_w4a8_mx_kimi_k3_tp8_situ_reuses_checkpoint_storage() -> None:
    """K3 TP8: K=3584, N=3072/8=384, gate/up checkpoint order."""

    _skip_if_unavailable()
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
    from b12x.moe.fused_moe import (
        plan_weights,
        prepare_weights,
        run,
    )
    from b12x.testing.reference.helpers import make_tp_moe_fp4_binding

    experts_count = 16
    hidden_size = 3584
    intermediate_size = 384
    topk = 16
    m = 8
    device = torch.device("cuda")
    weights = make_synthetic_mxfp4_moe(
        experts_count,
        hidden_size,
        intermediate_size,
        seed=20260731,
        device=device,
    )

    # The synthetic helper emits [up; gate]. K3 stores [w1/gate; w3/up].
    for name in ("w13_fp4", "w13_mx"):
        tensor = weights[name]
        first = tensor[:, :intermediate_size].clone()
        tensor[:, :intermediate_size].copy_(tensor[:, intermediate_size:])
        tensor[:, intermediate_size:].copy_(first)

    gen = torch.Generator(device=device)
    gen.manual_seed(20260801)
    x = torch.randn(
        m,
        hidden_size,
        dtype=torch.bfloat16,
        generator=gen,
        device=device,
    )
    logits = torch.randn(m, experts_count, generator=gen, device=device)
    topk_logits, topk_ids = torch.topk(logits, topk, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).float().contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    expected = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        experts_count,
        hidden_size,
        intermediate_size,
        activation="situ",
        w13_layout="w31",
    )
    source_ptrs = tuple(
        weights[name].untyped_storage().data_ptr()
        for name in ("w13_fp4", "w13_mx", "w2_fp4", "w2_mx")
    )
    weight_plan = plan_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="situ",
        params_dtype=torch.bfloat16,
        num_experts=experts_count,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w13_layout="w31",
    )
    prepared = prepare_weights(
        plan=weight_plan,
        w1_fp4=weights["w13_fp4"],
        w1_blockscale=weights["w13_mx"],
        w1_global_scale=weights["alphas"],
        a1_gscale=weights["input_scale"],
        w2_fp4=weights["w2_fp4"],
        w2_blockscale=weights["w2_mx"],
        w2_global_scale=weights["alphas"],
        a2_gscale=weights["input_scale"],
        params_dtype=torch.bfloat16,
    )
    runtime = prepared.representation_for("w4a8_mx")
    runtime_ptrs = tuple(
        tensor.untyped_storage().data_ptr()
        for tensor in (
            runtime.w13_rp,
            runtime.w13_sfb,
            runtime.w2_rp,
            runtime.w2_sfb,
        )
    )
    assert runtime_ptrs == source_ptrs

    output = torch.zeros(m, hidden_size, dtype=torch.bfloat16, device=device)
    bindings = ExitStack()
    binding = bindings.enter_context(
        make_tp_moe_fp4_binding(
            a=x,
            experts=prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            input_scales_static=True,
            quant_mode="w4a8_mx",
        )
    )
    run(binding=binding)
    torch.cuda.synchronize()

    assert output.isfinite().all()
    assert output.abs().sum().item() > 0
    cos = torch.nn.functional.cosine_similarity(
        output.float().flatten(),
        expected.float().flatten(),
        dim=0,
    ).item()
    assert cos > 0.998, cos

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        run(binding=binding)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    replay_cos = torch.nn.functional.cosine_similarity(
        output.float().flatten(),
        expected.float().flatten(),
        dim=0,
    ).item()
    assert replay_cos > 0.998, replay_cos
    del graph
    bindings.close()


@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_w4a8_mx_w31_layout_flip(activation: str) -> None:
    _skip_if_unavailable()
    weights = _weights()
    m = 16
    prepared = _prepare(weights, activation=activation)
    baseline = _run(m, prepared)
    repeat = _run(m, prepared)

    # Release the baseline owner, then build W31 from a fresh checkpoint.  The
    # half-row temporary avoids retaining complete W13 and W31 models together.
    del prepared, weights
    torch.cuda.empty_cache()
    weights = _weights()
    u8 = weights["w13_fp4"]
    tmp = u8[:, :_N].clone()
    u8[:, :_N].copy_(u8[:, _N:])
    u8[:, _N:].copy_(tmp)
    del tmp
    mx = weights["w13_mx"]
    tmp = mx[:, :_N].clone()
    mx[:, :_N].copy_(mx[:, _N:])
    mx[:, _N:].copy_(tmp)
    del tmp
    prepared = _prepare(weights, w13_layout="w31", activation=activation)
    flipped = _run(m, prepared)
    # Idempotency: a second pass over the same storage must not re-flip.
    flipped2 = _run(m, prepared)

    noise = (baseline.float() - repeat.float()).abs().max().item()
    bound = max(8.0 * noise, 1e-6 * baseline.float().abs().max().item())
    for label, got in (("first", flipped), ("repeat", flipped2)):
        err = (got.float() - baseline.float()).abs().max().item()
        assert err <= bound, (
            f"w31 flip mismatch ({label}): err={err} noise={noise} bound={bound}"
        )


@pytest.mark.parametrize("m", [1, 2, 3, 4])
# n=384 covers odd rp K-tile counts on FC2; 352/192/320 cover ceil-tiled
# tails (GLM-5.2 2048/TP6, DS4-Pro 3072/TP16 and 3072/TP10 native shards).
@pytest.mark.parametrize("n", [_N, 384, 352, 192, 320])
def test_w4a8_mx_small_band_matches_fp32_oracle(m: int, n: int) -> None:
    _skip_if_unavailable()
    from b12x.moe.fused_moe._impl import plan_b12x_fp4_moe_weights
    import b12x.moe.fused_moe._impl as tp_moe
    from b12x.moe._shared.kernels.reference import (
        moe_reference_w4a16_fp4_e8m0_k32,
    )

    weights = _weights(n=n)
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=_E,
        hidden_size=_K,
        intermediate_size=n,
    )
    plan = tp_moe.plan_tp_moe_execution(
        num_tokens=m,
        num_topk=_TOPK,
        device=torch.device("cuda"),
        weight_plan=weight_plan,
        quant_mode="w4a8_mx",
    )
    assert plan.implementation == "micro"

    x, topk_ids, topk_weights = _routed_inputs(m, 33)
    ref = moe_reference_w4a16_fp4_e8m0_k32(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        n,
        activation="silu",
        w13_layout="w13",
    )
    prepared = _prepare(weights, n=n)
    out = _run(m, prepared)
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"w4a8_mx tiny output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)


# ---------------------------------------------------------------------------
# Unified dynamic W4A8
# ---------------------------------------------------------------------------


def _oracle(m: int, weights: dict, seed: int = 33, *, n: int = _N):
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx

    x, topk_ids, topk_weights = _routed_inputs(m, seed)
    return moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        n,
        activation="silu",
    )


def test_w4a8_mx_dense_band_defaults_to_dynamic_and_matches_oracle() -> None:
    """The planner-selected dense band runs unified dynamic and matches its oracle."""
    _skip_if_unavailable()
    weights = _weights()
    m = 1024
    ref = _oracle(m, weights)
    prepared = _prepare(weights)
    out = _run(m, prepared)
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"dynamic output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)


@pytest.mark.parametrize("tile_m", [64, 128])
def test_w4a8_mx_materialized_dense_override_matches_oracle(
    monkeypatch, tile_m: int
) -> None:
    """The split dense-prefill specializations remain correct when forced."""

    _skip_if_unavailable()
    monkeypatch.setenv("B12X_DYNAMIC_TILE_MN", f"{tile_m}x128")
    weights = _weights()
    m = 1024
    ref = _oracle(m, weights)
    prepared = _prepare(weights)
    out = _run(m, prepared)
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"M{tile_m} dynamic output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)


def test_w4a8_mx_prepared_dynamic_runs_with_compacted_sources() -> None:
    """The serving representation must not retain logical checkpoint weights."""

    _skip_if_unavailable()
    from b12x.moe.fused_moe._impl import (
        plan_b12x_fp4_moe_weights,
        prepare_b12x_fp4_moe_weights,
    )
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
    from b12x.testing.reference.helpers import run_tp_moe_fp4

    n = 256
    weights = _weights(n=n, seed=91)
    m = 16
    x, topk_ids, topk_weights = _routed_inputs(m, 92)
    ref = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        n,
        activation="silu",
    )
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=_E,
        hidden_size=_K,
        intermediate_size=n,
        w13_layout="w13",
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=weight_plan,
        w1_fp4=weights["w13_fp4"],
        w1_blockscale=weights["w13_mx"],
        w1_global_scale=weights["alphas"],
        a1_gscale=weights["input_scale"],
        w2_fp4=weights["w2_fp4"],
        w2_blockscale=weights["w2_mx"],
        w2_global_scale=weights["alphas"],
        a2_gscale=weights["input_scale"],
        params_dtype=torch.bfloat16,
    )
    runtime = prepared.representation_for("w4a8_mx")
    assert tuple(
        tensor.untyped_storage().data_ptr()
        for tensor in (
            runtime.w13_rp,
            runtime.w13_sfb,
            runtime.w2_rp,
            runtime.w2_sfb,
        )
    ) == tuple(
        weights[name].untyped_storage().data_ptr()
        for name in ("w13_fp4", "w13_mx", "w2_fp4", "w2_mx")
    )

    out = run_tp_moe_fp4(
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )
    torch.cuda.synchronize()

    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos


def test_w4a8_mx_dynamic_graph_replay_tracks_routing_updates() -> None:
    """Capture unified dynamic in a CUDA graph (caller-owned output buffer);
    replay with routing/activations mutated IN PLACE must track the update
    (vs a fresh eager call on the same inputs, and vs the oracle)."""
    _skip_if_unavailable()
    from b12x.moe.fused_moe._impl import b12x_moe_fp4, clear_tp_moe_caches
    from b12x.testing.reference.helpers import make_tp_moe_fp4_binding

    clear_tp_moe_caches()
    device = torch.device("cuda")
    weights = _weights()
    m = 256
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx

    initial_x, initial_ids, initial_w = _routed_inputs(m, 33)
    rounds = []
    for round_idx in range(3):
        new_x, new_ids, new_w = _routed_inputs(m, 500 + round_idx)
        ref = moe_reference_w4a8_mx(
            new_x.float(),
            weights["w13_fp4"],
            weights["w13_mx"],
            None,
            weights["alphas"],
            weights["w2_fp4"],
            weights["w2_mx"],
            None,
            weights["alphas"],
            new_ids,
            new_w,
            _E,
            _K,
            _N,
            activation="silu",
        )
        rounds.append((new_x, new_ids, new_w, ref))
    prepared = _prepare(weights)
    x, topk_ids, topk_weights = initial_x, initial_ids, initial_w
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    graph_out = torch.zeros(m, _K, dtype=torch.bfloat16, device=device)
    eager_out = torch.zeros_like(graph_out)

    bindings = ExitStack()

    def _make_binding(out: torch.Tensor):
        return bindings.enter_context(
            make_tp_moe_fp4_binding(
                a=x,
                experts=prepared,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=out,
                input_scales_static=True,
                quant_mode="w4a8_mx",
            )
        )

    graph_binding = _make_binding(graph_out)
    eager_binding = _make_binding(eager_out)

    def _launch(binding) -> None:
        b12x_moe_fp4(binding=binding)

    # Warm the compiled dynamic launch, then capture.
    _launch(graph_binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        _launch(graph_binding)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    for round_idx, (new_x, new_ids, new_w, ref) in enumerate(rounds):
        x.copy_(new_x)
        topk_ids.copy_(new_ids)
        topk_weights.copy_(new_w)
        graph.replay()
        torch.cuda.synchronize()
        replayed = graph_out.clone()

        eager_out.zero_()
        _launch(eager_binding)
        torch.cuda.synchronize()

        assert replayed.abs().sum().item() > 0, round_idx
        replay_cos = torch.nn.functional.cosine_similarity(
            replayed.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        eager_cos = torch.nn.functional.cosine_similarity(
            eager_out.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        assert eager_cos > 0.998, (round_idx, "eager", eager_cos)
        assert replay_cos > 0.998, (round_idx, "replay", replay_cos)
        replay_eager_cos = torch.nn.functional.cosine_similarity(
            replayed.float().flatten(), eager_out.float().flatten(), dim=0
        ).item()
        assert replay_eager_cos > 0.9999, (round_idx, replay_eager_cos)
    del graph
    bindings.close()


def test_w4a8_mx_m9_graph_replay_with_aux_stream_work() -> None:
    """The M=9 two-CTA/SM grid must remain valid beside aux-stream work.

    Dynamic MoE synchronizes all CTAs between routing and compute phases.  At
    M=9 the W4A8 dispatcher uses the full two-CTA-per-SM resident grid (376
    CTAs on RTX 6000 Pro Blackwell), which requires a cooperative launch when
    shared-expert work is active on another stream.
    """
    _skip_if_unavailable()
    from b12x.moe.fused_moe._impl import b12x_moe_fp4, clear_tp_moe_caches
    from b12x.testing.reference.helpers import make_tp_moe_fp4_binding

    clear_tp_moe_caches()
    device = torch.device("cuda")
    m = 9
    weights = _weights(seed=91)
    prepared = _prepare(weights)
    x, topk_ids, topk_weights = _routed_inputs(m, 92)
    output = torch.zeros(m, _K, dtype=torch.bfloat16, device=device)
    bindings = ExitStack()
    binding = bindings.enter_context(
        make_tp_moe_fp4_binding(
            a=x,
            experts=prepared,
            topk_weights=topk_weights.contiguous(),
            topk_ids=topk_ids.contiguous(),
            output=output,
            input_scales_static=True,
            quant_mode="w4a8_mx",
        )
    )

    def _launch() -> None:
        b12x_moe_fp4(binding=binding)

    _launch()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        _launch()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    # Queue enough independent GEMM work to keep the auxiliary stream active
    # while the graph reaches the resident-grid kernel.
    aux_stream = torch.cuda.Stream()
    aux_a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    aux_b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    aux_out = torch.empty_like(aux_a)
    output.zero_()
    with torch.cuda.stream(aux_stream):
        for _ in range(16):
            torch.mm(aux_a, aux_b, out=aux_out)
    graph.replay()
    torch.cuda.current_stream().wait_stream(aux_stream)
    torch.cuda.synchronize()

    assert output.isfinite().all()
    assert output.abs().sum().item() > 0
    del graph
    bindings.close()


def test_w4a8_mx_dynamic_glm_shard_geometry() -> None:
    """GLM per-rank shard geometry: E=16, K=4096, n=256 (FC1 N=512, FC2
    N=4096 with K=256) through unified dynamic, gated vs the oracle."""
    _skip_if_unavailable()
    from b12x.moe.fused_moe._impl import clear_tp_moe_caches
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
    from b12x.testing.reference.helpers import run_tp_moe_fp4

    clear_tp_moe_caches()
    n = 256
    weights = _weights(n=n, seed=77)
    m = 256
    x, topk_ids, topk_weights = _routed_inputs(m, 44)
    ref = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        n,
        activation="silu",
    )
    prepared = _prepare(weights, n=n)
    out = run_tp_moe_fp4(
        a=x,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )
    torch.cuda.synchronize()
    n_out = out.float().norm().item()
    assert n_out > 0.01, f"dynamic output near-zero (norm={n_out})"
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cos > 0.998, cos
    n_ref = ref.float().norm().item()
    assert 0.8 < n_out / n_ref < 1.25, (n_out, n_ref)


@pytest.mark.parametrize("tile_m", (16, 32))
@pytest.mark.parametrize("max_active_clusters", (None, 1, 64, 128))
@pytest.mark.parametrize("capacity", (1, 8))
def test_repacked_decode_grid_reaches_launch_and_replays_without_allocation(
    monkeypatch,
    tile_m: int,
    max_active_clusters: int | None,
    capacity: int,
) -> None:
    _skip_if_unavailable()
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe import _impl
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    device = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if max_active_clusters is not None and max_active_clusters > 2 * sms:
        pytest.skip("Requested test grid exceeds this GPU's resident bound")
    for name in (
        "B12X_DYNAMIC_W4A8_DECODE_MAX_ACTIVE_CLUSTERS",
        "B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS",
        "B12X_LEVEL10_MAX_ACTIVE_CLUSTERS",
    ):
        monkeypatch.delenv(name, raising=False)
    _impl.clear_tp_moe_caches()
    counts = tuple(sorted({rows for rows in (1, 2, 6, capacity) if rows <= capacity}))
    weights = _weights(seed=281)
    x, ids, scales = _routed_inputs(capacity, 282)
    references = {
        rows: moe_reference_w4a8_mx(
            x[:rows].float(),
            weights["w13_fp4"],
            weights["w13_mx"],
            None,
            weights["alphas"],
            weights["w2_fp4"],
            weights["w2_mx"],
            None,
            weights["alphas"],
            ids[:rows],
            scales[:rows],
            _E,
            _K,
            _N,
            activation="silu",
        )
        for rows in counts
    }
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=_E, hidden_size=_K, intermediate_size=_N
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.PackedWeights(
            w13=weights["w13_fp4"],
            w2=weights["w2_fp4"],
            w13_block_scales=weights["w13_mx"],
            w2_block_scales=weights["w2_mx"],
            w13_global_scales=weights["alphas"],
            w2_global_scales=weights["alphas"],
            input_scale=weights["input_scale"],
            intermediate_scale=weights["input_scale"],
        ),
    )
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=_TOPK),
        invocation={"fast_math": False},
        override=MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=max_active_clusters,
            dynamic_tile_m=tile_m,
            dynamic_route_mode="grouped",
        ),
    )
    calls = []
    original = _impl._get_dynamic_kernel

    def record_grid(*args, **kwargs):
        compiled, grid = original(*args, **kwargs)
        calls.append((compiled, grid))
        return compiled, grid

    monkeypatch.setattr(_impl, "_get_dynamic_kernel", record_grid)

    def prepare(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        output = torch.empty_like(x)
        binding = state.bind(
            scratch=scratch,
            a=x,
            topk_ids=ids,
            topk_weights=scales,
            output=output,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(scratch, binding)
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(
            (plan.request(name="w4a8-resident-grid", prepare_call=prepare),)
        )
        expected_grid = min(2 * sms, max_active_clusters or 2 * sms)
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        output = torch.empty_like(x)
        storage = (*scratch, output, x, ids, scales)
        pointers = tuple(t.data_ptr() for t in storage)
        # Compile planning may request the same callable with a placeholder
        # grid. Observe an actual launch after preparation, not those requests.
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            topk_ids=ids,
            topk_weights=scales,
            output=output,
            input_scales_static=True,
        )
        calls.clear()
        session.freeze()
        with session.capture():
            fused_moe.run(binding=binding)
        torch.cuda.synchronize()
        assert calls and {grid for _, grid in calls} == {expected_grid}
        prepared_callable = calls[-1][0]
        for rows in counts:
            binding = fused_moe.bind(
                plan,
                scratch=scratch,
                a=x[:rows],
                topk_ids=ids[:rows],
                topk_weights=scales[:rows],
                output=output[:rows],
                input_scales_static=True,
            )
            calls.clear()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            assert calls and all(
                compiled is prepared_callable and grid == expected_grid
                for compiled, grid in calls
            )
            for _ in range(3):
                output.fill_(float("nan"))
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                )
                assert tuple(t.data_ptr() for t in storage) == pointers
                actual = output[:rows]
                assert actual.isfinite().all() and actual.abs().sum() > 0
                cosine = torch.nn.functional.cosine_similarity(
                    actual.float().flatten(),
                    references[rows].float().flatten(),
                    dim=0,
                ).item()
                assert cosine > 0.998, (rows, expected_grid, cosine)
                assert torch.isnan(output[rows:]).all()
            graph.reset()


@pytest.mark.parametrize("max_active_clusters", (None, 1, 24, 48))
@pytest.mark.parametrize(
    ("max_tokens", "unseen_counts", "expected_implementation", "route_planner"),
    (
        pytest.param(8, (1, 2, 7), "micro", "internal", id="micro-capacity"),
        pytest.param(64, (1, 2, 8, 16), "dynamic", "internal", id="dynamic-capacity"),
        pytest.param(64, (1, 2, 8, 16), "dynamic", "triton", id="dynamic-triton"),
    ),
)
def test_compact_n64_capacity_plan_reuses_one_callable_for_live_counts(
    max_tokens: int,
    unseen_counts: tuple[int, ...],
    expected_implementation: str,
    max_active_clusters: int | None,
    route_planner: str,
) -> None:
    _skip_if_unavailable()
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.moe import fused_moe
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    device = torch.device("cuda", torch.cuda.current_device())
    n = 192
    replay_counts = (*unseen_counts, max_tokens)
    weights = _weights(n=n, seed=117 + max_tokens)
    x, topk_ids, topk_weights = _routed_inputs(max_tokens, 118 + max_tokens)
    references = {}
    for rows in replay_counts:
        references[rows] = moe_reference_w4a8_mx(
            x[:rows].float(),
            weights["w13_fp4"],
            weights["w13_mx"],
            None,
            weights["alphas"],
            weights["w2_fp4"],
            weights["w2_mx"],
            None,
            weights["alphas"],
            topk_ids[:rows],
            topk_weights[:rows],
            _E,
            _K,
            n,
            activation="silu",
        )

    declaration = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=_E, hidden_size=_K, intermediate_size=n
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=declaration,
        weights=fused_moe.PackedWeights(
            w13=weights["w13_fp4"],
            w2=weights["w2_fp4"],
            w13_block_scales=weights["w13_mx"],
            w2_block_scales=weights["w2_mx"],
            w13_global_scales=weights["alphas"],
            w2_global_scales=weights["alphas"],
            input_scale=weights["input_scale"],
            intermediate_scale=weights["input_scale"],
        ),
    )
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=max_tokens, top_k=_TOPK),
        invocation={"fast_math": False},
        override=MoeDecodeConfig(
            backend=expected_implementation,
            route_planner=route_planner,
            max_active_clusters=max_active_clusters,
            dynamic_tile_m=16 if expected_implementation == "dynamic" else None,
            dynamic_route_mode="grouped"
            if expected_implementation == "dynamic"
            else None,
        ),
    )
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations

    def prepare(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        output = torch.empty_like(x)
        binding = state.bind(
            scratch=scratch,
            a=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(scratch, binding)
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((plan.request(name="compact-n64", prepare_call=prepare),))
        assert (
            plan.prepared.state.scratch.launch_plan.implementation
            == expected_implementation
        )
        assert plan.prepared.state.scratch.launch_plan.execution.tile_m == 16
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        output = torch.empty_like(x)
        storage = (*scratch, output, x, topk_ids, topk_weights)
        addresses = tuple(t.data_ptr() for t in storage)
        session.freeze()
        for rows in replay_counts:
            binding = fused_moe.bind(
                plan,
                scratch=scratch,
                a=x[:rows],
                topk_ids=topk_ids[:rows],
                topk_weights=topk_weights[:rows],
                output=output[:rows],
                input_scales_static=True,
            )
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            fused_moe.run(binding=binding)
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            for _ in range(3):
                if route_planner == "triton":
                    for tensor in scratch:
                        tensor.fill_(0x5A)
                output.fill_(float("nan"))
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                )
                assert tuple(t.data_ptr() for t in storage) == addresses
                replayed = output[:rows]
                assert replayed.isfinite().all() and replayed.abs().sum() > 0
                cosine = torch.nn.functional.cosine_similarity(
                    replayed.float().flatten(),
                    references[rows].float().flatten(),
                    dim=0,
                ).item()
                assert cosine > 0.998, (rows, cosine)
                assert torch.isnan(output[rows:]).all()
            graph.reset()


def test_compact_n64_micro_zero_and_tiny_blocks_use_common_mxfp8_scales() -> None:
    _skip_if_unavailable()
    from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
    from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
    from b12x.moe._shared.kernels.w4a8_compact_micro import (
        _layout,
        launch_w4a8_compact_micro,
        micro_scratch_nbytes,
    )

    n = 192
    capacity = 2
    weights = _weights(n=n, seed=191)
    _, topk_ids, topk_weights = _routed_inputs(capacity, 192)
    x = torch.zeros(capacity, _K, dtype=torch.bfloat16, device="cuda")
    x[1] = (
        torch.linspace(-1.0, 1.0, _K, dtype=torch.float32, device="cuda") * (2.0**-8)
    ).to(torch.bfloat16)
    experts = _prepare(weights, n=n)
    runtime = experts.representation_for("w4a8_mx")
    scratch = torch.empty(
        micro_scratch_nbytes(capacity, _K, n, _TOPK),
        dtype=torch.uint8,
        device="cuda",
    )

    launch_w4a8_compact_micro(
        scratch=scratch,
        a=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13=runtime.w13_rp,
        w13_scales=runtime.w13_sfb,
        w2=runtime.w2_rp,
        w2_scales=runtime.w2_sfb,
        alpha1=weights["alphas"],
        alpha2=weights["alphas"],
        input_scale=weights["input_scale"],
        down_scale=weights["input_scale"],
        max_tokens=capacity,
        num_topk=_TOPK,
        swiglu_limit=10.0,
        fast_math=False,
    )
    torch.cuda.synchronize()

    layout = _layout(capacity, _TOPK, _K, n)

    def region(name: str) -> torch.Tensor:
        offset, size = layout[name]
        return scratch.narrow(0, offset, size)

    input_scales = region("a_scales").view(capacity, _K // 32)
    assert torch.count_nonzero(input_scales[0]).item() == 0
    assert torch.count_nonzero(input_scales[1]).item() > 0

    pairs = capacity * _TOPK
    n_tiles = (n + 127) // 128
    n_padded = n_tiles * 128
    projections = region("projections").view(torch.bfloat16).view(pairs, 2 * n)
    gate = projections[:, :n].float().clamp(max=10.0)
    up = projections[:, n:].float().clamp(min=-10.0, max=10.0)
    activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    activated_padded = torch.zeros(
        pairs,
        n_padded,
        dtype=torch.bfloat16,
        device="cuda",
    )
    activated_padded[:, :n] = activated
    expected = empty_mxfp8_rows_for_dense_gemm(
        pairs,
        n_padded,
        device="cuda",
    )
    quantize_mxfp8_rows_cute(
        activated_padded,
        expected.values,
        expected.scale_rows,
        expected.scale_mma,
        expected_m=pairs,
    )
    torch.cuda.synchronize()

    intermediate = region("intermediate")
    payload_bytes = pairs * n_padded
    activation_scales = (
        intermediate[payload_bytes:]
        .view(n_tiles, pairs, 4)
        .permute(1, 0, 2)
        .reshape(pairs, n_padded // 32)
    )
    torch.testing.assert_close(
        activation_scales,
        expected.scale_rows.view(torch.uint8).reshape(pairs, n_padded // 32),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(activation_scales[:_TOPK]).item() == 0
    assert torch.count_nonzero(activation_scales[_TOPK:]).item() > 0


def test_compact_n64_grouped_prefill_honors_swiglu_limit() -> None:
    _skip_if_unavailable()
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.moe import fused_moe
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx

    device = torch.device("cuda", torch.cuda.current_device())
    n = 192
    tokens = 16
    weights = _weights(n=n, seed=211)
    x, topk_ids, topk_weights = _routed_inputs(tokens, 212)
    x = (x.float() * 32.0).to(torch.bfloat16)
    clamped = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        n,
        activation="silu",
        swiglu_limit=10.0,
    )
    unclamped = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        n,
        activation="silu",
    )
    assert not torch.allclose(clamped, unclamped, rtol=0.05, atol=0.05)

    declaration = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=10.0
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=_E, hidden_size=_K, intermediate_size=n
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=declaration,
        weights=fused_moe.PackedWeights(
            w13=weights["w13_fp4"],
            w2=weights["w2_fp4"],
            w13_block_scales=weights["w13_mx"],
            w2_block_scales=weights["w2_mx"],
            w13_global_scales=weights["alphas"],
            w2_global_scales=weights["alphas"],
            input_scale=weights["input_scale"],
            intermediate_scale=weights["input_scale"],
        ),
    )
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=tokens, top_k=_TOPK),
        invocation={"fast_math": False},
    )
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations

    def prepare(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        output = torch.empty_like(x)
        binding = state.bind(
            scratch=scratch,
            a=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(scratch, binding)
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((plan.request(name="compact-n64-clamp", prepare_call=prepare),))
        assert plan.prepared.state.scratch.launch_plan.implementation == "dynamic"
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        output = torch.empty_like(x)
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            input_scales_static=True,
        )
        session.freeze()
        actual = fused_moe.run(binding=binding)
        torch.cuda.synchronize()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            clamped.float().flatten(),
            dim=0,
        ).item()
        assert cosine > 0.998, cosine
