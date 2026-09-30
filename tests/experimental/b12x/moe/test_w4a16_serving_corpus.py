from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

import pytest
import torch

from b12x._lib.intrinsics import swizzle_block_scale
from b12x.moe._shared.kernels.w4a16.host import (
    plan_w4a16_buffers,
    select_route_block_size_m,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _DEFAULT_MAX_SHARED_MEM,
    compile_w4a16_fused_moe,
    compile_w4a16_topk_sum,
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_w4a16_modelopt_nvfp4_weights,
)
from b12x.testing.reference.w4a16_reference import moe_reference_w4a16


_NANO35_EXPERTS = 128
_NANO35_HIDDEN_SIZE = 2688
_NANO35_INTERMEDIATE_SIZE = 1856
_NANO35_TOPK = 6
_NANO35_ACTIVATION = "relu2"


@dataclass(frozen=True)
class _Nano35Weights:
    source: tuple[torch.Tensor, ...]
    prepared: object


def _storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    """Count unique backing CUDA allocations rather than tensor views."""
    storages: dict[tuple[torch.device, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (tensor.device, int(storage.data_ptr()))
        storages[key] = int(storage.nbytes())
    return sum(storages.values())


def _prepared_tensors(prepared: object) -> tuple[torch.Tensor, ...]:
    return tuple(
        value for value in vars(prepared).values() if isinstance(value, torch.Tensor)
    )


def _make_identical_expert_weights() -> tuple[torch.Tensor, ...]:
    """Make production-sized experts with identical contents.

    Identical experts let the test spread routes over all 128 serving experts
    while deriving the expected repeated output row from one exact GPU oracle
    route. Existing W4A16 tests cover expert-specific addressing with distinct
    weights; this corpus test is responsible for the production shape and exact
    serving-M specializations.
    """
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260718)
    experts = _NANO35_EXPERTS
    hidden_size = _NANO35_HIDDEN_SIZE
    intermediate_size = _NANO35_INTERMEDIATE_SIZE

    base_w13 = torch.randint(
        0,
        256,
        (1, intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    base_w2 = torch.randint(
        0,
        256,
        (1, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w13 = base_w13.expand(experts, -1, -1).contiguous()
    w2 = base_w2.expand(experts, -1, -1).contiguous()

    base_w13_scale = (
        torch.rand(
            (1, intermediate_size, hidden_size // 16),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.0625
        + 0.03125
    ).to(torch.float8_e4m3fn)
    base_w2_scale = (
        torch.rand(
            (1, hidden_size, intermediate_size // 16),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.0625
        + 0.03125
    ).to(torch.float8_e4m3fn)
    w13_scale = swizzle_block_scale(base_w13_scale.expand(experts, -1, -1).contiguous())
    w2_scale = swizzle_block_scale(base_w2_scale.expand(experts, -1, -1).contiguous())
    w13_global_scale = torch.full(
        (experts,), 0.0625, dtype=torch.float32, device="cuda"
    )
    w2_global_scale = torch.full((experts,), 0.0625, dtype=torch.float32, device="cuda")
    return (
        w13,
        w13_scale,
        w13_global_scale,
        w2,
        w2_scale,
        w2_global_scale,
    )


@pytest.fixture(scope="module")
def nano35_weights() -> _Nano35Weights:
    source = _make_identical_expert_weights()
    prepared = prepare_w4a16_modelopt_nvfp4_weights(
        *source,
        activation=_NANO35_ACTIVATION,
        params_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    return _Nano35Weights(source=source, prepared=prepared)


def _make_repeated_inputs(
    m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260719)
    prototype = (
        torch.randn(
            (1, _NANO35_HIDDEN_SIZE),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.125
    ).to(torch.bfloat16)
    # ``expand(1, -1)`` is already contiguous, so ``.contiguous()`` may return
    # the original ``prototype`` allocation for the M=1 specialization.  The
    # live-input graph check mutates ``x`` in place and must not overwrite the
    # saved oracle input that is later restored.
    x = prototype.expand(m, -1).clone()
    assert x.untyped_storage().data_ptr() != prototype.untyped_storage().data_ptr()

    token = torch.arange(m, dtype=torch.int64, device="cuda")[:, None]
    route_offsets = torch.arange(
        0,
        _NANO35_TOPK * 23,
        23,
        dtype=torch.int64,
        device="cuda",
    )[None, :]
    topk_ids = ((token + route_offsets) % _NANO35_EXPERTS).to(torch.int32)
    # Binary-exact weights sum to one. Since all expert tensors are identical,
    # the expected six-route reduction equals one unweighted oracle route.
    prototype_topk_weights = torch.tensor(
        (0.25, 0.25, 0.125, 0.125, 0.125, 0.125),
        dtype=torch.float32,
        device="cuda",
    )[None, :]
    topk_weights = prototype_topk_weights.expand(m, -1).contiguous()
    return x, topk_ids, topk_weights, prototype


def _one_route_oracle(
    prototype: torch.Tensor,
    source: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    w13, w13_scale, w13_global, w2, w2_scale, w2_global = source
    return moe_reference_w4a16(
        prototype,
        w13[:1],
        w13_scale[:1],
        w13_global[:1],
        w2[:1],
        w2_scale[:1],
        w2_global[:1],
        torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        torch.ones((1, 1), dtype=torch.float32, device="cuda"),
        1,
        _NANO35_HIDDEN_SIZE,
        _NANO35_INTERMEDIATE_SIZE,
        activation=_NANO35_ACTIVATION,
    )


def _assert_every_row_matches(
    actual: torch.Tensor,
    expected_row: torch.Tensor,
) -> dict[str, float]:
    expected_f32 = expected_row.float()
    expected_norm = float(expected_f32.norm().item())
    assert expected_norm > 0.0

    min_cos = 1.0
    max_relative_l2 = 0.0
    max_abs = 0.0
    saw_nonzero = False
    for start in range(0, int(actual.shape[0]), 1024):
        chunk = actual[start : start + 1024].float()
        assert bool(torch.isfinite(chunk).all().item())
        saw_nonzero = saw_nonzero or bool((chunk != 0).any().item())
        expected = expected_f32.expand_as(chunk)
        diff = chunk - expected
        row_norms = chunk.norm(dim=1)
        row_cos = (chunk * expected).sum(dim=1) / (row_norms * expected_norm).clamp_min(
            1e-24
        )
        row_relative_l2 = diff.norm(dim=1) / expected_norm
        min_cos = min(min_cos, float(row_cos.min().item()))
        max_relative_l2 = max(max_relative_l2, float(row_relative_l2.max().item()))
        max_abs = max(max_abs, float(diff.abs().max().item()))

    assert saw_nonzero
    assert min_cos >= 0.99, {
        "min_cos": min_cos,
        "max_relative_l2": max_relative_l2,
        "max_abs": max_abs,
    }
    # Cosine alone cannot catch a uniform output-scale error. The current
    # weight-only oracle and kernel are substantially tighter than this bound,
    # while 15% leaves room for their documented BF16 accumulation differences.
    assert max_relative_l2 <= 0.15, {
        "min_cos": min_cos,
        "max_relative_l2": max_relative_l2,
        "max_abs": max_abs,
    }
    return {
        "min_row_cos": min_cos,
        "max_row_relative_l2": max_relative_l2,
        "max_abs": max_abs,
    }


def _assert_bit_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    for start in range(0, int(actual.shape[0]), 4096):
        assert torch.equal(actual[start : start + 4096], expected[start : start + 4096])


def _run_nano35_serving_case(m: int, weights: _Nano35Weights) -> None:
    prepared = weights.prepared
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )
    block_size = select_route_block_size_m(m, _NANO35_TOPK, _NANO35_EXPERTS)
    direct_topk_routes = m <= 6
    plan = plan_w4a16_buffers(
        prepared,
        m=m,
        topk=_NANO35_TOPK,
        route_num_experts=_NANO35_EXPERTS,
        sms=sms,
    )
    max_m_blocks = m * _NANO35_TOPK if direct_topk_routes else plan.route_blocks

    free_before_buffers, device_total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    x, topk_ids, topk_weights, prototype = _make_repeated_inputs(m)
    live_prototype = (
        prototype.float() * -0.75 + 0.015625
    ).to(torch.bfloat16)
    live_topk_ids = (topk_ids + 1) % _NANO35_EXPERTS
    assert not torch.equal(live_prototype, prototype)
    assert not torch.equal(live_topk_ids, topk_ids)
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=_NANO35_TOPK,
        dtype=torch.bfloat16,
        device=device,
    )
    buffer_tensors = tuple(
        value for value in vars(buffers).values() if isinstance(value, torch.Tensor)
    )

    fused = compile_w4a16_fused_moe(
        size_m=m,
        hidden_size=_NANO35_HIDDEN_SIZE,
        intermediate_size=_NANO35_INTERMEDIATE_SIZE,
        num_experts=_NANO35_EXPERTS,
        top_k=_NANO35_TOPK,
        activation=_NANO35_ACTIVATION,
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block_size,
        max_m_blocks=max_m_blocks,
        element_dtype="bf16",
        fast_math=True,
        sms=sms,
        max_shared_mem=max_shared_mem,
        weight_layout="packed",
        scale_format="e4m3_k16",
        w13_layout="packed",
        direct_topk_routes=direct_topk_routes,
        tc_decode_fused_sum=False,
    )
    topk_sum = compile_w4a16_topk_sum(
        m=m,
        topk=_NANO35_TOPK,
        hidden_size=_NANO35_HIDDEN_SIZE,
        element_dtype="bf16",
    )
    assert (
        fused.size_m,
        fused.hidden_size,
        fused.intermediate_size,
        fused.num_experts,
        fused.top_k,
        fused.activation,
        fused.moe_block_size,
        fused.max_m_blocks,
        fused.weight_layout,
        fused.scale_format,
        fused.direct_topk_routes,
        fused.tc_decode_fused_sum,
    ) == (
        m,
        _NANO35_HIDDEN_SIZE,
        _NANO35_INTERMEDIATE_SIZE,
        _NANO35_EXPERTS,
        _NANO35_TOPK,
        _NANO35_ACTIVATION,
        block_size,
        max_m_blocks,
        "packed",
        "e4m3_k16",
        direct_topk_routes,
        False,
    )
    assert (topk_sum.topk, topk_sum.hidden_size) == (
        _NANO35_TOPK,
        _NANO35_HIDDEN_SIZE,
    )

    def launch() -> torch.Tensor:
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation=_NANO35_ACTIVATION,
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
            fused_launch=fused,
            topk_sum_launch=topk_sum,
        )

    eager = launch()
    expected_row = _one_route_oracle(prototype, weights.source)
    expected_live_row = _one_route_oracle(live_prototype, weights.source)
    assert not torch.allclose(expected_live_row, expected_row)
    torch.cuda.synchronize()
    eager_metrics = _assert_every_row_matches(eager, expected_row)
    eager_copy = eager.clone()

    if direct_topk_routes:
        actual_padded_route_slots = m * _NANO35_TOPK
        actual_route_blocks = m * _NANO35_TOPK
    else:
        actual_padded_route_slots = int(buffers.packed_route_count.item())
        assert actual_padded_route_slots % block_size == 0
        actual_route_blocks = actual_padded_route_slots // block_size
        assert m * _NANO35_TOPK <= actual_padded_route_slots <= plan.route_slots

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        launch()

    # Prove that capture retained both activation and routing inputs rather than
    # replaying values materialized during warmup. All experts intentionally
    # share weights, so changing route ids is numerically neutral while still
    # forcing the production routing path to consume new metadata; changing X
    # gives the replay an independently checkable output.
    x.copy_(live_prototype.expand_as(x))
    topk_ids.copy_(live_topk_ids)
    buffers.output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    live_graph_metrics = _assert_every_row_matches(
        buffers.output,
        expected_live_row,
    )

    # Restore the original live tensors and require a second replay to be
    # bit-exact with the eager production path.
    x.copy_(prototype.expand_as(x))
    topk_ids.copy_((live_topk_ids - 1) % _NANO35_EXPERTS)
    buffers.output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    graph_metrics = _assert_every_row_matches(buffers.output, expected_row)
    _assert_bit_exact(buffers.output, eager_copy)

    all_weight_tensors = weights.source + _prepared_tensors(prepared)
    record = {
        "activation": _NANO35_ACTIVATION,
        "actual_padded_route_slots": actual_padded_route_slots,
        "actual_route_blocks": actual_route_blocks,
        "buffer_plan": {
            "fc1_c_tmp_elements": plan.fc1_c_tmp_elements,
            "fc2_c_tmp_elements": plan.fc2_c_tmp_elements,
            "intermediate_cache13_elements": plan.intermediate_cache13_elements,
            "intermediate_cache2_elements": plan.intermediate_cache2_elements,
            "route_blocks_capacity": plan.route_blocks,
            "route_slots_capacity": plan.route_slots,
            "storage_bytes": _storage_bytes(buffer_tensors),
        },
        "compile_fake_m_extent": 1 if m == 1 else 2,
        "compile_kernels": [
            "moe.w4a16.fused_moe",
            "moe.w4a16.topk_sum",
        ],
        "cuda_graph_replay": True,
        "cuda_graph_live_input_replay": True,
        "device": props.name,
        "device_free_before_buffers": int(free_before_buffers),
        "device_total_bytes": int(device_total),
        "direct_topk_routes": direct_topk_routes,
        "eager_metrics": eager_metrics,
        "fused_compile": {
            "blocks_per_sm": fused.blocks_per_sm,
            "fc1_tile_k": fused.fc1_tile_k,
            "fc1_tile_n": fused.fc1_tile_n,
            "fc2_tile_k": fused.fc2_tile_k,
            "fc2_tile_n": fused.fc2_tile_n,
            "max_m_blocks": fused.max_m_blocks,
            "moe_block_size": fused.moe_block_size,
        },
        "graph_eager_bit_exact": True,
        "graph_metrics": graph_metrics,
        "live_graph_metrics": live_graph_metrics,
        "logical_m": m,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "num_experts": _NANO35_EXPERTS,
        "routed_rows": m * _NANO35_TOPK,
        "scale_format": "e4m3_k16",
        "shape": {
            "hidden_size": _NANO35_HIDDEN_SIZE,
            "intermediate_size": _NANO35_INTERMEDIATE_SIZE,
            "topk": _NANO35_TOPK,
        },
        "weight_layout": "packed",
        "weight_storage_bytes": _storage_bytes(all_weight_tensors),
    }
    print("W4A16_SERVING_CORPUS=" + json.dumps(record, sort_keys=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", (1, 2, 4, 8, 23, 33, 80), ids=lambda m: f"m{m}")
def test_w4a16_nano35_observed_serving_corpus(
    m: int,
    nano35_weights: _Nano35Weights,
) -> None:
    _run_nano35_serving_case(m, nano35_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "m",
    (8192, 16384, 24576, 32768),
    ids=lambda m: f"m{m}",
)
def test_w4a16_nano35_chunked_prefill_corpus(
    m: int,
    nano35_weights: _Nano35Weights,
) -> None:
    _run_nano35_serving_case(m, nano35_weights)
