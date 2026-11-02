from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from b12x.attention import sparse_mla
from b12x.attention._shared.mla import api as mla_api
from b12x.attention.sparse_mla import api as sparse_mla_api
from b12x.attention._shared.mla.reference import (
    _sparse_attention_reference,
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
    unpack_mla_kv_cache_reference,
)
from b12x.attention._shared.mla.kv_cache import (
    _glm_next_cache_byte_offset,
    _glm_next_cache_record_address,
    clear_glm_next_mla_kv_cache_kernel_cache,
    concat_and_cache_glm_next_mla,
)
from b12x.attention._shared.mla.traits import (
    ComputeMode,
    ModelType,
    ScaleFormat,
    infer_model_type,
    make_unified_traits,
    resolve_unplanned_traits,
)
from b12x.testing.reference.helpers import dequantize_nvfp4_mla_nope

from ..conftest import require_b12x as require_sm120


_GLM_NEXT_RECORD_BYTES = 528
_GLM_NEXT_NVFP4_RECORD_BYTES = 304
_GLM_NEXT_PAGE_SIZE = 64
_GLM_NEXT_HEAD_DIM = 512
_GLM_NEXT_SM_SCALE = 256**-0.5
_ALLOCATOR_COUNTERS = (
    "allocation.all.allocated",
    "allocation.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
    "num_alloc_retries",
    "num_ooms",
)


def _allocator_counters(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {name: int(stats.get(name, 0)) for name in _ALLOCATOR_COUNTERS}


def _glm_next_plan_and_binding(
    *,
    device: torch.device,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    selected: torch.Tensor,
    active: torch.Tensor,
    cache_seqlens: torch.Tensor,
    mode: str = "decode",
    page_size: int = _GLM_NEXT_PAGE_SIZE,
    return_lse: bool = False,
    lse_scale: str = "base2",
):
    is_nvfp4 = int(kv_cache.shape[-1]) == _GLM_NEXT_NVFP4_RECORD_BYTES
    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device=device,
            num_q_heads=int(q.shape[1]),
            max_q_rows=int(q.shape[0]),
            max_batch=int(cache_seqlens.shape[0]),
            max_width=int(selected.shape[1]),
            softmax_scale=_GLM_NEXT_SM_SCALE,
            max_kv_rows=max(int(active.max().item()), 1),
            kv_dtype=torch.uint8,
            head_dim=_GLM_NEXT_HEAD_DIM,
            v_head_dim=_GLM_NEXT_HEAD_DIM,
            page_size=page_size,
            model_type=ModelType.GLM_NEXT,
            scale_format=(
                ScaleFormat.NVFP4_E4M3 if is_nvfp4 else ScaleFormat.ARBITRARY_FP32
            ),
            cache_record_bytes=int(kv_cache.shape[-1]),
            fp8_rope=False,
            latent_scale_per_token=is_nvfp4,
            mode=mode,
            return_lse=return_lse,
            lse_scale=lse_scale,
        )
    )
    spec = plan.scratch_specs()[0]
    binding = sparse_mla.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device),
        q=q,
        selected_indices=selected,
        kv_cache=kv_cache,
        cache_lengths=cache_seqlens,
        selected_lengths=active,
    )
    return plan, binding


def _assert_glm_next_attention_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.reshape(-1), expected_f.reshape(-1), dim=0
    ).item()
    max_abs = (actual_f - expected_f).abs().max().item()
    assert cosine > 0.995, f"cosine similarity {cosine:.8f} did not exceed 0.995"
    assert max_abs < 0.03, f"max absolute error {max_abs:.8f} exceeded 0.03"


def _glm_next_dense_physical_reference(
    q: torch.Tensor,
    latent: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
) -> torch.Tensor:
    keys = torch.einsum("rl,hpl->rhp", latent.float(), w_uk_t.float()).to(
        torch.bfloat16
    )
    values = torch.einsum("rl,hlv->rhv", latent.float(), w_uv.float()).to(
        torch.bfloat16
    )
    logits = torch.einsum("rhp,shp->hrs", q.float(), keys.float())
    logits.mul_(_GLM_NEXT_SM_SCALE)
    causal_mask = torch.ones(
        (q.shape[0], q.shape[0]), dtype=torch.bool, device=q.device
    ).triu_(1)
    logits.masked_fill_(causal_mask.unsqueeze(0), -torch.inf)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.einsum("hrs,shv->rhv", probabilities, values.float()).to(
        torch.bfloat16
    )


def _pooled_selection_reference(
    pool_indices: torch.Tensor,
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    *,
    pool_size: int,
    block_size: int,
    block_stride_rows: int,
    num_cache_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, pool_topk = pool_indices.shape
    output_width = pool_topk * pool_size + pool_size - 1
    logical = torch.full(
        (rows, output_width), -1, dtype=torch.int64, device=pool_indices.device
    )
    sequence_lengths = positions + 1
    complete_pools = torch.div(sequence_lengths, pool_size, rounding_mode="floor")
    selected_pools = torch.minimum(
        complete_pools, torch.full_like(complete_pools, pool_topk)
    )
    selected_history_tokens = selected_pools * pool_size
    columns = torch.arange(output_width, dtype=torch.int64, device=pool_indices.device)
    pool_columns = torch.div(columns, pool_size, rounding_mode="floor").clamp_max(
        pool_topk - 1
    )
    selected_pool_ids = pool_indices[:, pool_columns].to(torch.int64)
    history = columns.unsqueeze(0) < selected_history_tokens.unsqueeze(1)
    history_values = selected_pool_ids * pool_size + columns.remainder(pool_size)
    logical.copy_(torch.where(history & (selected_pool_ids >= 0), history_values, -1))

    tail_start = complete_pools * pool_size
    tail_counts = sequence_lengths - tail_start
    tail_offsets = columns.unsqueeze(0) - selected_history_tokens.unsqueeze(1)
    in_tail = (tail_offsets >= 0) & (tail_offsets < tail_counts.unsqueeze(1))
    logical.copy_(torch.where(in_tail, tail_start.unsqueeze(1) + tail_offsets, logical))

    safe_logical = logical.clamp_min(0)
    block_ids = torch.div(safe_logical, block_size, rounding_mode="floor")
    valid = (logical >= 0) & (block_ids < block_table.shape[1])
    safe_blocks = block_ids.clamp_max(block_table.shape[1] - 1)
    pages = block_table[request_ids.to(torch.int64).unsqueeze(1), safe_blocks]
    valid &= (pages >= 0) & (pages < num_cache_blocks)
    physical = pages.to(torch.int64) * block_stride_rows + safe_logical % block_size
    output = torch.where(valid, physical, -1).to(torch.int32)
    active_counts = (selected_pools * pool_size + tail_counts).to(torch.int32)
    return output, active_counts


@torch.inference_mode()
@pytest.mark.parametrize("rows", [1, 7, 32])
@pytest.mark.parametrize("planned", [False, True])
def test_pooled_topk_expands_to_physical_slots(rows: int, planned: bool) -> None:
    device = require_sm120()
    pool_size = 4
    block_size = 256
    block_stride_rows = 256
    num_cache_blocks = 8_000_000
    requests = min(rows, 8)
    positions_data = [0, 1, 3, 4, 255, 256, 2047, 2048, 4095, 16383]
    positions = torch.tensor(
        [positions_data[row % len(positions_data)] for row in range(rows)],
        dtype=torch.int64,
        device=device,
    )
    request_ids = torch.arange(rows, dtype=torch.int32, device=device) % requests
    block_table = torch.arange(requests * 80, dtype=torch.int32, device=device).reshape(
        requests, 80
    )
    block_table.mul_(101).add_(7_000_000)
    block_table[:, -1] = -1
    pool_indices = torch.full((rows, 512), -1, dtype=torch.int32, device=device)
    for row, position in enumerate(positions.cpu().tolist()):
        selected = min((position + 1) // pool_size, 512)
        if selected:
            pool_indices[row, :selected] = torch.arange(
                selected - 1, -1, -1, dtype=torch.int32, device=device
            )
    output = torch.empty((rows, 2051), dtype=torch.int32, device=device)
    active_counts = torch.empty(rows, dtype=torch.int32, device=device)

    plan = None
    if planned:
        from b12x.preparation import PreparationSession, PreparedCall

        plan = sparse_mla.plan_pooled_selection(
            device=device,
            max_rows=32,
            page_size=block_size,
            max_page_table_width=80,
            num_cache_blocks=num_cache_blocks,
        )
        request = plan.request(
            name="pooled selection",
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(
                    pool_indices,
                    positions,
                    request_ids,
                    block_table,
                    output,
                    active_counts,
                    pool_size=pool_size,
                    block_size=block_size,
                    block_stride_rows=block_stride_rows,
                    num_cache_blocks=num_cache_blocks,
                )
            ),
        )
        PreparationSession(device=device, compile_workers=0).prepare((request,))

    sparse_mla.expand_pooled_topk_to_physical_slots(
        pool_indices,
        positions,
        request_ids,
        block_table,
        output,
        active_counts,
        pool_size=pool_size,
        block_size=block_size,
        block_stride_rows=block_stride_rows,
        num_cache_blocks=num_cache_blocks,
        plan=plan,
    )
    expected, expected_counts = _pooled_selection_reference(
        pool_indices,
        positions,
        request_ids,
        block_table,
        pool_size=pool_size,
        block_size=block_size,
        block_stride_rows=block_stride_rows,
        num_cache_blocks=num_cache_blocks,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(active_counts, expected_counts, rtol=0, atol=0)


@torch.inference_mode()
def test_pooled_topk_physical_expansion_replays_live_inputs() -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.preparation import PreparedCall, PreparationSession

    device = require_sm120()
    rows, requests = 7, 4
    pool_indices = torch.arange(512, dtype=torch.int32, device=device).repeat(rows, 1)
    positions = torch.full((rows,), 2047, dtype=torch.int64, device=device)
    request_ids = torch.arange(rows, dtype=torch.int32, device=device) % requests
    block_table = torch.arange(requests * 16, dtype=torch.int32, device=device).reshape(
        requests, 16
    )
    output = torch.empty((rows, 2051), dtype=torch.int32, device=device)
    active_counts = torch.empty(rows, dtype=torch.int32, device=device)

    def expand() -> None:
        sparse_mla.expand_pooled_topk_to_physical_slots(
            pool_indices,
            positions,
            request_ids,
            block_table,
            output,
            active_counts,
            pool_size=4,
            block_size=256,
            block_stride_rows=256,
            num_cache_blocks=8_000_000,
            plan=plan,
        )

    plan = sparse_mla.plan_pooled_selection(
        device=device,
        max_rows=rows,
        page_size=256,
        max_page_table_width=16,
        num_cache_blocks=8_000_000,
    )
    request = plan.request(
        name="glm-c4-selection",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(
                pool_indices,
                positions,
                request_ids,
                block_table,
                output,
                active_counts,
                pool_size=4,
                block_size=256,
                block_stride_rows=256,
                num_cache_blocks=8_000_000,
            ),
            owners=(
                pool_indices,
                positions,
                request_ids,
                block_table,
                output,
                active_counts,
            ),
        ),
    )
    session = PreparationSession(device=device, autotune=False, compile_workers=0)
    session.prepare((request,))
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("prepared C4 selection"), torch.cuda.graph(graph):
        expand()

    pool_indices.copy_(pool_indices.flip(dims=(1,)))
    positions.add_(1)
    request_ids.copy_(
        torch.tensor([3, 1, 2, 0, 3, 2, 1], dtype=torch.int32, device=device)
    )
    block_table.add_(7_000_000)
    output.fill_(37)
    active_counts.fill_(37)
    graph.replay()
    torch.cuda.synchronize(device)

    expected, expected_counts = _pooled_selection_reference(
        pool_indices,
        positions,
        request_ids,
        block_table,
        pool_size=4,
        block_size=256,
        block_stride_rows=256,
        num_cache_blocks=8_000_000,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(active_counts, expected_counts, rtol=0, atol=0)

    allocator_before = _allocator_counters(device)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize(device)
    assert _allocator_counters(device) == allocator_before


def test_pooled_topk_rejects_physical_slots_above_int32() -> None:
    pool_size = 4
    block_size = 256
    num_cache_blocks = torch.iinfo(torch.int32).max // block_size + 2
    with pytest.raises(
        ValueError, match="physical cache slots exceed the int32 index range"
    ):
        sparse_mla.expand_pooled_topk_to_physical_slots(
            torch.zeros((1, 1), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int64),
            torch.zeros((1,), dtype=torch.int32),
            torch.full((1, 1), num_cache_blocks - 1, dtype=torch.int32),
            torch.empty((1, 7), dtype=torch.int32),
            torch.empty((1,), dtype=torch.int32),
            pool_size=pool_size,
            block_size=block_size,
            block_stride_rows=block_size,
            num_cache_blocks=num_cache_blocks,
        )


def test_glm_next_traits_define_no_rope_record() -> None:
    traits = make_unified_traits(
        ModelType.GLM_NEXT,
        ComputeMode.FP8,
        ScaleFormat.ARBITRARY_FP32,
    )

    assert (traits.d_nope, traits.d_rope, traits.d_v) == (512, 0, 512)
    assert traits.kv_gmem_stride == 528
    assert traits.kv_smem_stride == 528
    assert traits.bulk_tx_bytes == 64 * 528
    assert traits.rope_payload_bytes == 0
    assert not traits.v_has_rope
    assert not traits.fp8_rope


def test_glm_next_requires_explicit_identity_for_ambiguous_head_width() -> None:
    assert infer_model_type(512, torch.uint8) == (
        ModelType.DSV4,
        ComputeMode.FP8,
        ScaleFormat.UE8M0_BYTE,
    )
    assert infer_model_type(
        512,
        torch.uint8,
        model_type=ModelType.GLM_NEXT,
    ) == (
        ModelType.GLM_NEXT,
        ComputeMode.FP8,
        ScaleFormat.ARBITRARY_FP32,
    )

    with pytest.raises(ValueError, match="requires q_head_dim=512"):
        infer_model_type(576, torch.uint8, model_type=ModelType.GLM_NEXT)


def test_glm_next_accepts_inline_scale_nvfp4_without_rope() -> None:
    nvfp4_traits = make_unified_traits(
        ModelType.GLM_NEXT,
        ComputeMode.BF16,
        ScaleFormat.NVFP4_E4M3,
        fp8_rope=False,
        latent_scale_per_token=True,
    )

    assert nvfp4_traits.kv_gmem_stride == _GLM_NEXT_NVFP4_RECORD_BYTES
    assert nvfp4_traits.kv_smem_stride == 288
    assert nvfp4_traits.d_rope == 0
    assert nvfp4_traits.compute_mode == ComputeMode.BF16

    compatibility_traits = resolve_unplanned_traits(
        512,
        torch.uint8,
        _GLM_NEXT_NVFP4_RECORD_BYTES,
        model_type=ModelType.GLM_NEXT,
        scale_format=ScaleFormat.NVFP4_E4M3,
    )
    assert compatibility_traits.latent_scale_per_token

    with pytest.raises(ValueError, match="inline per-token latent scale"):
        make_unified_traits(
            ModelType.GLM_NEXT,
            ComputeMode.BF16,
            ScaleFormat.NVFP4_E4M3,
        )
    with pytest.raises(ValueError, match="no RoPE cache payload"):
        make_unified_traits(
            ModelType.GLM_NEXT,
            ComputeMode.BF16,
            ScaleFormat.NVFP4_E4M3,
            fp8_rope=True,
            latent_scale_per_token=True,
        )
    with pytest.raises(ValueError, match="ComputeMode.FP8"):
        make_unified_traits(
            ModelType.GLM_NEXT,
            ComputeMode.BF16,
            ScaleFormat.ARBITRARY_FP32,
        )


def test_glm_next_nvfp4_unplanned_traits_enable_inline_latent_scale() -> None:
    traits = resolve_unplanned_traits(
        512,
        torch.uint8,
        _GLM_NEXT_NVFP4_RECORD_BYTES,
        model_type=ModelType.GLM_NEXT,
        scale_format=ScaleFormat.NVFP4_E4M3,
        fp8_rope=False,
    )

    assert traits.model_type == ModelType.GLM_NEXT
    assert traits.scale_format == ScaleFormat.NVFP4_E4M3
    assert traits.latent_scale_per_token is True
    assert traits.kv_gmem_stride == _GLM_NEXT_NVFP4_RECORD_BYTES


def test_traits_less_nvfp4_route_rejects_non_sm120_backend() -> None:
    q = torch.zeros((1, 1, _GLM_NEXT_HEAD_DIM), dtype=torch.bfloat16)
    cache = torch.zeros((1, 1, _GLM_NEXT_NVFP4_RECORD_BYTES), dtype=torch.uint8)
    selected = torch.zeros((1, 1), dtype=torch.int32)
    lengths = torch.ones((1,), dtype=torch.int32)
    workspace = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        page_size=1,
        cache_traits=None,
    )

    with pytest.raises(
        NotImplementedError, match="NVFP4 sparse MLA requires the active SM120"
    ):
        mla_api._run_sparse_mla(
            q_all=q,
            kv_cache=cache,
            selected_indices=selected,
            cache_seqlens_int32=lengths,
            active_token_counts=lengths,
            workspace=workspace,
            sm_scale=_GLM_NEXT_SM_SCALE,
            v_head_dim=_GLM_NEXT_HEAD_DIM,
            scale_format=ScaleFormat.NVFP4_E4M3,
            model_type=ModelType.GLM_NEXT,
        )


def test_glm_next_nvfp4_cache_abi_is_fixed_by_plan_and_binding() -> None:
    caps = sparse_mla.Caps(
        device="cpu",
        num_q_heads=8,
        max_q_rows=1,
        max_width=1,
        softmax_scale=_GLM_NEXT_SM_SCALE,
        kv_dtype=torch.uint8,
        head_dim=512,
        v_head_dim=512,
        page_size=1,
        model_type=ModelType.GLM_NEXT,
        scale_format=ScaleFormat.NVFP4_E4M3,
        cache_record_bytes=_GLM_NEXT_NVFP4_RECORD_BYTES,
        fp8_rope=False,
        latent_scale_per_token=True,
    )
    assert caps.scale_format == ScaleFormat.NVFP4_E4M3
    assert caps.cache_record_bytes == _GLM_NEXT_NVFP4_RECORD_BYTES
    assert caps.fp8_rope is False
    assert caps.latent_scale_per_token is True

    plan = sparse_mla.plan(caps)
    spec = plan.scratch_specs()[0]
    bind_kwargs = dict(
        scratch=torch.empty(spec.shape, dtype=spec.dtype),
        q=torch.empty((1, 8, 512), dtype=torch.bfloat16),
        selected_indices=torch.zeros((1, 1), dtype=torch.int32),
        cache_lengths=torch.ones((1,), dtype=torch.int32),
        selected_lengths=torch.ones((1,), dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="does not match the sparse MLA plan"):
        sparse_mla.bind(
            plan,
            kv_cache=torch.empty((1, 1, 528), dtype=torch.uint8),
            **bind_kwargs,
        )
    with pytest.raises(ValueError, match="page size does not match"):
        sparse_mla.bind(
            plan,
            kv_cache=torch.empty(
                (2, 2, _GLM_NEXT_NVFP4_RECORD_BYTES + 1), dtype=torch.uint8
            )[:, :, :_GLM_NEXT_NVFP4_RECORD_BYTES],
            **bind_kwargs,
        )
    binding = sparse_mla.bind(
        plan,
        kv_cache=torch.empty((1, 1, _GLM_NEXT_NVFP4_RECORD_BYTES), dtype=torch.uint8),
        **bind_kwargs,
    )
    assert binding.kv_cache is not None
    assert binding.runtime.scratch.cache_traits == caps.cache_traits


def test_binding_owned_cache_omits_runtime_recipe_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device="cpu",
            num_q_heads=1,
            max_q_rows=1,
            max_width=1,
            softmax_scale=_GLM_NEXT_SM_SCALE,
            kv_dtype=torch.uint8,
            head_dim=_GLM_NEXT_HEAD_DIM,
            v_head_dim=_GLM_NEXT_HEAD_DIM,
            page_size=1,
            model_type=ModelType.GLM_NEXT,
            scale_format=ScaleFormat.NVFP4_E4M3,
            latent_scale_per_token=True,
        )
    )
    spec = plan.scratch_specs()[0]
    cache = torch.empty((1, 1, _GLM_NEXT_NVFP4_RECORD_BYTES), dtype=torch.uint8)
    binding = sparse_mla.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype),
        q=torch.empty((1, 1, _GLM_NEXT_HEAD_DIM), dtype=torch.bfloat16),
        selected_indices=torch.zeros((1, 1), dtype=torch.int32),
        cache_lengths=torch.ones((1,), dtype=torch.int32),
        selected_lengths=torch.ones((1,), dtype=torch.int32),
        kv_cache=cache,
    )
    calls: dict[str, object] = {}

    def fake_run_decode(**kwargs: object) -> torch.Tensor:
        calls.update(kwargs)
        return torch.empty((1, 1, _GLM_NEXT_HEAD_DIM), dtype=torch.bfloat16)

    monkeypatch.setattr(sparse_mla_api, "_run_decode", fake_run_decode)

    sparse_mla.run(binding)

    assert binding.runtime.kv_cache is cache
    assert binding.runtime.cache_traits == plan.caps.cache_traits
    assert calls["binding"] is binding.runtime
    assert "kv_cache" not in calls
    assert "model_type" not in calls
    assert "scale_format" not in calls
    assert "fp8_rope" not in calls
    assert "latent_scale_per_token" not in calls


def test_glm_next_pooled_selection_maps_compact_physical_slots_and_replays() -> None:
    device = require_sm120()
    pool_ids = torch.full((2, 512), -1, dtype=torch.int32, device=device)
    pool_ids[0, :2] = torch.tensor([1, 0], dtype=torch.int32, device=device)
    pool_ids[1, 0] = 0
    positions = torch.tensor([9, 4], dtype=torch.int64, device=device)
    request_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    block_table = torch.tensor(
        [[3, 1, -1], [5, 2, -1]], dtype=torch.int32, device=device
    )
    output = torch.empty((2, 2051), dtype=torch.int32, device=device)
    active = torch.empty((2,), dtype=torch.int32, device=device)

    kwargs = dict(
        pool_size=4,
        block_size=8,
        block_stride_rows=16,
        num_cache_blocks=8,
    )
    sparse_mla.expand_pooled_topk_to_physical_slots(
        pool_ids,
        positions,
        request_ids,
        block_table,
        output,
        active,
        **kwargs,
    )
    torch.cuda.synchronize(device)

    assert active.cpu().tolist() == [10, 5]
    assert output[0, :10].cpu().tolist() == [52, 53, 54, 55, 48, 49, 50, 51, 16, 17]
    assert output[1, :5].cpu().tolist() == [80, 81, 82, 83, 84]
    assert bool(torch.all(output[:, 10:] == -1).item())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla.expand_pooled_topk_to_physical_slots(
            pool_ids,
            positions,
            request_ids,
            block_table,
            output,
            active,
            **kwargs,
        )
    allocator_counters = _allocator_counters(device)
    output.fill_(7)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize(device)
    assert _allocator_counters(device) == allocator_counters
    assert active.cpu().tolist() == [10, 5]
    assert output[0, :10].cpu().tolist() == [52, 53, 54, 55, 48, 49, 50, 51, 16, 17]
    assert output[1, :5].cpu().tolist() == [80, 81, 82, 83, 84]
    assert bool(torch.all(output[0, 10:] == -1).item())
    assert bool(torch.all(output[1, 5:] == -1).item())


def test_glm_next_reference_record_is_528_bytes() -> None:
    torch.manual_seed(20260826)
    latent = torch.randn((7, 512), dtype=torch.bfloat16) / 4

    packed = pack_mla_kv_cache_reference(latent)
    unpacked = unpack_mla_kv_cache_reference(packed)

    assert tuple(packed.shape) == (7, 1, 528)
    assert tuple(unpacked.shape) == (7, 1, 512)
    cosine = torch.nn.functional.cosine_similarity(
        unpacked[:, 0].flatten(), latent.float().flatten(), dim=0
    )
    assert float(cosine) > 0.999


def test_glm_next_packed_page_view_preserves_padded_page_stride() -> None:
    from b12x.attention._shared.mla.api import _is_supported_packed_kv_cache_view
    from b12x.attention._shared.mla.kernel import _cache_block_stride_bytes

    page_size = 64
    record_bytes = 528
    semantic_page_bytes = page_size * record_bytes
    pooled_tail_bytes = (page_size // 4) * 128 * 2
    padded_page_bytes = semantic_page_bytes + pooled_tail_bytes
    storage = torch.empty((3, padded_page_bytes), dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(3, page_size, record_bytes),
        stride=(padded_page_bytes, record_bytes, 1),
    )

    assert tuple(cache.shape) == (3, 64, 528)
    assert not cache.is_contiguous()
    assert _is_supported_packed_kv_cache_view(cache, page_size=page_size)
    assert (
        _cache_block_stride_bytes(
            cache,
            page_size=page_size,
            model_type=ModelType.GLM_NEXT,
            record_bytes=record_bytes,
        )
        == padded_page_bytes
    )


def test_glm_next_cache_writer_uses_wide_padded_page_offsets() -> None:
    page_size = 64
    page_stride = page_size * 528 + (page_size // 4) * 128 * 2
    high_page = 2**31 // page_stride + 1
    slot = high_page * page_size + 7

    offset = _glm_next_cache_byte_offset(
        slot,
        block_size=page_size,
        block_stride=page_stride,
    )

    assert offset == high_page * page_stride + 7 * 528
    assert offset > 2**31
    annotations = _glm_next_cache_record_address.__annotations__
    assert annotations["slot"] == "Int64"
    assert annotations["block_stride"] == "Int64"
    assert annotations["entry_stride"] == "Int64"
    assert annotations["return"] == "Int64"


def _valid_glm_next_writer_args() -> list[torch.Tensor]:
    page_size = 64
    page_stride = page_size * 528 + (page_size // 4) * 128 * 2
    backing = torch.empty((2, page_stride), dtype=torch.uint8)
    cache = torch.as_strided(
        backing,
        size=(2, page_size, 528),
        stride=(page_stride, 528, 1),
    )
    return [
        torch.empty((2, 512), dtype=torch.bfloat16),
        cache,
        torch.arange(2, dtype=torch.int64),
    ]


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("source_shape", ValueError, r"kv_c must be \(num_tokens, 512\)"),
        ("source_dtype", TypeError, "kv_c must be BF16"),
        ("cache_shape", ValueError, "kv_cache must be"),
        ("cache_dtype", TypeError, "kv_cache must be uint8"),
        ("slot_dtype", TypeError, "1-D int32 or int64"),
        ("short_source", ValueError, "must cover slot_mapping"),
        ("record_stride", ValueError, "packed semantic records"),
        ("overlapping_pages", ValueError, "page stride must cover"),
        ("cpu_device", ValueError, "all tensors must be on CUDA"),
        ("cpu_device_int32", ValueError, "all tensors must be on CUDA"),
    ],
)
def test_glm_next_cache_writer_rejects_invalid_contracts(
    case: str,
    error: type[Exception],
    match: str,
) -> None:
    kv_c, cache, slots = _valid_glm_next_writer_args()
    if case == "source_shape":
        kv_c = torch.empty((2, 511), dtype=torch.bfloat16)
    elif case == "source_dtype":
        kv_c = kv_c.float()
    elif case == "cache_shape":
        cache = torch.empty((2, 64, 527), dtype=torch.uint8)
    elif case == "cache_dtype":
        cache = torch.empty((2, 64, 528), dtype=torch.bfloat16)
    elif case == "slot_dtype":
        slots = slots.to(torch.int16)
    elif case == "short_source":
        kv_c = kv_c[:1]
    elif case == "record_stride":
        storage = torch.empty((2, 64, 544), dtype=torch.uint8)
        cache = torch.as_strided(
            storage,
            size=(2, 64, 528),
            stride=(64 * 544, 544, 1),
        )
    elif case == "overlapping_pages":
        semantic_page_bytes = 64 * 528
        storage = torch.empty(2 * semantic_page_bytes, dtype=torch.uint8)
        cache = torch.as_strided(
            storage,
            size=(2, 64, 528),
            stride=(semantic_page_bytes - 16, 528, 1),
        )
    elif case == "cpu_device_int32":
        slots = slots.to(torch.int32)
    elif case != "cpu_device":
        raise AssertionError(f"unknown case {case}")

    with pytest.raises(error, match=match):
        concat_and_cache_glm_next_mla(kv_c, cache, slots)


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_glm_next_cache_writer_preserves_padded_tail_and_record_abi(
    slot_dtype: torch.dtype,
) -> None:
    device = require_sm120()
    torch.manual_seed(20260826)
    page_size = 64
    num_pages = 3
    semantic_page_bytes = page_size * 528
    pooled_tail_bytes = (page_size // 4) * 128 * 2
    page_stride = semantic_page_bytes + pooled_tail_bytes
    sentinel = 0xA5
    backing = torch.full(
        (num_pages, page_stride),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )
    cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, 528),
        stride=(page_stride, 528, 1),
    )
    kv_c = (torch.randn((6, 512), device=device) / 4).to(torch.bfloat16)
    capacity = num_pages * page_size
    huge_slot = 2**30 + 17 if slot_dtype == torch.int32 else 2**40 + 17
    slots = torch.tensor(
        [0, -1, capacity, 65, huge_slot, 130],
        dtype=slot_dtype,
        device=device,
    )

    sparse_mla.concat_and_cache_glm_next_mla(kv_c, cache, slots)
    torch.cuda.synchronize(device)

    assert torch.all(backing[:, semantic_page_bytes:] == sentinel)
    changed = (cache != sentinel).any(dim=-1).nonzero(as_tuple=False)
    expected_changed = torch.tensor(
        [[0, 0], [1, 1], [2, 2]], dtype=torch.int64, device=device
    )
    assert torch.equal(changed, expected_changed)

    records = torch.stack((cache[0, 0], cache[1, 1], cache[2, 2])).cpu()
    dequantized = unpack_mla_kv_cache_reference(records.unsqueeze(1))[:, 0]
    source = (
        kv_c.index_select(0, torch.tensor([0, 3, 5], dtype=torch.int64, device=device))
        .float()
        .cpu()
    )
    cosine = torch.nn.functional.cosine_similarity(
        dequantized.flatten(), source.flatten(), dim=0
    )
    assert float(cosine) > 0.999

    actual_scales = records[:, 512:].contiguous().view(torch.float32)
    expected_scales = source.reshape(3, 4, 128).abs().amax(dim=-1) / 448.0
    expected_scales = torch.where(
        expected_scales > 0,
        expected_scales,
        torch.ones_like(expected_scales),
    )
    torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_glm_next_nvfp4_writer_uses_inline_scale_record(
    slot_dtype: torch.dtype,
) -> None:
    device = require_sm120()
    torch.manual_seed(20260831)
    cache = torch.full(
        (2, 64, _GLM_NEXT_NVFP4_RECORD_BYTES),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    latent = (torch.randn((4, 512), device=device) / 4).to(torch.bfloat16)
    slots = torch.tensor([0, -1, 65, 128], dtype=slot_dtype, device=device)

    sparse_mla.compile_glm_next_mla_cache_writer(latent, cache, slots)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
    torch.cuda.synchronize(device)

    assert torch.all(cache[0, 1] == 0xA5)
    records = torch.stack((cache[0, 0], cache[1, 1])).cpu()
    dequantized, _ = dequantize_nvfp4_mla_nope(
        records,
        nope_bytes=256,
        group_scales_offset=256,
        group_scales_end=288,
        latent_scale_offset=292,
    )
    source = (
        latent.index_select(0, torch.tensor([0, 2], dtype=torch.int64, device=device))
        .float()
        .cpu()
    )
    cosine = torch.nn.functional.cosine_similarity(
        dequantized.flatten(), source.flatten(), dim=0
    )
    assert float(cosine) > 0.97
    assert torch.all(records[:, 288:292] == 0)
    assert torch.all(records[:, 296:304] == 0)
    assert torch.all(records[:, 292:296].contiguous().view(torch.float32) > 0)


@torch.inference_mode()
def test_glm_next_nvfp4_writer_rejects_capture_compile_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    cache = torch.empty(
        (1, 64, _GLM_NEXT_NVFP4_RECORD_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    latent = torch.zeros((1, 512), dtype=torch.bfloat16, device=device)
    slots = torch.zeros((1,), dtype=torch.int64, device=device)
    clear_glm_next_mla_kv_cache_kernel_cache()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="compile miss during CUDA graph capture"):
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)


def test_glm_next_public_cpu_reference_path_preserves_model_identity() -> None:
    torch.manual_seed(20260826)
    rows, heads, cache_tokens, width = 2, 8, 16, 8
    latent = torch.randn((cache_tokens, 512), dtype=torch.bfloat16) / 4
    cache = pack_mla_kv_cache_reference(latent)
    q = torch.randn((rows, heads, 512), dtype=torch.bfloat16) / 4
    selected = torch.stack(
        [torch.randperm(cache_tokens)[:width].sort().values for _ in range(rows)]
    ).to(torch.int32)
    cache_seqlens = torch.full((rows,), cache_tokens, dtype=torch.int32)
    active = torch.full((rows,), width, dtype=torch.int32)

    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device="cpu",
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=width,
            softmax_scale=_GLM_NEXT_SM_SCALE,
            kv_dtype=torch.uint8,
            head_dim=512,
            v_head_dim=512,
            page_size=1,
            model_type=ModelType.GLM_NEXT,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.full(spec.shape, 0xA5, dtype=spec.dtype)
    before = scratch.clone()
    binding = sparse_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=cache,
        selected_indices=selected,
        cache_lengths=cache_seqlens,
        selected_lengths=active,
    )
    torch.testing.assert_close(scratch, before)
    assert tuple(inspect.signature(sparse_mla.run).parameters) == ("binding",)
    assert "run_decode" not in sparse_mla.__all__
    assert "run_extend" not in sparse_mla.__all__

    actual = sparse_mla.run(binding)
    expected = sparse_mla_reference(
        q_all=q,
        kv_cache=cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=256**-0.5,
        v_head_dim=512,
    )
    torch.testing.assert_close(actual, expected)

    assert sparse_mla.ModelType.GLM_NEXT == ModelType.GLM_NEXT
    assert callable(sparse_mla.compile_glm_next_mla_cache_writer)
    assert callable(sparse_mla.concat_and_cache_glm_next_mla)
    assert callable(sparse_mla.concat_and_cache_glm_next_mla_fp8)
    assert callable(sparse_mla.concat_and_cache_glm_next_mla_nvfp4)
    with pytest.raises(TypeError, match="binding must be sparse_mla.Binding"):
        sparse_mla.run(binding.runtime)


@pytest.mark.parametrize("container_width", [2051, 2112])
def test_glm_next_prefill_routes_exact_or_aligned_selector_width(
    monkeypatch: pytest.MonkeyPatch,
    container_width: int,
) -> None:
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill

    calls: list[dict] = []

    def fake_run_unified_prefill_mg(**kwargs):
        calls.append(kwargs)
        return kwargs["output"], kwargs["lse_out"]

    monkeypatch.setattr(
        prefill_mg,
        "run_unified_prefill_mg",
        fake_run_unified_prefill_mg,
    )

    q = torch.empty((2, 8, 512), dtype=torch.bfloat16)
    cache = torch.empty((4, 528), dtype=torch.uint8)
    selected = torch.full((2, container_width), -1, dtype=torch.int32)
    selected[:, :2051] = 0
    active = torch.full((2,), 2051, dtype=torch.int32)

    output, lse = run_unified_prefill(
        q=q,
        kv_cache=cache,
        topk_indices=selected,
        topk_length=active,
        sm_scale=256**-0.5,
        page_block_size=64,
        model_type=ModelType.GLM_NEXT,
    )

    assert output.shape == (2, 8, 512)
    assert lse.shape == (2, 8)
    assert len(calls) == 1
    assert calls[0]["model_type"] == ModelType.GLM_NEXT
    assert calls[0]["scale_format"] == ScaleFormat.ARBITRARY_FP32
    assert calls[0]["fp8_rope"] is False
    torch.testing.assert_close(calls[0]["topk_length"], active)


@torch.inference_mode()
def test_glm_next_production_decode_matches_packed_record_oracle() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260827)
    rows, heads, num_records, width = 2, 8, 3 * _GLM_NEXT_PAGE_SIZE, 129
    latent = (
        torch.randn(
            (num_records, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    cache = torch.empty(
        (
            num_records // _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_RECORD_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(num_records, dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)

    q = (
        torch.randn(
            (rows, heads, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    selected = torch.stack(
        [
            torch.randperm(num_records, generator=generator, device=device)[:width]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    active = torch.tensor([width, 65], dtype=torch.int32, device=device)
    cache_seqlens = torch.full((rows,), num_records, dtype=torch.int32, device=device)
    _, binding = _glm_next_plan_and_binding(
        device=device,
        kv_cache=cache,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
        return_lse=True,
    )

    actual, actual_lse = sparse_mla.run(binding)
    expected, expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=cache.view(num_records, 1, _GLM_NEXT_RECORD_BYTES),
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    torch.cuda.synchronize(device)

    _assert_glm_next_attention_close(actual, expected)
    assert bool(torch.isfinite(actual_lse).all().item())
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)


@torch.inference_mode()
def test_glm_next_nvfp4_decode_matches_dequantized_record_oracle() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260831)
    rows, heads, num_records, width = 2, 8, 3 * _GLM_NEXT_PAGE_SIZE, 129
    latent = (
        torch.randn(
            (num_records, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    cache = torch.empty(
        (
            num_records // _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_NVFP4_RECORD_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(num_records, dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)

    q = (
        torch.randn(
            (rows, heads, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    selected = torch.stack(
        [
            torch.randperm(num_records, generator=generator, device=device)[:width]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    active = torch.tensor([width, 65], dtype=torch.int32, device=device)
    cache_seqlens = torch.full((rows,), num_records, dtype=torch.int32, device=device)
    _, binding = _glm_next_plan_and_binding(
        device=device,
        kv_cache=cache,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
        return_lse=True,
    )

    actual, actual_lse = sparse_mla.run(binding)
    dequantized, _ = dequantize_nvfp4_mla_nope(
        cache.view(num_records, _GLM_NEXT_NVFP4_RECORD_BYTES),
        latent_scale_offset=292,
    )
    expected, expected_lse = _sparse_attention_reference(
        q_all=q,
        k_all=dequantized,
        v_all=dequantized,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        return_lse=True,
    )
    torch.cuda.synchronize(device)

    _assert_glm_next_attention_close(actual, expected)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        captured_output, captured_lse = sparse_mla.run(binding)
    assert captured_output.data_ptr() == actual.data_ptr()
    assert captured_lse.data_ptr() == actual_lse.data_ptr()

    q.copy_(
        (torch.randn(q.shape, generator=generator, device=device) / 4).to(
            torch.bfloat16
        )
    )
    latent.copy_(
        (torch.randn(latent.shape, generator=generator, device=device) / 4).to(
            torch.bfloat16
        )
    )
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
    dequantized, _ = dequantize_nvfp4_mla_nope(
        cache.view(num_records, _GLM_NEXT_NVFP4_RECORD_BYTES),
        latent_scale_offset=292,
    )
    replay_expected, replay_expected_lse = _sparse_attention_reference(
        q_all=q,
        k_all=dequantized,
        v_all=dequantized,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        return_lse=True,
    )
    allocator_before = _allocator_counters(device)
    graph.replay()
    torch.cuda.synchronize(device)

    assert _allocator_counters(device) == allocator_before
    _assert_glm_next_attention_close(captured_output, replay_expected)
    torch.testing.assert_close(captured_lse, replay_expected_lse, rtol=0.0, atol=0.05)


@torch.inference_mode()
def test_glm_next_hybrid_manager_page_replays_across_page_boundary() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260829)
    page_size = 2304
    num_pages = 2
    page_stride = (
        page_size * _GLM_NEXT_RECORD_BYTES
        + page_size // 4 * 128 * torch.bfloat16.itemsize
    )
    backing = torch.zeros((num_pages, page_stride), dtype=torch.uint8, device=device)
    cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, _GLM_NEXT_RECORD_BYTES),
        stride=(page_stride, _GLM_NEXT_RECORD_BYTES, 1),
    )
    slots = torch.tensor(
        [63, 64, 2302, 2303, 2304, 2305, 4607],
        dtype=torch.int64,
        device=device,
    )
    latent = (
        torch.randn(
            (slots.numel(), _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)

    q = (
        torch.randn(
            (1, 8, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    selected = slots.to(torch.int32).unsqueeze(0).contiguous()
    active = torch.full((1,), slots.numel(), dtype=torch.int32, device=device)
    cache_seqlens = torch.full(
        (1,), num_pages * page_size, dtype=torch.int32, device=device
    )
    _, binding = _glm_next_plan_and_binding(
        device=device,
        kv_cache=cache,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
        page_size=page_size,
        return_lse=True,
        lse_scale="natural",
    )

    actual, actual_lse = sparse_mla.run(binding)
    flat_cache = cache.contiguous().view(
        num_pages * page_size, 1, _GLM_NEXT_RECORD_BYTES
    )
    expected, expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=flat_cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    expected_lse.mul_(0.6931471805599453)
    torch.cuda.synchronize(device)

    assert cache.stride(0) == page_stride
    _assert_glm_next_attention_close(actual, expected)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        captured_output, captured_lse = sparse_mla.run(binding)
    assert captured_output.data_ptr() == actual.data_ptr()
    assert captured_lse.data_ptr() == actual_lse.data_ptr()

    for _ in range(2):
        q.copy_(
            (torch.randn(q.shape, generator=generator, device=device) / 4).to(
                torch.bfloat16
            )
        )
        latent.copy_(
            (
                torch.randn(
                    latent.shape,
                    generator=generator,
                    device=device,
                )
                / 4
            ).to(torch.bfloat16)
        )
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        flat_cache.copy_(
            cache.contiguous().view(num_pages * page_size, 1, _GLM_NEXT_RECORD_BYTES)
        )
        replay_expected, replay_expected_lse = sparse_mla_reference(
            q_all=q,
            kv_cache=flat_cache,
            page_table_1=selected,
            active_token_counts=active,
            sm_scale=_GLM_NEXT_SM_SCALE,
            v_head_dim=_GLM_NEXT_HEAD_DIM,
            return_lse=True,
        )
        replay_expected_lse.mul_(0.6931471805599453)
        allocator_before = _allocator_counters(device)
        graph.replay()
        torch.cuda.synchronize(device)

        assert _allocator_counters(device) == allocator_before
        _assert_glm_next_attention_close(captured_output, replay_expected)
        torch.testing.assert_close(
            captured_lse, replay_expected_lse, rtol=0.0, atol=0.05
        )


@torch.inference_mode()
def test_glm_next_production_prefill_2051_replays_without_allocation() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260828)
    rows, heads = 1, 8
    active_width = 2051
    container_width = 2112
    num_records = container_width
    latent = (
        torch.randn(
            (num_records, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    cache = torch.empty(
        (
            num_records // _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_RECORD_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(num_records, dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)

    q = (
        torch.randn(
            (rows, heads, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    selected = torch.full((rows, container_width), -1, dtype=torch.int32, device=device)
    selected[0, :active_width] = torch.randperm(
        num_records, generator=generator, device=device
    )[:active_width].to(torch.int32)
    active = torch.full((rows,), active_width, dtype=torch.int32, device=device)
    cache_seqlens = torch.full((rows,), num_records, dtype=torch.int32, device=device)
    _, binding = _glm_next_plan_and_binding(
        device=device,
        kv_cache=cache,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
        mode="extend",
        return_lse=True,
    )

    actual, actual_lse = sparse_mla.run(binding)
    expected, expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=cache.view(num_records, 1, _GLM_NEXT_RECORD_BYTES),
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    torch.cuda.synchronize(device)
    _assert_glm_next_attention_close(actual, expected)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)

    assert actual.data_ptr() == binding.runtime.scratch.output_buffer.data_ptr()
    assert binding.runtime.scratch.final_lse is not None
    assert actual_lse.data_ptr() == binding.runtime.scratch.final_lse.data_ptr()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        captured_output, captured_lse = sparse_mla.run(binding)
    assert captured_output.data_ptr() == actual.data_ptr()
    assert captured_lse.data_ptr() == actual_lse.data_ptr()

    q.copy_(
        (
            torch.randn(
                q.shape,
                generator=generator,
                device=device,
            )
            / 4
        ).to(torch.bfloat16)
    )
    replay_expected, replay_expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=cache.view(num_records, 1, _GLM_NEXT_RECORD_BYTES),
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    allocator_before = _allocator_counters(device)
    graph.replay()
    torch.cuda.synchronize(device)

    assert _allocator_counters(device) == allocator_before
    _assert_glm_next_attention_close(captured_output, replay_expected)
    torch.testing.assert_close(captured_lse, replay_expected_lse, rtol=0.0, atol=0.05)


@torch.inference_mode()
def test_glm_next_nvfp4_prefill_2051_matches_dequantized_record_oracle() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260831)
    rows, heads = 1, 8
    active_width = 2051
    container_width = 2112
    num_records = container_width
    latent = (
        torch.randn(
            (num_records, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    cache = torch.empty(
        (
            num_records // _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_NVFP4_RECORD_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(num_records, dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)

    q = (
        torch.randn(
            (rows, heads, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    selected = torch.full((rows, container_width), -1, dtype=torch.int32, device=device)
    selected[0, :active_width] = torch.randperm(
        num_records, generator=generator, device=device
    )[:active_width].to(torch.int32)
    active = torch.full((rows,), active_width, dtype=torch.int32, device=device)
    cache_seqlens = torch.full((rows,), num_records, dtype=torch.int32, device=device)
    _, binding = _glm_next_plan_and_binding(
        device=device,
        kv_cache=cache,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
        mode="extend",
        return_lse=True,
    )

    actual, actual_lse = sparse_mla.run(binding)
    dequantized, _ = dequantize_nvfp4_mla_nope(
        cache.view(num_records, _GLM_NEXT_NVFP4_RECORD_BYTES),
        latent_scale_offset=292,
    )
    expected, expected_lse = _sparse_attention_reference(
        q_all=q,
        k_all=dequantized,
        v_all=dequantized,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        return_lse=True,
    )
    torch.cuda.synchronize(device)

    _assert_glm_next_attention_close(actual, expected)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        captured_output, captured_lse = sparse_mla.run(binding)
    assert captured_output.data_ptr() == actual.data_ptr()
    assert captured_lse.data_ptr() == actual_lse.data_ptr()

    q.copy_(
        (torch.randn(q.shape, generator=generator, device=device) / 4).to(
            torch.bfloat16
        )
    )
    latent.copy_(
        (torch.randn(latent.shape, generator=generator, device=device) / 4).to(
            torch.bfloat16
        )
    )
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
    replay_dequantized, _ = dequantize_nvfp4_mla_nope(
        cache.view(num_records, _GLM_NEXT_NVFP4_RECORD_BYTES),
        latent_scale_offset=292,
    )
    replay_expected, replay_expected_lse = _sparse_attention_reference(
        q_all=q,
        k_all=replay_dequantized,
        v_all=replay_dequantized,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        return_lse=True,
    )
    allocator_before = _allocator_counters(device)
    graph.replay()
    torch.cuda.synchronize(device)

    assert _allocator_counters(device) == allocator_before
    _assert_glm_next_attention_close(captured_output, replay_expected)
    torch.testing.assert_close(captured_lse, replay_expected_lse, rtol=0.0, atol=0.05)


@torch.inference_mode()
def test_glm_next_tp4_hybrid_page_prefill_matches_oracle_and_replays() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260830)
    rows, heads = 26, 16
    page_size = 2176
    num_pages = 3
    container_width = 2051
    semantic_page_bytes = page_size * _GLM_NEXT_RECORD_BYTES
    pooled_tail_bytes = page_size // 4 * 128 * torch.bfloat16.itemsize
    page_stride = semantic_page_bytes + pooled_tail_bytes
    sentinel = 0xA5
    backing = torch.full(
        (num_pages, page_stride), sentinel, dtype=torch.uint8, device=device
    )
    cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, _GLM_NEXT_RECORD_BYTES),
        stride=(page_stride, _GLM_NEXT_RECORD_BYTES, 1),
    )
    latent = (
        torch.randn(
            (rows, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    slots = 2 * page_size + torch.arange(rows, dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)

    q = (
        torch.randn(
            (rows, heads, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    selected = torch.full((rows, container_width), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        selected[row, : row + 1] = slots[: row + 1].to(torch.int32)
    active = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    cache_seqlens = active.clone()
    _, binding = _glm_next_plan_and_binding(
        device=device,
        kv_cache=cache,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
        mode="extend",
        page_size=page_size,
        return_lse=True,
    )

    actual, actual_lse = sparse_mla.run(binding)
    flat_cache = cache.contiguous().view(
        num_pages * page_size, 1, _GLM_NEXT_RECORD_BYTES
    )
    expected, expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=flat_cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    torch.cuda.synchronize(device)

    assert cache.stride(0) == page_stride
    assert torch.all(backing[:, semantic_page_bytes:] == sentinel)
    _assert_glm_next_attention_close(actual, expected)
    per_row_cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(rows, -1),
        expected.float().reshape(rows, -1),
        dim=1,
    )
    assert float(per_row_cosine.min()) > 0.995
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        captured_output, captured_lse = sparse_mla.run(binding)
    assert captured_output.data_ptr() == actual.data_ptr()
    assert captured_lse.data_ptr() == actual_lse.data_ptr()

    q.copy_(
        (torch.randn(q.shape, generator=generator, device=device) / 4).to(
            torch.bfloat16
        )
    )
    latent.copy_(
        (torch.randn(latent.shape, generator=generator, device=device) / 4).to(
            torch.bfloat16
        )
    )
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
    flat_cache.copy_(
        cache.contiguous().view(num_pages * page_size, 1, _GLM_NEXT_RECORD_BYTES)
    )
    replay_expected, replay_expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=flat_cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    allocator_before = _allocator_counters(device)
    graph.replay()
    torch.cuda.synchronize(device)

    assert _allocator_counters(device) == allocator_before
    assert torch.all(backing[:, semantic_page_bytes:] == sentinel)
    _assert_glm_next_attention_close(captured_output, replay_expected)
    torch.testing.assert_close(captured_lse, replay_expected_lse, rtol=0.0, atol=0.05)


@torch.inference_mode()
def test_glm_next_writer_and_reader_use_int64_for_live_high_page() -> None:
    device = require_sm120()
    page_size = _GLM_NEXT_PAGE_SIZE
    semantic_page_bytes = page_size * _GLM_NEXT_RECORD_BYTES
    pooled_tail_bytes = (page_size // 4) * 128 * torch.bfloat16.itemsize
    page_stride_bytes = semantic_page_bytes + pooled_tail_bytes
    int32_max = torch.iinfo(torch.int32).max
    high_page = int32_max // page_stride_bytes + 2
    num_pages = high_page + 1
    required_bytes = num_pages * page_stride_bytes
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserve_bytes = 2 * 1024**3
    if free_bytes < required_bytes + reserve_bytes:
        pytest.skip(
            "live GLM_NEXT high-page test requires "
            f"{required_bytes + reserve_bytes} bytes free, found {free_bytes}"
        )
    try:
        backing = torch.empty(
            (num_pages, page_stride_bytes), dtype=torch.uint8, device=device
        )
    except torch.OutOfMemoryError:
        pytest.skip(
            "CUDA allocator could not reserve the required mostly-uninitialized "
            f"{required_bytes}-byte GLM_NEXT cache"
        )
    cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, _GLM_NEXT_RECORD_BYTES),
        stride=(page_stride_bytes, _GLM_NEXT_RECORD_BYTES, 1),
    )
    local_slot = 7
    high_slot = high_page * page_size + local_slot
    assert high_page * page_stride_bytes > int32_max
    assert high_slot < int32_max

    live_latent = (
        torch.linspace(
            -0.75,
            0.75,
            _GLM_NEXT_HEAD_DIM,
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .to(torch.bfloat16)
    )
    sources = torch.cat((torch.zeros_like(live_latent), live_latent))
    slots = torch.tensor([0, high_slot], dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(sources, cache, slots)

    q = torch.randn((1, 8, _GLM_NEXT_HEAD_DIM), dtype=torch.bfloat16, device=device)
    selected = torch.tensor([[high_slot]], dtype=torch.int32, device=device)
    active = torch.ones((1,), dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([high_slot + 1], dtype=torch.int32, device=device)
    _, binding = _glm_next_plan_and_binding(
        device=device,
        kv_cache=cache,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
    )
    actual = sparse_mla.run(binding)
    expected_value = unpack_mla_kv_cache_reference(
        cache[high_page, local_slot].reshape(1, 1, _GLM_NEXT_RECORD_BYTES)
    )[0, 0]
    expected = expected_value.view(1, 1, -1).expand_as(actual)
    torch.cuda.synchronize(device)

    assert (
        _glm_next_cache_byte_offset(
            high_slot,
            block_size=page_size,
            block_stride=page_stride_bytes,
        )
        > int32_max
    )
    _assert_glm_next_attention_close(actual, expected)


@torch.inference_mode()
def test_glm_next_nvfp4_writer_and_reader_use_int64_for_live_high_page() -> None:
    device = require_sm120()
    page_size = _GLM_NEXT_PAGE_SIZE
    index_page_bytes = 64 * 132
    semantic_page_bytes = page_size * _GLM_NEXT_NVFP4_RECORD_BYTES
    record_region_bytes = (
        (semantic_page_bytes + index_page_bytes - 1) // index_page_bytes
    ) * index_page_bytes
    pooled_tail_bytes = page_size * 33
    page_stride_bytes = record_region_bytes + pooled_tail_bytes
    int32_max = torch.iinfo(torch.int32).max
    high_page = int32_max // page_stride_bytes + 2
    num_pages = high_page + 1
    required_bytes = num_pages * page_stride_bytes
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserve_bytes = 2 * 1024**3
    if free_bytes < required_bytes + reserve_bytes:
        pytest.skip(
            "live GLM_NEXT NVFP4 high-page test requires "
            f"{required_bytes + reserve_bytes} bytes free, found {free_bytes}"
        )
    try:
        backing = torch.empty(
            (num_pages, page_stride_bytes), dtype=torch.uint8, device=device
        )
    except torch.OutOfMemoryError:
        pytest.skip(
            "CUDA allocator could not reserve the required mostly-uninitialized "
            f"{required_bytes}-byte GLM_NEXT NVFP4 cache"
        )
    cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, _GLM_NEXT_NVFP4_RECORD_BYTES),
        stride=(page_stride_bytes, _GLM_NEXT_NVFP4_RECORD_BYTES, 1),
    )
    local_slot = 7
    high_slot = high_page * page_size + local_slot
    assert high_page * page_stride_bytes > int32_max
    assert high_slot < int32_max

    live_latent = (
        torch.linspace(
            -0.75,
            0.75,
            _GLM_NEXT_HEAD_DIM,
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .to(torch.bfloat16)
    )
    sources = torch.cat((torch.zeros_like(live_latent), live_latent))
    slots = torch.tensor([0, high_slot], dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(sources, cache, slots)

    q = torch.randn((1, 8, _GLM_NEXT_HEAD_DIM), dtype=torch.bfloat16, device=device)
    selected = torch.tensor([[high_slot]], dtype=torch.int32, device=device)
    active = torch.ones((1,), dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([high_slot + 1], dtype=torch.int32, device=device)
    _, binding = _glm_next_plan_and_binding(
        device=device,
        kv_cache=cache,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
    )
    actual = sparse_mla.run(binding)
    expected_value, _ = dequantize_nvfp4_mla_nope(
        cache[high_page, local_slot].reshape(1, _GLM_NEXT_NVFP4_RECORD_BYTES),
        latent_scale_offset=292,
    )
    expected = expected_value[0].view(1, 1, -1).expand_as(actual)
    torch.cuda.synchronize(device)

    assert (
        _glm_next_cache_byte_offset(
            high_slot,
            block_size=page_size,
            block_stride=page_stride_bytes,
            entry_stride=_GLM_NEXT_NVFP4_RECORD_BYTES,
        )
        > int32_max
    )
    _assert_glm_next_attention_close(actual, expected)
