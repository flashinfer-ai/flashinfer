from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from b12x.attention import qsa
from b12x.attention.qsa import _contract as qsa_contract
from b12x.attention.qsa import _sparse_gqa as sparse_gqa
from b12x.attention.qsa import _sparse_gqa_cute_config as cute_config
from b12x.attention.qsa._contract import _target_splits
from b12x.attention.qsa._sparse_gqa import launch_sparse_paged_gqa

from ..conftest import require_b12x as require_sm120


@pytest.mark.parametrize("heads", [6, 12, 24])
@pytest.mark.parametrize("splits", [1, 16, 32, 64])
def test_split_merge_matches_softmax_with_empty_heads_and_changed_graph_inputs(
    heads: int,
    splits: int,
) -> None:
    from b12x.attention.paged._selected_forward import launch_sparse_gqa_merge

    device = require_sm120()
    torch.manual_seed(20260905)
    partials = torch.randn(16, splits, heads, 256, device=device)
    lse = torch.randn(16, splits, heads, device=device) * 8
    lse[:, 1::3] = -torch.inf
    lse[0, :, 0] = -torch.inf
    output = torch.empty(16, heads, 256, dtype=torch.bfloat16, device=device)
    output_lse = torch.empty(16, heads, dtype=torch.float32, device=device)

    for rows in (1, 4, 16):

        def launch() -> None:
            launch_sparse_gqa_merge(
                partial_output=partials[:rows],
                partial_lse=lse[:rows],
                output=output[:rows],
                output_lse=output_lse[:rows],
                rows=rows,
                splits=splits,
            )

        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for _ in range(2):
            partials.mul_(0.75)
            lse[:, :, -1].add_(2)
            graph.replay()
            probabilities = torch.softmax(lse[:rows].double(), dim=1).nan_to_num()
            expected = (partials[:rows].double() * probabilities[..., None]).sum(1)
            torch.testing.assert_close(
                output[:rows].double(),
                expected,
                rtol=4e-3,
                atol=3e-5,
            )
            torch.testing.assert_close(
                output_lse[:rows].double(),
                torch.logsumexp(lse[:rows].double(), dim=1),
                rtol=1e-5,
                atol=1e-5,
            )
            assert torch.count_nonzero(output[0, 0]) == 0
            assert torch.count_nonzero(output[:rows, -1]) > 0


@pytest.mark.parametrize("heads", [6, 12, 24])
def test_cooperative_merge_preserves_order_bitwise_under_changed_graph_inputs(
    monkeypatch,
    heads: int,
) -> None:
    """Tiling head dimensions cannot change the split-pair reduction tree."""
    from b12x.attention.paged import _selected_forward as implementation
    from b12x._lib.runtime_control import kernel_resolution_guard

    device = require_sm120()
    torch.manual_seed(20260905)
    geometries = ((1, 64), (4, 32), (16, 16), (4, 1), (64, 16))
    # Each graph consumes these caller-owned contiguous buffers directly.
    # Capturing a contiguous() adapter would also capture its staging copies.
    buffers = {
        (rows, splits): (
            torch.randn(rows, splits, heads, 256, device=device),
            torch.randn(rows, splits, heads, device=device) * 8,
            torch.empty(rows, heads, 256, device=device, dtype=torch.bfloat16),
            torch.empty(rows, heads, device=device, dtype=torch.float32),
        )
        for rows, splits in geometries
    }
    original = implementation._SparseGqaMergeKernel
    candidate = implementation._ShapeAdaptiveSparseGqaMergeKernel
    graphs = {}
    try:
        for label, kernel in (("serial", original), ("cooperative", candidate)):
            monkeypatch.setattr(implementation, "_MERGE_KERNEL", kernel)
            implementation.clear_caches()

            def launch(rows, splits):
                partials, lse, output, output_lse = buffers[rows, splits]
                implementation.launch_sparse_gqa_merge(
                    partial_output=partials,
                    partial_lse=lse,
                    output=output,
                    output_lse=output_lse,
                    rows=rows,
                    splits=splits,
                )

            launch(1, 64)
            warmed = tuple(implementation._MERGE_CACHE.items())
            with kernel_resolution_guard("cooperative QSA merge live rows/splits"):
                for rows, splits in geometries:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch(rows, splits)
                    graphs[label, rows, splits] = graph
                assert tuple(implementation._MERGE_CACHE.items()) == warmed
        for rows, splits in geometries:
            partials, lse, output, output_lse = buffers[rows, splits]
            for _ in range(3):
                partials.normal_()
                lse.normal_()
                lse[:, 1::3] = -torch.inf
                lse[0, :, 0] = -torch.inf
                pointers = tuple(
                    t.data_ptr() for t in (partials, lse, output, output_lse)
                )

                def replay(label):
                    output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graphs[label, rows, splits].replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                        assert after[key] == before[key]
                    assert tuple(
                        t.data_ptr() for t in (partials, lse, output, output_lse)
                    ) == pointers

                replay("serial")
                expected = output.clone()
                expected_lse = output_lse.clone()
                replay("cooperative")
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                torch.testing.assert_close(output_lse, expected_lse, rtol=0, atol=0)
                assert torch.count_nonzero(output[0, 0]) == 0
                assert torch.count_nonzero(output[:rows, -1]) > 0
    finally:
        implementation.clear_caches()


@pytest.mark.parametrize("selection_width", [2051, 2054])
def test_split_prewarm_matches_planned_selection_capacity(selection_width: int) -> None:
    """Warmup must cover both ordinary selection and a three-column MTP tail."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.paged import _selected_forward as implementation

    device = require_sm120()
    query = torch.randn(4, 6, 256, dtype=torch.bfloat16, device=device)
    keys = torch.randn(1, 16, 1, 256, dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)
    requests = torch.zeros(4, dtype=torch.int32, device=device)
    positions = torch.full((4,), 15, dtype=torch.int64, device=device)
    table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    selected = torch.full((4, selection_width), -1, dtype=torch.int32, device=device)
    selected[:, :16] = torch.arange(16, device=device)
    partials = torch.empty(4, 16, 6, 256, device=device)
    lse = torch.empty(4, 16, 6, device=device)
    implementation.clear_caches()
    implementation.precompile_sparse_gqa_split(
        query=query,
        key_cache=keys,
        value_cache=values,
        request_ids=requests,
        selection_width=selection_width,
    )
    with kernel_resolution_guard('planned selected-attention warmup capacity'):
        try:

            def launch(rows):
                implementation.launch_sparse_gqa_split(
                    query=query[:rows],
                    key_cache=keys,
                    value_cache=values,
                    k_descale=None,
                    v_descale=None,
                    block_table=table,
                    request_ids=requests[:rows],
                    selected_positions=selected[:rows],
                    query_positions=positions[:rows],
                    partial_output=partials[:rows],
                    partial_lse=lse[:rows],
                    softmax_scale=1 / 16,
                    splits=16,
                )

            # Compilation does not warm CUDA launch state or write caller storage.
            launch(1)
            for rows in (1, 4):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(rows)
                for _ in range(2):
                    query.normal_()
                    graph.replay()
                    probability = torch.softmax(lse[:rows].double(), dim=1).nan_to_num()
                    actual = (partials[:rows].double() * probability[..., None]).sum(1)
                    scores = (
                        torch.einsum(
                            "rhd,nd->rhn", query[:rows].double(), keys[0, :, 0].double()
                        )
                        / 16
                    )
                    expected = torch.einsum(
                        "rhn,nd->rhd", scores.softmax(-1), values[0, :, 0].double()
                    )
                    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.004)
                    assert torch.isfinite(actual).all()
                    assert torch.count_nonzero(actual) > 0
        finally:
            implementation.clear_caches()


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("live_counts", [(1, 4), (65, 66)])
def test_planned_selection_capacity_consumes_tail_with_frozen_graph_replay(
    kv_dtype: torch.dtype,
    live_counts: tuple[int, int],
) -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.paged import _selected_forward as implementation

    device = require_sm120()
    capacity, heads, width, splits = max(live_counts), 6, 2055, 64
    query = torch.zeros((capacity, heads, 256), device=device, dtype=torch.bfloat16)
    cache = torch.zeros((1, 16, 1, 256), device=device).to(kv_dtype)
    cache[0, 0].copy_(torch.full((1, 256), 0.5, device=device).to(kv_dtype))
    cache[0, 1].copy_(torch.full((1, 256), 1.0, device=device).to(kv_dtype))
    selected = torch.full((capacity, width), -1, device=device, dtype=torch.int32)
    selected[:, -1] = 0
    block_table = torch.zeros((1, 1), device=device, dtype=torch.int32)
    request_ids = torch.zeros(capacity, device=device, dtype=torch.int64)
    positions = torch.ones(capacity, device=device, dtype=torch.int64)
    output = torch.empty_like(query)
    partials = torch.empty((capacity, splits, heads, 256), device=device)
    lse = torch.empty((capacity, splits, heads), device=device)
    output_lse = torch.empty((capacity, heads), device=device)
    descale = torch.ones(1, device=device) if kv_dtype == torch.float8_e4m3fn else None
    buffers = (query, cache, selected, output, output_lse, partials, lse)
    pointers = tuple(t.data_ptr() for t in buffers)

    def launch(rows):
        return launch_sparse_paged_gqa(
            query=query[:rows], key_cache=cache, value_cache=cache,
            k_descale=descale, v_descale=descale, block_table=block_table,
            request_ids=request_ids[:rows], selected_positions=selected[:rows],
            query_positions=positions[:rows], output=output[:rows],
            output_lse=output_lse[:rows],
            partial_output=partials[:rows], partial_lse=lse[:rows],
            softmax_scale=1 / 16, block_n=16, splits=splits,
        )

    launch(live_counts[0])
    warmed = tuple(implementation._KERNEL_CACHE.items())
    warmed_merge = tuple(implementation._MERGE_CACHE.items())
    with kernel_resolution_guard('QSA fixed selected-position capacity across live rows'):
        for rows in live_counts:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(rows)
            for position, value in ((0, 0.5), (1, 1.0)):
                selected[:, -1] = position
                output.fill_(float("nan"))
                partials.fill_(float("nan"))
                lse.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                    assert after[key] == before[key]
                assert tuple(t.data_ptr() for t in buffers) == pointers
                assert torch.all(output[:rows] == value)
                assert tuple(implementation._KERNEL_CACHE.items()) == warmed
                assert tuple(implementation._MERGE_CACHE.items()) == warmed_merge


def test_qsa_caps_do_not_gate_architecture_or_tensor_parallel_layout() -> None:
    values = dict(
        device="cuda:0",
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_seq_len=32,
        num_main_cache_pages=2,
        num_compressed_cache_pages=2,
        main_page_size=16,
        compressed_page_size=4,
        q_heads=12,
        kv_heads=1,
        head_dim=256,
    )
    assert qsa.Caps(**values).device == torch.device("cuda:0")

    values["q_heads"] = 8
    values["kv_heads"] = 2
    assert qsa.Caps(**values).q_heads == 8

    values["q_heads"] = 7
    with pytest.raises(ValueError, match="q_heads must be divisible by kv_heads"):
        qsa.Caps(**values)


@pytest.mark.parametrize(
    ("q_heads", "kv_heads"),
    ((24, 1), (48, 3), (64, 4)),
)
def test_qsa_plan_hydrates_all_native_divisible_head_layouts(
    q_heads: int,
    kv_heads: int,
) -> None:
    caps = qsa.Caps(
        device="cuda:0",
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_seq_len=32,
        num_main_cache_pages=2,
        num_compressed_cache_pages=2,
        main_page_size=16,
        compressed_page_size=4,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=256,
    )
    declaration = qsa.plan(caps)
    declaration.contract.validate_query(declaration.query, None)

    hydrated = qsa_contract._caps_from_query(declaration.query, ordinal=0)
    assert (hydrated.q_heads, hydrated.kv_heads) == (q_heads, kv_heads)

@pytest.mark.parametrize("page_size", [16, 1504, 3008])
def test_qsa_caps_accepts_runtime_qwen_page_sizes(
    page_size: int,
) -> None:
    caps = qsa.Caps(
        device="cuda:0",
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_seq_len=page_size,
        num_main_cache_pages=1,
        num_compressed_cache_pages=1,
        main_page_size=page_size,
        compressed_page_size=page_size // 4,
        q_heads=12,
        kv_heads=1,
        head_dim=256,
    )
    assert caps.main_page_size == page_size


def _dense_gathered_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    rows, q_heads, head_dim = map(int, query.shape)
    page_size = int(key_cache.shape[1])
    kv_heads = int(key_cache.shape[2])
    heads_per_kv = q_heads // kv_heads
    result = torch.zeros_like(query)
    for row in range(rows):
        request_id = int(request_ids[row].item())
        query_position = int(query_positions[row].item())
        if request_id < 0 or request_id >= int(block_table.shape[0]):
            continue
        logical_positions = [
            int(position)
            for position in selected_positions[row].detach().cpu().tolist()
            if 0 <= int(position) <= query_position
        ]
        for query_head in range(q_heads):
            kv_head = query_head // heads_per_kv
            keys = []
            values = []
            for logical_position in logical_positions:
                logical_page = logical_position // page_size
                if logical_page >= int(block_table.shape[1]):
                    continue
                physical_page = int(block_table[request_id, logical_page].item())
                if physical_page < 0 or physical_page >= int(key_cache.shape[0]):
                    continue
                page_offset = logical_position % page_size
                keys.append(key_cache[physical_page, page_offset, kv_head])
                values.append(value_cache[physical_page, page_offset, kv_head])
            if not keys:
                continue
            gathered_key = torch.stack(keys).float()
            gathered_value = torch.stack(values).float()
            scores = (query[row, query_head].float() @ gathered_key.T) * softmax_scale
            result[row, query_head] = (
                torch.softmax(scores, dim=-1) @ gathered_value
            ).to(torch.bfloat16)
    return result


def _cache_layout(
    *,
    pages: int,
    page_size: int,
    kv_heads: int,
    head_dim: int,
    layout: str,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    def make() -> torch.Tensor:
        if layout == "contiguous":
            return torch.randn(
                (pages, page_size, kv_heads, head_dim),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=device, dtype=torch.bfloat16)
        if layout == "page_transposed":
            storage = torch.randn(
                (page_size, pages, kv_heads, head_dim),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=device, dtype=torch.bfloat16)
            return storage.permute(1, 0, 2, 3)
        if layout == "padded_inner":
            storage = torch.randn(
                (pages, page_size, kv_heads, head_dim + 8),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=device, dtype=torch.bfloat16)
            return storage[..., :head_dim]
        if layout == "interleaved_page":
            storage = torch.randn(
                (pages, 3, page_size, kv_heads, head_dim),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=device, dtype=torch.bfloat16)
            return storage[:, 1]
        raise AssertionError(f"unknown cache layout {layout}")

    return make(), make()


class _CandidateTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        strides: tuple[int, ...] | None = None,
        contiguous: bool = True,
    ) -> None:
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device("cuda", 0)
        self.is_cuda = True
        self._contiguous = contiguous
        if strides is None:
            running = 1
            reversed_strides = []
            for extent in reversed(shape):
                reversed_strides.append(running)
                running *= extent
            strides = tuple(reversed(reversed_strides))
        self._strides = strides

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        if dim is None:
            return self._strides
        return self._strides[dim]


@pytest.mark.parametrize(
    ("rows", "q_heads", "kv_heads", "expected"),
    [
        (2, 6, 1, True),
        (8, 6, 1, True),
        (9, 6, 1, True),
        (4, 12, 1, True),
        (5, 12, 1, True),
        (2, 24, 2, True),
        (4, 24, 2, True),
    ],
)
@pytest.mark.parametrize("splits", [1, 8, 64])
def test_cute_candidate_accepts_all_qwen_rows_for_interleaved_blh_cache_views(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    q_heads: int,
    kv_heads: int,
    expected: bool,
    splits: int,
) -> None:
    pages, layers, page_size = 128, 64, 1504
    allocation = torch.empty(
        (
            pages,
            layers,
            2,
            page_size,
            kv_heads * cute_config.HEAD_DIM,
        ),
        dtype=torch.bfloat16,
        device="meta",
    )
    layer_cache = allocation[:, layers // 2]
    key_view, value_view = (
        side.unflatten(-1, (kv_heads, cute_config.HEAD_DIM))
        for side in layer_cache.unbind(1)
    )
    assert not key_view.is_contiguous()
    assert not value_view.is_contiguous()
    query = _CandidateTensor((rows, q_heads, cute_config.HEAD_DIM), torch.bfloat16)
    key_cache = _CandidateTensor(
        (pages, page_size, kv_heads, cute_config.HEAD_DIM),
        torch.bfloat16,
        strides=key_view.stride(),
        contiguous=False,
    )
    value_cache = _CandidateTensor(
        (pages, page_size, kv_heads, cute_config.HEAD_DIM),
        torch.bfloat16,
        strides=value_view.stride(),
        contiguous=False,
    )

    assert (
        cute_config.is_candidate(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=_CandidateTensor((4, 128), torch.int32),
            request_ids=_CandidateTensor((rows,), torch.int64),
            selected_positions=_CandidateTensor(
                (rows, cute_config.SELECTION_WIDTH), torch.int32
            ),
            query_positions=_CandidateTensor((rows,), torch.int64),
            partial_output=_CandidateTensor(
                (
                    rows,
                    splits,
                    q_heads,
                    cute_config.HEAD_DIM,
                ),
                torch.float32,
            ),
            partial_lse=_CandidateTensor((rows, splits, q_heads), torch.float32),
            block_n=cute_config.BLOCK_N,
            splits=splits,
        )
        is expected
    )


def test_qwen_split_policy_uses_unsplit_large_rows() -> None:
    caps = SimpleNamespace(
        q_heads=12,
        kv_heads=1,
        head_dim=256,
        main_page_size=16,
        selection_width=2051,
    )

    assert _target_splits(caps, 1) == (16, 64)
    assert _target_splits(caps, 4) == (16, 32)
    assert _target_splits(caps, 32) == (16, 16)
    assert _target_splits(caps, 64) == (16, 16)
    assert _target_splits(caps, 65) == (16, 1)

    with pytest.raises(NotImplementedError, match="requires the CuTe Qwen"):
        _target_splits(
            SimpleNamespace(
                q_heads=8,
                kv_heads=3,
                head_dim=64,
                main_page_size=4,
                selection_width=67,
            ),
            3,
        )


def test_qwen_geometry_rejects_invalid_cute_contract_instead_of_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, q_heads, kv_heads = 5, 12, 1
    query = torch.empty((rows, q_heads, 256), device="meta")
    key_cache = torch.empty((1, 16, kv_heads, 256), device="meta")
    value_cache = torch.empty_like(key_cache)
    block_table = torch.empty((1, 1), dtype=torch.int32, device="meta")
    request_ids = torch.empty((rows,), dtype=torch.int64, device="meta")
    selected_positions = torch.empty((rows, 2051), dtype=torch.int32, device="meta")
    query_positions = torch.empty((rows,), dtype=torch.int64, device="meta")
    output = torch.empty_like(query)
    partial_output = torch.empty(
        (rows, 64, q_heads, 256), dtype=torch.float32, device="meta"
    )
    partial_lse = torch.empty((rows, 64, q_heads), dtype=torch.float32, device="meta")
    monkeypatch.setattr(
        sparse_gqa, "_validate_launch", lambda **_kwargs: (rows, q_heads, 256)
    )
    monkeypatch.setattr(sparse_gqa, "_cute_is_candidate", lambda **_kwargs: False)

    with pytest.raises(RuntimeError, match="requires its CuTe layout"):
        launch_sparse_paged_gqa(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            request_ids=request_ids,
            selected_positions=selected_positions,
            query_positions=query_positions,
            output=output,
            output_lse=torch.empty((rows, q_heads), dtype=torch.float32, device="meta"),
            partial_output=partial_output,
            partial_lse=partial_lse,
            softmax_scale=1.0 / 16.0,
            block_n=16,
            splits=64,
        )


def test_non_qwen_geometry_has_no_sparse_gqa_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, q_heads, kv_heads, head_dim = 3, 8, 2, 64
    query = torch.empty((rows, q_heads, head_dim), device="meta")
    key_cache = torch.empty((1, 4, kv_heads, head_dim), device="meta")
    value_cache = torch.empty_like(key_cache)
    selected_positions = torch.empty((rows, 67), dtype=torch.int32, device="meta")
    monkeypatch.setattr(
        sparse_gqa, "_validate_launch", lambda **_kwargs: (rows, q_heads, head_dim)
    )
    monkeypatch.setattr(
        sparse_gqa,
        "_cute_is_candidate",
        lambda **_kwargs: pytest.fail("non-Qwen geometry reached CuTe candidate"),
    )

    with pytest.raises(NotImplementedError, match="no CuTe implementation"):
        launch_sparse_paged_gqa(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=torch.empty((1, 1), dtype=torch.int32, device="meta"),
            request_ids=torch.empty((rows,), dtype=torch.int64, device="meta"),
            selected_positions=selected_positions,
            query_positions=torch.empty((rows,), dtype=torch.int64, device="meta"),
            output=torch.empty_like(query),
            output_lse=torch.empty(
                (rows, q_heads), dtype=torch.float32, device="meta"
            ),
            partial_output=torch.empty(
                (rows, 4, q_heads, head_dim), dtype=torch.float32, device="meta"
            ),
            partial_lse=torch.empty(
                (rows, 4, q_heads), dtype=torch.float32, device="meta"
            ),
            softmax_scale=1.0 / 8.0,
            block_n=16,
            splits=4,
        )


def test_large_prefill_rejects_layouts_not_supported_by_selected_abi() -> None:
    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 65, 6, 1, 256
    query = torch.empty(
        (q_heads, rows, head_dim), dtype=torch.bfloat16, device=device
    ).transpose(0, 1)
    key_cache = torch.empty(
        (1, 16, kv_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    value_cache = torch.empty_like(key_cache)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    request_ids = torch.zeros((rows,), dtype=torch.int64, device=device)
    selected_positions = torch.full((rows, 2051), -1, dtype=torch.int32, device=device)
    query_positions = torch.zeros((rows,), dtype=torch.int64, device=device)
    output = torch.empty(
        (q_heads, rows, head_dim), dtype=torch.bfloat16, device=device
    ).transpose(0, 1)
    launch_kwargs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "request_ids": request_ids,
        "selected_positions": selected_positions,
        "query_positions": query_positions,
        "output_lse": torch.empty(
            (rows, q_heads), dtype=torch.float32, device=device
        ),
        "partial_output": None,
        "partial_lse": None,
        "softmax_scale": 1.0 / math.sqrt(head_dim),
        "block_n": 16,
        "splits": 1,
    }

    with pytest.raises(ValueError, match="query must be contiguous"):
        launch_sparse_paged_gqa(
            query=query,
            output=query.contiguous(),
            **launch_kwargs,
        )

    with pytest.raises(ValueError, match="output must be contiguous"):
        launch_sparse_paged_gqa(
            query=query.contiguous(),
            output=output,
            **launch_kwargs,
        )

    overlapping_cache = torch.empty(
        16 + head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).as_strided((1, 16, kv_heads, head_dim), (16, 1, head_dim, 1))
    launch_kwargs["key_cache"] = overlapping_cache
    launch_kwargs["value_cache"] = overlapping_cache
    with pytest.raises(ValueError, match="must have non-overlapping rows"):
        launch_sparse_paged_gqa(
            query=query.contiguous(),
            output=query.contiguous(),
            **launch_kwargs,
        )


@pytest.mark.parametrize(
    (
        "rows",
        "q_heads",
        "kv_heads",
        "head_dim",
        "page_size",
        "selection_width",
        "block_n",
        "splits",
        "layout",
    ),
    [
        (1, 24, 2, 256, 16, 2051, 16, 64, "contiguous"),
        (1, 8, 2, 256, 16, 2051, 16, 64, "contiguous"),
        (1, 6, 1, 256, 16, 2051, 16, 64, "interleaved_page"),
        (5, 12, 1, 256, 16, 2051, 16, 64, "interleaved_page"),
    ],
)
def test_sparse_gqa_matches_gathered_dense_reference(
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    selection_width: int,
    block_n: int,
    splits: int,
    layout: str,
) -> None:
    device = require_sm120()
    generator = torch.Generator(device="cpu").manual_seed(
        92000 + rows + q_heads + head_dim
    )
    batches = 3
    table_width = 6
    pages = batches * table_width
    key_cache, value_cache = _cache_layout(
        pages=pages,
        page_size=page_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        layout=layout,
        device=device,
        generator=generator,
    )
    block_table = torch.stack(
        [
            torch.randperm(pages, generator=generator, dtype=torch.int64)[:table_width]
            for _ in range(batches)
        ]
    ).to(device=device, dtype=torch.int32)
    query = torch.randn(
        (rows, q_heads, head_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=torch.bfloat16)
    request_ids = (
        torch.arange(rows, dtype=torch.int64, device=device) % batches
    ).contiguous()
    query_positions = torch.tensor(
        [table_width * page_size - 2 - row for row in range(rows)],
        dtype=torch.int64,
        device=device,
    )
    selected_positions = torch.full(
        (rows + 2, selection_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    logical_capacity = table_width * page_size
    for row in range(rows):
        candidates = torch.randperm(
            logical_capacity, generator=generator, dtype=torch.int64
        )
        count = min(selection_width, logical_capacity)
        selected_positions[row, :count].copy_(
            candidates[:count].to(device=device, dtype=torch.int32)
        )

    output = torch.empty(
        (rows + 2, q_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    output_lse = torch.empty(
        (rows + 2, q_heads), dtype=torch.float32, device=device
    )
    partial_output = (
        torch.empty(
            (rows, splits, q_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
        if splits > 1
        else None
    )
    partial_lse = (
        torch.empty((rows, splits, q_heads), dtype=torch.float32, device=device)
        if splits > 1
        else None
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)
    actual = launch_sparse_paged_gqa(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        output_lse=output_lse,
        partial_output=partial_output,
        partial_lse=partial_lse,
        softmax_scale=softmax_scale,
        block_n=block_n,
        splits=splits,
    )
    expected = _dense_gathered_reference(
        query,
        key_cache,
        value_cache,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        softmax_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-2)


@pytest.mark.parametrize("page_size", [16, 3008])
@pytest.mark.parametrize(
    "kv_dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8_e4m3"],
)
def test_large_prefill_sparse_gqa_addresses_paged_cache(
    page_size: int,
    kv_dtype: torch.dtype,
) -> None:
    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 65, 24, 2, 256
    pages, batches = 4, 2
    storage = torch.randn(
        (pages, 2, page_size, kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    ).to(kv_dtype)
    key_cache, value_cache = storage.unbind(1)
    block_table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32, device=device)
    request_ids = torch.arange(rows, dtype=torch.int64, device=device) % batches
    request_ids[-1] = -1
    query_positions = torch.zeros(rows, dtype=torch.int64, device=device)
    selected_positions = torch.full((rows, 2051), -1, dtype=torch.int32, device=device)
    selected_positions[:, 0] = 0
    query = torch.randn((rows, q_heads, head_dim), dtype=torch.bfloat16, device=device)
    output = torch.full_like(query, float("nan"))
    output_lse = torch.empty((rows, q_heads), dtype=torch.float32, device=device)
    k_descale = None
    v_descale = None
    value_scale = 1.0
    if kv_dtype == torch.float8_e4m3fn:
        k_descale = torch.tensor([0.125], dtype=torch.float32, device=device)
        v_descale = torch.tensor([0.125], dtype=torch.float32, device=device)
        value_scale = float(v_descale)

    actual = launch_sparse_paged_gqa(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        output_lse=output_lse,
        partial_output=None,
        partial_lse=None,
        softmax_scale=1.0 / math.sqrt(head_dim),
        block_n=16,
        splits=1,
    )

    expected = torch.zeros_like(query)
    heads_per_kv = q_heads // kv_heads
    for row in range(rows - 1):
        request = int(request_ids[row])
        physical_page = int(block_table[request, 0])
        expected[row].copy_(
            (value_cache[physical_page, 0].float() * value_scale)
            .repeat_interleave(heads_per_kv, dim=0)
            .to(torch.bfloat16)
        )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-3)


@pytest.mark.parametrize(
    "kv_dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8_e4m3"],
)
@pytest.mark.parametrize("direct_kv_warps", [1, 2, 4])
def test_large_prefill_selected_paged_gqa_matches_split_for_full_selection(
    kv_dtype: torch.dtype,
    direct_kv_warps: int,
) -> None:
    from b12x.attention.qsa._sparse_gqa_cute import (
        launch_selected_paged_gqa_direct,
        launch_sparse_gqa_merge,
        launch_sparse_gqa_split,
    )

    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 65, 24, 2, 256
    page_size, pages, selection_width = 3008, 4, 2051
    generator = torch.Generator(device="cpu").manual_seed(93865)
    source = torch.randn(
        (pages, 2, page_size, kv_heads, head_dim),
        generator=generator,
        dtype=torch.bfloat16,
        device="cpu",
    ).to(device)
    k_descale = None
    v_descale = None
    if kv_dtype == torch.float8_e4m3fn:
        k_descale = torch.tensor([0.125], dtype=torch.float32, device=device)
        v_descale = torch.tensor([0.125], dtype=torch.float32, device=device)
        storage = torch.empty_like(source, dtype=kv_dtype)
        storage[:, 0].copy_((source[:, 0].float() / k_descale).to(kv_dtype))
        storage[:, 1].copy_((source[:, 1].float() / v_descale).to(kv_dtype))
    else:
        storage = source
    key_cache, value_cache = storage.unbind(1)
    block_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, -1)
    request_ids = torch.zeros(rows, dtype=torch.int64, device=device)
    query_positions = torch.full(
        (rows,), pages * page_size - 1, dtype=torch.int64, device=device
    )
    selected = torch.randperm(
        pages * page_size, generator=generator, dtype=torch.int64
    )[:selection_width]
    selected_positions = (
        selected.to(device=device, dtype=torch.int32)
        .view(1, -1)
        .expand(rows, -1)
        .contiguous()
    )
    query = torch.randn(
        (rows, q_heads, head_dim),
        generator=generator,
        dtype=torch.bfloat16,
        device="cpu",
    ).to(device)
    direct_output = torch.empty_like(query)
    cute_output = torch.empty_like(query)
    direct_lse = torch.empty((rows, q_heads), dtype=torch.float32, device=device)
    cute_lse = torch.empty_like(direct_lse)
    split_output = torch.empty(
        (rows, 16, q_heads, head_dim), dtype=torch.float32, device=device
    )
    split_lse = torch.empty((rows, 16, q_heads), dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    launch_selected_paged_gqa_direct(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=direct_output,
        output_lse=direct_lse,
        softmax_scale=scale,
        kv_warps=direct_kv_warps,
    )
    launch_sparse_gqa_split(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        partial_output=split_output,
        partial_lse=split_lse,
        softmax_scale=scale,
        splits=16,
    )
    launch_sparse_gqa_merge(
        partial_output=split_output,
        partial_lse=split_lse,
        output=cute_output,
        output_lse=cute_lse,
        rows=rows,
        splits=16,
    )
    torch.testing.assert_close(direct_output, cute_output, rtol=0.0, atol=3e-2)
    torch.testing.assert_close(direct_lse, cute_lse, rtol=0.0, atol=2e-3)


def test_sparse_gqa_matches_reference_for_1504_token_pages() -> None:
    device = require_sm120()
    generator = torch.Generator(device="cpu").manual_seed(93504)
    rows, q_heads, kv_heads, head_dim = 1, 12, 1, 256
    page_size, selection_width, splits = 1504, 2051, 64
    pages, table_width = 4, 3
    key_cache, value_cache = _cache_layout(
        pages=pages,
        page_size=page_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        layout="contiguous",
        device=device,
        generator=generator,
    )
    block_table = torch.tensor([[2, 0, 3]], dtype=torch.int32, device=device)
    query = torch.randn(
        (rows, q_heads, head_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=torch.bfloat16)
    request_ids = torch.zeros((rows,), dtype=torch.int64, device=device)
    query_positions = torch.tensor(
        [table_width * page_size - 1], dtype=torch.int64, device=device
    )
    selected_positions = torch.full(
        (rows, selection_width), -1, dtype=torch.int32, device=device
    )
    selected_positions[0, :6] = torch.tensor(
        [
            0,
            page_size - 1,
            page_size,
            2 * page_size - 1,
            2 * page_size,
            3 * page_size - 1,
        ],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty_like(query)
    output_lse = torch.empty((rows, q_heads), dtype=torch.float32, device=device)
    partial_output = torch.empty(
        (rows, splits, q_heads, head_dim), dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        (rows, splits, q_heads), dtype=torch.float32, device=device
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)
    actual = launch_sparse_paged_gqa(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        output_lse=output_lse,
        partial_output=partial_output,
        partial_lse=partial_lse,
        softmax_scale=softmax_scale,
        block_n=16,
        splits=splits,
    )
    expected = _dense_gathered_reference(
        query,
        key_cache,
        value_cache,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        softmax_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-2)


def test_sparse_gqa_fp8_3008_page_matches_reference_and_graph_replay() -> None:
    device = require_sm120()
    generator = torch.Generator(device="cpu").manual_seed(93803)
    rows, q_heads, kv_heads, head_dim = 1, 12, 1, 256
    page_size, selection_width, splits = 3008, 2051, 64
    pages = 6
    key_source, value_source = _cache_layout(
        pages=pages,
        page_size=page_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        layout="interleaved_page",
        device=device,
        generator=generator,
    )
    k_descale = torch.tensor([0.0125], dtype=torch.float32, device=device)
    v_descale = torch.tensor([0.01], dtype=torch.float32, device=device)
    key_cache = (key_source.float() / k_descale).to(torch.float8_e4m3fn)
    value_cache = (value_source.float() / v_descale).to(torch.float8_e4m3fn)
    block_table = torch.tensor([[4, 1, 5, 2]], dtype=torch.int32, device=device)
    query = torch.randn(
        (rows, q_heads, head_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=torch.bfloat16)
    request_ids = torch.zeros((rows,), dtype=torch.int64, device=device)
    query_positions = torch.tensor(
        [4 * page_size - 1], dtype=torch.int64, device=device
    )
    selected_positions = torch.full(
        (rows, selection_width), -1, dtype=torch.int32, device=device
    )
    selected_positions[0, :64] = torch.tensor(
        [
            0,
            page_size - 1,
            page_size,
            2 * page_size - 1,
            2 * page_size,
            3 * page_size - 1,
            *range(1, 59),
        ],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty_like(query)
    output_lse = torch.empty((rows, q_heads), dtype=torch.float32, device=device)
    partial_output = torch.empty(
        (rows, splits, q_heads, head_dim), dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        (rows, splits, q_heads), dtype=torch.float32, device=device
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)

    def launch() -> torch.Tensor:
        return launch_sparse_paged_gqa(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=block_table,
            request_ids=request_ids,
            selected_positions=selected_positions,
            query_positions=query_positions,
            output=output,
            output_lse=output_lse,
            partial_output=partial_output,
            partial_lse=partial_lse,
            softmax_scale=softmax_scale,
            block_n=16,
            splits=splits,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = launch()
    query.copy_(torch.randn_like(query))
    k_descale.fill_(0.015625)
    v_descale.fill_(0.0078125)
    expected = _dense_gathered_reference(
        query,
        key_cache.float() * k_descale,
        value_cache.float() * v_descale,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        softmax_scale,
    )
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == allocated_before
    assert captured_output.data_ptr() == output.data_ptr()
    torch.testing.assert_close(captured_output, expected, rtol=0.0, atol=3e-2)


def test_sparse_gqa_zeroes_padded_request_and_all_masked_rows() -> None:
    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 2, 24, 2, 256
    page_size, selection_width, splits = 16, 2051, 64
    query = torch.randn((rows, q_heads, head_dim), dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(
        (2, page_size, kv_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    request_ids = torch.tensor([-1, 0], dtype=torch.int64, device=device)
    query_positions = torch.tensor([7, 0], dtype=torch.int64, device=device)
    selected_positions = torch.full(
        (rows, selection_width), -1, dtype=torch.int32, device=device
    )
    selected_positions[1, :2] = torch.tensor([2, 3], dtype=torch.int32, device=device)
    output = torch.full(
        (rows, q_heads, head_dim),
        17,
        dtype=torch.bfloat16,
        device=device,
    )
    output_lse = torch.empty((rows, q_heads), dtype=torch.float32, device=device)
    partial_output = torch.empty(
        (rows, splits, q_heads, head_dim), dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        (rows, splits, q_heads), dtype=torch.float32, device=device
    )
    actual = launch_sparse_paged_gqa(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        output_lse=output_lse,
        partial_output=partial_output,
        partial_lse=partial_lse,
        softmax_scale=1.0 / math.sqrt(head_dim),
        block_n=16,
        splits=splits,
    )
    assert torch.count_nonzero(actual).item() == 0


def test_sparse_gqa_split_path_is_cuda_graph_replay_safe() -> None:
    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 1, 24, 2, 256
    page_size, selection_width, splits = 16, 2051, 64
    query = torch.randn((rows, q_heads, head_dim), dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(
        (8, page_size, kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(8, dtype=torch.int32, device=device).view(1, 8)
    request_ids = torch.zeros((rows,), dtype=torch.int64, device=device)
    query_positions = torch.tensor([95], dtype=torch.int64, device=device)
    selected_positions = torch.full(
        (rows, selection_width), -1, dtype=torch.int32, device=device
    )
    selected_positions[0, :96] = torch.randperm(
        96, dtype=torch.int64, device=device
    ).to(torch.int32)
    output = torch.empty_like(query)
    output_lse = torch.empty((rows, q_heads), dtype=torch.float32, device=device)
    partial_output = torch.empty(
        (rows, splits, q_heads, head_dim), dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        (rows, splits, q_heads), dtype=torch.float32, device=device
    )
    scale = 1.0 / math.sqrt(head_dim)

    launch_sparse_paged_gqa(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        output_lse=output_lse,
        partial_output=partial_output,
        partial_lse=partial_lse,
        softmax_scale=scale,
        block_n=16,
        splits=splits,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = launch_sparse_paged_gqa(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            request_ids=request_ids,
            selected_positions=selected_positions,
            query_positions=query_positions,
            output=output,
            output_lse=output_lse,
            partial_output=partial_output,
            partial_lse=partial_lse,
            softmax_scale=scale,
            block_n=16,
            splits=splits,
        )

    query.copy_(torch.randn_like(query))
    expected = _dense_gathered_reference(
        query,
        key_cache,
        value_cache,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        scale,
    )
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_output.data_ptr() == output.data_ptr()
    torch.testing.assert_close(captured_output, expected, rtol=0.0, atol=2e-2)


@pytest.mark.parametrize("request_id_dtype", [torch.int32, torch.int64])
def test_selected_paged_gqa_direct_path_is_cuda_graph_replay_safe(
    request_id_dtype: torch.dtype,
) -> None:
    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 65, 24, 2, 256
    page_size = 16
    query = torch.randn((rows, q_heads, head_dim), dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(
        (1, page_size, kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    request_ids = torch.zeros((rows,), dtype=request_id_dtype, device=device)
    query_positions = torch.zeros((rows,), dtype=torch.int64, device=device)
    selected_positions = torch.full((rows, 2051), -1, dtype=torch.int32, device=device)
    selected_positions[:, 0] = 0
    output = torch.empty_like(query)
    output_lse = torch.empty((rows, q_heads), dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    launch_sparse_paged_gqa(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        output_lse=output_lse,
        partial_output=None,
        partial_lse=None,
        softmax_scale=scale,
        block_n=16,
        splits=1,
        direct_kv_warps=4,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = launch_sparse_paged_gqa(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            request_ids=request_ids,
            selected_positions=selected_positions,
            query_positions=query_positions,
            output=output,
            output_lse=output_lse,
            partial_output=None,
            partial_lse=None,
            softmax_scale=scale,
            block_n=16,
            splits=1,
            direct_kv_warps=4,
        )

    value_cache.copy_(torch.randn_like(value_cache))
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)
    expected = (
        value_cache[0, 0]
        .repeat_interleave(q_heads // kv_heads, dim=0)
        .unsqueeze(0)
        .expand(rows, -1, -1)
    )
    assert allocated_after == allocated_before
    assert captured_output.data_ptr() == output.data_ptr()
    torch.testing.assert_close(captured_output, expected, rtol=0.0, atol=0.0)


def test_sparse_gqa_reuses_direct_binary_across_runtime_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.paged import _selected_forward as selected_impl

    device = require_sm120()
    q_heads, kv_heads, head_dim = 6, 1, 256
    compile_targets: list[str] = []
    original_compile = selected_impl.b12x_compile

    def traced_compile(target: object, *args: object, **kwargs: object) -> object:
        compile_targets.append(type(target).__name__)
        return original_compile(target, *args, **kwargs)

    def launch(rows: int) -> torch.Tensor:
        page_size = 16
        query = torch.randn(
            (rows, q_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        key_cache = torch.empty(
            (1, page_size, kv_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        value_cache = torch.empty_like(key_cache)
        block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        request_ids = torch.zeros((rows,), dtype=torch.int64, device=device)
        selected_positions = torch.full(
            (rows, 2051), -1, dtype=torch.int32, device=device
        )
        query_positions = torch.zeros((rows,), dtype=torch.int64, device=device)
        output = torch.empty_like(query)
        output_lse = torch.empty((rows, q_heads), dtype=torch.float32, device=device)
        return launch_sparse_paged_gqa(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            request_ids=request_ids,
            selected_positions=selected_positions,
            query_positions=query_positions,
            output=output,
            output_lse=output_lse,
            partial_output=None,
            partial_lse=None,
            softmax_scale=1.0 / math.sqrt(head_dim),
            block_n=16,
            splits=1,
            direct_kv_warps=4,
        )

    selected_impl.clear_caches()
    monkeypatch.setattr(selected_impl, "b12x_compile", traced_compile)
    try:
        assert torch.count_nonzero(launch(65)).item() == 0
        compiled_after_first_launch = tuple(compile_targets)
        assert (
            compiled_after_first_launch.count("_SelectedPositionPagedForwardKernel")
            == 1
        )

        with kernel_resolution_guard("QSA runtime-row cache reuse test"):
            for rows in (128, 1_024):
                assert torch.count_nonzero(launch(rows)).item() == 0
                assert tuple(compile_targets) == compiled_after_first_launch
    finally:
        selected_impl.clear_caches()


def test_large_prefill_direct_path_does_not_touch_split_scratch() -> None:
    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 65, 6, 1, 256
    query = torch.randn((rows, q_heads, head_dim), dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(
        (1, 16, kv_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    request_ids = torch.zeros((rows,), dtype=torch.int64, device=device)
    query_positions = torch.zeros((rows,), dtype=torch.int64, device=device)
    selected_positions = torch.full((rows, 2051), -1, dtype=torch.int32, device=device)
    selected_positions[:, 0] = 0
    output = torch.empty_like(query)
    output_lse = torch.empty((rows, q_heads), dtype=torch.float32, device=device)
    partial_output = torch.full(
        (rows, 64, q_heads, head_dim),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    partial_lse = torch.full(
        (rows, 64, q_heads),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    partial_output_before = partial_output.clone()
    partial_lse_before = partial_lse.clone()

    launch_sparse_paged_gqa(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        output_lse=output_lse,
        partial_output=partial_output,
        partial_lse=partial_lse,
        softmax_scale=1.0 / math.sqrt(head_dim),
        block_n=16,
        splits=1,
        direct_kv_warps=2,
    )
    torch.cuda.synchronize(device)

    torch.testing.assert_close(
        partial_output, partial_output_before, rtol=0.0, atol=0.0, equal_nan=True
    )
    torch.testing.assert_close(
        partial_lse, partial_lse_before, rtol=0.0, atol=0.0, equal_nan=True
    )


@pytest.mark.parametrize(
    "kv_dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8_e4m3"],
)
def test_sparse_gqa_uses_int64_for_high_physical_page_offsets(
    kv_dtype: torch.dtype,
) -> None:
    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 65, 24, 2, 256
    page_size = 16
    page_stride_elements = page_size * kv_heads * head_dim
    tail_page = math.ceil((1 << 31) / page_stride_elements)
    num_pages = tail_page + 1
    required_bytes = num_pages * page_stride_elements * kv_dtype.itemsize
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserve_bytes = 2 * 1024**3
    if free_bytes < required_bytes + reserve_bytes:
        pytest.skip(
            "high-page-id live allocation requires "
            f"{required_bytes + reserve_bytes} bytes free, found {free_bytes}"
        )
    try:
        cache = torch.empty(
            (num_pages, page_size, kv_heads, head_dim),
            dtype=kv_dtype,
            device=device,
        )
    except torch.OutOfMemoryError:
        pytest.skip(
            "CUDA allocator could not reserve the required mostly-uninitialized "
            f"{required_bytes}-byte high-page-id cache"
        )

    source_value = (
        torch.linspace(
            -1.0,
            1.0,
            kv_heads * head_dim,
            dtype=torch.float32,
            device=device,
        )
        .view(kv_heads, head_dim)
        .to(torch.bfloat16)
    )
    k_descale = None
    v_descale = None
    if kv_dtype == torch.float8_e4m3fn:
        k_descale = torch.tensor([0.015625], dtype=torch.float32, device=device)
        v_descale = torch.tensor([0.015625], dtype=torch.float32, device=device)
        live_value = (source_value.float() / v_descale).to(kv_dtype)
        expected_value = (live_value.float() * v_descale).to(torch.bfloat16)
    else:
        live_value = source_value
        expected_value = live_value
    cache[tail_page, 0].copy_(live_value)
    query = torch.randn((rows, q_heads, head_dim), dtype=torch.bfloat16, device=device)
    block_table = torch.tensor([[tail_page]], dtype=torch.int32, device=device)
    request_ids = torch.zeros((rows,), dtype=torch.int64, device=device)
    query_positions = torch.zeros((rows,), dtype=torch.int64, device=device)
    selected_positions = torch.full((rows, 2051), -1, dtype=torch.int32, device=device)
    selected_positions[:, 0] = 0
    output = torch.empty_like(query)
    output_lse = torch.empty((rows, q_heads), dtype=torch.float32, device=device)
    actual = launch_sparse_paged_gqa(
        query=query,
        key_cache=cache,
        value_cache=cache,
        k_descale=k_descale,
        v_descale=v_descale,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        output_lse=output_lse,
        partial_output=None,
        partial_lse=None,
        softmax_scale=1.0 / math.sqrt(head_dim),
        block_n=16,
        splits=1,
        direct_kv_warps=4,
    )
    expected = (
        torch.stack(
            [expected_value[head // (q_heads // kv_heads)] for head in range(q_heads)]
        )
        .unsqueeze(0)
        .expand(rows, -1, -1)
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
