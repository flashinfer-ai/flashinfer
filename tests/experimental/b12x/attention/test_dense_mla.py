"""Native dense MLA correctness, serving, and addressing gates."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import pytest
import torch

from b12x.attention import dense_mla

from ..conftest import require_b12x

FP8 = torch.float8_e4m3fn
HEADS = 8
QK_DIM = 576
VALUE_DIM = 512


_PREPARED: ContextVar[
    list[tuple[object, object]] | None
] = ContextVar("dense_mla_prepared", default=None)


@contextmanager
def _prepared_scope():
    """Scope prepared owners to the current numerical test."""
    prepared: list[tuple[object, object]] = []
    token = _PREPARED.set(prepared)
    try:
        yield
    finally:
        _PREPARED.reset(token)
        error = None
        for result, session in reversed(prepared):
            for close in (result.close, session.close):
                try:
                    close()
                except BaseException as caught:
                    if error is None:
                        error = caught
        if error is not None:
            raise error


@pytest.fixture(autouse=True)
def _prepared_dense_mla():
    with _prepared_scope():
        yield

def _scratch(spec: object) -> torch.Tensor:
    """Allocate caller-owned scratch from session-provided metadata."""
    return torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)


def _prepare(
    declaration: dense_mla.Plan,
    *,
    name: str,
    **binding_kwargs,
):
    """Admit and prime a declaration through the public preparation lifecycle."""
    from b12x.preparation import PreparationSession, PreparedCall

    scratch_spec = None

    def prepare_call(state):
        nonlocal scratch_spec
        (scratch_spec,) = state.scratch_specs()
        scratch = _scratch(scratch_spec)
        binding = state.bind(scratch=scratch, **binding_kwargs)
        state.prime(binding)
        return PreparedCall(
            run=lambda: state.run(binding),
            output=binding_kwargs["output"],
            owners=(scratch, binding),
        )

    session = PreparationSession(
        device=binding_kwargs["q"].device,
        autotune=False,
    )
    try:
        result = session.prepare((
            declaration.request(
                name=name,
                prepare_call=prepare_call,
            ),
        ))
    except BaseException:
        session.close()
        raise
    prepared = _PREPARED.get()
    if prepared is None:
        result.close()
        session.close()
        raise RuntimeError("dense MLA test preparation requires _prepared_scope")
    prepared.append((result, session))
    if scratch_spec is None:
        # A declaration prepared earlier in this scope is not primed again;
        # later bindings reuse its session-planned scratch contract.
        assert declaration.prepared is not None
        (scratch_spec,) = declaration.scratch_specs()
    return declaration, scratch_spec


def _bind(declaration: dense_mla.Plan, **binding_kwargs) -> dense_mla.Binding:
    """Prepare an independent owner, then bind the serving invocation to it."""
    plan, scratch_spec = _prepare(
        declaration,
        name=f"dense-mla-{binding_kwargs['q'].data_ptr():x}",
        **binding_kwargs,
    )
    return dense_mla.bind(
        plan, scratch=_scratch(scratch_spec), **binding_kwargs
    )


def _guarded_scratch(
    spec: object,
    *,
    guard_bytes: int = 16 * 1024 * 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact session-planned scratch surrounded by initialized canaries."""
    assert spec.dtype == torch.uint8
    storage = torch.full(
        (spec.nbytes + 2 * guard_bytes,),
        0xA5,
        dtype=torch.uint8,
        device=spec.device,
    )
    scratch = storage.narrow(0, guard_bytes, spec.nbytes)
    return storage, scratch


def _assert_matches(
    output: torch.Tensor,
    lse: torch.Tensor,
    reference_output: torch.Tensor,
    reference_lse: torch.Tensor,
) -> None:
    torch.cuda.synchronize()
    assert bool(torch.isfinite(output).all().item())
    assert bool(torch.isfinite(lse).all().item())
    assert int(torch.count_nonzero(output).item()) == output.numel()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().reshape(output.shape[0], -1),
        reference_output.float().reshape(output.shape[0], -1),
        dim=1,
    )
    assert float(cosine.min().item()) > 0.999
    torch.testing.assert_close(
        output.float(),
        reference_output.float(),
        rtol=2e-2,
        atol=5e-4,
    )
    torch.testing.assert_close(lse, reference_lse, rtol=2e-5, atol=2e-5)






@torch.inference_mode()
def test_fp8_physical_record_stride_ignores_padding() -> None:
    device = require_b12x()
    torch.manual_seed(20260813)
    rows = 2
    heads = 16
    pages = 3
    page_size = 64
    physical_width = 1088
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=FP8,
            num_q_heads=heads,
            page_size=page_size,
            max_total_q=rows,
            max_batch=rows,
            max_cache_tokens=65,
            max_page_table_width=2,
            num_cache_pages=pages,
            physical_record_width=physical_width,
        )
    )
    q_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    q = (torch.randn(rows, heads, QK_DIM, device=device) * 10).to(FP8)
    cache = torch.empty(
        pages,
        page_size,
        physical_width,
        dtype=FP8,
        device=device,
    )
    cache[..., :QK_DIM] = (
        torch.randn(pages, page_size, QK_DIM, device=device) * 10
    ).to(FP8)
    cache[..., QK_DIM:] = (
        (torch.randn(pages, page_size, physical_width - QK_DIM, device=device) * 100)
        .clamp(-448, 448)
        .to(FP8)
    )
    page_table = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([64, 65], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(rows + 1, dtype=torch.int32, device=device)
    output = torch.empty(rows, heads, VALUE_DIM, dtype=torch.bfloat16, device=device)
    binding = _bind(plan, q=q,
    kv_cache=cache,
    output=output,
    page_table=page_table,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,
    kv_scale=kv_scale,
    q_scale=q_scale,)
    actual, actual_lse = dense_mla.run(binding=binding)
    expected, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
    )
    _assert_matches(actual, actual_lse, expected, expected_lse)




@pytest.mark.parametrize("heads", [8, 12])
@torch.inference_mode()
def test_bf16_multi_request_decode_matches_reference(heads: int) -> None:
    device = require_b12x()
    torch.manual_seed(20260730)
    batch = 4
    page_size = 16
    pages = 24
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=heads,
            page_size=page_size,
            max_total_q=batch,
            max_batch=batch,
            max_cache_tokens=96,
            max_page_table_width=6,
            num_cache_pages=pages,
        )
    )
    q = (torch.randn(batch, heads, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.tensor(
        [
            [4, 7, 1, 5, 0, 3],
            [18, 2, 19, 6, 8, 21],
            [10, 9, 12, 16, 15, 11],
            [23, 17, 20, 14, 13, 22],
        ],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor(
        [1, 17, 63, 91],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.arange(
        batch + 1,
        dtype=torch.int32,
        device=device,
    )
    output = torch.full(
        (batch, heads, VALUE_DIM),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )

    # Compile first through a smaller live batch. The same capacity-planned
    # specialization must then accept the full batch without recompilation or
    # stale tensor-layout assumptions.
    small_output = torch.empty(
        1,
        heads,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    small_binding = _bind(plan, q=q[:1],
    kv_cache=cache,
    output=small_output,
    page_table=page_table[:1],
    cache_seqlens=cache_seqlens[:1],
    cu_seqlens_q=torch.tensor(
        [0, 1],
        dtype=torch.int32,
        device=device,
    ),)
    small_actual, small_lse = dense_mla.run(binding=small_binding)
    small_expected, small_expected_lse = dense_mla.reference(
        small_binding.q,
        small_binding.kv_cache,
        small_binding.page_table,
        small_binding.cache_seqlens,
        small_binding.cu_seqlens_q,
    )
    _assert_matches(
        small_actual,
        small_lse,
        small_expected,
        small_expected_lse,
    )

    binding = _bind(plan, q=q,
    kv_cache=cache,
    output=output,
    page_table=page_table,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,)

    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@pytest.mark.parametrize("heads", [8, 12])
@torch.inference_mode()
def test_fp8_query_tiled_causal_extend_matches_reference(heads: int) -> None:
    device = require_b12x()
    torch.manual_seed(20260731)
    query_rows = 5
    page_size = 16
    pages = 8
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="extend",
            kv_dtype=FP8,
            num_q_heads=heads,
            page_size=page_size,
            max_total_q=query_rows,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=pages,
        )
    )
    q_float = torch.randn(query_rows, heads, QK_DIM, device=device) * 0.14
    cache_float = torch.randn(pages, page_size, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.tensor(
        [[4, 7, 1, 5, 0, 3, 6, 2]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([77], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(
        [0, query_rows],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        query_rows,
        heads,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = _bind(plan, q=q,
    kv_cache=cache,
    output=output,
    page_table=page_table,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,
    q_scale=q_scale,
    kv_scale=kv_scale,)
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_bf16_query_tiled_causal_extend_matches_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260732)
    query_rows = 7
    page_size = 16
    pages = 8
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="extend",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=query_rows,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=pages,
        )
    )
    q = (torch.randn(query_rows, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.tensor(
        [[4, 7, 1, 5, 0, 3, 6, 2]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([101], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(
        [0, query_rows],
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        query_rows,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = _bind(plan, q=q,
    kv_cache=cache,
    output=output,
    page_table=page_table,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,)
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_padded_page_stride_matches_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260801)
    page_size = 16
    pages = 5
    page_payload = page_size * QK_DIM
    page_stride = page_payload + 128
    storage = torch.empty(
        pages * page_stride,
        dtype=torch.bfloat16,
        device=device,
    )
    cache = torch.as_strided(
        storage,
        size=(pages, page_size, QK_DIM),
        stride=(page_stride, QK_DIM, 1),
    )
    cache.normal_().mul_(0.1)
    caps = dense_mla.Caps(
        device=device,
        mode="decode",
        kv_dtype=torch.bfloat16,
        num_q_heads=HEADS,
        page_size=page_size,
        max_total_q=1,
        max_batch=1,
        max_cache_tokens=64,
        max_page_table_width=4,
        num_cache_pages=pages,
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    page_table = torch.tensor(
        [[4, 1, 3, 0]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([61], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    plan = dense_mla.plan(
        caps,
        invocation=dense_mla.invocation_from_tensors(
            caps,
            q=q,
            kv_cache=cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        ),
    )
    binding = _bind(plan, q=q,
    kv_cache=cache,
    output=output,
    page_table=page_table,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,)
    q_before = q.clone()
    cache_before = cache.clone()
    page_table_before = page_table.clone()
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.testing.assert_close(q, q_before, rtol=0, atol=0)
    torch.testing.assert_close(cache, cache_before, rtol=0, atol=0)
    torch.testing.assert_close(page_table, page_table_before, rtol=0, atol=0)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_cuda_graph_replay_is_allocation_stable_and_reads_live_inputs() -> None:
    device = require_b12x()
    torch.manual_seed(20260802)
    page_size = 16
    pages = 8
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=128,
            max_page_table_width=8,
            num_cache_pages=pages,
            use_cuda_graph=True,
        )
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(pages, page_size, QK_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    page_table = torch.arange(
        pages,
        dtype=torch.int32,
        device=device,
    ).reshape(1, pages)
    cache_seqlens = torch.tensor([97], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = _bind(plan, q=q,
    kv_cache=cache,
    output=output,
    page_table=page_table,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,)

    dense_mla.run(binding=binding)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated(device)
    q.mul_(0.75)
    cache_seqlens.fill_(65)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated(device)
    assert allocated_after == allocated_before

    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        captured_output,
        captured_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_fp8_production_split_plan_handles_short_live_sequence() -> None:
    """The 1M K3 plan must not touch inactive split storage or cache pages."""
    device = require_b12x()
    torch.manual_seed(20260803)
    page_size = 768
    page_width = (131_072 + page_size - 1) // page_size
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=FP8,
            num_q_heads=48,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=131_072,
            max_page_table_width=page_width,
            num_cache_pages=1,
            use_cuda_graph=True,
        )
    )

    q_float = torch.randn(1, 48, QK_DIM, device=device) * 0.1
    cache_float = torch.randn(1, page_size, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache = (cache_float / kv_scale).to(FP8)
    page_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([257], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        48,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    plan, scratch_spec = _prepare(
        plan,
        name="dense-mla-split",
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
        active_splits=1,
    )
    guarded_storage, scratch = _guarded_scratch(scratch_spec)
    binding = dense_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
        active_splits=1,
    )
    assert binding.active_splits == 1
    actual_output, actual_lse = dense_mla.run(binding=binding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_lse = dense_mla.run(binding=binding)
    graph.replay()
    torch.cuda.synchronize()
    guard_bytes = scratch.storage_offset()
    expected_guard = torch.full(
        (guard_bytes,),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    torch.testing.assert_close(guarded_storage[:guard_bytes], expected_guard)
    torch.testing.assert_close(guarded_storage[-guard_bytes:], expected_guard)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    _assert_matches(
        captured_output,
        captured_lse,
        expected_output,
        expected_lse,
    )

    # The active prefix preserves the maximum-context split boundaries and
    # only omits mathematically empty tail splits. It must therefore match a
    # full-plan launch exactly, including the BF16 output bits.
    full_output = torch.empty_like(output)
    full_binding = dense_mla.bind(
        plan,
        scratch=_scratch(scratch_spec),
        q=q,
        kv_cache=cache,
        output=full_output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    assert full_binding.active_splits > binding.active_splits
    full_actual, full_lse = dense_mla.run(binding=full_binding)
    torch.cuda.synchronize()
    torch.testing.assert_close(full_actual, captured_output, rtol=0, atol=0)
    torch.testing.assert_close(full_lse, captured_lse, rtol=0, atol=0)


@torch.inference_mode()
def test_page_ids_past_int32_scaled_offset_match_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260803)
    page_size = 16
    record_bytes = (
        QK_DIM
        * torch.empty(
            (),
            dtype=torch.bfloat16,
        ).element_size()
    )
    page_stride_bytes = page_size * record_bytes
    high_page = torch.iinfo(torch.int32).max // page_stride_bytes + 2
    live_pages = 2
    pages = high_page + live_pages
    assert high_page * page_stride_bytes > torch.iinfo(torch.int32).max

    # Roughly 2 GiB is intentionally mostly uninitialized. Only the live tail
    # pages are touched; this reproduces high recycled pool ids.
    cache = torch.empty(
        pages,
        page_size,
        QK_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    cache[high_page:].normal_().mul_(0.1)
    page_table = torch.arange(
        high_page,
        pages,
        dtype=torch.int32,
        device=device,
    ).reshape(1, live_pages)
    assert (
        int(page_table.min().item()) * cache.stride(0) * cache.element_size()
        > torch.iinfo(torch.int32).max
    )

    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=page_size * live_pages,
            max_page_table_width=live_pages,
            num_cache_pages=pages,
        )
    )
    q = (torch.randn(1, HEADS, QK_DIM, device=device) * 0.1).to(torch.bfloat16)
    cache_seqlens = torch.tensor(
        [page_size + 9],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = _bind(plan, q=q,
    kv_cache=cache,
    output=output,
    page_table=page_table,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,)
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )


@torch.inference_mode()
def test_fp8_page_ids_past_int32_scaled_offset_match_reference() -> None:
    device = require_b12x()
    torch.manual_seed(20260803)
    page_size = 768
    record_bytes = QK_DIM
    page_stride_bytes = page_size * record_bytes
    high_page = torch.iinfo(torch.int32).max // page_stride_bytes + 2
    live_pages = 2
    pages = high_page + live_pages
    assert high_page * page_stride_bytes > torch.iinfo(torch.int32).max

    cache = torch.empty(
        pages,
        page_size,
        QK_DIM,
        dtype=FP8,
        device=device,
    )
    cache_float = (
        torch.randn(
            live_pages,
            page_size,
            QK_DIM,
            device=device,
        )
        * 0.1
    )
    kv_scale = (cache_float.abs().max() / 400).reshape(1).float()
    cache[high_page:] = (cache_float / kv_scale).to(FP8)
    page_table = torch.arange(
        high_page,
        pages,
        dtype=torch.int32,
        device=device,
    ).reshape(1, live_pages)
    assert (
        int(page_table.min().item()) * cache.stride(0) * cache.element_size()
        > torch.iinfo(torch.int32).max
    )

    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            kv_dtype=FP8,
            num_q_heads=HEADS,
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=page_size * live_pages,
            max_page_table_width=live_pages,
            num_cache_pages=pages,
        )
    )
    q_float = torch.randn(1, HEADS, QK_DIM, device=device) * 0.1
    q_scale = (q_float.abs().max() / 400).reshape(1).float()
    q = (q_float / q_scale).to(FP8)
    cache_seqlens = torch.tensor(
        [page_size + 9],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.empty(
        1,
        HEADS,
        VALUE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    binding = _bind(plan, q=q,
    kv_cache=cache,
    output=output,
    page_table=page_table,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,
    q_scale=q_scale,
    kv_scale=kv_scale,)
    actual_output, actual_lse = dense_mla.run(binding=binding)
    expected_output, expected_lse = dense_mla.reference(
        q,
        cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    _assert_matches(
        actual_output,
        actual_lse,
        expected_output,
        expected_lse,
    )
