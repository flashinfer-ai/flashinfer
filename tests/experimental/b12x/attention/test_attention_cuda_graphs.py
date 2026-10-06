from __future__ import annotations

import math

import pytest
import torch

from b12x.attention.paged.reference import paged_attention_reference
from b12x.attention._shared.contiguous.api import clear_attention_caches
from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged._scratch import (
    B12XPagedAttentionScratchCaps,
    plan_paged_attention_scratch,
)
from b12x.attention.paged.planner import create_paged_plan
from b12x.attention.paged.planner import plan_extend_graph_capacity
from b12x.attention.paged.planner import plan_verify_graph_capacity

from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.paged_attention_helpers import (
    make_paged_inputs,
    quantize_paged_kv_cache_e4m3,
)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _lse_base2_to_natural(lse: torch.Tensor) -> torch.Tensor:
    return lse * math.log(2.0)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("num_q_heads", "page_size", "window_left"),
    [(24, 64, -1), (36, 128, 511)],
)
def test_laguna_fp8_verifier_replays_fixed_graph_across_context_lengths(
    num_q_heads: int,
    page_size: int,
    window_left: int,
) -> None:
    require_b12x()
    clear_attention_caches()
    torch.manual_seed(113)
    device = torch.device("cuda")
    query_len = 8
    num_kv_heads = 4
    head_dim = 128
    max_context = 8192
    num_pages = max_context // page_size
    q = torch.randn(
        query_len,
        num_q_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.25)
    combined_cache = (
        torch.randn(
            num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    k_cache, v_cache = combined_cache.unbind(1)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([max_context], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    descale = torch.ones((1,), dtype=torch.float32, device=device)
    capacity = plan_verify_graph_capacity(
        device=device,
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=page_size,
        batch=1,
        query_len=query_len,
        max_cache_page_count=num_pages,
        window_left=window_left,
    )
    cache_seqlens.fill_(capacity.representative_cache_seqlen)
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=device,
            mode="verify",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=query_len,
            max_batch=1,
            max_page_table_width=num_pages,
            max_work_items=capacity.max_work_items,
            max_partial_rows=capacity.max_partial_rows,
            num_cache_pages=num_pages,
            use_cuda_graph=True,
        )
    )
    scratch_plan.prepare_graph_replay_state(
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        active_total_q=query_len,
        window_left=window_left,
    )
    scratch = torch.empty(
        scratch_plan.layout.nbytes,
        dtype=torch.uint8,
        device=device,
    )
    output = torch.empty_like(q)

    def run() -> None:
        binding = scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=query_len,
            window_left=window_left,
            k_descale=descale,
            v_descale=descale,
        )
        paged_attention_forward(binding=binding)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for context in (1024, max_context):
        cache_seqlens.fill_(context)
        expected, _ = paged_attention_reference(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            k_descale=descale,
            v_descale=descale,
            causal=True,
            window_left=window_left,
        )
        graph.replay()
        torch.cuda.synchronize()
        assert (output - expected).abs().max().item() <= 0.01
        assert _cosine_similarity(output, expected) >= 0.999


@torch.inference_mode()
def test_laguna_fp8_extend_replays_smaller_uneven_query_capacity() -> None:
    """Live query partitions are packed on device under one fixed graph plan."""
    require_b12x()
    clear_attention_caches()
    torch.manual_seed(4129)
    device = torch.device("cuda")
    batch = 2
    capacity_q = 128
    live_total_q = 21
    page_size = 128
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 128
    num_pages = 8

    q = torch.randn(
        live_total_q,
        num_q_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.125)
    k_cache = (
        torch.randn(
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        .mul_(0.125)
        .to(torch.float8_e4m3fn)
    )
    v_cache = (
        torch.randn(
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        .mul_(0.125)
        .to(torch.float8_e4m3fn)
    )
    page_table = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([389, 501], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 8, live_total_q], dtype=torch.int32, device=device)
    prepare_cu_seqlens_q = torch.tensor(
        [0, 1, capacity_q], dtype=torch.int32, device=device
    )
    descale = torch.ones((batch,), dtype=torch.float32, device=device)
    output = torch.empty_like(q)

    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=device,
            mode="extend",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=capacity_q,
            max_batch=batch,
            max_page_table_width=page_table.shape[1],
            max_work_items=256,
            max_partial_rows=0,
            num_cache_pages=num_pages,
            use_cuda_graph=True,
            copy_runtime_metadata=False,
        )
    )
    scratch_plan.prepare_graph_replay_state(
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=prepare_cu_seqlens_q,
        active_total_q=capacity_q,
        disable_split_kv=True,
        window_left=-1,
    )
    (scratch_spec,) = scratch_plan.scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )

    def bind():
        return scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            disable_split_kv=True,
            window_left=-1,
            k_descale=descale,
            v_descale=descale,
        )

    eager_binding = bind()
    actual, _ = eager_binding.run()
    torch.cuda.synchronize()
    expected, _ = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=descale,
        v_descale=descale,
        causal=True,
    )
    assert actual.shape == q.shape
    assert (actual - expected).abs().max().item() <= 0.02
    assert _cosine_similarity(actual, expected) >= 0.999

    schedule_ptrs = tuple(
        int(getattr(eager_binding.scratch, name).data_ptr())
        for name in (
            "request_indices",
            "qo_tile_indices",
            "kv_tile_indices",
            "block_valid_mask",
        )
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_binding = bind()
        captured_binding.run()
    assert schedule_ptrs == tuple(
        int(getattr(captured_binding.scratch, name).data_ptr())
        for name in (
            "request_indices",
            "qo_tile_indices",
            "kv_tile_indices",
            "block_valid_mask",
        )
    )

    q.copy_(torch.randn_like(q).mul_(0.125))
    cache_seqlens.copy_(torch.tensor([257, 417], dtype=torch.int32, device=device))
    cu_seqlens_q[1] = 5
    graph.replay()
    torch.cuda.synchronize()
    expected_replay, _ = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=descale,
        v_descale=descale,
        causal=True,
    )
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output).item() > 0
    assert (output - expected_replay).abs().max().item() <= 0.02
    assert _cosine_similarity(output, expected_replay) >= 0.999


@torch.inference_mode()
def test_laguna_fp8_extend_reuses_dynamic_worklist_grid() -> None:
    """A shared compiled entry must launch the current graph-static capacity."""
    require_b12x()
    clear_attention_caches()
    torch.manual_seed(4130)
    device = torch.device("cuda")
    page_size = 128
    num_q_heads = 24
    num_kv_heads = 4
    head_dim = 128
    num_pages = 8
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor(
        [num_pages * page_size], dtype=torch.int32, device=device
    )
    combined_cache = (
        torch.randn(
            num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        .mul_(0.125)
        .to(torch.float8_e4m3fn)
    )
    k_cache, v_cache = combined_cache.unbind(1)
    descale = torch.ones((1,), dtype=torch.float32, device=device)

    def run_capacity(total_q: int) -> tuple[torch.Tensor, torch.Tensor]:
        q = torch.randn(
            total_q,
            num_q_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        ).mul_(0.125)
        cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device=device)
        capacity = plan_extend_graph_capacity(
            device=device,
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            batch=1,
            total_q_capacity=total_q,
            max_cache_page_count=num_pages,
            window_left=-1,
        )
        scratch_plan = plan_paged_attention_scratch(
            B12XPagedAttentionScratchCaps(
                device=device,
                mode="extend",
                dtype=q.dtype,
                kv_dtype=k_cache.dtype,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
                page_size=page_size,
                max_total_q=total_q,
                max_batch=1,
                max_page_table_width=num_pages,
                max_work_items=capacity.max_work_items,
                max_partial_rows=0,
                num_cache_pages=num_pages,
                use_cuda_graph=True,
                copy_runtime_metadata=False,
            )
        )
        scratch_plan.prepare_graph_replay_state(
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=total_q,
            disable_split_kv=True,
            window_left=-1,
        )
        scratch = torch.empty(
            scratch_plan.layout.nbytes,
            dtype=torch.uint8,
            device=device,
        )
        output = torch.full_like(q, float("nan"))
        binding = scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            disable_split_kv=True,
            window_left=-1,
            k_descale=descale,
            v_descale=descale,
        )
        actual, _ = binding.run()
        torch.cuda.synchronize()
        return q, actual

    # On the Laguna production geometry Q=128 uses the architecture wave,
    # while Q=1024 needs a larger direct worklist. Compile the small launch
    # first so a stale exact grid would leave the tail of the second output
    # poisoned.
    run_capacity(128)
    q, actual = run_capacity(1024)
    expected, _ = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        torch.tensor([0, q.shape[0]], dtype=torch.int32, device=device),
        k_descale=descale,
        v_descale=descale,
        causal=True,
    )
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual).item() > 0
    assert (actual - expected).abs().max().item() <= 0.02
    assert _cosine_similarity(actual, expected) >= 0.999


class _PagedGraphScratchHarness:
    def __init__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        mode: str,
    ) -> None:
        self.q = q
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.mode = mode
        self.plan = None
        self._scratch_plan = None
        self._scratch = None
        self._output = None
        self._k_descale = None
        self._v_descale = None
        self._page_table = None
        self._cache_seqlens = None
        self._cu_seqlens_q = None
        self._last_scratch = None

    def prepare(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        plan = create_paged_plan(
            self.q,
            self.k_cache,
            self.v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode=self.mode,
            enable_cuda_graph=True,
        )
        self.plan = plan
        if self._scratch_plan is None:
            # Graph plans carry a fixed resident launch grid in
            # padded_batch_size, even when the live work-item count is small.
            # Both the work metadata and block-valid plane must cover it.
            max_work_items = max(
                plan.new_batch_size,
                plan.padded_batch_size,
                1,
            )
            self._scratch_plan = plan_paged_attention_scratch(
                B12XPagedAttentionScratchCaps(
                    device=self.q.device,
                    mode=self.mode,
                    dtype=self.q.dtype,
                    kv_dtype=self.k_cache.dtype,
                    num_q_heads=self.q.shape[1],
                    num_kv_heads=self.k_cache.shape[2],
                    head_dim_qk=self.q.shape[2],
                    head_dim_vo=self.v_cache.shape[3],
                    page_size=self.k_cache.shape[1],
                    max_total_q=plan.total_q,
                    max_batch=page_table.shape[0],
                    max_page_table_width=page_table.shape[1],
                    max_work_items=max_work_items,
                    max_partial_rows=plan.total_num_partial_rows,
                    num_cache_pages=self.k_cache.shape[0],
                    use_cuda_graph=True,
                )
            )
            self._scratch = tuple(
                torch.empty(shape, dtype=dtype, device=self.q.device)
                for shape, dtype in self._scratch_plan.shapes_and_dtypes()
            )
            if self.mode == "decode":
                self._scratch_plan.prepare_decode_graph_replay_state(
                    batch=page_table.shape[0],
                    max_page_table_width=page_table.shape[1],
                )
            else:
                self._scratch_plan.prepare_graph_replay_state(
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    active_total_q=plan.total_q,
                )
        self._page_table = page_table
        self._cache_seqlens = cache_seqlens
        self._cu_seqlens_q = cu_seqlens_q
        if self._output is not None:
            self._bind()

    def _bind(self):
        assert self._scratch_plan is not None
        assert self._scratch is not None
        assert self._output is not None
        binding = self._scratch_plan.bind(
            scratch=self._scratch,
            q=self.q,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            output=self._output,
            page_table=self._page_table,
            cache_seqlens=self._cache_seqlens,
            cu_seqlens_q=self._cu_seqlens_q,
            k_descale=self._k_descale,
            v_descale=self._v_descale,
        )
        self._last_scratch = binding.scratch
        self.plan = binding.scratch.plan
        return binding

    def prepare_for_capture(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> None:
        """Materialize graph-mode scratch planning before capture begins."""
        self.q = q
        self.k_cache = k_cache
        self.v_cache = v_cache
        self._output = output
        self._k_descale = k_descale
        self._v_descale = v_descale
        self._bind()

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.q = q
        self.k_cache = k_cache
        self.v_cache = v_cache
        self._output = output
        self._k_descale = k_descale
        self._v_descale = v_descale
        return paged_attention_forward(binding=self._bind())

    def current_lse_view(self) -> torch.Tensor:
        assert self._last_scratch is not None
        return self._last_scratch.current_lse_view()


@torch.inference_mode()
@pytest.mark.xfail(
    reason="stale vs current decode-graph planner heuristics: capacity derivation from a live plan, LUT staging shape, and prepare ordering all need a refresh",
    strict=False,
)
def test_paged_attention_decode_replays_under_cuda_graph_with_variable_metadata() -> (
    None
):
    require_b12x()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 128, 192, 256],
        page_size=64,
        seed=73,
        page_table_width=64,
        num_pages=512,
    )
    _, _, _, page_table_max, cache_seqlens_max, cu_seqlens_q_max = make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[4096, 4096, 4096, 4096],
        page_size=64,
        seed=74,
        page_table_width=page_table.shape[1],
        num_pages=k_cache.shape[0],
    )
    workspace = _PagedGraphScratchHarness(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table_max, cache_seqlens_max, cu_seqlens_q_max)
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output = torch.empty_like(q)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(q, k_cache, v_cache, output=output)

    ref_out_1, ref_lse_1 = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out_1).abs().max().item() <= 0.02
    assert (
        _lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_1
    ).abs().max().item() <= 0.03
    assert _cosine_similarity(output, ref_out_1) >= 0.99999

    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = (
        make_paged_inputs(
            q_seqlens=[1, 1, 1, 1],
            cache_seqlens=[2048, 2048, 4096, 4096],
            page_size=64,
            seed=79,
            page_table_width=page_table.shape[1],
            num_pages=k_cache.shape[0],
        )
    )
    q.copy_(q_2)
    k_cache.copy_(k_cache_2)
    v_cache.copy_(v_cache_2)
    page_table.copy_(page_table_2)
    cache_seqlens.copy_(cache_seqlens_2)
    cu_seqlens_q.copy_(cu_seqlens_q_2)
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    ref_out_2, ref_lse_2 = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out_2).abs().max().item() <= 0.02
    assert (
        _lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_2
    ).abs().max().item() <= 0.03
    assert _cosine_similarity(output, ref_out_2) >= 0.99999


@torch.inference_mode()
@pytest.mark.xfail(
    reason="stale vs current decode-graph planner heuristics: capacity derivation from a live plan, LUT staging shape, and prepare ordering all need a refresh",
    strict=False,
)
def test_paged_attention_extend_replays_under_cuda_graph_with_smaller_metadata() -> (
    None
):
    require_b12x()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=83,
        page_table_width=4,
    )
    workspace = _PagedGraphScratchHarness(q, k_cache, v_cache, mode="extend")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output = torch.empty_like(q)
    workspace.prepare_for_capture(q, k_cache, v_cache, output=output)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(q, k_cache, v_cache, output=output)

    ref_out_1, ref_lse_1 = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output[: q.shape[0]] - ref_out_1).abs().max().item() <= 0.02
    assert (
        _lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_1
    ).abs().max().item() <= 0.03
    assert _cosine_similarity(output[: q.shape[0]], ref_out_1) >= 0.99999

    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = (
        make_paged_inputs(
            q_seqlens=[4, 4, 4, 4],
            cache_seqlens=[64, 97, 81, 113],
            page_size=64,
            seed=89,
            page_table_width=page_table.shape[1],
            num_pages=k_cache.shape[0],
        )
    )
    q.zero_()
    q[: q_2.shape[0]].copy_(q_2)
    k_cache.copy_(k_cache_2)
    v_cache.copy_(v_cache_2)
    page_table.copy_(page_table_2)
    cache_seqlens.copy_(cache_seqlens_2)
    cu_seqlens_q.copy_(cu_seqlens_q_2)
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    ref_out_2, ref_lse_2 = paged_attention_reference(
        q[: q_2.shape[0]],
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output[: q_2.shape[0]] - ref_out_2).abs().max().item() <= 0.02
    assert (
        _lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_2
    ).abs().max().item() <= 0.03
    assert _cosine_similarity(output[: q_2.shape[0]], ref_out_2) >= 0.99999


@torch.inference_mode()
@pytest.mark.xfail(
    reason="stale vs current decode-graph planner heuristics: capacity derivation from a live plan, LUT staging shape, and prepare ordering all need a refresh",
    strict=False,
)
def test_paged_attention_fp8_kv_replays_under_cuda_graph() -> None:
    require_b12x()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=97,
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _PagedGraphScratchHarness(q, k_fp8, v_fp8, mode="extend")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output = torch.empty_like(q)
    workspace.prepare_for_capture(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        workspace.run(
            q,
            k_fp8,
            v_fp8,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    ref_out_1, ref_lse_1 = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out_1).abs().max().item() <= 0.05
    assert (
        _lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_1
    ).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out_1) >= 0.9999

    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = (
        make_paged_inputs(
            q_seqlens=[6, 5, 7, 4],
            cache_seqlens=[97, 81, 113, 68],
            page_size=64,
            seed=101,
            page_table_width=page_table.shape[1],
            num_pages=k_cache.shape[0],
        )
    )
    k_fp8_2, v_fp8_2, k_descale_2, v_descale_2 = quantize_paged_kv_cache_e4m3(
        k_cache_2,
        v_cache_2,
        page_table_2,
        cache_seqlens_2,
    )
    q.copy_(q_2)
    k_fp8.copy_(k_fp8_2)
    v_fp8.copy_(v_fp8_2)
    k_descale.copy_(k_descale_2)
    v_descale.copy_(v_descale_2)
    page_table.copy_(page_table_2)
    cache_seqlens.copy_(cache_seqlens_2)
    cu_seqlens_q.copy_(cu_seqlens_q_2)
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    ref_out_2, ref_lse_2 = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (output - ref_out_2).abs().max().item() <= 0.05
    assert (
        _lse_base2_to_natural(workspace.current_lse_view()) - ref_lse_2
    ).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out_2) >= 0.9999
