"""Decode CUDA-graph replay ownership across shared paged-attention scratch."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from b12x.attention import paged
from b12x.attention.paged.reference import paged_attention_reference
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall, require_prepared

from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.paged_attention_helpers import make_paged_inputs


_SCHEDULE_FIELDS = (
    "request_indices",
    "qo_tile_indices",
    "kv_tile_indices",
    "merge_indptr",
    "o_indptr",
    "block_valid_mask",
    "kv_window_start_tokens",
    "kv_chunk_size_ptr",
    "total_num_rows_ptr",
)


def _schedule_ptrs(binding: object) -> dict[str, int]:
    scratch = binding.scratch
    return {
        name: int(getattr(scratch, name).data_ptr())
        for name in _SCHEDULE_FIELDS
    }


@torch.inference_mode()
def test_decode_graph_plan_owned_replay_state_survives_shared_scratch_and_big_pid() -> None:
    """Prepared decode plans retain independent schedules outside shared scratch."""
    device = require_b12x()
    batch = 2
    page_size = 64
    page_table_width = 66
    q, small_k, small_v, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=[4096, 3072],
        page_size=page_size,
        seed=4817,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        page_table_width=page_table_width,
        num_pages=256,
    )

    # page_stride = 64 * 1 * 256 * sizeof(bf16) = 32768 bytes.  Starting live
    # pages at 65537 puts every pool-scaled byte offset past signed Int32.
    high_page_base = 65_537
    k_cache = torch.empty(
        (high_page_base + int(small_k.shape[0]), *small_k.shape[1:]),
        dtype=small_k.dtype,
        device=device,
    )
    v_cache = torch.empty(
        (high_page_base + int(small_v.shape[0]), *small_v.shape[1:]),
        dtype=small_v.dtype,
        device=device,
    )
    k_cache[high_page_base:].copy_(small_k)
    v_cache[high_page_base:].copy_(small_v)
    page_table.add_(high_page_base)
    del small_k, small_v

    q_full, page_table_full, seqlens_full, cu_full = (
        q,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    q_window = q.clone().mul_(0.875)
    page_table_window = page_table.clone()
    seqlens_window = cache_seqlens.clone()
    cu_window = cu_seqlens_q.clone()
    output_full = torch.empty_like(q_full)
    output_window = torch.empty_like(q_window)
    window_left = 1024

    def declare(
        q_input: torch.Tensor,
        output: torch.Tensor,
        page_table_input: torch.Tensor,
        seqlens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        window: int,
    ):
        caps = paged.Caps(
            device=device,
            mode="decode",
            dtype=q_input.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q_input.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q_input.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=page_size,
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=512,
            max_partial_rows=512,
            num_cache_pages=k_cache.shape[0],
            use_cuda_graph=True,
        )
        invocation = paged.invocation_from_tensors(
            caps,
            q=q_input,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table_input,
            cache_seqlens=seqlens,
            cu_seqlens_q=cu_seqlens,
        )
        declaration = paged.plan(
            caps,
            invocation=FrozenMapping({**dict(invocation), "window_left": window}),
        )
        return declaration

    full_declaration = declare(
        q_full, output_full, page_table_full, seqlens_full, cu_full, window=-1
    )
    window_declaration = declare(
        q_window,
        output_window,
        page_table_window,
        seqlens_window,
        cu_window,
        window=window_left,
    )
    prepared_specs: dict[str, object] = {}

    def prime(
        name: str,
        q_input: torch.Tensor,
        output: torch.Tensor,
        page_table_input: torch.Tensor,
        seqlens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        window: int,
    ):
        def prepare_call(state: object) -> PreparedCall:
            (spec,) = state.scratch_plan.scratch_specs()
            state.prepare_decode_graph_replay_state(
                batch=batch,
                total_q_capacity=batch,
                max_page_table_width=page_table_width,
                max_cache_page_count=page_table_width,
                window_left=window,
            )
            prepared_specs[name] = spec
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = state.bind(
                scratch=scratch,
                q=q_input,
                k_cache=k_cache,
                v_cache=v_cache,
                output=output,
                page_table=page_table_input,
                cache_seqlens=seqlens,
                cu_seqlens_q=cu_seqlens,
                window_left=window,
                active_total_q=batch,
            )
            return PreparedCall(
                run=lambda: state.run(binding),
                output=output,
                owners=(scratch, binding),
            )

        return prepare_call

    full_request = full_declaration.request(
        name="decode-full",
        prepare_call=prime(
            "full", q_full, output_full, page_table_full, seqlens_full, cu_full, window=-1
        ),
    )
    window_request = window_declaration.request(
        name="decode-window",
        prepare_call=prime(
            "window",
            q_window,
            output_window,
            page_table_window,
            seqlens_window,
            cu_window,
            window=window_left,
        ),
    )

    with PreparationSession(device=device, autotune=False) as session:
        result = session.prepare((full_request, window_request))
        graph = torch.cuda.CUDAGraph()
        try:
            plan_full = result.plans["decode-full"]
            plan_window = result.plans["decode-window"]
            full_spec = prepared_specs["full"]
            window_spec = prepared_specs["window"]
            assert full_spec == window_spec
            shared_scratch = torch.empty(
                full_spec.shape,
                dtype=full_spec.dtype,
                device=device,
            )
            with session.capture(), torch.cuda.graph(graph):
                binding_full = paged.bind(
                    plan_full,
                    scratch=shared_scratch,
                    q=q_full,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    output=output_full,
                    page_table=page_table_full,
                    cache_seqlens=seqlens_full,
                    cu_seqlens_q=cu_full,
                    window_left=-1,
                    active_total_q=batch,
                )
                paged.run(binding=binding_full, plan=plan_full)
                binding_window = paged.bind(
                    plan_window,
                    scratch=shared_scratch,
                    q=q_window,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    output=output_window,
                    page_table=page_table_window,
                    cache_seqlens=seqlens_window,
                    cu_seqlens_q=cu_window,
                    window_left=window_left,
                    active_total_q=batch,
                )
                paged.run(binding=binding_window, plan=plan_window)

            # Public bindings share numerical workspace, while every prepared
            # plan owns the schedule addresses captured by its graph.
            assert binding_full.scratch._owner_scratch_plan is require_prepared(plan_full, "attention.gqa").scratch_plan
            assert binding_window.scratch._owner_scratch_plan is require_prepared(plan_window, "attention.gqa").scratch_plan
            full_ptrs = _schedule_ptrs(binding_full)
            window_ptrs = _schedule_ptrs(binding_window)
            assert all(full_ptrs[name] != window_ptrs[name] for name in _SCHEDULE_FIELDS)
            scratch_start = int(shared_scratch.data_ptr())
            scratch_end = scratch_start + int(shared_scratch.numel())
            assert all(
                not (scratch_start <= ptr < scratch_end)
                for ptr in (*full_ptrs.values(), *window_ptrs.values())
            )
            with pytest.raises(RuntimeError, match="cannot replace decode graph replay state"):
                require_prepared(plan_full, "attention.gqa").prepare_decode_graph_replay_state(
                    batch=batch,
                    total_q_capacity=batch,
                    max_page_table_width=page_table_width,
                    max_cache_page_count=page_table_width,
                    window_left=-1,
                )

            retained_schedule = weakref.ref(binding_full.scratch.merge_indptr)
            del binding_full, binding_window
            gc.collect()
            assert retained_schedule() is not None
            assert int(retained_schedule().data_ptr()) == full_ptrs["merge_indptr"]

            def replay_and_check(
                full_lengths: tuple[int, int],
                window_lengths: tuple[int, int],
                *,
                check_allocator: bool,
            ) -> None:
                seqlens_full.copy_(
                    torch.tensor(full_lengths, dtype=torch.int32, device=device)
                )
                seqlens_window.copy_(
                    torch.tensor(window_lengths, dtype=torch.int32, device=device)
                )
                expected_full, _ = paged_attention_reference(
                    q_full, k_cache, v_cache, page_table_full, seqlens_full, cu_full,
                    causal=True, window_left=-1,
                )
                expected_window, _ = paged_attention_reference(
                    q_window, k_cache, v_cache, page_table_window, seqlens_window, cu_window,
                    causal=True, window_left=window_left,
                )
                output_full.fill_(torch.nan)
                output_window.fill_(torch.nan)
                # Every byte of the common numerical allocation is invalid
                # between replays; captured schedule metadata must survive.
                shared_scratch.fill_(0x5A)
                allocated_before = torch.cuda.memory_allocated(device)
                reserved_before = torch.cuda.memory_reserved(device)
                graph.replay()
                torch.cuda.synchronize(device)
                if check_allocator:
                    assert torch.cuda.memory_allocated(device) == allocated_before
                    assert torch.cuda.memory_reserved(device) == reserved_before
                torch.testing.assert_close(
                    output_full.float(), expected_full.float(), atol=2e-2, rtol=2e-2
                )
                torch.testing.assert_close(
                    output_window.float(), expected_window.float(), atol=2e-2, rtol=2e-2
                )
                assert torch.isfinite(output_full).all()
                assert torch.isfinite(output_window).all()

            # The first replay settles CUDA graph bookkeeping; the second
            # proves replay creates no allocator growth.
            replay_and_check((512, 2048), (1536, 4096), check_allocator=False)
            replay_and_check((4096, 1024), (2048, 3072), check_allocator=True)
        finally:
            graph.reset()
            result.close()
