"""Contract and GPU tests for request-ordered Cake FMHA decode."""

from __future__ import annotations

import dataclasses
import inspect
import math
import weakref

import flashinfer
import flashinfer.cake_fmha as cake_api
import pytest
import torch


@pytest.mark.parametrize("q_len", (1, 6))
def test_request_order_plan_is_independent_of_length_and_order_contents(q_len) -> None:
    plan = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        (8193, 57345, 73729, 81921),
        q_len,
    )
    changed = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        (q_len, 129, 65, 1025),
        q_len,
        request_order_case="identity",
    )
    assert plan == changed
    assert plan.runtime_length_scheduler
    assert cake_api._is_authenticated_request_ordered_plan(plan)
    assert plan.workspace_size_bytes > 388
    assert not cake_api._is_authenticated_request_ordered_plan(
        dataclasses.replace(plan, workspace_parts=1)
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.total_tiles = 4


@pytest.mark.parametrize(
    ("q_len", "capacity_groups"),
    (
        (
            1,
            {
                256: (1,),
                128: (2,),
                64: (3, 4),
                32: (5, 8, 9),
                16: (10, 27, 32, 151, 152, 153, 256),
            },
        ),
        (6, {32: (1,), 16: (2, 8, 27, 32, 151, 152, 153, 256)}),
    ),
)
def test_request_order_runtime_family_reuses_source_across_batches(
    q_len, capacity_groups
) -> None:
    plans = [
        cake_api.plan_cake_fmha_request_ordered_paged_decode(
            [q_len + 1] * batch,
            q_len,
        )
        for batches in capacity_groups.values()
        for batch in batches
    ]
    assert all(plan.sm_count == 152 for plan in plans)
    for capacity, batches in capacity_groups.items():
        family = [plan for plan in plans if plan.batch_size in batches]
        assert {plan.workspace_parts for plan in family} == {capacity}
        assert len({plan.module_name for plan in family}) == 1
        assert all(plan.total_tiles == plan.batch_size * capacity for plan in family)
    assert len({plan.module_name for plan in plans}) == len(capacity_groups)
    assert all(cake_api._is_authenticated_request_ordered_plan(plan) for plan in plans)
    for plan in plans:
        assert plan == cake_api._default_cake_fmha_request_ordered_plan(
            batch_size=plan.batch_size,
            q_len=q_len,
            write_lse=False,
        )


@pytest.mark.parametrize("q_len", (1, 6))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_runtime_capture_private_descriptors_survive_caller_scratch_overwrite(
    q_len,
) -> None:
    """Two bindings of one plan keep descriptors outside reusable caller scratch."""
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("request-ordered Cake FMHA requires SM103")
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip("request-ordered Cake FMHA requires a 152-SM device")
    batch, length = 2, 2049
    pages = (length + 63) // 64
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(4832 + q_len)
    plan = cake_api.plan_cake_fmha_request_ordered_paged_decode([length] * batch, q_len)
    kv = torch.randn(
        (batch * pages, 2, 1, 64, 256), device=device, generator=generator
    ).to(torch.float8_e4m3fn)
    tables = torch.arange(batch * pages, dtype=torch.int32, device=device).view(
        batch, pages
    )
    lengths = torch.full((batch,), length, dtype=torch.int32, device=device)
    order = torch.arange(batch - 1, -1, -1, dtype=torch.int32, device=device)
    scale1 = torch.tensor([math.log2(math.e) / 16], dtype=torch.float32, device=device)
    scale2 = torch.ones(1, dtype=torch.float32, device=device)
    bindings = []
    for _ in range(2):
        query = torch.randn(
            (batch * q_len, 8, 256), device=device, generator=generator
        ).to(torch.bfloat16)
        bindings.append(
            {
                "query": query,
                "out": torch.empty_like(query),
                "workspace_buffer": torch.empty(
                    plan.workspace_size_bytes, dtype=torch.uint8, device=device
                ),
                "multi_ctas_kv_counter_buffer": torch.zeros(
                    batch * q_len, dtype=torch.int32, device=device
                ),
            }
        )

    def invoke(binding, preparation=None):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            **binding,
            kv_cache=(kv[:, 0], kv[:, 1]),
            block_tables=tables,
            seq_lens=lengths,
            max_seq_len=length,
            q_len_per_req=q_len,
            bmm1_scale_log2=scale1,
            bmm2_scale=scale2,
            backend="cake",
            enable_pdl=True,
            request_order=order,
            request_order_plan=plan,
            request_order_capture=preparation,
        )

    for binding in bindings:
        invoke(binding)
    torch.cuda.synchronize()
    expected = [binding["out"].clone() for binding in bindings]
    unprepared = cake_api.CakeFmhaRequestOrderedCapture([plan])
    rejected_graph = torch.cuda.CUDAGraph()
    try:
        with (
            torch.cuda.graph(rejected_graph),
            pytest.raises(RuntimeError, match="call prepare_workspace"),
        ):
            invoke(bindings[0], unprepared)
    finally:
        unprepared.discard()

    preparation = cake_api.CakeFmhaRequestOrderedCapture([plan])
    for binding in bindings:
        preparation.prepare_workspace(plan, binding["workspace_buffer"])
        preparation.prepare_workspace(plan, binding["workspace_buffer"])
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            with pytest.raises(RuntimeError, match="outside CUDA Graph capture"):
                preparation.prepare_workspace(plan, bindings[0]["workspace_buffer"])
            for binding in bindings:
                invoke(binding, preparation)
        preparation.finalize()
    except BaseException:
        preparation.discard()
        raise
    for poison in (0xA5, 0x5A):
        # This emulates another sequential operator's unrestricted scratch
        # writes. Explicit counters remain the same actual zero-reset storage.
        for binding in bindings:
            binding["workspace_buffer"].fill_(poison)
            binding["out"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for binding, reference_out in zip(bindings, expected, strict=False):
            torch.testing.assert_close(
                binding["out"], reference_out, atol=0.1, rtol=0.1
            )
            assert (
                torch.count_nonzero(binding["multi_ctas_kv_counter_buffer"]).item() == 0
            )


@pytest.mark.parametrize(
    ("batch", "q_len", "parts", "grid"),
    (
        (1, 1, 256, (1, 1, 152)),
        (2, 1, 128, (1, 1, 152)),
        (3, 1, 64, (1, 1, 150)),
        (5, 1, 32, (1, 1, 150)),
        (6, 1, 32, (1, 1, 150)),
        (7, 1, 32, (1, 1, 147)),
        (8, 1, 32, (1, 1, 152)),
        (9, 1, 32, (1, 1, 144)),
        (10, 1, 16, (1, 1, 150)),
        (27, 1, 16, (1, 1, 135)),
        (32, 1, 16, (1, 1, 128)),
        (64, 1, 16, (1, 1, 128)),
        (128, 1, 16, (1, 1, 128)),
        (256, 1, 16, (1, 1, 256)),
        (1, 6, 32, (6, 1, 25)),
        (2, 6, 16, (6, 1, 24)),
        (8, 6, 16, (6, 1, 24)),
        (27, 6, 16, (6, 1, 27)),
    ),
)
def test_runtime_length_grid_preserves_capacity_boundaries(
    batch, q_len, parts, grid
) -> None:
    plan = cake_api.plan_cake_fmha_request_ordered_paged_decode([q_len] * batch, q_len)
    assert plan.workspace_parts == parts
    assert plan.grid == grid
    assert cake_api._is_authenticated_request_ordered_plan(plan)
    assert not cake_api._is_authenticated_request_ordered_plan(
        dataclasses.replace(plan, grid=(grid[0], grid[1], grid[2] + batch))
    )


@pytest.mark.parametrize("q_len", (1, 6))
@pytest.mark.parametrize("small_external_counters", (False, True))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_runtime_length_workspace_reuse_preserves_internal_counters(
    q_len: int,
    small_external_counters: bool,
) -> None:
    """A smaller plan's real partial writes cannot corrupt a larger plan's counters."""
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("request-ordered Cake FMHA requires SM103")
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip("request-ordered Cake FMHA requires a 152-SM device")

    device = torch.device("cuda")
    length = 2049
    page_slots = (length + 63) // 64
    plans = {
        batch: flashinfer.plan_cake_fmha_request_ordered_paged_decode(
            [length] * batch,
            q_len,
        )
        for batch in (64, 1)
    }
    workspace = torch.full(
        (max(plan.workspace_size_bytes for plan in plans.values()),),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    qk = torch.tensor([math.log2(math.e) / 16], dtype=torch.float32, device=device)
    pv = torch.ones(1, dtype=torch.float32, device=device)
    inputs = {}
    for batch, plan in plans.items():
        query = torch.zeros(
            (batch * q_len, 8, 256), dtype=torch.bfloat16, device=device
        )
        key = torch.zeros((batch * page_slots, 1, 64, 256), device=device).to(
            torch.float8_e4m3fn
        )
        value = torch.ones_like(key, dtype=torch.float32).to(torch.float8_e4m3fn)
        inputs[batch] = dict(
            query=query,
            kv_cache=(key, value),
            out=torch.empty_like(query),
            block_tables=torch.arange(
                batch * page_slots, dtype=torch.int32, device=device
            ).view(batch, page_slots),
            seq_lens=torch.full((batch,), length, dtype=torch.int32, device=device),
            request_order=torch.arange(batch, dtype=torch.int32, device=device),
            request_order_plan=plan,
        )

    def invoke(batch, counters=None):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            **inputs[batch],
            workspace_buffer=workspace,
            max_seq_len=length,
            bmm1_scale_log2=qk,
            bmm2_scale=pv,
            backend="cake",
            enable_pdl=True,
            q_len_per_req=q_len,
            multi_ctas_kv_counter_buffer=counters,
        )
        torch.testing.assert_close(
            inputs[batch]["out"],
            torch.ones_like(inputs[batch]["out"]),
            atol=0.1,
            rtol=0.1,
        )

    # Poisoned storage must be initialized before the first use. Inspect the
    # full reserved counter region, including capacity outside the active B64.
    counter_begin, _, partial_begin, _, _ = cake_api._runtime_length_workspace_layout(
        plans[64]
    )
    reserved_counters = workspace[counter_begin:partial_begin].view(torch.int32)
    invoke(64)
    assert torch.count_nonzero(reserved_counters).item() == 0

    # Clear only the small plan's partial O region, so nonzero values after its
    # actual FMHA invocation demonstrate that this fixture exercises partials.
    _, _, small_partial_begin, small_partial_end, _ = (
        cake_api._runtime_length_workspace_layout(plans[1])
    )
    small_partials = workspace[small_partial_begin:small_partial_end].view(
        torch.bfloat16
    )
    small_partials.zero_()
    external = (
        torch.zeros(q_len, dtype=torch.int32, device=device)
        if small_external_counters
        else None
    )
    invoke(1, external)
    assert torch.count_nonzero(small_partials).item() > 0
    if external is not None:
        assert torch.count_nonzero(external).item() == 0
    # Check before launching A again so the regression reports corrupted
    # counters directly instead of entering a final-owner wait with bad state.
    assert torch.count_nonzero(reserved_counters).item() == 0
    invoke(64)
    assert torch.count_nonzero(reserved_counters).item() == 0


def test_decode_api_exposes_order_pointer_and_host_plan_at_the_end() -> None:
    parameters = list(
        inspect.signature(
            flashinfer.decode.trtllm_batch_decode_with_kv_cache
        ).parameters
    )
    assert parameters[-3:] == [
        "request_order",
        "request_order_plan",
        "request_order_capture",
    ]


def test_request_order_requires_explicit_cake_backend() -> None:
    tensor = torch.empty(1)
    with pytest.raises(ValueError, match="explicit backend='cake'"):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            1,
            request_order=torch.empty(1, dtype=torch.int32),
        )


def test_host_plan_requires_device_order_tensor() -> None:
    tensor = torch.empty(1)
    plan = cake_api.CakeFmhaRequestOrderedDecodePlan(
        module_name="cake_fmha_request_ordered_paged_decode_test",
        batch_size=1,
        q_len=1,
        workspace_parts=1,
        grid=(1, 1, 1),
        total_tiles=1,
        write_lse=False,
    )
    with pytest.raises(ValueError, match="requires a device request_order"):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            1,
            backend="cake",
            request_order_plan=plan,
        )


@pytest.mark.parametrize("q_len", (1, 2, 3, 4, 5, 6, 8, 17, 65))
@pytest.mark.parametrize(("num_q_heads", "num_kv_heads"), ((8, 1), (32, 2)))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_request_ordered_capture_prepares_actual_producer_q(
    q_len: int, num_q_heads: int, num_kv_heads: int
) -> None:
    """Two live graphs retain their own real captured-Q descriptor bindings."""
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("request-ordered Cake FMHA requires SM103")
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip("request-ordered Cake FMHA requires a 152-SM device")

    batch = 2
    generic = q_len in (1, 6) and (num_q_heads, num_kv_heads) == (8, 1)
    page_slots = 3 if generic else 4
    lengths = (73, 137)
    current_lengths = lengths
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(48320 + q_len)
    base = torch.randn(
        batch * q_len,
        num_q_heads,
        256,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    key, value = (
        torch.randn(
            batch * page_slots,
            num_kv_heads,
            64,
            256,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).to(torch.float8_e4m3fn)
        for _ in range(2)
    )
    tables = torch.arange(batch * page_slots, dtype=torch.int32, device=device).view(
        batch, page_slots
    )
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    order = torch.arange(batch, dtype=torch.int32, device=device)
    qk = torch.tensor([math.log2(math.e) / 16], dtype=torch.float32, device=device)
    pv = torch.ones(1, dtype=torch.float32, device=device)
    plan = flashinfer.plan_cake_fmha_request_ordered_paged_decode(
        lengths, q_len, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads
    )

    def invoke(query, workspace, output, preparation=None):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=(key, value),
            workspace_buffer=workspace,
            out=output,
            block_tables=tables,
            seq_lens=seq_lens,
            max_seq_len=max(lengths),
            bmm1_scale_log2=qk,
            bmm2_scale=pv,
            backend="cake",
            enable_pdl=True,
            q_len_per_req=q_len,
            request_order=order,
            request_order_plan=plan,
            request_order_capture=preparation,
        )

    def reference(query):
        rows = []
        for request, length in enumerate(current_lengths):
            k = key[tables[request].long()].float().permute(0, 2, 1, 3)
            v = value[tables[request].long()].float().permute(0, 2, 1, 3)
            k = k.reshape(-1, num_kv_heads, 256)[:length]
            v = v.reshape(-1, num_kv_heads, 256)[:length]
            head_indices = torch.arange(num_q_heads, device=device) // (
                num_q_heads // num_kv_heads
            )
            k, v = k[:, head_indices], v[:, head_indices]
            q = query.view(batch, q_len, num_q_heads, 256)[request].float()
            scores = torch.einsum("qhd,khd->hqk", q, k) / 16
            visible = length - q_len + torch.arange(q_len, device=device) + 1
            mask = torch.arange(length, device=device)[None, :] < visible[:, None]
            probabilities = scores.masked_fill(~mask[None, :, :], -torch.inf).softmax(
                -1
            )
            rows.append(torch.einsum("hqk,khd->qhd", probabilities, v))
        return torch.cat(rows).to(torch.bfloat16)

    # Keep the eager producer result alive to force an actual new Q allocation
    # in capture. Module/resource warming still follows the ordinary public API.
    warm_query = base * 1.0
    warm_workspace = torch.empty(
        plan.workspace_size_bytes, dtype=torch.uint8, device=device
    )
    warm_output = torch.empty_like(base)
    invoke(warm_query, warm_workspace, warm_output)
    torch.cuda.synchronize()

    # A second Q cannot overwrite a pending descriptor slot in the same graph.
    # Discard this unpublished graph, then prove the workspace claim is released
    # by preparing and running an ordinary launch through that same allocation.
    rejected_workspace = torch.empty(
        plan.workspace_size_bytes, dtype=torch.uint8, device=device
    )
    rejected_output = torch.empty_like(base)
    if generic:
        invoke(warm_query, rejected_workspace, rejected_output)
    rejected_preparation = cake_api.CakeFmhaRequestOrderedCapture([plan])
    rejected_preparation.prepare_workspace(plan, rejected_workspace)
    rejected_graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(rejected_graph):
            first_query = base * 3.0
            second_query = base * 4.0
            invoke(
                first_query, rejected_workspace, rejected_output, rejected_preparation
            )
            with pytest.raises(RuntimeError, match="different tensor bindings"):
                invoke(
                    second_query,
                    rejected_workspace,
                    rejected_output,
                    rejected_preparation,
                )
    finally:
        rejected_preparation.discard()
    assert not rejected_preparation.finalized
    with pytest.raises(RuntimeError, match="already finished"):
        rejected_preparation.finalize()
    del rejected_graph, first_query, second_query
    invoke(warm_query, rejected_workspace, rejected_output)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        rejected_output, reference(warm_query), atol=0.1, rtol=0.1
    )
    del rejected_workspace, rejected_output, rejected_preparation

    graphs = []
    for factor in (1.0, 2.0):
        workspace = torch.empty(
            plan.workspace_size_bytes, dtype=torch.uint8, device=device
        )
        workspace_ref = weakref.ref(workspace)
        output = torch.empty_like(base)
        if generic:
            invoke(warm_query, workspace, output)
        preparation = cake_api.CakeFmhaRequestOrderedCapture([plan])
        preparation.prepare_workspace(plan, workspace)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                produced_query = base * factor
                captured_query_ptr = produced_query.data_ptr()
                invoke(produced_query, workspace, output, preparation)
                # This reversible lifecycle error must leave the recorded
                # transaction available for proper finalization after capture.
                with pytest.raises(RuntimeError, match="after capture ends"):
                    preparation.finalize()
            preparation.finalize()
        except BaseException:
            preparation.discard()
            raise
        assert captured_query_ptr != warm_query.data_ptr()
        assert preparation.finalized
        with pytest.raises(RuntimeError, match="already finished"):
            preparation.finalize()
        del produced_query, workspace
        assert workspace_ref() is not None
        graphs.append((graph, preparation, output, factor, workspace_ref))

    # Both graphs replay after both descriptor sets have been finalized. Their
    # workspaces live through their preparation objects, not a shared scratch slot.
    for replay_index, permutation in enumerate(((1, 0), (0, 1))):
        if generic:
            current_lengths = (q_len + 1, 129) if replay_index == 0 else lengths
            seq_lens.copy_(
                torch.tensor(current_lengths, dtype=torch.int32, device=device)
            )
        order.copy_(torch.tensor(permutation, dtype=torch.int32, device=device))
        base.mul_(0.75)
        for graph, preparation, output, factor, workspace_ref in graphs:
            assert preparation.finalized and workspace_ref() is not None
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output, reference(base * factor), atol=0.1, rtol=0.1
            )


@pytest.mark.parametrize(
    ("uses_shared_paged_kv_idx", "q_len"),
    ((True, 1), (False, 6)),
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_request_ordered_public_api_graph_replays_device_permutations(
    uses_shared_paged_kv_idx: bool,
    q_len: int,
) -> None:
    _check_request_ordered_graph_permutations(uses_shared_paged_kv_idx, q_len, 8, 1)


@pytest.mark.parametrize("q_len", (2, 8))
@pytest.mark.parametrize("uses_shared_paged_kv_idx", (True, False))
@pytest.mark.parametrize(("num_q_heads", "num_kv_heads"), ((8, 1), (32, 2)))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_request_ordered_runtime_q_graph_permutations(
    uses_shared_paged_kv_idx: bool,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    _check_request_ordered_graph_permutations(
        uses_shared_paged_kv_idx, q_len, num_q_heads, num_kv_heads
    )


@pytest.mark.parametrize("uses_shared_paged_kv_idx", (True, False))
@pytest.mark.parametrize(
    ("batch_size", "q_len", "num_kv_splits", "write_lse"),
    (
        pytest.param(1, 1, 4, True, id="b1-q1-s4"),
        pytest.param(8, 1, 4, True, id="b8-q1-s4"),
        pytest.param(1, 6, 2, True, id="b1-q6-s2"),
        pytest.param(8, 6, 2, True, id="b8-q6-s2"),
        pytest.param(1, 6, 6, False, id="b1-q6-fused-s6"),
        pytest.param(8, 6, 6, False, id="b8-q6-fused-s6"),
        pytest.param(32, 6, 6, False, id="b32-q6-fused-s6"),
    ),
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_request_ordered_explicit_split_graph_permutations(
    uses_shared_paged_kv_idx: bool,
    batch_size: int,
    q_len: int,
    num_kv_splits: int,
    write_lse: bool,
) -> None:
    """Native split/combine follows live producer Q and device metadata."""
    _check_request_ordered_graph_permutations(
        uses_shared_paged_kv_idx,
        q_len,
        32,
        2,
        batch_size=batch_size,
        num_kv_splits=num_kv_splits,
        write_lse=write_lse,
    )


def test_b1_q6_s76_plan_and_exact_scratch() -> None:
    kwargs = dict(num_q_heads=32, num_kv_heads=2)
    default = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        (136193,), 6, **kwargs
    )
    assert default.workspace_parts == 1
    wide = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        (136193,), 6, num_kv_splits=6, **kwargs
    )
    assert wide.module_name.endswith("6c47e3e71cb98a8c1ee1")
    plan = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        (136193,), 6, num_kv_splits=76, **kwargs
    )
    assert plan.grid == (76, 2, 1) and plan.total_tiles == 1
    assert plan.write_lse is False and cake_api._is_authenticated_request_ordered_plan(
        plan
    )
    partial_o_bytes, partial_stats_bytes = (
        cake_api._request_ordered_partial_workspace_bytes(plan)
    )
    assert (partial_o_bytes, partial_stats_bytes) == (9961472, 155648)
    assert (32 << 20) + partial_o_bytes + partial_stats_bytes == 43671552
    with pytest.raises(ValueError, match="split"):
        cake_api.plan_cake_fmha_request_ordered_paged_decode(
            (136193,), 6, num_kv_splits=76, write_lse=True, **kwargs
        )
    with pytest.raises(ValueError, match="split"):
        cake_api.plan_cake_fmha_request_ordered_paged_decode(
            (136193, 136193), 6, num_kv_splits=76, **kwargs
        )


@pytest.mark.parametrize("uses_shared_paged_kv_idx", (True, False))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_b1_q6_s76_exact_scratch_graph(uses_shared_paged_kv_idx: bool) -> None:
    _check_request_ordered_graph_permutations(
        uses_shared_paged_kv_idx,
        6,
        32,
        2,
        batch_size=1,
        num_kv_splits=76,
        write_lse=False,
        candidate_workspace_bytes=43671552,
    )


def _check_request_ordered_graph_permutations(
    uses_shared_paged_kv_idx: bool,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    batch_size: int = 4,
    num_kv_splits: int | None = None,
    write_lse: bool = True,
    candidate_workspace_bytes: int | None = None,
) -> None:
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("request-ordered Cake FMHA requires SM103")
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip("request-ordered Cake FMHA requires a 152-SM device")

    device = torch.device("cuda")
    if num_kv_splits is None:
        seq_lens_host = (65, 129, 257, 385)
    elif batch_size == 1:
        seq_lens_host = (1537,)
    else:
        length_pattern = (65, 129, 257, 385, 513, 769, 1025, 1537)
        seq_lens_host = tuple(length_pattern[i % 8] for i in range(batch_size))
    page_slots = 4 * math.ceil(max(seq_lens_host) / 256)
    num_pages = batch_size * page_slots
    generator = torch.Generator(device=device).manual_seed(4832 + q_len)
    query = torch.randn(
        (batch_size * q_len, num_q_heads, 256),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    key = torch.randn(
        (num_pages, num_kv_heads, 64, 256),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    value = torch.randn(
        (num_pages, num_kv_heads, 64, 256),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    shared_tables = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        batch_size, page_slots
    )
    if uses_shared_paged_kv_idx:
        block_tables = shared_tables
    else:
        value_tables = (
            shared_tables.roll(1, dims=-1)
            if batch_size == 1
            else shared_tables.flip(0).contiguous()
        )
        block_tables = torch.stack((shared_tables, value_tables), dim=1)
    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device=device)
    bmm1_scale = 1.0 / math.sqrt(256)
    bmm1_scale_log2 = torch.tensor(
        [bmm1_scale * math.log2(math.e)], dtype=torch.float32, device=device
    )
    bmm2_scale = torch.ones(1, dtype=torch.float32, device=device)
    reference_workspace = torch.empty(64 << 20, dtype=torch.uint8, device=device)
    candidate_workspace = (
        torch.empty_like(reference_workspace)
        if candidate_workspace_bytes is None
        else torch.empty(candidate_workspace_bytes, dtype=torch.uint8, device=device)
    )
    reference_out = torch.empty_like(query)
    reference_lse = torch.empty(query.shape[:-1], dtype=torch.float32, device=device)
    candidate_out = torch.empty_like(query)
    candidate_lse = torch.empty_like(reference_lse)

    common = {
        "query": query,
        "kv_cache": (key, value),
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "max_seq_len": max(seq_lens_host),
        "bmm1_scale": bmm1_scale,
        "bmm2_scale": bmm2_scale,
        "kv_layout": "HND",
        "enable_pdl": True,
        "q_len_per_req": q_len,
        "uses_shared_paged_kv_idx": uses_shared_paged_kv_idx,
        "return_lse": True,
        "bmm1_scale_log2": bmm1_scale_log2,
    }
    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        workspace_buffer=reference_workspace,
        out=reference_out,
        lse=reference_lse,
        backend="trtllm-gen",
        **common,
    )
    plan = flashinfer.plan_cake_fmha_request_ordered_paged_decode(
        seq_lens_host,
        q_len,
        write_lse=write_lse,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_kv_splits=num_kv_splits,
    )
    request_order = torch.arange(batch_size, dtype=torch.int32, device=device)
    candidate_counter = (
        torch.zeros(
            batch_size * q_len * (num_q_heads // 8),
            dtype=torch.int32,
            device=device,
        )
        if plan.workspace_parts > 1
        else None
    )

    def run_candidate(candidate_query=query, preparation=None) -> None:
        candidate_common = dict(common, query=candidate_query, return_lse=write_lse)
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            workspace_buffer=candidate_workspace,
            out=candidate_out,
            lse=candidate_lse if write_lse else None,
            backend="cake",
            request_order=request_order,
            request_order_plan=plan,
            request_order_capture=preparation,
            multi_ctas_kv_counter_buffer=candidate_counter,
            **candidate_common,
        )

    run_candidate()
    torch.cuda.synchronize()
    torch.testing.assert_close(candidate_out, reference_out, atol=0.1, rtol=0.1)
    if write_lse:
        torch.testing.assert_close(candidate_lse, reference_lse, atol=1e-2, rtol=1e-2)
    graph = torch.cuda.CUDAGraph()
    if num_kv_splits is None and not plan.runtime_length_scheduler:
        with torch.cuda.graph(graph):
            run_candidate()
        permutations = ((3, 1, 0, 2), (1, 3, 2, 0))
    else:
        preparation = cake_api.CakeFmhaRequestOrderedCapture([plan])
        preparation.prepare_workspace(plan, candidate_workspace)
        try:
            with torch.cuda.graph(graph):
                produced_query = query * 1.0
                run_candidate(produced_query, preparation)
            preparation.finalize()
        except BaseException:
            preparation.discard()
            raise
        permutations = (
            tuple(reversed(range(batch_size))),
            tuple((3 * i + 1) % batch_size for i in range(batch_size)),
        )

    for replay_index, permutation in enumerate(permutations):
        if num_kv_splits is not None or plan.runtime_length_scheduler:
            query.mul_(0.875)
            seq_lens.copy_(
                torch.tensor(
                    [length - 1 - 16 * replay_index for length in seq_lens_host],
                    dtype=torch.int32,
                    device=device,
                )
            )
            block_tables.copy_(block_tables.roll(1, dims=-1))
        physical_order = torch.tensor(permutation, dtype=torch.int64, device=device)
        inverse_order = torch.argsort(physical_order)
        physical_inputs = dict(common)
        physical_inputs.update(
            query=query.view(batch_size, q_len, num_q_heads, 256)[
                physical_order
            ].flatten(0, 1),
            block_tables=block_tables[physical_order],
            seq_lens=seq_lens[physical_order],
        )
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            workspace_buffer=reference_workspace,
            out=reference_out,
            lse=reference_lse,
            backend="trtllm-gen",
            **physical_inputs,
        )
        expected_out = reference_out.view(batch_size, q_len, num_q_heads, 256)[
            inverse_order
        ].flatten(0, 1)
        expected_lse = reference_lse.view(batch_size, q_len, num_q_heads)[
            inverse_order
        ].flatten(0, 1)
        request_order.copy_(physical_order)
        candidate_out.fill_(float("nan"))
        candidate_lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        if candidate_counter is not None:
            assert torch.count_nonzero(candidate_counter).item() == 0
        torch.testing.assert_close(
            candidate_out,
            expected_out,
            atol=0.1,
            rtol=0.1,
        )
        if write_lse:
            torch.testing.assert_close(
                candidate_lse,
                expected_lse,
                atol=1e-2,
                rtol=1e-2,
            )


@pytest.mark.parametrize("q_len", (2, 8, 17, 257))
@pytest.mark.parametrize("write_lse", (False, True))
@pytest.mark.parametrize(("num_q_heads", "num_kv_heads"), ((8, 1), (32, 2)))
def test_runtime_q_plan_uses_authenticated_generated_binding(
    q_len: int, write_lse: bool, num_q_heads: int, num_kv_heads: int
) -> None:
    from flashinfer.jit.cake_fmha_request_ordered import (
        get_cake_fmha_request_ordered_module_spec,
    )

    plan = flashinfer.plan_cake_fmha_request_ordered_paged_decode(
        (513, 769),
        q_len,
        write_lse=write_lse,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )
    assert "_runtime_q_" in plan.module_name
    assert plan.grid == (q_len, num_q_heads // 8, 2)
    assert plan.total_tiles == 2 * q_len * (num_q_heads // 8)
    assert plan.workspace_parts == 1
    assert cake_api._is_authenticated_request_ordered_plan(plan)
    assert not cake_api._is_authenticated_request_ordered_plan(
        dataclasses.replace(plan, grid=(q_len + 1, num_q_heads // 8, 2))
    )
    spec = get_cake_fmha_request_ordered_module_spec(plan.module_name)
    assert spec.tma_workspace_bytes == 384


@pytest.mark.parametrize("batch_size", (1, 8, 32, 160, 192, 224, 256))
def test_fused_q6_plan_uses_authenticated_physical_kv_groups(batch_size: int) -> None:
    plan = flashinfer.plan_cake_fmha_request_ordered_paged_decode(
        (513,) * batch_size,
        6,
        num_q_heads=32,
        num_kv_heads=2,
        num_kv_splits=6,
    )
    assert plan.grid == (6, 2, batch_size)
    assert plan.workspace_parts == 6 and plan.total_tiles == 1
    assert cake_api._is_authenticated_request_ordered_plan(plan)
    assert not cake_api._is_authenticated_request_ordered_plan(
        dataclasses.replace(plan, grid=(6, 4, batch_size))
    )
    with pytest.raises(ValueError, match="no exported dynamic request-order schedule"):
        flashinfer.plan_cake_fmha_request_ordered_paged_decode(
            (513,) * batch_size,
            6,
            write_lse=True,
            num_q_heads=32,
            num_kv_heads=2,
            num_kv_splits=6,
        )


@pytest.mark.parametrize("q_len", (0, -1, 2.5, True))
def test_request_order_rejects_invalid_q_before_tensor_processing(q_len) -> None:
    tensor = torch.empty(1)
    with pytest.raises(ValueError, match="uniform positive integer q_len_per_req"):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            1,
            backend="cake",
            request_order=torch.empty(1, dtype=torch.int32),
            q_len_per_req=q_len,
        )
