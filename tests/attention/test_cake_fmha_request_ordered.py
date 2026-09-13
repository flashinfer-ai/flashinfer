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


def _route(
    *,
    module: str,
    route_slug: str,
    batch: int,
    q_len: int,
    kv_lens: tuple[int, ...],
    request_order_case: str,
    write_lse: bool,
    grid: tuple[int, int, int],
    total_tiles: int,
    workspace_parts: int = 1,
    segmented_clc: bool = False,
) -> dict:
    return {
        "shape": f"{route_slug}_{batch}",
        "module_name": module,
        "args": {
            "q_lens": [q_len] * batch,
            "kv_lens": list(kv_lens),
            "real_batch_size": batch,
            "request_order_case": request_order_case,
            "provide_lse": write_lse,
        },
        "build_plan": {
            "route_slug": route_slug,
            "q_len": q_len,
            "ordered": True,
            "num_split": 2 if segmented_clc else 1,
            "workspace_parts": workspace_parts,
            "write_lse": write_lse,
            "segmented_clc": segmented_clc,
            "static_one_tile": False,
            "grid": list(grid),
            "total_tiles": total_tiles,
        },
    }


def _manifest(num_q_heads: int = 8, num_kv_heads: int = 1) -> dict:
    assert (num_q_heads, num_kv_heads) == (8, 1)
    exact_lengths = (8193, 57345, 73729, 81921)
    return {
        "routes": [
            _route(
                module="cake_fmha_request_ordered_paged_decode_exact",
                route_slug="two_wave_q1",
                batch=4,
                q_len=1,
                kv_lens=exact_lengths,
                request_order_case="length_desc",
                write_lse=False,
                grid=(1, 1, 4),
                total_tiles=8,
                workspace_parts=16,
                segmented_clc=True,
            ),
            _route(
                module="cake_fmha_request_ordered_paged_decode_fallback_q1",
                route_slug="fallback_q1",
                batch=1,
                q_len=1,
                kv_lens=(8193,),
                request_order_case="identity",
                write_lse=False,
                grid=(1, 1, 1),
                total_tiles=1,
            ),
            _route(
                module="cake_fmha_request_ordered_paged_decode_fallback_q1_lse",
                route_slug="fallback_q1_ordered_s1_lse",
                batch=1,
                q_len=1,
                kv_lens=(8193,),
                request_order_case="identity",
                write_lse=True,
                grid=(1, 1, 1),
                total_tiles=1,
            ),
        ]
    }


def test_request_order_plan_selects_exact_exported_schedule(monkeypatch) -> None:
    monkeypatch.setattr(cake_api, "get_cake_fmha_request_ordered_manifest", _manifest)
    plan = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        (8193, 57345, 73729, 81921),
        1,
    )
    assert plan.module_name.endswith("_exact")
    assert plan.grid == (1, 1, 4)
    assert plan.total_tiles == 8
    assert plan.workspace_parts == 16
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.total_tiles = 4


def test_request_order_plan_uses_graph_safe_fallback_for_other_lengths(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cake_api, "get_cake_fmha_request_ordered_manifest", _manifest)
    plan = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        (64, 128, 192),
        1,
    )
    assert plan.module_name.endswith("_fallback_q1")
    assert plan.batch_size == 3
    assert plan.grid == (1, 1, 3)
    assert plan.total_tiles == 3
    assert plan.workspace_parts == 1


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


@pytest.mark.parametrize("q_len", (1, 6))
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

    batch, page_slots = 2, 4
    lengths = (73, 137)
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
    assert plan.workspace_parts == 1

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
        for request, length in enumerate(lengths):
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
    warm_workspace = torch.empty(388, dtype=torch.uint8, device=device)
    warm_output = torch.empty_like(base)
    invoke(warm_query, warm_workspace, warm_output)
    torch.cuda.synchronize()

    # A second Q cannot overwrite a pending descriptor slot in the same graph.
    # Discard this unpublished graph, then prove the workspace claim is released
    # by preparing and running an ordinary launch through that same allocation.
    rejected_workspace = torch.empty(388, dtype=torch.uint8, device=device)
    rejected_output = torch.empty_like(base)
    rejected_preparation = cake_api.CakeFmhaRequestOrderedCapture([plan])
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
        workspace = torch.empty(388, dtype=torch.uint8, device=device)
        workspace_ref = weakref.ref(workspace)
        output = torch.empty_like(base)
        preparation = cake_api.CakeFmhaRequestOrderedCapture([plan])
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
    for permutation in ((1, 0), (0, 1)):
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
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("request-ordered Cake FMHA requires SM103")
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip("request-ordered Cake FMHA requires a 152-SM device")

    device = torch.device("cuda")
    batch_size, page_slots = 4, 8
    seq_lens_host = (65, 129, 257, 385)
    num_pages = batch_size * page_slots
    generator = torch.Generator(device=device).manual_seed(4832 + q_len)
    query = torch.randn(
        (batch_size * q_len, 8, 256),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    key = torch.randn(
        (num_pages, 1, 64, 256),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    value = torch.randn(
        (num_pages, 1, 64, 256),
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
        value_tables = shared_tables.flip(0).contiguous()
        block_tables = torch.stack((shared_tables, value_tables), dim=1)
    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device=device)
    bmm1_scale = 1.0 / math.sqrt(256)
    bmm1_scale_log2 = torch.tensor(
        [bmm1_scale * math.log2(math.e)], dtype=torch.float32, device=device
    )
    bmm2_scale = torch.ones(1, dtype=torch.float32, device=device)
    reference_workspace = torch.empty(64 << 20, dtype=torch.uint8, device=device)
    candidate_workspace = torch.empty_like(reference_workspace)
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
        write_lse=True,
    )
    request_order = torch.arange(batch_size, dtype=torch.int32, device=device)

    def run_candidate() -> None:
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            workspace_buffer=candidate_workspace,
            out=candidate_out,
            lse=candidate_lse,
            backend="cake",
            request_order=request_order,
            request_order_plan=plan,
            **common,
        )

    run_candidate()
    torch.cuda.synchronize()
    torch.testing.assert_close(candidate_out, reference_out, atol=0.1, rtol=0.1)
    torch.testing.assert_close(candidate_lse, reference_lse, atol=1e-2, rtol=1e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_candidate()

    for permutation in ((3, 1, 0, 2), (1, 3, 2, 0)):
        physical_order = torch.tensor(permutation, dtype=torch.int64, device=device)
        inverse_order = torch.argsort(physical_order)
        physical_inputs = dict(common)
        physical_inputs.update(
            query=query.view(batch_size, q_len, 8, 256)[physical_order].flatten(0, 1),
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
        expected_out = reference_out.view(batch_size, q_len, 8, 256)[
            inverse_order
        ].flatten(0, 1)
        expected_lse = reference_lse.view(batch_size, q_len, 8)[inverse_order].flatten(
            0, 1
        )
        request_order.copy_(physical_order)
        candidate_out.fill_(float("nan"))
        candidate_lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            candidate_out,
            expected_out,
            atol=0.1,
            rtol=0.1,
        )
        torch.testing.assert_close(
            candidate_lse,
            expected_lse,
            atol=1e-2,
            rtol=1e-2,
        )
