"""Contract and GPU tests for request-ordered Cake FMHA decode."""

from __future__ import annotations

import dataclasses
import inspect
import json
import math

import flashinfer
import flashinfer.cake_fmha as cake_api
import flashinfer.cake_fmha_request_ordered as request_ordered_api
import flashinfer.jit.cake_fmha_request_ordered as request_ordered_jit
import pytest
import torch


_EXACT_MODULE = "cake_fmha_request_ordered_paged_decode_exact"
_FALLBACK_Q1 = "cake_fmha_request_ordered_paged_decode_fallback_q1"
_FALLBACK_Q1_LSE = "cake_fmha_request_ordered_paged_decode_fallback_q1_lse"
_FALLBACK_Q1_SPLIT4_LSE = "cake_fmha_request_ordered_paged_decode_fallback_q1_s4_lse"
_EXACT_PERIOD = (8193, 57345, 73729, 81921)


def _module(
    *,
    name: str,
    kind: str,
    route_slug: str,
    q_len: int,
    num_split: int,
    write_lse: bool,
) -> dict:
    return {
        "name": name,
        "kind": kind,
        "route_slug": route_slug,
        "q_len": q_len,
        "num_split": num_split,
        "write_lse": write_lse,
        "defines": {"Q_LEN": q_len, "NUM_SPLIT": num_split, "WRITE_LSE": int(write_lse)}
        if kind == "persistent"
        else {},
    }


def _exact_route(
    *,
    module: str,
    shape: str,
    period: tuple[int, ...],
    count: int,
    q_len: int,
    real_batch_size: int,
    request_order_case: str,
    write_lse: bool,
    grid: tuple[int, int, int],
    total_tiles: int,
    workspace_parts: int,
) -> dict:
    return {
        "shape": shape,
        "module": module,
        "q_len": q_len,
        "kv_lens": {"period": list(period), "count": count},
        "real_batch_size": real_batch_size,
        "request_order_case": request_order_case,
        "write_lse": write_lse,
        "grid": list(grid),
        "total_tiles": total_tiles,
        "workspace_parts": workspace_parts,
    }


def _manifest() -> dict:
    """Module table in the shipped manifest schema: programs plus exact exported routes."""

    return {
        "modules": [
            _module(
                name=_EXACT_MODULE,
                kind="two_wave",
                route_slug="two_wave_q1",
                q_len=1,
                num_split=2,
                write_lse=False,
            ),
            _module(
                name=_FALLBACK_Q1,
                kind="persistent",
                route_slug="fallback_q1",
                q_len=1,
                num_split=1,
                write_lse=False,
            ),
            _module(
                name=_FALLBACK_Q1_LSE,
                kind="persistent",
                route_slug="fallback_q1_ordered_s1_lse",
                q_len=1,
                num_split=1,
                write_lse=True,
            ),
            _module(
                name=_FALLBACK_Q1_SPLIT4_LSE,
                kind="persistent",
                route_slug="fallback_q1_ordered_s4_lse",
                q_len=1,
                num_split=4,
                write_lse=True,
            ),
        ],
        "exact_routes": [
            _exact_route(
                module=_EXACT_MODULE,
                shape="perf_ragged_b8_q1",
                period=_EXACT_PERIOD,
                count=8,
                q_len=1,
                real_batch_size=8,
                request_order_case="length_desc",
                write_lse=False,
                grid=(1, 1, 8),
                total_tiles=16,
                workspace_parts=16,
            ),
            _exact_route(
                module=_FALLBACK_Q1_SPLIT4_LSE,
                shape="correctness_split4_q1",
                period=(897, 1025, 1153, 1281),
                count=4,
                q_len=1,
                real_batch_size=4,
                request_order_case="random",
                write_lse=True,
                grid=(1, 4, 4),
                total_tiles=16,
                workspace_parts=4,
            ),
        ],
    }


def test_request_order_plan_selects_exact_exported_schedule(monkeypatch) -> None:
    monkeypatch.setattr(
        request_ordered_api, "get_cake_fmha_request_ordered_manifest", _manifest
    )
    plan = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        _EXACT_PERIOD * 2,
        1,
    )
    assert plan.module_name == _EXACT_MODULE
    assert plan.batch_size == 8
    assert plan.grid == (1, 1, 8)
    assert plan.total_tiles == 16
    assert plan.workspace_parts == 16
    assert plan.write_lse is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.total_tiles = 4


def test_request_order_plan_uses_graph_safe_fallback_for_other_lengths(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        request_ordered_api, "get_cake_fmha_request_ordered_manifest", _manifest
    )
    plan = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        (64, 128, 192),
        1,
    )
    assert plan.module_name == _FALLBACK_Q1
    assert plan.batch_size == 3
    assert plan.grid == (1, 1, 3)
    assert plan.total_tiles == 3
    assert plan.workspace_parts == 1

    # One exported period with a different request count is not an exact route.
    truncated = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        _EXACT_PERIOD,
        1,
    )
    assert truncated.module_name == _FALLBACK_Q1
    assert truncated.grid == (1, 1, 4)

    # The single-split LSE program is the fallback; the split-4 module is exact only.
    lse_plan = cake_api.plan_cake_fmha_request_ordered_paged_decode(
        (64, 128, 192),
        1,
        write_lse=True,
    )
    assert lse_plan.module_name == _FALLBACK_Q1_LSE
    assert lse_plan.workspace_parts == 1
    assert lse_plan.write_lse is True


def test_decode_api_exposes_order_pointer_and_host_plan_at_the_end() -> None:
    parameters = list(
        inspect.signature(
            flashinfer.decode.trtllm_batch_decode_with_kv_cache
        ).parameters
    )
    assert parameters[-2:] == ["request_order", "request_order_plan"]


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


def _shipped_manifest_copy(tmp_path) -> dict:
    """The shipped module table, with empty stand-in sources under ``tmp_path``."""

    root = request_ordered_jit._source_root()
    payload = json.loads((root / request_ordered_jit._MANIFEST_NAME).read_text())
    for module in payload["modules"]:
        for key in ("device_path", "binding_path"):
            stand_in = tmp_path.joinpath(*module[key].split("/"))
            stand_in.parent.mkdir(parents=True, exist_ok=True)
            stand_in.touch()
    return payload


def _validate_manifest(monkeypatch, tmp_path, payload: dict) -> dict:
    (tmp_path / request_ordered_jit._MANIFEST_NAME).write_text(json.dumps(payload))
    monkeypatch.setattr(request_ordered_jit, "_source_root", lambda: tmp_path)
    return request_ordered_jit.get_cake_fmha_request_ordered_manifest.__wrapped__()


def test_manifest_validator_accepts_the_shipped_module_table(
    monkeypatch, tmp_path
) -> None:
    payload = _shipped_manifest_copy(tmp_path)
    assert _validate_manifest(monkeypatch, tmp_path, payload)["modules"]


def test_manifest_validator_rejects_defines_that_disagree_with_the_module(
    monkeypatch, tmp_path
) -> None:
    payload = _shipped_manifest_copy(tmp_path)
    module = next(module for module in payload["modules"] if module["defines"])
    module["defines"]["Q_LEN"] = 6 if module["q_len"] == 1 else 1
    with pytest.raises(ValueError, match="defines disagree"):
        _validate_manifest(monkeypatch, tmp_path, payload)


@pytest.mark.parametrize("field", ["q_len", "write_lse"])
def test_manifest_validator_rejects_exact_routes_that_disagree_with_their_module(
    monkeypatch, tmp_path, field: str
) -> None:
    payload = _shipped_manifest_copy(tmp_path)
    route = payload["exact_routes"][0]
    route[field] = (
        (6 if route[field] == 1 else 1) if field == "q_len" else not route[field]
    )
    with pytest.raises(ValueError, match="disagrees with module"):
        _validate_manifest(monkeypatch, tmp_path, payload)
