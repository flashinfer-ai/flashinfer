"""CPU-side contract tests for request-ordered Cake FMHA decode."""

from __future__ import annotations

import dataclasses
import inspect

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


def _manifest() -> dict:
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
    monkeypatch.setattr(
        cake_api, "get_cake_fmha_request_ordered_manifest", _manifest
    )
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
    monkeypatch.setattr(
        cake_api, "get_cake_fmha_request_ordered_manifest", _manifest
    )
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
        inspect.signature(flashinfer.trtllm_batch_decode_with_kv_cache).parameters
    )
    assert parameters[-2:] == ["request_order", "request_order_plan"]


def test_request_order_requires_explicit_cake_backend() -> None:
    tensor = torch.empty(1)
    with pytest.raises(ValueError, match="explicit backend='cake'"):
        flashinfer.trtllm_batch_decode_with_kv_cache(
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
        flashinfer.trtllm_batch_decode_with_kv_cache(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            1,
            backend="cake",
            request_order_plan=plan,
        )
