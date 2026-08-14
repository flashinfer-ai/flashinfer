"""CPU-side validation for DCP speculative FMHA JIT specialization keys."""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from flashinfer.dcp import (
    _select_num_split,
    get_dcp_spec_counter_bytes,
    get_dcp_spec_workspace_size_bytes,
    run_dcp_spec_decode,
)
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.jit.dcp import get_dcp_spec_uri
from flashinfer.trace.templates.attention import (
    trtllm_batch_decode_dcp_spec_split_kv_trace,
    trtllm_batch_decode_dcp_spec_trace,
    trtllm_batch_decode_trace_dispatch,
)


def test_dcp_spec_uri_covers_full_parameterized_domain() -> None:
    v1_uri = get_dcp_spec_uri("v1", "sm100f", 64, 5, 32, 4, 8, 1)
    assert v1_uri.startswith("cake_fmha_dcp_spec_bf16_v1_")
    assert v1_uri.endswith("_b64_q5_hq32_hkv4_cp8_retain1")
    v4_uri = get_dcp_spec_uri("v4", "sm100a", 1, 8, 64, 8, 4, 16)
    assert v4_uri.startswith("cake_fmha_dcp_spec_bf16_v4_")
    assert v4_uri.endswith("_b1_q8_hq64_hkv8_cp4_split16")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("v1", "sm100f", 1, 3, 32, 4, 4, 0), "q_len"),
        (("v1", "sm100f", 1, 4, 32, 4, 3, 0), "cp_world"),
        (("v1", "sm100f", 1, 4, 64, 4, 4, 0), "group ratio"),
        (("v4", "sm100f", 1, 4, 32, 4, 4, 1), "num_split"),
    ],
)
def test_dcp_spec_uri_rejects_unsupported_specialization(args, message) -> None:
    with pytest.raises(ValueError, match=message):
        get_dcp_spec_uri(*args)


def test_public_decode_api_adds_optional_dcp_arguments() -> None:
    parameters = inspect.signature(trtllm_batch_decode_with_kv_cache).parameters
    assert parameters["cp_world"].default == 1
    assert parameters["cp_rank"].default == 0
    assert parameters["causal_seqlens_kv_global"].default is None


def test_dcp_workspace_and_counter_sizes_are_caller_owned_exact_views() -> None:
    # B=8, Q=4, Hq=64, split=6: BF16 O[...128] + FP32 LSE per row.
    rows = 8 * 4 * 64 * 6
    assert get_dcp_spec_workspace_size_bytes(8, 4, 64, 6) == rows * (128 * 2 + 4)
    assert get_dcp_spec_counter_bytes(8, 4, 8) == 8 * 4 * 8 * 4


def test_dcp_split_selector_matches_promoted_policy() -> None:
    assert _select_num_split(logical_tiles=32, sm_count=148, local_blocks=9) == 1
    assert _select_num_split(logical_tiles=8, sm_count=148, local_blocks=128) == 16
    assert _select_num_split(logical_tiles=64, sm_count=148, local_blocks=128) == 2


@pytest.mark.parametrize(
    ("capability", "target"),
    [
        ((10, 0), "sm100a"),
        ((10, 3), "sm100f"),
    ],
)
def test_dcp_target_keeps_independent_architecture_baselines(
    monkeypatch, capability, target
) -> None:
    dcp = importlib.import_module("flashinfer.dcp")
    monkeypatch.setattr(dcp, "get_compute_capability", lambda _device: capability)
    monkeypatch.setattr(dcp, "_is_cuda_version_at_least", lambda _version: True)
    assert dcp._select_target(None) == target


def test_dcp_trace_dispatch_distinguishes_combined_and_split_kv() -> None:
    marker = object()
    assert (
        trtllm_batch_decode_trace_dispatch(
            causal_seqlens_kv_global=marker, kv_cache=marker
        )
        is trtllm_batch_decode_dcp_spec_trace
    )
    assert (
        trtllm_batch_decode_trace_dispatch(
            causal_seqlens_kv_global=marker, kv_cache=(marker, marker)
        )
        is trtllm_batch_decode_dcp_spec_split_kv_trace
    )


def _empty_rank_inputs(seq_lens_dtype=torch.int32):
    return {
        "query": torch.empty((1, 8, 128), dtype=torch.bfloat16),
        "k_cache": torch.empty((1, 8, 16, 128), dtype=torch.bfloat16),
        "v_cache": torch.empty((1, 8, 16, 128), dtype=torch.bfloat16),
        "workspace_buffer": torch.empty(1, dtype=torch.uint8),
        "block_tables": torch.zeros((1, 1), dtype=torch.int32),
        "seq_lens": torch.zeros((1,), dtype=seq_lens_dtype),
        "causal_seqlens_kv_global": torch.zeros((1,), dtype=torch.int32),
        "max_local_seq_len": 0,
        "sm_scale": 128**-0.5,
        "cp_world": 8,
        "cp_rank": 7,
        "q_len_per_req": 1,
        "out": torch.empty((1, 8, 128), dtype=torch.bfloat16),
        "lse": torch.empty((1, 8), dtype=torch.float32),
        "completion_buffer": None,
    }


def test_dcp_all_empty_rank_reaches_native_v1_route(monkeypatch) -> None:
    dcp = importlib.import_module("flashinfer.dcp")
    launches = []
    module = SimpleNamespace(run=lambda *args: launches.append(args))
    jit_dcp = importlib.import_module("flashinfer.jit.dcp")
    monkeypatch.setattr(dcp, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(dcp, "_select_target", lambda _device: "sm100a")
    monkeypatch.setattr(jit_dcp, "load_dcp_spec_module", lambda *args: module)

    run_dcp_spec_decode(**_empty_rank_inputs())

    assert len(launches) == 1


def test_dcp_rejects_non_int32_local_seq_lens() -> None:
    with pytest.raises(ValueError, match="contiguous int32"):
        run_dcp_spec_decode(**_empty_rank_inputs(torch.int64))
