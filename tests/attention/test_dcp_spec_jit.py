"""CPU-side validation for DCP speculative FMHA JIT specialization keys."""

import importlib
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.dcp import (
    _select_fp8_num_split,
    _select_num_split,
    get_dcp_spec_counter_bytes,
    get_dcp_spec_workspace_size_bytes,
    run_dcp_spec_decode,
)
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.jit.dcp import get_dcp_spec_fp8_uri, get_dcp_spec_uri
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
    fp8_uri = get_dcp_spec_fp8_uri("sm100a", 256, 3, 64, 8, 4, 3, 1)
    assert fp8_uri == (
        "cake_fmha_dcp_spec_bf16_fp8_sm100a_"
        "b256_q3_hq64_hkv8_cp4_split3_retain1"
    )


def test_dcp_jit_selects_the_route_specialized_source_family(monkeypatch) -> None:
    jit_dcp = importlib.import_module("flashinfer.jit.dcp")
    source_dir = Path(__file__).resolve().parents[2] / "csrc" / "dcp"
    monkeypatch.setattr(jit_dcp, "_get_csrc_dir", lambda: source_dir)
    monkeypatch.setattr(
        jit_dcp,
        "gen_jit_spec",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    jit_dcp.gen_dcp_spec_module.cache_clear()
    jit_dcp.gen_dcp_spec_fp8_module.cache_clear()

    try:
        v1 = jit_dcp.gen_dcp_spec_module("v1", "sm100a", 1, 1, 64, 8, 1, 1)
        v4 = jit_dcp.gen_dcp_spec_module("v4", "sm100a", 1, 1, 64, 8, 1, 16)
        fp8 = jit_dcp.gen_dcp_spec_fp8_module("sm100f", 64, 3, 64, 8, 4, 3, 1)

        assert Path(v1.sources[0]).name == "cake_fmha_dcp_spec_bf16_v1_retain1.cu"
        assert Path(v4.sources[0]).name == "cake_fmha_dcp_spec_bf16_v4_split16.cu"
        assert Path(fp8.sources[0]).name == "cake_fmha_dcp_spec_bf16_fp8.cu"
        assert Path(fp8.sources[1]).name == "cake_fmha_dcp_spec_bf16_fp8_binding.cu"
        assert "-DRETAIN_KV_L2=1" not in v1.extra_cuda_cflags
        assert "-DNUM_SPLIT=16" not in v4.extra_cuda_cflags
        assert "-DQ_LEN=3" in fp8.extra_cuda_cflags
        assert "-DNUM_SPLIT=3" in fp8.extra_cuda_cflags
        assert "-DRETAIN_KV_L2=1" in fp8.extra_cuda_cflags
    finally:
        jit_dcp.gen_dcp_spec_module.cache_clear()
        jit_dcp.gen_dcp_spec_fp8_module.cache_clear()


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


def test_fp8_dcp_spec_uri_supports_q3_but_rejects_other_gaps() -> None:
    assert "_q3_" in get_dcp_spec_fp8_uri("sm100f", 64, 3, 64, 8, 4, 3, 1)
    with pytest.raises(ValueError, match="q_len"):
        get_dcp_spec_fp8_uri("sm100f", 64, 7, 64, 8, 4, 3, 1)
    with pytest.raises(ValueError, match="num_split"):
        get_dcp_spec_fp8_uri("sm100f", 64, 4, 64, 8, 4, 5, 1)
    with pytest.raises(ValueError, match="retain_kv_l2"):
        get_dcp_spec_fp8_uri("sm100f", 64, 4, 64, 8, 4, 3, 2)


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
    assert _select_num_split(logical_tiles=32, sm_count=148, local_blocks=32) == 4
    assert _select_num_split(logical_tiles=8, sm_count=148, local_blocks=128) == 16
    assert _select_num_split(logical_tiles=64, sm_count=148, local_blocks=128) == 2
    assert (
        _select_fp8_num_split(
            logical_tiles=32, sm_count=148, local_blocks=64, cp_world=1
        )
        == 4
    )
    assert (
        _select_fp8_num_split(
            logical_tiles=32, sm_count=148, local_blocks=64, cp_world=4
        )
        == 3
    )
    assert (
        _select_fp8_num_split(
            logical_tiles=148, sm_count=148, local_blocks=64, cp_world=4
        )
        == 1
    )


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


def _empty_rank_inputs(
    seq_lens_dtype=torch.int32,
    *,
    kv_dtype=torch.bfloat16,
    q_len_per_req=1,
    bmm2_scale=1.0,
):
    page_size = 64 if kv_dtype == torch.float8_e4m3fn else 16
    return {
        "query": torch.empty((q_len_per_req, 8, 128), dtype=torch.bfloat16),
        "k_cache": torch.empty((1, 8, page_size, 128), dtype=kv_dtype),
        "v_cache": torch.empty((1, 8, page_size, 128), dtype=kv_dtype),
        "workspace_buffer": torch.empty(1, dtype=torch.uint8),
        "block_tables": torch.zeros((1, 1), dtype=torch.int32),
        "seq_lens": torch.zeros((1,), dtype=seq_lens_dtype),
        "causal_seqlens_kv_global": torch.zeros((1,), dtype=torch.int32),
        "max_local_seq_len": 0,
        "bmm1_scale": 128**-0.5,
        "bmm2_scale": bmm2_scale,
        "cp_world": 8,
        "cp_rank": 7,
        "q_len_per_req": q_len_per_req,
        "out": torch.empty((q_len_per_req, 8, 128), dtype=torch.bfloat16),
        "lse": torch.empty((q_len_per_req, 8), dtype=torch.float32),
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


def test_fp8_page64_q3_reaches_single_native_launch_with_fused_scales(
    monkeypatch,
) -> None:
    dcp = importlib.import_module("flashinfer.dcp")
    launches = []
    module = SimpleNamespace(run=lambda *args: launches.append(args))
    jit_dcp = importlib.import_module("flashinfer.jit.dcp")
    monkeypatch.setattr(dcp, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(dcp, "_select_target", lambda _device: "sm100a")
    monkeypatch.setattr(jit_dcp, "load_dcp_spec_fp8_module", lambda *args: module)

    inputs = _empty_rank_inputs(
        kv_dtype=torch.float8_e4m3fn,
        q_len_per_req=3,
        bmm2_scale=0.25,
    )
    inputs["bmm1_scale"] = 0.125
    run_dcp_spec_decode(**inputs)

    assert len(launches) == 1
    args = launches[0]
    assert args[1].dtype == torch.uint8
    assert args[2].dtype == torch.uint8
    assert args[13] == pytest.approx(0.125 / math.log(2.0))
    assert args[14] == pytest.approx(0.25)


def test_fp8_page64_underfill_uses_split3_and_caller_owned_scratch(
    monkeypatch,
) -> None:
    dcp = importlib.import_module("flashinfer.dcp")
    launches = []
    loader_calls = []
    module = SimpleNamespace(run=lambda *args: launches.append(args))
    jit_dcp = importlib.import_module("flashinfer.jit.dcp")
    monkeypatch.setattr(dcp, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(dcp, "_select_target", lambda _device: "sm100a")
    monkeypatch.setattr(
        jit_dcp,
        "load_dcp_spec_fp8_module",
        lambda *args: loader_calls.append(args) or module,
    )

    inputs = _empty_rank_inputs(
        kv_dtype=torch.float8_e4m3fn,
        q_len_per_req=4,
        bmm2_scale=0.25,
    )
    inputs["block_tables"] = torch.zeros((1, 128), dtype=torch.int32)
    inputs["seq_lens"] = torch.full((1,), 8192, dtype=torch.int32)
    inputs["max_local_seq_len"] = 8192
    inputs["workspace_buffer"] = torch.empty(
        get_dcp_spec_workspace_size_bytes(1, 4, 8, 3), dtype=torch.uint8
    )
    inputs["completion_buffer"] = torch.zeros(
        get_dcp_spec_counter_bytes(1, 4, 8), dtype=torch.uint8
    )
    run_dcp_spec_decode(**inputs)

    assert len(loader_calls) == 1
    assert loader_calls[0][-2:] == (3, 0)
    assert len(launches) == 1
    args = launches[0]
    assert args[3].data_ptr() == inputs["workspace_buffer"].data_ptr()
    assert args[7].data_ptr() == inputs["completion_buffer"].data_ptr()


def test_bf16_page16_rejects_nonunit_bmm2_scale(monkeypatch) -> None:
    dcp = importlib.import_module("flashinfer.dcp")
    monkeypatch.setattr(dcp, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(dcp, "_select_target", lambda _device: "sm100a")
    with pytest.raises(ValueError, match="BF16/page16"):
        run_dcp_spec_decode(**_empty_rank_inputs(bmm2_scale=0.5))


def test_q3_is_restricted_to_fp8_page64() -> None:
    with pytest.raises(ValueError, match="q_len_per_req"):
        run_dcp_spec_decode(**_empty_rank_inputs(q_len_per_req=3))


def test_dcp_rejects_non_int32_local_seq_lens() -> None:
    with pytest.raises(ValueError, match="contiguous int32"):
        run_dcp_spec_decode(**_empty_rank_inputs(torch.int64))
