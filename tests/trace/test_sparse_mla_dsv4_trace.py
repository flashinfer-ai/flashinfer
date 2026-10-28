# SPDX-License-Identifier: Apache-2.0
import inspect
import json
from unittest.mock import ANY

import pytest
import torch

from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4 as run
from flashinfer.trace import template
from tests.trace.template_registry import collect_registered_trace_templates

B, Q, H, D = 2, 2, 64, 512


def _empty(*shape, dtype):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _const_axes(definition):
    return {
        name: axis["value"]
        for name, axis in definition["axes"].items()
        if axis["type"] == "const"
    }


def _trtllm_gen_kwargs(**overrides):
    kwargs = dict(
        query=_empty(B, Q, H, D, dtype=torch.bfloat16),
        swa_kv_cache=_empty(16, 1, 64, D, dtype=torch.float8_e4m3fn),
        workspace_buffer=_empty(1 << 20, dtype=torch.uint8),
        sparse_indices=_empty(B * Q, 640, dtype=torch.int32),
        compressed_kv_cache=_empty(32, 1, 64, D, dtype=torch.float8_e4m3fn),
        sparse_topk_lens=_empty(B * Q, dtype=torch.int32),
        seq_lens=_empty(B, dtype=torch.int32),
    )
    kwargs.update(overrides)
    return kwargs


def _sm120_kwargs(kv_cache_format, record_bytes=528):
    return dict(
        query=_empty(B * Q, H, D, dtype=torch.bfloat16),
        swa_kv_cache=_empty(16, 64, record_bytes, dtype=torch.uint8),
        workspace_buffer=_empty(1 << 20, dtype=torch.uint8),
        sparse_indices=_empty(B * Q, 128, dtype=torch.int32),
        compressed_kv_cache=_empty(32, 64, record_bytes, dtype=torch.uint8),
        swa_topk_lens=_empty(B * Q, dtype=torch.int32),
        extra_sparse_indices=_empty(B * Q, 512, dtype=torch.int32),
        extra_sparse_topk_lens=_empty(B * Q, dtype=torch.int32),
        cum_seq_lens_q=_empty(B + 1, dtype=torch.int32),
        max_q_len=Q,
        backend="sparse",
        kv_cache_format=kv_cache_format,
    )


def test_sparse_mla_dsv4_trace_registration():
    labels = {
        label
        for func, _, label in collect_registered_trace_templates()
        if func.__qualname__ == run.__qualname__
    }
    assert {
        "sparse_mla_dsv4_dense_hnd",
        "sparse_mla_dsv4_ragged_3d",
        "sparse_mla_dsv4_dense_nhd_rope_quant",
        "sparse_mla_dsv4_hca_hnd",
        "sparse_mla_dsv4_hca_page_aligned_nhd",
    } <= labels
    assert len(labels) == 18


def test_sparse_mla_dsv4_trace_trtllm_gen_combined_table(tmp_path):
    definition = run.fi_trace(save_dir=tmp_path, **_trtllm_gen_kwargs())
    assert definition["name"] == (
        "sparse_mla_dsv4_dense_hnd_h64_swaps64_swakvd512_cps64_ckvd512_topk640"
    )
    assert definition["op_type"] == "sparse_mla_dsv4"
    assert _const_axes(definition)["compressed_page_size"] == 64
    assert definition["inputs"]["swa_kv_cache"]["shape"] == [
        "swa_num_pages",
        "kv_pad_dim",
        "swa_page_size",
        "swa_kv_dim",
    ]
    assert definition["inputs"]["swa_kv_cache"]["dtype"] == "float8_e4m3fn"
    assert definition["outputs"]["output"] == {
        "shape": ["batch_size", "q_len_per_request", "num_heads", "head_dim_v"],
        "dtype": "bfloat16",
        "param": "out",
    }
    assert "unknown" not in str(
        {k: v for k, v in definition["inputs"].items() if not v.get("optional")}
    )
    saved = json.loads((tmp_path / f"{definition['name']}.json").read_text())
    assert saved == definition


def test_sparse_mla_dsv4_trace_sm120_segments_and_inline_records():
    footer = run.fi_trace(**_sm120_kwargs("fp8_dsv41"))
    inline = run.fi_trace(**_sm120_kwargs("fp8_dsv41_inline"))
    assert footer["name"] == (
        "sparse_mla_dsv4_ragged_3d_h64_swaps64_swakvd528_cps64_ckvd528_topk128_xtopk512"
    )
    assert inline["name"] == footer["name"] + "_inline1"
    assert footer["inputs"]["swa_kv_cache"]["shape"] == [
        "swa_num_pages",
        "swa_page_size",
        "swa_kv_dim",
    ]
    assert footer["inputs"]["query"]["shape"] == ["num_tokens", "num_heads", "head_dim"]
    assert footer["outputs"]["output"]["shape"][0] == "num_tokens"


def test_sparse_mla_dsv4_trace_without_compressed_pool():
    kwargs = _sm120_kwargs("nvfp4", record_bytes=384)
    for name in ("compressed_kv_cache", "extra_sparse_indices"):
        kwargs.pop(name)
    definition = run.fi_trace(**kwargs)
    assert "compressed_page_size" not in definition["axes"]
    assert "extra_topk" not in definition["axes"]
    assert definition["name"].endswith("swakvd384_topk128")


def test_sparse_mla_dsv4_trace_nhd_layout():
    kwargs = _trtllm_gen_kwargs(
        swa_kv_cache=_empty(16, 64, 1, D, dtype=torch.float8_e4m3fn),
        compressed_kv_cache=_empty(32, 128, 1, D, dtype=torch.float8_e4m3fn),
        kv_layout="NHD",
    )
    definition = run.fi_trace(**kwargs)
    assert definition["name"].startswith("sparse_mla_dsv4_dense_nhd_")
    axes = _const_axes(definition)
    assert (axes["swa_page_size"], axes["compressed_page_size"]) == (64, 128)


def test_sparse_mla_dsv4_trace_rope_quant():
    kwargs = _trtllm_gen_kwargs(
        query=_empty(B, Q, 128, D, dtype=torch.float8_e4m3fn),
        dsv4_inv_rope_cos_sin_cache=_empty(4096, 64, dtype=torch.float32),
        backend="trtllm-gen",
    )
    definition = run.fi_trace(**kwargs)
    assert definition["name"].startswith("sparse_mla_dsv4_dense_hnd_rope_quant_h128_")
    outputs = definition["outputs"]
    assert outputs["output"]["dtype"] == "float8_e4m3fn"
    assert outputs["output_scale"]["param"] == "dsv4_output_scale"
    assert _const_axes(definition)["rope_quant_group_width"] == 4096


def test_sparse_mla_dsv4_trace_hca():
    kwargs = dict(
        query=_empty(B, Q, H, D, dtype=torch.float8_e4m3fn),
        swa_kv_cache=_empty(16, 64, D, dtype=torch.float8_e4m3fn),
        workspace_buffer=_empty(1 << 20, dtype=torch.uint8),
        compressed_kv_cache=_empty(32, 64, D, dtype=torch.float8_e4m3fn),
        sparse_topk_lens=_empty(B * Q, dtype=torch.int32),
        swa_topk_lens=_empty(B * Q, dtype=torch.int32),
        hca_swa_indices=_empty(B * Q, 128, dtype=torch.int32),
        hca_compressed_block_tables=_empty(B * Q, 8, dtype=torch.int32),
        hca_seq_lens=_empty(B, dtype=torch.int32),
        backend="cute-dsl",
    )
    definition = run.fi_trace(**kwargs)
    assert definition["name"] == (
        "sparse_mla_dsv4_hca_3d_h64_swaps64_swakvd512_cps64_ckvd512_win128"
    )
    assert "route:hca" in definition["tags"]
    with pytest.raises(NotImplementedError, match="dense queries"):
        run.fi_trace(
            **{
                **kwargs,
                "query": _empty(B * Q, H, D, dtype=torch.float8_e4m3fn),
                "cum_seq_lens_q": _empty(B + 1, dtype=torch.int32),
            }
        )


def test_sparse_mla_dsv4_trace_hca_compressed_page_aligned():
    from flashinfer.api_logging import _TRACE_DISPATCHERS
    from flashinfer.trace_apply.adapt import build_candidate_kwargs

    kwargs = _trtllm_gen_kwargs(
        query=_empty(B, Q, H, D, dtype=torch.float8_e4m3fn),
        backend="cute-dsl",
        hca_sparse_indices_format="compressed-page-aligned",
        hca_use_persistent=True,
    )
    definition = run.fi_trace(**kwargs)
    default_name = run.fi_trace(**_trtllm_gen_kwargs())["name"]
    assert definition["name"] == default_name.replace(
        "sparse_mla_dsv4_dense_hnd_", "sparse_mla_dsv4_hca_page_aligned_hnd_"
    )
    inputs = definition["inputs"]
    for name in ("backend", "hca_sparse_indices_format"):
        assert inputs[name] == {"shape": None, "dtype": "string", "description": ANY}
    for name in ("hca_is_causal", "hca_use_persistent"):
        assert inputs[name]["optional"] is True
    for name in ("sparse_indices", "sparse_topk_lens", "seq_lens"):
        assert "optional" not in inputs[name]
    assert not {
        "hca_swa_indices",
        "hca_compressed_block_tables",
        "hca_seq_lens",
        "swa_topk_lens",
    } & set(inputs)

    dispatch = _TRACE_DISPATCHERS[inspect.unwrap(run)]
    replayed = build_candidate_kwargs(dispatch(**kwargs), kwargs)
    assert replayed["hca_sparse_indices_format"] == "compressed-page-aligned"
    assert replayed["backend"] == "cute-dsl"
    assert replayed["hca_use_persistent"] is True


def test_sparse_mla_dsv4_trace_autodump(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(template, "_DUMPED_NAMES", set())
    kwargs = {
        name: torch.empty(value.shape, dtype=value.dtype)
        for name, value in _trtllm_gen_kwargs().items()
    }
    with pytest.raises(ValueError):
        run(**kwargs)
    (dumped,) = tmp_path.glob("*.json")
    definition = json.loads(dumped.read_text())
    assert definition["name"].startswith("sparse_mla_dsv4_dense_hnd_")
    assert definition["tags"][0] == (
        "fi_api:flashinfer.mla._core.trtllm_batch_decode_sparse_mla_dsv4"
    )
