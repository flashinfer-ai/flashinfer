# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import json

import pytest
import torch

from flashinfer.fi_trace import fi_trace
from flashinfer.msa_ops import minimax_m3_sparse_attn_decode


def test_minimax_m3_init_cpu():
    from flashinfer.trace.templates.minimax_m3 import _minimax_m3_init

    inputs = _minimax_m3_init(
        total_q=8, batch_size=4, num_pages=8, max_pages=2, device="cpu"
    )
    assert inputs["workspace"] is None
    assert inputs["q"].shape == (8, 16, 128)
    assert inputs["topk_idx"].shape == (1, 8, 16)


def test_minimax_m3_check_is_per_query_head():
    from flashinfer.trace.templates.minimax_m3 import _minimax_m3_check

    expected = torch.full((2, 16, 128), 0.001)
    assert _minimax_m3_check(expected, expected.clone())
    wrong = expected.clone()
    wrong[0, 0] = 0
    assert not _minimax_m3_check(expected, wrong)


@pytest.mark.parametrize("k_rank,v_rank", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_minimax_m3_trace_cpu(tmp_path, k_rank, v_rank):
    q = torch.empty(8, 64, 128, dtype=torch.bfloat16)
    definition = fi_trace(
        minimax_m3_sparse_attn_decode,
        save_dir=tmp_path,
        q=q,
        kv_cache=torch.empty(32, 4, 128, 256, dtype=torch.float8_e4m3fn),
        topk_idx=torch.empty(4, 8, 16, dtype=torch.int32),
        block_table=torch.empty(2, 16, dtype=torch.int32),
        seq_lens=torch.empty(2, dtype=torch.int32),
        k_scale=torch.ones((1,) * k_rank),
        v_scale=torch.ones((1,) * v_rank),
        out=torch.empty_like(q),
        workspace=None,
    )
    suffix = "" if (k_rank, v_rank) == (1, 1) else f"_k{k_rank}v{v_rank}"
    assert definition["name"] == (
        f"minimax_m3_sparse_attn_decode{suffix}_h64_kv4_d128_p128_packed256_topk16"
    )
    assert definition["inputs"]["k_scale"]["shape"] == ["scale_size"] * k_rank
    assert definition["inputs"]["v_scale"]["shape"] == ["scale_size"] * v_rank
    assert definition["inputs"]["kv_cache"]["dtype"] == "float8_e4m3fn"
    assert definition["outputs"]["out"]["dtype"] == "bfloat16"
    assert "reference" in definition
    assert "init" in definition
    saved = tmp_path / (definition["name"] + ".json")
    assert saved.read_bytes().endswith(b"\n")
    assert json.loads(saved.read_text()) == definition


@pytest.mark.parametrize("k_rank,v_rank", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_minimax_m3_rendered_init_preserves_scale_ranks(tmp_path, k_rank, v_rank):
    from flashinfer.trace.templates.minimax_m3 import _MINIMAX_M3_TRACES

    template = _MINIMAX_M3_TRACES[(k_rank, v_rank)]
    small = dict(
        total_q=8, batch_size=4, num_pages=8, max_pages=2, scale_size=1, device="cpu"
    )
    inputs = template.init(**small)
    assert inputs["k_scale"].ndim == k_rank
    assert inputs["v_scale"].ndim == v_rank
    definition = fi_trace(minimax_m3_sparse_attn_decode, save_dir=tmp_path, **inputs)
    namespace = {}
    exec(compile(definition["init"], "<minimax_m3_trace_init>", "exec"), namespace)
    restored = namespace[template.init.__name__](**small)
    assert restored["k_scale"].ndim == k_rank
    assert restored["v_scale"].ndim == v_rank
    roundtrip = fi_trace(minimax_m3_sparse_attn_decode, **restored)
    assert roundtrip["name"] == definition["name"]
    assert roundtrip["inputs"] == definition["inputs"]


def test_minimax_m3_autodump_keeps_all_scale_forms(tmp_path, monkeypatch):
    import flashinfer.trace.template as trace_module
    from flashinfer.trace.templates.minimax_m3 import _MINIMAX_M3_TRACES

    monkeypatch.setattr(trace_module, "_DUMPED_NAMES", set())
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path))
    for template in _MINIMAX_M3_TRACES.values():
        inputs = template.init(
            total_q=8, batch_size=4, num_pages=8, max_pages=2, device="cpu"
        )
        # Auto-dump happens before validation, so exercise it without a GPU.
        with pytest.raises(TypeError, match="workspace"):
            minimax_m3_sparse_attn_decode(**inputs)
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 4
    emitted_ranks = set()
    for path in paths:
        assert path.read_bytes().endswith(b"\n")
        definition = json.loads(path.read_text())
        emitted_ranks.add(
            tuple(
                len(definition["inputs"][name]["shape"])
                for name in ("k_scale", "v_scale")
            )
        )
    assert emitted_ranks == set(_MINIMAX_M3_TRACES)


def test_minimax_m3_does_not_misdescribe_higher_rank_scales():
    assert (
        fi_trace(
            minimax_m3_sparse_attn_decode,
            k_scale=torch.ones(1, 1),
            v_scale=torch.ones(1),
        )
        == {}
    )


def test_tensor_sm_scale_rejected_before_other_validation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor data must not be converted")

    monkeypatch.setattr(torch.Tensor, "__float__", forbidden)
    with pytest.raises(TypeError, match="sm_scale must be a host float"):
        minimax_m3_sparse_attn_decode(
            None,
            None,
            None,
            None,
            None,
            k_scale=None,
            v_scale=None,
            out=None,
            workspace=None,
            sm_scale=torch.tensor(0.1),
        )


@pytest.mark.parametrize(
    "sm_scale", [0, -0.125, float("inf"), float("-inf"), float("nan")]
)
def test_invalid_host_sm_scale_rejected_before_other_validation(sm_scale):
    with pytest.raises(ValueError, match="sm_scale must be finite and positive"):
        minimax_m3_sparse_attn_decode(
            None,
            None,
            None,
            None,
            None,
            k_scale=None,
            v_scale=None,
            out=None,
            workspace=None,
            sm_scale=sm_scale,
        )
