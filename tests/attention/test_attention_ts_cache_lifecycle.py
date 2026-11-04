# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer.attention.prims_ts import _compile_cache as cc
from flashinfer.attention.prims_ts import mla_decode as mla

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


@pytest.mark.parametrize("extra_topk", [0, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sparse_prepared_launch_rebinds_dynamic_inputs(extra_topk, dtype, monkeypatch):
    torch.manual_seed(41)
    device = torch.device("cuda", torch.cuda.current_device())
    rows, heads, topk = 4, 16, 128
    query = torch.randn(rows, 1, heads, 512, device=device).to(dtype)
    pool = torch.randn(256, 1, 512, device=device).to(dtype)
    extra = (
        torch.randn_like(pool)
        if extra_topk and dtype == torch.bfloat16
        else (pool.clone() if extra_topk else None)
    )
    indices = torch.randint(0, 256, (rows, topk), device=device, dtype=torch.int32)
    lengths = torch.full((rows,), topk, device=device, dtype=torch.int32)
    extra_indices = indices.clone() if extra_topk else None
    extra_lengths = lengths.clone() if extra_topk else None
    sinks = [torch.randn(heads, device=device) for _ in range(2)]
    output = torch.empty_like(query, dtype=torch.bfloat16)
    wrapper = mla.BatchSparseMLADecodePagedTSWrapper()
    plan = dict(
        max_topk=topk,
        max_extra_topk=extra_topk,
        q_dtype=dtype,
        use_attention_sinks=True,
    )
    wrapper.plan(device, rows, heads, **plan)

    def run(sink, validate=False, **kwargs):
        return wrapper.run(
            query,
            pool,
            indices,
            lengths,
            extra,
            extra_indices=extra_indices,
            extra_lengths=extra_lengths,
            attention_sinks=sink,
            out=output,
            validate=validate,
            **kwargs,
        )

    prepared_calls = []
    original = wrapper._run_prepared

    def prepared_run(*args):
        prepared_calls.append(1)
        return original(*args)

    monkeypatch.setattr(wrapper, "_run_prepared", prepared_run)
    references = [run(sink, validate=True).clone() for sink in sinks]
    assert len(wrapper._prepared) == 2
    for sink, reference in zip(sinks, references, strict=False):
        torch.testing.assert_close(run(sink), reference, rtol=0, atol=0)
    assert len(prepared_calls) == 2
    with pytest.raises(ValueError, match="out"):
        wrapper._run_prepared(
            next(iter(wrapper._prepared.values())),
            query,
            indices,
            lengths,
            extra_indices,
            extra_lengths,
            output.flatten(),
        )
    # New metadata at an identical sink address cannot hit the prepared entry.
    with pytest.raises(ValueError, match="attention_sinks"):
        run(sinks[0].view(1, heads))
    with pytest.raises(TypeError, match="Python"):
        run(sinks[0], output_scale=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(sinks[0])
    query.copy_(torch.randn(query.shape, device=device).to(dtype))
    indices.copy_(torch.roll(indices, 5, 1))
    lengths.fill_(topk // 2)
    sinks[0].add_(0.25)
    expected = run(sinks[0], validate=True).clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        run(sinks[0], output_scale=0.5).clone(),
        run(sinks[0], output_scale=0.5, validate=True),
        rtol=0,
        atol=0,
    )
    wrapper.plan(device, rows, heads, **plan)
    assert not wrapper._prepared


def test_sparse_precompile_probe_then_reload(monkeypatch, tmp_path):
    from flashinfer.jit import env

    monkeypatch.setattr(env, "FLASHINFER_JIT_DIR", tmp_path)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    mla._get_compiled_mla_decode.cache_clear()
    wrapper = mla.BatchSparseMLADecodePagedTSWrapper()
    with cc.probe_cached(), pytest.raises(cc.KernelNotCached):
        wrapper.plan("cuda", 1, 16, max_topk=128)
    with cc.precompile_only():
        wrapper.plan("cuda", 1, 16, max_topk=128)
    assert cc.is_placeholder(wrapper._state["compiled"])
    with cc.probe_cached():
        wrapper.plan("cuda", 1, 16, max_topk=128)
    wrapper.plan("cuda", 1, 16, max_topk=128)
    assert not cc.is_placeholder(wrapper._state["compiled"])
    query = torch.randn(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(128, 1, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(128, device="cuda", dtype=torch.int32)[None]
    lengths = torch.tensor([128], device="cuda", dtype=torch.int32)
    first = wrapper.run(query, pool, indices, lengths).clone()
    mla._get_compiled_mla_decode.cache_clear()
    wrapper.plan("cuda", 1, 16, max_topk=128)
    torch.testing.assert_close(
        wrapper.run(query, pool, indices, lengths), first, rtol=0, atol=0
    )
    resolved = mla._resolve_sparse_mla_plan("cuda", 1, 16, max_topk=128)
    assert mla._resolve_sparse_mla_plan("cuda", 1, 16, max_topk=128) is resolved
