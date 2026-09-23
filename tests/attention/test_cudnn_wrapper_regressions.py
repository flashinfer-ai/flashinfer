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

import math
import weakref
from dataclasses import replace

import pytest
import torch

import flashinfer
from flashinfer.cudnn import prefill


@pytest.mark.parametrize(
    "field",
    ["table_width", "table_stride", "length_dtype", "length_stride", "kv_dtype"],
)
def test_prefill_descriptor_key(field):
    q = torch.empty(5, 8, 128, device="meta", dtype=torch.bfloat16)
    k = torch.empty(8, 2, 16, 128, device="meta", dtype=q.dtype)
    v = torch.empty_like(k)
    kwargs = dict(
        max_token_seq_q=3,
        max_sequence_kv=64,
        actual_seq_lens_q=torch.empty(2, 1, 1, 1, device="meta", dtype=torch.int32),
        actual_seq_lens_kv=torch.empty(2, 1, 1, 1, device="meta", dtype=torch.int32),
        block_tables=torch.empty(2, 4, device="meta", dtype=torch.int32),
    )
    before = prefill._sdpa_prefill_key_fn(q, k, v, 0.125, **kwargs)
    if field == "table_width":
        kwargs["block_tables"] = torch.empty(2, 8, device="meta", dtype=torch.int32)
    elif field == "table_stride":
        kwargs["block_tables"] = torch.empty(2, 8, device="meta", dtype=torch.int32)[
            :, ::2
        ]
    elif field == "length_dtype":
        kwargs["actual_seq_lens_kv"] = kwargs["actual_seq_lens_kv"].to(torch.int64)
    elif field == "length_stride":
        kwargs["actual_seq_lens_kv"] = torch.empty(
            4, 1, 1, 1, device="meta", dtype=torch.int32
        )[::2]
    else:
        v = v.to(torch.float16)
    assert before != prefill._sdpa_prefill_key_fn(q, k, v, 0.125, **kwargs)


def test_prefill_override_respects_indptr_layout(monkeypatch):
    monkeypatch.setattr(prefill, "_cudnn_supports_shape_override", lambda: True)
    q = torch.empty(5, 8, 128, device="meta", dtype=torch.bfloat16)
    k = torch.empty(9, 2, 128, device="meta", dtype=q.dtype)
    indptr = torch.empty(3, device="meta", dtype=torch.int32)
    offset_names = (
        "cu_seq_lens_q",
        "cu_seq_lens_kv",
        "batch_offsets_q",
        "batch_offsets_k",
        "batch_offsets_v",
        "batch_offsets_o",
    )
    metadata = prefill._PrefillMetadata(
        3, 7, True, False, **dict.fromkeys(offset_names, indptr)
    )
    declared = metadata.override_shape(q, k)
    assert declared is not None
    graph = prefill.CudnnPrefillGraph(
        prefill._sdpa_prefill_key_fn(q, k, k, 0.125, **metadata.graph_kwargs(declared)),
        None,
        override_cache=declared,
        return_lse=False,
    )
    larger = replace(
        metadata,
        **dict.fromkeys(offset_names, torch.empty(4, device="meta", dtype=torch.int32)),
    )
    assert graph.matches(q, k, k, 0.125, larger)
    strided = replace(
        metadata, batch_offsets_q=torch.empty(6, device="meta", dtype=torch.int32)[::2]
    )
    assert strided.override_shape(q, k) is None
    assert not graph.matches(q, k, k, 0.125, strided)


def _paged_inputs():
    torch.manual_seed(7)
    q = torch.randn(5, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(6, 16, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    qo = torch.tensor([0, 3, 5], dtype=torch.int32)
    ip = torch.tensor([0, 3, 5], dtype=torch.int32)
    ix = torch.tensor([3, 0, 4, 2, 1], dtype=torch.int32)
    last = torch.tensor([1, 1], dtype=torch.int32)
    return q, k, v, qo, ip, ix, last


def _reference(q, k, v, qo, ip, ix, last, *, causal, scale, sinks=None):
    outputs, stats = [], []
    for b in range(len(qo) - 1):
        kb = k[ix[ip[b] : ip[b + 1]].long()].flatten(0, 1)
        vb = v[ix[ip[b] : ip[b + 1]].long()].flatten(0, 1)
        length = (ip[b + 1] - ip[b] - 1) * k.shape[1] + last[b]
        kb = kb[:length].float().repeat_interleave(q.shape[1] // k.shape[2], 1)
        vb = vb[:length].float().repeat_interleave(q.shape[1] // v.shape[2], 1)
        qb = q[qo[b] : qo[b + 1]].float()
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale
        if causal:
            rows = torch.arange(qb.shape[0], device=q.device)
            cols = torch.arange(kb.shape[0], device=q.device)
            scores.masked_fill_(
                cols[None, :] > rows[:, None] + kb.shape[0] - qb.shape[0], -torch.inf
            )
        denominator = scores.logsumexp(-1)
        if sinks is not None:
            denominator = torch.logaddexp(denominator, sinks[:, None])
        outputs.append(
            torch.einsum("hqk,khd->qhd", (scores - denominator[..., None]).exp(), vb)
        )
        stats.append(denominator.transpose(0, 1))
    return torch.cat(outputs), torch.cat(stats)


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_prefill_warm_run_does_not_rebuild_plan_metadata(
    monkeypatch, paged, return_lse
):
    if not prefill._cudnn_supports_direct_seqlens(torch.bfloat16, mixed=paged):
        pytest.skip("requires direct cuDNN cumulative sequence lengths")
    q, k, v, qo, ip, ix, last = _paged_inputs()
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
    if paged:
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="cudnn")
        plan = lambda: w.plan(qo, ip, ix, last, 8, 2, 128, 16, q_data_type=q.dtype)
        run = lambda query, value: w.run(query, (k, value), return_lse=return_lse)
        value = v
    else:
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="cudnn")
        kv = torch.tensor([0, 33, 50], dtype=torch.int32)
        keys = torch.cat([k[ix[:3]].flatten(0, 1)[:33], k[ix[3:]].flatten(0, 1)[:17]])
        value = torch.cat([v[ix[:3]].flatten(0, 1)[:33], v[ix[3:]].flatten(0, 1)[:17]])
        plan = lambda: w.plan(qo, kv, 8, 2, 128, q_data_type=q.dtype)
        run = lambda query, value: w.run(query, keys, value, return_lse=return_lse)
    plan()
    run(q, value)
    prepared = w._cudnn_prepared
    assert prepared is not None
    # Replanning new metadata with the same descriptor keeps prepared overrides.
    plan()

    def unexpected(*args, **kwargs):
        pytest.fail("warm run rebuilt plan metadata or override descriptors")

    # Guard the four kinds of static work, without patching their callers too.
    monkeypatch.setattr(prefill._PrefillMetadata, "resolve_from_plan", unexpected)
    for name in (
        "_prefill_descriptor_key",
        "_prefill_plan_bindings",
        "_override_execute_kwargs",
    ):
        monkeypatch.setattr(prefill, name, unexpected)
    # Fresh pointers must still bind correctly, with independent output/LSE checks.
    query, value = -q, -value
    result = run(query, value)
    out, lse = result if return_lse else (result, None)
    ref, stats = _reference(
        query, k, -v, qo, ip, ix, last, causal=False, scale=128**-0.5
    )
    torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
    if return_lse:
        torch.testing.assert_close(
            lse, stats * math.log2(math.e), atol=0.003, rtol=0.003
        )
    assert w._cudnn_prepared is prepared


def test_prefill_plan_snapshots_layout_and_replan_rekeys(monkeypatch):
    monkeypatch.setattr(prefill, "_cudnn_supports_shape_override", lambda: False)
    lengths = torch.empty(2, device="meta", dtype=torch.int32)
    table = torch.empty(2, 4, device="meta", dtype=torch.int32)

    def metadata():
        return prefill._PrefillMetadata(
            3,
            64,
            False,
            True,
            actual_seq_lens_q=lengths,
            actual_seq_lens_kv=lengths,
            block_tables=table,
        )

    plan = prefill._CudnnPrefillPlan.prepare(metadata(), torch.bfloat16, lengths.device)
    assert (
        prefill._CudnnPrefillPlan.prepare(
            metadata(), torch.bfloat16, lengths.device, plan
        )
        is plan
    )
    table.as_strided_((2, 4), (1, 2))
    assert plan.metadata.block_tables.stride() == (4, 1)
    replanned = prefill._CudnnPrefillPlan.prepare(
        metadata(), torch.bfloat16, lengths.device, plan
    )
    assert replanned.exact_keys != plan.exact_keys


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("force_legacy", [False, True])
def test_ragged_replan_reuses_owned_mirrors_without_writing_caller_buffers(
    monkeypatch, force_legacy
):
    if force_legacy:
        monkeypatch.setattr(
            prefill, "_cudnn_supports_direct_seqlens", lambda *a, **k: False
        )
    direct = prefill._cudnn_supports_direct_seqlens(torch.bfloat16)
    q = torch.randn(5, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(50, 2, 128, device=q.device, dtype=q.dtype)
    v = torch.randn_like(k)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device=q.device), backend="cudnn"
    )
    qo_gpu = torch.tensor([0, 3, 5], device=q.device, dtype=torch.int32)
    kv_gpu = torch.tensor([0, 33, 50], device=q.device, dtype=torch.int32)
    w.plan(qo_gpu, kv_gpu, 8, 2, 128, q_data_type=q.dtype)
    w.run(q, k, v)
    for split in (2, 3):
        qo = torch.tensor([0, split, 5], dtype=torch.int32)
        kv = torch.tensor([0, 31, 50], dtype=torch.int32)
        w.plan(qo, kv, 8, 2, 128, q_data_type=q.dtype)
        if split == 2:
            pointer = w._qo_indptr_buf.data_ptr()
            plan = w._cudnn_plan
        else:
            assert w._qo_indptr_buf.data_ptr() == pointer
            if direct:
                assert w._cudnn_plan is plan
        out, lse = w.run(q, k, v, return_lse=True)
        ref, stats = _reference(
            q,
            k.unsqueeze(1),
            v.unsqueeze(1),
            qo,
            kv,
            torch.arange(50, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
            causal=False,
            scale=128**-0.5,
        )
        torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
        torch.testing.assert_close(
            lse, stats * math.log2(math.e), atol=0.003, rtol=0.003
        )
    torch.testing.assert_close(qo_gpu.cpu(), torch.tensor([0, 3, 5], dtype=torch.int32))
    torch.testing.assert_close(
        kv_gpu.cpu(), torch.tensor([0, 33, 50], dtype=torch.int32)
    )


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("lse_base", ["ln", "log2"])
@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_paged_prefill_default_scale_layout_lse(layout, lse_base, explicit_metadata):
    q, k, v, qo, ip, ix, last = _paged_inputs()
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, layout, backend="cudnn")
    metadata = {}
    if explicit_metadata:
        metadata = dict(
            seq_lens=torch.tensor([33, 17], device=q.device, dtype=torch.int32),
            seq_lens_q=torch.tensor([3, 2], device=q.device, dtype=torch.int32),
            block_tables=torch.tensor(
                [[3, 0, 4], [2, 1, 0]], device=q.device, dtype=torch.int32
            ),
            max_token_per_sequence=3,
            max_sequence_kv=48,
        )
    w.plan(
        qo, ip, ix, last, 8, 2, 128, 16, causal=True, q_data_type=q.dtype, **metadata
    )
    cache = (k, v) if layout == "NHD" else (k.transpose(1, 2), v.transpose(1, 2))
    out, lse = w.run(q, cache, return_lse=True, lse_base=lse_base)
    ref, lse_ref = _reference(q, k, v, qo, ip, ix, last, causal=True, scale=128**-0.5)
    torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
    torch.testing.assert_close(
        lse,
        lse_ref * (math.log2(math.e) if lse_base == "log2" else 1),
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
def test_decode_capture_replan_keeps_metadata_alive():
    q, k, v, _, ip, ix, last = _paged_inputs()
    q = q[:2].contiguous()
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws,
        "NHD",
        backend="cudnn",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.empty(3, device=q.device, dtype=torch.int32),
        paged_kv_indices_buffer=torch.empty(5, device=q.device, dtype=torch.int32),
        paged_kv_last_page_len_buffer=torch.empty(
            2, device=q.device, dtype=torch.int32
        ),
    )

    def plan(lengths):
        w.plan(ip, ix, lengths, 8, 2, 128, 16, q_data_type=q.dtype)

    plan(last)
    out = torch.empty_like(q)
    w.run(q, (k, v), out=out)
    owner = weakref.ref(w._cudnn_prepared.seq_lens_q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(q, (k, v), out=out)
    last = torch.tensor([5, 3], dtype=torch.int32)
    plan(last)
    poison = [
        torch.zeros(2, 1, 1, 1, device=q.device, dtype=torch.int32) for _ in range(128)
    ]
    out.fill_(torch.nan)
    graph.replay()
    ref, _ = _reference(
        q, k, v, torch.tensor([0, 1, 2]), ip, ix, last, causal=False, scale=128**-0.5
    )
    torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
    assert owner() is not None
    assert len(poison) == 128
    # Preparing another output contract must not release the captured Q lengths.
    w.run(q, (k, v), return_lse=True)
    out.fill_(torch.nan)
    graph.replay()
    torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
def test_paged_single_token_gqa_rejects_incomplete_lse():
    q, k, v, _, ip, ix, last = _paged_inputs()
    q = q[:2]
    qo = torch.tensor([0, 1, 2], dtype=torch.int32)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, device=q.device, dtype=torch.uint8),
        "NHD",
        backend="cudnn",
    )
    w.plan(qo, ip, ix, last, 8, 2, 128, 16, q_data_type=q.dtype)
    out = w.run(q, (k, v))
    ref, _ = _reference(q, k, v, qo, ip, ix, last, causal=False, scale=128**-0.5)
    torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
    with pytest.raises(NotImplementedError, match="LSE"):
        w.run(q, (k, v), return_lse=True)


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
def test_decode_replan_reuses_preparation(monkeypatch):
    import flashinfer.decode as decode

    q, k, v, _, ip, ix, last = _paged_inputs()
    q = q[:2].contiguous()
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend="cudnn")
    calls = []
    prepare = decode.prepare_cudnn_batch_decode

    def counted(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(decode, "prepare_cudnn_batch_decode", counted)
    for scale in (0.125, 0.125, 0.25):
        w.plan(ip, ix, last, 8, 2, 128, 16, q_data_type=q.dtype, sm_scale=scale)
        out = w.run(q, (k, v))
        ref, _ = _reference(
            q, k, v, torch.tensor([0, 1, 2]), ip, ix, last, causal=False, scale=scale
        )
        torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
    assert len(calls) == 2


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("strided", [False, True])
def test_decode_rebinds_sinks_without_repreparing(monkeypatch, strided):
    import flashinfer.decode as decode

    q, k, v, _, ip, ix, last = _paged_inputs()
    q = q[:4].contiguous()
    ws = torch.empty(128 << 20, device=q.device, dtype=torch.uint8)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, "NHD", backend="cudnn", use_tensor_cores=True
    )
    w.plan(ip, ix, last, 8, 2, 128, 16, q_data_type=q.dtype, q_len_per_req=2)
    calls = []
    prepare = decode.prepare_cudnn_batch_decode

    def counted(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(decode, "prepare_cudnn_batch_decode", counted)
    for value in (1.0, 4.0, -1.0):
        storage = torch.full((16 if strided else 8,), value, device=q.device)
        sinks = storage[::2] if strided else storage
        out = w.run(q, (k, v), sinks=sinks)
        ref, _ = _reference(
            q,
            k,
            v,
            torch.tensor([0, 2, 4]),
            ip,
            ix,
            last,
            causal=True,
            scale=128**-0.5,
            sinks=sinks,
        )
        torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
    assert len(calls) == 1
    # A strided sink copy must read fresh values on replay as well.
    out = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(q, (k, v), sinks=sinks, out=out)
    storage.fill_(5)
    graph.replay()
    ref, _ = _reference(
        q,
        k,
        v,
        torch.tensor([0, 2, 4]),
        ip,
        ix,
        last,
        causal=True,
        scale=128**-0.5,
        sinks=sinks,
    )
    torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("kind", ["paged", "ragged"])
@pytest.mark.parametrize(
    "unsupported",
    [
        dict(window_left=4),
        dict(pos_encoding_mode="ALIBI"),
        dict(logits_soft_cap=2),
        dict(packed_custom_mask=torch.ones(1, dtype=torch.uint8)),
        dict(prefix_len_ptr=torch.ones(1, dtype=torch.int32)),
    ],
)
def test_prefill_rejects_unsupported_semantics(kind, unsupported):
    ws = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    qo = torch.tensor([0, 2], dtype=torch.int32)
    kv = torch.tensor([0, 4], dtype=torch.int32)
    if kind == "paged":
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="cudnn")
        args = (
            qo,
            kv,
            torch.arange(4, dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            8,
            2,
            128,
            16,
        )
    else:
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="cudnn")
        args = (qo, kv, 8, 2, 128)
    with pytest.raises(NotImplementedError, match="cuDNN"):
        w.plan(*args, q_data_type=torch.bfloat16, **unsupported)


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
def test_ragged_prefill_rejects_output_scale():
    q = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(4, 2, 128, device=q.device, dtype=q.dtype)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 << 20, device=q.device, dtype=torch.uint8), backend="cudnn"
    )
    w.plan(
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32),
        8,
        2,
        128,
        q_data_type=q.dtype,
    )
    with pytest.raises(NotImplementedError, match="o_scale"):
        w.run(q, kv, kv, o_scale=0.5)


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("scale_type", ["default", "scalar", "tensor"])
def test_fp8_prefill_scales_capture_replay(paged, scale_type):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuDNN's unified FP8 prefill engine requires Blackwell or newer")
    if not prefill._cudnn_supports_direct_seqlens(torch.float8_e4m3fn, mixed=paged):
        pytest.skip("requires direct cuDNN FP8 cumulative sequence lengths")
    q, k, v, qo, ip, ix, last = _paged_inputs()
    q, k, v = (t.to(torch.float8_e4m3fn) for t in (q, k, v))
    ws = torch.empty(128 << 20, device=q.device, dtype=torch.uint8)
    plan_kwargs = dict(causal=True, q_data_type=q.dtype, o_data_type=torch.bfloat16)
    if paged:
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="cudnn")
        w.plan(qo, ip, ix, last, 8, 2, 128, 16, **plan_kwargs)
        inputs = (q, (k, v))
    else:
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="cudnn")
        w.plan(
            qo, torch.tensor([0, 33, 50], dtype=torch.int32), 8, 2, 128, **plan_kwargs
        )
        keys, values = (
            torch.cat([t[ix[:3]].flatten(0, 1)[:33], t[ix[3:]].flatten(0, 1)[:17]])
            for t in (k, v)
        )
        inputs = (q, keys, values)

    scale_names = ("q_scale", "k_scale", "v_scale")
    scales = (1.0, 1.0, 1.0) if scale_type == "default" else (0.5, 0.25, 0.75)
    kwargs = (
        {} if scale_type == "default" else dict(zip(scale_names, scales, strict=True))
    )
    if scale_type == "tensor":
        kwargs = {
            name: torch.full((1,), value, dtype=torch.float32, device=q.device)
            for name, value in kwargs.items()
        }

    def run(scale_kwargs):
        return w.run(*inputs, return_lse=True, lse_base="ln", **scale_kwargs)

    def check(out, lse, scale_values):
        qs, ks, vs = scale_values
        ref, lse_ref = _reference(
            q.float(),
            k.float(),
            v.float() * vs,
            qo,
            ip,
            ix,
            last,
            causal=True,
            scale=128**-0.5 * qs * ks,
        )
        torch.testing.assert_close(lse, lse_ref, atol=0.003, rtol=0.003)
        # sdpa_fp8 also rounds probabilities to FP8; retain the eager math gate.
        assert torch.linalg.vector_norm(
            out.float() - ref
        ) < 0.05 * torch.linalg.vector_norm(ref)
        if scale_type != "default":
            torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.03)

    out, lse = run(kwargs)
    check(out, lse, scales)
    if scale_type == "default":
        explicit, _ = run(dict.fromkeys(scale_names, 1.0))
        torch.testing.assert_close(out, explicit, atol=0.001, rtol=0.001)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = run(kwargs)

    # Replaying must read current inputs and GPU scales, while Python scalars
    # stay fixed in this capture even after an eager call with different values.
    q.copy_(-q.float())
    alternate_scales = (0.75, 0.5, 0.25)
    if scale_type == "tensor":
        for name, value in zip(scale_names, alternate_scales, strict=True):
            kwargs[name].fill_(value)
        scales = alternate_scales
    elif scale_type == "scalar":
        eager_out, eager_lse = run(
            dict(zip(scale_names, alternate_scales, strict=True))
        )
        check(eager_out, eager_lse, alternate_scales)
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    check(out, lse, scales)


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
def test_attention_handles_follow_device_and_thread():
    import concurrent.futures
    from flashinfer.cudnn.decode import _create_cudnn_handle as decode_handle
    from flashinfer.cudnn.prefill import _create_cudnn_handle as prefill_handle

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    # Leave device 0 current while explicitly requesting a handle on device 1.
    with torch.cuda.device(0):
        first = decode_handle(torch.cuda.current_stream(0))
        second = prefill_handle(torch.cuda.current_stream(1))
        assert first != second
        assert decode_handle(torch.cuda.current_stream(0)) == first
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            other = pool.submit(
                lambda: decode_handle(torch.cuda.current_stream(0))
            ).result()
            assert other != first
