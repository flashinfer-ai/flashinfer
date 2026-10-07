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
    plan = prefill._CudnnPrefillPlan.prepare(larger, q.dtype, q.device)
    assert graph.matches_plan(q, k, k, 0.125, plan, False)
    strided = replace(
        metadata, batch_offsets_q=torch.empty(6, device="meta", dtype=torch.int32)[::2]
    )
    assert strided.override_shape(q, k) is None
    plan = prefill._CudnnPrefillPlan.prepare(strided, q.dtype, q.device)
    assert not graph.matches_plan(q, k, k, 0.125, plan, False)


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
@pytest.mark.parametrize("unit_scales", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_warm_run_does_not_rebuild_plan_metadata(
    monkeypatch, paged, return_lse, unit_scales, dtype
):
    if not prefill._cudnn_supports_direct_seqlens(dtype, mixed=paged):
        pytest.skip("requires direct cuDNN cumulative sequence lengths")
    q, k, v, qo, ip, ix, last = _paged_inputs()
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    scales = dict(q_scale=1.0, k_scale=1, v_scale=1.0) if unit_scales else {}
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
    if paged:
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="cudnn")
        plan = lambda: w.plan(qo, ip, ix, last, 8, 2, 128, 16, q_data_type=q.dtype)
        run = lambda query, value: w.run(
            query, (k, value), return_lse=return_lse, **scales
        )
        value = v
    else:
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="cudnn")
        kv = torch.tensor([0, 33, 50], dtype=torch.int32)
        keys = torch.cat([k[ix[:3]].flatten(0, 1)[:33], k[ix[3:]].flatten(0, 1)[:17]])
        value = torch.cat([v[ix[:3]].flatten(0, 1)[:33], v[ix[3:]].flatten(0, 1)[:17]])
        plan = lambda: w.plan(qo, kv, 8, 2, 128, q_data_type=q.dtype)
        run = lambda query, value: w.run(
            query, keys, value, return_lse=return_lse, **scales
        )
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
    if unit_scales:
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                captured = run(query, value)
            query.mul_(0.5)
            value.mul_(0.5)
            graph.replay()
            out, lse = captured if return_lse else (captured, None)
            ref, stats = _reference(
                query,
                k,
                -v * 0.5,
                qo,
                ip,
                ix,
                last,
                causal=False,
                scale=128**-0.5,
            )
            torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
            if return_lse:
                torch.testing.assert_close(
                    lse, stats * math.log2(math.e), atol=0.003, rtol=0.003
                )
        finally:
            graph.reset()


def test_prefill_plan_snapshots_layout_and_replan_rekeys(monkeypatch):
    monkeypatch.setattr(prefill, "_cudnn_supports_shape_override", lambda: False)
    lengths = torch.empty(2, dtype=torch.int32)
    table = torch.empty(2, 4, dtype=torch.int32)

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


@pytest.mark.parametrize("change", ["view_alias", "set_storage", "set_offset"])
def test_prefill_replan_tracks_storage_not_tensor_identity(monkeypatch, change):
    """Fresh views reuse bindings; rebinding the same tensor must replace them."""
    monkeypatch.setattr(prefill, "_cudnn_supports_shape_override", lambda: False)
    lengths = torch.tensor([3, 2], dtype=torch.int32)
    table = torch.arange(16, dtype=torch.int32)[:8].view(2, 4)

    def metadata():
        return prefill._PrefillMetadata(
            3,
            64,
            False,
            True,
            actual_seq_lens_q=lengths.view_as(lengths),
            actual_seq_lens_kv=lengths.view_as(lengths),
            block_tables=table,
        )

    initial = metadata()
    original_q, original_kv = initial.actual_seq_lens_q, initial.actual_seq_lens_kv
    plan = prefill._CudnnPrefillPlan.prepare(initial, torch.bfloat16, lengths.device)
    replace_storage = change != "view_alias"
    if replace_storage:
        old_table = plan.metadata.block_tables
        if change == "set_storage":
            table.set_(table.flip(1).contiguous())
        else:
            table.set_(table.untyped_storage(), 4, table.size(), table.stride())
        # Reuse the same original tensor objects, as the former identity check
        # did. Their detached plan snapshots still own the old allocation.
        current = prefill._PrefillMetadata(
            3,
            64,
            False,
            True,
            actual_seq_lens_q=original_q,
            actual_seq_lens_kv=original_kv,
            block_tables=table,
        )
    else:
        table = table.view_as(table)
        current = metadata()
    new = prefill._CudnnPrefillPlan.prepare(
        current, torch.bfloat16, lengths.device, plan
    )
    assert (new is plan) is (not replace_storage)
    assert new.metadata.block_tables.is_set_to(table)
    if replace_storage:
        torch.testing.assert_close(
            old_table, torch.arange(8, dtype=torch.int32).view(2, 4)
        )
        torch.testing.assert_close(new.metadata.block_tables, table)


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("force_legacy", [False, True])
@pytest.mark.parametrize("lse_layout", ["NH", "HN"])
def test_ragged_replan_reuses_owned_mirrors_without_writing_caller_buffers(
    monkeypatch, force_legacy, lse_layout
):
    if force_legacy:
        # The wrapper imports its own alias; both dispatch and metadata
        # resolution must see the missing capability.
        for module in (flashinfer.prefill, prefill):
            monkeypatch.setattr(
                module, "_cudnn_supports_direct_seqlens", lambda *a, **k: False
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
    if not direct:
        assert w._cudnn_plan is None
    w.run(q, k, v)
    for split in (2, 3):
        qo = torch.tensor([0, split, 5], dtype=torch.int32)
        kv = torch.tensor([0, 31, 50], dtype=torch.int32)
        w.plan(qo, kv, 8, 2, 128, q_data_type=q.dtype)
        if not direct:
            assert w._cudnn_plan is None
        if split == 2:
            pointer = w._qo_indptr_buf.data_ptr()
            plan = w._cudnn_plan
        else:
            assert w._qo_indptr_buf.data_ptr() == pointer
            if direct:
                assert w._cudnn_plan is plan
        out, lse = w.run(q, k, v, return_lse=True, lse_layout=lse_layout)
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
            lse.T if lse_layout == "HN" else lse,
            stats * math.log2(math.e),
            atol=0.003,
            rtol=0.003,
        )
    torch.testing.assert_close(qo_gpu.cpu(), torch.tensor([0, 3, 5], dtype=torch.int32))
    torch.testing.assert_close(
        kv_gpu.cpu(), torch.tensor([0, 33, 50], dtype=torch.int32)
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(q, k, v, out=out, lse=lse, return_lse=True, lse_layout=lse_layout)
    w.plan(qo.clone(), kv.clone(), 8, 2, 128, q_data_type=q.dtype)
    # Native capture holds raw metadata pointers. Reusing freed indptr storage
    # must not silently change the work done by subsequent replays.
    poison = [torch.zeros(3, dtype=torch.int32, device=q.device) for _ in range(128)]
    out.fill_(torch.nan)
    lse.fill_(torch.nan)
    graph.replay()
    torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
    torch.testing.assert_close(
        lse.T if lse_layout == "HN" else lse,
        stats * math.log2(math.e),
        atol=0.003,
        rtol=0.003,
    )
    assert len(poison) == 128


@pytest.mark.parametrize("backend", ["fa2", "auto"])
@pytest.mark.parametrize("length_dtype", [None, torch.int32, torch.uint32])
def test_paged_prefill_explicit_max_preserves_fallback_metadata(backend, length_dtype):
    q, k, v, qo, ip, ix, last = _paged_inputs()
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, device=q.device, dtype=torch.uint8),
        "NHD",
        backend=backend,
        use_cuda_graph=True,
        qo_indptr_buf=qo.cuda(),
        paged_kv_indptr_buf=ip.cuda(),
        paged_kv_indices_buf=ix.cuda(),
        paged_kv_last_page_len_buf=last.cuda(),
    )

    def plan(last):
        lengths = (ip[1:] - ip[:-1] - 1) * 16 + last
        w.plan(
            qo,
            ip,
            ix,
            last,
            8,
            2,
            128,
            16,
            causal=True,
            q_data_type=q.dtype,
            max_token_per_sequence=8,
            max_sequence_kv=48,
            seq_lens=None
            if length_dtype is None
            else lengths.to(q.device, length_dtype),
        )
        assert w._max_kv_len == 48

    def check(out, lse, last):
        ref, stats = _reference(q, k, v, qo, ip, ix, last, causal=True, scale=128**-0.5)
        torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
        torch.testing.assert_close(
            lse, stats * math.log2(math.e), atol=0.003, rtol=0.003
        )

    plan(last)
    out, lse = w.run(q, (k, v), return_lse=True)
    check(out, lse, last)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(q, (k, v), out=out, lse=lse, return_lse=True)
    last = last + 4
    plan(last)
    out.fill_(torch.nan)
    lse.fill_(torch.nan)
    graph.replay()
    check(out, lse, last)


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("lse_base", ["ln", "log2"])
@pytest.mark.parametrize("explicit_metadata", [False, True])
@pytest.mark.parametrize("lse_layout", ["NH", "HN"])
def test_paged_prefill_default_scale_layout_lse(
    layout, lse_base, explicit_metadata, lse_layout
):
    q, k, v, qo, ip, ix, last = _paged_inputs()
    last = last + 4
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, layout, backend="cudnn")
    metadata = {}
    if explicit_metadata:
        metadata = dict(
            seq_lens=torch.tensor([37, 21], device=q.device, dtype=torch.int32),
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
    out, lse = w.run(
        q, cache, return_lse=True, lse_base=lse_base, lse_layout=lse_layout
    )
    ref, lse_ref = _reference(q, k, v, qo, ip, ix, last, causal=True, scale=128**-0.5)
    torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
    torch.testing.assert_close(
        lse.T if lse_layout == "HN" else lse,
        lse_ref * (math.log2(math.e) if lse_base == "log2" else 1),
        atol=2e-3,
        rtol=2e-3,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(
            q,
            cache,
            out=out,
            lse=lse,
            return_lse=True,
            lse_base=lse_base,
            lse_layout=lse_layout,
        )
    lengths_ptr = w._seq_lens_kv.data_ptr()
    last = last - 2
    if explicit_metadata:
        metadata["seq_lens"].sub_(2)
    # Explicit GPU lengths and host-known bounds need no device readback.
    # Keep the real captured replay below as the metadata-lifetime check.
    previous_sync_mode = torch.cuda.get_sync_debug_mode()
    if explicit_metadata:
        torch.cuda.set_sync_debug_mode("error")
    try:
        w.plan(
            qo,
            ip,
            ix,
            last,
            8,
            2,
            128,
            16,
            causal=True,
            q_data_type=q.dtype,
            **metadata,
        )
    finally:
        torch.cuda.set_sync_debug_mode(previous_sync_mode)
    assert w._seq_lens_kv.data_ptr() == lengths_ptr
    out.fill_(torch.nan)
    lse.fill_(torch.nan)
    graph.replay()
    ref, lse_ref = _reference(q, k, v, qo, ip, ix, last, causal=True, scale=128**-0.5)
    torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)
    torch.testing.assert_close(
        lse.T if lse_layout == "HN" else lse,
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
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("length_shape", [(-1,), (-1, 1, 1, 1)])
def test_paged_prefill_plan_stages_device_lengths_without_sync(
    length_dtype, length_shape
):
    # Explicit device lengths with host-known bounds must be staged into the
    # wrapper-owned buffers on the device: a host readback would block plan()
    # until all queued GPU work drains (v0.7.1rc1 regression from #5350).
    q, k, v, qo, ip, ix, last = _paged_inputs()
    last = last + 4
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, device=q.device, dtype=torch.uint8),
        "NHD",
        backend="cudnn",
    )
    table = torch.tensor([[3, 0, 4], [2, 1, 0]], device=q.device, dtype=torch.int32)

    def device_lengths(last):
        kv = (ip[1:] - ip[:-1] - 1) * k.shape[1] + last
        return tuple(
            x.to(q.device, length_dtype).view(length_shape)
            for x in (kv, qo[1:] - qo[:-1])
        )

    def plan(last, kv_lens, q_lens):
        w.plan(
            qo,
            ip,
            ix,
            last,
            8,
            2,
            128,
            16,
            causal=True,
            q_data_type=q.dtype,
            seq_lens=kv_lens,
            seq_lens_q=q_lens,
            block_tables=table,
            max_token_per_sequence=3,
            max_sequence_kv=48,
        )

    def check(last):
        ref, _ = _reference(q, k, v, qo, ip, ix, last, causal=True, scale=128**-0.5)
        out = w.run(q, (k, v))
        torch.testing.assert_close(out.float(), ref, atol=0.015, rtol=0.015)

    plan(last, *device_lengths(last))
    check(last)
    kv_ptr, q_ptr = w._seq_lens_kv.data_ptr(), w._seq_lens_q.data_ptr()

    last = last - 2
    kv_lens, q_lens = device_lengths(last)
    torch.cuda.synchronize()
    # Keep the stream busy: a plan() that waits on the device returns only
    # after this event completes.
    torch.cuda._sleep(1 << 30)
    busy = torch.cuda.Event()
    busy.record()
    previous_sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        plan(last, kv_lens, q_lens)
    finally:
        torch.cuda.set_sync_debug_mode(previous_sync_mode)
    assert not busy.query(), "plan() waited for queued GPU work"
    # The wrapper keeps its own copies: clobbering the caller's tensors after
    # plan() (stream-ordered) must not change the planned lengths.
    assert (w._seq_lens_kv.data_ptr(), w._seq_lens_q.data_ptr()) == (kv_ptr, q_ptr)
    kv_lens.zero_()
    q_lens.zero_()
    del kv_lens, q_lens
    check(last)


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
    # FP8 inputs with BF16 output require SM10x; SM12x is not supported.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("cuDNN FP8-input/BF16-output prefill requires SM10x")
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


@pytest.mark.parametrize("backend", ["cudnn", "fa2"])
@pytest.mark.parametrize("declared_capacity", [None, 128])
def test_paged_graph_query_capacity_grows_and_replays(backend, declared_capacity):
    """A short first plan must not consume an explicitly larger graph capacity."""
    if backend == "cudnn" and not prefill.CUDNN_AVAILABLE:
        pytest.skip("requires cuDNN graph support")
    device = "cuda"
    q = torch.randn(128, 8, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(8, 16, 2, 128, device=device, dtype=q.dtype)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    ip = torch.tensor([0, 8], dtype=torch.int32)
    ix = torch.arange(8, device=device, dtype=torch.int32)
    last = torch.tensor([16], dtype=torch.int32)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device=device),
        backend=backend,
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty(2, device=device, dtype=torch.int32),
        paged_kv_indptr_buf=torch.empty(2, device=device, dtype=torch.int32),
        paged_kv_indices_buf=torch.empty_like(ix),
        paged_kv_last_page_len_buf=torch.empty(1, device=device, dtype=torch.int32),
        **(
            {"max_total_num_rows": declared_capacity}
            if declared_capacity is not None
            else {}
        ),
    )

    def plan(length):
        qo = torch.tensor([0, length], dtype=torch.int32)
        w.plan(qo, ip, ix, last, 8, 2, 128, 16, causal=True, q_data_type=q.dtype)
        return qo

    def check(length, qo):
        ref, _ = _reference(
            q[:length], k, v, qo, ip, ix, last, causal=True, scale=128**-0.5
        )
        torch.testing.assert_close(out[:length].float(), ref, atol=0.015, rtol=0.015)

    small_qo = plan(64)
    w.run(q[:64], (k, v), out=out[:64])
    check(64, small_qo)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            w.run(q[:64], (k, v), out=out[:64])
        if declared_capacity is None:
            with pytest.raises(ValueError, match="cannot exceed"):
                plan(128)
        else:
            large_qo = plan(128)
            w.run(q, (k, v), out=out)
            check(128, large_qo)
            with pytest.raises(ValueError, match="cannot exceed"):
                plan(129)
        small_qo = plan(64)
        q.mul_(0.5)
        v.mul_(0.75)
        out.fill_(torch.nan)
        graph.replay()
        check(64, small_qo)
        assert torch.isnan(out[64:]).all()
    finally:
        graph.reset()


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("mode", ["auto", "hn_declined", "legacy", "old_frontend"])
def test_prefill_hn_layout_capacity_switch_keeps_old_capture(monkeypatch, mode):
    """HN/NH graph keys and output bindings survive replan with fewer packed tokens."""
    if mode == "legacy":
        for module in (flashinfer.prefill, prefill):
            monkeypatch.setattr(
                module, "_cudnn_supports_direct_seqlens", lambda *a, **k: False
            )
    elif not prefill._cudnn_supports_direct_seqlens(torch.bfloat16, mixed=True):
        pytest.skip("requires direct sequence lengths")
    hn_attempts = []
    if mode == "hn_declined" and not flashinfer.prefill._CUDNN_NATIVE_HN_SUPPORTED:
        pytest.skip("native HN requires FE Stats stride override support")
    if mode == "old_frontend":
        monkeypatch.setattr(flashinfer.prefill, "_CUDNN_NATIVE_HN_SUPPORTED", False)
    if mode == "hn_declined":
        original = prefill._build_prefill_graph

        def build(*args, **kwargs):
            if kwargs.get("stats_head_stride", 0):
                hn_attempts.append(kwargs["stats_head_stride"])
                raise prefill.cudnn.cudnnGraphNotSupportedError(
                    "test engine declines HN"
                )
            return original(*args, **kwargs)

        monkeypatch.setattr(prefill, "_build_prefill_graph", build)
    q, k, v, qo, ip, ix, last = _paged_inputs()
    last = last + 4
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device=q.device),
        "NHD",
        backend="cudnn",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty_like(qo, device=q.device),
        paged_kv_indptr_buf=torch.empty_like(ip, device=q.device),
        paged_kv_indices_buf=torch.empty_like(ix, device=q.device),
        paged_kv_last_page_len_buf=torch.empty_like(last, device=q.device),
        max_total_num_rows=5,
    )

    def plan(offsets):
        w.plan(offsets, ip, ix, last, 8, 2, 128, 16, causal=True, q_data_type=q.dtype)

    def run(query, layout, **kwargs):
        return w.run(
            query, (k, v), return_lse=True, lse_base="ln", lse_layout=layout, **kwargs
        )

    plan(qo)
    out, lse = run(q, "HN")
    if mode == "auto":
        assert w._cudnn_prepared.stats_head_stride in (0, q.shape[0])
    elif mode in ("hn_declined", "old_frontend"):
        assert w._cudnn_prepared.stats_head_stride == 0
    else:
        assert w._cudnn_plan is None
    hn_graph = w._cudnn_prepared.graph if w._cudnn_prepared is not None else None
    native_hn = bool(
        w._cudnn_prepared is not None and w._cudnn_prepared.stats_head_stride
    )
    run(q, "HN", out=out, lse=lse)
    if mode == "hn_declined":
        assert hn_attempts == [5], "warm runs must not retry the unsupported layout"
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run(q, "HN", out=out, lse=lse)
        # Switch the prepared wrapper's layout and then its physical head stride.
        _, nh = run(q, "NH")
        torch.testing.assert_close(lse, nh.T, atol=2e-3, rtol=2e-3)
        qo2 = torch.tensor([0, 2, 4], dtype=torch.int32)
        plan(qo2)
        q.mul_(0.75)
        v.mul_(0.5)
        out2, lse2 = run(q[:4], "HN")
        if native_hn and w._cudnn_prepared.override_cache is not None:
            assert w._cudnn_prepared.graph is hn_graph
        ref, stats = _reference(
            q[:4], k, v, qo2, ip, ix, last, causal=True, scale=128**-0.5
        )
        torch.testing.assert_close(out2.float(), ref, atol=0.015, rtol=0.015)
        torch.testing.assert_close(lse2.T, stats, atol=2e-3, rtol=2e-3)
        out.fill_(torch.nan)
        lse.fill_(torch.nan)
        graph.replay()
        torch.testing.assert_close(out[:4].float(), ref, atol=0.015, rtol=0.015)
        torch.testing.assert_close(lse[:, :4].T, stats, atol=2e-3, rtol=2e-3)
        assert bool(torch.isnan(out[4:]).all())
        if native_hn:
            assert bool(torch.isnan(lse[:, 4:]).all())
    finally:
        graph.reset()


@pytest.mark.parametrize("backend", ["cudnn", "cutlass"])
def test_ragged_cpu_prefixes_avoid_sync_and_preserve_device_bindings(backend):
    if backend == "cutlass" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("CUTLASS prefill requires Blackwell")
    if backend == "cudnn" and not prefill.CUDNN_AVAILABLE:
        pytest.skip("requires cuDNN graph support")
    q = torch.randn(5, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(50, 8, 128, device=q.device, dtype=q.dtype)
    v = torch.randn_like(k)
    qo = torch.tensor([0, 3, 5], dtype=torch.int32)
    kv = torch.tensor([0, 33, 50], dtype=torch.int32)
    qo_gpu, kv_gpu = qo.cuda(), kv.cuda()
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 << 20, device=q.device, dtype=torch.uint8), backend=backend
    )
    kwargs = dict(q_data_type=q.dtype, qo_indptr_cpu=qo, kv_indptr_cpu=kv)
    w.plan(qo_gpu, kv_gpu, 8, 8, 128, **kwargs)
    w.run(q, k, v)
    for split in (2, 3):
        qo[1] = split
        qo_gpu.copy_(qo)
        torch.cuda.synchronize()
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            w.plan(qo_gpu, kv_gpu, 8, 8, 128, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        assert w._qo_indptr_buf.data_ptr() == qo_gpu.data_ptr()
        assert w._kv_indptr_buf.data_ptr() == kv_gpu.data_ptr()
        out, lse = w.run(q, k, v, return_lse=True, lse_base="ln")
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
        torch.testing.assert_close(lse, stats, atol=0.003, rtol=0.003)


@pytest.mark.parametrize("name", ["qo_indptr_cpu", "kv_indptr_cpu"])
@pytest.mark.parametrize("bad", ["device", "shape", "dtype"])
def test_ragged_rejects_invalid_cpu_prefix_mirror(name, bad):
    qo = torch.tensor([0, 3, 5], dtype=torch.int32)
    kv = torch.tensor([0, 33, 50], dtype=torch.int32)
    hints = dict(qo_indptr_cpu=qo, kv_indptr_cpu=kv)
    if bad == "device":
        hints[name] = hints[name].cuda()
    elif bad == "shape":
        hints[name] = hints[name][:2]
    else:
        hints[name] = hints[name].float()
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 << 20, device="cuda", dtype=torch.uint8), backend="cudnn"
    )
    with pytest.raises(ValueError, match=name):
        w.plan(qo.cuda(), kv.cuda(), 8, 8, 128, q_data_type=torch.bfloat16, **hints)


@pytest.mark.parametrize("counts", [[0, 3, 1], [257, 1, 0, 513], [0, 0, 0], [4097]])
@pytest.mark.parametrize("index_device", ["cpu", "cuda"])
@pytest.mark.parametrize("strided", [False, True])
def test_page_table_staging_capacity_and_strided_indices(counts, index_device, strided):
    from flashinfer.prefill import _build_block_tables_from_paged_kv_indices

    step = 2 if strided else 1
    offsets = torch.tensor([0, *counts], dtype=torch.int64).cumsum(0)
    host_storage = torch.full(((len(offsets) + 1) * step,), -1, dtype=torch.int64)
    host_storage[: len(offsets) * step : step] = offsets
    host_storage[-step] = int(offsets[-1]) + 3
    host = host_storage[::step]
    idx_storage = torch.arange(
        (sum(counts) + 1) * step, dtype=torch.int64, device=index_device
    )
    indices = idx_storage[::step][: sum(counts)]
    expected = torch.zeros((len(counts), max(counts)), dtype=torch.int32)
    for b, n in enumerate(counts):
        expected[b, :n] = indices.cpu()[int(offsets[b]) : int(offsets[b + 1])]
    got = _build_block_tables_from_paged_kv_indices(
        host, indices, len(counts), torch.device("cuda")
    )
    torch.testing.assert_close(got.cpu(), expected)
    # Reuse larger capacity, with nontrivial row stride and poisoned tails.
    backing = torch.full(
        (len(counts), max(counts) + 7), -777, device="cuda", dtype=torch.int32
    )
    out = backing[:, : max(counts) + 3]
    same = _build_block_tables_from_paged_kv_indices(
        host, indices, len(counts), torch.device("cuda"), out=out
    )
    assert same.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out[:, : max(counts)].cpu(), expected)
    assert torch.count_nonzero(out[:, max(counts) :]).item() == 0
    assert (backing[:, max(counts) + 3 :] == -777).all().item()


@pytest.mark.skipif(not prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lse_layout", ["NH", "HN"])
def test_paged_capture_replan_changes_indices_and_lengths(dtype, lse_layout):
    q, k, v, qo, ip, ix, last = _paged_inputs()
    q, k, v, ix = q.to(dtype), k.to(dtype), v.to(dtype), ix.cuda()
    out = torch.empty_like(q)
    lse = torch.empty((5, 8) if lse_layout == "NH" else (8, 5), device=q.device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, device=q.device, dtype=torch.uint8),
        backend="cudnn",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty_like(qo, device=q.device),
        paged_kv_indptr_buf=torch.empty_like(ip, device=q.device),
        paged_kv_indices_buf=torch.empty_like(ix),
        paged_kv_last_page_len_buf=torch.empty_like(last, device=q.device),
    )

    def plan():
        wrapper.plan(qo, ip, ix, last, 8, 2, 128, 16, causal=True, q_data_type=dtype)

    def run():
        wrapper.run(
            q,
            (k, v),
            out=out,
            lse=lse,
            return_lse=True,
            lse_layout=lse_layout,
            lse_base="ln",
        )

    plan()
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run()
        table_address = wrapper._cudnn_block_tables.data_ptr()
        # Swap the shorter/longer requests and replace the page IDs. Capacity
        # stays fixed, so the captured table must be updated in place.
        ip = torch.tensor([0, 2, 5], dtype=torch.int32)
        ix = torch.tensor([5, 2, 4, 3, 1], device=q.device, dtype=torch.int32)
        plan()
        assert wrapper._cudnn_block_tables.data_ptr() == table_address
        v.mul_(0.75)
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        expected, expected_lse = _reference(
            q, k, v, qo, ip, ix, last, causal=True, scale=128**-0.5
        )
        torch.testing.assert_close(out.float(), expected, atol=0.015, rtol=0.015)
        actual_lse = lse if lse_layout == "NH" else lse.T
        torch.testing.assert_close(actual_lse, expected_lse, atol=0.001, rtol=0.001)
    finally:
        graph.reset()
