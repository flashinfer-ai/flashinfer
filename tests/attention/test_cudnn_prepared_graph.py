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

"""Prepared graph reuse must preserve rebinding, replan, and workspace contracts."""

import math

import pytest
import torch

import flashinfer
from flashinfer.cudnn import prefill as cudnn_prefill

pytestmark = pytest.mark.skipif(
    not cudnn_prefill.CUDNN_AVAILABLE,
    reason="requires the cudnn-frontend python package (cuDNN graph backend)",
)


def _reference(q, k, v, q_lens, kv_lens, scale):
    outputs, stats = [], []
    q_start = kv_start = 0
    for lq, lkv in zip(q_lens, kv_lens, strict=True):
        qi = q[q_start : q_start + lq].float()
        ki = k[kv_start : kv_start + lkv].repeat_interleave(2, 1).float()
        vi = v[kv_start : kv_start + lkv].repeat_interleave(2, 1).float()
        scores = torch.einsum("qhd,khd->hqk", qi, ki) * scale
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vi))
        stats.append(scores.logsumexp(-1).T / math.log(2))
        q_start += lq
        kv_start += lkv
    return torch.cat(outputs), torch.cat(stats)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_prepared_history_alternating_shapes_and_retained_capture(paged, return_lse):
    """Revisit preparations with fresh bindings, then replay after their eviction."""
    if not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16, mixed=paged):
        pytest.skip("requires direct cuDNN cumulative sequence lengths")
    torch.manual_seed(79)
    workspace = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    cls = (
        flashinfer.BatchPrefillWithPagedKVCacheWrapper
        if paged
        else flashinfer.BatchPrefillWithRaggedKVCacheWrapper
    )
    wrapper = cls(workspace, "NHD", backend="cudnn")
    scale = 128**-0.5
    cases = []
    for batch in [1, 3]:
        q = torch.randn(batch * 4, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch * 16, 2, 128, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        cases.append(
            (q, k, v, torch.empty_like(q), torch.empty(batch * 4, 4, device="cuda"))
        )

    def plan(case, scale_value):
        q, k, v, _, _ = case
        batch = q.size(0) // 4
        qo = torch.arange(batch + 1, dtype=torch.int32) * 4
        kv = torch.arange(batch + 1, dtype=torch.int32) * 16
        if paged:
            wrapper.plan(
                qo,
                torch.arange(batch + 1, dtype=torch.int32),
                torch.arange(batch, dtype=torch.int32),
                torch.full((batch,), 16, dtype=torch.int32),
                4,
                2,
                128,
                16,
                q_data_type=q.dtype,
                sm_scale=scale_value,
                seq_lens=torch.full((batch,), 16, dtype=torch.int32, device="cuda"),
                seq_lens_q=torch.full((batch,), 4, dtype=torch.int32, device="cuda"),
                max_token_per_sequence=4,
                max_sequence_kv=16,
                block_tables=torch.arange(batch, dtype=torch.int32, device="cuda").view(
                    batch, 1
                ),
            )
        else:
            wrapper.plan(qo, kv, 4, 2, 128, q_data_type=q.dtype, sm_scale=scale_value)

    def run(case):
        q, k, v, out, lse = case
        kwargs = dict(out=out, return_lse=return_lse)
        if return_lse:
            kwargs["lse"] = lse
        if paged:
            wrapper.run(q, (k.view(-1, 16, 2, 128), v.view(-1, 16, 2, 128)), **kwargs)
        else:
            wrapper.run(q, k, v, **kwargs)

    def check(case, scale_value):
        q, k, v, out, lse = case
        batch = q.size(0) // 4
        ref, stats = _reference(q, k, v, [4] * batch, [16] * batch, scale_value)
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
        if return_lse:
            torch.testing.assert_close(lse, stats, atol=1e-3, rtol=1e-3)

    seen = {}
    for i in range(6):
        index = i % 2
        case = cases[index]
        # Replace runtime Q/K/V/O objects even when returning to the same shape.
        if i >= 2:
            case = cases[index] = tuple(t.clone() for t in case)
            case[0].mul_(0.75)
            case[2].neg_()
        plan(case, scale)
        run(case)
        check(case, scale)
        assert wrapper._cudnn_prepared is seen.setdefault(
            index, wrapper._cudnn_prepared
        )

    captured_case = cases[1]
    # Explicitly retain the metadata passed to the captured call. Eager wrappers
    # do not promise to update a previous capture's metadata on later plan().
    captured_metadata = wrapper._cudnn_plan
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        run(captured_case)
    for multiplier in [2, 3, 4, 5]:
        plan(cases[0], scale * multiplier)
        run(cases[0])
        check(cases[0], scale * multiplier)
    assert all(
        prepared not in wrapper._cudnn_prepared_cache for prepared in seen.values()
    )
    assert len(wrapper._cudnn_prepared_cache) == 4
    assert captured_metadata is not wrapper._cudnn_plan
    captured_case[0].mul_(0.5)
    captured_case[2].neg_()
    captured_case[3].fill_(torch.nan)
    captured_case[4].fill_(torch.nan)
    capture.replay()
    check(captured_case, scale)


@pytest.mark.parametrize("backend", ["auto", "cudnn"])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_first_ragged_auto_plan_preserves_captured_prefixes(
    monkeypatch, backend, graph_mode
):
    if not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("requires direct cuDNN cumulative sequence lengths")
    # Exercise auto resolving to cuDNN without pinning the selection heuristics.
    monkeypatch.setattr(
        flashinfer.prefill, "determine_attention_backend", lambda *a, **kw: "cudnn"
    )
    torch.manual_seed(53)
    q = torch.randn(20, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(42, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    qo = torch.tensor([0, 3, 20], dtype=torch.int32)
    kv = torch.tensor([0, 9, 42], dtype=torch.int32)
    buffers = (
        dict(qo_indptr_buf=qo.cuda(), kv_indptr_buf=kv.cuda()) if graph_mode else {}
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 << 20, device="cuda", dtype=torch.uint8),
        backend=backend,
        use_cuda_graph=graph_mode,
        **buffers,
    )

    def plan():
        wrapper.plan(qo, kv, 4, 2, 128, q_data_type=q.dtype)

    plan()
    assert wrapper._backend == "cudnn"
    ref, ref_lse = _reference(q, k, v, [3, 17], [9, 33], 128**-0.5)
    out, lse = wrapper.run(q, k, v, return_lse=True)
    prefixes = (wrapper._qo_indptr_buf.data_ptr(), wrapper._kv_indptr_buf.data_ptr())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, out=out, lse=lse, return_lse=True)
    plan()
    # Capture stores these addresses. Check their stability before freed storage
    # can be reused, making the old-code failure deterministic and safe.
    assert prefixes == (
        wrapper._qo_indptr_buf.data_ptr(),
        wrapper._kv_indptr_buf.data_ptr(),
    )
    rebound, rebound_lse = wrapper.run(q.clone(), k.clone(), -v, return_lse=True)
    torch.testing.assert_close(rebound.float(), -ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(rebound_lse, ref_lse, atol=1e-3, rtol=1e-3)
    v.mul_(0.5)
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref * 0.5, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "require_ordered", [False, True], ids=["compatible", "ordered"]
)
def test_prepared_ragged_interleaved_wrappers_capture_replan(
    monkeypatch, require_ordered
):
    """Shared graph-cache entries must not share shapes or changing bindings."""
    if not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("requires direct cuDNN cumulative sequence lengths")
    if require_ordered and not cudnn_prefill.supports_ordered_cudnn_execution(
        cudnn_prefill.cudnn.pygraph
    ):
        pytest.skip("requires FE ordered tensor execution support")
    torch.manual_seed(52)
    scale = 128**-0.5
    cases = []
    for q_lens, kv_lens in (([3, 14], [9, 22]), ([9, 2], [7, 16])):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(128 << 20, device="cuda", dtype=torch.uint8),
            backend="cudnn",
            use_cuda_graph=True,
            qo_indptr_buf=torch.empty(3, device="cuda", dtype=torch.int32),
            kv_indptr_buf=torch.empty(3, device="cuda", dtype=torch.int32),
        )
        q = torch.randn(sum(q_lens), 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(sum(kv_lens), 2, 128, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:2], device=q.device)
        cases.append((wrapper, q, k, v, out, lse, q_lens, kv_lens))

    def plan(case):
        wrapper, _, _, _, _, _, q_lens, kv_lens = case
        wrapper.plan(
            torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32),
            torch.tensor([0, kv_lens[0], sum(kv_lens)], dtype=torch.int32),
            4,
            2,
            128,
            causal=False,
            q_data_type=torch.bfloat16,
            sm_scale=scale,
        )

    def run(case):
        wrapper, q, k, v, out, lse, _, _ = case
        return wrapper.run(q, k, v, out=out, lse=lse, return_lse=True)

    def check(case):
        _, q, k, v, out, lse, q_lens, kv_lens = case
        ref, ref_lse = _reference(q, k, v, q_lens, kv_lens, scale)
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)

    # Prepare both before capture. Their shapes fall in the same override class.
    for case in cases:
        plan(case)
        run(case)
        check(case)
    if require_ordered:
        original_execute = cudnn_prefill.cudnn.pygraph.execute

        def require_tensor_sequence(self, tensors, *args, tensor_uids=None, **kwargs):
            assert tensor_uids is not None, "ordered execution fell back to a mapping"
            assert isinstance(tensors, (tuple, list))
            return original_execute(
                self, tensors, *args, tensor_uids=tensor_uids, **kwargs
            )

        # Include graph instances created after replan/workspace replacement.
        # The same consumer contract applies to native and FROST providers.
        monkeypatch.setattr(
            cudnn_prefill.cudnn.pygraph, "execute", require_tensor_sequence
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(cases[0])

    # A second wrapper and replacement workspace cannot mutate captured bindings.
    other = cases[1][0]
    other.reset_workspace_buffer(
        torch.empty_like(other._float_workspace_buffer), other._int_workspace_buffer
    )
    run(cases[1])
    check(cases[1])

    # Replan within the captured bounds, updating the registered indptr buffers.
    first = cases[0]
    first[-2].reverse()
    first[-1].reverse()
    plan(first)
    first[1].mul_(0.5)
    first[4].fill_(torch.nan)
    first[5].fill_(torch.nan)
    graph.replay()
    check(first)

    # The next eager call must use the caller's current stream, not capture's.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        first[3].neg_()
        run(first)
    torch.cuda.current_stream().wait_stream(stream)
    check(first)


@pytest.mark.parametrize("return_lse", [False, True])
def test_prepared_ragged_rebind_replan_workspace(return_lse):
    if not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("requires direct cuDNN cumulative sequence lengths")
    torch.manual_seed(42)
    q = torch.randn(12, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(40, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        backend="cudnn",
    )
    scale = 128**-0.5
    for q_lens, kv_lens, factor in (([7, 5], [17, 23], 1), ([4, 8], [21, 19], 2)):
        qo = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32)
        kv = torch.tensor([0, kv_lens[0], sum(kv_lens)], dtype=torch.int32)
        wrapper.plan(
            qo, kv, 4, 2, 128, q_data_type=torch.bfloat16, sm_scale=scale * factor
        )
        ref, ref_lse = _reference(q, k, v, q_lens, kv_lens, scale * factor)
        actual = wrapper.run(q, k, v, return_lse=return_lse)
        out, lse = actual if return_lse else (actual, None)
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
        if return_lse:
            torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)
        prepared = wrapper._cudnn_prepared
        assert prepared is not None
        # A second call rebinds all buffers while reusing the prepared graph.
        actual = wrapper.run(q, k, -v, return_lse=return_lse)
        out = actual[0] if return_lse else actual
        torch.testing.assert_close(out.float(), -ref, atol=1e-2, rtol=1e-2)
        assert wrapper._cudnn_prepared is prepared

    # Replacing the workspace must reconsider the prepared graph's requirement.
    # Backend plans may fall back to an exact graph; FROST can retain the same
    # override class when its workspace requirement fits the new allocation.
    if prepared.override_cache is not None:
        workspace_bytes = 64 * 1024
        guarded = torch.full(
            (3 * workspace_bytes,), 0xA5, dtype=torch.uint8, device="cuda"
        )
        wrapper.reset_workspace_buffer(
            guarded[workspace_bytes : 2 * workspace_bytes],
            wrapper._int_workspace_buffer,
        )
        actual = wrapper.run(q, k, v, return_lse=return_lse)
        assert wrapper._cudnn_prepared is not prepared
        assert wrapper._cudnn_prepared.graph.get_workspace_size() <= workspace_bytes
        assert torch.all(guarded[:workspace_bytes] == 0xA5)
        assert torch.all(guarded[2 * workspace_bytes :] == 0xA5)
        out, lse = actual if return_lse else (actual, None)
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
        if return_lse:
            torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("return_lse", [False, True])
def test_prepared_decode_rebind_replan(return_lse):
    torch.manual_seed(42)
    q = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(8, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="cudnn",
    )
    scale = 128**-0.5
    for indices, factor in (([2, 0, 7, 1], 1), ([6, 3, 1, 5], 2)):
        wrapper.plan(
            torch.tensor([0, 2, 4], dtype=torch.int32),
            torch.tensor(indices, dtype=torch.int32),
            torch.tensor([15, 9], dtype=torch.int32),
            4,
            2,
            128,
            16,
            q_data_type=torch.bfloat16,
            sm_scale=scale * factor,
        )
        actual = wrapper.run(q, (k, v), return_lse=return_lse)
        packed_k = torch.cat(
            [k[indices[:2]].flatten(0, 1)[:31], k[indices[2:]].flatten(0, 1)[:25]]
        )
        packed_v = torch.cat(
            [v[indices[:2]].flatten(0, 1)[:31], v[indices[2:]].flatten(0, 1)[:25]]
        )
        ref, ref_lse = _reference(
            q, packed_k, packed_v, [1, 1], [31, 25], scale * factor
        )
        out, lse = actual if return_lse else (actual, None)
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
        if return_lse:
            torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)
        prepared = wrapper._cudnn_prepared
        assert prepared is not None
        actual = wrapper.run(q, (k, -v), return_lse=return_lse)
        out = actual[0] if return_lse else actual
        torch.testing.assert_close(out.float(), -ref, atol=1e-2, rtol=1e-2)
        assert wrapper._cudnn_prepared is prepared

    wrapper.reset_workspace_buffer(
        torch.empty_like(wrapper._float_workspace_buffer),
        wrapper._int_workspace_buffer,
    )
    assert wrapper._cudnn_prepared is None
    actual = wrapper.run(q, (k, v), return_lse=return_lse)
    out, lse = actual if return_lse else (actual, None)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    if return_lse:
        torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "rebound", ["q", "k", "v", "workspace", "block_tables", "kv_lens", "seq_lens_q"]
)
def test_prepared_decode_rejects_wrong_device_before_execute(monkeypatch, rebound):
    q = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(4, 16, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="cudnn",
    )
    wrapper.plan(
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.arange(4, dtype=torch.int32),
        torch.tensor([15, 9], dtype=torch.int32),
        4,
        2,
        128,
        16,
        q_data_type=q.dtype,
    )
    wrapper.run(q, (k, v), enable_pdl=False)
    prepared = wrapper._cudnn_prepared
    assert prepared is not None

    def unexpected_execute(*args, **kwargs):
        pytest.fail("a buffer on the wrong device reached graph.execute")

    monkeypatch.setattr(prepared.graph, "execute", unexpected_execute)
    if rebound == "q":
        q = q.cpu()
    elif rebound == "k":
        k = k.cpu()
    elif rebound == "v":
        v = v.cpu()
    elif rebound == "workspace":
        wrapper.reset_workspace_buffer(
            torch.empty(128 * 1024, dtype=torch.uint8), wrapper._int_workspace_buffer
        )
    elif rebound == "block_tables":
        wrapper._block_tables = wrapper._block_tables.cpu()
    elif rebound == "kv_lens":
        wrapper._cudnn_kv_lens_view = wrapper._cudnn_kv_lens_view.cpu()
    else:
        prepared.seq_lens_q = prepared.seq_lens_q.cpu()
    with pytest.raises(ValueError, match="same device"):
        wrapper.run(q, (k, v), enable_pdl=False)


@pytest.mark.parametrize("warmed", [False, True])
@pytest.mark.parametrize("buffer", ["out", "lse"])
def test_prepared_ragged_rejects_noncontiguous_buffers(monkeypatch, warmed, buffer):
    q = torch.randn(6, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(10, 2, 128, device="cuda", dtype=q.dtype)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        backend="cudnn",
    )
    wrapper.plan(
        torch.tensor([0, 2, 6], dtype=torch.int32),
        torch.tensor([0, 3, 10], dtype=torch.int32),
        4,
        2,
        128,
        q_data_type=q.dtype,
    )
    if warmed:
        wrapper.run(q, k, k, return_lse=True)
        prepared = wrapper._cudnn_prepared
        assert prepared is not None

        def unexpected_execute(*args, **kwargs):
            pytest.fail("a noncontiguous output buffer reached graph.execute")

        monkeypatch.setattr(prepared.graph, "execute", unexpected_execute)
    if buffer == "out":
        storage = torch.full((6, 4, 256), 123, device="cuda", dtype=q.dtype)
        invalid = storage[..., ::2]
    else:
        storage = torch.full((6, 8), 123, device="cuda", dtype=torch.float32)
        invalid = storage[:, ::2]
    with pytest.raises(ValueError, match=f"{buffer} must be contiguous"):
        wrapper.run(q, k, k, return_lse=True, **{buffer: invalid})
    assert torch.all(storage == 123)


@pytest.mark.parametrize("buffer", ["out", "lse"])
def test_cudnn_paged_rejects_noncontiguous_buffers_before_dispatch(monkeypatch, buffer):
    q = torch.randn(6, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 16, 2, 128, device="cuda", dtype=q.dtype)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="cudnn",
    )
    wrapper.plan(
        torch.tensor([0, 2, 6], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([3, 7], dtype=torch.int32),
        4,
        2,
        128,
        16,
        q_data_type=q.dtype,
        seq_lens=torch.tensor([3, 7], dtype=torch.int32, device="cuda"),
        seq_lens_q=torch.tensor([2, 4], dtype=torch.int32, device="cuda"),
        max_token_per_sequence=4,
        max_sequence_kv=7,
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32, device="cuda"),
    )

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("a noncontiguous output buffer reached cuDNN dispatch")

    monkeypatch.setattr(
        flashinfer.prefill, "_reuse_cudnn_prefill_preparation", unexpected_dispatch
    )
    monkeypatch.setattr(
        flashinfer.prefill, "cudnn_batch_prefill_with_kv_cache", unexpected_dispatch
    )
    if buffer == "out":
        storage = torch.full((6, 4, 256), 123, device="cuda", dtype=q.dtype)
        invalid = storage[..., ::2]
    else:
        storage = torch.full((6, 8), 123, device="cuda", dtype=torch.float32)
        invalid = storage[:, ::2]
    with pytest.raises(ValueError, match=f"{buffer} must be contiguous"):
        wrapper.run(q, (k, k), return_lse=True, **{buffer: invalid})
    assert torch.all(storage == 123)
