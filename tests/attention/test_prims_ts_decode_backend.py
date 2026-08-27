# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ``backend="prims-ts"`` path of BatchDecodeWithPagedKVCacheWrapper.

The rejection tests run on any CUDA device because they fire before the kernel
is planned. The numerical tests need SM100a, where the kernel is qualified.
"""

import math

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="prims-ts decode requires nvidia-cutlass-dsl==4.7.0",
)

import flashinfer  # noqa: E402
from flashinfer.utils import is_sm100a_supported  # noqa: E402


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

requires_prims_ts_gpu = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0)
    or not is_sm100a_supported(torch.device("cuda")),
    reason="prims-ts decode is qualified on SM100a",
)


NUM_QO_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 16


def _make_page_table(kv_lens, page_size, device):
    """Build contiguous CSR page metadata covering the given kv lengths."""

    pages_per_req = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    offsets = [0]
    for num_pages in pages_per_req:
        offsets.append(offsets[-1] + num_pages)
    indptr = torch.tensor(offsets, dtype=torch.int32, device=device)
    indices = torch.arange(sum(pages_per_req), dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        [
            kv_len - (n - 1) * page_size
            for kv_len, n in zip(kv_lens, pages_per_req, strict=True)
        ],
        dtype=torch.int32,
        device=device,
    )
    return indptr, indices, last_page_len


def _make_wrapper(backend, kv_layout="HND", device="cuda", **kwargs):
    workspace = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout, backend=backend, **kwargs
    )


def _plan_args(kv_lens, device, page_size=PAGE_SIZE):
    indptr, indices, last_page_len = _make_page_table(kv_lens, page_size, device)
    return (
        indptr,
        indices,
        last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
    )


def _reference_decode(
    q, k_cache, v_cache, kv_lens, q_len_per_req, is_causal, page_size
):
    """Dense or bottom-right causal paged decode over HND k/v pages."""

    batch_size = len(kv_lens)
    group_size = NUM_QO_HEADS // NUM_KV_HEADS
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    q = q.view(batch_size, q_len_per_req, NUM_QO_HEADS, HEAD_DIM).float()
    out = torch.empty_like(q)
    page_id = 0
    for b, kv_len in enumerate(kv_lens):
        num_pages = (kv_len + page_size - 1) // page_size
        pages = slice(page_id, page_id + num_pages)
        page_id += num_pages
        # HND pages are [num_pages, num_kv_heads, page_size, head_dim].
        k = k_cache[pages].permute(1, 0, 2, 3).reshape(NUM_KV_HEADS, -1, HEAD_DIM)
        v = v_cache[pages].permute(1, 0, 2, 3).reshape(NUM_KV_HEADS, -1, HEAD_DIM)
        k = k[:, :kv_len].float()
        v = v[:, :kv_len].float()
        for h in range(NUM_QO_HEADS):
            kv_h = h // group_size
            scores = (q[b, :, h] @ k[kv_h].transpose(0, 1)) * sm_scale
            if is_causal:
                rows = torch.arange(q_len_per_req, device=q.device).unsqueeze(1)
                cols = torch.arange(kv_len, device=q.device).unsqueeze(0)
                scores = scores.masked_fill(
                    cols > kv_len - q_len_per_req + rows, float("-inf")
                )
            out[b, :, h] = torch.softmax(scores, dim=-1) @ v[kv_h]
    return out.view(batch_size * q_len_per_req, NUM_QO_HEADS, HEAD_DIM)


def _make_cache(kv_lens, dtype, device, page_size=PAGE_SIZE):
    num_pages = sum((kv_len + page_size - 1) // page_size for kv_len in kv_lens)
    shape = (num_pages, NUM_KV_HEADS, page_size, HEAD_DIM)
    k = torch.randn(shape, device=device).to(dtype)
    v = torch.randn(shape, device=device).to(dtype)
    return k, v


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


@requires_cuda
def test_rejects_nhd_layout():
    with pytest.raises(NotImplementedError, match="kv_layout"):
        _make_wrapper("prims-ts", kv_layout="NHD")


@requires_cuda
def test_rejects_jit_args():
    with pytest.raises(NotImplementedError, match="jit_args"):
        _make_wrapper("prims-ts", jit_args=[1])


def _graph_buffers(batch_size, max_pages, device="cuda"):
    return dict(
        paged_kv_indptr_buffer=torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        ),
        paged_kv_indices_buffer=torch.zeros(
            max_pages, dtype=torch.int32, device=device
        ),
        paged_kv_last_page_len_buffer=torch.zeros(
            batch_size, dtype=torch.int32, device=device
        ),
    )


@requires_prims_ts_gpu
def test_graph_plan_rejects_small_workspace():
    workspace = torch.zeros(64, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "HND",
        backend="prims-ts",
        use_cuda_graph=True,
        **_graph_buffers(2, max_pages=64),
    )
    with pytest.raises(ValueError, match="workspace bytes"):
        wrapper.plan(*_plan_args([32, 48], "cuda"), q_data_type=torch.bfloat16)


@requires_cuda
def test_rejects_fast_decode_plan():
    wrapper = _make_wrapper("prims-ts")
    with pytest.raises(NotImplementedError, match="fast_decode_plan"):
        flashinfer.decode.fast_decode_plan(
            wrapper,
            *_plan_args([32, 48], "cuda"),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )


@requires_cuda
def test_rejects_mismatched_seq_lens():
    wrapper = _make_wrapper("prims-ts")
    with pytest.raises(ValueError, match="seq_lens must match"):
        wrapper.plan(
            *_plan_args([32, 48], "cuda"),
            q_data_type=torch.bfloat16,
            seq_lens=torch.tensor([32, 40], dtype=torch.int32, device="cuda"),
        )


def test_plan_trace_captures_explicit_causal_mode():
    """The plan trace includes kwargs passed through compatibility API."""

    plan_trace = flashinfer.BatchDecodeWithPagedKVCacheWrapper.plan.fi_trace
    indptr, indices, last_page_len, *_ = _plan_args([2, 3], "cpu")
    definition = plan_trace(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=PAGE_SIZE,
        kwargs={"q_len_per_req": 4, "is_causal": False},
    )

    assert definition["op_type"] == "gqa_paged_plan"
    assert definition["inputs"]["q_len_per_req"]["optional"] is True
    assert definition["inputs"]["is_causal"] == {
        "shape": None,
        "dtype": "bool",
        "optional": True,
        "description": "Whether the planned attention mask is causal.",
    }


@requires_cuda
def test_workspace_size_rejects_prims_ts():
    wrapper = _make_wrapper("prims-ts")
    with pytest.raises(NotImplementedError, match="prims-ts"):
        wrapper.workspace_size(
            *_plan_args([2, 3], "cuda"),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            q_len_per_req=4,
        )


@requires_cuda
def test_rejects_logits_soft_cap():
    wrapper = _make_wrapper("prims-ts")
    with pytest.raises(NotImplementedError, match="logits_soft_cap"):
        wrapper.plan(*_plan_args([64, 96], "cuda"), logits_soft_cap=30.0)


@requires_cuda
def test_rejects_pos_encoding_mode():
    wrapper = _make_wrapper("prims-ts")
    with pytest.raises(NotImplementedError, match="pos_encoding_mode"):
        wrapper.plan(*_plan_args([64, 96], "cuda"), pos_encoding_mode="ROPE_LLAMA")


@requires_cuda
@pytest.mark.parametrize(
    "kwargs", [{"fixed_split_size": 4}, {"disable_split_kv": True}]
)
def test_rejects_split_kv_knobs(kwargs):
    wrapper = _make_wrapper("prims-ts")
    with pytest.raises(NotImplementedError, match="split-kv"):
        wrapper.plan(*_plan_args([64, 96], "cuda"), **kwargs)


@requires_cuda
def test_rejects_mixed_q_kv_dtype():
    wrapper = _make_wrapper("prims-ts")
    with pytest.raises(NotImplementedError, match="q_data_type == kv_data_type"):
        wrapper.plan(
            *_plan_args([64, 96], "cuda"),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
        )


@requires_cuda
@pytest.mark.parametrize("q_len_per_req,is_causal", [(4, False), (1, True)])
def test_explicit_is_causal_rejected_on_other_backends(q_len_per_req, is_causal):
    wrapper = _make_wrapper("fa2", kv_layout="NHD")
    with pytest.raises(NotImplementedError, match="is_causal"):
        wrapper.plan(
            *_plan_args([64, 96], "cuda"),
            q_data_type=torch.bfloat16,
            q_len_per_req=q_len_per_req,
            is_causal=is_causal,
        )


@requires_cuda
def test_default_is_causal_leaves_other_backends_unchanged():
    kv_lens = [64, 96]
    torch.manual_seed(0)
    k_cache, v_cache = _make_cache(kv_lens, torch.bfloat16, "cuda")
    kv_cache = (
        k_cache.transpose(-3, -2).contiguous(),
        v_cache.transpose(-3, -2).contiguous(),
    )
    q = torch.randn(
        len(kv_lens), NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )

    outs = []
    for is_causal in (None, False):
        wrapper = _make_wrapper("fa2", kv_layout="NHD")
        wrapper.plan(
            *_plan_args(kv_lens, "cuda"),
            q_data_type=torch.bfloat16,
            is_causal=is_causal,
        )
        outs.append(wrapper.run(q, kv_cache))
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@requires_prims_ts_gpu
@pytest.mark.parametrize("q_len_per_req", [1, 4])
@pytest.mark.parametrize("is_causal", [True, False])
def test_matches_reference(q_len_per_req, is_causal):
    kv_lens = [64, 96, 128]
    torch.manual_seed(0)
    k_cache, v_cache = _make_cache(kv_lens, torch.bfloat16, "cuda")
    q = torch.randn(
        len(kv_lens) * q_len_per_req,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    wrapper = _make_wrapper("prims-ts")
    wrapper.plan(
        *_plan_args(kv_lens, "cuda"),
        q_data_type=torch.bfloat16,
        q_len_per_req=q_len_per_req,
        is_causal=is_causal,
    )
    out = wrapper.run(q, (k_cache, v_cache))

    reference = _reference_decode(
        q, k_cache, v_cache, kv_lens, q_len_per_req, is_causal, PAGE_SIZE
    )
    torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)


@requires_prims_ts_gpu
def test_dense_and_causal_differ():
    kv_lens = [128, 128]
    q_len_per_req = 4
    torch.manual_seed(0)
    k_cache, v_cache = _make_cache(kv_lens, torch.bfloat16, "cuda")
    q = torch.randn(
        len(kv_lens) * q_len_per_req,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    outs = []
    for is_causal in (True, False):
        wrapper = _make_wrapper("prims-ts")
        wrapper.plan(
            *_plan_args(kv_lens, "cuda"),
            q_data_type=torch.bfloat16,
            q_len_per_req=q_len_per_req,
            is_causal=is_causal,
        )
        outs.append(wrapper.run(q, (k_cache, v_cache)))
    # Only the last query row shares the same visible range under both masks.
    assert not torch.allclose(outs[0][0], outs[1][0], rtol=1e-2, atol=1e-2)


@requires_prims_ts_gpu
def test_dense_allows_kv_len_below_q_len():
    kv_lens = [2, 3]
    q_len_per_req = 4
    torch.manual_seed(0)
    k_cache, v_cache = _make_cache(kv_lens, torch.bfloat16, "cuda")
    q = torch.randn(
        len(kv_lens) * q_len_per_req,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    wrapper = _make_wrapper("prims-ts")
    wrapper.plan(
        *_plan_args(kv_lens, "cuda"),
        q_data_type=torch.bfloat16,
        q_len_per_req=q_len_per_req,
        is_causal=False,
    )
    out = wrapper.run(q, (k_cache, v_cache))

    reference = _reference_decode(
        q, k_cache, v_cache, kv_lens, q_len_per_req, False, PAGE_SIZE
    )
    torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_sliding_window_requires_causal():
    wrapper = _make_wrapper("prims-ts")
    with pytest.raises(ValueError, match="window_left"):
        wrapper.plan(
            *_plan_args([64, 96], "cuda"),
            q_data_type=torch.bfloat16,
            q_len_per_req=4,
            is_causal=False,
            window_left=16,
        )


@requires_prims_ts_gpu
@pytest.mark.parametrize(
    "run_kwargs,match",
    [
        ({"return_lse": True}, "LSE"),
        ({"skip_softmax_threshold_scale_factor": 1.0}, "skip_softmax"),
        ({"sinks": torch.zeros(NUM_QO_HEADS)}, "sinks"),
    ],
)
def test_run_rejects_unsupported_options(run_kwargs, match):
    kv_lens = [64, 96]
    torch.manual_seed(0)
    k_cache, v_cache = _make_cache(kv_lens, torch.bfloat16, "cuda")
    q = torch.randn(
        len(kv_lens), NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )

    wrapper = _make_wrapper("prims-ts")
    wrapper.plan(*_plan_args(kv_lens, "cuda"), q_data_type=torch.bfloat16)
    with pytest.raises(NotImplementedError, match=match):
        wrapper.run(q, (k_cache, v_cache), **run_kwargs)


@requires_prims_ts_gpu
def test_matching_seq_lens_accepted():
    kv_lens = [32, 48]
    wrapper = _make_wrapper("prims-ts")
    wrapper.plan(
        *_plan_args(kv_lens, "cuda"),
        q_data_type=torch.bfloat16,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
    )
    q = torch.randn(
        len(kv_lens), NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    k_cache, v_cache = _make_cache(kv_lens, torch.bfloat16, "cuda")
    out = wrapper.run(q, (k_cache, v_cache))
    assert out.shape == q.shape


@requires_prims_ts_gpu
def test_rejects_noncontiguous_multi_q_tensors():
    kv_lens = [64, 96]
    q_len_per_req = 4
    wrapper = _make_wrapper("prims-ts")
    wrapper.plan(
        *_plan_args(kv_lens, "cuda"),
        q_data_type=torch.bfloat16,
        q_len_per_req=q_len_per_req,
        is_causal=False,
    )
    k_cache, v_cache = _make_cache(kv_lens, torch.bfloat16, "cuda")
    num_tokens = len(kv_lens) * q_len_per_req
    q = torch.randn(
        num_tokens, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    q_strided = torch.randn(
        num_tokens, HEAD_DIM, NUM_QO_HEADS, dtype=torch.bfloat16, device="cuda"
    ).transpose(1, 2)
    assert not q_strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        wrapper.run(q_strided, (k_cache, v_cache))
    out_strided = torch.empty_like(q_strided)
    with pytest.raises(ValueError, match="contiguous"):
        wrapper.run(q, (k_cache, v_cache), out=out_strided)


@requires_prims_ts_gpu
@pytest.mark.parametrize("q_len_per_req,is_causal", [(1, False), (4, False), (4, True)])
def test_cuda_graph_replan_then_replay(q_len_per_req, is_causal):
    kv_lens_sets = [[64, 96], [40, 200], [128, 128]]
    max_pages = 64
    torch.manual_seed(0)
    k_cache, v_cache = _make_cache(
        [max_pages * PAGE_SIZE // 2] * 2, torch.bfloat16, "cuda"
    )
    q = torch.randn(
        2 * q_len_per_req, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    wrapper = _make_wrapper(
        "prims-ts", use_cuda_graph=True, **_graph_buffers(2, max_pages=max_pages)
    )

    def plan(kv_lens):
        wrapper.plan(
            *_plan_args(kv_lens, "cuda"),
            q_data_type=torch.bfloat16,
            q_len_per_req=q_len_per_req,
            is_causal=is_causal,
        )

    def reference(kv_lens):
        return _reference_decode(
            q, k_cache, v_cache, kv_lens, q_len_per_req, is_causal, PAGE_SIZE
        )

    plan(kv_lens_sets[0])
    out = torch.empty_like(q)
    wrapper.run(q, (k_cache, v_cache), out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, (k_cache, v_cache), out=out)

    for kv_lens in kv_lens_sets:
        plan(kv_lens)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.float(), reference(kv_lens), rtol=2e-2, atol=2e-2
        )


@requires_prims_ts_gpu
def test_cuda_graph_matches_eager_path():
    kv_lens = [64, 96]
    torch.manual_seed(0)
    k_cache, v_cache = _make_cache(kv_lens, torch.bfloat16, "cuda")
    q = torch.randn(2, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    plan_args = _plan_args(kv_lens, "cuda")

    eager = _make_wrapper("prims-ts")
    eager.plan(*plan_args, q_data_type=torch.bfloat16)
    expected = eager.run(q, (k_cache, v_cache))

    graph_wrapper = _make_wrapper(
        "prims-ts", use_cuda_graph=True, **_graph_buffers(2, max_pages=32)
    )
    graph_wrapper.plan(*plan_args, q_data_type=torch.bfloat16)
    got = graph_wrapper.run(q, (k_cache, v_cache))
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


@requires_prims_ts_gpu
def test_graph_capture_of_fixed_plan_replays():
    """Eager-mode wrapper: capturing run() after a plan replays that plan."""

    kv_lens = [64, 96]
    q_len_per_req = 4
    torch.manual_seed(0)
    k_cache, v_cache = _make_cache(kv_lens, torch.bfloat16, "cuda")
    q = torch.randn(
        len(kv_lens) * q_len_per_req,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    wrapper = _make_wrapper("prims-ts")
    wrapper.plan(
        *_plan_args(kv_lens, "cuda"),
        q_data_type=torch.bfloat16,
        q_len_per_req=q_len_per_req,
        is_causal=False,
    )
    eager = wrapper.run(q, (k_cache, v_cache))

    out = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, (k_cache, v_cache), out=out)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
