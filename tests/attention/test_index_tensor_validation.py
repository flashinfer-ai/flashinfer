"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import itertools

import pytest
import torch
from tests.test_helpers.jit_utils import gen_prefill_attention_modules

import flashinfer
from flashinfer.utils import has_flashinfer_jit_cache, is_sm90a_supported

# Issue #1654: an index tensor whose dtype disagrees with the compiled module
# used to be misread by the kernel instead of rejected. One kernel variant
# (fp16, head_dim 128) keeps JIT time small; most tests raise before any
# module is fetched.

NUM_QO_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 16
DTYPE = torch.float16


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_prefill_attention_modules(
            [DTYPE],  # q_dtypes
            [DTYPE],  # kv_dtypes
            [HEAD_DIM],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


def _workspace_buffer(device="cuda:0"):
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)


def _make_paged_inputs(device="cuda:0"):
    """A tiny 3-request paged-KV problem, mirroring the issue #1654 repro."""
    kv_lens = [40, 5, 33]
    qo_lens = [4, 1, 7]
    num_pages = [(l + PAGE_SIZE - 1) // PAGE_SIZE for l in kv_lens]
    total_pages = sum(num_pages)

    qo_indptr = torch.tensor(
        [0] + list(itertools.accumulate(qo_lens)), dtype=torch.int32, device=device
    )
    kv_indptr = torch.tensor(
        [0] + list(itertools.accumulate(num_pages)), dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        [((l - 1) % PAGE_SIZE) + 1 for l in kv_lens], dtype=torch.int32, device=device
    )
    kv_cache = (
        torch.randn(
            total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        ),
        torch.randn(
            total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        ),
    )
    q = torch.randn(sum(qo_lens), NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return {
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "last_page_len": last_page_len,
        "kv_cache": kv_cache,
        "q": q,
        "batch_size": len(kv_lens),
    }


def _make_ragged_inputs(device="cuda:0"):
    qo_lens = [4, 1, 7]
    kv_lens = [5, 2, 6]
    qo_indptr = torch.tensor(
        [0] + list(itertools.accumulate(qo_lens)), dtype=torch.int32, device=device
    )
    kv_indptr = torch.tensor(
        [0] + list(itertools.accumulate(kv_lens)), dtype=torch.int32, device=device
    )
    q = torch.randn(sum(qo_lens), NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    k = torch.randn(sum(kv_lens), NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    v = torch.randn(sum(kv_lens), NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return {"qo_indptr": qo_indptr, "kv_indptr": kv_indptr, "q": q, "k": k, "v": v}


def _strided_copy(x: torch.Tensor) -> torch.Tensor:
    """Return a non-contiguous view holding the same values as ``x``."""
    buf = torch.full((2 * x.shape[0],), -1, dtype=x.dtype, device=x.device)
    buf[0::2] = x
    strided = buf[0::2]
    assert not strided.is_contiguous()
    assert torch.equal(strided, x)
    return strided


# ---------------------------------------------------------------------------
# Regression: mixed int32/int64 last_page_len (the exact issue #1654 repro).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_paged_prefill_rejects_int64_last_page_len(backend):
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend=backend
    )
    with pytest.raises(ValueError, match="paged_kv_last_page_len must be torch.int32"):
        wrapper.plan(
            p["qo_indptr"],
            p["kv_indptr"],
            p["kv_indices"],
            p["last_page_len"].to(torch.int64),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            causal=True,
        )


@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_decode_rejects_int64_last_page_len(use_tensor_cores):
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", use_tensor_cores=use_tensor_cores
    )
    with pytest.raises(ValueError, match="last_page_len must be torch.int32"):
        wrapper.plan(
            p["kv_indptr"],
            p["kv_indices"],
            p["last_page_len"].to(torch.int64),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )


# ---------------------------------------------------------------------------
# Other index tensors, wrong lengths, and wrong rank.
# ---------------------------------------------------------------------------


def test_paged_prefill_rejects_int64_qo_indptr():
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match="qo_indptr must be torch.int32"):
        wrapper.plan(
            p["qo_indptr"].to(torch.int64),
            p["kv_indptr"],
            p["kv_indices"],
            p["last_page_len"],
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            causal=True,
        )


def test_paged_prefill_rejects_int64_paged_kv_indptr():
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match="paged_kv_indptr must be torch.int32"):
        wrapper.plan(
            p["qo_indptr"],
            p["kv_indptr"].to(torch.int64),
            p["kv_indices"],
            p["last_page_len"],
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            causal=True,
        )


def test_paged_prefill_rejects_int64_paged_kv_indices():
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match="paged_kv_indices must be torch.int32"):
        wrapper.plan(
            p["qo_indptr"],
            p["kv_indptr"],
            p["kv_indices"].to(torch.int64),
            p["last_page_len"],
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            causal=True,
        )


def test_paged_prefill_rejects_wrong_length_paged_kv_indptr():
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match="paged_kv_indptr must have length"):
        wrapper.plan(
            p["qo_indptr"],
            p["kv_indptr"][:-1].contiguous(),
            p["kv_indices"],
            p["last_page_len"],
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            causal=True,
        )


def test_paged_prefill_rejects_2d_last_page_len():
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match="must be a 1D tensor"):
        wrapper.plan(
            p["qo_indptr"],
            p["kv_indptr"],
            p["kv_indices"],
            p["last_page_len"].unsqueeze(-1),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            causal=True,
        )


def test_ragged_prefill_rejects_int64_qo_indptr():
    r = _make_ragged_inputs()
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match="qo_indptr must be torch.int32"):
        wrapper.plan(
            r["qo_indptr"].to(torch.int64),
            r["kv_indptr"],
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            causal=True,
        )


def test_ragged_prefill_rejects_int64_kv_indptr():
    r = _make_ragged_inputs()
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match="kv_indptr must be torch.int32"):
        wrapper.plan(
            r["qo_indptr"],
            r["kv_indptr"].to(torch.int64),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            causal=True,
        )


def test_ragged_prefill_rejects_wrong_length_kv_indptr():
    r = _make_ragged_inputs()
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match="kv_indptr must have length"):
        wrapper.plan(
            r["qo_indptr"],
            torch.cat([r["kv_indptr"], r["kv_indptr"][-1:]]),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            causal=True,
        )


def test_decode_rejects_int64_indptr():
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", use_tensor_cores=True
    )
    with pytest.raises(ValueError, match="indptr must be torch.int32"):
        wrapper.plan(
            p["kv_indptr"].to(torch.int64),
            p["kv_indices"],
            p["last_page_len"],
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )


def test_decode_rejects_int64_indices():
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", use_tensor_cores=True
    )
    with pytest.raises(ValueError, match="indices must be torch.int32"):
        wrapper.plan(
            p["kv_indptr"],
            p["kv_indices"].to(torch.int64),
            p["last_page_len"],
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )


def test_decode_rejects_wrong_length_indptr():
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", use_tensor_cores=True
    )
    with pytest.raises(ValueError, match="indptr must have length"):
        wrapper.plan(
            torch.cat([p["kv_indptr"], p["kv_indptr"][-1:]]),
            p["kv_indices"],
            p["last_page_len"],
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )


# ---------------------------------------------------------------------------
# Non-contiguous plan() inputs must be silently made contiguous, not rejected:
# the kernel output must match a contiguous run bit-for-bit.
# ---------------------------------------------------------------------------


def test_paged_prefill_strided_last_page_len_matches_contiguous():
    p = _make_paged_inputs()
    strided = _strided_copy(p["last_page_len"])

    ref_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend="fa2"
    )
    ref_wrapper.plan(
        p["qo_indptr"],
        p["kv_indptr"],
        p["kv_indices"],
        p["last_page_len"],
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
    )
    ref = ref_wrapper.run(p["q"], p["kv_cache"])

    strided_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend="fa2"
    )
    strided_wrapper.plan(
        p["qo_indptr"],
        p["kv_indptr"],
        p["kv_indices"],
        strided,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
    )
    out = strided_wrapper.run(p["q"], p["kv_cache"])
    assert torch.equal(out, ref)


def test_decode_strided_last_page_len_matches_contiguous():
    p = _make_paged_inputs()
    strided = _strided_copy(p["last_page_len"])
    qd = torch.randn(
        p["batch_size"], NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda:0"
    )

    ref_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", use_tensor_cores=True
    )
    ref_wrapper.plan(
        p["kv_indptr"],
        p["kv_indices"],
        p["last_page_len"],
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    ref = ref_wrapper.run(qd, p["kv_cache"])

    strided_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", use_tensor_cores=True
    )
    strided_wrapper.plan(
        p["kv_indptr"],
        p["kv_indices"],
        strided,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    out = strided_wrapper.run(qd, p["kv_cache"])
    assert torch.equal(out, ref)


def test_ragged_prefill_strided_kv_indptr_matches_contiguous():
    r = _make_ragged_inputs()
    outputs = []
    for kv_indptr in (r["kv_indptr"], _strided_copy(r["kv_indptr"])):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            _workspace_buffer(), "NHD", backend="fa2"
        )
        wrapper.plan(
            r["qo_indptr"],
            kv_indptr,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            causal=False,
        )
        outputs.append(wrapper.run(r["q"], r["k"], r["v"]))
    assert torch.equal(outputs[0], outputs[1])


# ---------------------------------------------------------------------------
# fast_decode_plan (the copy-free plan used by SGLang) validates the same way.
# ---------------------------------------------------------------------------


def test_fast_decode_plan_rejects_int64_last_page_len():
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", use_tensor_cores=True
    )
    with pytest.raises(ValueError, match="last_page_len must be torch.int32"):
        flashinfer.fast_decode_plan(
            wrapper,
            p["kv_indptr"],
            p["kv_indices"],
            p["last_page_len"].to(torch.int64),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )


def test_fast_decode_plan_strided_last_page_len_matches_contiguous():
    p = _make_paged_inputs()
    qd = torch.randn(
        p["batch_size"], NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda:0"
    )
    outputs = []
    for last_page_len in (p["last_page_len"], _strided_copy(p["last_page_len"])):
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _workspace_buffer(), "NHD", use_tensor_cores=True
        )
        # fast_decode_plan reuses the module that a regular plan() cached.
        wrapper.plan(
            p["kv_indptr"],
            p["kv_indices"],
            p["last_page_len"],
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )
        flashinfer.fast_decode_plan(
            wrapper,
            p["kv_indptr"],
            p["kv_indices"],
            last_page_len,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )
        outputs.append(wrapper.run(qd, p["kv_cache"]))
    assert torch.equal(outputs[0], outputs[1])


# ---------------------------------------------------------------------------
# The C++ run binding checks the index dtypes against the compiled IdType, so a
# buffer swapped behind the wrapper's back is rejected too.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_paged_prefill_binding_rejects_mismatched_buffer_dtype(backend):
    if backend == "fa3" and not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("fa3 backend requires SM90a")
    p = _make_paged_inputs()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace_buffer(), "NHD", backend=backend
    )
    wrapper.plan(
        p["qo_indptr"],
        p["kv_indptr"],
        p["kv_indices"],
        p["last_page_len"],
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
    )
    wrapper._paged_kv_last_page_len_buf = wrapper._paged_kv_last_page_len_buf.to(
        torch.int64
    )
    with pytest.raises(RuntimeError, match="paged_kv_last_page_len"):
        wrapper.run(p["q"], p["kv_cache"])


# ---------------------------------------------------------------------------
# CUDA-graph buffers: validated once in __init__, not per plan() call.
# ---------------------------------------------------------------------------


def test_prefill_paged_cuda_graph_buffers_reject_int64():
    device = "cuda:0"
    batch_size = 3
    with pytest.raises(ValueError, match="paged_kv_indptr_buf must be torch.int32"):
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _workspace_buffer(),
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=torch.zeros(batch_size + 1, dtype=torch.int32, device=device),
            paged_kv_indptr_buf=torch.zeros(
                batch_size + 1, dtype=torch.int64, device=device
            ),
            paged_kv_indices_buf=torch.zeros(16, dtype=torch.int32, device=device),
            paged_kv_last_page_len_buf=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
        )


def test_prefill_paged_cuda_graph_buffers_reject_noncontiguous():
    device = "cuda:0"
    batch_size = 3
    strided_indptr = _strided_copy(
        torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    )
    with pytest.raises(ValueError, match="paged_kv_indptr_buf must be contiguous"):
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _workspace_buffer(),
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=torch.zeros(batch_size + 1, dtype=torch.int32, device=device),
            paged_kv_indptr_buf=strided_indptr,
            paged_kv_indices_buf=torch.zeros(16, dtype=torch.int32, device=device),
            paged_kv_last_page_len_buf=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
        )


def test_ragged_prefill_cuda_graph_buffers_reject_int64():
    device = "cuda:0"
    batch_size = 3
    with pytest.raises(ValueError, match="kv_indptr_buf must be torch.int32"):
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            _workspace_buffer(),
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=torch.zeros(batch_size + 1, dtype=torch.int32, device=device),
            kv_indptr_buf=torch.zeros(batch_size + 1, dtype=torch.int64, device=device),
        )


def test_decode_cuda_graph_buffers_reject_int64():
    device = "cuda:0"
    batch_size = 3
    with pytest.raises(ValueError, match="paged_kv_indptr_buffer must be torch.int32"):
        flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _workspace_buffer(),
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=torch.zeros(
                batch_size + 1, dtype=torch.int64, device=device
            ),
            paged_kv_indices_buffer=torch.zeros(16, dtype=torch.int32, device=device),
            paged_kv_last_page_len_buffer=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
        )


def test_decode_cuda_graph_buffers_reject_noncontiguous():
    device = "cuda:0"
    batch_size = 3
    strided_last_page_len = _strided_copy(
        torch.zeros(batch_size, dtype=torch.int32, device=device)
    )
    with pytest.raises(
        ValueError, match="paged_kv_last_page_len_buffer must be contiguous"
    ):
        flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _workspace_buffer(),
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            ),
            paged_kv_indices_buffer=torch.zeros(16, dtype=torch.int32, device=device),
            paged_kv_last_page_len_buffer=strided_last_page_len,
        )
