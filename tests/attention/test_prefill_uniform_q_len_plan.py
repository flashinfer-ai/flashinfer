"""
Copyright (c) 2026 by FlashInfer team.

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

import inspect

import pytest
import torch

import flashinfer
import flashinfer.prefill as prefill_mod
from flashinfer.prefill import _resolve_uniform_q_len

# --------------------------------------------------------------------------
# Host-side guard (no GPU needed): the value forwarded to the FA2 scheduler
# must be 0 whenever the C++ per-request check would reject the hint.
# --------------------------------------------------------------------------

CALLER = "BatchPrefillWithPagedKVCacheWrapper.plan"


def _indptr(lens):
    return torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32
    )


@pytest.fixture
def warnings(monkeypatch):
    """Capture the (once-per-message) planner warnings."""
    seen = []
    monkeypatch.setattr(
        prefill_mod.logger,
        "warning_once",
        lambda msg, *args: seen.append(msg % args if args else msg),
    )
    return seen


def test_none_or_non_positive_means_no_hint(warnings):
    indptr = _indptr([4, 4, 4])
    assert _resolve_uniform_q_len(None, indptr, True, "fa2", CALLER) == 0
    assert _resolve_uniform_q_len(0, indptr, True, "fa2", CALLER) == 0
    assert _resolve_uniform_q_len(-3, indptr, True, "fa2", CALLER) == 0
    assert warnings == []


def test_uniform_cuda_graph_batch_is_forwarded(warnings):
    indptr = _indptr([4] * 8)
    assert _resolve_uniform_q_len(4, indptr, True, "fa2", CALLER) == 4
    # accepts tensor / numpy-style ints too
    assert _resolve_uniform_q_len(torch.tensor(4), indptr, True, "fa2", CALLER) == 4
    assert _resolve_uniform_q_len(4, _indptr([4]), True, "fa2", CALLER) == 4
    assert warnings == []


@pytest.mark.parametrize(
    "lens",
    [
        [4, 4, 3, 4],  # one short request
        [4, 4, 5],  # one long request
        [1, 1, 1],  # constant stride, but not the promised one
        [2, 6, 4, 4],  # averages to 4: the scheduler checks every request
        [],  # empty batch
    ],
)
def test_non_uniform_batch_falls_back_with_a_warning(lens, warnings):
    indptr = _indptr(lens)
    assert _resolve_uniform_q_len(4, indptr, True, "fa2", CALLER) == 0
    assert len(warnings) == 1
    assert "does not describe this batch" in warnings[0]
    assert CALLER in warnings[0]


def test_eager_plans_and_other_backends_do_not_forward(warnings):
    indptr = _indptr([4] * 4)
    assert _resolve_uniform_q_len(4, indptr, False, "fa2", CALLER) == 0
    assert _resolve_uniform_q_len(4, indptr, True, "fa3", CALLER) == 0
    assert _resolve_uniform_q_len(4, indptr, True, "cudnn", CALLER) == 0
    assert _resolve_uniform_q_len(4, indptr, True, "trtllm-gen", CALLER) == 0
    assert warnings == []


def test_plan_signatures_expose_the_keyword():
    for cls in (
        flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper,
        flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper,
    ):
        params = inspect.signature(cls.plan).parameters
        assert "uniform_q_len" in params
        assert params["uniform_q_len"].default is None
        assert params["uniform_q_len"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert (
        "uniform_q_len"
        in inspect.signature(
            flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.workspace_size
        ).parameters
    )


# --------------------------------------------------------------------------
# On-device: a CUDA-graph fa2 plan with the hint must produce the same output
# as the stock plan for a uniform spec-decode-shaped batch (GQA 4:1, 4 query
# rows per request), with and without a custom mask.
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("kv_len", [1023, 4096])
@pytest.mark.parametrize("use_custom_mask", [False, True])
def test_uniform_q_len_plan_matches_stock_plan_on_device(
    batch_size, kv_len, use_custom_mask, monkeypatch
):
    torch.manual_seed(0)
    qo_len, num_qo_heads, num_kv_heads, head_dim, page_size = 4, 4, 1, 128, 1
    device = torch.device("cuda:0")
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=device, dtype=torch.float16
    )
    total_pages = batch_size * kv_len
    kv_data = torch.randn(
        total_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * kv_len
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32)
    kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32)
    custom_mask = None
    if use_custom_mask:
        # verify-style mask: row r of a request sees the prefix plus the first
        # r+1 new tokens
        blocks = []
        for _ in range(batch_size):
            m = torch.ones(qo_len, kv_len, dtype=torch.bool)
            m[:, kv_len - qo_len :] = torch.tril(
                torch.ones(qo_len, qo_len, dtype=torch.bool)
            )
            blocks.append(m.flatten())
        custom_mask = torch.cat(blocks).to(device)

    forwarded = []
    real_resolve = prefill_mod._resolve_uniform_q_len

    def spy(*args, **kwargs):
        value = real_resolve(*args, **kwargs)
        forwarded.append(value)
        return value

    monkeypatch.setattr(prefill_mod, "_resolve_uniform_q_len", spy)

    outputs = []
    for hint in (None, qo_len):
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
            paged_kv_indptr_buf=torch.empty(
                batch_size + 1, dtype=torch.int32, device=device
            ),
            paged_kv_indices_buf=torch.empty(
                total_pages, dtype=torch.int32, device=device
            ),
            paged_kv_last_page_len_buf=torch.empty(
                batch_size, dtype=torch.int32, device=device
            ),
            custom_mask_buf=torch.empty(
                (batch_size * qo_len * kv_len + 7) // 8,
                dtype=torch.uint8,
                device=device,
            )
            if use_custom_mask
            else None,
            mask_indptr_buf=torch.empty(
                batch_size + 1, dtype=torch.int32, device=device
            )
            if use_custom_mask
            else None,
            backend="fa2",
        )
        forwarded.clear()
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            custom_mask=custom_mask,
            causal=not use_custom_mask,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            uniform_q_len=hint,
        )
        assert forwarded == [qo_len if hint else 0]
        o = wrapper.run(q, kv_data)
        # graph-capture the run and replay once: the hint's whole point is the
        # captured plan
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            wrapper.run(q, kv_data)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o_graph = wrapper.run(q, kv_data)
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(o, o_graph, rtol=1e-3, atol=1e-3)
        outputs.append(o.float())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-2, atol=1e-2)
