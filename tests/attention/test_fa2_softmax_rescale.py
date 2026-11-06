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

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] < 8,
    reason="This BF16 and large-head FA2 regression requires SM80 or newer",
)


def run_prefill(q, k, v, *, layout, mask=None, causal=False, split=False, graph=False):
    """One-request public API call, with an independent graph-enabled plan."""
    device = q.device
    qi = torch.tensor([0, len(q)], dtype=torch.int32, device=device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    options = dict(use_cuda_graph=graph, backend="fa2")
    if graph and mask is not None:
        options.update(
            custom_mask_buf=torch.empty(
                (mask.numel() + 7) // 8, dtype=torch.uint8, device=device
            ),
            mask_indptr_buf=torch.empty(2, dtype=torch.int32, device=device),
        )
    plan_options = dict(
        causal=causal,
        custom_mask=mask,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        disable_split_kv=not split,
        sm_scale=q.shape[-1] ** -0.5,
    )
    if layout == "ragged":
        ki = torch.tensor([0, len(k)], dtype=torch.int32, device=device)
        if graph:
            options.update(qo_indptr_buf=qi, kv_indptr_buf=ki)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, **options)
        wrapper.plan(qi, ki, q.shape[1], k.shape[1], q.shape[-1], **plan_options)
        inputs = (q, k, v)
    else:
        page_size = 16
        pages = (len(k) + page_size - 1) // page_size
        kp = torch.zeros(pages * page_size, *k.shape[1:], dtype=k.dtype, device=device)
        vp = torch.zeros_like(kp)
        kp[: len(k)], vp[: len(v)] = k, v
        ki = torch.tensor([0, pages], dtype=torch.int32, device=device)
        indices = torch.arange(pages, dtype=torch.int32, device=device)
        last = torch.tensor(
            [(len(k) - 1) % page_size + 1], dtype=torch.int32, device=device
        )
        if graph:
            options.update(
                qo_indptr_buf=qi,
                paged_kv_indptr_buf=ki,
                paged_kv_indices_buf=indices,
                paged_kv_last_page_len_buf=last,
            )
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, **options)
        wrapper.plan(
            qi,
            ki,
            indices,
            last,
            q.shape[1],
            k.shape[1],
            q.shape[-1],
            page_size,
            **plan_options,
        )
        inputs = (
            q,
            (
                kp.view(pages, page_size, *k.shape[1:]),
                vp.view(pages, page_size, *v.shape[1:]),
            ),
        )
    out, lse = wrapper.run(*inputs, return_lse=True)
    if graph:
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            out, lse = wrapper.run(*inputs, return_lse=True)
        for _ in range(5):
            captured.replay()
    torch.cuda.synchronize()
    return out, lse


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("layout", ["ragged", "paged"])
def test_equal_maximum_preserves_history(dtype, head_dim, layout):
    """Preserve an unchanged dominant maximum across KV tiles in Eager and Graph runs."""
    # An early dominant key followed by zero scores keeps the maximum unchanged.
    # These are independently constructed, exactly representable finite inputs.
    q = torch.full((64, 1, head_dim), 8192, device="cuda", dtype=dtype)
    k = torch.zeros((32768, 1, head_dim), device="cuda", dtype=dtype)
    k[0] = 160
    v = torch.zeros_like(k)
    v[0] = 1

    # Full-KV FP64 reference for one of the identical query rows.
    scores = q[0, 0].double() @ k[:, 0].double().T * head_dim**-0.5
    reference = torch.softmax(scores, dim=-1) @ v[:, 0].double()
    reference_lse = torch.logsumexp(scores, dim=-1) / math.log(2)
    torch.testing.assert_close(reference, torch.ones_like(reference), atol=0, rtol=0)
    for graph in [False, True]:
        out, lse = run_prefill(q, k, v, layout=layout, graph=graph)
        assert torch.isfinite(out).all()
        assert torch.isfinite(lse).all()
        torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=2e-3)
        # The returned FP32 log2 LSE has a few-ULP rounding allowance at large scores.
        torch.testing.assert_close(
            lse.double(), reference_lse.expand_as(lse), rtol=2e-7, atol=1e-2
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("layout", ["ragged", "paged"])
@pytest.mark.parametrize("valid_key", [None, 0, 256])
def test_masked_history(dtype, head_dim, layout, valid_key):
    """Keep fully masked rows empty and retain mass before or after masked KV tiles."""
    q = torch.ones(4, 1, head_dim, dtype=dtype, device="cuda")
    k = torch.ones(257, 1, head_dim, dtype=dtype, device="cuda")
    v = torch.ones_like(k)
    mask = torch.zeros(4, 257, dtype=torch.bool, device="cuda")
    if valid_key is not None:
        mask[:, valid_key] = True
    for graph in [False, True]:
        out, lse = run_prefill(q, k, v, layout=layout, mask=mask, graph=graph)
        assert torch.isfinite(out).all()
        expected = torch.zeros_like(out) if valid_key is None else torch.ones_like(out)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        if valid_key is None:
            assert torch.isneginf(lse).all()
        else:
            expected_lse = math.sqrt(head_dim) / math.log(2)
            torch.testing.assert_close(
                lse, torch.full_like(lse, expected_lse), rtol=1e-6, atol=1e-5
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("layout", ["ragged", "paged"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_random_fp64_reference(dtype, head_dim, layout, causal, split):
    """Compare output and LSE with full-KV FP64 for causal and split-planner options."""
    torch.manual_seed(20261020)
    q = torch.randn(17, 4, head_dim, dtype=dtype, device="cuda") / 3
    k = torch.randn(513, 2, head_dim, dtype=dtype, device="cuda") / 3
    v = torch.randn_like(k)
    k64 = k.double().repeat_interleave(2, dim=1).transpose(0, 1)
    v64 = v.double().repeat_interleave(2, dim=1).transpose(0, 1)
    scores = q.double().transpose(0, 1) @ k64.transpose(-1, -2) * head_dim**-0.5
    if causal:
        allowed = torch.arange(len(k), device="cuda")[None, :] <= (
            len(k) - len(q) + torch.arange(len(q), device="cuda")[:, None]
        )
        scores.masked_fill_(~allowed, -torch.inf)
    reference = (torch.softmax(scores, dim=-1) @ v64).transpose(0, 1)
    reference_lse = (torch.logsumexp(scores, dim=-1) / math.log(2)).T
    for graph in [False, True]:
        out, lse = run_prefill(
            q, k, v, layout=layout, causal=causal, split=split, graph=graph
        )
        assert torch.isfinite(out).all() and torch.isfinite(lse).all()
        torch.testing.assert_close(out.double(), reference, rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(lse.double(), reference_lse, rtol=2e-4, atol=2e-4)
