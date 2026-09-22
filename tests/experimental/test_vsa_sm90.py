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

"""Hopper VSA planning, numerical, lifetime, and API regression coverage."""

import pytest
import torch

from flashinfer.experimental.vsa_sm90.metadata import prepare_metadata


def _descriptors(h=2, mb=3, nb=5):
    mask = torch.ones((h, mb, nb), dtype=torch.bool)
    rows = torch.full((h, mb), 64, dtype=torch.int32)
    cols = torch.full((h, nb), 64, dtype=torch.int32)
    return mask, rows, cols


def test_metadata_snapshot_sorted_prefixes():
    mask, rows, cols = _descriptors()
    mask[0, 0] = torch.tensor([False, True, False, True, False])
    original = mask.clone()
    plan = prepare_metadata(mask, rows, cols, 2, 2, 128)
    mask.fill_(False)
    rows.fill_(1)
    cols.fill_(1)
    for h in range(2):
        for row in range(3):
            expected = original[h, row].nonzero().flatten().to(torch.int32)
            count = int(plan.counts[h, row])
            torch.testing.assert_close(plan.indices[h, row, :count], expected)
            assert bool((plan.indices[h, row, count:] == -1).all())
    assert (plan.qo_len, plan.kv_len) == (192, 320)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"causal": True}, "noncausal"),
        ({"pos_encoding_mode": "ALIBI"}, "positional"),
        ({"use_fp16_qk_reduction": True}, "FP32"),
        ({"logits_soft_cap": 2.0}, "logits_soft_cap"),
        ({"q_data_type": torch.float16}, "BF16"),
        ({"kv_data_type": torch.float16}, "BF16"),
        ({"sm_scale": float("nan")}, "finite"),
        ({"sm_scale": float("inf")}, "finite"),
        ({"sm_scale": 1e39}, "finite"),
    ],
)
def test_unsupported_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        prepare_metadata(*_descriptors(), 2, 2, 128, **kwargs)


@pytest.mark.parametrize("mutation", ["empty", "row64", "col64", "dtype", "shape"])
def test_invalid_sparse_metadata(mutation):
    mask, rows, cols = _descriptors()
    if mutation == "empty":
        mask[0, 1] = False
    elif mutation == "row64":
        rows[0, 0] = 32
    elif mutation == "col64":
        cols[1, 0] = 63
    elif mutation == "dtype":
        mask = mask.to(torch.int32)
    else:
        rows = rows[:, :1]
    with pytest.raises(ValueError):
        prepare_metadata(mask, rows, cols, 2, 2, 128)


@pytest.mark.parametrize("h,kv_h,d", [(2, 1, 128), (2, 2, 64), (0, 0, 128)])
def test_unsupported_dimensions(h, kv_h, d):
    with pytest.raises(ValueError):
        prepare_metadata(*_descriptors(), h, kv_h, d)


def test_sparse_capacity_limit():
    with pytest.raises(ValueError, match="at most 64"):
        prepare_metadata(*_descriptors(nb=65), 2, 2, 128)


@pytest.mark.parametrize("h,mb", [(2, 160), (2, 600)])
def test_load_balancing_visits_each_query_block_once(h, mb):
    mask, rows, cols = _descriptors(h, mb, 16)
    mask[:, ::2, 1:] = False
    plan = prepare_metadata(mask, rows, cols, h, h, 128)
    torch.testing.assert_close(plan.order.sort().values, torch.arange(h * mb))
    counts = plan.counts.flatten()[plan.order]
    if h * mb <= 1056:
        assert bool((counts[:-1] >= counts[1:]).all())
    else:
        counts = counts.reshape(h, mb)
        assert bool((counts[:, :-1] >= counts[:, 1:]).all())


@pytest.mark.parametrize(
    "mb,nb,scale,expected",
    [
        (1, 1, 0.125, "single"),
        (32, 3, 0.125, "dsplit"),
        (128, 3, 0.125, "general"),
        (128, 12, 0.125, "pipelined"),
        (1, 1, 0.0, "general"),
        (1, 1, -0.125, "general"),
    ],
)
def test_dispatch_uses_shape_and_scale(mb, nb, scale, expected):
    plan = prepare_metadata(*_descriptors(1, mb, nb), 1, 1, 128, sm_scale=scale)
    assert plan.schedule == expected


requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Requires an SM90 GPU and CuTe DSL",
)


def _wrapper(mask, backend="vsa_sm90_blk64", scale=None):
    from flashinfer.sparse import VariableBlockSparseAttentionWrapper

    h, mb, nb = mask.shape
    workspace_size = 0 if backend == "vsa_sm90_blk64" else 128 * 1024 * 1024
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(workspace_size, device="cuda", dtype=torch.uint8), backend=backend
    )
    wrapper.plan(
        mask,
        torch.full((h, mb), 64, dtype=torch.int32, device=mask.device),
        torch.full((h, nb), 64, dtype=torch.int32, device=mask.device),
        h,
        h,
        128,
        q_data_type=torch.bfloat16,
        sm_scale=scale,
        non_blocking=False,
    )
    return wrapper


def _inputs(h, mb, nb):
    return tuple(
        torch.randn((h, length * 64, 128), device="cuda", dtype=torch.bfloat16)
        for length in (mb, nb, nb)
    )


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity",
    [(2, 1, 2, 1), (1, 16, 6, 3), (1, 128, 8, 4), (1, 128, 16, 12), (2, 160, 16, 12)],
)
def test_all_schedules_against_fa3(h, mb, nb, capacity):
    torch.manual_seed(42)
    mask = torch.zeros((h, mb, nb), dtype=torch.bool, device="cuda")
    for head in range(h):
        for row in range(mb):
            count = capacity if row == 0 else 1 + row % capacity
            mask[head, row, torch.randperm(nb, device="cuda")[:count]] = True
    candidate = _wrapper(mask)
    reference = _wrapper(mask, "fa3")
    q, k, v = _inputs(h, mb, nb)
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        expected = reference.run(q, k, v, enable_pdl=False)
        actual = candidate.run(q, k, v, out=out, enable_pdl=False)
        assert actual.data_ptr() == out.data_ptr()
        assert actual.shape == q.shape
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
        assert float((actual.float() - expected.float()).abs().max()) <= 0.03
        q.normal_()
        v.normal_()


@requires_hopper
def test_replan_and_independent_wrappers():
    mask, rows, cols = _descriptors(2, 2, 3)
    mask[:, :, 1:] = False
    first = _wrapper(mask)
    q, k, v = _inputs(2, 2, 3)
    expected_first = first.run(q, k, v).clone()
    mask[:, :, 0] = False
    mask[:, :, 2] = True
    second = _wrapper(mask)
    expected_second = second.run(q, k, v).clone()
    torch.testing.assert_close(first.run(q, k, v), expected_first, atol=0, rtol=0)
    first.plan(mask, rows, cols, 2, 2, 128, q_data_type=torch.bfloat16)
    torch.testing.assert_close(first.run(q, k, v), expected_second, atol=0, rtol=0)
    torch.testing.assert_close(second.run(q, k, v), expected_second, atol=0, rtol=0)


@requires_hopper
@pytest.mark.parametrize("scale", [0.0, -0.125, 1e-7, 128**-0.5])
def test_padding_uses_attention_math(scale):
    # Separate mathematical control for the known FA3 finite-padding defect.
    mask, _, _ = _descriptors(1, 1, 2)
    mask[:, :, 1] = False
    q, k, v = _inputs(1, 1, 2)
    q.fill_(-32)
    k.fill_(32)
    actual = _wrapper(mask, scale=scale).run(q, k, v)
    expected = v[:, :64].float().mean(dim=1, keepdim=True).expand_as(q).to(q.dtype)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@requires_hopper
@pytest.mark.parametrize("scale", [0.0, -0.125, 1e-7, 1e-4])
def test_signed_and_tiny_scales_against_math(scale):
    mask, _, _ = _descriptors(2, 2, 3)
    mask[0, 0, 1] = False
    mask[1, 1, 2] = False
    q, k, v = _inputs(2, 2, 3)
    actual = _wrapper(mask, scale=scale).run(q, k, v)
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    dense_mask = mask.repeat_interleave(64, dim=1).repeat_interleave(64, dim=2)
    scores.masked_fill_(~dense_mask.to("cuda"), float("-inf"))
    expected = (scores.softmax(dim=-1) @ v.float()).to(q.dtype)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@requires_hopper
def test_offset_views_and_rejected_output_alias():
    mask, _, _ = _descriptors(1, 2, 3)
    wrapper = _wrapper(mask)
    q, k, v = _inputs(1, 2, 3)
    expected = wrapper.run(q, k, v)
    views = []
    for tensor in (q, k, v):
        storage = torch.empty(tensor.numel() + 1, device="cuda", dtype=tensor.dtype)
        view = storage[1:].view_as(tensor)
        view.copy_(tensor)
        views.append(view)
    storage = torch.full((q.numel() + 2,), 7, device="cuda", dtype=q.dtype)
    out = storage[1:-1].view(-1, 1, 128)
    actual = wrapper.run(*views, out=out)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert storage[0].item() == 7 and storage[-1].item() == 7
    with pytest.raises(ValueError, match="overlap"):
        wrapper.run(q, k, v, out=q.view(-1, 1, 128))
    with pytest.raises(ValueError, match="log-sum-exp"):
        wrapper.run(q, k, v, return_lse=True)
    with pytest.raises(ValueError, match="PDL"):
        wrapper.run(q, k, v, enable_pdl=True)


@requires_hopper
def test_plan_stream_and_graph_lifetime():
    mask, rows, cols = _descriptors(1, 2, 3)
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        wrapper = _wrapper(mask)
    # Deliberately no producer wait: plan's ready event owns that dependency.
    q, k, v = _inputs(1, 2, 3)
    expected = wrapper.run(q, k, v).clone()
    out = torch.empty((128, 1, 128), device="cuda", dtype=q.dtype)
    wrapper.run(q, k, v, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, out=out)
    graph.replay()
    torch.testing.assert_close(out.view_as(q), expected, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="captured"):
        wrapper.plan(mask, rows, cols, 1, 1, 128, q_data_type=torch.bfloat16)
    # Other wrappers must not evict this graph's descriptor storage.
    other = _wrapper(mask)
    other.run(q, k, v)
    graph.replay()
    torch.testing.assert_close(out.view_as(q), expected, atol=0, rtol=0)
