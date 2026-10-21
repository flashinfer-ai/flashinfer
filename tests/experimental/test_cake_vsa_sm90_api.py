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

"""Public surface of the Cake SM90 VSA backend: the ``flashinfer.cake_vsa_sm90``
facade and ``backend="cake"`` (alias ``"cake_cute"``) of
:class:`flashinfer.sparse.VariableBlockSparseAttentionWrapper`.

The facade is a re-export of ``flashinfer.cake_vsa_sm90_backend``; this module
checks that identity and the wrapper route end to end (numerics against FA3
and an FP32 reference, DPS output ABI, graph capture, re-planning, rejected
options) and leaves the planner, the plan object and the package contract to
``test_cake_vsa_sm90.py``, which imports only the backend package.
"""


import warnings

import pytest
import torch

import flashinfer
import flashinfer.cake_vsa_sm90 as facade
import flashinfer.cake_vsa_sm90_backend as backend
from flashinfer.sparse import VariableBlockSparseAttentionWrapper


def _random_mask(h, mb, nb, capacity, seed=0, ragged=True, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros((h, mb, nb), dtype=torch.bool)
    for head in range(h):
        for row in range(mb):
            count = (
                capacity
                if (row == 0 or not ragged)
                else 1 + (row * 7 + head) % capacity
            )
            mask[head, row, torch.randperm(nb, generator=g)[:count]] = True
    return mask.to(device)


def _descriptors(h=2, mb=3, nb=5, device="cpu"):
    mask = torch.ones((h, mb, nb), dtype=torch.bool, device=device)
    rows = torch.full((h, mb), 64, dtype=torch.int32, device=device)
    cols = torch.full((h, nb), 64, dtype=torch.int32, device=device)
    return mask, rows, cols


requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Requires an SM90 GPU",
)


def _wrapper(mask, backend="cake", scale=None):
    from flashinfer.sparse import VariableBlockSparseAttentionWrapper

    h, mb, nb = mask.shape
    workspace_size = 0 if backend == "cake" else 128 * 1024 * 1024
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


def _inputs(h, mb, nb, seed=7):
    """Deterministic BF16 Q/K/V (HND) for the kernel-vs-reference checks."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(
            (h, length * 64, 128), device="cuda", dtype=torch.bfloat16, generator=g
        )
        for length in (mb, nb, nb)
    )


def _reference(q, k, v, mask, scale):
    """Independent FP32 reference: masked dense softmax attention per head.

    Returned unrounded (FP32): the BF16 kernel output is compared against the
    exact value, so the check bounds the kernel's own rounding (at most half a
    BF16 ulp plus the FP32 accumulation difference) instead of asking two
    independently rounded BF16 results to agree, which any two implementations
    fail by one ulp (0.03125 for |o| in [4, 8)) on a fraction of the elements.
    """
    scores = torch.einsum("hmd,hnd->hmn", q.float(), k.float()) * float(scale)
    dense = mask.to(q.device).repeat_interleave(64, dim=1).repeat_interleave(64, dim=2)
    scores.masked_fill_(~dense, float("-inf"))
    return torch.einsum("hmn,hnd->hmd", torch.softmax(scores, dim=-1), v.float())


def _topk_mask(h, mb, nb, k, seed=42):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.rand((h, mb, nb), generator=g).topk(k, dim=-1).indices
    mask = torch.zeros((h, mb, nb), dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def _odd_single_block_mask():
    """Pair plan with one odd block per head that has a single KV block.

    129 query blocks per head at 64 KV blocks each pair up to 64 pair tiles and
    leave query block 128 (one KV block) alone on warpgroup 0 (tile mode 2).
    """
    g = torch.Generator().manual_seed(3)
    mask = torch.zeros((2, 129, 1024), dtype=torch.bool)
    for head in range(2):
        for row in range(129):
            count = 1 if row == 128 else 64
            mask[head, row, torch.randperm(1024, generator=g)[:count]] = True
    return mask


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
@pytest.mark.parametrize(
    "h,mb,nb,capacity,ragged,scale",
    [
        # boundary / tail: one block, odd block counts, full 64-block capacity,
        # unequal Q/KV lengths, lone tile of a pair plan, custom scale
        (1, 1, 1, 1, False, None),
        (2, 3, 5, 2, True, None),
        (2, 3, 3, 3, False, 0.5),
        (1, 1, 64, 64, False, None),
        (2, 33, 64, 64, True, None),
        (3, 7, 64, 64, False, None),
        (4, 16, 8, 4, True, None),
        (5, 17, 9, 5, True, None),
        (8, 64, 64, 32, False, None),
        (7, 64, 64, 16, True, 0.125),
        (2, 64, 64, 3, True, None),
    ],
)
def test_held_out_shapes_against_fp32_reference(h, mb, nb, capacity, ragged, scale):
    mask = _random_mask(
        h, mb, nb, capacity, seed=h * 1000 + mb, ragged=ragged, device="cuda"
    )
    torch.manual_seed(h * 31 + mb)
    q, k, v = _inputs(h, mb, nb)
    wrapper = _wrapper(mask, scale=scale)
    actual = wrapper.run(q, k, v)
    expected = _reference(q, k, v, mask, 128**-0.5 if scale is None else scale)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.01)
    assert float((actual.float() - expected.float()).abs().max()) <= 0.03


@requires_hopper
def test_replan_and_independent_wrappers():
    mask, rows, cols = _descriptors(2, 2, 3, device="cuda")
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
    mask, _, _ = _descriptors(1, 1, 2, device="cuda")
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
    mask, _, _ = _descriptors(2, 2, 3, device="cuda")
    mask[0, 0, 1] = False
    mask[1, 1, 2] = False
    q, k, v = _inputs(2, 2, 3)
    actual = _wrapper(mask, scale=scale).run(q, k, v)
    expected = _reference(q, k, v, mask, scale)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@requires_hopper
def test_offset_views_and_rejected_output_alias():
    mask, _, _ = _descriptors(1, 2, 3, device="cuda")
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
    mask, rows, cols = _descriptors(device="cuda")
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(0, device="cuda", dtype=torch.uint8), backend="cake"
    )
    plan_kwargs = {"q_data_type": torch.bfloat16, **kwargs}
    with pytest.raises(ValueError, match=match):
        wrapper.plan(mask, rows, cols, 2, 2, 128, **plan_kwargs)


@requires_hopper
@pytest.mark.parametrize(
    "mutation", ["gqa", "head64", "row32", "col63", "dtype", "shape", "empty"]
)
def test_invalid_sparse_metadata(mutation):
    mask, rows, cols = _descriptors(device="cuda")
    h, kv_h, d = 2, 2, 128
    if mutation == "gqa":
        kv_h = 1
    elif mutation == "head64":
        d = 64
    elif mutation == "row32":
        rows[0, 0] = 32
    elif mutation == "col63":
        cols[1, 0] = 63
    elif mutation == "dtype":
        mask = mask.to(torch.int32)
    elif mutation == "shape":
        rows = rows[:, :1]
    else:
        mask[0, 1] = False
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(0, device="cuda", dtype=torch.uint8), backend="cake"
    )
    with pytest.raises(ValueError):
        wrapper.plan(mask, rows, cols, h, kv_h, d, q_data_type=torch.bfloat16)


@requires_hopper
def test_plan_stream_and_graph_lifetime():
    mask, rows, cols = _descriptors(1, 2, 3, device="cuda")
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
    # Other wrappers must not disturb this graph's plan.
    other = _wrapper(mask)
    other.run(q, k, v)
    q.normal_()
    graph.replay()
    torch.testing.assert_close(
        out.view_as(q).float(),
        _reference(q, k, v, mask, 128**-0.5),
        atol=0.01,
        rtol=0.01,
    )
    with warnings.catch_warnings():
        # The capture is abandoned before any launch; torch warns about the empty graph.
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="preallocated"):
            graph2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph2):
                wrapper.run(q, k, v)


@requires_hopper
def test_odd_single_block_tile_never_stores_from_the_idle_warpgroup():
    # On a mode-2 tile both warpgroups address the same 64 output rows; the
    # idle warpgroup's accumulator is 0 x rcp(0) = NaN and its store must not
    # happen (it raced the owner's store).  NaN-filled output, many runs.
    mask = _odd_single_block_mask().cuda()
    torch.manual_seed(11)
    q, k, v = _inputs(2, 129, 1024)
    wrapper = _wrapper(mask)
    expected = _reference(q, k, v, mask, 128**-0.5)
    for _ in range(20):
        out = torch.full(
            (2 * 129 * 64, 1, 128), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        actual = wrapper.run(q, k, v, out=out)
        assert bool(torch.isfinite(actual).all()), (
            "an output row was never written or holds NaN"
        )
        torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.01)


@requires_hopper
def test_backend_cake_cute_is_an_alias_of_cake():
    """``backend="cake_cute"`` (the name of the removed CuTe DSL build) plans and
    runs the ``cake`` kernels: same stage, bit-identical output, same DPS ABI."""
    h, mb, nb = 2, 16, 16
    mask = _topk_mask(h, mb, nb, 12).cuda()
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    wrappers = {}
    for name in ("cake", "cake_cute"):
        wrapper = VariableBlockSparseAttentionWrapper(
            torch.empty(0, device="cuda", dtype=torch.uint8), backend=name
        )
        wrapper.plan(
            mask, rows, cols, h, h, 128, q_data_type=torch.bfloat16, non_blocking=False
        )
        wrappers[name] = wrapper
    plan_a = wrappers["cake"]._cake_vsa_sm90_plan
    plan_b = wrappers["cake_cute"]._cake_vsa_sm90_plan
    for name in ("small_kmax", "small_split", "small_cluster", "num_ctas"):
        assert getattr(plan_a, name) == getattr(plan_b, name), name
    q, k, v = _inputs(h, mb, nb)
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    expected = wrappers["cake"].run(q, k, v, enable_pdl=False)
    actual = wrappers["cake_cute"].run(q, k, v, out=out, enable_pdl=False)
    assert actual.data_ptr() == out.data_ptr() and actual.shape == q.shape
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_facade_re_exports_the_backend():
    """``flashinfer.cake_vsa_sm90`` is the stable import path of the backend package: the plan object,
    ``create_plan`` and the planner entry points are the backend's own objects."""
    for name in (
        "CakeVsaSm90Plan",
        "create_plan",
        "cluster_capacity",
        "cluster_variant_for",
        "plan_small",
        "plan_vsa_sm90",
        "small_route",
        "split_cluster_variant_for",
        "split_kmax",
        "is_cake_vsa_sm90_supported",
    ):
        assert getattr(facade, name) is getattr(backend, name), name
    assert facade.ARCHES == backend.ARCHES == ("sm_90a",)
    assert facade.SUPPORTED_COMPUTE_CAPABILITIES == ((9, 0),)
    assert set(facade.__all__) <= set(dir(backend))
    assert len(facade.STAGES) == 18
    assert backend.__name__ == "flashinfer.cake_vsa_sm90_backend"
    assert (
        flashinfer.sparse.VariableBlockSparseAttentionWrapper
        is VariableBlockSparseAttentionWrapper
    )


def test_facade_support_predicate_follows_the_device():
    assert facade.is_cake_vsa_sm90_supported(torch.device("cpu")) is False
    if torch.cuda.is_available():
        expected = torch.cuda.get_device_capability() == (9, 0)
        assert facade.is_cake_vsa_sm90_supported(torch.device("cuda")) is expected
