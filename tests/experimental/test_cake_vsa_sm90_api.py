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
options; the Sage FP8 route: ``q_data_type=torch.float8_e4m3fn`` with the three
scales at ``plan()``, its validation messages, BF16 output, graph replay; the
grouped-query route: ``num_qo_heads = g x num_kv_heads`` with ``g`` in {2, 4, 8}
on the kv-head-shaped mask and scales, other ratios rejected) and leaves the
planner, the plan object and the package contract to ``test_cake_vsa_sm90.py``,
which imports only the backend package.
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


def _gqa_inputs(h, h_kv, mb, nb, seed=7):
    """``q`` over the ``h`` query heads, ``k`` / ``v`` over the ``h_kv`` kv heads (``_inputs``' draw at
    ``h_kv`` heads, so the MHA route on ``k.repeat_interleave(g, 0)`` is the replicated-input baseline)."""
    q = _inputs(h, mb, nb, seed)[0]
    _, k, v = _inputs(h_kv, mb, nb, seed)
    return q, k, v


def _gqa_wrapper(mask_kv, num_qo_heads, backend="cake", scale=None, **plan_kwargs):
    """A wrapper planned on the kv-head-shaped mask for ``num_qo_heads`` query heads."""
    h_kv, mb, nb = mask_kv.shape
    workspace_size = 0 if backend == "cake" else 128 * 1024 * 1024
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(workspace_size, device="cuda", dtype=torch.uint8), backend=backend
    )
    wrapper.plan(
        mask_kv,
        torch.full((h_kv, mb), 64, dtype=torch.int32, device=mask_kv.device),
        torch.full((h_kv, nb), 64, dtype=torch.int32, device=mask_kv.device),
        num_qo_heads,
        h_kv,
        128,
        sm_scale=scale,
        non_blocking=False,
        **({"q_data_type": torch.bfloat16} | plan_kwargs),
    )
    return wrapper


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


FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_K_GROUP = 16


def _torch_sage_quantize_hnd(q, k, v):
    """Torch Sage recipe on HND tensors: E4M3 q/k/v and the FP32 scales in the shapes ``plan()`` takes
    (``q_scale (1, H, Sq)`` per token, ``k_scale (1, H, ceil(Sk / 16))`` per 16-token group, ``v_scale (H, 128)``
    per channel); the SM100 recipe with ``batch := H`` and one head.  ``k`` / ``v`` may carry the kv heads of a
    GQA problem (every tensor is quantized per its own head)."""
    _h, sq, dim = q.shape
    h, sk = k.shape[0], k.shape[1]
    q32, k32, v32 = (t.float() for t in (q, k, v))
    q_scale = q32.abs().amax(dim=-1).clamp_min(1e-12) / _FP8_MAX
    q8 = (q32 / q_scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX).to(FP8)
    groups = -(-sk // _K_GROUP)
    k_pad = torch.nn.functional.pad(k32, (0, 0, 0, groups * _K_GROUP - sk))
    k_grouped = k_pad.view(h, groups, _K_GROUP, dim)
    k_scale = k_grouped.abs().amax(dim=(-1, -2)).clamp_min(1e-12) / _FP8_MAX
    k8 = (k_grouped / k_scale[..., None, None]).clamp(-_FP8_MAX, _FP8_MAX)
    k8 = k8.view(h, groups * _K_GROUP, dim)[:, :sk].to(FP8)
    v_scale = v32.abs().amax(dim=-2).clamp_min(1e-12) / _FP8_MAX
    v8 = (v32 / v_scale.unsqueeze(-2)).clamp(-_FP8_MAX, _FP8_MAX).to(FP8)
    return (
        q8.contiguous(),
        k8.contiguous(),
        v8.contiguous(),
        q_scale.unsqueeze(0).contiguous(),
        k_scale.unsqueeze(0).contiguous(),
        v_scale.contiguous(),
    )


def _dequantize(q8, k8, v8, q_scale, k_scale, v_scale):
    """FP32 q/k/v the E4M3 tensors and their scales denote (the inputs of reference (b))."""
    sk = k8.shape[1]
    k_tok = k_scale[0].repeat_interleave(_K_GROUP, dim=-1)[:, :sk]
    return (
        q8.float() * q_scale[0].unsqueeze(-1),
        k8.float() * k_tok.unsqueeze(-1),
        v8.float() * v_scale.unsqueeze(-2),
    )


def _fp8_plan(mask, q, k, v, scale=None, backend="cake", quantize=None):
    """A ``backend="cake"`` wrapper planned on the FP8 route plus the quantized inputs it runs on.  ``quantize``
    is the facade's kernel quantizer by default; pass ``_torch_sage_quantize_hnd`` for the torch recipe."""
    h, mb, nb = mask.shape
    q8, k8, v8, q_scale, k_scale, v_scale = (quantize or facade.sage_fp8_quantize_sm90)(
        q, k, v
    )
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(0, device="cuda", dtype=torch.uint8), backend=backend
    )
    wrapper.plan(
        mask,
        torch.full((h, mb), 64, dtype=torch.int32, device=mask.device),
        torch.full((h, nb), 64, dtype=torch.int32, device=mask.device),
        h,
        h,
        128,
        sm_scale=scale,
        q_data_type=FP8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        non_blocking=False,
    )
    return wrapper, (q8, k8, v8), (q_scale, k_scale, v_scale)


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
        ({"q_data_type": FP8}, "q_scale, k_scale and v_scale together"),
        ({"q_data_type": FP8, "kv_data_type": torch.bfloat16}, "kv_data_type equal"),
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
    "mutation",
    ["gqa3", "gqa_nonmultiple", "head64", "row32", "col63", "dtype", "shape", "empty"],
)
def test_invalid_sparse_metadata(mutation):
    mask, rows, cols = _descriptors(device="cuda")
    h, kv_h, d = 2, 2, 128
    if mutation == "gqa3":  # three query heads per kv head is not a supported group
        h = 6
    elif mutation == "gqa_nonmultiple":
        h = 3
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
@pytest.mark.parametrize(
    "g,h_kv,mb,nb,capacity,scale",
    [
        (2, 2, 4, 8, 3, None),  # small route
        (4, 1, 16, 8, 4, 0.5),  # MQA
        (8, 1, 4, 4, 1, None),  # MQA, one block per query block
        (4, 2, 160, 16, 12, -0.125),  # persistent pair plan
        (2, 3, 33, 64, 64, None),  # full lists, odd kv-head and query-block counts
    ],
)
def test_gqa_head_ratios_run_the_gqa_kernels(g, h_kv, mb, nb, capacity, scale):
    """``backend="cake"`` accepts ``num_qo_heads = g x num_kv_heads`` for ``g`` in {2, 4, 8} (MQA included)
    with the kv-head-shaped mask and descriptors: the GQA stage of the MHA route runs, the output (DPS
    ``[H*M, 1, 128]``, HND return value) matches the FP32 reference and -- bit for bit on the static programs
    -- the MHA route on replicated inputs; two launches are bit-identical."""
    h = g * h_kv
    mask = _random_mask(
        h_kv, mb, nb, capacity, seed=h_kv * 1000 + mb, ragged=True, device="cuda"
    )
    mask_rep = mask.repeat_interleave(g, dim=0)
    q, k, v = _gqa_inputs(h, h_kv, mb, nb)
    k_rep, v_rep = k.repeat_interleave(g, dim=0), v.repeat_interleave(g, dim=0)
    gqa = _gqa_wrapper(mask, h, scale=scale)
    mha = _wrapper(mask_rep, scale=scale)
    plan, plan_mha = gqa._cake_vsa_sm90_plan, mha._cake_vsa_sm90_plan
    assert (plan.gqa, plan.num_qo_heads, plan.num_kv_heads) == (g, h, h_kv)
    assert backend.stage_gqa(plan.stage) and not backend.stage_gqa(plan_mha.stage)
    assert plan.mode == plan_mha.mode
    out = torch.full(
        (h * mb * 64, 1, 128), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    actual = gqa.run(q, k, v, out=out, enable_pdl=False)
    assert actual.data_ptr() == out.data_ptr() and actual.shape == q.shape
    assert bool(torch.isfinite(actual).all())
    expected = mha.run(q, k_rep, v_rep, enable_pdl=False)
    static = plan.small_kmax is None or not (plan.small_split or plan.small_cluster)
    if static:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    reference = _reference(
        q, k_rep, v_rep, mask_rep, 128**-0.5 if scale is None else scale
    )
    torch.testing.assert_close(actual.float(), reference, atol=0.01, rtol=0.01)
    assert float((actual.float() - reference).abs().max()) <= 0.03
    torch.testing.assert_close(gqa.run(q, k, v), actual, atol=0, rtol=0)
    with pytest.raises(ValueError, match="k must have shape"):
        gqa.run(q, k_rep, v_rep)


@requires_hopper
@pytest.mark.parametrize(
    "h,kv_h,match",
    [
        (6, 2, r"g in \(2, 4, 8\)"),
        (10, 2, r"g in \(2, 4, 8\)"),
        (12, 2, r"g in \(2, 4, 8\)"),
        (32, 2, r"g in \(2, 4, 8\)"),
        (5, 2, r"must be a multiple of num_kv_heads"),
        (3, 2, r"must be a multiple of num_kv_heads"),
        (4, 1, r"must equal num_kv_heads \(1\)"),  # descriptors given per query head
    ],
)
def test_gqa_unsupported_head_ratios_are_rejected(h, kv_h, match):
    """Other multiples (3, 5, 6, 16 query heads per kv head), non-multiples and a mask shaped by the query
    heads are refused by name at ``plan()``."""
    mask, rows, cols = _descriptors(device="cuda")  # two (kv) heads
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(0, device="cuda", dtype=torch.uint8), backend="cake"
    )
    with pytest.raises(ValueError, match=match):
        wrapper.plan(mask, rows, cols, h, kv_h, 128, q_data_type=torch.bfloat16)


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
        "stage_gqa",
        "GQA_PERSISTENT_STAGES",
        "GQA_SMALL_STAGES",
        "FP8_GQA_PERSISTENT_STAGES",
        "FP8_GQA_SMALL_STAGES",
    ):
        assert getattr(facade, name) is getattr(backend, name), name
    assert facade.ARCHES == backend.ARCHES == ("sm_90a",)
    assert facade.SUPPORTED_COMPUTE_CAPABILITIES == ((9, 0),)
    assert facade.GQA_GROUPS == backend.GQA_GROUPS == (1, 2, 4, 8)
    assert set(facade.__all__) <= set(dir(backend))
    # 19 BF16 + 19 FP8 twins + 2 quantizers + 19 + 19 GQA stages + the two round-2 FP8 GQA programs
    # + the round-16 position-selected FP8 GQA static program
    assert len(facade.STAGES) == 81
    assert len(facade.GQA_PERSISTENT_STAGES) + len(facade.GQA_SMALL_STAGES) == 19
    assert (
        len(facade.FP8_GQA_PERSISTENT_STAGES) + len(facade.FP8_GQA_SMALL_STAGES) == 22
    )
    assert facade.sage_fp8_quantize_sm90 is backend.sage_fp8_quantize_sm90
    assert facade.FP8_DTYPE is torch.float8_e4m3fn and facade.K_SCALE_GROUP == 16
    assert "sage_fp8_quantize_sm90" in facade.__all__
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


# --------------------------------------------------------------------------- Sage FP8 (E4M3) route


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity,scale",
    [
        (2, 3, 5, 2, None),  # small kernel
        (1, 16, 16, 16, 0.5),  # cluster
        (2, 8, 64, 64, None),  # split-KV, full 64-block lists
        (2, 160, 16, 12, -0.125),  # persistent pair plan
    ],
)
def test_fp8_route_against_the_dequantized_reference(h, mb, nb, capacity, scale):
    """``q_data_type=torch.float8_e4m3fn`` with the three scales plans the FP8 twin of the route the BF16
    plan takes; the BF16 output matches the exact softmax of the dequantized E4M3 inputs (reference (b):
    atol = rtol = 5e-2, no element outside) and three launches are bit-identical."""
    mask = _random_mask(h, mb, nb, capacity, seed=5, ragged=True, device="cuda")
    q, k, v = _inputs(h, mb, nb)
    wrapper, (q8, k8, v8), scales = _fp8_plan(mask, q, k, v, scale)
    plan = wrapper._cake_vsa_sm90_plan
    bf16 = _wrapper(mask, scale=scale)._cake_vsa_sm90_plan
    assert plan.fp8 and plan.stage in backend.MODULES
    assert plan.stage == (
        bf16.stage.replace("attention", "attention_fp8")
        if bf16.stage.startswith("attention")
        else bf16.stage.replace("small_", "small_fp8_")
    )
    assert plan.mode == bf16.mode
    outs = [wrapper.run(q8, k8, v8).clone() for _ in range(3)]
    assert outs[0].dtype == torch.bfloat16 and outs[0].shape == q.shape
    for later in outs[1:]:
        torch.testing.assert_close(later, outs[0], atol=0, rtol=0)
    exact = _reference(
        *_dequantize(q8, k8, v8, *scales), mask, 128**-0.5 if scale is None else scale
    )
    torch.testing.assert_close(outs[0].float(), exact, atol=5e-2, rtol=5e-2)


@requires_hopper
def test_fp8_output_is_bf16_dps():
    """The DPS ``out`` of an FP8 plan is the BF16 ``[H*M, 1, 128]`` buffer (the return value its HND view);
    an E4M3 or wrongly shaped ``out`` and BF16 ``q/k/v`` on an FP8 plan are refused by name."""
    mask = _random_mask(2, 4, 6, 3, seed=9, device="cuda")
    q, k, v = _inputs(2, 4, 6)
    wrapper, (q8, k8, v8), _ = _fp8_plan(mask, q, k, v)
    out = torch.full(
        (2 * 4 * 64, 1, 128), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    result = wrapper.run(q8, k8, v8, out=out)
    torch.cuda.synchronize()
    assert result.data_ptr() == out.data_ptr() and result.shape == q.shape
    assert not torch.isnan(out).any(), "every output element is written"
    with pytest.raises(ValueError, match="out must have shape .* and dtype bfloat16"):
        wrapper.run(q8, k8, v8, out=out.view(FP8))
    with pytest.raises(
        ValueError, match="q must have shape .* and dtype float8_e4m3fn"
    ):
        wrapper.run(q, k, v)
    with pytest.raises(
        ValueError, match="k must have shape .* and dtype float8_e4m3fn"
    ):
        wrapper.run(q8, k, v8)


@requires_hopper
def test_fp8_scales_are_rejected_for_other_backends():
    mask, rows, cols = _descriptors(device="cuda")
    scale = torch.ones((1, 2, 3 * 64), device="cuda")
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8), backend="fa3"
    )
    with pytest.raises(
        ValueError,
        match="only supported for backend='cake' \\(Sage FP8 on SM90\\), got backend='fa3'",
    ):
        wrapper.plan(
            mask, rows, cols, 2, 2, 128, q_data_type=torch.bfloat16, q_scale=scale
        )


@requires_hopper
@pytest.mark.parametrize(
    "given", [(), ("q_scale",), ("q_scale", "k_scale"), ("k_scale", "v_scale")]
)
def test_fp8_requires_the_three_scales_together(given):
    mask, rows, cols = _descriptors(device="cuda")
    full = {
        "q_scale": torch.ones((1, 2, 3 * 64), device="cuda"),
        "k_scale": torch.ones((1, 2, 5 * 64 // 16), device="cuda"),
        "v_scale": torch.ones((2, 128), device="cuda"),
    }
    kwargs = {name: full[name] for name in given}
    missing = ", ".join(name for name in full if name not in given)
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(0, device="cuda", dtype=torch.uint8), backend="cake"
    )
    with pytest.raises(ValueError, match=f"together \\(missing: {missing}\\)"):
        wrapper.plan(mask, rows, cols, 2, 2, 128, q_data_type=FP8, **kwargs)
    # the same scales on a BF16 plan are refused the other way round
    with pytest.raises(ValueError, match="only with q_data_type=float8_e4m3fn"):
        wrapper.plan(mask, rows, cols, 2, 2, 128, q_data_type=torch.bfloat16, **full)


@requires_hopper
@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            "q_shape",
            r"q_scale must have shape \(1, 2, 192\) and dtype float32, got \(2, 192\) float32",
        ),
        (
            "k_groups",
            r"k_scale must have shape \(1, 2, 20\) and dtype float32, got \(1, 2, 19\) float32",
        ),
        (
            "v_shape",
            r"v_scale must have shape \(2, 128\) and dtype float32, got \(1, 2, 128\) float32",
        ),
        (
            "q_dtype",
            r"q_scale must have shape \(1, 2, 192\) and dtype float32, got \(1, 2, 192\) bfloat16",
        ),
        ("k_strided", r"k_scale must be contiguous on the planned device"),
        ("v_cpu", r"v_scale must be contiguous on the planned device"),
        # GQA: k_scale / v_scale are per kv head (query-head-shaped ones are refused by shape); g = 3 refused
        (
            "gqa_k_scale_at_query_heads",
            r"k_scale must have shape \(1, 1, 20\) and dtype float32, got \(1, 2, 20\) float32",
        ),
        ("gqa_ratio_3", r"g in \(2, 4, 8\)"),
    ],
)
def test_fp8_scale_validation(mutation, match):
    """Exact shapes ``(1, num_qo_heads, Sq)`` / ``(1, num_kv_heads, ceil(Sk / 16))`` / ``(num_kv_heads, 128)``,
    FP32, contiguous on the plan device; a GQA request is validated against the kv-head-shaped mask and scales."""
    mask, rows, cols = _descriptors(
        device="cuda"
    )  # H = 2, MB = 3, NB = 5 -> Sq 192, Sk 320, 20 groups
    h, kv_h = 2, 2
    q_scale = torch.ones((1, 2, 192), device="cuda")
    k_scale = torch.ones((1, 2, 20), device="cuda")
    v_scale = torch.ones((2, 128), device="cuda")
    if mutation == "q_shape":
        q_scale = q_scale[0]
    elif mutation == "k_groups":
        k_scale = k_scale[..., :19]
    elif mutation == "v_shape":
        v_scale = v_scale.unsqueeze(0)
    elif mutation == "q_dtype":
        q_scale = q_scale.bfloat16()
    elif mutation == "k_strided":
        k_scale = torch.ones((1, 2, 40), device="cuda")[..., ::2]
    elif mutation == "v_cpu":
        v_scale = v_scale.cpu()
    elif mutation == "gqa_k_scale_at_query_heads":
        mask, rows, cols = mask[:1], rows[:1], cols[:1]  # one kv head, two query heads
        kv_h = 1
        v_scale = v_scale[:1]
    else:  # gqa_ratio_3: six query heads over two kv heads
        h = 6
        q_scale = torch.ones((1, 6, 192), device="cuda")
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(0, device="cuda", dtype=torch.uint8), backend="cake"
    )
    with pytest.raises(ValueError, match=match):
        wrapper.plan(
            mask,
            rows,
            cols,
            h,
            kv_h,
            128,
            q_data_type=FP8,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )


@requires_hopper
def test_fp8_empty_rows_raise_the_bf16_error():
    """An empty query block is refused with the same ValueError on both data types."""
    mask, rows, cols = _descriptors(device="cuda")
    mask[0, 1] = False
    q_scale = torch.ones((1, 2, 192), device="cuda")
    k_scale = torch.ones((1, 2, 20), device="cuda")
    v_scale = torch.ones((2, 128), device="cuda")
    messages = {}
    for dtype in (torch.bfloat16, FP8):
        wrapper = VariableBlockSparseAttentionWrapper(
            torch.empty(0, device="cuda", dtype=torch.uint8), backend="cake"
        )
        scales = (
            {"q_scale": q_scale, "k_scale": k_scale, "v_scale": v_scale}
            if dtype is FP8
            else {}
        )
        with pytest.raises(
            ValueError, match="empty sparse rows|at least one KV block"
        ) as info:
            wrapper.plan(mask, rows, cols, 2, 2, 128, q_data_type=dtype, **scales)
        messages[dtype] = str(info.value)
    assert messages[torch.bfloat16] == messages[FP8]


@requires_hopper
def test_fp8_plan_stream_and_graph_lifetime():
    """An FP8 plan built on another stream is captured and replayed bit-exactly; the scales are plan data,
    so re-quantizing into the same scale tensors in place is what a replay sees."""
    h, mb, nb = 1, 16, 16
    mask = _random_mask(h, mb, nb, 16, seed=17, ragged=False, device="cuda")
    q, k, v = _inputs(h, mb, nb)
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        wrapper, (q8, k8, v8), (q_scale, k_scale, v_scale) = _fp8_plan(mask, q, k, v)
    plan = wrapper._cake_vsa_sm90_plan
    assert plan.fp8 and plan.stage.startswith("small_fp8_")
    expected = wrapper.run(q8, k8, v8).clone()
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    wrapper.run(q8, k8, v8, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q8, k8, v8, out=out)
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.view_as(expected), expected, atol=0, rtol=0)
    # re-quantize new inputs into the captured buffers (data and scales alike): the replay follows
    q.normal_()
    v.mul_(3.0)
    new = facade.sage_fp8_quantize_sm90(q, k, v)
    for captured, fresh in zip(
        (q8, k8, v8, q_scale, k_scale, v_scale), new, strict=True
    ):
        captured.copy_(fresh)
    graph.replay()
    exact = _reference(
        *_dequantize(q8, k8, v8, q_scale, k_scale, v_scale), mask, 128**-0.5
    )
    torch.testing.assert_close(
        out.view_as(expected).float(), exact, atol=5e-2, rtol=5e-2
    )
    with pytest.raises(RuntimeError, match="captured VSA plan cannot be replaced"):
        wrapper.plan(
            mask,
            torch.full((h, mb), 64, dtype=torch.int32, device="cuda"),
            torch.full((h, nb), 64, dtype=torch.int32, device="cuda"),
            h,
            h,
            128,
            q_data_type=FP8,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )


@requires_hopper
def test_fp8_and_bf16_plans_share_the_planner_bytes():
    """The FP8 twin consumes exactly the BF16 planner's ``meta`` / ``hdr`` (persistent) or ``plan`` (small)
    tensors: same route, same bytes, only the stage name and the bound scales differ."""
    for mask in (
        _random_mask(2, 3, 5, 2, seed=5, device="cuda"),  # small
        _random_mask(2, 160, 16, 12, seed=5, device="cuda"),  # persistent
    ):
        h, mb, nb = mask.shape
        q, k, v = _inputs(h, mb, nb)
        fp8 = _fp8_plan(mask, q, k, v)[0]._cake_vsa_sm90_plan
        bf16 = _wrapper(mask)._cake_vsa_sm90_plan
        assert (fp8.fp8, bf16.fp8) == (True, False) and fp8.mode == bf16.mode
        if fp8.small_kmax is None:
            assert torch.equal(fp8.meta, bf16.meta) and torch.equal(fp8.hdr, bf16.hdr)
            assert fp8.tile_stride == bf16.tile_stride and fp8.num_ctas == bf16.num_ctas
        else:
            assert (
                torch.equal(fp8.plan_param, bf16.plan_param)
                and fp8.num_items == bf16.num_items
            )
        assert fp8.k_scale_groups == -(-(nb * 64) // 16)


@requires_hopper
@pytest.mark.parametrize(
    "g,h_kv,mb,nb,capacity,scale",
    [
        (2, 2, 3, 5, 2, None),  # small kernel
        (4, 1, 16, 16, 16, 0.5),  # cluster
        (8, 1, 8, 64, 64, None),  # MQA, split-KV, full 64-block lists
        (4, 2, 160, 16, 12, -0.125),  # persistent pair plan
    ],
)
def test_fp8_gqa_route_against_the_dequantized_reference(
    g, h_kv, mb, nb, capacity, scale
):
    """``q_data_type=torch.float8_e4m3fn`` on a GQA problem: the package quantizer takes ``k`` / ``v`` over the kv
    heads and returns the kv-head-shaped scales, ``plan()`` takes them with the kv-head mask, the FP8 GQA twin
    of the BF16 GQA route runs, three launches are bit-identical, the BF16 output matches the exact softmax of
    the dequantized inputs (reference (b), 5e-2) and -- bit for bit on the static programs -- the FP8 MHA route
    on replicated inputs."""
    h = g * h_kv
    mask = _random_mask(h_kv, mb, nb, capacity, seed=5, ragged=True, device="cuda")
    mask_rep = mask.repeat_interleave(g, dim=0)
    q, k, v = _gqa_inputs(h, h_kv, mb, nb)
    q8, k8, v8, q_scale, k_scale, v_scale = facade.sage_fp8_quantize_sm90(q, k, v)
    groups = -(-(nb * 64) // 16)
    assert tuple(q_scale.shape) == (1, h, mb * 64)
    assert tuple(k_scale.shape) == (1, h_kv, groups) and tuple(v_scale.shape) == (
        h_kv,
        128,
    )
    wrapper = _gqa_wrapper(
        mask,
        h,
        scale=scale,
        q_data_type=FP8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    plan = wrapper._cake_vsa_sm90_plan
    bf16 = _gqa_wrapper(mask, h, scale=scale)._cake_vsa_sm90_plan
    assert plan.fp8 and plan.gqa == g and backend.stage_gqa(plan.stage)
    assert backend.stage_dtype(plan.stage) == "e4m3" and plan.mode == bf16.mode
    expected_stage = (
        bf16.stage.replace("attention", "attention_fp8")
        if bf16.stage.startswith("attention")
        else bf16.stage.replace("small_", "small_fp8_")
    )
    # the Sage-FP8 route serves static grouped-query plans whose lightest CTA runs at least
    # PIN_MIN_CTA_LOOP_ITERS main-loop iterations on the position-selected static program
    # (the pair plan of this list runs 13 and takes it from the threshold 8 on); its BF16 twin
    # is the plain static program, and the bitwise checks below hold on it (pin == static
    # bits).  The expectation comes from the planner decision on the plan's own device, not
    # from the stage the wrapper picked, so a plan routed away from the pin program fails here.
    sms = torch.cuda.get_device_properties(plan.device).multi_processor_count
    if backend.fp8_gqa_pin(
        backend.plan_vsa_sm90(mask.cpu(), sms=sms, gqa=g, split_tile_order="group")
    ):
        assert expected_stage == "attention_fp8_gqa"
        expected_stage = "attention_fp8_gqa_pin"
    assert plan.stage == expected_stage
    outs = [wrapper.run(q8, k8, v8).clone() for _ in range(3)]
    assert outs[0].dtype == torch.bfloat16 and outs[0].shape == q.shape
    for later in outs[1:]:
        torch.testing.assert_close(later, outs[0], atol=0, rtol=0)
    k8_rep, v8_rep = k8.repeat_interleave(g, dim=0), v8.repeat_interleave(g, dim=0)
    scales_rep = (
        q_scale,
        k_scale.repeat_interleave(g, dim=1),
        v_scale.repeat_interleave(g, dim=0),
    )
    exact = _reference(
        *_dequantize(q8, k8_rep, v8_rep, *scales_rep),
        mask_rep,
        128**-0.5 if scale is None else scale,
    )
    torch.testing.assert_close(outs[0].float(), exact, atol=5e-2, rtol=5e-2)
    # the FP8 MHA route on replicated inputs quantized through the MHA quantizer path: the same FP8 bits
    mha, (q8_m, k8_m, v8_m), scales_m = _fp8_plan(
        mask_rep, q, k.repeat_interleave(g, dim=0), v.repeat_interleave(g, dim=0), scale
    )
    assert torch.equal(q8_m.view(torch.uint8), q8.view(torch.uint8))
    assert torch.equal(k8_m.view(torch.uint8), k8_rep.view(torch.uint8))
    assert torch.equal(v8_m.view(torch.uint8), v8_rep.view(torch.uint8))
    for a, b in zip(scales_m, scales_rep, strict=True):
        assert torch.equal(a, b)
    expected = mha.run(q8_m, k8_m, v8_m)
    plan_mha = mha._cake_vsa_sm90_plan
    assert not backend.stage_gqa(plan_mha.stage) and plan_mha.mode == plan.mode
    if plan.small_kmax is None or not (plan.small_split or plan.small_cluster):
        torch.testing.assert_close(outs[0], expected, atol=0, rtol=0)
    else:
        torch.testing.assert_close(
            outs[0].float(), expected.float(), atol=1e-2, rtol=1e-2
        )
