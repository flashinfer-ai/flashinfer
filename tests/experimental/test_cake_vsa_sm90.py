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

"""Cake SM90 VSA (``backend="cake"`` on Hopper): planning, numerics, lifetime, API."""


import warnings

import pytest
import torch

from flashinfer.cake_vsa_sm90 import (
    MAX_OWN,
    META_NOWN,
    META_NSEQ,
    META_NTILES,
    META_OWN_OFF,
    META_SEQ_OFF,
    META_WORDS,
    OWN_WORDS,
    _schedule_feasible,
    plan_vsa_sm90,
)


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


# ---------------------------------------------------------------------------
# Host planner (CPU only)
# ---------------------------------------------------------------------------


def _decode_tiles(plan):
    """Yield (head, qb0, qb1, mode, positions[(blk_a, blk_b)], owns[2]) per tile."""
    meta = plan["meta"].numpy().view("uint32")
    stride = plan["tile_stride"]
    for c in range(plan["num_ctas"]):
        n_tiles = int(meta[c * stride, META_NTILES])
        assert 1 <= n_tiles <= stride
        for i in range(n_tiles):
            row = meta[c * stride + i]
            head, qb0, qb1, mode = (int(x) for x in row[:4])
            n_seq = int(row[META_NSEQ])
            words = row[META_SEQ_OFF : META_SEQ_OFF + n_seq]
            positions = []
            for w in words:
                a, b = int(w & 0xFFFF), int(w >> 16)
                positions.append((a, b if b != 0xFFFF else -1))
            owns = []
            for wg in range(2):
                n_own = int(row[META_NOWN + wg])
                assert n_own <= MAX_OWN
                base = META_OWN_OFF + wg * OWN_WORDS
                packed = row[base : base + (n_own + 1) // 2]
                entries = []
                for w in packed:
                    entries.extend((int(w & 0xFFFF), int(w >> 16)))
                owns.append(entries[:n_own])
            yield head, qb0, qb1, mode, positions, owns


@pytest.mark.parametrize(
    "h,mb,nb,capacity,ragged",
    [
        (1, 1, 1, 1, False),
        (2, 3, 5, 2, True),
        (4, 16, 8, 4, True),
        (8, 64, 64, 32, False),
        (7, 64, 64, 16, True),
    ],
)
def test_plan_covers_every_query_block_once_and_is_deadlock_free(
    h, mb, nb, capacity, ragged
):
    mask = _random_mask(h, mb, nb, capacity, ragged=ragged)
    plan = plan_vsa_sm90(mask, sms=132)
    assert plan["meta"].shape[1] == META_WORDS
    assert plan["num_ctas"] == min(plan["num_tiles"], 132)
    seen = set()
    for head, qb0, qb1, mode, positions, owns in _decode_tiles(plan):
        blocks = {qb0} if mode else {qb0, qb1}
        assert mode in (0, 1, 2)
        for qb in blocks:
            assert (head, qb) not in seen
            seen.add((head, qb))
        # Every selected KV block of each covered query block travels exactly once.
        bits = [0] * len(positions)
        served = {qb: [] for qb in blocks}
        for wg, entries in enumerate(owns):
            for entry in entries:
                pos, has2, use = entry >> 3, (entry >> 2) & 1, entry & 3
                assert use & (1 << wg)
                assert has2 == (positions[pos][1] >= 0)
                bits[pos] |= use
        for pos, (a, b) in enumerate(positions):
            for blk in (a, b):
                if blk >= 0:
                    consumers = (
                        [qb0, qb1]
                        if (mode == 0 and bits[pos] == 3)
                        else ([qb0] if (mode or bits[pos] == 1) else [qb1])
                    )
                    for qb in consumers:
                        served[qb].append(blk)
        for qb in blocks:
            expected = mask[head, qb].nonzero().flatten().tolist()
            assert sorted(served[qb]) == expected, (head, qb)
        assert _schedule_feasible([bt for bt in bits])
    assert seen == {(head, qb) for head in range(h) for qb in range(mb)}


def test_plan_rejects_unsupported_masks():
    with pytest.raises(ValueError, match="empty"):
        mask = torch.ones((1, 2, 3), dtype=torch.bool)
        mask[0, 1] = False
        plan_vsa_sm90(mask, sms=132)
    with pytest.raises(ValueError, match="at most 64"):
        plan_vsa_sm90(torch.ones((1, 1, 65), dtype=torch.bool), sms=132)
    with pytest.raises(ValueError, match="boolean"):
        plan_vsa_sm90(torch.ones((1, 1, 2), dtype=torch.int32), sms=132)


def test_plan_mode_selection_prefers_split_below_full_occupancy():
    mask = _random_mask(8, 16, 16, 12, ragged=False)
    assert plan_vsa_sm90(mask, sms=132)["mode"] == "split"
    dense = _random_mask(8, 64, 64, 32, ragged=False)
    assert plan_vsa_sm90(dense, sms=132)["mode"] == "pair"


# ---------------------------------------------------------------------------
# GPU coverage (Hopper only)
# ---------------------------------------------------------------------------

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


def _inputs(h, mb, nb):
    return tuple(
        torch.randn((h, length * 64, 128), device="cuda", dtype=torch.bfloat16)
        for length in (mb, nb, nb)
    )


def _reference(q, k, v, mask, scale):
    """Independent FP32 reference: masked dense softmax attention per head."""
    scores = torch.einsum("hmd,hnd->hmn", q.float(), k.float()) * float(scale)
    dense = mask.to(q.device).repeat_interleave(64, dim=1).repeat_interleave(64, dim=2)
    scores.masked_fill_(~dense, float("-inf"))
    return torch.einsum("hmn,hnd->hmd", torch.softmax(scores, dim=-1), v.float()).to(
        q.dtype
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
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
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
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


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
    from flashinfer.sparse import VariableBlockSparseAttentionWrapper

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
    from flashinfer.sparse import VariableBlockSparseAttentionWrapper

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
        out.view_as(q), _reference(q, k, v, mask, 128**-0.5), atol=0.01, rtol=0.01
    )
    with warnings.catch_warnings():
        # The capture is abandoned before any launch; torch warns about the empty graph.
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="preallocated"):
            graph2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph2):
                wrapper.run(q, k, v)
