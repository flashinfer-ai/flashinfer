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

"""GPU coverage for the common probability-descending, index-ascending order."""

import pytest
import torch

import flashinfer


def _renorm_reference(probs, *, k=None, p=None):
    order = probs.argsort(dim=-1, descending=True, stable=True)
    sorted_probs = probs.gather(1, order).double()
    keep = torch.ones_like(sorted_probs, dtype=torch.bool)
    if k is not None:
        k = torch.as_tensor(k, device=probs.device).reshape(-1, 1)
        keep &= torch.arange(probs.shape[1], device=probs.device) < k
    if p is not None:
        p = torch.as_tensor(p, dtype=torch.float32, device=probs.device).reshape(-1, 1)
        before = sorted_probs.cumsum(-1) - sorted_probs
        keep &= before < p.double() * sorted_probs.sum(-1, keepdim=True)
    selected = torch.where(keep, sorted_probs, 0)
    selected /= selected.sum(-1, keepdim=True)
    return torch.zeros_like(selected).scatter(1, order, selected).to(probs.dtype)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("mode", ["top_k", "top_p", "joint"])
@pytest.mark.parametrize("tensor_params", [False, True])
def test_rejected_boundary_candidate_is_never_returned(
    mode, deterministic, tensor_params
):
    # The old loop can return its rejected candidate when pivot_0 == pivot_1 == high.
    # Lower indices win, so the only legal token is 2, not 3.
    probs = torch.tensor([[8, 10, 23, 23]], dtype=torch.float32, device="cuda") / 64
    indices = torch.zeros(32768, dtype=torch.int32, device="cuda")
    k = torch.tensor([1], dtype=torch.int32, device="cuda") if tensor_params else 1
    p = torch.tensor([0.3], device="cuda") if tensor_params else 0.3
    kwargs = dict(
        indices=indices,
        deterministic=deterministic,
        seed=42,
        return_valid=True,
    )
    if mode == "top_k":
        samples, valid = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, **kwargs
        )
    elif mode == "top_p":
        samples, valid = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, **kwargs
        )
    else:
        p = torch.tensor([0.99], device="cuda") if tensor_params else 0.99
        samples, valid = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, k, p, filter_apply_order="joint", **kwargs
        )
    assert valid.all()
    assert (samples == 2).all(), samples.bincount(minlength=4)


@pytest.mark.parametrize("vocab_size", [4, 257, 131072])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tensor_params", [False, True])
def test_top_k_filters_keep_exact_index_prefix(vocab_size, dtype, tensor_params):
    # The all-tie row crosses CTA boundaries at large vocab sizes. Mixed k also
    # exercises persistent groups after an unfiltered row.
    logits = torch.zeros((3, vocab_size), device="cuda", dtype=dtype)
    logits[0, -1] = 2
    logits[0, 1] = 1
    logits[2, ::3] = 1
    probs = torch.softmax(logits.float(), dim=-1).to(dtype)
    k = (
        torch.tensor([vocab_size, 2, 1], dtype=torch.int32, device="cuda")
        if tensor_params
        else 2
    )
    expected_probs = _renorm_reference(probs, k=k)
    actual_probs = flashinfer.sampling.top_k_renorm_probs(probs, k)
    assert torch.equal(actual_probs > 0, expected_probs > 0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=0.01, atol=1e-6)

    masked = flashinfer.sampling.top_k_mask_logits(logits, k)
    order = logits.argsort(dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order).scatter_(
        1, order, torch.arange(vocab_size, device="cuda").expand_as(order)
    )
    ks = torch.as_tensor(k, device="cuda").reshape(-1, 1)
    expected_keep = rank < ks
    assert torch.equal(torch.isfinite(masked), expected_keep)
    assert torch.equal(masked[expected_keep], logits[expected_keep])


@pytest.mark.parametrize("vocab_size", [4, 257, 131072])
@pytest.mark.parametrize("p", [0.3, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("tensor_params", [False, True])
def test_top_p_renorm_keeps_shortest_index_prefix(vocab_size, p, tensor_params):
    probs = torch.ones((2, vocab_size), dtype=torch.float32, device="cuda")
    probs[0, 0] = 4
    probs /= probs.sum(-1, keepdim=True)
    param = torch.tensor([p, p], device="cuda") if tensor_params else p
    expected = _renorm_reference(probs, p=param)
    actual = flashinfer.sampling.top_p_renorm_probs(probs, param)
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("mode", ["top_k", "top_p", "joint"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_boundary_sampling_matches_renorm_distribution(mode, deterministic):
    probs = torch.tensor([[4, 3, 3, 0]], dtype=torch.float32, device="cuda") / 10
    indices = torch.zeros(65536, dtype=torch.int32, device="cuda")
    kwargs = dict(indices=indices, deterministic=deterministic, seed=123)
    if mode == "top_k":
        expected = flashinfer.sampling.top_k_renorm_probs(probs, 2)
        samples = flashinfer.sampling.top_k_sampling_from_probs(probs, 2, **kwargs)
    elif mode == "top_p":
        expected = flashinfer.sampling.top_p_renorm_probs(probs, 0.6)
        samples = flashinfer.sampling.top_p_sampling_from_probs(probs, 0.6, **kwargs)
    else:
        expected = flashinfer.sampling.top_k_renorm_probs(probs, 2)
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, 2, 1.0, filter_apply_order="joint", **kwargs
        )
    counts = samples.long().bincount(minlength=4)
    assert counts[2] == 0 and counts[3] == 0
    torch.testing.assert_close(
        counts.float() / counts.sum(), expected[0], rtol=0, atol=0.015
    )


@pytest.mark.parametrize("order", ["joint", "top_k_first"])
@pytest.mark.parametrize("fast_path", [False, True])
def test_top_k_first_and_joint_share_tie_convention(order, fast_path):
    vocab_size = 65536 if fast_path else 257
    probs = torch.zeros((32, vocab_size), dtype=torch.float32, device="cuda")
    probs[:, [1, 3]] = 0.5
    samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs, 1, 1.0, filter_apply_order=order, seed=42
    )
    assert (samples == 1).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_mask_treats_signed_zero_as_a_tie(dtype):
    logits = torch.tensor([[-0.0, 0.0, -1.0, -2.0]], device="cuda", dtype=dtype)
    actual = flashinfer.sampling.top_k_mask_logits(logits, 1)
    assert torch.equal(
        torch.isfinite(actual),
        torch.tensor([[True, False, False, False]], device="cuda"),
    )


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("from_logits", [False, True])
def test_fast_path_preserves_original_order_inside_top_p_boundary(
    deterministic, from_logits
):
    probs = torch.zeros((32, 65536), dtype=torch.float32, device="cuda")
    probs[:, [0, 1, 512, 1023]] = torch.tensor([0.1, 0.4, 0.4, 0.1], device="cuda")
    if from_logits:
        samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            probs.log(), 3, 0.3, deterministic=deterministic, seed=42
        )
    else:
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, 3, 0.3, deterministic=deterministic, seed=42
        )
    assert (samples == 1).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [1, 17, 256])
def test_explicit_small_index_selection_across_index_tiles(dtype, k):
    scores = torch.ones((2, 65536), device="cuda", dtype=dtype)
    values, indices = flashinfer.top_k(
        scores,
        k,
        sorted=True,
        deterministic=True,
        tie_break=flashinfer.TopKTieBreak.SMALL,
    )
    expected = torch.arange(k, device="cuda").expand_as(indices)
    assert torch.equal(indices, expected)
    assert torch.equal(values, torch.ones_like(values))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_uniform_ties_remain_exact_through_persistent_rows(dtype):
    vocab = 131072
    probs = torch.full((1024, vocab), 1 / vocab, device="cuda", dtype=dtype)
    logits = torch.zeros_like(probs)
    k = torch.tensor([vocab, 17, 1, 513], device="cuda", dtype=torch.int32).repeat(256)
    expected = torch.arange(vocab, device="cuda") < k.unsqueeze(1)
    for _ in range(10):
        renormed = flashinfer.sampling.top_k_renorm_probs(probs, k)
        masked = flashinfer.sampling.top_k_mask_logits(logits, k)
        assert torch.equal(renormed > 0, expected), (
            renormed.gt(0) != expected
        ).nonzero()[:16]
        assert torch.equal(torch.isfinite(masked), expected), (
            torch.isfinite(masked) != expected
        ).nonzero()[:16]


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("tensor_params", [False, True])
@pytest.mark.parametrize("from_logits", [False, True])
def test_fast_path_ties_created_by_softmax_use_original_indices(
    deterministic, tensor_params, from_logits
):
    logits = torch.full((32, 65536), -float("inf"), device="cuda")
    logits[:, 0] = -1
    logits[:, 1] = 0
    logits[:, 512] = 1e-8
    probs = logits.softmax(-1)
    assert torch.equal(probs[:, 1], probs[:, 512])
    k = torch.full((32,), 3, device="cuda", dtype=torch.int32) if tensor_params else 3
    kwargs = dict(deterministic=deterministic, seed=42)
    if tensor_params:
        kwargs["top_k_max"] = 3
    if from_logits:
        samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, 0.3, **kwargs
        )
    else:
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, k, 0.3, **kwargs
        )
    assert (samples == 1).all()
