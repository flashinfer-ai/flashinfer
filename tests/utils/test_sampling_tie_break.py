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
        keep &= (before < p.double()) | (p >= 1)
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


@pytest.mark.parametrize("vocab_size", [8192, 131072])
@pytest.mark.parametrize("mode", ["top_k", "top_p", "joint"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_large_vocab_sampling_resolves_boundary_ties(vocab_size, mode, deterministic):
    # A modest output batch keeps Hopper's split-row dispatch active. Ties span
    # multiple chunks; indices also check per-input-row threshold lookup.
    probs = torch.zeros((5, vocab_size), dtype=torch.float32, device="cuda")
    positions = torch.tensor(
        [3, vocab_size // 4 - 1, vocab_size // 2 + 1, vocab_size - 1],
        device="cuda",
    )
    probs[0, positions] = 0.25
    probs[1, positions] = torch.tensor([8, 10, 23, 23], device="cuda") / 64
    probs[2, 0] = 0.5
    probs[2, positions[:2]] = 0.125
    probs[2, positions[2:]] = 0.0625
    probs[2, 1:3] = 0.0625
    probs[3, positions[:3]] = torch.tensor([0.5, 0.2, 0.2], device="cuda")
    probs[4, positions[:3]] = torch.tensor([0.5, 0.5, 2**-20], device="cuda")
    k = torch.tensor([2, 1, 5, 2, vocab_size], dtype=torch.int32, device="cuda")
    p = torch.tensor([0.5, 0.3, 0.875, 0.52, 1.0], device="cuda")
    if mode == "joint":
        p[1] = 0.99
    expected = _renorm_reference(
        probs, k=k if mode != "top_p" else None, p=p if mode != "top_k" else None
    )
    indices = torch.arange(256, device="cuda", dtype=torch.int64) % len(probs)
    for offset in (0, 17, 65537):
        kwargs = dict(
            indices=indices,
            deterministic=deterministic,
            seed=42,
            offset=offset,
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
            samples, valid = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, k, p, filter_apply_order="joint", **kwargs
            )
        assert valid.all()
        assert (expected[indices, samples.long()] > 0).all()


@pytest.mark.parametrize("vocab_size", [8192, 8193, 131072])
@pytest.mark.parametrize("mode", ["top_k", "top_p", "joint"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("higher_at_end", [False, True])
def test_large_vocab_ties_preserve_legal_proposals_and_distribution(
    vocab_size, mode, deterministic, higher_at_end
):
    # Dyadic masses make the first CDF draw identical across the single-CTA
    # and cluster hierarchies. A legal first proposal must never be redrawn.
    probs = torch.zeros((1, vocab_size), device="cuda")
    ties = torch.tensor(
        [3, vocab_size // 4 - 1, vocab_size // 2 + 1, vocab_size - 1],
        device="cuda",
    )
    higher_index = vocab_size - 2 if higher_at_end else 0
    probs[0, higher_index] = 0.5
    probs[0, ties] = 0.125
    indices = torch.zeros(8192, dtype=torch.int64, device="cuda")
    support = torch.cat(
        [torch.tensor([higher_index], dtype=torch.int64, device="cuda"), ties[:2]]
    )
    counts = torch.zeros(3, device="cuda")
    for offset in (0, 65537):
        kwargs = dict(
            indices=indices,
            deterministic=deterministic,
            seed=42,
            offset=offset,
            return_valid=True,
        )
        proposal, proposal_valid = flashinfer.sampling.sampling_from_probs(
            probs, **kwargs
        )
        if mode == "top_k":
            samples, valid = flashinfer.sampling.top_k_sampling_from_probs(
                probs, 3, **kwargs
            )
        elif mode == "top_p":
            samples, valid = flashinfer.sampling.top_p_sampling_from_probs(
                probs, 0.75, **kwargs
            )
        else:
            samples, valid = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, 3, 0.9, filter_apply_order="joint", **kwargs
            )
        assert proposal_valid.all() and valid.all()
        legal_proposal = (proposal[:, None] == support).any(-1)
        torch.testing.assert_close(
            samples[legal_proposal], proposal[legal_proposal], atol=0, rtol=0
        )
        assert (samples[:, None] == support).any(-1).all()
        counts += (samples[:, None] == support).sum(0)
    torch.testing.assert_close(
        counts / counts.sum(),
        torch.tensor([2 / 3, 1 / 6, 1 / 6], device="cuda"),
        atol=0.025,
        rtol=0,
    )


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


@pytest.mark.parametrize("vocab_size", [4, 4096])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    "values,p",
    [([0.5, 0.2, 0.2], 0.52), ([0.5, 0.2, 0.19], 0.52), ([0.5, 0.3, 0.1], 0.55)],
)
def test_top_p_uses_absolute_threshold_with_and_without_ties(
    vocab_size, deterministic, values, p
):
    # Cover both the small-vocab search and AIR. The sum differs from one
    # intentionally: a tied boundary must not change the threshold definition.
    probs = torch.zeros((1, vocab_size), device="cuda")
    probs[0, :3] = torch.tensor(values, device="cuda")
    expected = _renorm_reference(probs, p=p)
    actual = flashinfer.sampling.top_p_renorm_probs(probs, p)
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected)
    indices = torch.zeros(8192, dtype=torch.int64, device="cuda")
    for mode in ("top_p", "joint"):
        if mode == "top_p":
            samples = flashinfer.sampling.top_p_sampling_from_probs(
                probs, p, indices=indices, deterministic=deterministic, seed=42
            )
        else:
            samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs,
                2,
                p,
                indices=indices,
                filter_apply_order="joint",
                deterministic=deterministic,
                seed=42,
            )
        counts = samples.bincount(minlength=vocab_size).float() / indices.numel()
        assert not counts[2:].any()
        torch.testing.assert_close(counts, expected[0], rtol=0, atol=0.025)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("mode", ["top_k", "top_p", "joint"])
def test_whole_tied_groups_before_boundary_keep_sampling_distribution(
    mode, deterministic
):
    probs = torch.tensor(
        [[0.5, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625, 0]],
        device="cuda",
        dtype=torch.bfloat16,
    ).float()
    expected = _renorm_reference(probs, k=5, p=0.875)
    indices = torch.zeros(8192, dtype=torch.int32, device="cuda")
    kwargs = dict(indices=indices, deterministic=deterministic, seed=19)
    if mode == "top_k":
        samples = flashinfer.sampling.top_k_sampling_from_probs(probs, 5, **kwargs)
    elif mode == "top_p":
        samples = flashinfer.sampling.top_p_sampling_from_probs(probs, 0.875, **kwargs)
    else:
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, 5, 0.875, filter_apply_order="joint", **kwargs
        )
    counts = samples.bincount(minlength=8).float() / indices.numel()
    assert not counts[5:].any()
    torch.testing.assert_close(counts, expected[0], rtol=0, atol=0.025)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_rejection_kernels_launch_across_vector_widths(deterministic, indices_dtype):
    # Exercise public fp32 vector widths and promoted fp16/bf16 inputs. On
    # SM90/SM100 these small rows still use the 1024-thread launch budget.
    for vocab in (257, 258, 260, 264, 272):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            probs = torch.full((1, vocab), 1 / vocab, device="cuda", dtype=dtype)
            indices = torch.zeros(16, device="cuda", dtype=indices_dtype)
            kwargs = dict(indices=indices, deterministic=deterministic, seed=7)
            outputs = (
                flashinfer.sampling.top_k_sampling_from_probs(probs, 1, **kwargs),
                flashinfer.sampling.top_p_sampling_from_probs(
                    probs, 0.25 / vocab, **kwargs
                ),
                flashinfer.sampling.top_k_top_p_sampling_from_probs(
                    probs, 1, 0.99, filter_apply_order="joint", **kwargs
                ),
            )
            torch.cuda.synchronize()
            for output in outputs:
                assert output.dtype == indices_dtype
                assert (output == 0).all()


@pytest.mark.parametrize("is_deterministic", [False, True])
@pytest.mark.parametrize("p", [0.625, 0.8125, 1.0])
def test_air_sparse_boundary_ties_cross_index_halves(p, is_deterministic):
    probs = torch.zeros((1, 131072), device="cuda")
    probs[0, 0] = 0.5625
    positions = torch.tensor(
        [3, 1023, 1024, 65535, 65536, 65537, 131071], device="cuda"
    )
    probs[0, positions] = 0.0625
    expected = _renorm_reference(probs, p=p)
    actual = flashinfer.sampling.top_p_renorm_probs(probs, p, is_deterministic)
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("vocab_size", [4, 4096])
def test_top_p_one_keeps_all_tokens_despite_row_sum_roundoff(vocab_size):
    probs = torch.zeros((1, vocab_size), device="cuda")
    probs[0, :3] = torch.tensor([0.5, 0.5, 2**-20], device="cuda")
    actual = flashinfer.sampling.top_p_renorm_probs(probs, 1.0)
    assert torch.equal(actual > 0, probs > 0)
    torch.testing.assert_close(actual, probs / probs.sum(-1, keepdim=True))


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
@pytest.mark.parametrize("is_deterministic", [False, True])
def test_sparse_tie_prefix_scratch_is_reused_across_rows(dtype, is_deterministic):
    # More rows than SMs guarantees persistent iterations even with one CTA/group.
    batch = torch.cuda.get_device_properties(0).multi_processor_count + 4
    vocab = 32768
    columns = torch.arange(vocab, device="cuda")
    logits = (columns % 3 == 0).to(dtype).expand(batch, -1).contiguous()
    probs = logits / (vocab // 3 + 1)
    k = torch.tensor([vocab, 17, 1, 513], device="cuda", dtype=torch.int32)
    k = k.repeat((batch + 3) // 4)[:batch]
    expected = (columns % 3 == 0) & (columns // 3 < k.unsqueeze(1))
    for _ in range(3):
        renormed = flashinfer.sampling.top_k_renorm_probs(probs, k, is_deterministic)
        masked = flashinfer.sampling.top_k_mask_logits(logits, k)
        assert torch.equal(renormed > 0, expected)
        assert torch.equal(torch.isfinite(masked), expected | (k == vocab).unsqueeze(1))


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
