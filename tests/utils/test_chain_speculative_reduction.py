"""Regression coverage for chain speculative sampling across vocabulary tiles."""

import pytest
import torch

import flashinfer


def _draft_and_target(batch_size, num_speculate_tokens, vocab_size, device="cuda"):
    pre_norm_draft = torch.rand(
        batch_size, num_speculate_tokens, vocab_size, device=device
    )
    draft_probs = pre_norm_draft / pre_norm_draft.sum(dim=-1, keepdim=True)
    draft_token_ids = torch.randint(
        vocab_size, (batch_size, num_speculate_tokens), device=device, dtype=torch.int32
    )
    pre_norm_target = torch.rand(
        batch_size, num_speculate_tokens + 1, vocab_size, device=device
    )
    target_probs = pre_norm_target / pre_norm_target.sum(dim=-1, keepdim=True)
    return draft_probs, draft_token_ids, target_probs


@pytest.mark.parametrize(
    "vocab_size", [33, 1023, 1024, 1025, 4095, 4096, 4097, 32001, 151937]
)
@pytest.mark.parametrize("num_speculate_tokens", [1, 3])
@pytest.mark.parametrize("deterministic", [False, True])
def test_chain_spec_tile_boundaries(vocab_size, num_speculate_tokens, deterministic):
    # The residual sum spans every vocabulary tile, so vocabularies just above and
    # below a tile edge exercise the masked tail of the final tile.
    torch.manual_seed(20260913)
    draft_probs, draft_token_ids, target_probs = _draft_and_target(
        4, num_speculate_tokens, vocab_size
    )
    accepted = torch.zeros(4, dtype=torch.int32, device="cuda")
    emitted = torch.zeros(4, dtype=torch.int32, device="cuda")
    output, accepted, emitted = flashinfer.chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        target_probs,
        accepted,
        emitted,
        deterministic=deterministic,
        seed=12345,
        offset=17,
    )
    assert output.shape == (4, num_speculate_tokens + 1)
    # Position 0 is always written, so every row emits at least one real token.
    # Without this the range check below could pass on an all-padding output.
    assert torch.all(output[:, 0] >= 0)
    valid = output[output >= 0]
    assert valid.numel() >= output.shape[0]
    assert torch.all(valid < vocab_size)
    assert torch.all(emitted + 1 == (output != -1).sum(dim=1))


@pytest.mark.parametrize("vocab_size", [4097, 151937])
@pytest.mark.parametrize("deterministic", [False, True])
def test_chain_spec_onehot_target_is_exact(vocab_size, deterministic):
    # A one-hot target that agrees with every draft token must accept the whole
    # chain, which pins the output regardless of how the residual sum is reduced.
    torch.manual_seed(20260913)
    batch_size, num_speculate_tokens = 4, 3
    pre_norm_draft = torch.rand(
        batch_size, num_speculate_tokens, vocab_size, device="cuda"
    )
    draft_probs = pre_norm_draft / pre_norm_draft.sum(dim=-1, keepdim=True)
    draft_token_ids = torch.randint(
        vocab_size,
        (batch_size, num_speculate_tokens),
        device="cuda",
        dtype=torch.int32,
    )
    target_token_ids = torch.randint(
        vocab_size,
        (batch_size, num_speculate_tokens + 1),
        device="cuda",
        dtype=torch.int32,
    )
    target_token_ids[..., :num_speculate_tokens] = draft_token_ids
    target_probs = torch.zeros(
        (batch_size, num_speculate_tokens + 1, vocab_size), device="cuda"
    )
    target_probs.scatter_(2, target_token_ids.long().unsqueeze(-1), 1)
    accepted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    emitted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    output, accepted, emitted = flashinfer.chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        target_probs,
        accepted,
        emitted,
        deterministic=deterministic,
        seed=12345,
        offset=17,
    )
    torch.testing.assert_close(output, target_token_ids, rtol=0, atol=0)


@pytest.mark.parametrize("vocab_size", [1024, 8193])
def test_chain_spec_all_draft_rejected(vocab_size):
    # Disjoint draft and target supports force rejection at the first position,
    # so every row samples from the residual and the tail is padded with -1.
    torch.manual_seed(20260913)
    batch_size, num_speculate_tokens = 4, 3
    half = vocab_size // 2
    draft_probs = torch.zeros(
        batch_size, num_speculate_tokens, vocab_size, device="cuda"
    )
    draft_probs[..., :half] = 1.0 / half
    draft_token_ids = torch.zeros(
        (batch_size, num_speculate_tokens), device="cuda", dtype=torch.int32
    )
    target_probs = torch.zeros(
        batch_size, num_speculate_tokens + 1, vocab_size, device="cuda"
    )
    target_probs[..., half:] = 1.0 / (vocab_size - half)
    accepted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    emitted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    output, accepted, emitted = flashinfer.chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        target_probs,
        accepted,
        emitted,
        seed=12345,
        offset=17,
    )
    # The first token is resampled from the residual, which lives in the upper half.
    assert torch.all(output[:, 0] >= half)
    assert torch.all(output[:, 1:] == -1)
    assert torch.all(emitted == 0)
    # No draft token can be accepted when the supports are disjoint.
    assert torch.all(accepted == 0)


@pytest.mark.parametrize("vocab_size", [1025, 32001])
def test_chain_spec_residual_has_one_nonzero_index(vocab_size):
    # The draft puts all of its mass on index 0, so relu(target - draft) is zero
    # everywhere except the last index. The residual sum therefore has exactly one
    # contributing term no matter which tile it lands in, and the resampled token
    # is pinned to that index.
    torch.manual_seed(20260913)
    batch_size = 4
    draft_probs = torch.zeros(batch_size, 1, vocab_size, device="cuda")
    draft_probs[..., 0] = 1.0
    draft_token_ids = torch.zeros((batch_size, 1), device="cuda", dtype=torch.int32)
    target_probs = torch.zeros(batch_size, 2, vocab_size, device="cuda")
    target_probs[:, 0, vocab_size - 1] = 1.0
    target_probs[:, 1, 0] = 1.0
    accepted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    emitted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    output, accepted, emitted = flashinfer.chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        target_probs,
        accepted,
        emitted,
        seed=12345,
        offset=17,
    )
    # The target gives the drafted token zero probability, so it is always
    # rejected and every row resamples the one index carrying residual mass.
    assert torch.all(output[:, 0] == vocab_size - 1)
    assert torch.all(output[:, 1:] == -1)
    assert torch.all(accepted == 0)


@pytest.mark.parametrize("vocab_size", [4097, 32000, 32001, 128256, 151937])
def test_chain_spec_residual_normalizer_selects_token(vocab_size):
    # The tests above pin a token by which index carries residual mass, so they
    # cannot see an error in the residual sum itself: u = uniform * sum, and a
    # single-index residual absorbs any u. Here the residual is spread over many
    # indices with unequal mass, so the value of the sum decides where the
    # inverse-CDF walk stops. A reduction that drops tiles, drops vector lanes,
    # or skips the cross-thread step samples from a truncated distribution and
    # shifts the mean of the sampled index.
    batch_size = 512
    draft_probs = torch.zeros(batch_size, 1, vocab_size, device="cuda")
    draft_probs[..., 0] = 1.0
    draft_token_ids = torch.zeros((batch_size, 1), device="cuda", dtype=torch.int32)
    target_probs = torch.zeros(batch_size, 2, vocab_size, device="cuda")
    # Mass on every 65th index, rising with position, so the cumulative walk
    # crosses many tiles before it terminates. The stride must not be a multiple
    # of the vector width, or every mass-bearing index shares one vector lane
    # and a reduction that drops the other lanes would only drop zeros.
    idx = torch.arange(1, vocab_size, 65, device="cuda")
    weights = torch.arange(1, idx.numel() + 1, device="cuda", dtype=torch.float32)
    probs = weights / weights.sum()
    target_probs[:, 0, idx] = probs
    target_probs[:, 1, 0] = 1.0

    # Analytic mean and standard error of the index under the correct residual.
    expected_mean = (idx.double() * probs.double()).sum()
    variance = ((idx.double() - expected_mean) ** 2 * probs.double()).sum()
    samples = []
    for seed in (12345, 999, 7):
        accepted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        emitted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        output, accepted, emitted = flashinfer.chain_speculative_sampling(
            draft_probs,
            draft_token_ids,
            target_probs,
            accepted,
            emitted,
            seed=seed,
            offset=17,
        )
        # The drafted token has zero target mass, so every row resamples from
        # the residual and must land on one of the weighted indices.
        assert torch.all(accepted == 0)
        assert torch.all(output[:, 1:] == -1)
        assert torch.all(torch.isin(output[:, 0], idx.to(output.dtype)))
        samples.append(output[:, 0].double())

    drawn = torch.cat(samples)
    standard_error = (variance / drawn.numel()).sqrt()
    # Six standard errors: a correct kernel exceeds this about 1 time in 5e8,
    # while a reduction that under-counts the residual pulls the mean down by
    # far more than that.
    assert (drawn.mean() - expected_mean).abs() < 6 * standard_error


@pytest.mark.parametrize("deterministic", [False, True])
def test_chain_spec_graph_replay_updates_input(deterministic):
    # Replay has to read the current contents of the captured tensors. Rewriting
    # the residual between replays moves the only resampleable index, so a replay
    # that reused stale input would return the previous token.
    batch_size, vocab_size = 4, 8193
    draft_probs = torch.zeros(batch_size, 1, vocab_size, device="cuda")
    draft_probs[..., 0] = 1.0
    draft_token_ids = torch.zeros((batch_size, 1), device="cuda", dtype=torch.int32)
    target_probs = torch.zeros(batch_size, 2, vocab_size, device="cuda")
    target_probs[:, 1, 0] = 1.0
    accepted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    emitted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    seed = torch.tensor([12345], device="cuda", dtype=torch.uint64)
    offset = torch.tensor([17], device="cuda", dtype=torch.uint64)

    def sample():
        return flashinfer.chain_speculative_sampling(
            draft_probs,
            draft_token_ids,
            target_probs,
            accepted,
            emitted,
            deterministic=deterministic,
            seed=seed,
            offset=offset,
        )

    target_probs[:, 0, 8192] = 1.0
    sample()  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _, _ = sample()
    for token in (8192, 4096, 1023, 1):
        target_probs[:, 0].zero_()
        target_probs[:, 0, token] = 1.0
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(output[:, 0] == token)


def test_chain_spec_counters_accumulate_across_calls():
    # accepted and emitted are accumulators the kernel adds into, so repeated
    # calls with the same pinned inputs must advance them by the same amount
    # each time.
    batch_size, vocab_size = 4, 4097
    draft_probs = torch.zeros(batch_size, 1, vocab_size, device="cuda")
    draft_probs[..., 0] = 1.0
    draft_token_ids = torch.zeros((batch_size, 1), device="cuda", dtype=torch.int32)
    target_probs = torch.zeros(batch_size, 2, vocab_size, device="cuda")
    target_probs[:, 0, 0] = 1.0
    target_probs[:, 1, 0] = 1.0
    accepted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    emitted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for call in range(1, 4):
        output, accepted, emitted = flashinfer.chain_speculative_sampling(
            draft_probs,
            draft_token_ids,
            target_probs,
            accepted,
            emitted,
            seed=12345,
            offset=17,
        )
        # The target agrees with the draft, so every call accepts its one token
        # and emits it, adding exactly one to each counter.
        assert torch.all(output[:, 0] == 0)
        assert torch.all(accepted == call)
        assert torch.all(emitted == call)
