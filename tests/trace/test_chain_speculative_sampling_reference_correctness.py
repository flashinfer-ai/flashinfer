"""Reference correctness test for the chain_speculative_sampling trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(batch_size=3, num_speculative=4, vocab_size=128),
        dict(batch_size=2, num_speculative=3, vocab_size=96),
    ],
)
def test_chain_speculative_sampling_reference_correctness(shape_kwargs):
    """Chain speculative sampling kernel vs reference.

    Uses one-hot draft+target distributions where target matches draft on
    all draft positions (→ all draft tokens accepted) and picks a fixed
    token for the final bonus slot, so kernel and argmax-reference agree.
    """
    import flashinfer
    from flashinfer.trace.templates.sampling import chain_speculative_sampling_trace

    inputs = chain_speculative_sampling_trace.init(**shape_kwargs)
    draft_ids = inputs["draft_token_ids"]
    B, S = draft_ids.shape
    V = inputs["draft_probs"].shape[-1]
    bonus_ids = torch.randint(0, V, (B,), dtype=torch.int64, device="cuda")
    # One-hot draft probs: shape [B, S, V]
    draft_probs = inputs["draft_probs"]
    draft_probs.zero_()
    draft_probs.scatter_(2, draft_ids.to(torch.int64).unsqueeze(-1), 1.0)
    # One-hot target probs: shape [B, S+1, V]; matches draft for first S slots.
    target_ids = torch.cat([draft_ids.to(torch.int64), bonus_ids.unsqueeze(-1)], dim=1)
    target_probs = inputs["target_probs"]
    target_probs.zero_()
    target_probs.scatter_(2, target_ids.unsqueeze(-1), 1.0)
    accepted_num = torch.zeros(B, dtype=torch.int32, device="cuda")
    emitted_num = torch.zeros(B, dtype=torch.int32, device="cuda")
    api_out, _, _ = flashinfer.chain_speculative_sampling(
        draft_probs,
        draft_ids,
        target_probs,
        accepted_num,
        emitted_num,
        deterministic=True,
    )
    ref_out = chain_speculative_sampling_trace.reference(
        draft_probs, draft_ids, target_probs
    )
    _close(api_out.to(torch.int32), ref_out, atol=0.0, rtol=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
