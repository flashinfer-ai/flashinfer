"""Serialize chain speculative sampling outputs for comparison across revisions.

The residual block sum is accumulated in a different order by the deferred
reduction, so outputs are compared as a divergence rate rather than asserted
bit-exact. Run unchanged on each revision and diff the resulting files.
"""

import json
import sys

import torch

import flashinfer

torch.manual_seed(73915)
records = []
for vocab in (33, 111, 1023, 1024, 1025, 4095, 4096, 4097, 8193, 32000, 128256):
    for num_speculate_tokens in (1, 3, 5):
        batch_size = 17
        pre_norm_draft = torch.rand(
            batch_size, num_speculate_tokens, vocab, device="cuda"
        )
        draft_probs = pre_norm_draft / pre_norm_draft.sum(dim=-1, keepdim=True)
        draft_token_ids = torch.randint(
            vocab,
            (batch_size, num_speculate_tokens),
            device="cuda",
            dtype=torch.int32,
        )
        pre_norm_target = torch.rand(
            batch_size, num_speculate_tokens + 1, vocab, device="cuda"
        )
        target_probs = pre_norm_target / pre_norm_target.sum(dim=-1, keepdim=True)
        for deterministic in (False, True):
            for seed_mode in ("scalar", "tensor"):
                seed, offset = 12345, 17
                if seed_mode == "tensor":
                    seed = torch.tensor([seed], device="cuda", dtype=torch.uint64)
                    offset = torch.tensor([offset], device="cuda", dtype=torch.uint64)
                accepted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
                emitted = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
                output, accepted, emitted = flashinfer.chain_speculative_sampling(
                    draft_probs,
                    draft_token_ids,
                    target_probs,
                    accepted,
                    emitted,
                    deterministic=deterministic,
                    seed=seed,
                    offset=offset,
                )
                records.append(
                    {
                        "vocab": vocab,
                        "num_speculate_tokens": num_speculate_tokens,
                        "deterministic": deterministic,
                        "seed_mode": seed_mode,
                        "tokens": output.tolist(),
                        "accepted": accepted.tolist(),
                        "emitted": emitted.tolist(),
                    }
                )
with open(sys.argv[1], "w") as f:
    json.dump(records, f)
print(
    len(records),
    "seeded cases",
    sum(len(r["tokens"]) * len(r["tokens"][0]) for r in records),
    "tokens",
    flush=True,
)
