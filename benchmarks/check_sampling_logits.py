"""Serialize public-API outputs for comparison with the unmodified revision."""

import json
import sys

import torch

import flashinfer

torch.manual_seed(73915)
records = []
for vocab in (33, 111, 1023, 1024, 1025, 4095, 4096, 4097, 8193, 32000, 128256, 151936):
    random_logits = torch.randn(17, vocab, device="cuda")
    for kind in ("random", "ties", "masked_ties", "all_masked"):
        logits = random_logits.clone()
        if kind == "ties":
            logits.fill_(1e30)
        elif kind == "masked_ties":
            logits.fill_(-torch.inf)
            logits[:, ::7] = 1e30
        elif kind == "all_masked":
            logits.fill_(-torch.inf)
        for deterministic in (False, True):
            for mode in ("direct", "int32", "int64"):
                indices = (
                    None
                    if mode == "direct"
                    else torch.tensor(
                        [16, 3, 3, 1, 0, 9, 13, 2],
                        device="cuda",
                        dtype=torch.int32 if mode == "int32" else torch.int64,
                    )
                )
                for seed_mode in ("scalar", "tensor"):
                    seed, offset = 12345, 17
                    if seed_mode == "tensor":
                        seed = torch.tensor([seed], device="cuda", dtype=torch.uint64)
                        offset = torch.tensor(
                            [offset], device="cuda", dtype=torch.uint64
                        )
                    output = flashinfer.sampling_from_logits(
                        logits,
                        indices=indices,
                        deterministic=deterministic,
                        seed=seed,
                        offset=offset,
                    )
                    records.append(
                        {
                            "vocab": vocab,
                            "kind": kind,
                            "deterministic": deterministic,
                            "indices": mode,
                            "seed_mode": seed_mode,
                            "tokens": output.tolist(),
                        }
                    )
with open(sys.argv[1], "w") as f:
    json.dump(records, f)
print(
    len(records),
    "seeded cases",
    sum(len(r["tokens"]) for r in records),
    "tokens",
    flush=True,
)
