# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Explicit cuDNN Top-K with a changed-input CUDA graph replay."""

import torch
import flashinfer


def main():
    scores = torch.randn((8, 8192), dtype=torch.bfloat16, device="cuda")
    lengths = torch.full((8,), 8192, dtype=torch.int32, device="cuda")
    indices = torch.empty((8, 512), dtype=torch.int32, device="cuda")

    def execute():
        return flashinfer.top_k_varlen(
            scores, lengths, 512, out_indices=indices, backend="cudnn"
        )

    execute()  # prepare outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    scores.normal_()
    lengths.fill_(4096)
    graph.replay()
    picked = indices.long()
    assert bool(((picked >= 0) & (picked < 4096)).all())
    ordered = picked.sort(dim=1).values
    assert bool((ordered[:, 1:] != ordered[:, :-1]).all())
    actual = scores.gather(1, picked).sort(dim=1).values
    expected = scores[:, :4096].topk(512, dim=1).values.sort(dim=1).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("cuDNN changed-input graph replay passed")


if __name__ == "__main__":
    main()
