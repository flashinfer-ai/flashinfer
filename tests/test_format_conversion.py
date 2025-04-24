import numpy as np
import pytest
import torch

from flashinfer.triton import build_pos_ids_from_segment_offsets_and_lengths


@pytest.mark.parametrize("n_segments", [1, 9, 19, 99, 999])
@pytest.mark.parametrize("multiple_of", [57, 128, 256])
def test_build_pos_ids_from_segment_offsets_and_lengths(
    n_segments: int,
    multiple_of: int,
):
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    # generate random segment offsets and lengths
    lengths = torch.randint(1, 1024, (n_segments,), device=device)
    packed_segment_offsets = torch.zeros(
        n_segments + 1, dtype=torch.int32, device=device
    )
    packed_segment_offsets[1:] = lengths.cumsum(dim=0)
    lengths_padded = (lengths + multiple_of - 1) // multiple_of * multiple_of
    padded_segment_offsets = torch.zeros(
        n_segments + 1, dtype=torch.int32, device=device
    )
    padded_segment_offsets[1:] = lengths_padded.cumsum(dim=0)

    padded_pos_ids = build_pos_ids_from_segment_offsets_and_lengths(
        padded_segment_offsets, packed_segment_offsets, out=None
    )

    padded_pos_ids_ref = []
    for i in range(n_segments):
        padded_pos_ids_ref.append(
            torch.arange(
                padded_segment_offsets[i],
                padded_segment_offsets[i] + lengths[i],
                dtype=torch.int32,
                device=device,
            )
        )
    padded_pos_ids_ref = torch.cat(padded_pos_ids_ref)

    torch.testing.assert_close(padded_pos_ids, padded_pos_ids_ref)
