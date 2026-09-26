import pytest
import torch

from flashinfer.parallel_attention import split_varlen_input


@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_split_varlen_input_handles_sequences_shorter_than_world_size(
    tensor_layout,
):
    """Keep trailing ranks empty and pad both supported tensor layouts."""
    values = torch.arange(7, dtype=torch.float32)
    tensor = (
        values.reshape(1, 7, 1) if tensor_layout == "HND" else values.reshape(7, 1, 1)
    )
    chunk_dim = 1 if tensor_layout == "HND" else 0

    expected = (
        [0, 2, 3],
        [1, 4, 5],
        [6, 0, 0],
        [0, 0, 0],
    )
    for rank, expected_values in enumerate(expected):
        local = split_varlen_input(
            tensor,
            seq_len_list=[2, 5],
            world_size=4,
            rank=rank,
            tensor_layout=tensor_layout,
        )

        assert local.shape[chunk_dim] == 3
        assert local.flatten().tolist() == expected_values


def test_split_varlen_input_preserves_even_split_behavior():
    """Retain the existing partition for evenly divisible sequences."""
    tensor = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)

    chunks = [
        split_varlen_input(tensor, [4, 8], world_size=4, rank=rank).flatten().tolist()
        for rank in range(4)
    ]

    assert chunks == [
        [0, 4, 5],
        [1, 6, 7],
        [2, 8, 9],
        [3, 10, 11],
    ]
