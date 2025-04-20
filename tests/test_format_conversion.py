import numpy as np
import pytest
import torch

from flashinfer import pad_ragged_tensor_to_multiple_of


def pad_ragged_tensor_to_multiple_of_pytorch_fill_zeros(
    ragged_tensor, indptr, multiple_of
):
    """PyTorch baseline implementation of pad_ragged_tensor_to_multiple_of."""
    n_rows = indptr.shape[0] - 1
    dim = ragged_tensor.shape[1]

    # Compute padded lengths for each row
    row_lengths = indptr[1:] - indptr[:-1]
    padded_lengths = ((row_lengths + multiple_of - 1) // multiple_of) * multiple_of

    # Compute padded indptr
    padded_indptr = torch.zeros_like(indptr)
    padded_indptr[1:] = torch.cumsum(padded_lengths, dim=0)

    # Allocate padded tensor
    total_padded_length = padded_indptr[-1].item()
    padded_ragged_tensor = torch.zeros(
        (total_padded_length, dim),
        dtype=ragged_tensor.dtype,
        device=ragged_tensor.device,
    )

    # Copy data from original tensor to padded tensor
    for i in range(n_rows):
        row_start = indptr[i].item()
        row_end = indptr[i + 1].item()
        row_length = row_end - row_start

        padded_row_start = padded_indptr[i].item()

        # Copy the original data
        padded_ragged_tensor[padded_row_start : padded_row_start + row_length] = (
            ragged_tensor[row_start:row_end]
        )

    return padded_ragged_tensor, padded_indptr


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("n_rows", [1, 2, 5, 10])
@pytest.mark.parametrize("dim", [64, 128, 1024])
@pytest.mark.parametrize("multiple_of", [8, 16, 32, 64, 128])
def test_pad_ragged_tensor_to_multiple_of(dtype, n_rows, dim, multiple_of):
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    # Create random row lengths
    row_lengths = torch.randint(1, 100, (n_rows,), device=device)

    # Create indptr
    indptr = torch.zeros(n_rows + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(row_lengths, dim=0)

    # Create ragged tensor
    nnz = indptr[-1].item()
    ragged_tensor = torch.randn(nnz, dim, dtype=dtype, device=device)

    # Run both implementations
    padded_ragged_tensor, padded_indptr = pad_ragged_tensor_to_multiple_of(
        ragged_tensor, indptr, multiple_of, fill_zeros=True
    )

    padded_ragged_tensor_ref, padded_indptr_ref = (
        pad_ragged_tensor_to_multiple_of_pytorch_fill_zeros(
            ragged_tensor, indptr, multiple_of
        )
    )

    # Check shapes
    assert padded_ragged_tensor.shape == padded_ragged_tensor_ref.shape
    assert padded_indptr.shape == padded_indptr_ref.shape

    # Check indptr values
    assert torch.allclose(padded_indptr, padded_indptr_ref)

    # Check tensor values
    assert torch.allclose(
        padded_ragged_tensor, padded_ragged_tensor_ref, rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    test_pad_ragged_tensor_to_multiple_of(
        dtype=torch.float16, n_rows=100, dim=1024, multiple_of=128
    )
