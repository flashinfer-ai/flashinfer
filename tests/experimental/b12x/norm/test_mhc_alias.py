"""Disjoint mHC arena views are valid; overlapping byte ranges remain invalid."""

import pytest
import torch

from b12x.norm.mhc._alias import overlaps
from b12x.norm.mhc._impl import _lagged_mix_views, _validate_lagged_y_output


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_disjoint_coefficients_share_storage(device):
    storage = torch.empty(256, device=device)
    incoming, predicted = storage[:8].view(2, 4), storage[8:16].view(2, 4)
    _lagged_mix_views(
        incoming,
        predicted,
        tokens=2,
        device=torch.device(device),
        inputs=(storage[16:32],),
        outputs=(storage[32:64],),
    )
    _validate_lagged_y_output(storage[64:128], (storage[:64], storage[128:]))
    with pytest.raises(ValueError, match="must not alias"):
        _lagged_mix_views(
            incoming,
            storage[4:12].view(2, 4),
            tokens=2,
            device=torch.device(device),
            inputs=(),
            outputs=(),
        )
    with pytest.raises(ValueError, match="must not alias"):
        _validate_lagged_y_output(storage[64:128], (storage[127:],))


def test_byte_ranges_cover_dtype_views_strides_and_empty_tensors():
    storage = torch.empty(128)
    assert not overlaps(storage[:8], storage[8:16])
    assert overlaps(storage[:8], storage[7:16])
    assert overlaps(storage[:8], storage.view(torch.uint8)[31:33])
    assert not overlaps(storage[:8], storage.view(torch.uint8)[32:64])
    assert overlaps(storage[:32:2], storage[30:31])
    assert not overlaps(storage[:32:2], storage[32:])
    assert not overlaps(storage[:0], storage)
    assert not overlaps(storage, torch.empty_like(storage))
