import pytest

from b12x.moe._shared.kernels.w4a16.host import (
    route_pack_warmup_token_counts,
)


@pytest.mark.parametrize(
    ("capacity", "expected"),
    [
        (1, (1,)),
        (5, (1, 2, 3, 4, 5)),
        (6, (1, 2, 3, 4, 5, 6)),
        (32, (1, 2, 3, 4, 5, 8, 9, 16, 17, 32)),
        (
            3072,
            (
                1,
                2,
                3,
                4,
                5,
                8,
                9,
                16,
                17,
                32,
                33,
                64,
                65,
                128,
                129,
                256,
                257,
                512,
                513,
                1024,
                1025,
                2048,
                2049,
                3072,
            ),
        ),
    ],
)
def test_route_pack_warmup_covers_capacity_buckets(
    capacity: int, expected: tuple[int, ...]
) -> None:
    assert route_pack_warmup_token_counts(capacity) == expected


def test_route_pack_warmup_rejects_empty_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        route_pack_warmup_token_counts(0)
