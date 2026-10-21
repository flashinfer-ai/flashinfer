import pytest
import torch

from b12x.moe.fused_moe._route_pack_cache import route_pack_prewarm_key


def _key(
    *,
    device_type: str = "cuda",
    device_index: int = 0,
    route_ids_dtype: torch.dtype = torch.int32,
    token_count: int = 3,
    top_k: int = 1,
    packed_route_slots: int = 256,
    route_blocks: int = 4,
    block_size: int = 64,
    num_experts: int = 4,
    mapped: bool = False,
) -> tuple[object, ...]:
    return route_pack_prewarm_key(
        device_type,
        device_index,
        route_ids_dtype,
        token_count,
        top_k,
        packed_route_slots,
        route_blocks,
        block_size,
        num_experts,
        mapped,
    )


def test_route_pack_prewarm_key_includes_each_launch_dimension() -> None:
    baseline = _key()
    assert baseline != _key(device_type="cpu")
    assert baseline != _key(device_index=1)
    assert baseline != _key(route_ids_dtype=torch.int64)
    assert baseline != _key(token_count=4)
    assert baseline != _key(top_k=2)
    assert baseline != _key(packed_route_slots=320)
    assert baseline != _key(route_blocks=5)
    assert baseline != _key(block_size=128)
    assert baseline != _key(num_experts=8)
    assert baseline != _key(mapped=True)


@pytest.mark.parametrize(
    "dimension",
    [
        "token_count",
        "top_k",
        "packed_route_slots",
        "route_blocks",
        "block_size",
        "num_experts",
    ],
)
def test_route_pack_prewarm_rejects_nonpositive_dimensions(dimension: str) -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        _key(**{dimension: 0})
