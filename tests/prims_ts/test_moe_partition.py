# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

import flashinfer.prims_ts.moe.partition as partition_module
from flashinfer.prims_ts.moe.partition import (
    CudaForkJoin,
    MoePartitionResources,
    PreparedWeightLayout,
    prepared_weight_layouts,
    require_partition_pair,
    shard_prepared_moe_weights,
    validate_partition_shape,
    validate_partition_tactic,
)
from flashinfer.utils import get_compute_capability


def _pack_mma_scale_factors(
    logical: torch.Tensor,
    *,
    logical_k: int,
    layout: str,
    k_group_size: int,
) -> torch.Tensor:
    """Pure-torch reference for flattened R128c4 MMA scale storage."""
    assert layout == "r128c4"
    num_experts, logical_rows, sf_cols = map(int, logical.shape)
    assert sf_cols * k_group_size == logical_k
    assert logical_rows % 128 == 0 and sf_cols % 4 == 0
    return (
        logical.reshape(num_experts, logical_rows // 128, 4, 32, sf_cols // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(-1)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] < 10,
    reason="block_scale_interleave requires an SM100+ GPU",
)
def test_pack_mma_scale_factors_matches_block_scale_interleave():
    from flashinfer import block_scale_interleave, shuffle_matrix_a, shuffle_matrix_sf_a

    experts, rows, sf_cols = 2, 256, 8
    logical = torch.randint(
        0, 256, (experts, rows, sf_cols), dtype=torch.uint8, device="cuda"
    )
    reference = _pack_mma_scale_factors(
        logical, logical_k=sf_cols * 16, layout="r128c4", k_group_size=16
    )
    torch.testing.assert_close(block_scale_interleave(logical), reference)

    shuffled = shuffle_matrix_sf_a(logical[0], 128)
    reference = _pack_mma_scale_factors(
        shuffle_matrix_a(logical[0], 128)[None],
        logical_k=sf_cols * 16,
        layout="r128c4",
        k_group_size=16,
    )
    torch.testing.assert_close(shuffled.reshape(-1), reference)


def _shuffle_rows(tensor: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor:
    block = 32 if epilogue_tile_m % 128 == 0 else 16
    groups = block // 8
    out = torch.empty_like(tensor)
    for row in range(tensor.shape[-2]):
        row_in_block = row % block
        destination = (
            row // block * block + row_in_block % groups * 8 + row_in_block // groups
        )
        out[..., destination, :] = tensor[..., row, :]
    return out


def _prepare_fc1(raw: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor:
    half = raw.shape[-2] // 2
    reordered = torch.empty_like(raw)
    reordered[..., 0::2, :] = raw[..., :half, :]
    reordered[..., 1::2, :] = raw[..., half:, :]
    return _shuffle_rows(reordered, epilogue_tile_m)


def _prepare_fc2(raw: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor:
    return _shuffle_rows(raw, epilogue_tile_m)


def _raw_fc1_shard(raw: torch.Tensor, partition_id: int) -> torch.Tensor:
    intermediate = raw.shape[-2] // 2
    half = intermediate // 2
    begin = partition_id * half
    end = begin + half
    return torch.cat(
        (
            raw[..., begin:end, :],
            raw[..., intermediate + begin : intermediate + end, :],
        ),
        dim=-2,
    ).contiguous()


def _to_block_major_k(
    tensor: torch.Tensor, *, storage_bits: int, block_bytes: int
) -> torch.Tensor:
    logical_k_per_block = block_bytes * 8 // storage_bits
    storage_per_block = block_bytes // tensor.element_size()
    assert (
        tensor.shape[-1]
        * tensor.element_size()
        * 8
        // storage_bits
        % (logical_k_per_block)
        == 0
    )
    experts, rows, _ = tensor.shape
    return (
        tensor.view(experts, rows, -1, storage_per_block)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


@pytest.mark.parametrize("epilogue_tile_m", [64, 128])
@pytest.mark.parametrize("layout_kind", ["major_k", "block_major_k"])
@pytest.mark.parametrize(
    ("dtype", "storage_bits"),
    [
        pytest.param(torch.bfloat16, 16, id="bf16"),
        pytest.param(torch.uint8, 8, id="fp8-storage"),
        pytest.param(torch.uint8, 4, id="fp4-packed-storage"),
    ],
)
def test_row_shards_commute_with_gated_reorder_and_shuffle(
    epilogue_tile_m, layout_kind, dtype, storage_bits
):
    experts, hidden, intermediate = 2, 256, 128
    element_bits = torch.empty((), dtype=dtype).element_size() * 8
    fc1_k_storage = hidden * storage_bits // element_bits
    fc2_k_storage = intermediate * storage_bits // element_bits
    fc1_raw = torch.randint(
        0, 127, (experts, 2 * intermediate, fc1_k_storage), dtype=torch.int32
    ).to(dtype)
    fc2_raw = torch.randint(
        0, 127, (experts, hidden, fc2_k_storage), dtype=torch.int32
    ).to(dtype)
    block_bytes = 16
    fc1_prepared = _prepare_fc1(fc1_raw, epilogue_tile_m)
    fc2_prepared = _prepare_fc2(fc2_raw, epilogue_tile_m)
    if layout_kind == "block_major_k":
        fc1_prepared = _to_block_major_k(
            fc1_prepared, storage_bits=storage_bits, block_bytes=block_bytes
        )
        fc2_prepared = _to_block_major_k(
            fc2_prepared, storage_bits=storage_bits, block_bytes=block_bytes
        )
    layout = PreparedWeightLayout(
        layout_kind,
        storage_bits=storage_bits,
        block_major_k_bytes=block_bytes if layout_kind == "block_major_k" else None,
    )

    shards = shard_prepared_moe_weights(
        fc1_prepared,
        fc2_prepared,
        strategy="tp_n_tp_n",
        hidden_size=hidden,
        intermediate_size=intermediate,
        fc1_layout=layout,
        fc2_layout=layout,
        fc1_epilogue_tile_m=epilogue_tile_m,
        fc2_epilogue_tile_m=epilogue_tile_m,
    )

    for partition_id, (fc1_shard, fc2_shard) in enumerate(
        zip(shards.fc1, shards.fc2, strict=True)
    ):
        expected_fc1 = _prepare_fc1(
            _raw_fc1_shard(fc1_raw, partition_id), epilogue_tile_m
        )
        expected_fc2 = _prepare_fc2(
            fc2_raw[
                :,
                partition_id * (hidden // 2) : (partition_id + 1) * (hidden // 2),
                :,
            ].contiguous(),
            epilogue_tile_m,
        )
        if layout_kind == "block_major_k":
            expected_fc1 = _to_block_major_k(
                expected_fc1, storage_bits=storage_bits, block_bytes=block_bytes
            )
            expected_fc2 = _to_block_major_k(
                expected_fc2, storage_bits=storage_bits, block_bytes=block_bytes
            )
        torch.testing.assert_close(fc1_shard, expected_fc1)
        torch.testing.assert_close(fc2_shard, expected_fc2)
        assert fc1_shard.is_contiguous()
        assert fc2_shard.is_contiguous()


@pytest.mark.parametrize("layout_kind", ["major_k", "block_major_k"])
def test_packed_fp4_fc2_k_shards_commute_with_preparation(layout_kind):
    experts, hidden, intermediate = 2, 256, 256
    block_bytes = 16
    fc1_raw = torch.randint(
        0, 256, (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8
    )
    fc2_raw = torch.randint(
        0, 256, (experts, hidden, intermediate // 2), dtype=torch.uint8
    )
    fc2_scale_raw = torch.randint(
        0, 256, (experts, hidden, intermediate // 16), dtype=torch.uint8
    )
    fc1_prepared = _prepare_fc1(fc1_raw, 128)
    fc2_major = _prepare_fc2(fc2_raw, 128)
    fc2_scale = _pack_mma_scale_factors(
        _prepare_fc2(fc2_scale_raw, 128),
        logical_k=intermediate,
        layout="r128c4",
        k_group_size=16,
    )
    if layout_kind == "block_major_k":
        fc1_prepared = _to_block_major_k(
            fc1_prepared, storage_bits=4, block_bytes=block_bytes
        )
        fc2_prepared = _to_block_major_k(
            fc2_major, storage_bits=4, block_bytes=block_bytes
        )
        layout = PreparedWeightLayout(
            "block_major_k",
            storage_bits=4,
            block_major_k_bytes=block_bytes,
            scale_layout="r128c4",
            scale_k_group_size=16,
        )
    else:
        fc2_prepared = fc2_major
        layout = PreparedWeightLayout(
            "major_k",
            storage_bits=4,
            scale_layout="r128c4",
            scale_k_group_size=16,
        )
    fc1_layout = PreparedWeightLayout(
        layout_kind,
        storage_bits=4,
        block_major_k_bytes=block_bytes if layout_kind == "block_major_k" else None,
    )

    shards = shard_prepared_moe_weights(
        fc1_prepared,
        fc2_prepared,
        strategy="tp_n_tp_k",
        hidden_size=hidden,
        intermediate_size=intermediate,
        fc1_layout=fc1_layout,
        fc2_layout=layout,
        fc2_scale=fc2_scale,
    )

    bytes_per_k_shard = intermediate // 4
    assert shards.fc2_scale is not None
    for partition_id, (fc2_shard, fc2_scale_shard) in enumerate(
        zip(shards.fc2, shards.fc2_scale, strict=True)
    ):
        raw_k_shard = fc2_raw[
            ...,
            partition_id * bytes_per_k_shard : (partition_id + 1) * bytes_per_k_shard,
        ].contiguous()
        expected = _prepare_fc2(raw_k_shard, 128)
        if layout_kind == "block_major_k":
            expected = _to_block_major_k(
                expected, storage_bits=4, block_bytes=block_bytes
            )
        torch.testing.assert_close(fc2_shard, expected)
        expected_scale = _pack_mma_scale_factors(
            _prepare_fc2(
                fc2_scale_raw[
                    ...,
                    partition_id * 8 : (partition_id + 1) * 8,
                ].contiguous(),
                128,
            ),
            logical_k=intermediate // 2,
            layout="r128c4",
            k_group_size=16,
        )
        torch.testing.assert_close(fc2_scale_shard, expected_scale)
        assert fc2_shard.is_contiguous()


def test_r128c4_scale_row_shards_commute_with_preparation():
    experts, hidden, intermediate = 2, 256, 128
    k_group_size = 16
    fc1_raw = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // k_group_size),
        dtype=torch.uint8,
    )
    fc1_prepared_logical = _prepare_fc1(fc1_raw, 128)
    fc1_scale = _pack_mma_scale_factors(
        fc1_prepared_logical,
        logical_k=hidden,
        layout="r128c4",
        k_group_size=k_group_size,
    )
    weight_layout = PreparedWeightLayout(
        "major_k",
        storage_bits=16,
        scale_layout="r128c4",
        scale_k_group_size=k_group_size,
    )
    fc1_weight = torch.empty((experts, 2 * intermediate, hidden), dtype=torch.bfloat16)
    fc2_weight = torch.empty((experts, hidden, intermediate), dtype=torch.bfloat16)

    shards = shard_prepared_moe_weights(
        fc1_weight,
        fc2_weight,
        strategy="tp_n_tp_n",
        hidden_size=hidden,
        intermediate_size=intermediate,
        fc1_layout=weight_layout,
        fc2_layout=PreparedWeightLayout("major_k", storage_bits=16),
        fc1_scale=fc1_scale,
    )

    assert shards.fc1_scale is not None
    for partition_id, scale_shard in enumerate(shards.fc1_scale):
        logical_shard = _prepare_fc1(_raw_fc1_shard(fc1_raw, partition_id), 128)
        expected = _pack_mma_scale_factors(
            logical_shard,
            logical_k=hidden,
            layout="r128c4",
            k_group_size=k_group_size,
        )
        torch.testing.assert_close(scale_shard, expected)


def test_r128c4_rejects_partial_scale_row_tile():
    experts, hidden, intermediate = 1, 256, 64
    logical_scale = torch.zeros(
        (experts, 2 * intermediate, hidden // 16), dtype=torch.uint8
    )
    scale = _pack_mma_scale_factors(
        logical_scale,
        logical_k=hidden,
        layout="r128c4",
        k_group_size=16,
    )
    scale_layout = PreparedWeightLayout(
        "major_k",
        storage_bits=16,
        scale_layout="r128c4",
        scale_k_group_size=16,
    )

    with pytest.raises(ValueError, match="requires 128-row alignment"):
        shard_prepared_moe_weights(
            torch.empty((experts, 2 * intermediate, hidden), dtype=torch.bfloat16),
            torch.empty((experts, hidden, intermediate), dtype=torch.bfloat16),
            strategy="tp_n_tp_n",
            hidden_size=hidden,
            intermediate_size=intermediate,
            fc1_layout=scale_layout,
            fc2_layout=PreparedWeightLayout("major_k", storage_bits=16),
            fc1_scale=scale,
        )


def test_r128c4_rejects_partial_scale_k_tile():
    experts, hidden, intermediate = 1, 256, 128
    fc1_scale = _pack_mma_scale_factors(
        torch.zeros(
            (experts, 2 * intermediate, hidden // 32),
            dtype=torch.uint8,
        ),
        logical_k=hidden,
        layout="r128c4",
        k_group_size=32,
    )
    fc2_scale = _pack_mma_scale_factors(
        torch.zeros(
            (experts, hidden, intermediate // 32),
            dtype=torch.uint8,
        ),
        logical_k=intermediate,
        layout="r128c4",
        k_group_size=32,
    )
    scale_layout = PreparedWeightLayout(
        "major_k",
        storage_bits=8,
        scale_layout="r128c4",
        scale_k_group_size=32,
    )

    with pytest.raises(ValueError, match="four-scale blocks"):
        shard_prepared_moe_weights(
            torch.empty((experts, 2 * intermediate, hidden), dtype=torch.uint8),
            torch.empty((experts, hidden, intermediate), dtype=torch.uint8),
            strategy="tp_n_tp_k",
            hidden_size=hidden,
            intermediate_size=intermediate,
            fc1_layout=scale_layout,
            fc2_layout=scale_layout,
            fc1_scale=fc1_scale,
            fc2_scale=fc2_scale,
        )


def _validate_shape(resources, shards, layout, *, strategy="tp_n_tp_n"):
    validate_partition_shape(
        resources,
        strategy=strategy,
        fc1_weights=shards.fc1,
        fc2_weights=shards.fc2,
        fc1_scales=shards.fc1_scale,
        fc2_scales=shards.fc2_scale,
        fc1_layout=layout,
        fc2_layout=layout,
        num_experts=1,
        hidden_size=256,
        intermediate_size=128,
    )


def test_validate_partition_accepts_materialized_contract():
    layout = PreparedWeightLayout("major_k", storage_bits=16)
    shards = shard_prepared_moe_weights(
        torch.empty((1, 256, 256), dtype=torch.bfloat16),
        torch.empty((1, 256, 128), dtype=torch.bfloat16),
        strategy="tp_n_tp_n",
        hidden_size=256,
        intermediate_size=128,
        fc1_layout=layout,
        fc2_layout=layout,
    )
    resources = MoePartitionResources(
        [SimpleNamespace(device=torch.device("cpu")) for _ in range(2)]
    )
    _validate_shape(resources, shards, layout)


def test_validate_partition_rejects_non_contiguous_shard():
    layout = PreparedWeightLayout("major_k", storage_bits=16)
    shards = shard_prepared_moe_weights(
        torch.empty((1, 256, 256), dtype=torch.bfloat16),
        torch.empty((1, 256, 128), dtype=torch.bfloat16),
        strategy="tp_n_tp_n",
        hidden_size=256,
        intermediate_size=128,
        fc1_layout=layout,
        fc2_layout=layout,
    )
    resources = MoePartitionResources(
        [SimpleNamespace(device=torch.device("cpu")) for _ in range(2)]
    )
    _validate_shape(resources, shards, layout)

    shard = shards.fc1[0]
    shard.as_strided_(shard.shape, (shard.stride(0), 1, shard.shape[1]))
    assert not shard.is_contiguous()
    with pytest.raises(ValueError, match="must be contiguous"):
        _validate_shape(resources, shards, layout)


def test_validate_partition_tactic_accepts_partial_sf_group():
    common = dict(
        hidden_size=256,
        intermediate_size=128,
        fc1_tile_m=64,
        fc2_tile_m=64,
        fc1_epilogue_tile_m=128,
        fc2_epilogue_tile_m=128,
        fc1_is_swap_ab=True,
        fc2_is_swap_ab=True,
        fc1_partition_width=192,
        fc1_output_sf_block_size=32,
    )
    validate_partition_tactic(strategy="tp_n_tp_n", **common)
    validate_partition_tactic(strategy="tp_n_tp_k", **common)

    with pytest.raises(ValueError, match="SF block width 32"):
        validate_partition_tactic(
            strategy="tp_n_tp_n",
            **{**common, "fc1_partition_width": 48},
        )


def test_validate_partition_tactic_memo_is_scoped_to_geometry():
    common = dict(
        strategy="tp_n_tp_k",
        hidden_size=256,
        intermediate_size=128,
        fc1_tile_m=64,
        fc2_tile_m=64,
        fc1_epilogue_tile_m=128,
        fc2_epilogue_tile_m=128,
        fc1_is_swap_ab=True,
        fc2_is_swap_ab=True,
    )

    validate_partition_tactic.cache_clear()
    validate_partition_tactic(**common)
    validate_partition_tactic(**common)

    assert validate_partition_tactic.cache_info().misses == 1
    assert validate_partition_tactic.cache_info().hits == 1


def test_partition_resources_requires_exactly_two_streams():
    with pytest.raises(ValueError, match="exactly two partition streams"):
        MoePartitionResources([SimpleNamespace(device=torch.device("cpu"))])


def test_partition_resources_launch_guard_is_reusable():
    resources = MoePartitionResources(
        [SimpleNamespace(device=torch.device("cpu")) for _ in range(2)]
    )

    with resources.launch_guard():
        pass
    with resources.launch_guard():
        pass


def test_tp_n_tp_k_rejects_k_shard_cutting_block_major_k():
    experts, hidden, intermediate = 1, 256, 128
    block_bytes = 64
    layout = PreparedWeightLayout(
        "block_major_k", storage_bits=4, block_major_k_bytes=block_bytes
    )
    fc1 = torch.empty((experts, 2, 2 * intermediate, block_bytes), dtype=torch.uint8)
    fc2 = torch.empty(
        (experts, 1, hidden, block_bytes),
        dtype=torch.uint8,
    )

    with pytest.raises(ValueError, match="cuts through a physical K block"):
        shard_prepared_moe_weights(
            fc1,
            fc2,
            strategy="tp_n_tp_k",
            hidden_size=hidden,
            intermediate_size=intermediate,
            fc1_layout=layout,
            fc2_layout=layout,
        )


def test_shard_allocator_places_every_partition_payload():
    layout = PreparedWeightLayout("major_k", storage_bits=16)
    calls = []
    storage = {0: [], 1: []}

    def alloc(partition_id, shape, dtype):
        calls.append((partition_id, shape, dtype))
        tensor = torch.empty(shape, dtype=dtype)
        storage[partition_id].append(tensor)
        return tensor

    shards = shard_prepared_moe_weights(
        torch.empty((1, 256, 256), dtype=torch.bfloat16),
        torch.empty((1, 256, 128), dtype=torch.bfloat16),
        strategy="tp_n_tp_n",
        hidden_size=256,
        intermediate_size=128,
        fc1_layout=layout,
        fc2_layout=layout,
        alloc=alloc,
    )
    assert len(calls) == 4
    assert [partition_id for partition_id, _, _ in calls] == [0, 0, 1, 1]
    for partition_id in range(2):
        assert shards.fc1[partition_id] is storage[partition_id][0]
        assert shards.fc2[partition_id] is storage[partition_id][1]


def test_shard_allocator_rejects_misaligned_block_scale_storage():
    experts, hidden, intermediate = 1, 256, 128
    fc1_scale = _pack_mma_scale_factors(
        torch.zeros((experts, 2 * intermediate, hidden // 16), dtype=torch.uint8),
        logical_k=hidden,
        layout="r128c4",
        k_group_size=16,
    )

    def alloc(partition_id, shape, dtype):
        if dtype == torch.uint8:
            return torch.empty(shape[0] + 1, dtype=dtype)[1:]
        return torch.empty(shape, dtype=dtype)

    with pytest.raises(ValueError, match="32-byte aligned block-scale storage"):
        shard_prepared_moe_weights(
            torch.empty((experts, 2 * intermediate, hidden), dtype=torch.bfloat16),
            torch.empty((experts, hidden, intermediate), dtype=torch.bfloat16),
            strategy="tp_n_tp_n",
            hidden_size=hidden,
            intermediate_size=intermediate,
            fc1_layout=PreparedWeightLayout(
                "major_k",
                storage_bits=16,
                scale_layout="r128c4",
                scale_k_group_size=16,
            ),
            fc2_layout=PreparedWeightLayout("major_k", storage_bits=16),
            fc1_scale=fc1_scale,
            alloc=alloc,
        )


def test_shard_without_allocator_never_aliases_source():
    layout = PreparedWeightLayout("major_k", storage_bits=16)
    fc1 = torch.randn((1, 256, 256), dtype=torch.bfloat16)
    fc2 = torch.randn((1, 256, 128), dtype=torch.bfloat16)
    shards = shard_prepared_moe_weights(
        fc1,
        fc2,
        strategy="tp_n_tp_n",
        hidden_size=256,
        intermediate_size=128,
        fc1_layout=layout,
        fc2_layout=layout,
    )
    expected = [shard.clone() for shard in (*shards.fc1, *shards.fc2)]
    fc1.zero_()
    fc2.zero_()
    for shard, reference in zip((*shards.fc1, *shards.fc2), expected, strict=True):
        torch.testing.assert_close(shard, reference)


def test_require_partition_pair():
    require_partition_pair("fc1_scales", None)
    require_partition_pair("fc1_weights", (object(), object()))
    with pytest.raises(ValueError, match="fc1_weights must contain exactly 2"):
        require_partition_pair("fc1_weights", (object(),))


def test_prepared_weight_layout_rejects_fp32_storage():
    with pytest.raises(ValueError, match="storage_bits must be one of 4, 8, or 16"):
        PreparedWeightLayout("major_k", storage_bits=32)
    with pytest.raises(ValueError, match="Unsupported prepared weight dtype"):
        prepared_weight_layouts(
            weight_dtype=torch.float32,
            weight_layout=0,
            use_shuffled_weight=True,
        )


def test_prepared_weight_layouts_derives_physical_contract():
    fc1, fc2 = prepared_weight_layouts(
        weight_dtype=torch.float8_e4m3fn,
        weight_layout=0,
        use_shuffled_weight=True,
        scale_k_group_size=32,
    )
    assert fc1 == fc2
    assert fc1.storage_bits == 8
    assert fc1.weight_dtype == torch.float8_e4m3fn
    assert fc1.scale_layout == "r128c4"


def test_prepared_weight_layouts_requires_per_fc_block_bytes():
    with pytest.raises(ValueError, match="require the prepared FC1 and FC2"):
        prepared_weight_layouts(
            weight_dtype=torch.uint8,
            weight_layout=2,
            use_shuffled_weight=True,
        )

    fc1, fc2 = prepared_weight_layouts(
        weight_dtype=torch.uint8,
        weight_layout=2,
        use_shuffled_weight=True,
        fc1_block_major_k_bytes=64,
        fc2_block_major_k_bytes=128,
    )
    assert fc1.block_major_k_bytes == 64
    assert fc2.block_major_k_bytes == 128


def test_validate_partition_rejects_wrong_weight_dtype():
    layout = PreparedWeightLayout(
        "major_k", storage_bits=16, weight_dtype=torch.bfloat16
    )
    shards = shard_prepared_moe_weights(
        torch.empty((1, 256, 256), dtype=torch.float16),
        torch.empty((1, 256, 128), dtype=torch.float16),
        strategy="tp_n_tp_n",
        hidden_size=256,
        intermediate_size=128,
        fc1_layout=PreparedWeightLayout("major_k", storage_bits=16),
        fc2_layout=PreparedWeightLayout("major_k", storage_bits=16),
    )
    resources = MoePartitionResources(
        [SimpleNamespace(device=torch.device("cpu")) for _ in range(2)]
    )
    with pytest.raises(ValueError, match="must have dtype torch.bfloat16"):
        validate_partition_shape(
            resources,
            strategy="tp_n_tp_n",
            fc1_weights=shards.fc1,
            fc2_weights=shards.fc2,
            fc1_scales=None,
            fc2_scales=None,
            fc1_layout=layout,
            fc2_layout=layout,
            num_experts=1,
            hidden_size=256,
            intermediate_size=128,
        )


def test_validate_partition_rejects_mismatched_shard_dtypes():
    layout = PreparedWeightLayout("major_k", storage_bits=16)
    fc1 = (
        torch.empty((1, 128, 256), dtype=torch.bfloat16),
        torch.empty((1, 128, 256), dtype=torch.float16),
    )
    fc2 = tuple(torch.empty((1, 128, 128), dtype=torch.bfloat16) for _ in range(2))
    resources = MoePartitionResources(
        [SimpleNamespace(device=torch.device("cpu")) for _ in range(2)]
    )
    with pytest.raises(ValueError, match=r"fc1_weights\[1\].*dtype"):
        validate_partition_shape(
            resources,
            strategy="tp_n_tp_n",
            fc1_weights=fc1,
            fc2_weights=fc2,
            fc1_scales=None,
            fc2_scales=None,
            fc1_layout=layout,
            fc2_layout=layout,
            num_experts=1,
            hidden_size=256,
            intermediate_size=128,
        )


def test_cuda_fork_join_records_explicit_stream_dependencies(monkeypatch):
    records = []

    class FakeEvent:
        def record(self, stream):
            records.append(("record", self, stream))

    class FakeStream:
        device = torch.device("cuda:0")

        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            records.append(("wait", self, event))

    monkeypatch.setattr(partition_module.torch.cuda, "Event", FakeEvent)
    streams = [FakeStream("partition0"), FakeStream("partition1")]
    caller = FakeStream("caller")
    fork_join = CudaForkJoin(streams)

    fork_join.fork(caller)
    fork_join.join(caller)
    fork_join.fork(caller)

    fork_event = fork_join._fork_event
    done0, done1 = fork_join._done_events
    assert records == [
        ("record", fork_event, caller),
        ("wait", streams[0], fork_event),
        ("wait", streams[1], fork_event),
        ("record", done0, streams[0]),
        ("wait", caller, done0),
        ("record", done1, streams[1]),
        ("wait", caller, done1),
        ("record", fork_event, caller),
        ("wait", streams[0], fork_event),
        ("wait", streams[1], fork_event),
    ]
