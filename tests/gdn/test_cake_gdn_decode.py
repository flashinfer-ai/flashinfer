# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest

from flashinfer.jit import cake_gdn as cake_gdn


def _prefill(**overrides):
    params = {
        "arch": "sm_100a",
        "io_dtype": "float16",
        "state_dtype": "float32",
        "num_seqs": 1,
        "total_seq_len": 16384,
        "max_seq_len": 16384,
        "num_q_heads": 2,
        "num_k_heads": 2,
        "num_v_heads": 8,
        "use_initial_state": True,
        "store_final_state": True,
        "checkpoint_every_n_tokens": 0,
        "use_state_indices": False,
        "gates_present": False,
    }
    params.update(overrides)
    return cake_gdn.select_cake_gdn_prefill_variant(**params)


def _decode(**overrides):
    params = {
        "arch": "sm_100a",
        "batch_size": 1,
        "io_dtype": "bfloat16",
        "state_dtype": "float32",
        "head_size": 128,
        "layout": "nontranspose",
        "num_k_heads": 16,
        "num_q_heads": 16,
        "num_v_heads": 32,
        "scale": 128**-0.5,
        "seq_len": 1,
        "use_qk_l2norm": True,
    }
    params.update(overrides)
    return cake_gdn.select_cake_gdn_decode_variant(**params)


def test_manifest_lists_loadable_source_only_variants() -> None:
    # The manifest is the only registry: every variant names one kernel source
    # (shared with the other specializations of the same body), the -D defines
    # that specialize it, and one host shim.  Nothing is pinned by checksum.
    manifest = cake_gdn._manifest()
    root = cake_gdn._source_dir()
    variants = manifest["variants"]
    assert len(variants) > 0
    assert len({record["name"] for record in variants}) == len(variants)
    assert manifest["source_only"] is True
    assert manifest["binary_artifacts"] is False
    assert "sha256" not in json.dumps(manifest)
    architectures = set(manifest["architectures"])
    for header in manifest["cuda_headers"]:
        assert (root / header["path"]).is_file()
    for record in variants:
        (output,) = record["outputs"]
        assert set(output["architectures"]) <= architectures
        assert (root / output["path"]).is_file()
        assert (root / record["host_binding"]["path"]).is_file()
        for key, value in record["defines"].items():
            assert key.isidentifier()
            assert isinstance(value, str) and value
        assert set(record["defines"]) <= set(record["specializations"])


def test_prefill_resolver_selects_dvsplit_full_and_single_chunk() -> None:
    dvsplit = _prefill()
    assert dvsplit.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert "dvsplit_initial_f16io" in dvsplit.variant_name

    full = _prefill(
        arch="sm_103a",
        num_seqs=16,
        total_seq_len=16 * 8192,
        max_seq_len=8192,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
    )
    assert full.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
    assert "dvsplit" not in full.variant_name

    single = _prefill(
        io_dtype="bfloat16",
        num_seqs=4,
        total_seq_len=4 * 64,
        max_seq_len=64,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=False,
        store_final_state=False,
    )
    assert single.route_id == "flashinfer.gdn_prefill.noncp.single_chunk.dvsplit"
    assert "single_chunk" in single.variant_name


def test_prefill_resolver_selects_frozen_dynamic_head_specializations() -> None:
    dynamic_heads = _prefill(
        num_seqs=1,
        total_seq_len=64,
        max_seq_len=64,
        num_q_heads=3,
        num_k_heads=3,
        num_v_heads=3,
        use_initial_state=False,
        store_final_state=True,
    )
    dynamic_group = _prefill(
        num_q_heads=6,
        num_k_heads=2,
        num_v_heads=2,
    )

    assert dynamic_heads.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert dynamic_group.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    heads_record = cake_gdn._kernel_record(dynamic_heads.variant_name)
    group_record = cake_gdn._kernel_record(dynamic_group.variant_name)
    assert heads_record["specializations"]["NUM_O_HEADS_LOG2"] == -1
    assert heads_record["specializations"]["HEAD_GROUP_LOG2"] == 0
    assert group_record["specializations"]["NUM_O_HEADS_LOG2"] == -1
    assert group_record["specializations"]["HEAD_GROUP_LOG2"] == -1


def test_prefill_resolver_selects_sglang_tp4_bf16_indexed_row() -> None:
    route = _prefill(
        arch="sm_103a",
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        num_seqs=5,
        total_seq_len=5 * 64,
        max_seq_len=64,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=True,
        store_final_state=True,
        use_state_indices=True,
    )

    assert route.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    record = cake_gdn._kernel_record(route.variant_name)
    assert record["specializations"] == {
        "ENABLE_CHECKPOINTS": 0,
        "HEAD_GROUP_LOG2": 1,
        "IS_GQA": 0,
        "NUM_O_HEADS_LOG2": 3,
        "SINGLE_CHUNK_NO_STATE": 0,
        "STORE_FINAL_STATE": 1,
        "UNIT_GATES": 0,
        "USE_INITIAL_STATE": 1,
        "USE_STATE_INDICES": 1,
    }


def test_prefill_resolver_selects_exact_sglang_tp4_checkpoint_row() -> None:
    route = _prefill(
        arch="sm_103a",
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        num_seqs=7,
        total_seq_len=421,
        max_seq_len=107,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=64,
        use_state_indices=True,
        seq_lens=(52, 93, 15, 107, 72, 61, 21),
    )

    assert route.route_id == "flashinfer.gdn_prefill.noncp.checkpoints.dvsplit"
    record = cake_gdn._kernel_record(route.variant_name)
    assert record["specializations"] == {
        "ENABLE_CHECKPOINTS": 1,
        "HEAD_GROUP_LOG2": 1,
        "IS_GQA": 0,
        "NUM_O_HEADS_LOG2": 3,
        "SINGLE_CHUNK_NO_STATE": 0,
        "STORE_FINAL_STATE": 1,
        "UNIT_GATES": 0,
        "USE_INITIAL_STATE": 1,
        "USE_STATE_INDICES": 1,
    }


def test_prefill_resolver_fails_closed_for_unpromoted_rows() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="checkpoint route requires the frozen FP16/FP32 packed contract",
    ):
        _prefill(
            io_dtype="bfloat16",
            use_initial_state=False,
            checkpoint_every_n_tokens=64,
        )

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="SGLang TP4 BF16 indexed DV-split",
    ):
        _prefill(
            io_dtype="bfloat16",
            state_dtype="bfloat16",
            num_seqs=7,
            total_seq_len=421,
            max_seq_len=107,
            num_q_heads=4,
            num_k_heads=4,
            num_v_heads=8,
            use_initial_state=True,
            store_final_state=True,
            checkpoint_every_n_tokens=64,
            use_state_indices=True,
        )

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="low-precision state requires BF16 I/O",
    ):
        _prefill(state_dtype="float16")


def test_kernel_loader_fails_closed_for_unsupported_architecture() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="unsupported Cake GDN architecture",
    ):
        cake_gdn.load_cake_gdn_kernel("unused", "sm_90a")  # type: ignore[arg-type]


def test_decode_resolver_selects_all_promoted_physical_routes() -> None:
    small = _decode()
    assert small.route_id.endswith("nontranspose_small")
    assert "nontranspose_fp32_t1_small" in small.variant_name

    large = _decode(arch="sm_103a", batch_size=32)
    assert large.route_id.endswith("nontranspose_large")
    assert "nontranspose_fp32_t1_" in large.variant_name
    assert "small" not in large.variant_name

    pretranspose = _decode(layout="pretranspose")
    assert pretranspose.route_id == "flashinfer.gdn_decode.indexed_fp32_t1_splitv8"
    assert "pretranspose_splitv8" in pretranspose.variant_name


def test_decode_resolver_selects_exact_promoted_fp32_mtp_rows() -> None:
    rows = (
        (
            dict(
                batch_size=1,
                seq_len=2,
                disable_state_update=True,
                cache_steps=2,
            ),
            "indexed_fp32_mtp_t2.inline_tile8_verify_cache",
            "mtp_t2_inline_tile8",
        ),
        (
            dict(batch_size=4, seq_len=4, cache_steps=4),
            "indexed_fp32_mtp_t4.splitv8_update_cache",
            "mtp_t4_splitv8",
        ),
        *(
            (
                dict(batch_size=batch_size, seq_len=4, cache_steps=4),
                "indexed_fp32_mtp_t4.tile64_update_cache",
                "mtp_t4_splitv2_tile64",
            )
            for batch_size in (16, 64)
        ),
    )
    for overrides, route_suffix, variant_fragment in rows:
        route = _decode(
            layout="pretranspose",
            strided_inputs=True,
            cache_intermediate_states=True,
            **overrides,
        )
        assert route.route_id.endswith(route_suffix)
        assert variant_fragment in route.variant_name


def test_decode_resolver_fails_closed_for_unpromoted_fp32_mtp_rows() -> None:
    base = {
        "layout": "pretranspose",
        "strided_inputs": True,
        "cache_intermediate_states": True,
        "seq_len": 4,
        "cache_steps": 4,
    }
    for overrides in (
        {"batch_size": 5},
        {"batch_size": 4, "strided_inputs": False},
        {"batch_size": 4, "cache_intermediate_states": False},
        {"batch_size": 4, "cache_steps": 5},
        {"batch_size": 4, "num_v_heads": 64},
    ):
        with pytest.raises(
            cake_gdn.CakeGDNUnsupportedError,
            match="FP32 MTP decode is limited",
        ):
            _decode(**(base | overrides))


def test_decode_resolver_selects_exact_promoted_bf16_rows() -> None:
    rows = (
        (
            dict(
                batch_size=4,
                seq_len=2,
                num_v_heads=32,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=4,
            ),
            "indexed_bf16_verify_t2.wide32",
        ),
        (
            dict(
                batch_size=8,
                seq_len=3,
                num_v_heads=64,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=3,
            ),
            "indexed_bf16_verify_t3.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_v_heads=64,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=4,
            ),
            "indexed_bf16_verify_t4.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_v_heads=32,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=4,
            ),
            "indexed_bf16_verify_t4.wide32",
        ),
        (
            dict(
                batch_size=8,
                seq_len=2,
                num_v_heads=64,
                strided_inputs=True,
            ),
            "indexed_bf16_update_t2.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_v_heads=64,
                strided_inputs=True,
                cache_intermediate_states=True,
                cache_steps=5,
            ),
            "indexed_bf16_checkpoint_t4.wide64",
        ),
    )
    for overrides, route_suffix in rows:
        route = _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            **overrides,
        )
        assert route.route_id.endswith(route_suffix)
        assert "bf16state_wide128" in route.variant_name

    tp4_rows = (
        (
            dict(batch_size=4, seq_len=1),
            "indexed_bf16_t1.vec8_t16",
            "t1_bf16state_vec8",
        ),
        (
            dict(batch_size=4, seq_len=1, arch="sm_103a"),
            "indexed_bf16_t1.vec8r56_t16",
            "t1_bf16state_vec8r56",
        ),
        *(
            (
                dict(
                    batch_size=batch_size,
                    seq_len=4,
                    disable_state_update=True,
                    cache_intermediate_states=True,
                    cache_steps=4,
                ),
                "indexed_bf16_verify_t4.tile16_fullwarp",
                "t4_bf16state_tile16",
            )
            for batch_size in range(1, 9)
        ),
    )
    for overrides, route_suffix, variant_fragment in tp4_rows:
        route = _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            num_k_heads=4,
            num_q_heads=4,
            num_v_heads=8,
            strided_inputs=True,
            **overrides,
        )
        assert route.route_id.endswith(route_suffix)
        assert variant_fragment in route.variant_name


_QWEN35_BF16_T1_GEOMETRIES = (
    # (H, HV): Qwen3.5-35B-A3B TP1/TP2/TP4, Qwen3.5-397B-A17B TP2/TP4/TP8
    (16, 32),
    (8, 16),
    (4, 8),
    (8, 32),
    (4, 16),
    (2, 8),
)
_SGLANG_GRAPH_BATCHES = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    1000,
)


def test_bf16_t1_route_rule_follows_the_state_head_count() -> None:
    bands = cake_gdn.CAKE_GDN_BF16_T1_ROUTE_BANDS
    bounds = [band[0] for band in bands]
    assert bounds[-1] is None and None not in bounds[:-1]
    assert bounds[:-1] == sorted(bounds[:-1])
    for _, body, tile_v in bands:
        assert body in cake_gdn.CAKE_GDN_BF16_T1_BODIES and tile_v in (16, 32, 64, 128)
    rule = cake_gdn.cake_gdn_bf16_t1_route
    assert set(cake_gdn.CAKE_GDN_BF16_T1_ROUTE_ARCH_BODIES) == {"sm_103a"}
    for arch, overrides in (
        ("sm_100a", {}),
        ("sm_103a", cake_gdn.CAKE_GDN_BF16_T1_ROUTE_ARCH_BODIES["sm_103a"]),
    ):
        previous = 0
        for max_state_heads, body, tile_v in bands[:-1]:
            expected = (overrides.get(max_state_heads, body), tile_v)
            assert expected[0] in cake_gdn.CAKE_GDN_BF16_T1_BODIES
            assert rule(1, previous + 1, arch) == expected
            assert rule(1, max_state_heads, arch) == expected
            previous = max_state_heads
        assert rule(1, previous + 1, arch) == bands[-1][1:]
        assert rule(512, 32, arch) == ("wide", 128)
    assert rule(1, 8, "sm_103a") == ("vec8", 32)
    assert rule(1, 32, "sm_100a") == ("vec8", 16)
    assert rule(1, 32, "sm_103a") == ("vec8r56", 16)
    assert rule(4, 32, "sm_100a") == ("vec8", 16)
    assert rule(4, 32, "sm_103a") == ("vec8r56", 16)
    assert rule(16, 32, "sm_103a") == ("vec8occ", 64)
    assert rule(24, 32, "sm_103a") == ("vec8", 64)
    assert rule(32, 32, "sm_103a") == ("wide", 128)
    assert (
        cake_gdn.cake_gdn_bf16_route_tile_v(
            "flashinfer.gdn_decode.indexed_bf16_t1.vec8_t16"
        )
        == 16
    )
    assert (
        cake_gdn.cake_gdn_bf16_route_tile_v(
            "flashinfer.gdn_decode.indexed_bf16_t1.vec8r56_t16"
        )
        == 16
    )
    assert (
        cake_gdn.cake_gdn_bf16_route_tile_v(
            "flashinfer.gdn_decode.indexed_bf16_t1.vec8occ_t64"
        )
        == 64
    )
    assert (
        cake_gdn.cake_gdn_bf16_route_tile_v(
            "flashinfer.gdn_decode.indexed_bf16_t1.wide128"
        )
        == 128
    )
    assert (
        cake_gdn.cake_gdn_bf16_route_tile_v(
            "flashinfer.gdn_decode.indexed_bf16_verify_t4.tile16_fullwarp"
        )
        == 16
    )
    assert (
        cake_gdn.cake_gdn_bf16_route_tile_v(
            "flashinfer.gdn_decode.indexed_bf16_verify_t4.wide64"
        )
        == 64
    )
    assert (
        cake_gdn.cake_gdn_bf16_route_tile_v(
            "flashinfer.gdn_decode.indexed_bf16_verify_t7.wide32"
        )
        == 32
    )
    with pytest.raises(cake_gdn.CakeGDNUnsupportedError, match="carries no grid tile"):
        cake_gdn.cake_gdn_bf16_route_tile_v(
            "flashinfer.gdn_decode.indexed_fp32_t1_splitv8"
        )


@pytest.mark.parametrize("heads", _QWEN35_BF16_T1_GEOMETRIES)
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
def test_decode_resolver_admits_every_qwen35_bf16_t1_geometry_at_any_batch(
    arch, heads
) -> None:
    num_q_heads, num_v_heads = heads
    for batch_size in _SGLANG_GRAPH_BATCHES:
        for strided_inputs in (True, False):
            route = _decode(
                arch=arch,
                state_dtype="bfloat16",
                layout="pretranspose",
                batch_size=batch_size,
                num_k_heads=num_q_heads,
                num_q_heads=num_q_heads,
                num_v_heads=num_v_heads,
                seq_len=1,
                strided_inputs=strided_inputs,
            )
            body, tile_v = cake_gdn.cake_gdn_bf16_t1_route(
                batch_size, num_v_heads, arch
            )
            if body == "wide":
                assert (
                    route.route_id
                    == f"flashinfer.gdn_decode.indexed_bf16_t1.wide{tile_v}"
                )
                assert "mtp_t4_bf16state_wide128" in route.variant_name
            else:
                assert (
                    route.route_id
                    == f"flashinfer.gdn_decode.indexed_bf16_t1.{body}_t{tile_v}"
                )
                assert f"t1_bf16state_{body}_" in route.variant_name
            assert cake_gdn.cake_gdn_bf16_route_tile_v(route.route_id) == tile_v
            record = cake_gdn._kernel_record(route.variant_name)
            assert record["specializations"]["H"] == num_q_heads
            assert record["specializations"]["HV"] == num_v_heads
            assert record["specializations"]["STRIDED_INPUTS"] == 1
            assert (
                record["specializations"].get(
                    "TILE_V_WIDE", record["specializations"].get("TILE_V")
                )
                == tile_v
            )
            assert record["specializations"].get("T_STEPS", 1) == 1
            assert record["specializations"].get("UPDATE_STATE", 1) == 1
            assert arch in record["architectures"]


def test_decode_resolver_fails_closed_for_unlisted_bf16_t1_geometry_and_controls() -> (
    None
):
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError, match="no exact frozen Cake GDN variant"
    ):
        _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            batch_size=4,
            num_k_heads=16,
            num_q_heads=16,
            num_v_heads=64,
            seq_len=1,
            strided_inputs=True,
        )
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError, match="updates the state and caches nothing"
    ):
        _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            batch_size=4,
            num_v_heads=32,
            seq_len=1,
            strided_inputs=True,
            disable_state_update=True,
        )
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError, match="pretranspose state-pool layout"
    ):
        _decode(
            state_dtype="bfloat16",
            layout="nontranspose",
            batch_size=4,
            num_v_heads=32,
            seq_len=1,
        )


def test_decode_resolver_fails_closed_for_unpromoted_bf16_shape() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="exact promoted verify/update rows",
    ):
        _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            batch_size=9,
            num_k_heads=4,
            num_q_heads=4,
            num_v_heads=8,
            seq_len=4,
            strided_inputs=True,
            disable_state_update=True,
            cache_intermediate_states=True,
            cache_steps=4,
        )


def test_decode_resolver_fails_closed_outside_child_contract() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="requires BF16 I/O and FP32 or BF16 state",
    ):
        _decode(state_dtype="float16")

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="requires in-kernel Q/K L2 normalization",
    ):
        _decode(use_qk_l2norm=False)


def test_architecture_mapping_is_exact() -> None:
    assert cake_gdn.arch_for_compute_capability(10, 0) == "sm_100a"
    assert cake_gdn.arch_for_compute_capability(10, 3) == "sm_103a"
    assert cake_gdn.arch_for_compute_capability(10, 7) == "sm_107a"
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="supports only SM100a/SM103a/SM107a",
    ):
        cake_gdn.arch_for_compute_capability(12, 0)


def test_prefill_resolver_selects_exact_gated_physical_schedules() -> None:
    gatepipe = _prefill(gates_present=True)
    gatepipe_record = cake_gdn._kernel_record(gatepipe.variant_name)
    assert gatepipe.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert "gatepipe4" in gatepipe.variant_name
    assert gatepipe_record["tma_abi"] == "grid_constant"
    assert gatepipe_record["specializations"]["UNIT_GATES"] == 0

    fullgrid_vhold = _prefill(
        num_seqs=8,
        total_seq_len=8 * 128,
        max_seq_len=128,
        gates_present=True,
    )
    fullgrid_record = cake_gdn._kernel_record(fullgrid_vhold.variant_name)
    assert fullgrid_vhold.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert "fullgrid_vhold" in fullgrid_vhold.variant_name
    assert fullgrid_record["tma_abi"] == "grid_constant"
    assert fullgrid_record["specializations"]["UNIT_GATES"] == 0


def test_prefill_resolver_accepts_dynamic_sglang_tp4_checkpoint_batches() -> None:
    common = {
        "arch": "sm_103a",
        "io_dtype": "bfloat16",
        "state_dtype": "bfloat16",
        "num_q_heads": 4,
        "num_k_heads": 4,
        "num_v_heads": 8,
        "use_initial_state": True,
        "store_final_state": True,
        "checkpoint_every_n_tokens": 64,
        "use_state_indices": True,
        "gates_present": True,
    }
    frozen = cake_gdn.select_cake_gdn_prefill_variant(
        **common,
        num_seqs=7,
        total_seq_len=421,
        max_seq_len=107,
        seq_lens=(52, 93, 15, 107, 72, 61, 21),
    )
    live = cake_gdn.select_cake_gdn_prefill_variant(
        **common,
        num_seqs=5,
        total_seq_len=4296,
        max_seq_len=897,
        seq_lens=(849, 835, 862, 897, 853),
    )
    assert (
        frozen.route_id
        == live.route_id
        == "flashinfer.gdn_prefill.noncp.checkpoints.dvsplit"
    )
    assert frozen.variant_name == live.variant_name


def test_prefill_resolver_keeps_sglang_tp4_checkpoint_family_fail_closed() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="checkpoint route requires the frozen",
    ):
        cake_gdn.select_cake_gdn_prefill_variant(
            arch="sm_103a",
            io_dtype="bfloat16",
            state_dtype="bfloat16",
            num_seqs=5,
            total_seq_len=4296,
            max_seq_len=897,
            num_q_heads=4,
            num_k_heads=4,
            num_v_heads=8,
            use_initial_state=True,
            store_final_state=True,
            checkpoint_every_n_tokens=128,
            use_state_indices=True,
            gates_present=True,
            seq_lens=(849, 835, 862, 897, 853),
        )

    for arch, num_seqs in (("sm_100a", 10), ("sm_103a", 11), ("sm_107a", 14)):
        with pytest.raises(
            cake_gdn.CakeGDNUnsupportedError,
            match="indexed DV-split contract",
        ):
            cake_gdn.select_cake_gdn_prefill_variant(
                arch=arch,
                io_dtype="bfloat16",
                state_dtype="bfloat16",
                num_seqs=num_seqs,
                total_seq_len=64 * num_seqs,
                max_seq_len=64,
                num_q_heads=4,
                num_k_heads=4,
                num_v_heads=8,
                use_initial_state=True,
                store_final_state=True,
                checkpoint_every_n_tokens=64,
                use_state_indices=True,
                gates_present=True,
                seq_lens=(64,) * num_seqs,
            )


def test_nvcc_identity_captures_resolved_path_and_exact_version_output(
    tmp_path, monkeypatch
) -> None:
    cuda_root = tmp_path / "cuda"
    nvcc = cuda_root / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("compiler", encoding="utf-8")
    version_output = "nvcc: NVIDIA (R) Cuda compiler driver\nBuild exact-output\n"
    calls = []

    class Result:
        returncode = 0
        stdout = version_output

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr(cake_gdn, "get_cuda_path", lambda: str(cuda_root))
    monkeypatch.setattr(cake_gdn.subprocess, "run", fake_run)

    observed_nvcc, observed_version = cake_gdn._nvcc_identity()

    assert observed_nvcc == nvcc.resolve()
    assert observed_version == version_output
    assert calls == [
        (
            [str(nvcc.resolve()), "--version"],
            {
                "stdout": cake_gdn.subprocess.PIPE,
                "stderr": cake_gdn.subprocess.STDOUT,
                "text": True,
                "check": False,
            },
        )
    ]


def test_compile_cache_digest_isolated_by_nvcc_identity(tmp_path) -> None:
    common = {
        "arch": "sm_100a",
        "cuda_sha256": "cuda-source",
        "header_sha256s": ("header-a", "header-b"),
        "compile_options": ("--use_fast_math",),
    }
    baseline = cake_gdn._compile_cache_digest(
        **common,
        nvcc=tmp_path / "cuda-a" / "bin" / "nvcc",
        nvcc_version="release 12.9\nBuild A\n",
    )
    changed_path = cake_gdn._compile_cache_digest(
        **common,
        nvcc=tmp_path / "cuda-b" / "bin" / "nvcc",
        nvcc_version="release 12.9\nBuild A\n",
    )
    changed_version = cake_gdn._compile_cache_digest(
        **common,
        nvcc=tmp_path / "cuda-a" / "bin" / "nvcc",
        nvcc_version="release 12.9\nBuild B\n",
    )

    assert len(baseline) == 64
    assert len({baseline, changed_path, changed_version}) == 3
    assert len({baseline[:16], changed_path[:16], changed_version[:16]}) == 3


def test_compile_cubin_adds_manifest_header_include_paths(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "gdn" / "cake" / "cuda" / "kernel.cu"
    source.parent.mkdir(parents=True)
    source.write_text('#include "cake_gdn_common.cuh"\n', encoding="utf-8")
    include_path = tmp_path / "gdn"
    (include_path / "cake_gdn_common.cuh").write_text("// header\n", encoding="utf-8")
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("compiler", encoding="utf-8")
    calls = []

    class Result:
        returncode = 0
        stdout = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b"cubin")
        return Result()

    monkeypatch.setattr(cake_gdn.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setattr(cake_gdn.subprocess, "run", fake_run)

    cubin = cake_gdn._compile_cubin(
        source,
        arch="sm_100a",
        digest="digest",
        compile_options=("--use_fast_math",),
        include_paths=(include_path,),
        nvcc=nvcc,
    )

    assert cubin == b"cubin"
    assert calls[0][0] == [
        str(nvcc),
        "--cubin",
        "--std=c++17",
        "-O3",
        "--gpu-architecture=sm_100a",
        f"-I{include_path}",
        "--use_fast_math",
        *cake_gdn.get_nvcc_parallelism_flags(),
        str(source),
        "-o",
        calls[0][0][-1],
    ]
