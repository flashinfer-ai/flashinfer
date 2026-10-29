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

import hashlib
import json
import re
from pathlib import Path

import pytest

from flashinfer.jit import cake_gdn as cake_gdn

# The -D knobs a record passes when it shares its body with a schedule twin
# (its kernel symbol and the constant its deleted clone body baked in); the
# resolver key (``specializations``) does not carry them.
_BODY_SHARING_DEFINES = {
    "CAKE_GDN_KERNEL_SYMBOL",
    "LAUNCH_MIN_BLOCKS",
    "T_STEPS",
    "CAKE_GDN_METADATA_I64",
    "CAKE_GDN_FP8_X2",
}


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


@pytest.mark.cpu_only
def test_manifest_lists_loadable_source_only_variants() -> None:
    # The manifest is the only registry: every variant names one kernel source
    # (shared with the other specializations of the same body), the -D defines
    # that specialize it, and its host-shim template.  Nothing is pinned by
    # checksum.
    manifest = cake_gdn._manifest()
    root = cake_gdn._source_dir()
    variants = manifest["variants"]
    assert len(variants) > 0
    assert len({record["name"] for record in variants}) == len(variants)
    assert manifest["source_only"] is True
    assert manifest["binary_artifacts"] is False
    assert "sha256" not in json.dumps(manifest)
    architectures = set(manifest["architectures"])
    # Every shared header's sha256 enters every variant's compile-cache digest,
    # so the manifest lists only headers that a listed kernel body includes.
    header_names = {Path(header["path"]).name for header in manifest["cuda_headers"]}
    for header in manifest["cuda_headers"]:
        assert (root / header["path"]).is_file()
    included_headers: set[str] = set()
    for record in variants:
        # One body per architecture: a kernel whose lowering differs on one
        # architecture (the sm_107a F64-free q scale) carries that architecture
        # in its own output; the outputs partition the record's architectures.
        served: list[str] = []
        for output in record["outputs"]:
            assert output["architectures"]
            assert set(output["architectures"]) <= architectures
            assert (root / output["path"]).is_file()
            source = (root / output["path"]).read_text(encoding="utf-8")
            included_headers.update(
                name for name in header_names if f'#include "{name}"' in source
            )
            served.extend(output["architectures"])
        assert sorted(served) == sorted(record["architectures"])
        assert (root / record["host_binding"]["template"]).is_file()
        # A record spells out ``defines`` only where its -D flags are not its
        # specializations (see test_manifest_defines_default_to_specializations).
        for key, value in record.get("defines", {}).items():
            assert key.isidentifier()
            assert isinstance(value, str) and value
        # A record sharing its body with a schedule twin passes the knobs
        # its own text used to bake in on the compile line; those are not
        # specializations of the resolver key.
        extra_defines = set(record.get("defines", {})) - set(record["specializations"])
        assert extra_defines <= _BODY_SHARING_DEFINES
    assert included_headers == header_names


@pytest.mark.cpu_only
def test_host_shims_render_to_the_former_per_variant_text() -> None:
    # Every variant's host shim is rendered from a class template under
    # host/templates/; the sidecar pins the rendered text to the sha256 of the
    # per-variant shim file it replaced.  A record that shares its body with a
    # schedule twin passes its shim's kernel symbol as CAKE_GDN_KERNEL_SYMBOL.
    pinned_path = Path(__file__).with_name("cake_gdn_host_shim_sha256.json")
    pinned = json.loads(pinned_path.read_text(encoding="utf-8"))
    variants = cake_gdn._manifest()["variants"]
    assert sorted(pinned) == sorted(record["name"] for record in variants)
    for record in variants:
        host = record["host_binding"]
        text = cake_gdn.render_cake_gdn_host_shim(record)
        assert "@" not in text, record["name"]
        assert f"TVM_FFI_EMBED_CUBIN({host['module_ident']});" in text
        assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({host['entry']}, " in text
        assert f'"{host["cuda_symbol"]}"' in text
        defines = record.get("defines", {})
        if "CAKE_GDN_KERNEL_SYMBOL" in defines:
            assert defines["CAKE_GDN_KERNEL_SYMBOL"] == host["cuda_symbol"]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert digest == pinned[record["name"]], record["name"]


@pytest.mark.cpu_only
def test_manifest_defines_default_to_specializations() -> None:
    # The -D flags of a record are its specializations unless it spells out
    # ``defines``: the decode bodies that bake their constants in-file take
    # none, and some bodies take only a subset of the specialization keys.  A
    # record never repeats its specializations as ``defines``.
    variants = cake_gdn._manifest()["variants"]
    explicit = 0
    for record in variants:
        flags = cake_gdn._define_flags(record)
        from_specializations = tuple(
            f"-D{key}={value}" for key, value in record["specializations"].items()
        )
        if "defines" in record:
            explicit += 1
            assert flags == tuple(
                f"-D{key}={value}" for key, value in record["defines"].items()
            )
            assert flags != from_specializations
        else:
            assert flags == from_specializations
    assert 0 < explicit < len(variants)


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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

    # The SGLang TP4 BF16 indexed checkpoint route is structural: the same row
    # with the TP2 head split (8, 8, 8) stays DV-split (2 * 7 * 8 = 112 tiles)
    # but is not the frozen contract.
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
            num_q_heads=8,
            num_k_heads=8,
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


@pytest.mark.cpu_only
def test_prefill_resolver_takes_no_device_derived_lengths() -> None:
    """The resolver reads nothing back from the device.

    ``max_seq_len`` is an upper bound: the SGLang TP4 checkpoint row resolves
    from the packed length alone, and the per-sequence ``seq_lens`` keyword
    that used to feed a host-side consistency check is gone.
    """

    common = {
        "arch": "sm_103a",
        "io_dtype": "bfloat16",
        "state_dtype": "bfloat16",
        "num_seqs": 7,
        "total_seq_len": 421,
        "num_q_heads": 4,
        "num_k_heads": 4,
        "num_v_heads": 8,
        "use_initial_state": True,
        "store_final_state": True,
        "checkpoint_every_n_tokens": 64,
        "use_state_indices": True,
    }
    exact = _prefill(**common, max_seq_len=107)
    bound = _prefill(**common, max_seq_len=421)
    assert (
        exact.route_id
        == bound.route_id
        == "flashinfer.gdn_prefill.noncp.checkpoints.dvsplit"
    )
    assert exact.variant_name == bound.variant_name

    with pytest.raises(TypeError, match="seq_lens"):
        _prefill(**common, max_seq_len=107, seq_lens=(52, 93, 15, 107, 72, 61, 21))


@pytest.mark.cpu_only
def test_prefill_resolver_single_chunk_follows_the_host_bound() -> None:
    """``single_chunk`` needs a bound of at most 64 tokens, not device lengths.

    Four 64-token stateless sequences (the only shipped single-chunk head split,
    H4/HV8) resolve to the single-chunk kernel with the exact bound and fall to
    the general stateless specialization -- unshipped for this head split, so
    fail closed -- when only the packed length is known.
    """

    common = {
        "io_dtype": "bfloat16",
        "num_seqs": 4,
        "total_seq_len": 4 * 64,
        "num_q_heads": 4,
        "num_k_heads": 4,
        "num_v_heads": 8,
        "use_initial_state": False,
        "store_final_state": False,
    }
    exact = _prefill(**common, max_seq_len=64)
    assert exact.route_id == "flashinfer.gdn_prefill.noncp.single_chunk.dvsplit"
    assert "single_chunk" in exact.variant_name

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError, match="no exact frozen Cake GDN variant"
    ):
        _prefill(**common, max_seq_len=4 * 64)


@pytest.mark.cpu_only
def test_kernel_loader_fails_closed_for_unsupported_architecture() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="unsupported Cake GDN architecture",
    ):
        cake_gdn.load_cake_gdn_kernel("unused", "sm_90a")  # type: ignore[arg-type]


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_bf16_t1_route_rule_follows_the_state_head_count() -> None:
    bands = cake_gdn.CAKE_GDN_BF16_T1_ROUTE_BANDS
    bounds = [band[0] for band in bands]
    assert bounds[-1] is None and None not in bounds[:-1]
    assert bounds[:-1] == sorted(bounds[:-1])
    for _, body, tile_v in bands:
        assert body in cake_gdn.CAKE_GDN_BF16_T1_BODIES and tile_v in (16, 32, 64, 128)
    rule = cake_gdn.cake_gdn_bf16_t1_route
    assert set(cake_gdn.CAKE_GDN_BF16_T1_ROUTE_ARCH_BODIES) == {"sm_103a"}
    assert set(cake_gdn.CAKE_GDN_BF16_T1_ROUTE_ARCH_BANDS) == {"sm_107a"}
    for arch, arch_bands, overrides in (
        ("sm_100a", bands, {}),
        ("sm_103a", bands, cake_gdn.CAKE_GDN_BF16_T1_ROUTE_ARCH_BODIES["sm_103a"]),
        ("sm_107a", cake_gdn.CAKE_GDN_BF16_T1_ROUTE_ARCH_BANDS["sm_107a"], {}),
    ):
        arch_bounds = [band[0] for band in arch_bands]
        assert arch_bounds[-1] is None and None not in arch_bounds[:-1]
        assert arch_bounds[:-1] == sorted(arch_bounds[:-1])
        previous = 0
        for max_state_heads, body, tile_v in arch_bands[:-1]:
            expected = (overrides.get(max_state_heads, body), tile_v)
            assert expected[0] in cake_gdn.CAKE_GDN_BF16_T1_BODIES
            assert tile_v in (16, 32, 64, 128)
            assert rule(1, previous + 1, arch) == expected
            assert rule(1, max_state_heads, arch) == expected
            previous = max_state_heads
        assert rule(1, previous + 1, arch) == arch_bands[-1][1:]
        assert rule(512, 32, arch) == ("wide", 128)
    # sm_107a (Rubin R200) carries a complete table of its own: TILE_V=16 of the
    # occupancy-first body below 192 heads and again at 369-416, TILE_V=32 in
    # between and up to 3072 heads, the wide body only above that.
    assert rule(1, 32, "sm_107a") == ("vec8occ", 16)
    assert rule(6, 32, "sm_107a") == ("vec8occ", 16)
    assert rule(8, 32, "sm_107a") == ("vec8", 32)
    assert rule(9, 32, "sm_107a") == ("vec8occ", 32)
    assert rule(23, 16, "sm_107a") == ("vec8occ", 32)
    assert rule(24, 16, "sm_107a") == ("vec8occ", 16)
    assert rule(13, 32, "sm_107a") == ("vec8occ", 16)
    assert rule(14, 32, "sm_107a") == ("vec8", 32)
    assert rule(24, 32, "sm_107a") == ("vec8", 32)
    assert rule(32, 32, "sm_107a") == ("vec8occ", 32)
    assert rule(96, 32, "sm_107a") == ("vec8occ", 32)
    assert rule(128, 32, "sm_107a") == ("wide", 128)
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("heads", _QWEN35_BF16_T1_GEOMETRIES)
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a", "sm_107a"))
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
            # The loader fails closed on a record that does not list the
            # architecture, so every routed instance must carry it.
            assert arch in record["architectures"]
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a", "sm_107a"))
def test_decode_resolver_routes_the_bf16_verify_tile16_body_per_architecture(
    arch,
) -> None:
    # Rubin (sm_107a) runs the v-prefetch / unrolled `tile16_vpre` schedule of the
    # full-warp tile-v16 verify kernel; B200 / B300 keep the shipped body.  The
    # route id (grid tile) is the same on every architecture.
    assert cake_gdn.CAKE_GDN_BF16_VERIFY_TILE16_ARCH_BODIES == {
        "sm_107a": "tile16_vpre_es_pro"
    }
    body = "tile16_vpre_es_pro" if arch == "sm_107a" else "tile16"
    rows = [
        *(
            dict(
                batch_size=batch_size,
                seq_len=seq_len,
                num_k_heads=8,
                num_q_heads=8,
                num_v_heads=16,
                cache_steps=seq_len,
            )
            for batch_size, seq_len in ((1, 7), (2, 7), (3, 7), (4, 7), (1, 8))
        ),
        *(
            dict(
                batch_size=batch_size,
                seq_len=4,
                num_k_heads=4,
                num_q_heads=4,
                num_v_heads=8,
                cache_steps=4,
            )
            for batch_size in range(1, 9)
        ),
    ]
    for overrides in rows:
        route = _decode(
            arch=arch,
            state_dtype="bfloat16",
            layout="pretranspose",
            strided_inputs=True,
            disable_state_update=True,
            cache_intermediate_states=True,
            **overrides,
        )
        assert route.route_id.endswith(".tile16_fullwarp")
        assert re.search(
            rf"_t4_bf16state_{body}_[0-9a-f]{{12}}$", route.variant_name
        ), (
            arch,
            overrides,
            route.variant_name,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_decode_resolver_routes_the_fp32_mtp_bodies_per_architecture(arch) -> None:
    # The promoted fp32-state MTP update row (B4 T4, cache) runs the `_pro` twin of
    # `mtp_t4_splitv8` on Rubin (sm_107a): same kernel contract, bitwise-identical
    # output, only the schedule of the producer phase differs.  The verify row
    # (B1 T2, `mtp_t2_inline_tile8`) runs its `_bb` twin on Rubin (branch-light token
    # body, bitwise-identical output); the tile64 update rows (B16 / B64 T4) run the
    # `_sl2` state-load twin on Rubin (split cp.async commit groups,
    # bitwise-identical output); every other architecture keeps the shipped
    # bodies.  Route ids never change.
    assert cake_gdn.CAKE_GDN_FP32_MTP_ARCH_BODIES == {
        (
            "sm_107a",
            "gdn_decode_pretranspose_mtp_t4_splitv8",
        ): "gdn_decode_pretranspose_mtp_t4_splitv8_pro",
        (
            "sm_107a",
            "gdn_decode_pretranspose_mtp_t4_splitv2_tile64",
        ): "gdn_decode_pretranspose_mtp_t4_splitv2_tile64_sl2",
        (
            "sm_107a",
            "gdn_decode_pretranspose_mtp_t2_inline_tile8",
        ): "gdn_decode_pretranspose_mtp_t2_inline_tile8_bb",
    }
    suffix = "_pro" if arch == "sm_107a" else ""
    inline_suffix = "_bb" if arch == "sm_107a" else ""
    rows = [
        (
            dict(batch_size=1, seq_len=2, disable_state_update=True, cache_steps=2),
            "flashinfer.gdn_decode.indexed_fp32_mtp_t2.inline_tile8_verify_cache",
            "_mtp_t2_inline_tile8",
            inline_suffix,
        ),
        (
            dict(batch_size=4, seq_len=4, disable_state_update=False, cache_steps=4),
            "flashinfer.gdn_decode.indexed_fp32_mtp_t4.splitv8_update_cache",
            "_mtp_t4_splitv8",
            suffix,
        ),
    ]
    for overrides, route_id, body, body_suffix in rows:
        route = _decode(
            arch=arch,
            state_dtype="float32",
            layout="pretranspose",
            strided_inputs=True,
            cache_intermediate_states=True,
            num_k_heads=16,
            num_q_heads=16,
            num_v_heads=32,
            **overrides,
        )
        assert route.route_id == route_id, (arch, overrides, route.route_id)
        assert re.search(rf"{body}{body_suffix}_[0-9a-f]{{12}}$", route.variant_name), (
            arch,
            overrides,
            route.variant_name,
        )


@pytest.mark.cpu_only
def test_fp32_t1_route_rule_follows_the_state_head_count() -> None:
    # Two thresholds, one table (sm_107a only): the shipped `splitv8` body up to 64 state
    # heads, the register-direct `splitv8_reg_ost` body (the out-store twin of `splitv8_reg`) below 256, the `splitv8_tilev32` twin
    # from 256 on; B200 / B300 keep `splitv8`.
    assert cake_gdn.CAKE_GDN_FP32_T1_SPLITV8_TILEV32_MIN_STATE_HEADS == 256
    assert cake_gdn.CAKE_GDN_FP32_T1_SPLITV8_REG_MIN_STATE_HEADS == 64
    assert set(cake_gdn.CAKE_GDN_FP32_T1_SPLITV8_ARCH_BANDS) == {"sm_107a"}
    assert cake_gdn.CAKE_GDN_FP32_T1_SPLITV8_ARCH_BANDS["sm_107a"] == (
        (64, "splitv8"),
        (255, "splitv8_reg_ost"),
        (None, "splitv8_tilev32"),
    )
    assert cake_gdn.CAKE_GDN_FP32_T1_CTAS_PER_STATE == {
        "gdn_decode_pretranspose_splitv8": 8,
        "gdn_decode_pretranspose_splitv8_tilev32": 4,
        "gdn_decode_pretranspose_splitv8_reg_ost": 8,
    }
    rule = cake_gdn.cake_gdn_fp32_t1_route
    for arch in ("sm_100a", "sm_103a"):
        for batch_size, num_v_heads in (
            (1, 32),
            (2, 32),
            (7, 32),
            (8, 32),
            (16, 16),
            (64, 32),
            (512, 32),
        ):
            assert rule(batch_size, num_v_heads, arch) == "splitv8"
    # the public bench_pretranspose_b* rows (HV=32): b1 and the 64-head edge keep splitv8,
    # 64 < heads < 256 (b4 .. b7) take splitv8_reg_ost, b8 .. b512 splitv8_tilev32
    assert rule(1, 32, "sm_107a") == "splitv8"
    assert rule(2, 32, "sm_107a") == "splitv8"
    assert rule(3, 32, "sm_107a") == "splitv8_reg_ost"
    assert rule(4, 32, "sm_107a") == "splitv8_reg_ost"
    assert rule(6, 32, "sm_107a") == "splitv8_reg_ost"
    assert rule(7, 32, "sm_107a") == "splitv8_reg_ost"
    assert rule(15, 16, "sm_107a") == "splitv8_reg_ost"
    assert rule(8, 32, "sm_107a") == "splitv8_tilev32"
    assert rule(16, 16, "sm_107a") == "splitv8_tilev32"
    assert rule(16, 32, "sm_107a") == "splitv8_tilev32"
    assert rule(32, 32, "sm_107a") == "splitv8_tilev32"
    assert rule(64, 32, "sm_107a") == "splitv8_tilev32"
    assert rule(512, 32, "sm_107a") == "splitv8_tilev32"


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_decode_resolver_routes_the_fp32_t1_pretranspose_body_per_state_head_band(
    arch,
) -> None:
    # The indexed fp32-state T=1 row (H=16 / HV=32) runs the `splitv8_tilev32` twin on
    # Rubin (sm_107a) from 256 state heads: same kernel contract, bitwise-identical
    # output, four CTAs per state head instead of eight.  The route id names the twin
    # and the host takes the grid from the resolved record, not from a fixed eight.
    for batch_size in (1, 2, 4, 7, 8, 16, 32, 64):
        route = _decode(arch=arch, layout="pretranspose", batch_size=batch_size)
        body = cake_gdn.cake_gdn_fp32_t1_route(batch_size, 32, arch)
        heads = batch_size * 32
        if arch != "sm_107a" or heads <= 64:
            expected = "splitv8"
        elif heads < 256:
            expected = "splitv8_reg_ost"
        else:
            expected = "splitv8_tilev32"
        assert body == expected, (arch, batch_size, body)
        tilev32 = body == "splitv8_tilev32"
        suffix = body.removeprefix("splitv8")
        assert (
            route.route_id == f"flashinfer.gdn_decode.indexed_fp32_t1_splitv8{suffix}"
        ), (arch, batch_size, route.route_id)
        assert re.search(
            rf"pretranspose_splitv8{suffix}_[0-9a-f]{{12}}$", route.variant_name
        ), (arch, batch_size, route.variant_name)
        record = cake_gdn._kernel_record(route.variant_name)
        # The loader fails closed on a record that does not list the architecture,
        # so every routed body must carry it (the twin is exported for sm_107a only).
        assert arch in record["architectures"]
        assert record["specializations"]["H"] == 16
        assert record["specializations"]["HV"] == 32
        assert cake_gdn.cake_gdn_fp32_t1_ctas_per_state(route.variant_name) == (
            4 if tilev32 else 8
        )
    nontranspose = _decode(arch=arch, layout="nontranspose")
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError, match="not an fp32-state T=1 pretranspose"
    ):
        cake_gdn.cake_gdn_fp32_t1_ctas_per_state(nontranspose.variant_name)


@pytest.mark.cpu_only
def test_nontranspose_fp32_t1_body_rule_follows_the_batch_band() -> None:
    # One table (sm_107a only): the `mid4` twin in the small band (B < 32), the
    # `tilev32_x2` twin in the large band's wave-fit range (32 <= B < 37), the `hqsplit`
    # twin from B >= 37, the shipped small / large bodies everywhere
    # else; B200 / B300 keep the shipped bodies.
    assert cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_SMALL_BATCH_THRESHOLD == 32
    assert cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_LARGE_WAVEFIT_BATCH_LIMIT == 37
    assert cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_BASE_SCHEDULES == {
        "small": "gdn_decode_nontranspose_fp32_t1_small",
        "large": "gdn_decode_nontranspose_fp32_t1",
    }
    assert cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_RULE_BANDS == {
        "small": "small",
        "large_wavefit": "large",
        "large": "large",
    }
    assert cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_ARCH_BODIES == {
        ("sm_107a", "small"): "gdn_decode_nontranspose_fp32_t1_mid4",
        ("sm_107a", "large_wavefit"): "gdn_decode_nontranspose_fp32_t1_tilev32_x2",
        ("sm_107a", "large"): "gdn_decode_nontranspose_fp32_t1_hqsplit",
    }
    assert cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_CTAS_PER_STATE == {
        "gdn_decode_nontranspose_fp32_t1_small": 8,
        "gdn_decode_nontranspose_fp32_t1": 1,
        "gdn_decode_nontranspose_fp32_t1_mid4": 4,
        "gdn_decode_nontranspose_fp32_t1_tilev32_x2": 2,
        "gdn_decode_nontranspose_fp32_t1_hqsplit": 1,
    }
    rule_band = cake_gdn.cake_gdn_nontranspose_fp32_t1_rule_band
    for batch_size, expected in (
        (1, "small"),
        (31, "small"),
        (32, "large_wavefit"),
        (34, "large_wavefit"),
        (36, "large_wavefit"),
        (37, "large"),
        (512, "large"),
    ):
        assert rule_band(batch_size) == expected, batch_size
    rule = cake_gdn.cake_gdn_nontranspose_fp32_t1_route
    small = (
        "gdn_decode_nontranspose_fp32_t1_small",
        "flashinfer.gdn_decode.direct_fp32_t1_nontranspose_small",
    )
    large = (
        "gdn_decode_nontranspose_fp32_t1",
        "flashinfer.gdn_decode.direct_fp32_t1_nontranspose_large",
    )
    mid4 = (
        "gdn_decode_nontranspose_fp32_t1_mid4",
        "flashinfer.gdn_decode.direct_fp32_t1_nontranspose_small_mid4",
    )
    x2 = (
        "gdn_decode_nontranspose_fp32_t1_tilev32_x2",
        "flashinfer.gdn_decode.direct_fp32_t1_nontranspose_large_tilev32_x2",
    )
    hqsplit = (
        "gdn_decode_nontranspose_fp32_t1_hqsplit",
        "flashinfer.gdn_decode.direct_fp32_t1_nontranspose_large_hqsplit",
    )
    for arch in ("sm_100a", "sm_103a"):
        for batch_size in (1, 4, 8, 16, 31):
            assert rule(batch_size, arch) == small
        for batch_size in (32, 33, 34, 35, 36, 37, 64, 128, 512):
            assert rule(batch_size, arch) == large
    # the public nontranspose rows (HV=32) on sm_107a: b1 .. b31 take mid4, the wave-fit
    # range b32 .. b36 takes tilev32_x2, b37 .. b512 take hqsplit
    for batch_size in (1, 4, 8, 16, 31):
        assert rule(batch_size, "sm_107a") == mid4
    for batch_size in (32, 33, 34, 35, 36):
        assert rule(batch_size, "sm_107a") == x2
    for batch_size in (37, 38, 64, 128, 512):
        assert rule(batch_size, "sm_107a") == hqsplit


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_decode_resolver_routes_the_fp32_nontranspose_t1_body_per_batch_band(
    arch,
) -> None:
    # The K-major fp32-state T=1 rows run the `mid4` twin on Rubin (sm_107a) below 32
    # batch rows (four CTAs per state head instead of eight) and the `tilev32_x2` twin
    # in the wave-fit range 32 <= B < 37 (two CTAs per state head instead of one) and the
    # `hqsplit` twin from B >= 37 (one CTA per state head): same
    # kernel contract, bitwise-identical output and state.  The route id names the twin
    # and the host takes the grid from the resolved record, not from a fixed eight / one.
    for batch_size in (1, 4, 8, 16, 31, 32, 33, 34, 35, 36, 37, 38, 64, 128):
        route = _decode(arch=arch, layout="nontranspose", batch_size=batch_size)
        band = "small" if batch_size < 32 else "large"
        mid4 = arch == "sm_107a" and batch_size < 32
        x2 = arch == "sm_107a" and 32 <= batch_size < 37
        hqsplit = arch == "sm_107a" and batch_size >= 37
        suffix = (
            "_mid4" if mid4 else "_tilev32_x2" if x2 else "_hqsplit" if hqsplit else ""
        )
        assert (
            route.route_id
            == f"flashinfer.gdn_decode.direct_fp32_t1_nontranspose_{band}{suffix}"
        ), (arch, batch_size, route.route_id)
        schedule = (
            "nontranspose_fp32_t1_mid4"
            if mid4
            else "nontranspose_fp32_t1_tilev32_x2"
            if x2
            else "nontranspose_fp32_t1_hqsplit"
            if hqsplit
            else "nontranspose_fp32_t1_small"
            if band == "small"
            else "nontranspose_fp32_t1"
        )
        assert re.search(rf"{schedule}_[0-9a-f]{{12}}$", route.variant_name), (
            arch,
            batch_size,
            route.variant_name,
        )
        record = cake_gdn._kernel_record(route.variant_name)
        # The loader fails closed on a record that does not list the architecture,
        # so every routed body must carry it (the twin is exported for sm_107a only).
        assert arch in record["architectures"]
        assert record["specializations"]["H"] == 16
        assert record["specializations"]["HV"] == 32
        assert cake_gdn.cake_gdn_nontranspose_fp32_t1_ctas_per_state(
            route.variant_name
        ) == (4 if mid4 else 2 if x2 else 8 if band == "small" else 1)
    # the H4 / HV8 trace geometry ships a small record of its own and follows the rule
    trace = _decode(
        arch=arch,
        layout="nontranspose",
        batch_size=4,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
    )
    assert ("mid4" in trace.variant_name) == (arch == "sm_107a")
    pretranspose = _decode(arch=arch, layout="pretranspose")
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError, match="not an fp32-state T=1 nontranspose"
    ):
        cake_gdn.cake_gdn_nontranspose_fp32_t1_ctas_per_state(pretranspose.variant_name)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_decode_resolver_routes_the_bf16_wide_body_per_architecture(arch) -> None:
    # The multi-token rows outside the tile16 verify kernel run the wide
    # (TILE_V_WIDE 32/64) MTP body; Rubin (sm_107a) takes its v-prefetch /
    # unrolled `wide128_vpre` schedule, B200 / B300 keep the shipped body.  The
    # route id (flavour + grid tile) is the same on every architecture, and the
    # T=1 band rows keep the shipped wide body everywhere.
    assert cake_gdn.CAKE_GDN_BF16_WIDE_ARCH_BODIES == {"sm_107a": "wide128_vpre_xchg"}
    body = "wide128_vpre_xchg" if arch == "sm_107a" else "wide128"
    rows = [
        # (overrides, expected route id)
        (
            dict(
                batch_size=5,
                seq_len=7,
                num_k_heads=8,
                num_q_heads=8,
                num_v_heads=16,
                cache_steps=7,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
            ),
            "flashinfer.gdn_decode.indexed_bf16_verify_t7.wide32",
        ),
        (
            dict(
                batch_size=1,
                seq_len=7,
                num_k_heads=16,
                num_q_heads=16,
                num_v_heads=32,
                cache_steps=7,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
            ),
            "flashinfer.gdn_decode.indexed_bf16_verify_t7.wide32",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_k_heads=16,
                num_q_heads=16,
                num_v_heads=32,
                cache_steps=4,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
            ),
            "flashinfer.gdn_decode.indexed_bf16_verify_t4.wide32",
        ),
        (
            dict(
                batch_size=4,
                seq_len=2,
                num_k_heads=16,
                num_q_heads=16,
                num_v_heads=32,
                cache_steps=4,
                strided_inputs=False,
                disable_state_update=True,
                cache_intermediate_states=True,
            ),
            "flashinfer.gdn_decode.indexed_bf16_verify_t2.wide32",
        ),
        (
            dict(
                batch_size=8,
                seq_len=3,
                num_k_heads=16,
                num_q_heads=16,
                num_v_heads=64,
                cache_steps=3,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
            ),
            "flashinfer.gdn_decode.indexed_bf16_verify_t3.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=2,
                num_k_heads=16,
                num_q_heads=16,
                num_v_heads=64,
                cache_steps=0,
                strided_inputs=True,
                disable_state_update=False,
                cache_intermediate_states=False,
            ),
            "flashinfer.gdn_decode.indexed_bf16_update_t2.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_k_heads=16,
                num_q_heads=16,
                num_v_heads=64,
                cache_steps=5,
                strided_inputs=True,
                disable_state_update=False,
                cache_intermediate_states=True,
            ),
            "flashinfer.gdn_decode.indexed_bf16_checkpoint_t4.wide64",
        ),
    ]
    for overrides, route_id in rows:
        route = _decode(
            arch=arch, state_dtype="bfloat16", layout="pretranspose", **overrides
        )
        assert route.route_id == route_id, (arch, overrides, route.route_id)
        assert re.search(
            rf"_mtp_t4_bf16state_{body}_[0-9a-f]{{12}}$", route.variant_name
        ), (
            arch,
            overrides,
            route.variant_name,
        )
    # T=1 rows that the band table sends to the wide body keep the shipped body on every arch.
    t1 = _decode(
        arch=arch,
        state_dtype="bfloat16",
        layout="pretranspose",
        batch_size=256,
        seq_len=1,
        num_k_heads=16,
        num_q_heads=16,
        num_v_heads=32,
        strided_inputs=True,
    )
    if t1.route_id.startswith("flashinfer.gdn_decode.indexed_bf16_t1.wide"):
        assert re.search(r"_mtp_t4_bf16state_wide128_[0-9a-f]{12}$", t1.variant_name), (
            arch,
            t1.variant_name,
        )


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_architecture_mapping_is_exact() -> None:
    assert cake_gdn.arch_for_compute_capability(10, 0) == "sm_100a"
    assert cake_gdn.arch_for_compute_capability(10, 3) == "sm_103a"
    assert cake_gdn.arch_for_compute_capability(10, 7) == "sm_107a"
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="supports only SM100a/SM103a/SM107a",
    ):
        cake_gdn.arch_for_compute_capability(12, 0)


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("state_dtype", ("float32", "bfloat16"))
def test_prefill_resolver_routes_the_bf16_dvsplit_bodies_to_the_sm100a_late_state_inp_twins(
    state_dtype: str,
) -> None:
    """Lever L2 stays off on the sm_100a bf16-I/O DV-split bodies (Cake rule mirror); sm_103a takes the base."""
    common = dict(
        io_dtype="bfloat16",
        state_dtype=state_dtype,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=True,
        store_final_state=True,
        gates_present=True,
    )
    late = _prefill(arch="sm_100a", **common)
    early = _prefill(arch="sm_103a", **common)
    assert late.route_id == early.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert "_sm100a_state_inp_late_" in late.variant_name
    assert "state_inp_late" not in early.variant_name
    late_record = cake_gdn._kernel_record(late.variant_name)
    early_record = cake_gdn._kernel_record(early.variant_name)
    assert late_record["architectures"] == ["sm_100a"]
    assert late_record["specializations"] == early_record["specializations"]
    assert late_record["tma_abi"] == early_record["tma_abi"] == "grid_constant"
    base = early_record["source_schedule"]
    assert late_record["source_schedule"] == f"{base}_sm100a_state_inp_late"


@pytest.mark.cpu_only
def test_prefill_schedule_attr_pins_the_sm100a_late_state_inp_twin_names() -> None:
    for use_initial_state in (False, True):
        for state_dtype in ("float32", "bfloat16"):
            kwargs = dict(
                dvsplit=True,
                use_initial_state=use_initial_state,
                io_dtype="bfloat16",
                state_dtype=state_dtype,
                single_chunk=False,
                gatepipe4=False,
                fullgrid_vhold=False,
            )
            base = cake_gdn._prefill_schedule_attr(
                sm100a_state_inp_late=False,
                sm103a_role_rebalanced=False,
                gate_stream_full=False,
                **kwargs,
            )
            twin = cake_gdn._prefill_schedule_attr(
                sm100a_state_inp_late=True,
                sm103a_role_rebalanced=False,
                gate_stream_full=False,
                **kwargs,
            )
            assert base.startswith("flashinfer_blackwell_gdn_prefill_dvsplit")
            assert twin == f"{base}_sm100a_state_inp_late"
    # The rule never fires for sm_103a / sm_107a, f16 I/O, single-chunk or indexed-state routes.
    # Hq16/Hv32 (Qwen3.5-35B TP1): the FP32-state DV-split base record lists sm_103a
    # and sm_107a; the FP32-state TP2/TP4/TP8 base records list sm_103a only (sm_100a
    # runs the late-state-input twins, which are the only records listing it).
    for arch in ("sm_103a", "sm_107a"):
        assert (
            "state_inp_late"
            not in _prefill(
                arch=arch,
                io_dtype="bfloat16",
                num_q_heads=16,
                num_k_heads=16,
                num_v_heads=32,
                gates_present=True,
            ).variant_name
        )
    assert (
        "state_inp_late" not in _prefill(gates_present=True).variant_name
    )  # f16 I/O (gatepipe4)


def _requested_prefill_source(monkeypatch, **overrides):
    """The (schedule attr, specializations, route) the resolver asks the manifest for, whatever the manifest ships."""
    captured: dict = {}

    def fake_variant_for(*, arch, domain, schedule_attr, specializations):
        captured.update(
            arch=arch,
            domain=domain,
            schedule_attr=schedule_attr,
            specializations=dict(specializations),
        )
        return {"name": f"stub:{schedule_attr}"}

    cake_gdn.select_cake_gdn_prefill_variant.cache_clear()
    monkeypatch.setattr(cake_gdn, "_variant_for", fake_variant_for)
    try:
        route = _prefill(**overrides)
    finally:
        cake_gdn.select_cake_gdn_prefill_variant.cache_clear()
    assert captured["domain"] == "prefill"
    return captured["schedule_attr"], captured["specializations"], route.route_id


# Hq16/Hv16, fp16 I/O, fp32 state, initial + final state, 4 x 16 x 2 = 128 DV-split tiles: neither gatepipe4
# (tiles != 128) nor, off sm_100a, fullgrid_vhold -- the exact SM103a role-rebalance compile key.
_HQ16_HV16_F16_STATEFUL_128_TILES = dict(
    io_dtype="float16",
    state_dtype="float32",
    num_seqs=4,
    total_seq_len=4 * 1024,
    max_seq_len=1024,
    num_q_heads=16,
    num_k_heads=16,
    num_v_heads=16,
    use_initial_state=True,
    store_final_state=True,
    gates_present=True,
)


@pytest.mark.cpu_only
def test_prefill_resolver_routes_the_sm103a_hq16_hv16_fp16_stateful_key_to_the_role_rebalanced_twin(
    monkeypatch,
) -> None:
    """Cake rule mirror (``_should_use_sm103a_initial_state_role_rebalance``): sm_103a builds the 224/56 role-split
    twin for exactly this key; sm_107a (CC 10.7) takes the ``l7_vec2`` sequential twin of the plain body
    (``CAKE_GDN_PREFILL_SEQ_ARCH_BODIES``), sm_100a the full-grid V-hold body."""
    base = "flashinfer_blackwell_gdn_prefill_dvsplit_initial_f16io"
    key = _HQ16_HV16_F16_STATEFUL_128_TILES
    twin, twin_specs, route = _requested_prefill_source(
        monkeypatch, arch="sm_103a", **key
    )
    assert twin == f"{base}_sm103a_role_rebalanced"
    assert route == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert (
        twin_specs["NUM_O_HEADS_LOG2"] == 4
        and twin_specs["HEAD_GROUP_LOG2"] == 0
        and twin_specs["IS_GQA"] == 1
    )
    assert twin_specs["USE_INITIAL_STATE"] == 1 and twin_specs["STORE_FINAL_STATE"] == 1
    assert (
        twin_specs["ENABLE_CHECKPOINTS"] == 0 and twin_specs["USE_STATE_INDICES"] == 0
    )
    assert twin_specs["SINGLE_CHUNK_NO_STATE"] == 0 and twin_specs["UNIT_GATES"] == 0
    plain, plain_specs, _ = _requested_prefill_source(
        monkeypatch, arch="sm_107a", **key
    )
    assert plain == f"{base}_sm107a_l7_vec2" and plain_specs == twin_specs
    vhold, vhold_specs, _ = _requested_prefill_source(
        monkeypatch, arch="sm_100a", **key
    )
    assert vhold == f"{base}_fullgrid_vhold" and vhold_specs == twin_specs
    # Without gates the gatepipe4 schedule is never selected, so the rule fires at any tile count on sm_103a.
    no_gates, _, _ = _requested_prefill_source(
        monkeypatch,
        arch="sm_103a",
        **{**key, "num_seqs": 2, "total_seq_len": 2048, "gates_present": False},
    )
    assert no_gates == twin
    # Every neighbour of the key stays on its base on sm_103a: gatepipe4 (gated, tiles != 128), bf16 I/O,
    # Hq8/Hv8, Hq16/Hv32 (full DV), no final state, indexed state.
    for change in (
        dict(num_seqs=2, total_seq_len=2048),
        dict(io_dtype="bfloat16"),
        dict(num_q_heads=8, num_k_heads=8, num_v_heads=8),
        dict(num_v_heads=32),
        dict(store_final_state=False),
        dict(use_state_indices=True),
    ):
        attr, _, _ = _requested_prefill_source(
            monkeypatch, arch="sm_103a", **{**key, **change}
        )
        assert "role_rebalanced" not in attr, (change, attr)


@pytest.mark.cpu_only
def test_prefill_schedule_attr_pins_the_sm103a_role_rebalanced_twin_name() -> None:
    kwargs = dict(
        dvsplit=True,
        use_initial_state=True,
        io_dtype="float16",
        state_dtype="float32",
        single_chunk=False,
        gatepipe4=False,
        fullgrid_vhold=False,
        sm100a_state_inp_late=False,
    )
    base = cake_gdn._prefill_schedule_attr(
        sm103a_role_rebalanced=False, gate_stream_full=False, **kwargs
    )
    assert base == "flashinfer_blackwell_gdn_prefill_dvsplit_initial_f16io"
    assert (
        cake_gdn._prefill_schedule_attr(
            sm103a_role_rebalanced=True, gate_stream_full=False, **kwargs
        )
        == f"{base}_sm103a_role_rebalanced"
    )


# Hq16/Hv32, bf16 I/O, fp32 state, initial + final state, 32 x 256-token sequences: 1024 full-DV work tiles on the
# 148 / 160 / 212-CTA grid of the grid rule (seven, seven and five tiles per CTA), mean sequence length 256 -- a
# gate-stream key on every architecture (Cake rule ``gate_stream_flat_default``: >= 2 tiles per CTA, mean <= 512).
_HQ16_HV32_BF16_STATEFUL_32X256 = dict(
    io_dtype="bfloat16",
    state_dtype="float32",
    num_seqs=32,
    total_seq_len=32 * 256,
    max_seq_len=256,
    num_q_heads=16,
    num_k_heads=16,
    num_v_heads=32,
    use_initial_state=True,
    store_final_state=True,
    gates_present=True,
)


@pytest.mark.cpu_only
def test_prefill_resolver_routes_many_short_full_dv_tiles_to_the_gatestream_twin(
    monkeypatch,
) -> None:
    """Cake rule mirror (``gate_stream_flat_default``, exporter ``_prefill_plan``): the full-DV stateful bodies take
    their ``_gatestream_full`` twin on every architecture when a CTA owns at least two work tiles under the grid rule
    and the mean sequence length is at most 512 tokens; every other key stays on its base.  On sm_107a the seq twin
    rule applies on top: the gate-stream twin's own ``_sm107a_i64meta`` twin (no sm_107a record is a base body)."""
    base = "flashinfer_blackwell_gdn_prefill_initial"
    key = _HQ16_HV32_BF16_STATEFUL_32X256
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        twin, specs, route = _requested_prefill_source(monkeypatch, arch=arch, **key)
        seq_twin = "_sm107a_i64meta" if arch == "sm_107a" else ""
        assert twin == f"{base}_gatestream_full{seq_twin}", arch
        assert route == "flashinfer.gdn_prefill.noncp.full_dv"
        assert (
            specs["NUM_O_HEADS_LOG2"] == 5
            and specs["HEAD_GROUP_LOG2"] == 1
            and specs["IS_GQA"] == 0
        )
        assert specs["USE_INITIAL_STATE"] == 1 and specs["STORE_FINAL_STATE"] == 1
        assert specs["ENABLE_CHECKPOINTS"] == 0 and specs["USE_STATE_INDICES"] == 0
        assert specs["SINGLE_CHUNK_NO_STATE"] == 0 and specs["UNIT_GATES"] == 0
    # The bf16-state body has its own twin; 16 x 512 (512 tiles, four per CTA, mean 512) and the Hq8/Hv32 and
    # Hq4/Hv16 16 x 512 head splits (512 / 256 tiles on 128 CTAs) qualify as well.
    for change, expected in (
        (dict(state_dtype="bfloat16"), f"{base}_bf16state_gatestream_full"),
        (
            dict(num_seqs=16, total_seq_len=16 * 512, max_seq_len=512),
            f"{base}_gatestream_full",
        ),
        (
            dict(
                num_seqs=16,
                total_seq_len=16 * 512,
                max_seq_len=512,
                num_q_heads=8,
                num_k_heads=8,
                state_dtype="bfloat16",
            ),
            f"{base}_bf16state_gatestream_full",
        ),
        (
            dict(
                num_seqs=16,
                total_seq_len=16 * 512,
                max_seq_len=512,
                num_q_heads=4,
                num_k_heads=4,
                num_v_heads=16,
            ),
            f"{base}_gatestream_full",
        ),
    ):
        attr, _, _ = _requested_prefill_source(
            monkeypatch, arch="sm_103a", **{**key, **change}
        )
        assert attr == expected, (change, attr)
    # Off: 8 x 1024 (256 tiles, two per CTA, but a 1024-token mean), 16 x 513 (mean above 512), 16 x 512 Hq4/Hv8
    # (128 tiles, one CTA per tile), 4 x 256 Hq4/Hv8 (DV-split: one CTA per tile; sm_100a takes the L2 twin), f16
    # I/O and the stateless body (no gate-stream twin), indexed state (the ``_indexed`` body, no twin).
    for change, expected in (
        (dict(num_seqs=8, total_seq_len=8 * 1024, max_seq_len=1024), base),
        (dict(num_seqs=16, total_seq_len=16 * 513, max_seq_len=513), base),
        (
            dict(
                num_seqs=16,
                total_seq_len=16 * 512,
                max_seq_len=512,
                num_q_heads=4,
                num_k_heads=4,
                num_v_heads=8,
            ),
            base,
        ),
        (
            dict(
                num_seqs=4,
                total_seq_len=4 * 256,
                num_q_heads=4,
                num_k_heads=4,
                num_v_heads=8,
            ),
            "flashinfer_blackwell_gdn_prefill_dvsplit_initial_sm100a_state_inp_late",
        ),
        (dict(io_dtype="float16"), f"{base}_f16io"),
        (
            dict(use_initial_state=False, store_final_state=False),
            "flashinfer_blackwell_gdn_prefill",
        ),
        (dict(use_state_indices=True), base),
    ):
        attr, _, _ = _requested_prefill_source(
            monkeypatch, arch="sm_100a", **{**key, **change}
        )
        assert attr == expected, (change, attr)


@pytest.mark.cpu_only
def test_prefill_resolver_runs_the_base_record_where_no_gatestream_twin_ships() -> None:
    """The ``_gatestream_full`` twin is exported for the head geometries the representative rows reach; a key
    whose twin is not in the manifest (here the Qwen3.5-397B TP4 Hq4/Hv16 FP32-state record, 64 x 256 full-DV
    tiles) resolves to its base ``_initial`` record on every architecture that ships it -- the route upstream
    served before the twin existed -- while the Hq16/Hv32 key of the same shape takes the twin."""
    shape = dict(
        io_dtype="bfloat16",
        state_dtype="float32",
        num_seqs=64,
        total_seq_len=64 * 256,
        max_seq_len=256,
        use_initial_state=True,
        store_final_state=True,
        gates_present=True,
    )
    for arch in ("sm_100a", "sm_103a"):
        base = _prefill(
            arch=arch, num_q_heads=4, num_k_heads=4, num_v_heads=16, **shape
        )
        assert base.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
        assert "gatestream" not in base.variant_name, base
        record = cake_gdn._kernel_record(base.variant_name)
        assert record["source_schedule"].endswith(
            ":flashinfer_blackwell_gdn_prefill_initial"
        )
        assert (
            record["specializations"]["HEAD_GROUP_LOG2"],
            record["specializations"]["IS_GQA"],
            record["specializations"]["NUM_O_HEADS_LOG2"],
        ) == (2, 0, 4)
        twin = _prefill(
            arch=arch, num_q_heads=16, num_k_heads=16, num_v_heads=32, **shape
        )
        assert twin.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
        assert twin.variant_name.startswith(
            "prefill_flashinfer_blackwell_gdn_prefill_initial_gatestream_full"
        ), twin
    # CC 10.7 runs the TP4 key through its `i64meta` twin record (the sm_107a default body:
    # the same schedule reading int32 or int64 metadata directly); the base record lists the
    # other architectures only.  No gatestream twin exists for the key on any architecture.
    seven = _prefill(
        arch="sm_107a", num_q_heads=4, num_k_heads=4, num_v_heads=16, **shape
    )
    assert seven.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
    assert seven.variant_name != base.variant_name, (seven, base)
    assert seven.metadata_i64 is True and base.metadata_i64 is False
    seven_record = cake_gdn._kernel_record(seven.variant_name)
    assert seven_record["source_schedule"].endswith(
        ":flashinfer_blackwell_gdn_prefill_initial_sm107a_i64meta"
    )
    assert seven_record["architectures"] == ["sm_107a"]
    assert seven_record["specializations"] == record["specializations"]
    assert "sm_107a" not in cake_gdn._kernel_record(base.variant_name)["architectures"]


@pytest.mark.cpu_only
def test_prefill_resolver_runs_an_i64_metadata_twin_for_every_key_on_sm_107a() -> None:
    """P1-I64META: on CC 10.7 every non-CP prefill key resolves to an sm_107a-only twin record that
    declares ``metadata_i64`` (the routed vec2 bodies and the ``i64meta`` default alike); on the
    other architectures the resolved record never declares it."""
    keys = (
        # full-DV ``_initial`` TP4 key (64 x 256 tokens, Hq4/Hv16): the ``i64meta`` default body
        dict(
            io_dtype="bfloat16",
            num_seqs=64,
            total_seq_len=64 * 256,
            max_seq_len=256,
            num_q_heads=4,
            num_k_heads=4,
            num_v_heads=16,
            gates_present=True,
        ),
        # dvsplit fp16-I/O fp32-state Hq2/Hv8 (the ``_prefill`` default): the routed ``l7_vec2`` body
        dict(),
        # dvsplit bf16-state Hq2/Hv8: the routed ``l7b_vec2lp`` body (sm_100a runs its late-input twin)
        dict(io_dtype="bfloat16", state_dtype="bfloat16"),
        # the single-chunk stateless record (Hq4/Hv8, 64 tokens): the ``i64meta`` default body
        dict(
            io_dtype="bfloat16",
            num_q_heads=4,
            num_k_heads=4,
            num_v_heads=8,
            max_seq_len=64,
            total_seq_len=64,
            use_initial_state=False,
            store_final_state=False,
        ),
    )
    for key in keys:
        seven = _prefill(arch="sm_107a", **key)
        record = cake_gdn._kernel_record(seven.variant_name)
        assert seven.metadata_i64 is True, (key, seven)
        assert record.get("metadata_i64") is True and record["architectures"] == [
            "sm_107a"
        ], (key, record)
        assert "_sm107a_" in record["source_schedule"], (key, record)
        for arch in ("sm_100a", "sm_103a"):
            other = _prefill(arch=arch, **key)
            assert other.metadata_i64 is False, (key, other)
            assert "metadata_i64" not in cake_gdn._kernel_record(other.variant_name), (
                key,
                other,
            )
    # Every sm_107a non-CP prefill record of the manifest is a twin with the field; no base record lists sm_107a.
    for record in cake_gdn._manifest()["variants"]:
        if record["domain"] != "prefill":
            continue
        twin = "_sm107a_" in record["source_schedule"]
        assert ("sm_107a" in record["architectures"]) == twin, record["name"]
        assert record.get("metadata_i64", False) is twin, record["name"]
        if twin:
            assert record["architectures"] == ["sm_107a"], record["name"]


@pytest.mark.cpu_only
def test_prefill_schedule_attr_pins_the_gatestream_twin_names() -> None:
    kwargs = dict(
        dvsplit=False,
        io_dtype="bfloat16",
        single_chunk=False,
        gatepipe4=False,
        fullgrid_vhold=False,
        sm100a_state_inp_late=False,
        sm103a_role_rebalanced=False,
    )
    for state_dtype in ("float32", "bfloat16"):
        base = cake_gdn._prefill_schedule_attr(
            use_initial_state=True,
            state_dtype=state_dtype,
            gate_stream_full=False,
            **kwargs,
        )
        assert base == "flashinfer_blackwell_gdn_prefill_initial" + (
            "_bf16state" if state_dtype == "bfloat16" else ""
        )
        twin = cake_gdn._prefill_schedule_attr(
            use_initial_state=True,
            state_dtype=state_dtype,
            gate_stream_full=True,
            **kwargs,
        )
        assert twin == f"{base}_gatestream_full"
        # No twin without initial state: the flag leaves the stateless name unchanged.
        stateless = cake_gdn._prefill_schedule_attr(
            use_initial_state=False,
            state_dtype=state_dtype,
            gate_stream_full=False,
            **kwargs,
        )
        assert (
            cake_gdn._prefill_schedule_attr(
                use_initial_state=False,
                state_dtype=state_dtype,
                gate_stream_full=True,
                **kwargs,
            )
            == stateless
        )
    # The rule itself: tiles per CTA from the grid rule, mean sequence length from the host scalars.
    assert cake_gdn._cake_gdn_prefill_gate_stream_full(
        num_seqs=32, total_seq_len=32 * 256, total_tiles=1024, grid_x=148
    )
    assert cake_gdn._cake_gdn_prefill_gate_stream_full(
        num_seqs=16, total_seq_len=16 * 512, total_tiles=256, grid_x=128
    )
    assert not cake_gdn._cake_gdn_prefill_gate_stream_full(
        num_seqs=16, total_seq_len=16 * 513, total_tiles=256, grid_x=128
    )
    assert not cake_gdn._cake_gdn_prefill_gate_stream_full(
        num_seqs=8, total_seq_len=8 * 1024, total_tiles=256, grid_x=128
    )
    assert not cake_gdn._cake_gdn_prefill_gate_stream_full(
        num_seqs=16, total_seq_len=16 * 512, total_tiles=128, grid_x=128
    )
    assert cake_gdn._cake_gdn_prefill_grid_x(1024, 148, False, 256) == 148
    assert cake_gdn._cake_gdn_prefill_grid_x(512, 160, False, 512) == 128
    assert cake_gdn._cake_gdn_prefill_grid_x(128, 148, True, 1024) == 128


@pytest.mark.cpu_only
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
    )
    live = cake_gdn.select_cake_gdn_prefill_variant(
        **common,
        num_seqs=5,
        total_seq_len=4296,
        max_seq_len=897,
    )
    assert (
        frozen.route_id
        == live.route_id
        == "flashinfer.gdn_prefill.noncp.checkpoints.dvsplit"
    )
    assert frozen.variant_name == live.variant_name


@pytest.mark.cpu_only
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
            )


# Qwen3-Next / Qwen3.5 per-rank linear-attention head splits of the vLLM
# prefill contract (BF16 I/O, packed FP32 initial+final state) as frozen
# (HEAD_GROUP_LOG2, IS_GQA, NUM_O_HEADS_LOG2) triples: H/HV = 16/32, 8/16,
# 4/8, 2/4 (Qwen3-Next-80B and Qwen3.5-35B-A3B at TP1/TP2/TP4/TP8) and
# 16/64, 8/32, 4/16, 2/8 (Qwen3.5-397B-A17B at TP1/TP2/TP4/TP8).
_QWEN_TP_FP32_STATE_GEOMETRIES = frozenset(
    {
        (1, 0, 2),
        (1, 0, 3),
        (1, 0, 4),
        (1, 0, 5),
        (2, 0, 3),
        (2, 0, 4),
        (2, 0, 5),
        (2, 0, 6),
    }
)


def _prefill_heads(geometry):
    """``(num_q_heads, num_k_heads, num_v_heads)`` of one power-of-two geometry triple."""

    head_group_log2, is_gqa, num_o_heads_log2 = geometry
    num_o_heads = 1 << num_o_heads_log2
    min_heads = num_o_heads >> head_group_log2
    if is_gqa:
        return num_o_heads, min_heads, min_heads
    return min_heads, min_heads, num_o_heads


def _prefill_geometry_of(route):
    specs = cake_gdn._kernel_record(route.variant_name)["specializations"]
    return (specs["HEAD_GROUP_LOG2"], specs["IS_GQA"], specs["NUM_O_HEADS_LOG2"])


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
def test_prefill_fp32_state_contract_covers_every_qwen_tp_geometry(arch) -> None:
    contract = {
        "arch": arch,
        "io_dtype": "bfloat16",
        "state_dtype": "float32",
        "use_initial_state": True,
        "store_final_state": True,
    }
    for dvsplit in (True, False):
        geometries = cake_gdn.cake_gdn_prefill_geometries(**contract, dvsplit=dvsplit)
        missing = _QWEN_TP_FP32_STATE_GEOMETRIES - geometries
        assert not missing, (dvsplit, sorted(missing))
    for geometry in sorted(_QWEN_TP_FP32_STATE_GEOMETRIES):
        num_q_heads, num_k_heads, num_v_heads = _prefill_heads(geometry)
        # One sequence keeps 2 * num_o_heads <= 128 tiles on the DV-split
        # schedule; 64 sequences exceed every arch's active-cluster count.
        for num_seqs, suffix in ((1, ".dvsplit"), (64, ".full_dv")):
            route = cake_gdn.select_cake_gdn_prefill_variant(
                arch=arch,
                io_dtype="bfloat16",
                state_dtype="float32",
                num_seqs=num_seqs,
                total_seq_len=256 * num_seqs,
                max_seq_len=256,
                num_q_heads=num_q_heads,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                use_initial_state=True,
                store_final_state=True,
                checkpoint_every_n_tokens=0,
                use_state_indices=False,
                gates_present=True,
            )
            assert route.route_id == f"flashinfer.gdn_prefill.noncp{suffix}"
            assert _prefill_geometry_of(route) == geometry


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a", "sm_107a"))
def test_prefill_geometries_mirror_the_resolver(arch) -> None:
    # The helper tabulates nothing: every power-of-two triple it lists for a
    # contract resolves, on that schedule, to a record carrying that triple.
    for io_dtype, state_dtype in (
        ("bfloat16", "float32"),
        ("bfloat16", "bfloat16"),
        ("float16", "float32"),
    ):
        for use_initial_state, store_final_state in (
            (True, True),
            (False, True),
            (False, False),
        ):
            for dvsplit in (True, False):
                geometries = cake_gdn.cake_gdn_prefill_geometries(
                    arch=arch,
                    io_dtype=io_dtype,
                    state_dtype=state_dtype,
                    use_initial_state=use_initial_state,
                    store_final_state=store_final_state,
                    dvsplit=dvsplit,
                )
                for geometry in sorted(geometries):
                    if -1 in geometry:
                        continue
                    num_q_heads, num_k_heads, num_v_heads = _prefill_heads(geometry)
                    # 256 sequences of 128 tokens exceed every arch's cluster
                    # count at any head count (and are not single-chunk).
                    num_seqs = 1 if dvsplit else 256
                    route = cake_gdn.select_cake_gdn_prefill_variant(
                        arch=arch,
                        io_dtype=io_dtype,
                        state_dtype=state_dtype,
                        num_seqs=num_seqs,
                        total_seq_len=128 * num_seqs,
                        max_seq_len=128,
                        num_q_heads=num_q_heads,
                        num_k_heads=num_k_heads,
                        num_v_heads=num_v_heads,
                        use_initial_state=use_initial_state,
                        store_final_state=store_final_state,
                        checkpoint_every_n_tokens=0,
                        use_state_indices=False,
                        # FP16-I/O gated calls take the gatepipe4/fullgrid_vhold twins.
                        gates_present=False,
                    )
                    assert route.route_id.endswith(
                        ".dvsplit" if dvsplit else ".full_dv"
                    ), route.route_id
                    assert _prefill_geometry_of(route) == geometry, route


@pytest.mark.cpu_only
def test_variant_resolution_fails_closed_for_an_unlisted_architecture(
    monkeypatch,
) -> None:
    schedule_attr = "flashinfer_blackwell_gdn_prefill_initial"
    specializations = {
        "ENABLE_CHECKPOINTS": 0,
        "HEAD_GROUP_LOG2": 1,
        "IS_GQA": 0,
        "NUM_O_HEADS_LOG2": 4,
        "SINGLE_CHUNK_NO_STATE": 0,
        "STORE_FINAL_STATE": 1,
        "UNIT_GATES": 0,
        "USE_INITIAL_STATE": 1,
        "USE_STATE_INDICES": 0,
    }
    record = {
        "name": "prefill_two_architecture_record",
        "domain": "prefill",
        "source_schedule": f"flashinfer.cake_gdn.generated.prefill:{schedule_attr}",
        "tma_abi": "grid_constant",
        "architectures": ["sm_100a", "sm_103a"],
        "outputs": [{"architectures": ["sm_100a", "sm_103a"], "path": "cuda/x.cu"}],
        "specializations": specializations,
    }
    monkeypatch.setattr(cake_gdn, "_manifest", lambda: {"variants": [record]})

    def resolve(arch):
        return cake_gdn._variant_for(
            arch=arch,
            domain="prefill",
            schedule_attr=schedule_attr,
            specializations=specializations,
        )

    assert resolve("sm_100a") is record
    assert resolve("sm_103a") is record
    # An architecture the row was never exported for fails closed at
    # resolution (CakeGDNUnsupportedError), not as the loader's ValueError;
    # the message names the architecture, the schedule and the key.
    with pytest.raises(cake_gdn.CakeGDNUnsupportedError, match="sm_107a") as info:
        resolve("sm_107a")
    assert not isinstance(info.value, ValueError)
    assert schedule_attr in str(info.value)
    assert repr(specializations) in str(info.value)

    contract = {
        "io_dtype": "bfloat16",
        "state_dtype": "float32",
        "use_initial_state": True,
        "store_final_state": True,
        "dvsplit": False,
    }
    assert cake_gdn.cake_gdn_prefill_geometries(arch="sm_103a", **contract) == {
        (1, 0, 4)
    }
    assert cake_gdn.cake_gdn_prefill_geometries(arch="sm_107a", **contract) == set()

    # Listing the architecture without a body for it fails closed the same way.
    record["outputs"] = [{"architectures": ["sm_100a"], "path": "cuda/x.cu"}]
    with pytest.raises(cake_gdn.CakeGDNUnsupportedError, match="sm_103a"):
        resolve("sm_103a")
    assert resolve("sm_100a") is record
    assert cake_gdn.cake_gdn_prefill_geometries(arch="sm_103a", **contract) == set()


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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
