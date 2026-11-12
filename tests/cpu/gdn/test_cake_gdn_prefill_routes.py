"""Route table of the Cake GDN non-CP prefill resolver: the sm_107a sequential twins.

Every sm_107a non-CP prefill record is a twin of a shipped schedule: the explicit table
(``CAKE_GDN_PREFILL_SEQ_ARCH_BODIES``) routes the DV-split state-I/O bodies, and the
architecture default (``CAKE_GDN_PREFILL_SEQ_DEFAULT_BODIES``: ``i64meta``) covers every
other key.  Each twin record lists sm_107a only, carries ``metadata_i64`` (its kernel
reads int32 or int64 launch metadata directly) and its base record no longer lists sm_107a.
"""

import json

import pytest

from flashinfer.jit import cake_gdn

PREFIX = "flashinfer.cake_gdn.generated.prefill:"
TABLE = cake_gdn.CAKE_GDN_PREFILL_SEQ_ARCH_BODIES
DEFAULT = cake_gdn.CAKE_GDN_PREFILL_SEQ_DEFAULT_BODIES
ROUTED = sorted(
    {
        schedule.removesuffix("_indexed")
        for (arch, schedule) in TABLE
        if arch == "sm_107a"
    }
)


@pytest.mark.cpu_only
def test_sm107a_default_body_is_the_int64_metadata_twin():
    assert DEFAULT == {"sm_107a": "i64meta"}
    assert all(arch == "sm_107a" for (arch, _schedule) in TABLE)
    assert "i64meta" not in TABLE.values()


@pytest.mark.cpu_only
@pytest.mark.parametrize("schedule", ROUTED)
def test_sm107a_routes_the_vec2_state_io_twins_per_shipped_schedule(schedule):
    body = TABLE[("sm_107a", schedule)]
    # the DV-split state-I/O twins and the fp8-state packed final-state store twin
    assert body in ("l7_vec2", "l7b_vec2lp", "st2s_vec2lp", "st8s_pack")
    twin = cake_gdn.cake_gdn_prefill_seq_schedule(
        "sm_107a", schedule, use_state_indices=False
    )
    assert twin == f"{schedule}_sm107a_{body}"
    # An indexed-state plan without an explicit entry takes the architecture default.
    indexed_body = TABLE.get(("sm_107a", f"{schedule}_indexed"), DEFAULT["sm_107a"])
    indexed = cake_gdn.cake_gdn_prefill_seq_schedule(
        "sm_107a", schedule, use_state_indices=True
    )
    assert indexed == f"{schedule}_sm107a_{indexed_body}"
    for arch in ("sm_100a", "sm_103a"):
        for use_state_indices in (False, True):
            shipped = cake_gdn.cake_gdn_prefill_seq_schedule(
                arch, schedule, use_state_indices=use_state_indices
            )
            assert shipped == schedule


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "schedule",
    (
        "flashinfer_blackwell_gdn_prefill",
        "flashinfer_blackwell_gdn_prefill_initial",
        "flashinfer_blackwell_gdn_prefill_initial_bf16state",
        "flashinfer_blackwell_gdn_prefill_initial_gatestream_full",
        "flashinfer_blackwell_gdn_prefill_dvsplit_single_chunk",
    ),
)
def test_sm107a_routes_every_other_schedule_to_its_int64_metadata_twin(schedule):
    assert ("sm_107a", schedule) not in TABLE
    for use_state_indices in (False, True):
        twin = cake_gdn.cake_gdn_prefill_seq_schedule(
            "sm_107a", schedule, use_state_indices=use_state_indices
        )
        assert twin == f"{schedule}_sm107a_i64meta"
        for arch in ("sm_100a", "sm_103a"):
            assert (
                cake_gdn.cake_gdn_prefill_seq_schedule(
                    arch, schedule, use_state_indices=use_state_indices
                )
                == schedule
            )


def _prefill_records_by_key():
    by_key: dict = {}
    for record in cake_gdn._manifest()["variants"]:
        if record["domain"] != "prefill":
            continue
        key = (
            record["source_schedule"],
            json.dumps(record["specializations"], sort_keys=True),
        )
        by_key.setdefault(key, []).append(record)
    return by_key


@pytest.mark.cpu_only
def test_every_shipped_sm107a_record_is_a_twin_with_int64_metadata():
    twins = 0
    for records in _prefill_records_by_key().values():
        for record in records:
            schedule = record["source_schedule"].removeprefix(PREFIX)
            is_twin = "_sm107a_" in schedule
            assert ("sm_107a" in record["architectures"]) is is_twin, record["name"]
            assert bool(record.get("metadata_i64", False)) is is_twin, record["name"]
            if is_twin:
                assert record["architectures"] == ["sm_107a"], record["name"]
                twins += 1
    assert twins > 0


@pytest.mark.cpu_only
def test_every_shipped_base_record_the_rule_routes_has_its_sm107a_twin_record():
    by_key = _prefill_records_by_key()
    checked = 0
    for (source, spec_json), records in by_key.items():
        schedule = source.removeprefix(PREFIX)
        if "_sm107a_" in schedule:
            continue
        for record in records:
            twin = cake_gdn.cake_gdn_prefill_seq_schedule(
                "sm_107a",
                schedule,
                use_state_indices=bool(record["specializations"]["USE_STATE_INDICES"]),
            )
            assert twin != schedule
            twin_records = by_key.get((PREFIX + twin, spec_json))
            if twin_records is None:
                # A base key no sm_107a record ever served (a record exported for a subset of
                # the architectures); the resolver fails closed on sm_107a for it.
                assert "sm_107a" not in record["architectures"], record["name"]
                continue
            (twin_record,) = twin_records
            assert twin_record["architectures"] == ["sm_107a"]
            assert twin_record["metadata_i64"] is True
            # The twin's host shim is its base's shim class reading the int32 / int64
            # metadata: same launch parameters, the twin's own names and template.
            assert (
                twin_record["host_binding"]["params"]
                == record["host_binding"]["params"]
            )
            assert twin_record["tma_abi"] == record["tma_abi"] == "grid_constant"
            checked += 1
    assert checked > 0


_GATESTREAM_SM107A = (
    "prefill_flashinfer_blackwell_gdn_prefill_initial_gatestream_full_sm107a_i64meta_"
)
_GATESTREAM_PF_SM107A = "prefill_flashinfer_blackwell_gdn_prefill_initial_gatestream_full_pf_sm107a_i64meta_"
_INITIAL_SM107A = "prefill_flashinfer_blackwell_gdn_prefill_initial_sm107a_i64meta_"


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("num_q_heads", "num_v_heads", "num_seqs", "seq_len", "suffix"),
    (
        # Qwen3-Next TP8 and Qwen3.5 TP2/TP4/TP8 per-rank FP32-state splits whose
        # full-DV CTAs own at least two short tiles: the gate-stream twin, as in Cake,
        # as its prefetch twin (the launch grid gives some CTA a second tile).
        (2, 4, 64, 256, "b00507dd0f31"),
        (8, 32, 16, 512, "12da800ed86f"),
        (8, 32, 32, 512, "12da800ed86f"),
        (8, 32, 64, 256, "12da800ed86f"),
        (4, 16, 16, 512, "6fefb73716fe"),
        (4, 16, 32, 512, "6fefb73716fe"),
        (4, 16, 64, 256, "6fefb73716fe"),
        (2, 8, 32, 512, "41cbb32105a5"),
        (2, 8, 64, 256, "41cbb32105a5"),
    ),
)
def test_sm107a_fp32_state_gate_stream_twin_per_head_split(
    num_q_heads, num_v_heads, num_seqs, seq_len, suffix
):
    common = dict(
        io_dtype="bfloat16",
        state_dtype="float32",
        num_seqs=num_seqs,
        total_seq_len=num_seqs * seq_len,
        max_seq_len=seq_len,
        num_q_heads=num_q_heads,
        num_k_heads=num_q_heads,
        num_v_heads=num_v_heads,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=0,
        use_state_indices=False,
        gates_present=True,
    )
    route = cake_gdn.select_cake_gdn_prefill_variant(arch="sm_107a", **common)
    assert route.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
    assert route.variant_name == _GATESTREAM_PF_SM107A + suffix
    assert route.metadata_i64
    # The twin reuses the shipped gate-stream body and its base's launch parameters; its
    # prefetch twin is the same key and launch on the prefetching body (its own kernel symbol).
    records = {r["name"]: r for r in cake_gdn._manifest()["variants"]}
    twin = records[_GATESTREAM_SM107A + suffix]
    prefetch = records[route.variant_name]
    for field in ("specializations", "metadata_i64"):
        assert prefetch[field] == twin[field], field
    for field in ("params", "template"):
        assert prefetch["host_binding"][field] == twin["host_binding"][field], field
    assert (
        prefetch["host_binding"]["cuda_symbol"]
        == twin["host_binding"]["cuda_symbol"] + "_pf"
    )
    assert prefetch["outputs"] != twin["outputs"]
    (base,) = (
        r
        for name, r in records.items()
        if name.startswith(_INITIAL_SM107A)
        and r["specializations"] == twin["specializations"]
    )
    assert twin["host_binding"]["params"] == base["host_binding"]["params"]
    assert twin["host_binding"]["template"] == base["host_binding"]["template"]
    shipped = {
        r["host_binding"]["cuda_symbol"]: r["outputs"]
        for name, r in records.items()
        if name.startswith(_GATESTREAM_SM107A)
    }
    assert len(shipped) == 1
    assert twin["outputs"] == next(iter(shipped.values()))
    shipped_prefetch = {
        r["host_binding"]["cuda_symbol"]: r["outputs"]
        for name, r in records.items()
        if name.startswith(_GATESTREAM_PF_SM107A)
    }
    assert set(shipped_prefetch) == {symbol + "_pf" for symbol in shipped}
    assert prefetch["outputs"] == next(iter(shipped_prefetch.values()))
    # Long sequences keep the base body (mean length above 512 tokens).
    long = cake_gdn.select_cake_gdn_prefill_variant(
        arch="sm_107a",
        **{**common, "total_seq_len": num_seqs * 2048, "max_seq_len": 2048},
    )
    assert long.variant_name == base["name"]


_MIXED16 = (8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("num_q_heads", "num_v_heads", "full_dv", "dvsplit"),
    (
        # Qwen3-Next / Qwen3.5 TP2/TP4/TP8 FP32-state splits on the ragged 16-sequence batch
        # (one 8192-token sequence first): with ``cu_seqlens_cpu`` the batch runs the
        # DV-split body of its key, one CTA per half tile.
        (8, 16, "f16a8d42d552", "5d5e7cd3bb78"),
        (4, 8, "f11f48fee0a6", "aa51e505aec0"),
        (8, 32, "73606292b9c9", "9d051157bbf3"),
        (4, 16, "df3cde37e265", "d04755b8fc11"),
        (2, 8, "a4094b019dd7", "def1226b2f12"),
    ),
)
def test_sm107a_fp32_state_ragged_batch_takes_the_dvsplit_body_with_host_lengths(
    num_q_heads, num_v_heads, full_dv, dvsplit
):
    common = dict(
        arch="sm_107a",
        io_dtype="bfloat16",
        state_dtype="float32",
        num_seqs=len(_MIXED16),
        total_seq_len=sum(_MIXED16),
        max_seq_len=max(_MIXED16),
        num_q_heads=num_q_heads,
        num_k_heads=num_q_heads,
        num_v_heads=num_v_heads,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=0,
        use_state_indices=False,
        gates_present=True,
    )
    base = cake_gdn.select_cake_gdn_prefill_variant(**common)
    assert base.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
    assert base.variant_name == _INITIAL_SM107A + full_dv
    assert base.one_cta_per_tile is False
    routed = cake_gdn.select_cake_gdn_prefill_variant(**common, host_seq_lens=_MIXED16)
    assert routed.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert routed.variant_name == (
        "prefill_flashinfer_blackwell_gdn_prefill_dvsplit_initial_sm107a_l7_vec2_"
        + dvsplit
    )
    assert routed.metadata_i64
    assert routed.one_cta_per_tile is True


@pytest.mark.cpu_only
def test_sm107a_fp32_state_ragged_batch_under_the_tile_budget_keeps_its_grid():
    # Hq2/Hv4: 2 * 16 * 4 = 128 half tiles fit the 212 SMs, DV-split already.
    common = dict(
        arch="sm_107a",
        io_dtype="bfloat16",
        state_dtype="float32",
        num_seqs=len(_MIXED16),
        total_seq_len=sum(_MIXED16),
        max_seq_len=max(_MIXED16),
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=4,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=0,
        use_state_indices=False,
        gates_present=True,
    )
    routed = cake_gdn.select_cake_gdn_prefill_variant(**common, host_seq_lens=_MIXED16)
    assert routed == cake_gdn.select_cake_gdn_prefill_variant(**common)
    assert routed.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert routed.variant_name.endswith("e5e48bfdef0b")
    assert routed.one_cta_per_tile is False


_BF16STATE_SM107A = "prefill_flashinfer_blackwell_gdn_prefill_initial_bf16state_"
_GATESTREAM_BF16_SM107A = _BF16STATE_SM107A + "gatestream_full_sm107a_i64meta_"
_GATESTREAM_BF16_PF_SM107A = _BF16STATE_SM107A + "gatestream_full_pf_sm107a_i64meta_"
# A ragged BF16-state batch: 64 sequences of mean 295 tokens, Hq4/Hv16 (1024 full-DV tiles).
_RAGGED64 = (
    700,
    20,
    450,
    1,
    333,
    96,
    512,
    64,
    1024,
    8,
    250,
    200,
    600,
    40,
    128,
    300,
) * 4


def _bf16_state_key(arch, num_q_heads, num_v_heads, lens, *, state_dtype="bfloat16"):
    return dict(
        arch=arch,
        io_dtype="bfloat16",
        state_dtype=state_dtype,
        num_seqs=len(lens),
        total_seq_len=sum(lens),
        max_seq_len=max(lens),
        num_q_heads=num_q_heads,
        num_k_heads=num_q_heads,
        num_v_heads=num_v_heads,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=0,
        use_state_indices=False,
        gates_present=True,
        host_seq_lens=tuple(lens),
    )


@pytest.mark.cpu_only
def test_sm107a_gate_stream_prefetch_twin_follows_the_launch_grid():
    # The gate-stream decision stays on the nominal 212 clusters; the prefetch twin needs a
    # launch grid (the launch's SM count) that gives some CTA a second tile.  The ragged
    # 64-sequence batch launches 212 CTAs on 212 SMs and one CTA per tile on 216 SMs (the
    # ragged grid route), where the gate-stream twin runs without the prefetch.
    ragged = _bf16_state_key("sm_107a", 4, 16, _RAGGED64)
    assert cake_gdn._cake_gdn_prefill_grid_x(1024, 212, False, 1024, _RAGGED64) == 212
    assert cake_gdn._cake_gdn_prefill_grid_x(1024, 216, False, 1024, _RAGGED64) == 1024
    for sm_count, name in (
        (None, _GATESTREAM_BF16_PF_SM107A + "c13e8b260387"),
        (212, _GATESTREAM_BF16_PF_SM107A + "c13e8b260387"),
        (216, _GATESTREAM_BF16_SM107A + "c13e8b260387"),
    ):
        route = cake_gdn.select_cake_gdn_prefill_variant(**ragged, sm_count=sm_count)
        assert route == cake_gdn.CakeGDNRoute(
            "flashinfer.gdn_prefill.noncp.full_dv", name, True, False
        ), sm_count
    # Persistent on both counts (2048 tiles on 212 / 216 CTAs): the prefetch twin.
    persistent = _bf16_state_key("sm_107a", 8, 32, (256,) * 64)
    for sm_count in (None, 212, 216):
        route = cake_gdn.select_cake_gdn_prefill_variant(
            **persistent, sm_count=sm_count
        )
        assert route.variant_name == _GATESTREAM_BF16_PF_SM107A + "aeb81abb92c2"


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "key",
    (
        # sm_107a without the gate-stream twin: a 1024-token mean (base body) and one tile a
        # CTA (128 tiles on 128 CTAs).
        _bf16_state_key("sm_107a", 8, 32, (1024,) * 8, state_dtype="float32"),
        _bf16_state_key("sm_107a", 2, 8, (512,) * 16),
        # The gate-stream twin of the other architectures (no prefetch twin).
        _bf16_state_key("sm_100a", 16, 32, (256,) * 32, state_dtype="float32"),
        _bf16_state_key("sm_103a", 16, 32, (256,) * 32, state_dtype="float32"),
    ),
    ids=(
        "sm107a_long",
        "sm107a_one_tile_per_cta",
        "sm100a_gatestream",
        "sm103a_gatestream",
    ),
)
def test_sm_count_leaves_every_other_route_unchanged(key):
    routes = {
        sm_count: cake_gdn.select_cake_gdn_prefill_variant(**key, sm_count=sm_count)
        for sm_count in (None, 148, 160, 212, 216)
    }
    assert len(set(routes.values())) == 1, routes
    name = routes[None].variant_name
    assert "_pf_" not in name
    assert ("_gatestream_full" in name) is (key["arch"] != "sm_107a")


_F16IO_SM107A = "prefill_flashinfer_blackwell_gdn_prefill_initial_f16io_"
_F16IO_BASE_SM107A = _F16IO_SM107A + "sm107a_i64meta_"
_GATESTREAM_F16IO_SM107A = _F16IO_SM107A + "gatestream_full_sm107a_i64meta_"
_GATESTREAM_F16IO_PF_SM107A = _F16IO_SM107A + "gatestream_full_pf_sm107a_i64meta_"


def _f16_io_key(arch, num_q_heads, num_v_heads, lens):
    return dict(
        _bf16_state_key(arch, num_q_heads, num_v_heads, lens, state_dtype="float32"),
        io_dtype="float16",
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("num_q_heads", "num_v_heads", "lens", "suffix", "base_suffix"),
    (
        # The seven fixed-head FP16-I/O FP32-state keys with several short full-DV tiles per
        # CTA (2048, 2048, 1024, 512, 1024, 512 and 256 tiles; the last on 128 CTAs).
        (8, 32, (256,) * 64, "3400019c1c9d", "0a518d13901e"),
        (16, 32, (256,) * 64, "4b324091b0c8", "31dac57f676c"),
        (4, 16, (256,) * 64, "892e1bbebce0", "4806529ee52a"),
        (2, 8, (256,) * 64, "1e9b588b8066", "6b1531103546"),
        (16, 64, (512,) * 16, "926c590d59ab", "723b365a5ec7"),
        (32, 32, (512,) * 16, "a7c45dca3f9d", "7dadd112e990"),
        (16, 16, (512,) * 16, "de5a927b5f1c", "6fe2e96200a2"),
    ),
)
def test_sm107a_f16io_gate_stream_twin_and_its_prefetch_twin(
    num_q_heads, num_v_heads, lens, suffix, base_suffix
):
    # The FP16-I/O full-DV body takes its gate-stream twin under the same rule as the
    # BF16-I/O bodies (nominal 212 clusters), as its prefetch twin when the launch grid
    # gives some CTA a second tile, else the twin without the prefetch.
    key = _f16_io_key("sm_107a", num_q_heads, num_v_heads, lens)
    tiles = len(lens) * max(num_q_heads, num_v_heads)
    for sm_count in (None, 212, 216):
        route = cake_gdn.select_cake_gdn_prefill_variant(**key, sm_count=sm_count)
        assert route == cake_gdn.CakeGDNRoute(
            "flashinfer.gdn_prefill.noncp.full_dv",
            _GATESTREAM_F16IO_PF_SM107A + suffix,
            True,
            False,
        ), sm_count
    one_cta_per_tile = cake_gdn.select_cake_gdn_prefill_variant(**key, sm_count=tiles)
    assert one_cta_per_tile.variant_name == _GATESTREAM_F16IO_SM107A + suffix
    records = {r["name"]: r for r in cake_gdn._manifest()["variants"]}
    base = records[_F16IO_BASE_SM107A + base_suffix]
    twin = records[_GATESTREAM_F16IO_SM107A + suffix]
    prefetch = records[_GATESTREAM_F16IO_PF_SM107A + suffix]
    for record in (twin, prefetch):
        assert record["architectures"] == ["sm_107a"]
        for field in ("specializations", "metadata_i64", "tma_abi"):
            assert record[field] == base[field], field
        for field in ("params", "template"):
            assert record["host_binding"][field] == base["host_binding"][field], field
    assert (
        prefetch["host_binding"]["cuda_symbol"]
        == twin["host_binding"]["cuda_symbol"] + "_pf"
    )
    assert twin["outputs"] != base["outputs"] != prefetch["outputs"] != twin["outputs"]
    for prefix, record in (
        (_GATESTREAM_F16IO_SM107A, twin),
        (_GATESTREAM_F16IO_PF_SM107A, prefetch),
    ):
        shipped = {
            (r["host_binding"]["cuda_symbol"], json.dumps(r["outputs"]))
            for name, r in records.items()
            if name.startswith(prefix)
        }
        assert shipped == {
            (record["host_binding"]["cuda_symbol"], json.dumps(record["outputs"]))
        }
    # Long sequences keep the base body (mean length above 512 tokens).
    long = cake_gdn.select_cake_gdn_prefill_variant(
        **_f16_io_key("sm_107a", num_q_heads, num_v_heads, (2048,) * len(lens))
    )
    assert long.variant_name == base["name"]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "key",
    (
        # One tile a CTA (128 tiles on 128 CTAs): no twin.
        _f16_io_key("sm_107a", 2, 8, (512,) * 16),
        # The other architectures ship no FP16-I/O gate-stream twin: the base record.
        _f16_io_key("sm_100a", 8, 32, (256,) * 64),
        _f16_io_key("sm_103a", 8, 32, (256,) * 64),
        _f16_io_key("sm_100a", 16, 32, (256,) * 32),
        _f16_io_key("sm_103a", 16, 32, (256,) * 32),
    ),
    ids=(
        "sm107a_one_tile_per_cta",
        "sm100a_hq8_hv32",
        "sm103a_hq8_hv32",
        "sm100a_hq16_hv32",
        "sm103a_hq16_hv32",
    ),
)
def test_f16io_keys_without_a_shipped_gate_stream_twin_keep_their_base(key):
    routes = {
        sm_count: cake_gdn.select_cake_gdn_prefill_variant(**key, sm_count=sm_count)
        for sm_count in (None, 148, 160, 212, 216)
    }
    assert len(set(routes.values())) == 1, routes
    name = routes[None].variant_name
    assert "_gatestream_full" not in name, name
    record = cake_gdn._kernel_record(name)
    schedule = record["source_schedule"].removeprefix(PREFIX)
    assert schedule.startswith("flashinfer_blackwell_gdn_prefill_initial_f16io"), (
        schedule
    )


@pytest.mark.cpu_only
def test_sm107a_bf16_state_hq2_hv8_gate_stream_twin_and_its_prefetch_twin():
    # The Qwen3.5 TP8 BF16-state split (Hq2/Hv8) with several short full-DV tiles per CTA
    # (64 x 256 tokens: 512 tiles) takes the BF16-state gate-stream twin, as its prefetch
    # twin when the launch grid gives some CTA a second tile, else the twin without it.
    key = _bf16_state_key("sm_107a", 2, 8, (256,) * 64)
    for sm_count in (None, 212, 216):
        route = cake_gdn.select_cake_gdn_prefill_variant(**key, sm_count=sm_count)
        assert route == cake_gdn.CakeGDNRoute(
            "flashinfer.gdn_prefill.noncp.full_dv",
            _GATESTREAM_BF16_PF_SM107A + "d28a977c404a",
            True,
            False,
        ), sm_count
    one_cta_per_tile = cake_gdn.select_cake_gdn_prefill_variant(**key, sm_count=512)
    assert one_cta_per_tile.variant_name == _GATESTREAM_BF16_SM107A + "d28a977c404a"
    records = {r["name"]: r for r in cake_gdn._manifest()["variants"]}
    base = records[_BF16STATE_SM107A + "sm107a_i64meta_d5225057d28d"]
    for prefix in (_GATESTREAM_BF16_SM107A, _GATESTREAM_BF16_PF_SM107A):
        record = records[prefix + "d28a977c404a"]
        for field in ("specializations", "metadata_i64", "tma_abi"):
            assert record[field] == base[field], field
        for field in ("params", "template"):
            assert record["host_binding"][field] == base["host_binding"][field], field
        # The gate-stream program's own body and unit, shared with its other records.
        shipped = {
            (r["host_binding"]["cuda_symbol"], json.dumps(r["outputs"]))
            for name, r in records.items()
            if name.startswith(prefix)
        }
        assert shipped == {
            (record["host_binding"]["cuda_symbol"], json.dumps(record["outputs"]))
        }
    # Long sequences (mean above 512 tokens) leave the gate-stream twins: up to 2048 tokens per
    # sequence the key's vec2-full twin of the base record, above that the base record itself.
    long = cake_gdn.select_cake_gdn_prefill_variant(
        **_bf16_state_key("sm_107a", 2, 8, (2048,) * 64)
    )
    assert long.variant_name == base["name"].replace(
        "_sm107a_i64meta_", "_sm107a_vec2full_i64meta_"
    )
    longer = cake_gdn.select_cake_gdn_prefill_variant(
        **_bf16_state_key("sm_107a", 2, 8, (8192,) * 64)
    )
    assert longer.variant_name == base["name"]
