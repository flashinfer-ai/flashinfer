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
