"""Route table of the Cake GDN non-CP prefill resolver: the sm_107a sequential twins."""

import json

import pytest

from flashinfer.jit import cake_gdn

PREFIX = "flashinfer.cake_gdn.generated.prefill:"
TABLE = cake_gdn.CAKE_GDN_PREFILL_SEQ_ARCH_BODIES
ROUTED = sorted(
    {
        schedule.removesuffix("_indexed")
        for (arch, schedule) in TABLE
        if arch == "sm_107a"
    }
)


@pytest.mark.parametrize("schedule", ROUTED)
def test_sm107a_routes_the_vec2_state_io_twins_per_shipped_schedule(schedule):
    body = TABLE[("sm_107a", schedule)]
    assert body in ("l7_vec2", "l7b_vec2lp", "st2s_vec2lp")
    twin = cake_gdn.cake_gdn_prefill_seq_schedule(
        "sm_107a", schedule, use_state_indices=False
    )
    assert twin == f"{schedule}_sm107a_{body}"
    indexed_body = TABLE.get(("sm_107a", f"{schedule}_indexed"), "base")
    indexed = cake_gdn.cake_gdn_prefill_seq_schedule(
        "sm_107a", schedule, use_state_indices=True
    )
    expected = (
        schedule if indexed_body == "base" else f"{schedule}_sm107a_{indexed_body}"
    )
    assert indexed == expected
    for arch in ("sm_100a", "sm_103a"):
        for use_state_indices in (False, True):
            shipped = cake_gdn.cake_gdn_prefill_seq_schedule(
                arch, schedule, use_state_indices=use_state_indices
            )
            assert shipped == schedule


def test_every_shipped_sm107a_record_of_a_routed_schedule_has_its_twin_record():
    variants = cake_gdn._manifest()["variants"]
    by_source: dict = {}
    for record in variants:
        key = (
            record["source_schedule"],
            json.dumps(record["specializations"], sort_keys=True),
        )
        by_source.setdefault(key, []).append(record)
    checked = 0
    for record in variants:
        if record["domain"] != "prefill" or "sm_107a" not in record["architectures"]:
            continue
        schedule = record["source_schedule"].removeprefix(PREFIX)
        if schedule not in ROUTED:
            continue
        twin = cake_gdn.cake_gdn_prefill_seq_schedule(
            "sm_107a",
            schedule,
            use_state_indices=bool(record["specializations"]["USE_STATE_INDICES"]),
        )
        if twin == schedule:
            continue
        key = (PREFIX + twin, json.dumps(record["specializations"], sort_keys=True))
        assert key in by_source, f"no sm_107a twin record for {record['name']}"
        (twin_record,) = by_source[key]
        assert twin_record["architectures"] == ["sm_107a"]
        assert twin_record["threads"] == record["threads"]
        assert twin_record["tma_abi"] == record["tma_abi"] == "grid_constant"
        checked += 1
    assert checked > 0
