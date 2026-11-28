from dataclasses import replace
from unittest.mock import patch

from . import tuned_table
from .heuristic import MegaMoEHeuristicInput, MegaMoEHeuristicOverrides, select_megamoe_config

TABLE = dict(schema=1, tables=[
    dict(model="glm53", signature=[4096, 4096, 8, 288], topology="ep4_1numa", entries=[
        dict(tokens=16, knobs=dict(k1_ctas_per_sm=2, k2_stages=3, fc1_utility_postprocess="on")),
        dict(tokens=512, knobs=dict()),
    ]),
    dict(model="glm53", signature=[4096, 4096, 8, 288], topology="ep8_2numa", entries=[
        dict(tokens=16, knobs=dict(backend="p2p_split")),
        dict(tokens=2048, knobs=dict(backend="gin_hybrid")),
    ]),
])


def shape(tokens, ep=4, same=3, cross=0):
    return MegaMoEHeuristicInput(
        tokens_per_rank=tokens, hidden=4096, intermediate=4096, num_topk=8, num_total_experts=288,
        data_parallel_size=1, tensor_parallel_size=1, expert_parallel_size=ep,
        ep_same_numa_peer_count=same, ep_cross_numa_peer_count=cross, num_sms=110,
        sm_min_partition=8, sm_partition_alignment=8)


def test_nearest_anchor_and_utility():
    with patch.object(tuned_table, "_load", return_value=TABLE):
        assert tuned_table.lookup(shape(90))["tokens"] == 16
        assert tuned_table.lookup(shape(91))["tokens"] == 512
        assert tuned_table.utility_for(shape(32)) is True
        assert tuned_table.utility_for(shape(4096)) is None


def test_ep8_backend_selection_and_folded_view():
    with patch.object(tuned_table, "_load", return_value=TABLE):
        real = shape(32, ep=8, same=3, cross=4)
        assert tuned_table.preferred_ep8_backend(real) == "p2p_split"
        assert tuned_table.preferred_ep8_backend(replace(real, tokens_per_rank=4096)) == "gin_hybrid"
        folded = shape(32, ep=8, same=7, cross=0)  # p2p_split view of the same EP
        assert tuned_table.topology_of(folded) == "ep8_2numa"
        assert tuned_table.lookup(folded)["tokens"] == 16
        assert tuned_table.lookup(real) is None  # hybrid view, entry tuned for p2p_split
        assert tuned_table.preferred_ep8_backend(folded) is None


def test_rule_without_table():
    import os
    from .runtime import driver_caps

    x = tuned_table.SPLIT_RULE_MAX_TOKENS
    driver_caps.hybrid_sm_split_available.cache_clear()
    try:
        with patch.object(tuned_table, "_load", return_value=dict(schema=1, tables=[])), \
                patch.dict(os.environ, {driver_caps.HYBRID_SM_SPLIT_ENV: "1"}):
            real = shape(64, ep=8, same=3, cross=4)
            assert tuned_table.preferred_ep8_backend(real) == "p2p_split"
            assert tuned_table.preferred_ep8_backend(replace(real, tokens_per_rank=x)) == "p2p_split"
            assert tuned_table.preferred_ep8_backend(replace(real, tokens_per_rank=x + 1)) == "gin_hybrid"
    finally:
        driver_caps.hybrid_sm_split_available.cache_clear()


def test_overrides_mapping():
    o = tuned_table.overrides_for(dict(backend="gin_hybrid", k1_n=16, k2_n=32, k2_warps=8, sms=[72, 38],
                                       k2_tail_reclaim="off", dispatch_rank_cache="on",
                                       k2_ready_queue_bundle=2))
    assert o == dict(comm_backend="gin_hybrid", k1_tile=(64, 16, 128), k2_tile=(64, 32, 128), k2_warps=8,
                     k1_sms=72, tx_sms=0, rx_sms=0, k2_sms=38, k2_tail_reclaim=False,
                     dispatch_rank_cache=True, ready_queue_bundle=2)


def test_table_knobs_reach_the_selection():
    with patch.object(tuned_table, "_load", return_value=TABLE):
        c = select_megamoe_config(shape(16))
        assert (c.k1_ctas_per_sm, c.dispatch_warps, c.k2_stages) == (2, 1, 3)
        explicit = select_megamoe_config(shape(16), MegaMoEHeuristicOverrides(k1_ctas_per_sm=1))
        assert explicit.k1_ctas_per_sm == 1


def _shipped_cases():
    for table in tuned_table._load()["tables"]:
        h, i, k, e = table["signature"]
        for entry in table["entries"]:
            ep, real = {"ep4_1numa": (4, (3, 0)), "ep8_2numa": (8, (3, 4)),
                        "ep16_2x2numa": (16, (3, 12))}[table["topology"]]
            split = entry["knobs"].get("backend") == "p2p_split"
            same, cross = (ep - 1, 0) if split else real
            yield MegaMoEHeuristicInput(
                tokens_per_rank=entry["tokens"], hidden=h, intermediate=i, num_topk=k, num_total_experts=e,
                data_parallel_size=1, tensor_parallel_size=1, expert_parallel_size=ep,
                ep_same_numa_peer_count=same, ep_cross_numa_peer_count=cross, num_sms=110,
                sm_min_partition=8, sm_partition_alignment=8), entry


def test_shipped_table_entries_select_legal_configs():
    from .heuristic import legality_errors, semantic_errors
    cases = list(_shipped_cases())
    for s, entry in cases:
        assert tuned_table.lookup(s) is entry
        c = select_megamoe_config(s)
        assert not legality_errors(s, c), (s.tokens_per_rank, entry)
        assert not semantic_errors(c, "ordinary_topk"), (s.tokens_per_rank, entry)
        for key, value in tuned_table.overrides_for(entry["knobs"]).items():
            if key != "comm_backend":
                assert getattr(c, key) == value, (s.tokens_per_rank, key, entry)


def test_ep16_two_host_topology_uses_the_split_rule():
    with patch.object(tuned_table, "_load", return_value=TABLE):
        real = shape(64, ep=16, same=3, cross=12)
        assert tuned_table.topology_of(real) == "ep16_2x2numa"
        assert tuned_table.preferred_ep8_backend(real) == "p2p_split"
        folded = shape(64, ep=16, same=15, cross=0)
        assert tuned_table.topology_of(folded) == "ep16_2x2numa"
        assert tuned_table.preferred_ep8_backend(folded) is None
        assert tuned_table.lookup(folded) is None  # nothing tabulated for EP16
        assert tuned_table.topology_of(shape(64, ep=16, same=7, cross=8)) is None


def test_every_shipped_entry_resolves_and_passes_the_launch_guard():
    """Shipped anchors (this module sees the real table): every entry resolves
    through the selection and the transport overlay (which runs
    check_launch_safety) without error."""
    from .heuristic import select_transport_overlay

    folded = {"ep4_1numa": (4, 3, 0), "ep8_2numa": (8, 7, 0), "ep16_2x2numa": (16, 15, 0)}
    real = {"ep8_2numa": (8, 3, 4)}
    count = 0
    for table in tuned_table._load()["tables"]:
        h, i, k, e = table["signature"]
        for entry in table["entries"]:
            backend = entry["knobs"].get("backend")
            ep, same, cross = (real[table["topology"]] if backend == "gin_hybrid"
                               else folded[table["topology"]])
            shp = MegaMoEHeuristicInput(
                tokens_per_rank=entry["tokens"], hidden=h, intermediate=i, num_topk=k, num_total_experts=e,
                data_parallel_size=1, tensor_parallel_size=1, expert_parallel_size=ep,
                ep_same_numa_peer_count=same, ep_cross_numa_peer_count=cross, num_sms=110,
                sm_min_partition=8, sm_partition_alignment=8)
            o = MegaMoEHeuristicOverrides(rank_local_combine=False) if backend == "p2p_split" else None
            select_transport_overlay(shp, select_megamoe_config(shp, o))
            count += 1
    assert count >= 50
