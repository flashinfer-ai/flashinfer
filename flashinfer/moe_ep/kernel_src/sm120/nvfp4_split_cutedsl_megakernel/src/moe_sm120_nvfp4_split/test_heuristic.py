"""Host regression tests for candidate selection and numerical permission.

The configuration fixtures come from completed GPU validation runs, rather
than recomputing expected configurations from the selector under test.
"""
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from .heuristic import (
    MegaMoEHeuristicInput, MegaMoEHeuristicOverrides, derive_features,
    enumerate_candidates, explain_selection, legality_errors, semantic_errors,
    select_megamoe_config, select_compatibility_config,
)

CASES = json.loads((Path(__file__).parent/'selector_regression_cases.json').read_text())


def shape(tokens=8192, **updates):
    values = dict(tokens_per_rank=tokens, hidden=4096, intermediate=4096,
                  num_topk=6, num_total_experts=256, data_parallel_size=1,
                  tensor_parallel_size=1, expert_parallel_size=4,
                  ep_same_numa_peer_count=3, ep_cross_numa_peer_count=0,
                  num_sms=110, sm_min_partition=8, sm_partition_alignment=8)
    return MegaMoEHeuristicInput(**dict(values, **updates))


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['name'])
def test_selected_configuration_matches_measured_fixture(case):
    s = MegaMoEHeuristicInput(**case['shape'])
    c = select_megamoe_config(s, MegaMoEHeuristicOverrides(k1_compact_tokens=False, k2_compact_tokens=False), numerical_contract='owner_bf16')
    assert json.loads(json.dumps(asdict(c))) == dict(case['config'], k1_compact_tokens=False, k2_compact_tokens=False)
    assert not legality_errors(s, c)
    assert not semantic_errors(c, 'owner_bf16')


@pytest.mark.parametrize('tokens', [16, 512, 1024, 8192, 24576])
def test_default_contract_cannot_change_bf16_association(tokens):
    s = shape(tokens)
    assert not select_megamoe_config(s).rank_local_combine
    assert all(not c.config.rank_local_combine for c in enumerate_candidates(s))
    with pytest.raises(ValueError):
        select_megamoe_config(s, MegaMoEHeuristicOverrides(rank_local_combine=True))


def test_owner_contract_allows_rlc_but_opt_out_is_authoritative():
    s = shape()
    assert select_megamoe_config(s, numerical_contract='owner_bf16').rank_local_combine
    c = select_megamoe_config(s, MegaMoEHeuristicOverrides(rank_local_combine=False),
                             numerical_contract='owner_bf16')
    assert not c.rank_local_combine


@pytest.mark.parametrize('overrides', [
    MegaMoEHeuristicOverrides(ready_queue_bundle=16),
    MegaMoEHeuristicOverrides(k2_stages=2),
    MegaMoEHeuristicOverrides(dispatch_rank_cache=False),
    MegaMoEHeuristicOverrides(dispatch_warps=4),
    MegaMoEHeuristicOverrides(k1_sms=80, k2_sms=30, tx_sms=0, rx_sms=0),
])
def test_explicit_fields_survive_both_selection_layers(overrides):
    c = select_megamoe_config(shape(), overrides, numerical_contract='owner_bf16')
    for key, value in asdict(overrides).items():
        if value is not None:
            assert getattr(c, key) == value


def test_redundant_backend_override_is_idempotent():
    s = shape()
    a = select_megamoe_config(s, numerical_contract='owner_bf16')
    b = select_megamoe_config(s, MegaMoEHeuristicOverrides(comm_backend='p2p_direct'),
                             numerical_contract='owner_bf16')
    assert a == b


@pytest.mark.parametrize('updates', [
    dict(tokens_per_rank=24577), dict(num_sms=188),
    dict(data_parallel_size=2), dict(tensor_parallel_size=2),
    dict(expert_parallel_size=8, ep_same_numa_peer_count=3, ep_cross_numa_peer_count=4),
])
def test_uncalibrated_shapes_use_compatibility_path(updates):
    s = replace(shape(), **updates)
    assert enumerate_candidates(s) == ()
    assert select_megamoe_config(s) == select_compatibility_config(s)
    assert not select_megamoe_config(s).rank_local_combine
    assert explain_selection(s)['mode'] == 'legacy_compatibility_fallback'


@pytest.mark.parametrize('contract', ['invalid', '', None])
def test_unknown_numerical_contract_rejected(contract):
    with pytest.raises(ValueError):
        select_megamoe_config(shape(), numerical_contract=contract)


def test_host_counts_obey_route_conservation_and_worst_owner_work():
    s = shape(); c = select_megamoe_config(s, numerical_contract='owner_bf16')
    counts = (3072,)*64 + (0,)*192
    assert sum(counts) == s.tokens_per_rank*s.expert_parallel_size*s.num_topk
    f = derive_features(s, c, counts)
    assert f.fc2_tiles == 64*((3072+c.k2_tile[1]-1)//c.k2_tile[1])
    assert f.fc2_tiles > derive_features(s, c).fc2_tiles
    info = explain_selection(s, numerical_contract='owner_bf16', host_expert_counts=counts)
    assert info['route_statistics'] == 'caller_host_counts'
    assert not info['robust_policy']['changed']
    for invalid in [counts[:-1], (-1,)+counts[1:], (0,)*256]:
        with pytest.raises(ValueError):
            select_megamoe_config(s, host_expert_counts=invalid)


def test_large_policy_is_scoped_and_reports_predicted_risk():
    info = explain_selection(shape(), numerical_contract='owner_bf16')
    policy = info['robust_policy']
    assert policy['mode'] == 'large_joint_stage_queue' and policy['changed']
    assert policy['mean_log_p50'] < -.005
    assert policy['worst_log_p50'] <= .005 and policy['worst_log_p90'] <= .01
    assert info['selected']['k2_stages'] == 3 and info['selected']['ready_queue_bundle'] == 8
    wide = explain_selection(shape(hidden=7168, intermediate=6144, num_total_experts=384),
                             numerical_contract='owner_bf16')
    assert not wide['robust_policy']['changed']


def test_partial_sm_override_and_unqualified_cross_numa_rejected():
    with pytest.raises(ValueError):
        select_megamoe_config(shape(), MegaMoEHeuristicOverrides(k1_sms=80))
    s = shape(ep_same_numa_peer_count=1, ep_cross_numa_peer_count=2)
    with pytest.raises(ValueError, match='cross-NUMA'):
        select_megamoe_config(s, MegaMoEHeuristicOverrides(
            comm_backend='p2p_direct', dispatch_warps=4, dispatch_rank_cache=False))
