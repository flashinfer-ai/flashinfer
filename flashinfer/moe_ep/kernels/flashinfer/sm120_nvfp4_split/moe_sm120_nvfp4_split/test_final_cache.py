"""Hot-path and cache-isolation regressions; synthetic host data only."""
from dataclasses import asdict, replace
from unittest.mock import patch

from . import heuristic as h
from . import model_policy, tuning_config as tc
from .api import MegaMoEProblemSpec, select_compile_spec
from .test_tuning_config import shape


def test_final_rule_is_constructed_and_validated_once():
    h._select_with_profiles.cache_clear()
    s = shape(16384)
    with patch.object(model_policy, 'select', wraps=model_policy.select) as policy, \
         patch.object(model_policy, 'config_from_dict', wraps=tc.config_from_dict) as validate:
        first = h.select_megamoe_config(s, numerical_contract='owner_bf16')
        for _ in range(20):
            assert h.select_megamoe_config(s, numerical_contract='owner_bf16') is first
        assert first.k2_stages == 3
        assert policy.call_count == validate.call_count == 1


def test_cached_diagnostics_cannot_be_mutated_through_explanation():
    s = shape(16384)
    first = h.explain_selection(s, numerical_contract='owner_bf16')
    first['profiles']['builtin'].clear()
    second = h.explain_selection(s, numerical_contract='owner_bf16')
    assert second['profiles']['builtin']['matched']
    second['profiles']['builtin']['evidence'].clear()
    assert h.explain_selection(s, numerical_contract='owner_bf16')['profiles']['builtin']['evidence']


def test_offline_content_and_hardware_separate_final_cache_entries():
    s = shape()
    base = h.select_megamoe_config(s, numerical_contract='owner_bf16')
    tuned = replace(base, k1_tile=(64, 64, 128), k2_tile=(64, 64, 128), k1_compact_tokens=False, k2_compact_tokens=False)
    entry = dict(shape=asdict(s), numerical_contract='owner_bf16', config=asdict(tuned),
                 validation='qualified', evidence={'fixture': 'synthetic'})
    doc = tc.make_document('host-test', [entry], {})
    profile = tc.parse_tuning_config(doc)
    kw = dict(numerical_contract='owner_bf16', tuning_config=profile, hardware='host-test')
    assert h.select_megamoe_config(s, **kw) == tuned
    assert h.select_megamoe_config(s, **dict(kw, hardware='different')) == base
    entry['config'] = asdict(base)
    other = tc.parse_tuning_config(tc.make_document('host-test', [entry], {}))
    assert h.select_megamoe_config(s, **dict(kw, tuning_config=other)) == base
    with patch.object(model_policy, 'select', side_effect=AssertionError('warm path')):
        assert h.select_megamoe_config(s, **kw) == tuned


def test_contract_override_and_counts_separate_cache_entries():
    s = shape(16384)
    owner = h.select_megamoe_config(s, numerical_contract='owner_bf16')
    ordinary = h.select_megamoe_config(s)
    assert not ordinary.rank_local_combine
    o = h.MegaMoEHeuristicOverrides(k2_stages=2)
    assert h.select_megamoe_config(s, o, numerical_contract='owner_bf16').k2_stages == 2
    q, r = divmod(s.tokens_per_rank * 4 * 8, 288)
    counts = (q + 1,) * r + (q,) * (288 - r)
    assert h.select_megamoe_config(s, numerical_contract='owner_bf16', host_expert_counts=counts) == h._select_cached(s, None, 'owner_bf16', counts)
    assert h.select_megamoe_config(s, numerical_contract='owner_bf16') is owner


def test_compile_spec_reuse_and_rank_isolation():
    p = MegaMoEProblemSpec(tokens_per_rank=16384, hidden=4096, intermediate=4096,
                          num_topk=8, num_total_experts=288, expert_parallel_size=4,
                          expert_parallel_rank=0)
    kw = dict(problem=p, num_sms=110, sm_min_partition=8, sm_partition_alignment=8,
              ep_same_numa_peer_count=3, ep_cross_numa_peer_count=0,
              numerical_contract='owner_bf16')
    select_compile_spec.cache_clear()
    first = select_compile_spec(**kw)
    with patch.object(MegaMoEProblemSpec, 'validate', side_effect=AssertionError('warm validation')):
        assert select_compile_spec(**kw) is first
    other = select_compile_spec(**dict(kw, problem=replace(p, expert_parallel_rank=1)))
    assert other.problem.expert_parallel_rank == 1
    assert other is not first
    assert h._select_with_profiles.cache_info().maxsize == 256
    assert select_compile_spec.cache_info().maxsize == 256


def test_hardware_identity_separates_compiler_stacks():
    import json
    from types import SimpleNamespace
    props = SimpleNamespace(name='test-device', major=12, minor=0,
                            multi_processor_count=110, total_memory=1)
    packages = [SimpleNamespace(metadata={'Name': 'nvidia-cutlass-dsl'}),
                SimpleNamespace(metadata={'Name': 'nvidia-cutlass-dsl-libs-cu13'}),
                SimpleNamespace(metadata={'Name': 'irrelevant-package'})]
    with patch('torch.cuda.get_device_properties', return_value=props), \
         patch('importlib.metadata.distributions', return_value=packages), \
         patch('importlib.metadata.version', return_value='4.8.0'):
        first = tc.hardware_identity()
    with patch('torch.cuda.get_device_properties', return_value=props), \
         patch('importlib.metadata.distributions', return_value=packages), \
         patch('importlib.metadata.version', return_value='4.9.0'):
        second = tc.hardware_identity()
    assert first != second
    assert json.loads(first)['compiler_libraries'] == {
        'nvidia-cutlass-dsl': '4.8.0', 'nvidia-cutlass-dsl-libs-cu13': '4.8.0'}
