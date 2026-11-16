"""Policy safety and cache tests; fixture profiles are not production evidence."""
from dataclasses import asdict, replace
import json
import pytest
from . import utility_policy
from .api import MegaMoEProblemSpec, MegaMoECompileSpec, SplitKernelBuildOptions, select_compile_spec
from .jit_config import Sm120JitConfig
from .heuristic import MegaMoEHeuristicInput, MegaMoEHeuristicOverrides


def args(t=32,ep=4,k=6,e=256):
    return dict(problem=MegaMoEProblemSpec(t,k,e,4096,4096,ep,0,gate_up_clamp=10),
                ep_same_numa_peer_count=3,ep_cross_numa_peer_count=ep-4,
                num_sms=110,sm_min_partition=8,sm_partition_alignment=8,
                numerical_contract='owner_bf16')


def profile(a,c):
    p=a['problem']
    shape=MegaMoEHeuristicInput(tokens_per_rank=p.tokens_per_rank,hidden=p.hidden,
        intermediate=p.intermediate,num_topk=p.num_topk,num_total_experts=p.num_total_experts,
        data_parallel_size=p.data_parallel_size,tensor_parallel_size=p.tensor_parallel_size,
        expert_parallel_size=p.expert_parallel_size,ep_same_numa_peer_count=a['ep_same_numa_peer_count'],
        ep_cross_numa_peer_count=a['ep_cross_numa_peer_count'],num_sms=a['num_sms'],
        sm_min_partition=a['sm_min_partition'],sm_partition_alignment=a['sm_partition_alignment'])
    return {'key':json.loads(json.dumps(dict(shape=asdict(shape),config=asdict(c),contract=a['numerical_contract'])))}

@pytest.fixture
def qualified(monkeypatch):
    # This fixture tests the exact-profile utility layer in isolation. Continuous
    # decode defaults have separate interval and API/native parity validation.
    from . import decode_policy
    monkeypatch.setattr(decode_policy, 'GEOMETRY_RULES', ())
    monkeypatch.setattr(decode_policy, 'CACHE_RULES', ())
    select_compile_spec.cache_clear()
    a=args();off=select_compile_spec(**a,jit=Sm120JitConfig(fc1_utility_postprocess=False))
    monkeypatch.setattr(utility_policy,'PROFILES',(profile(a,off.kernel),))
    select_compile_spec.cache_clear()
    yield a,off
    select_compile_spec.cache_clear()


def test_default_resolves_to_bool_with_distinct_cache(qualified):
    a,off=qualified
    auto=select_compile_spec(**a)
    on=select_compile_spec(**a,jit=Sm120JitConfig(fc1_utility_postprocess=True))
    assert auto.kernel==off.kernel
    assert auto.jit.fc1_utility_postprocess is True
    assert auto.cache_key==on.cache_key!=off.cache_key
    assert auto.canonical_dict()['jit']['fc1_utility_postprocess'] is True
    direct=MegaMoECompileSpec(problem=off.problem,kernel=off.kernel)
    assert direct.jit.fc1_utility_postprocess is False

@pytest.mark.parametrize('change', ['neighbor','contract','topology','stages','build','trace'])
def test_unqualified_context_falls_back(qualified,change):
    a,_=qualified
    if change=='neighbor':a['problem']=replace(a['problem'],tokens_per_rank=33)
    if change=='contract':a['numerical_contract']='ordinary_topk'
    if change=='topology':a.update(ep_same_numa_peer_count=1,ep_cross_numa_peer_count=2)
    if change=='stages':a['overrides']=MegaMoEHeuristicOverrides(k1_stages=2)
    if change=='build':a['build']=SplitKernelBuildOptions(group_hint=40)
    if change=='trace':a['jit']=Sm120JitConfig(enable_globaltimer=True)
    s=select_compile_spec(**a)
    assert s.jit.fc1_utility_postprocess is False

@pytest.mark.parametrize('t', [287,606,512,1024,2048,8192,24576])
def test_large_and_compact_configs_keep_original_geometry(t):
    a=args(t=t)
    auto=select_compile_spec(**a)
    off=select_compile_spec(**a,jit=Sm120JitConfig(fc1_utility_postprocess=False))
    assert auto.kernel==off.kernel
    assert auto.jit.fc1_utility_postprocess is False
    with pytest.raises(ValueError,match='noncompact N16/N32'):
        select_compile_spec(**a,jit=Sm120JitConfig(fc1_utility_postprocess=True))

@pytest.mark.parametrize('value', ['1','0','on','off','true','false'])
def test_environment_override(monkeypatch,value):
    monkeypatch.setenv('MEGA_FC1_UTILITY_POSTPROCESS',value)
    assert Sm120JitConfig.from_environment().fc1_utility_postprocess is (value in ('1','on','true'))

def test_missing_env_means_auto(monkeypatch):
    monkeypatch.delenv('MEGA_FC1_UTILITY_POSTPROCESS',raising=False)
    assert Sm120JitConfig.from_environment().fc1_utility_postprocess is None
    with pytest.raises(ValueError):Sm120JitConfig(fc1_utility_postprocess='on')


def test_all_ranks_share_policy_but_have_distinct_cache_keys(qualified):
    a,_=qualified
    keys=set()
    for rank in range(4):
        a['problem']=replace(a['problem'],expert_parallel_rank=rank)
        spec=select_compile_spec(**a)
        assert spec.jit.fc1_utility_postprocess is True
        keys.add(spec.cache_key)
    assert len(keys)==4


def test_unqualified_clamp_falls_back(qualified):
    a,_=qualified
    a['problem']=replace(a['problem'],gate_up_clamp=8.0)
    assert select_compile_spec(**a).jit.fc1_utility_postprocess is False
