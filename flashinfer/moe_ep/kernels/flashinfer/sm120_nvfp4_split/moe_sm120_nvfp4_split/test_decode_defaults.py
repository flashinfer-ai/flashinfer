"""Safety contracts for continuous defaults, independent of measured speedups."""
from dataclasses import asdict, replace
from unittest.mock import patch
import pytest
from . import api, decode_policy as policy
from .heuristic import MegaMoEHeuristicInput, MegaMoEHeuristicOverrides, MegaMoEKernelConfig
from .jit_config import Sm120JitConfig


def shape(ep=8,model='dsv',t=32):
    return MegaMoEHeuristicInput(tokens_per_rank=t,hidden=4096,intermediate=4096,
        num_topk=6 if model=='dsv' else 8,num_total_experts=256 if model=='dsv' else 288,
        data_parallel_size=1,tensor_parallel_size=1,expert_parallel_size=ep,
        ep_same_numa_peer_count=3,ep_cross_numa_peer_count=ep-4,
        num_sms=110,sm_min_partition=8,sm_partition_alignment=8)


def select(s,**kw):
    p=api.MegaMoEProblemSpec(tokens_per_rank=s.tokens_per_rank,hidden=s.hidden,
        intermediate=s.intermediate,num_topk=s.num_topk,num_total_experts=s.num_total_experts,
        data_parallel_size=s.data_parallel_size,tensor_parallel_size=s.tensor_parallel_size,
        expert_parallel_size=s.expert_parallel_size,expert_parallel_rank=kw.pop('rank',0),gate_up_clamp=10.)
    return api.select_compile_spec(problem=p,ep_same_numa_peer_count=s.ep_same_numa_peer_count,
        ep_cross_numa_peer_count=s.ep_cross_numa_peer_count,num_sms=s.num_sms,
        sm_min_partition=s.sm_min_partition,sm_partition_alignment=s.sm_partition_alignment,
        numerical_contract='owner_bf16',**kw)


def baseline(s):
    with patch.object(policy,'GEOMETRY_RULES',()),patch.object(policy,'CACHE_RULES',()):
        api.select_compile_spec.cache_clear()
        out=select(s)
    api.select_compile_spec.cache_clear()
    return out


@pytest.fixture(autouse=True)
def clean_selector():
    api.select_compile_spec.cache_clear()
    yield
    api.select_compile_spec.cache_clear()


@pytest.mark.parametrize('model,ep',[('dsv',4),('glm',4),('dsv',8),('glm',8)])
def test_explicit_complete_baseline_configuration_wins(model,ep):
    s=shape(ep,model);b=baseline(s)
    fields=MegaMoEHeuristicOverrides.__dataclass_fields__
    override=MegaMoEHeuristicOverrides(**{k:v for k,v in asdict(b.kernel).items() if k in fields})
    got=select(s,overrides=override,jit=Sm120JitConfig(fc1_utility_postprocess=b.jit.fc1_utility_postprocess,
        weight_eviction_policy='default',activation_eviction_policy='default'))
    assert got.kernel==b.kernel
    assert got.jit.fc1_utility_postprocess==b.jit.fc1_utility_postprocess
    assert got.jit.weight_eviction_policy==got.jit.activation_eviction_policy=='default'


@pytest.mark.parametrize('model,ep',[('dsv',4),('glm',4),('dsv',8),('glm',8)])
def test_explicit_cache_flags_independent_and_canonical(model,ep):
    s=shape(ep,model)
    specs=[select(s,jit=Sm120JitConfig(weight_eviction_policy=w,activation_eviction_policy=a))
        for w,a in [('default','default'),('evict_first','default'),('default','evict_last'),('evict_first','evict_last')]]
    assert len({x.cache_key for x in specs})==4
    assert all(x.kernel==specs[0].kernel for x in specs)
    assert [(x.jit.weight_eviction_policy,x.jit.activation_eviction_policy) for x in specs]==[
        ('default','default'),('evict_first','default'),('default','evict_last'),('evict_first','evict_last')]


def test_rule_full_anchor_guards_all_config_fields():
    assert policy.GEOMETRY_RULES
    rule=policy.GEOMETRY_RULES[0];s=shape(rule['ep'],rule['model'],rule['lo'])
    c=MegaMoEKernelConfig(**dict(rule['anchors'][0],expected_rows_per_expert=s.tokens_per_rank*s.expert_parallel_size*s.num_topk/s.num_total_experts))
    for field,value in asdict(c).items():
        if field in ('k1_ctas_per_sm','k2_launch_factor','weight_prefetch_ahead','xnuma_fsplit','k1_t0_prefetch','push_dispatch','xnuma_pack'):continue  # validated at construction
        replacement=(999,999,999) if isinstance(value,tuple) else '__unqualified__' if isinstance(value,str) else 999999
        if isinstance(value,bool):replacement=not value
        altered=replace(c,**{field:replacement})
        # If toggling a boolean happens to be another measured full anchor,
        # this isn't an unqualified mutation and is covered by other anchors.
        if field!='expected_rows_per_expert' and policy._config_key(altered) in rule['anchors']:continue
        with patch.object(policy,'GEOMETRY_RULES',(rule,)),patch.object(policy,'CACHE_RULES',()):
            got,jit=policy.select_decode_defaults(s,altered,None,Sm120JitConfig(),'owner_bf16',eligible=True,clamp=10.)
        assert got==altered and jit==Sm120JitConfig(),field


def test_runtime_and_shape_exclusions():
    s=shape();b=baseline(s);j=Sm120JitConfig()
    excluded=[dict(eligible=False),dict(clamp=0.),dict(tuning=object()),dict(counts=(1,)),dict(contract='ordinary_topk')]
    for change in excluded:
        kw=dict(contract='owner_bf16',counts=None,eligible=True,clamp=10.,tuning=None);kw.update(change)
        assert policy.select_decode_defaults(s,b.kernel,None,j,**kw)==(b.kernel,j)
    for change in [dict(hidden=8192),dict(intermediate=8192),dict(num_sms=108),dict(sm_min_partition=4),
        dict(sm_partition_alignment=4),dict(tensor_parallel_size=2),dict(data_parallel_size=2),
        dict(ep_same_numa_peer_count=0),dict(ep_cross_numa_peer_count=7),dict(num_topk=5),dict(tokens_per_rank=0)]:
        altered=replace(s,**change)
        assert policy.select_decode_defaults(altered,b.kernel,None,j,'owner_bf16',eligible=True,clamp=10.)==(b.kernel,j)


def test_rank_and_cache_identity():
    s=shape();specs=[select(s,rank=r) for r in range(8)]
    assert len({x.cache_key for x in specs})==8
    assert all(x.kernel==specs[0].kernel for x in specs)


def test_partial_sms_errors_preserved():
    with pytest.raises(ValueError):select(shape(),overrides=MegaMoEHeuristicOverrides(k1_sms=56))


def test_invalid_cache_hint_rejected():
    with pytest.raises(ValueError):Sm120JitConfig(weight_eviction_policy='evict_last')
    with pytest.raises(ValueError):Sm120JitConfig(activation_eviction_policy='evict_first')


def test_existing_jit_positional_arguments_preserved():
    old_style = Sm120JitConfig(None, False, 3, 4, False, 2, True, False)
    explicit = Sm120JitConfig(fc1_utility_postprocess=None, k1_ready_major=False,
        k2_token_stripe=3, fc2_ready_bundle_k_tiles=4, fc2_packed_store=False,
        num_ab_stages_override=2, enable_globaltimer=True, enable_k2_tile_trace=False)
    assert old_style == explicit
    import inspect
    signature = inspect.signature(Sm120JitConfig)
    for name in ('weight_eviction_policy', 'activation_eviction_policy'):
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
