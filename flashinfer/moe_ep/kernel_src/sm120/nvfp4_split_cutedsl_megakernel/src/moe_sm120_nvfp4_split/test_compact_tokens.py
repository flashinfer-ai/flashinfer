from dataclasses import asdict,replace
import json
import pytest
from moe_sm120_nvfp4_split import compact_policy as policy
from moe_sm120_nvfp4_split.heuristic import MegaMoEHeuristicInput,MegaMoEHeuristicOverrides,select_megamoe_config,legality_errors
from moe_sm120_nvfp4_split.api import MegaMoEProblemSpec,MegaMoECompileSpec,KERNEL_CACHE_ABI

def fixture():
 shape=MegaMoEHeuristicInput(tokens_per_rank=512,hidden=4096,intermediate=4096,num_topk=8,num_total_experts=288,data_parallel_size=1,tensor_parallel_size=1,expert_parallel_size=4,ep_same_numa_peer_count=3,ep_cross_numa_peer_count=0,num_sms=110,sm_min_partition=8,sm_partition_alignment=8)
 cfg=select_megamoe_config(shape,numerical_contract='owner_bf16')
 assert cfg.k1_tile[1]==cfg.k2_tile[1]==128
 return shape,cfg

def test_cache_separates_both_specializations():
 shape,cfg=fixture();problem=MegaMoEProblemSpec(tokens_per_rank=512,num_topk=8,num_total_experts=288,hidden=4096,intermediate=4096,expert_parallel_size=4,expert_parallel_rank=0)
 specs=[MegaMoECompileSpec(problem,replace(cfg,k1_compact_tokens=a,k2_compact_tokens=b)) for a,b in [(False,False),(True,False),(True,True),(False,True)]]
 assert KERNEL_CACHE_ABI==2812
 assert len({s.cache_key for s in specs})==4
 assert len({s.canonical_dict()['kernel']['k1_tile'] for s in specs})==1

def test_policy_requires_exact_shape_contract_and_baseline(monkeypatch):
 shape,cfg=fixture();record=dict(shape=json.loads(json.dumps(asdict(shape))),contract='owner_bf16',baseline_config=json.loads(json.dumps(asdict(cfg))),flags=dict(k1_compact_tokens=True,k2_compact_tokens=True),evidence='unit')
 monkeypatch.setattr(policy,'PROFILES',[record]);selected,info=policy.select(shape,cfg,None,'owner_bf16')
 assert info['matched'] and selected.k1_compact_tokens and selected.k2_compact_tokens
 for s,c,contract,override,counts in [(replace(shape,tokens_per_rank=513),cfg,'owner_bf16',None,None),(shape,replace(cfg,k2_stages=2),'owner_bf16',None,None),(shape,cfg,'ordinary_topk',None,None),(shape,cfg,'owner_bf16',MegaMoEHeuristicOverrides(k1_compact_tokens=False),None),(shape,cfg,'owner_bf16',None,(0,))]:
  actual,info=policy.select(s,c,override,contract,counts);assert actual==c and not info['matched']

def test_explicit_flags_survive_selector_and_legacy_fallback():
 for ep in [4,8]:
  shape,cfg=fixture();shape=replace(shape,expert_parallel_size=ep,ep_cross_numa_peer_count=0 if ep==4 else 4)
  chosen=select_megamoe_config(shape,MegaMoEHeuristicOverrides(k1_compact_tokens=True,k2_compact_tokens=True),numerical_contract='owner_bf16' if ep==4 else 'ordinary_topk')
  assert chosen.k1_compact_tokens and chosen.k2_compact_tokens
  assert not legality_errors(shape,chosen)

def test_narrow_tile_does_not_accept_compact_flag():
 shape,cfg=fixture()
 assert any('requires an N128' in e for e in legality_errors(shape,replace(cfg,k1_tile=(64,64,128),k1_compact_tokens=True)))
 assert any('must be bool' in e for e in legality_errors(shape,replace(cfg,k1_compact_tokens=1)))
