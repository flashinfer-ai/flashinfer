"""Scope of the measured GLM rule, including rejected generalizations."""
from dataclasses import asdict, replace
import unittest
from .test_tuning_config import shape
from .heuristic import (select_megamoe_config, explain_selection, _select_cached,
                        MegaMoEHeuristicOverrides)
from .api import MegaMoEProblemSpec, select_compile_spec
from .jit_config import Sm120JitConfig


class ModelPolicyTest(unittest.TestCase):
    def test_qualified_glm_capacity_selects_stage3(self):
        s=shape(16384)
        old=_select_cached(s,None,'owner_bf16',None)
        new=select_megamoe_config(s,numerical_contract='owner_bf16')
        self.assertEqual(old.k2_stages,2)
        self.assertEqual(new.k2_stages,3)
        changed={k for k,v in asdict(old).items() if asdict(new)[k]!=v and k not in ('k1_compact_tokens','k2_compact_tokens')}
        self.assertEqual(changed,{'k2_stages'})
        self.assertEqual(explain_selection(s,numerical_contract='owner_bf16')['profiles']['source'],'builtin_model')

    def test_rejected_n64_and_unqualified_neighbors_keep_general(self):
        for t in [448,512,576,8192,15872,16383,16385,16896]:
            s=shape(t)
            self.assertEqual(replace(select_megamoe_config(s,numerical_contract='owner_bf16'),k1_compact_tokens=False,k2_compact_tokens=False),
                             _select_cached(s,None,'owner_bf16',None),t)

    def test_dsv4_flash_and_other_scopes_keep_general(self):
        for updates in [dict(num_topk=6,num_total_experts=256),dict(ep_same_numa_peer_count=1,ep_cross_numa_peer_count=2),dict(num_sms=120)]:
            s=shape(16384,**updates)
            self.assertEqual(replace(select_megamoe_config(s,numerical_contract='owner_bf16'),k1_compact_tokens=False,k2_compact_tokens=False),
                             _select_cached(s,None,'owner_bf16',None))

    def test_explicit_stage2_override_wins(self):
        s=shape(16384);o=MegaMoEHeuristicOverrides(k2_stages=2)
        self.assertEqual(select_megamoe_config(s,o,numerical_contract='owner_bf16').k2_stages,2)

    def test_already_host_counts_keep_existing_count_aware_selection(self):
        s=shape(16384);total=s.tokens_per_rank*s.num_topk*s.expert_parallel_size
        q,r=divmod(total,s.num_total_experts);counts=(q+1,)*r+(q,)*(s.num_total_experts-r)
        self.assertEqual(select_megamoe_config(s,numerical_contract='owner_bf16',host_expert_counts=counts),
                         _select_cached(s,None,'owner_bf16',counts))

    def test_ordinary_contract_never_uses_owner_rule(self):
        s=shape(16384)
        self.assertEqual(select_megamoe_config(s),_select_cached(s,None,'ordinary_topk',None))
        self.assertFalse(select_megamoe_config(s).rank_local_combine)

    def test_api_custom_jit_retains_compatibility(self):
        p=MegaMoEProblemSpec(tokens_per_rank=16384,hidden=4096,intermediate=4096,
                            num_topk=8,num_total_experts=288,expert_parallel_size=4,expert_parallel_rank=0)
        args=dict(problem=p,numerical_contract='owner_bf16',num_sms=110,
                  sm_min_partition=8,sm_partition_alignment=8,ep_same_numa_peer_count=3,ep_cross_numa_peer_count=0)
        self.assertEqual(select_compile_spec(**args).kernel.k2_stages,3)
        from .heuristic import select_compatibility_config
        custom=select_compile_spec(**args,jit=Sm120JitConfig(k2_token_stripe=3))
        self.assertEqual(custom.kernel,select_compatibility_config(shape(16384),numerical_contract='owner_bf16'))


if __name__=='__main__':unittest.main()
