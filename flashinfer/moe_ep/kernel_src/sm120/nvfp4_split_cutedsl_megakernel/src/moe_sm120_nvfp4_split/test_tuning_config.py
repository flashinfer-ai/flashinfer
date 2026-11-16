"""Synthetic host tests; numbers here are not GPU performance evidence."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from .heuristic import (MegaMoEHeuristicInput, MegaMoEHeuristicOverrides,
                        select_megamoe_config, explain_selection)
from .tuning_config import (make_document, parse_tuning_config, load_tuning_config,
                            config_from_dict)
from .tuning_search import candidate_configs, qualify_pairs


def shape(tokens=512, **kw):
    return MegaMoEHeuristicInput(**dict(dict(tokens_per_rank=tokens,hidden=4096,
        intermediate=4096,num_topk=8,num_total_experts=288,data_parallel_size=1,
        tensor_parallel_size=1,expert_parallel_size=4,ep_same_numa_peer_count=3,
        ep_cross_numa_peer_count=0,num_sms=110,sm_min_partition=8,sm_partition_alignment=8),**kw))


class TuningConfigTest(unittest.TestCase):
    def setUp(self):
        self.s=shape(); self.base=select_megamoe_config(self.s,numerical_contract='owner_bf16')
        self.candidate=replace(self.base,k1_tile=(64,64,128),k2_tile=(64,64,128),k1_compact_tokens=False,k2_compact_tokens=False)
        self.doc=make_document('synthetic-device',[
            dict(shape=asdict(self.s),numerical_contract='owner_bf16',
                 config=asdict(self.candidate),validation='qualified',
                 evidence={'fixture':'host-test-only'})],{'kind':'synthetic'})

    def select(self,doc=None,**kw):
        return select_megamoe_config(self.s,numerical_contract='owner_bf16',
            tuning_config=parse_tuning_config(doc or self.doc),hardware='synthetic-device',**kw)

    def test_exact_profile_wins_and_is_explained(self):
        self.assertEqual(self.select(),self.candidate)
        info=explain_selection(self.s,numerical_contract='owner_bf16',
            tuning_config=parse_tuning_config(self.doc),hardware='synthetic-device')
        self.assertEqual(info['profiles']['source'],'offline_tuning')

    def test_unknown_token_uses_general_not_nearest(self):
        s=shape(513)
        self.assertEqual(select_megamoe_config(s,numerical_contract='owner_bf16',
            tuning_config=parse_tuning_config(self.doc),hardware='synthetic-device'),
            select_megamoe_config(s,numerical_contract='owner_bf16'))

    def test_hardware_mismatch_and_missing_identity_fall_back(self):
        for hardware in ('other-device',None):
            self.assertEqual(select_megamoe_config(self.s,numerical_contract='owner_bf16',
                tuning_config=parse_tuning_config(self.doc),hardware=hardware),self.base)

    def test_source_and_abi_mismatch_fall_back(self):
        for key,val in [('implementation_sha256','old'),('kernel_cache_abi',0)]:
            d=copy.deepcopy(self.doc);d[key]=val
            self.assertEqual(self.select(d),self.base)

    def test_ordinary_contract_cannot_load_owner_profile(self):
        result=select_megamoe_config(self.s,tuning_config=parse_tuning_config(self.doc),hardware='synthetic-device')
        self.assertFalse(result.rank_local_combine)

    def test_explicit_conflicting_override_is_authoritative(self):
        o=MegaMoEHeuristicOverrides(k2_stages=2)
        self.assertEqual(self.select(overrides=o),select_megamoe_config(self.s,o,numerical_contract='owner_bf16'))
        self.assertEqual(self.select(overrides=o).k2_stages,2)

    def test_redundant_override_preserves_profile(self):
        self.assertEqual(self.select(overrides=MegaMoEHeuristicOverrides(comm_backend='p2p_direct')),self.candidate)

    def test_illegal_config_falls_back(self):
        for key,val in [('k1_sms',73),('k2_min_blocks_per_sm',0),('dispatch_warps',True),('k1_tile',[64,7,128]),('expected_rows_per_expert',0),('rank_local_combine','false')]:
            d=copy.deepcopy(self.doc);d['entries'][0]['config'][key]=val
            self.assertEqual(self.select(d),self.base,key)

    def test_duplicate_unqualified_and_unknown_fields_rejected(self):
        d=copy.deepcopy(self.doc);d['entries']*=2
        self.assertTrue(parse_tuning_config(d).error)
        d=copy.deepcopy(self.doc);d['entries'][0]['validation']='pending'
        self.assertTrue(parse_tuning_config(d).error)
        d=copy.deepcopy(self.doc);d['extra']=1
        self.assertTrue(parse_tuning_config(d).error)

    def test_reload_same_path_changes_content_cache_key(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'tuning.json';p.write_text(json.dumps(self.doc));a=load_tuning_config(p)
            d=copy.deepcopy(self.doc);d['entries'][0]['config']=asdict(self.base)
            p.write_text(json.dumps(d));b=load_tuning_config(p)
            self.assertNotEqual(a.sha256,b.sha256)
            for profile,want in [(a,self.candidate),(b,self.base),(a,self.candidate)]:
                self.assertEqual(select_megamoe_config(self.s,numerical_contract='owner_bf16',
                    tuning_config=profile,hardware='synthetic-device'),want)

    def test_unreadable_and_malformed_file_fall_back(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'broken.json'
            with self.assertWarns(RuntimeWarning): result=load_tuning_config(p)
            self.assertTrue(result.error)
            p.write_text('{')
            with self.assertWarns(RuntimeWarning): result=load_tuning_config(p)
            self.assertTrue(result.error)

    def test_compact_baseline_does_not_hide_narrow_geometry_candidates(self):
        base = replace(self.base, k1_compact_tokens=True, k2_compact_tokens=True)
        candidates = candidate_configs(self.s, base, 'owner_bf16', 12)
        self.assertIn(base, candidates)
        for n in (32, 64):
            expected = replace(base, k1_tile=(64, n, 128), k2_tile=(64, n, 128),
                               k1_compact_tokens=False, k2_compact_tokens=False)
            self.assertIn(expected, candidates)

    def test_search_covers_low_model_rank_n64_and_stage3(self):
        self.assertIn(self.candidate,candidate_configs(self.s,self.base,'owner_bf16',4))
        s=shape(16384);base=select_megamoe_config(s,numerical_contract='owner_bf16')
        self.assertIn(replace(base,k2_stages=3),candidate_configs(s,base,'owner_bf16',4))


class QualificationTest(unittest.TestCase):
    def setUp(self):
        self.pairs=[]
        for route in ['balanced','power_law','zero_count_peer']:
            for repeat in range(2):
                base=dict(status='PASS',shape={'t':512},hardware='synthetic',
                    implementation_sha256='fixture',fixtures=[str(repeat)],outputs=['same'],
                    contract='owner_bf16',p50_us=100.,p90_us=110.,spec=dict(seed=repeat))
                candidate=dict(base,p50_us=95.,p90_us=105.)
                self.pairs.append(dict(route=route,baseline=base,candidate=candidate))
        self.routes=['balanced','power_law','zero_count_peer']

    def test_complete_consistent_gain_qualifies(self):
        self.assertTrue(qualify_pairs(self.pairs,self.routes)['qualified'])

    def test_balanced_win_cannot_hide_zero_owner_regression(self):
        self.pairs[-1]['candidate']['p50_us']=146.
        self.assertFalse(qualify_pairs(self.pairs,self.routes)['qualified'])

    def test_p90_regression_rejected(self):
        self.pairs[-1]['candidate']['p90_us']=112.
        self.assertFalse(qualify_pairs(self.pairs,self.routes)['qualified'])

    def test_repeated_seed_and_dynamic_timing_rejected(self):
        self.pairs[-1]['baseline']['spec']['seed']=0
        self.assertFalse(qualify_pairs(self.pairs,self.routes)['qualified'])

    def test_missing_holdout_or_fixture_mismatch_rejected(self):
        self.assertFalse(qualify_pairs(self.pairs[:-1],self.routes)['qualified'])
        self.pairs[-1]['candidate']['fixtures']=['different']
        self.assertFalse(qualify_pairs(self.pairs,self.routes)['qualified'])


if __name__=='__main__': unittest.main()
