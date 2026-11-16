"""CPU tests: python -m unittest moe_sm120_nvfp4_split.test_ep8_resource_policy -v.

Executes the native runner's actual host selection block via AST extraction,
without importing its GPU dependencies or initializing any GPU context.
"""
import ast
from dataclasses import asdict, replace
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import api, ep8_resource_policy as policy
from .heuristic import (
    MegaMoEHeuristicInput, MegaMoEHeuristicOverrides,
    select_megamoe_config, select_compatibility_config,
)
from .jit_config import Sm120JitConfig
from .tuning_config import TuningConfig
from .utility_policy import select_utility


def args(model='dsv'):
    k,e=(6,256) if model=='dsv' else (8,288)
    return dict(problem=api.MegaMoEProblemSpec(32,k,e,4096,4096,8,0,gate_up_clamp=10),
        ep_same_numa_peer_count=3,ep_cross_numa_peer_count=4,num_sms=110,
        sm_min_partition=8,sm_partition_alignment=8,numerical_contract='owner_bf16')


def shape(a):
    p=a['problem']
    return MegaMoEHeuristicInput(tokens_per_rank=p.tokens_per_rank,hidden=p.hidden,
        intermediate=p.intermediate,num_topk=p.num_topk,num_total_experts=p.num_total_experts,
        data_parallel_size=p.data_parallel_size,tensor_parallel_size=p.tensor_parallel_size,
        expert_parallel_size=p.expert_parallel_size,
        **{k:a[k] for k in ('ep_same_numa_peer_count','ep_cross_numa_peer_count',
                           'num_sms','sm_min_partition','sm_partition_alignment')})


def sms(config):
    return tuple(getattr(config,k) for k in policy.SM_FIELDS)


def native_select(a, *, utility=None, full=True, dual=2, build=None, jit=None,
                  overrides=None, tuning=None):
    """Run verbatim current native host selection/utility code; no GPU imports."""
    path=Path(__file__).with_name('mega_runner.py')
    tree=ast.parse(path.read_text(),filename=str(path))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='selection_jit' for t in n.targets))
    stop=next(i for i,n in enumerate(main.body[start:],start)
              if isinstance(n,ast.If) and ast.unparse(n.test)=='rank == 0')
    code=compile(ast.Module(body=main.body[start:stop],type_ignores=[]),str(path),'exec')
    b=build or api.SplitKernelBuildOptions()
    launch=SimpleNamespace(skip_ref_check=not full,group_hint=b.group_hint,
        flag_batch=b.flag_batch,epi_flag_batch=repr(b.epi_flag_batch),
        split_launch='green_graph' if b.concurrent_k1_k2 else 'serial',
        numerical_contract=a['numerical_contract'],gate_up_clamp=a['problem'].gate_up_clamp)
    ns=dict(os=os,args=launch,heuristic_input=shape(a),heuristic_overrides=overrides,
        offline_tuning=tuning,offline_hardware=None,
        Sm120JitConfig=Sm120JitConfig,SplitKernelBuildOptions=api.SplitKernelBuildOptions,
        is_calibrated_execution=api.is_calibrated_execution,
        select_megamoe_config=select_megamoe_config,
        select_compatibility_config=select_compatibility_config,
        _parse_tuple=lambda value:ast.literal_eval(value))
    env={'MEGA_DUAL_K2_CTAS_PER_SM':str(dual)}
    if utility is not None:env['MEGA_FC1_UTILITY_POSTPROCESS']='1' if utility else '0'
    if jit is not None:
        # Preserve the exact runner block but substitute its env parser result
        # through an isolated facade, without changing the immutable JIT class.
        class Facade:
            @staticmethod
            def from_environment():return jit
        ns['Sm120JitConfig']=Facade
    with patch.dict(os.environ,env,clear=True):exec(code,ns)
    return ns['kernel_config'],ns['selection_jit']


class Ep8ResourcePolicyTests(unittest.TestCase):
    def setUp(self):
        # Isolate the original SM-only policy; continuous policy has its own
        # integrated API/native/interval/override tests.
        from . import decode_policy
        self.geometry_patch=patch.object(decode_policy,'GEOMETRY_RULES',())
        self.cache_patch=patch.object(decode_policy,'CACHE_RULES',())
        self.geometry_patch.start();self.cache_patch.start()
        api.select_compile_spec.cache_clear()
    def tearDown(self):
        api.select_compile_spec.cache_clear()
        self.cache_patch.stop();self.geometry_patch.stop()

    def test_api_native_default_and_utility_parity(self):
        for model in ('dsv','glm'):
            a=args(model);spec=api.select_compile_spec(**a);c,j=native_select(a)
            self.assertEqual(spec.kernel,c)
            self.assertEqual(spec.jit,j)
            self.assertEqual(sms(c),policy.ALLOCATIONS[model])
            self.assertEqual(j.fc1_utility_postprocess,model=='dsv')
            self.assertEqual(c.k1_tile,(64,32,128));self.assertEqual(c.k2_tile,(64,32,128))
            self.assertEqual(c.combine_chunk_rows,16)

    def test_explicit_utility_booleans_win_in_both_entries(self):
        for model in ('dsv','glm'):
            for flag in (False,True):
                a=args(model)
                spec=api.select_compile_spec(**a,jit=Sm120JitConfig(fc1_utility_postprocess=flag))
                c,j=native_select(a,utility=flag)
                self.assertEqual(spec.kernel,c);self.assertEqual(spec.jit,j)
                self.assertIs(j.fc1_utility_postprocess,flag)

    def test_old_explicit_sm_and_arbitrary_override_precedence(self):
        for model in ('dsv','glm'):
            for counts in ((48,30,16,16),policy.ALLOCATIONS[model],(64,30,8,8)):
                o=MegaMoEHeuristicOverrides(**dict(zip(policy.SM_FIELDS,counts)))
                a=args(model);spec=api.select_compile_spec(**a,overrides=o)
                c,j=native_select(a,overrides=o)
                self.assertEqual(sms(c),counts);self.assertEqual(spec.kernel,c)
                self.assertEqual(spec.jit,j)
                if counts==(48,30,16,16):self.assertEqual(j.fc1_utility_postprocess,model=='dsv')

    def test_partial_sm_error_preserved(self):
        a=args();o=MegaMoEHeuristicOverrides(k1_sms=56)
        with self.assertRaises(ValueError):api.select_compile_spec(**a,overrides=o)
        with self.assertRaises(ValueError):native_select(a,overrides=o)

    def test_anchor_fields_and_other_overrides_exclude_rule(self):
        a=args();s=shape(a);base=select_megamoe_config(s,numerical_contract='owner_bf16')
        for field,value in vars(base).items():
            changed=(not value if isinstance(value,bool) else value+1 if isinstance(value,(int,float))
                     else 'different' if isinstance(value,str) else (64,16,128) if isinstance(value,tuple) else 3)
            other=replace(base,**{field:changed})
            with self.subTest(field=field):
                self.assertIs(policy.select_ep8_resources(s,other,None,'owner_bf16',
                    eligible=True,gate_up_clamp=10),other)
        for o in (MegaMoEHeuristicOverrides(combine_chunk_rows=4),
                  MegaMoEHeuristicOverrides(comm_backend='gin'),
                  MegaMoEHeuristicOverrides(k1_tile=(64,16,128))):
            spec=api.select_compile_spec(**a,overrides=o);c,j=native_select(a,overrides=o)
            self.assertEqual(spec.kernel,c);self.assertEqual(spec.jit,j)
            self.assertEqual(sms(c),(48,30,16,16))
        o=MegaMoEHeuristicOverrides(comm_backend='gin_hybrid')
        self.assertEqual(sms(api.select_compile_spec(**a,overrides=o).kernel),policy.ALLOCATIONS['dsv'])

    def test_all_shape_boundaries_are_exact(self):
        a=args();s=shape(a);base=select_megamoe_config(s,numerical_contract='owner_bf16')
        cases=dict(tokens_per_rank=(1,16,31,33,64,128,159,160,161,24576),
            hidden=(4128,),intermediate=(4160,),num_topk=(5,8),num_total_experts=(264,288),
            data_parallel_size=(2,),tensor_parallel_size=(2,),expert_parallel_size=(4,),
            ep_same_numa_peer_count=(2,),ep_cross_numa_peer_count=(3,),num_sms=(108,120),
            sm_min_partition=(4,),sm_partition_alignment=(4,))
        for field,values in cases.items():
            for value in values:
                with self.subTest(field=field,value=value):
                    self.assertIs(policy.select_ep8_resources(replace(s,**{field:value}),base,None,
                        'owner_bf16',eligible=True,gate_up_clamp=10),base)

    def test_context_exclusions_and_api_native_parity(self):
        for model in ('dsv','glm'):
            for change in ('clamp','contract','trace','stage','build','tuning'):
                a=args(model);kw={};native={}
                if change=='clamp':a['problem']=replace(a['problem'],gate_up_clamp=8)
                if change=='contract':a['numerical_contract']='ordinary_topk'
                if change=='trace':kw['jit']=Sm120JitConfig(enable_globaltimer=True);native['jit']=kw['jit']
                if change=='stage':kw['jit']=Sm120JitConfig(num_ab_stages_override=3);native['jit']=kw['jit']
                if change=='build':kw['build']=api.SplitKernelBuildOptions(group_hint=40);native['build']=kw['build']
                if change=='tuning':kw['tuning_config']=TuningConfig('','','intentional invalid CPU fixture');native['tuning']=kw['tuning_config']
                with self.subTest(model=model,change=change):
                    spec=api.select_compile_spec(**a,**kw);c,j=native_select(a,**native)
                    # Explicit JIT stages become canonical in MegaMoECompileSpec.
                    if change=='stage':c=replace(c,k1_stages=3)
                    self.assertEqual(spec.kernel,c);self.assertEqual(spec.jit,j)
                    self.assertEqual(sms(c),(48,30,16,16))

    def test_host_counts_and_incomplete_runner_excluded(self):
        for model in ('dsv','glm'):
            a=args(model);counts=(32*8*a['problem'].num_topk,)+(0,)*(a['problem'].num_total_experts-1)
            self.assertEqual(sms(api.select_compile_spec(**a,host_expert_counts=counts).kernel),(48,30,16,16))
            for kw in ({'full':False},{'dual':1}):
                c,j=native_select(a,**kw);self.assertEqual(sms(c),(48,30,16,16))
                self.assertFalse(j.fc1_utility_postprocess)

    def test_neighbor_specs_and_keys_unchanged(self):
        for model in ('dsv','glm'):
            for t in (16,31,33,64,128,159,160,161):
                a=args(model);a['problem']=replace(a['problem'],tokens_per_rank=t)
                actual=api.select_compile_spec(**a)
                original=select_megamoe_config(shape(a),numerical_contract='owner_bf16')
                enabled=select_utility(shape(a),original,'owner_bf16')
                expected=api.MegaMoECompileSpec(a['problem'],original,
                    jit=Sm120JitConfig(fc1_utility_postprocess=enabled))
                self.assertEqual(actual.kernel,expected.kernel);self.assertEqual(actual.cache_key,expected.cache_key)

    def test_all_ranks_and_cache_identity(self):
        for model in ('dsv','glm'):
            a=args(model);keys=set()
            for rank in range(8):
                a['problem']=replace(a['problem'],expert_parallel_rank=rank)
                spec=api.select_compile_spec(**a);c,j=native_select(a)
                self.assertEqual(spec.kernel,c);self.assertEqual(spec.jit,j);keys.add(spec.cache_key)
            self.assertEqual(len(keys),8)
            old=api.select_compile_spec(**a,overrides=MegaMoEHeuristicOverrides(
                k1_sms=48,k2_sms=30,tx_sms=16,rx_sms=16))
            self.assertNotEqual(old.cache_key,spec.cache_key)
            off=replace(spec,jit=Sm120JitConfig(fc1_utility_postprocess=False))
            on=replace(spec,jit=Sm120JitConfig(fc1_utility_postprocess=True))
            self.assertNotEqual(off.cache_key,on.cache_key)
            self.assertIs(api.select_compile_spec(**a),spec)
            self.assertEqual(spec.canonical_dict()['cache_abi'],api.KERNEL_CACHE_ABI)

    def test_production_utility_old_new_exact_and_glm_hold(self):
        from .utility_profiles import PROFILES
        ep8=[p for p in PROFILES if p['key']['shape']['expert_parallel_size']==8]
        self.assertFalse(any(p['key']['shape']['num_topk']==8 for p in ep8))
        for counts in ((48,30,16,16),policy.ALLOCATIONS['dsv']):
            self.assertTrue(any(tuple(p['key']['config'][k] for k in policy.SM_FIELDS)==counts
                and p['key']['shape']['tokens_per_rank']==32 for p in ep8))

    def test_every_nondefault_jit_field_excludes_resources(self):
        changes=dict(k1_ready_major=False,k2_token_stripe=4,fc2_ready_bundle_k_tiles=8,
            fc2_packed_store=True,num_ab_stages_override=3,enable_globaltimer=True,
            enable_k2_tile_trace=True)
        self.assertEqual(set(changes),set(vars(Sm120JitConfig()))-{'fc1_utility_postprocess','weight_eviction_policy','activation_eviction_policy'})
        for model in ('dsv','glm'):
            for field,value in changes.items():
                with self.subTest(model=model,field=field):
                    jit=replace(Sm120JitConfig(),**{field:value})
                    spec=api.select_compile_spec(**args(model),jit=jit)
                    c,j=native_select(args(model),jit=jit)
                    self.assertEqual(sms(spec.kernel),(48,30,16,16))
                    self.assertEqual(sms(c),(48,30,16,16))
                    self.assertFalse(spec.jit.fc1_utility_postprocess)
                    self.assertFalse(j.fc1_utility_postprocess)

    def test_every_nondefault_build_field_excludes_api_resources(self):
        changes=dict(group_hint=40,flag_batch=8,epi_flag_batch=(2,1),
            concurrent_k1_k2=False,k1_active_clusters=48,k2_active_clusters=30)
        self.assertEqual(set(changes),set(vars(api.SplitKernelBuildOptions())))
        for model in ('dsv','glm'):
            for field,value in changes.items():
                with self.subTest(model=model,field=field):
                    build=replace(api.SplitKernelBuildOptions(),**{field:value})
                    spec=api.select_compile_spec(**args(model),build=build)
                    self.assertEqual(sms(spec.kernel),(48,30,16,16))
                    self.assertFalse(spec.jit.fc1_utility_postprocess)

    def test_runner_green_and_completion_guards_still_apply(self):
        self.assertFalse(api.is_calibrated_execution(complete_pipeline=False))
        self.assertFalse(api.is_calibrated_execution(dual_k2_ctas_per_sm=1))
        self.assertFalse(api.is_calibrated_execution(build=api.SplitKernelBuildOptions(concurrent_k1_k2=False)))

if __name__=='__main__':unittest.main()
