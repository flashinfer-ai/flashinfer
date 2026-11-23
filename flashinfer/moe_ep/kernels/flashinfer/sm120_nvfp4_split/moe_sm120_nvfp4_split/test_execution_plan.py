"""Host-only contracts for empty and partially empty EP collectives."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from .api import MegaMoEProblemSpec, SplitKernelBuildOptions, select_compile_spec
from .execution_plan import select_collective_execution_plan
from .jit_config import Sm120JitConfig

OPTIONS = dict(ep_same_numa_peer_count=3, ep_cross_numa_peer_count=0,
               num_sms=110, sm_min_partition=8, sm_partition_alignment=8)


def problem(tokens=8192):
    return MegaMoEProblemSpec(tokens_per_rank=tokens, num_topk=6,
                             num_total_experts=256, hidden=4096, intermediate=4096,
                             expert_parallel_size=4, expert_parallel_rank=0)


def test_entire_empty_collective_skips_compilation_and_launch():
    with patch('moe_sm120_nvfp4_split.execution_plan.select_compile_spec') as select:
        plan = select_collective_execution_plan(problem=problem(0), ep_token_counts=(0,0,0,0))
        select.assert_not_called()
    assert plan.is_empty and not plan.requires_collective_launch
    assert plan.make_empty_output(device='cpu').shape == (0,4096)


def test_locally_empty_rank_still_participates_with_shared_capacity():
    plan = select_collective_execution_plan(problem=problem(32), ep_token_counts=(0,32,7,0), **OPTIONS)
    assert plan.requires_collective_launch and plan.local_output_tokens == 0
    assert plan.compile_spec.problem.tokens_per_rank == 32
    with pytest.raises(ValueError, match='must still join'):
        plan.make_empty_output(device='cpu')


@pytest.mark.parametrize('counts', [(0,0,0), (0,-1,0,0), (0,True,0,0), (0,64,0,0)])
def test_invalid_or_undersized_collective_rejected(counts):
    with pytest.raises(ValueError):
        select_collective_execution_plan(problem=problem(32), ep_token_counts=counts, **OPTIONS)


def test_api_default_contract_disables_rlc_and_owner_contract_enables_it():
    ordinary = select_compile_spec(problem=problem(), **OPTIONS)
    owner = select_compile_spec(problem=problem(), numerical_contract='owner_bf16', **OPTIONS)
    assert not ordinary.kernel.rank_local_combine and owner.kernel.rank_local_combine
    assert ordinary.cache_key != owner.cache_key


def test_explicit_jit_stage_is_visible_in_specialization():
    spec = select_compile_spec(problem=problem(), jit=replace(Sm120JitConfig(), num_ab_stages_override=2), **OPTIONS)
    assert spec.kernel.k1_stages == 2
    assert spec.canonical_dict()['kernel']['k1_stages'] == 2


def test_tail_reclaim_cannot_be_used_with_sequential_execution():
    from .heuristic import MegaMoEHeuristicOverrides
    with pytest.raises(ValueError, match='tail reclaim'):
        select_compile_spec(problem=problem(64), build=replace(SplitKernelBuildOptions(), concurrent_k1_k2=False),
                            overrides=MegaMoEHeuristicOverrides(k2_tail_reclaim=True), **OPTIONS)
