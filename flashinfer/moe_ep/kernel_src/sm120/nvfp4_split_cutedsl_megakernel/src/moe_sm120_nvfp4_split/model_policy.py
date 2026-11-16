"""Qualified GLM5.3-flash rules. Evidence and scope are explicit.

Model names never alter numerical permission or bypass correctness exclusions.
"""
from dataclasses import replace
from .tuning_config import overrides_match, config_from_dict

POLICY_VERSION = 'glm-qualified-v1'
# Populated only after paired route/neighbor and dynamic qualification.
QUALIFIED_RULES = ({'name': 'glm53_flash_t16384_k2_stage3',
 'tokens': (16384,),
 'anchor': {'comm_backend': 'p2p_direct',
            'kernel_comm_backend': 'p2p_direct',
            'token_back_mode': 'epi_warps',
            'k1_tile': (64, 128, 128),
            'k2_tile': (64, 128, 128),
            'k2_stages': 2,
            'k2_warps': 8,
            'k1_sms': 72,
            'k2_sms': 38,
            'tx_sms': 0,
            'rx_sms': 0,
            'sm_min_partition': 8,
            'sm_partition_alignment': 8,
            'dispatch_chunk_tokens': 64,
            'combine_chunk_rows': 16,
            'dispatch_channels': 2,
            'combine_channels': 2,
            'dispatch_slots': 2,
            'combine_slots': 2,
            'combine_blocking_put': False,
            'dispatch_local_handoff_windows': 1,
            'dispatch_remote_handoff_windows': 1,
            'ibgda_rc_per_pe': 1,
            'ibgda_rc_mapping': 'warp',
            'tp_k3_chunks': 1,
            'dispatch_warps': 1,
            'dispatch_compute_overlap': True,
            'dispatch_rank_cache': True,
            'rank_local_combine': True,
            'k1_ready_queue': True,
            'k1_ready_queue_m_rotation': 0,
            'k2_ready_queue': True,
            'ready_queue_bundle': 4,
            'k2_natural_regs': False,
            'k2_min_blocks_per_sm': 1,
            'k2_tail_reclaim': False,
            'expected_rows_per_expert': 1820.4444444444443,
            'k1_stages': 3},
 'changes': {'k2_stages': 3},
 'evidence': {'campaign': 'SM120-SELECTOR-TUNING-20261105',
              'reference_head': '22c08eb561ab821839fa921ee8c48ab21592cc9a',
              'kernel_abi': 2737,
              'routes': {'balanced': {'p50_change_percent': -3.1571490166554694,
                                      'p90_change_percent': -3.244131413942819,
                                      'pairs': [[0.9686653637465902, 0.9682568840761562],
                                                [0.9681917138348065, 0.9668609911071948]]},
                         'power_law': {'p50_change_percent': -1.6903763169018204,
                                       'p90_change_percent': -1.6960237799969025,
                                       'pairs': [[0.9824345569931557, 0.9824572729600779],
                                                 [0.9837583623170241, 0.9836225967921168]]},
                         'zero_count_peer': {'p50_change_percent': -1.3147136666957326,
                                             'p90_change_percent': -1.2858232256533153,
                                             'pairs': [[0.9869364866876941, 0.9874008390414856],
                                                       [0.9867692470638172, 0.9868827644198054]]}},
              'dynamic_replays': 100,
              'scope': 'Exact T16384; synthetic holdout routes; no global-optimum claim'}},)


def select(shape, anchor, overrides, contract, counts):
    info = dict(source='builtin_model', matched=False, version=POLICY_VERSION)
    if ((shape.hidden, shape.intermediate, shape.num_topk, shape.num_total_experts)
            != (4096, 4096, 8, 288)
            or (shape.data_parallel_size, shape.tensor_parallel_size, shape.expert_parallel_size) != (1, 1, 4)
            or (shape.num_sms, shape.sm_min_partition, shape.sm_partition_alignment) != (110, 8, 8)
            or shape.ep_same_numa_peer_count != 3 or shape.ep_cross_numa_peer_count != 0
            or contract != 'owner_bf16' or counts is not None):
        return anchor, dict(info, reason='outside qualified model/runtime/route scope')
    for rule in QUALIFIED_RULES:
        if shape.tokens_per_rank not in rule['tokens']:
            continue
        if any(getattr(anchor, k) != v for k, v in rule['anchor'].items()):
            continue
        candidate = replace(anchor, **rule['changes'])
        if not overrides_match(candidate, overrides):
            continue
        from dataclasses import asdict
        candidate = config_from_dict(asdict(candidate), shape, contract)
        return candidate, dict(info, matched=True, rule=rule['name'], evidence=rule['evidence'])
    return anchor, dict(info, reason='no qualified rule at this token capacity')
