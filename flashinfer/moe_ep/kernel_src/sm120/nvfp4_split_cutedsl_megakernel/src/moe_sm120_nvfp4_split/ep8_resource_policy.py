"""Exact native EP8 T32 compute/communication allocation policy.

Numerical permission, launch eligibility and utility auto-policy remain separate.
Qualified native-entry evidence: release_changes/ep8_resources_20261114.md.
"""
from dataclasses import asdict, replace

POLICY_VERSION = 'ep8-exact-t32-resources-v1'
SM_FIELDS = ('k1_sms', 'k2_sms', 'tx_sms', 'rx_sms')
ALLOCATIONS = {'dsv': (56, 38, 8, 8), 'glm': (64, 30, 8, 8)}
# Complete resolved production anchors, not generated from the selector at runtime.
BASE_CONFIGS = {'dsv': {'comm_backend': 'gin_hybrid',
         'kernel_comm_backend': 'gin',
         'token_back_mode': 'reuse_dispatch_warps',
         'k1_tile': [64, 32, 128],
         'k2_tile': [64, 32, 128],
         'k2_stages': 2,
         'k2_warps': 8,
         'k1_sms': 48,
         'k2_sms': 30,
         'tx_sms': 16,
         'rx_sms': 16,
         'sm_min_partition': 8,
         'sm_partition_alignment': 8,
         'dispatch_chunk_tokens': 64,
         'combine_chunk_rows': 16,
         'dispatch_channels': 2,
         'combine_channels': 2,
         'dispatch_slots': 2,
         'combine_slots': 4,
         'combine_blocking_put': False,
         'dispatch_local_handoff_windows': 1,
         'dispatch_remote_handoff_windows': 2,
         'ibgda_rc_per_pe': 1,
         'ibgda_rc_mapping': 'warp',
         'tp_k3_chunks': 1,
         'dispatch_warps': 4,
         'dispatch_compute_overlap': False,
         'dispatch_rank_cache': False,
         'rank_local_combine': False,
         'k1_ready_queue': True,
         'k1_ready_queue_m_rotation': 0,
         'k2_ready_queue': True,
         'ready_queue_bundle': 4,
         'k2_natural_regs': True,
         'k2_min_blocks_per_sm': 2,
         'k2_tail_reclaim': False,
         'expected_rows_per_expert': 6.0,
         'k1_compact_tokens': False,
         'k2_compact_tokens': False,
         'k1_stages': None},
 'glm': {'comm_backend': 'gin_hybrid',
         'kernel_comm_backend': 'gin',
         'token_back_mode': 'reuse_dispatch_warps',
         'k1_tile': [64, 32, 128],
         'k2_tile': [64, 32, 128],
         'k2_stages': 2,
         'k2_warps': 8,
         'k1_sms': 48,
         'k2_sms': 30,
         'tx_sms': 16,
         'rx_sms': 16,
         'sm_min_partition': 8,
         'sm_partition_alignment': 8,
         'dispatch_chunk_tokens': 64,
         'combine_chunk_rows': 16,
         'dispatch_channels': 2,
         'combine_channels': 2,
         'dispatch_slots': 2,
         'combine_slots': 4,
         'combine_blocking_put': False,
         'dispatch_local_handoff_windows': 1,
         'dispatch_remote_handoff_windows': 2,
         'ibgda_rc_per_pe': 1,
         'ibgda_rc_mapping': 'warp',
         'tp_k3_chunks': 1,
         'dispatch_warps': 4,
         'dispatch_compute_overlap': False,
         'dispatch_rank_cache': False,
         'rank_local_combine': False,
         'k1_ready_queue': True,
         'k1_ready_queue_m_rotation': 0,
         'k2_ready_queue': True,
         'ready_queue_bundle': 4,
         'k2_natural_regs': True,
         'k2_min_blocks_per_sm': 2,
         'k2_tail_reclaim': False,
         'expected_rows_per_expert': 7.111111111111111,
         'k1_compact_tokens': False,
         'k2_compact_tokens': False,
         'k1_stages': None}}


def select_ep8_resources(shape, anchor, overrides, contract, host_expert_counts=None,
                         *, eligible, gate_up_clamp, tuning_config=None):
    """Apply only after ordinary selection and override validation.

    eligible must attest default calibrated build/JIT and complete execution.
    Both API and native runner call this before resolving FC1 utility auto.
    """
    if (not eligible or gate_up_clamp != 10.0 or tuning_config is not None
            or host_expert_counts is not None or contract != 'owner_bf16'
            or (shape.tokens_per_rank, shape.hidden, shape.intermediate) != (32,4096,4096)
            or (shape.data_parallel_size, shape.tensor_parallel_size,
                shape.expert_parallel_size) != (1,1,8)
            or (shape.ep_same_numa_peer_count, shape.ep_cross_numa_peer_count) != (3,4)
            or (shape.num_sms, shape.sm_min_partition, shape.sm_partition_alignment) != (110,8,8)):
        return anchor
    model = {(6,256): 'dsv', (8,288): 'glm'}.get((shape.num_topk,shape.num_total_experts))
    if model is None:
        return anchor
    values=asdict(anchor)
    # JSON anchor records encode tuples as lists. Normalize only tile containers;
    # backend names and all resource/transport fields stay exact.
    values['k1_tile']=list(values['k1_tile'])
    values['k2_tile']=list(values['k2_tile'])
    if values != BASE_CONFIGS[model]:
        return anchor
    sms=ALLOCATIONS[model]
    if len(sms)!=4 or any(type(value) is not int for value in sms):
        raise ValueError('EP8 allocation must contain four integer SM counts')
    candidate=replace(anchor,**dict(zip(SM_FIELDS,sms)))
    from .tuning_config import overrides_match
    if not overrides_match(candidate,overrides):
        return anchor
    from .heuristic import legality_errors, semantic_errors, correctness_qualification_errors
    errors=(legality_errors(shape,candidate)+semantic_errors(candidate,contract)
            +correctness_qualification_errors(shape,candidate))
    if errors:
        raise ValueError('; '.join(errors))
    return candidate
