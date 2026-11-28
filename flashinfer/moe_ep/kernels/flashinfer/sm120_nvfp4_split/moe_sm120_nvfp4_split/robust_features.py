"""Shared workload/configuration features. No model-name or token whitelist."""

import math
from dataclasses import replace, asdict


def feature_vector(s, c, f):
    lg = lambda x: math.log2(max(float(x), 1e-10))
    geom = [
        lg(c.k1_tile[1] / 128),
        lg(c.k2_tile[1] / 128),
        lg(c.k1_sms / 72),
        lg(c.k2_sms / 38),
        float((c.k1_stages or 2) - 2),
        float(c.k2_stages - 2),
        lg(c.ready_queue_bundle / 4),
        float(c.rank_local_combine),
        float(c.k2_warps == 12),
        float(c.dispatch_compute_overlap),
        float(c.dispatch_rank_cache),
        lg(c.dispatch_warps),
        float(c.k2_tail_reclaim),
    ]
    work = [
        lg(max(s.expected_rows_per_expert, 0.01) / 128),
        lg(s.num_topk / 6),
        lg(s.num_total_experts / 256),
        lg(max(s.tokens_per_rank, 1) / 1024),
        1 / (1 + f.k1_waves),
        1 / (1 + f.k2_waves),
    ]
    direct = [
        lg(max(f.padded_fc1_flops, 1) / max(f.useful_fc1_flops, 1)),
        lg(max(f.padded_fc2_flops, 1) / max(f.useful_fc2_flops, 1)),
        lg(1 + f.token_back_bytes / max(f.dispatch_bytes, 1)),
        f.grain_relative_to_worker_work,
    ]
    return geom + direct + [g * w for g in geom for w in work]


def candidates(h, s, anchor, contract="owner_bf16", overrides=None):
    raw = [("anchor", anchor)]
    if anchor.k2_warps == 12:
        raw += [
            (
                f"dispatch{w}_tail{int(tail)}",
                replace(anchor, dispatch_warps=w, k2_tail_reclaim=tail),
            )
            for w in (1, 4)
            for tail in (False, True)
        ]
    else:
        raw += [
            ("sm64", replace(anchor, k1_sms=64, k2_sms=s.num_sms - 64)),
            ("sm80", replace(anchor, k1_sms=80, k2_sms=s.num_sms - 80)),
            ("k1s2", replace(anchor, k1_stages=2)),
            (
                f"k2s{5-anchor.k2_stages}",
                replace(anchor, k2_stages=5 - anchor.k2_stages),
            ),
            (
                "n64off",
                replace(
                    anchor,
                    k1_tile=(64, 64, 128),
                    k2_tile=(64, 64, 128),
                    rank_local_combine=False,
                    k2_stages=3,
                ),
            ),
        ]
    out = []
    for label, c in raw:
        if any(c == v for _, v in out):
            continue
        if (
            h.legality_errors(s, c)
            or h.semantic_errors(c, contract)
            or h.correctness_qualification_errors(s, c)
        ):
            continue
        if overrides is not None and any(
            v is not None and getattr(c, k, None) != v
            for k, v in asdict(overrides).items()
        ):
            continue
        out.append((label, c))
    return out


def in_domain(s):
    return (
        s.hidden == 4096
        and s.intermediate == 4096
        and 6 <= s.num_topk <= 8
        and 256 <= s.num_total_experts <= 288
        and s.expert_parallel_size == 4
        and s.tensor_parallel_size == 1
        and s.data_parallel_size == 1
        and s.num_sms == 110
        and s.ep_cross_numa_peer_count == 0
        and 0 < s.tokens_per_rank <= 24576
    )


def work_vector(s, anchor, f):
    """Small dimensionless workload basis for an isolated config change."""
    return [
        1.0,
        math.log2(max(s.expected_rows_per_expert, 0.01) / 128),
        math.log2(s.num_topk / 6),
        math.log2(s.num_total_experts / 256),
        math.log2(max(f.k1_waves, 1)),
        math.log2(max(f.k2_waves, 1)),
        math.log2(max(f.padded_fc1_flops, 1) / max(f.useful_fc1_flops, 1)),
        math.log2(1 + f.token_back_bytes / max(f.dispatch_bytes, 1)),
    ]
