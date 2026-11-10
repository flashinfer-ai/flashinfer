"""Joint stage/queue paired cost correction, with unknown-route risk gates."""

import math
from dataclasses import replace, asdict
from .large_policy_parameters import PARAMETERS as P
from .robust_features import in_domain


def select(h, s, anchor, overrides, contract, counts):
    info = dict(
        mode="large_joint_stage_queue",
        changed=False,
        route_statistics="unknown; offline scenarios only",
    )
    if (
        not in_domain(s)
        or contract != "owner_bf16"
        or counts is not None
        or not P["density_min"] <= s.expected_rows_per_expert <= P["density_max"]
    ):
        return anchor, dict(info, reason="outside_calibrated_scope")
    if not (
        anchor.k1_tile == (64, 128, 128)
        and anchor.k2_tile == (64, 128, 128)
        and anchor.k1_stages == 3
        and anchor.k2_stages == 2
        and anchor.ready_queue_bundle == 4
        and anchor.k1_sms == 72
        and anchor.k2_sms == 38
        and anchor.k2_warps == 8
    ):
        return anchor, dict(info, reason="anchor_configuration_outside_paired_scope")
    c = replace(anchor, k2_stages=3, ready_queue_bundle=8)
    if (
        h.legality_errors(s, c)
        or h.semantic_errors(c, contract)
        or h.correctness_qualification_errors(s, c)
    ):
        return anchor, dict(info, reason="illegal_candidate")
    if overrides is not None and any(
        v is not None and getattr(c, k, None) != v for k, v in asdict(overrides).items()
    ):
        return anchor, dict(info, reason="explicit_overrides")
    x = [
        1.0,
        math.log2(s.expected_rows_per_expert / 768),
        math.log(s.num_topk / 6) / math.log(8 / 6),
    ]
    estimates = {
        route: [sum(a * w[j] for a, w in zip(x, P["models"][route])) for j in (0, 1)]
        for route in P["routes"]
    }
    mean = sum(v[0] for v in estimates.values()) / len(estimates)
    worst = max(v[0] for v in estimates.values())
    tail = max(v[1] for v in estimates.values())
    change = (
        mean < -P["mean_log_gain_margin"]
        and worst <= P["worst_log_p50_limit"]
        and tail <= P["worst_log_p90_limit"]
    )
    return (c if change else anchor), dict(
        info,
        changed=change,
        route_predictions=estimates,
        mean_log_p50=mean,
        worst_log_p50=worst,
        worst_log_p90=tail,
        reason="paired_model_risk_gate",
    )
