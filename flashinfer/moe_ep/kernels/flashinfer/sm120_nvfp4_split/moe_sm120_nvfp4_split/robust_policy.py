"""Unknown-route robust comparison against frozen v11 incumbent."""

from .robust_features import feature_vector, work_vector, candidates, in_domain
from .robust_parameters import PARAMETERS as P


def select_small(h, s, anchor, overrides, contract, counts):
    if (
        not in_domain(s)
        or not P["density_bounds"][0]
        <= s.expected_rows_per_expert
        <= P["density_bounds"][1]
        or contract != "owner_bf16"
        or counts is not None
    ):
        return anchor, dict(mode="baseline_outside_robust_scope", changed=False)
    choices = candidates(h, s, anchor, contract, overrides)
    base = feature_vector(s, anchor, h.derive_features(s, anchor))
    ranked = []
    for label, c in choices:
        z = [
            (a - b) / scale
            for a, b, scale in zip(
                feature_vector(s, c, h.derive_features(s, c)), base, P["scales"]
            )
        ]
        if P.get("mode") == "family":
            if c == anchor:
                estimates = {route: [0.0, 0.0] for route in P["routes"]}
            else:
                models = P["family_models"].get(label, {})
                if any(route not in models for route in P["routes"]):
                    continue
                xx = [
                    a / scale
                    for a, scale in zip(
                        work_vector(s, anchor, h.derive_features(s, anchor)),
                        P["work_scales"],
                    )
                ]
                estimates = {
                    route: [
                        sum(a * w[j] for a, w in zip(xx, models[route])) for j in [0, 1]
                    ]
                    for route in P["routes"]
                }
        else:
            estimates = {
                route: [
                    sum(a * w[j] for a, w in zip(z, P["models"][route])) for j in [0, 1]
                ]
                for route in P["routes"]
            }
        p50 = sum(x[0] for x in estimates.values()) / len(estimates)
        worst = max(x[0] for x in estimates.values())
        p90 = max(x[1] for x in estimates.values())
        ranked.append((p50, p90, label, c, estimates, worst))
    eligible = [
        r
        for r in ranked
        if r[0] < -P["log_margin"].get(r[2], 999.0) and r[5] <= 0.005 and r[1] <= 0.01
    ]
    best = (
        min(eligible, key=lambda r: r[0])
        if eligible
        else next(r for r in ranked if r[3] == anchor)
    )
    return best[3], dict(
        mode="route_robust_pairwise_ridge",
        changed=best[3] != anchor,
        action=best[2],
        log_margin=P["log_margin"],
        route_predictions=best[4],
        mean_predicted_log_p50=best[0],
        worst_predicted_log_p50=best[5],
        worst_predicted_log_p90=best[1],
        route_statistics="unknown; predictions over offline route scenarios, not observed counts",
        candidate_count=len(ranked),
    )


def select(h, s, anchor, overrides, contract, counts):
    from .large_policy import select as select_large

    large, info = select_large(h, s, anchor, overrides, contract, counts)
    if info["changed"]:
        return large, info
    small, small_info = select_small(h, s, anchor, overrides, contract, counts)
    return small, dict(small_info, large_joint=info)
