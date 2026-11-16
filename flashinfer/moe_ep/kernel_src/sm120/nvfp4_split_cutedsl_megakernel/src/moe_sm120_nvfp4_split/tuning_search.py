"""Bounded offline search and conservative per-route qualification."""
from dataclasses import asdict, replace
import math
import statistics


def auto_tokens(experts, topk, ep):
    """Representative capacities first, then probes around token-tile edges."""
    result = [16,32,64,128,512,1024,2048,4096,8192,12288,16384,20480,24576]
    for target, tile in [(512,64),(8192,128),(16384,128)]:
        period = experts * tile / (ep * topk)
        boundary = max(1, round(target / period)) * period
        for t in (math.floor(boundary)-1, math.floor(boundary), math.floor(boundary)+1):
            if 0 < t <= 24576 and t not in result:
                result.append(t)
    return result


def candidate_configs(shape, baseline, contract, limit=12):
    from .heuristic import enumerate_candidates
    from .tuning_config import config_from_dict
    # Local geometry/stage ablations precede the cost-ranked catalog. A model
    # misranking cannot hide the N64/72+38/RLC proposal from a short search.
    proposals = [baseline]
    if baseline.k2_warps == 8:
        proposals += [replace(baseline, k1_tile=(64, n, 128), k2_tile=(64, n, 128),
                              k1_compact_tokens=baseline.k1_compact_tokens and n == 128,
                              k2_compact_tokens=baseline.k2_compact_tokens and n == 128) for n in (64, 128, 32)]
        proposals += [replace(baseline, k2_stages=n) for n in (2, 3)]
        proposals += [replace(baseline, k1_stages=n) for n in (2, 3)]
        proposals += [replace(baseline, ready_queue_bundle=q) for q in (4, 8, 16)]
        proposals += [replace(baseline, k1_sms=n, k2_sms=shape.num_sms-n) for n in (64, 72, 80, 88)]
    catalog = enumerate_candidates(shape, numerical_contract=contract, full_catalog=True)
    # Add a best-cost representative of every geometry/SM/stage family before
    # filling with raw cost order, so large budgets expand coverage predictably.
    families = set()
    for item in catalog:
        c = item.config
        family = (c.k1_tile, c.k2_tile, c.k1_sms, c.k1_stages, c.k2_stages, c.k2_warps)
        if family not in families:
            proposals.append(c); families.add(family)
    proposals += [item.config for item in catalog]
    out = []
    for c in proposals:
        # V1 preserves the baseline reduction association. RLC transitions
        # need distinct arithmetic/equivalence qualification, not just timing.
        if c.rank_local_combine != baseline.rank_local_combine or c in out:
            continue
        try:
            config_from_dict(asdict(c), shape, contract)
        except (ValueError, TypeError, ZeroDivisionError):
            continue
        out.append(c)
        if len(out) >= limit:
            break
    return out


def qualify_pairs(pairs, required_routes, min_gain=.01):
    """Each route needs independently confirmed, matched measurements.

    Geometric means are over process quantiles, never pooled inner samples.
    No claim of a formal confidence interval is made.
    """
    reasons, metrics = [], {}
    for route in required_routes:
        group = [p for p in pairs if p['route'] == route]
        if len(group) < 2:
            reasons.append(route + ': missing independent pair'); continue
        seeds = [p['baseline'].get('spec', {}).get('seed') for p in group]
        if None in seeds or len(set(seeds)) < 2:
            reasons.append(route + ': missing independent holdout seeds'); continue
        ratios50, ratios90 = [], []
        for p in group:
            b, c = p['baseline'], p['candidate']
            if (b.get('status') != 'PASS' or c.get('status') != 'PASS'
                    or b['shape'] != c['shape'] or b['hardware'] != c['hardware']
                    or b['implementation_sha256'] != c['implementation_sha256']
                    or b['fixtures'] != c['fixtures'] or b['outputs'] != c['outputs']
                    or b['spec']['seed'] != c.get('spec', {}).get('seed')
                    or b['spec'].get('dynamic', 0) or c.get('spec', {}).get('dynamic', 0)
                    or b['contract'] != c['contract']):
                reasons.append(route + ': invalid or unmatched evidence'); continue
            if any(not math.isfinite(x) or x <= 0 for x in (b['p50_us'], c['p50_us'], b['p90_us'], c['p90_us'])):
                reasons.append(route + ': invalid latency'); continue
            ratios50.append(c['p50_us']/b['p50_us']); ratios90.append(c['p90_us']/b['p90_us'])
        if len(ratios50) != len(group):
            continue
        r50 = math.exp(statistics.mean(map(math.log, ratios50)))
        r90 = math.exp(statistics.mean(map(math.log, ratios90)))
        metrics[route] = dict(p50_ratio=r50, p90_ratio=r90,
                              per_pair_p50=ratios50, per_pair_p90=ratios90)
        if r50 > 1-min_gain or r90 > 1 or any(x > 1 for x in ratios50+ratios90):
            reasons.append(route + ': gain/P90/repeatability gate failed')
    return dict(qualified=not reasons and bool(metrics), reasons=reasons, metrics=metrics)
