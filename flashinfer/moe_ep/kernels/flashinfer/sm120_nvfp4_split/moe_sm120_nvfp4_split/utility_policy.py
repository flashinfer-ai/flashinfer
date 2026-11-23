"""Evidence-qualified FC1 utility defaults; exact shapes/configs only.

Unmatched shapes, explicit JIT/build variants and unsupported tiles retain the
original implementation. This policy never changes kernel geometry or precision.
"""
from dataclasses import asdict
from .utility_profiles import PROFILES

def select_utility(shape, config, contract, *, eligible=True):
    if (not eligible or config.k1_tile[1] not in (16, 32)
            or config.k1_compact_tokens):
        return False
    # JSON-normalize tuples once for comparison with immutable evidence records.
    from .heuristic import policy_asdict
    import json
    key = json.loads(json.dumps(dict(shape=asdict(shape), config=policy_asdict(config), contract=contract)))
    return any(key == profile["key"] for profile in PROFILES)
