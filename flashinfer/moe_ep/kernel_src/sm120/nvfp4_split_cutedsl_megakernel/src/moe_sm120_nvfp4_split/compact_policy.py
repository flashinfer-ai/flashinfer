"""Evidence-qualified compact token specialization; no changes to tile geometry."""
from dataclasses import asdict, replace
import json
from .compact_profiles import PROFILES
def select(shape, base, overrides, contract, counts=None):
    # Explicit caller flags/configuration and route-aware selection retain priority.
    if counts is not None or (overrides and any(getattr(overrides,k,None) is not None for k in ("k1_compact_tokens","k2_compact_tokens"))):
        return base, dict(matched=False,reason="explicit caller specialization")
    shape_dict=json.loads(json.dumps(asdict(shape)))
    config=json.loads(json.dumps(asdict(base)))
    for profile in PROFILES:
        if profile["shape"]==shape_dict and profile["contract"]==contract and profile["baseline_config"]==config:
            return replace(base,**profile["flags"]),dict(matched=True,evidence=profile["evidence"],flags=profile["flags"])
    return base,dict(matched=False,reason="no qualified exact configuration")
