# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Shape-independent identities and legality for frozen grouped MoE kernels."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_DIMENSIONS = ("s", "n", "k", "experts", "groups")
_DTYPES = ("bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4")


def dimension_matches(value: int, rule: Any) -> bool:
    if isinstance(rule, int):
        return value == rule
    if not isinstance(rule, dict):
        return False
    multiple = int(rule.get("multiple_of", 1))
    if multiple <= 0:
        raise RuntimeError("cuDNN Frost dimension 'multiple_of' must be positive")
    return (
        value >= int(rule.get("min", 0))
        and ("max" not in rule or value <= int(rule["max"]))
        and value % multiple == 0
    )


def can_implement(kernel, dimensions: dict[str, int]) -> bool:
    """Check artifact dimension rules; tensor layouts/ABIs remain dtype-specific."""
    return all(
        value > 0 and dimension_matches(value, kernel.contract.get(name))
        for name, value in dimensions.items()
    )


def generic_contract(raw: dict[str, Any]) -> dict[str, Any]:
    """Conservative common bounds for the frozen symbolic grouped-MoE recipes.

    128-element N/K alignment covers BF16 vector/TMA stores and every supported
    block-scale layout. Offsets remain unaligned, int32 and one per routed group;
    aligned-offset and split-K recipes require a separate launch contract.
    """
    parameters = raw.get("source", {}).get("parameters", {})
    if parameters.get("version") != 1:
        raise ValueError("generic Frost contracts require frozen source parameters")
    constants = dict(parameters["constants"])
    if (
        constants.get("split_k_slices") != "1"
        or constants.get("moe_token_alignment") != "1"
        or constants.get("moe_aligned_offsets", "False") != "False"
        or constants.get("offset_cutlass_dtype") != "cutlass.Int32"
    ):
        raise ValueError("unsupported generic Frost grouped-MoE recipe")
    contract = deepcopy(raw["contract"])
    contract.update(
        s={"min": 1},
        n={"min": 128, "max": 1 << 20, "multiple_of": 128},
        k={"min": 128, "max": 1 << 20, "multiple_of": 128},
        experts={"min": 1, "max": 1024},
        groups={"min": 1, "max": 1024},
    )
    return contract


def kernel_id(raw: dict[str, Any], dtype: str) -> str:
    """Identify logical axes; source/compiler hashes qualify compiled tactics."""
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported Frost catalog dtype: {dtype}")
    from .dimension_catalog import logical_id

    return logical_id(raw, dtype)


def generalize_records(
    records: list[dict[str, Any]], dtype: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Coalesce verified implementations and return the old-to-new shortlist IDs."""
    catalog: dict[str, dict[str, Any]] = {}
    identities: dict[str, str] = {}
    for original in records:
        raw = deepcopy(original)
        raw["contract"] = generic_contract(raw)
        identity = kernel_id(raw, dtype)
        identities[original["id"]] = identity
        raw["id"] = identity
        previous = catalog.get(identity)
        if previous is not None:
            if any(
                previous[key] != raw[key]
                for key in (
                    "source",
                    "abi",
                    "contract",
                    "tactic",
                    "workspace_bytes",
                    "launch",
                )
            ):
                raise RuntimeError("conflicting Frost implementation identity")
            revisions = set(previous.get("producer_revisions", ()))
            revisions.update(
                r["producer_revision"]
                for r in (previous, raw)
                if "producer_revision" in r
            )
            if len(revisions) > 1:
                previous["producer_revisions"] = sorted(revisions)
        else:
            catalog[identity] = raw
    return list(catalog.values()), identities


def select_kernels(
    first,
    second,
    fused,
    profiles,
    *,
    tokens,
    hidden,
    intermediate,
    experts,
    topk,
    sm_count,
):
    """Use measured choices or bound the unmeasured search for every dtype."""
    from .heuristics import select_stages
    from .shortlist import select_profiles

    if not first or not second:
        return (), ()
    choices = None
    if profiles and tokens <= max(profiles):
        bucket = min(n for n in profiles if n >= tokens)
        choices = profiles[bucket]
    # A measured tile can be unavailable on this device. Manifest IDs are
    # validated when reading the shortlist; rank the remaining legal pool.
    if choices is None or any(
        not set(identities) <= {kernel.artifact_id for kernel in pool}
        for pool, identities in zip((first, second), choices, strict=True)
    ):
        return select_stages(
            first,
            second,
            fused,
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            topk=topk,
            sm_count=sm_count,
        )
    first, second = select_profiles(profiles, tokens, first, second)
    tiles = {kernel.tactic_metadata["tile"] for kernel in first}
    return first + tuple(k for k in fused if k.tactic_metadata["tile"] in tiles), second
