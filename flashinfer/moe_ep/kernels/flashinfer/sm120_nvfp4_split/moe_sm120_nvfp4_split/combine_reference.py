# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure PyTorch combine references; no kernel, CUDA or communication-runtime imports.

Input terms are BF16 expert outputs, *after* FC2's output conversion. Arithmetic
validation must obtain them from an independent expert reference. Transport
validation instead reconstructs from actual staging, and cannot certify GEMMs.

FLAT: slot-ordered FP32 sum, then one BF16 conversion.
GROUPED: slot-ordered FP32 sum per owner, BF16 partial conversion, then
owner-rank-ordered FP32 sum and final BF16 conversion. This extra rounding is
intentional in the current RLC numerical contract, not an equivalence claim
with FLAT. Neither reduction is replaced with torch.sum (unspecified tree).
"""

from __future__ import annotations

import torch


def flat_topk_bf16_reference(terms: torch.Tensor) -> torch.Tensor:
    """Ordered FP32 additions over axis 1 of BF16 [token, slot, hidden]."""
    if terms.dtype != torch.bfloat16 or terms.ndim != 3:
        raise ValueError("expected BF16 [token, slot, hidden]")
    acc = torch.zeros(
        (terms.shape[0], terms.shape[2]), dtype=torch.float32, device=terms.device
    )
    for slot in range(terms.shape[1]):
        acc = acc + terms[:, slot, :].float()
    return acc.to(torch.bfloat16)


def owner_bf16_partials_reference(
    terms: torch.Tensor, topk_idx: torch.Tensor, *,
    world_size: int, num_experts_per_rank: int,
) -> torch.Tensor:
    """Return BF16 [token, owner, hidden]; absent owners contribute positive 0."""
    if terms.dtype != torch.bfloat16 or terms.ndim != 3:
        raise ValueError("expected BF16 [token, slot, hidden]")
    if topk_idx.shape != terms.shape[:2] or topk_idx.device != terms.device:
        raise ValueError("route shape/device must match terms")
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("route must contain integer expert IDs")
    if world_size <= 0 or num_experts_per_rank <= 0:
        raise ValueError("world_size and num_experts_per_rank must be positive")
    if bool(((topk_idx < 0) | (topk_idx >= world_size * num_experts_per_rank)).any()):
        raise ValueError("expert ID outside EP group")
    owners = topk_idx // num_experts_per_rank
    partials = torch.zeros(
        (terms.shape[0], world_size, terms.shape[2]),
        dtype=torch.bfloat16, device=terms.device,
    )
    for owner in range(world_size):
        acc = torch.zeros_like(partials[:, owner], dtype=torch.float32)
        for slot in range(terms.shape[1]):
            # where, not value*mask: an unselected NaN/Inf must not contaminate
            # a different owner, including count=0.
            acc = acc + torch.where(
                (owners[:, slot] == owner)[:, None], terms[:, slot].float(), 0.0
            )
        partials[:, owner] = acc.to(torch.bfloat16)
    return partials


def owner_grouped_bf16_reference(
    terms: torch.Tensor, topk_idx: torch.Tensor, *,
    world_size: int, num_experts_per_rank: int,
) -> torch.Tensor:
    """RLC arithmetic contract, including the owner partial BF16 rounding."""
    return flat_topk_bf16_reference(owner_bf16_partials_reference(
        terms, topk_idx, world_size=world_size,
        num_experts_per_rank=num_experts_per_rank,
    ))


def byte_difference(actual: torch.Tensor, expected: torch.Tensor, *, chunk_bytes=16*1024**2) -> int:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("byte comparison requires equal shapes and dtypes")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    a = actual.contiguous().view(torch.uint8).reshape(-1)
    b = expected.contiguous().view(torch.uint8).reshape(-1)
    # Reduce each bounded chunk before creating the next one. A full boolean
    # sum may materialize an int64 temporary eight times the byte-buffer size.
    return sum(int((a[start:start+chunk_bytes] != b[start:start+chunk_bytes]).sum().item())
               for start in range(0, a.numel(), chunk_bytes))


def numerical_comparison(actual, expected, *, atol=0.05, rtol=0.05) -> dict:
    """Finite values and tolerance are mandatory, even for matching NaN/Inf."""
    if actual.shape != expected.shape:
        raise ValueError("numerical comparison requires equal shapes")
    a, b = actual.float(), expected.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a - b).abs()
    bad = ~finite | (delta > atol + rtol * b.abs())
    nonfinite = int((~finite).sum().item())
    return {
        "status": "PASS" if not bool(bad.any()) else "FAIL",
        "elements": a.numel(),
        "different_elements": int((a != b).sum().item()),
        "over_tolerance_elements": int(bad.sum().item()),
        "nonfinite_elements": nonfinite,
        # None keeps structured output valid JSON for nonfinite inputs.
        "max_abs": float(delta.max()) if a.numel() and not nonfinite else None,
        "mean_abs": float(delta.mean()) if a.numel() and not nonfinite else None,
        "atol": atol, "rtol": rtol,
    }


def make_combine_report(actual, flat_reference, *, grouped_reference=None,
                        transport=None, independent_grouped_arithmetic=False) -> dict:
    """Independent verdicts, deliberately no aggregate `passed` field."""
    grouped = {"status": "NOT_APPLICABLE", "reason": "RLC is disabled"}
    if grouped_reference is not None:
        diff = byte_difference(actual, grouped_reference)
        nonfinite = int((~torch.isfinite(actual)
                         | ~torch.isfinite(grouped_reference)).sum().item())
        grouped = {
            "status": "PASS" if diff == 0 and nonfinite == 0 else "FAIL",
            "byte_differences": diff, "nonfinite_elements": nonfinite,
            "contract": "slot_FP32_owner_BF16_then_owner_FP32_output_BF16",
        }
    extra = {}
    if grouped_reference is not None and independent_grouped_arithmetic:
        # Independent dense GEMM has a numerical accuracy contract. Its BF16
        # terms need not be bitwise identical to native FP32 MMA accumulation.
        # Exact association/transport is instead checked from actual FC2 staging.
        extra["independent_grouped_byte_equivalence"] = dict(grouped)
        extra["grouped_arithmetic"] = numerical_comparison(actual, grouped_reference)
        errors = (transport or {}).get("errors", {})
        exact = (transport is not None and transport.get("status") == "PASS"
                 and transport.get("source") == "actual_FC2_BF16_staging"
                 and {"partial_bytes", "output_bytes", "ready"} <= set(errors)
                 and not any(errors.values()))
        grouped = {
            "status": "PASS" if exact else "FAIL",
            "source": "actual_FC2_BF16_staging",
            "certifies_FC1_FC2_arithmetic": False,
            "contract": "slot_FP32_owner_BF16_then_owner_FP32_output_BF16",
        }
    return {
        **extra,
        "grouped_implementation": grouped,
        "ordinary_topk": numerical_comparison(actual, flat_reference),
        "staging_transport": transport if transport is not None else {
            "status": "NOT_CHECKED", "reason": "no staging oracle was run",
        },
    }


def enforce_combine_reports(rank_reports: list[dict], *, strict: bool) -> None:
    """Call on every EP rank *after* gathering reports, avoiding partial exit."""
    failures = []
    for rank, report in enumerate(rank_reports):
        for name in ("grouped_implementation", "staging_transport", "grouped_arithmetic"):
            if report.get(name, {}).get("status") == "FAIL":
                failures.append(f"rank {rank} {name}")
        if strict and report["ordinary_topk"]["status"] == "FAIL":
            failures.append(f"rank {rank} ordinary_topk")
    if failures:
        raise AssertionError("Combine validation failed: " + ", ".join(failures))


def reconstruct_owner_from_staging(staged, mapping, metadata, routes, *,
                                   owner, tokens_per_rank, experts_per_rank):
    """Actual staging -> owner partial, safe even when mapping is corrupt.

    Returns a partial and error counts; callers collectively report failures
    before any rank raises. Invalid indices are masked before indexing.
    """
    total, topk = routes.shape
    if mapping.shape != routes.shape or staged.ndim != 2:
        raise ValueError("staging/mapping shape mismatch")
    expected_valid = routes // experts_per_rank == owner
    valid = mapping >= 0
    bounds = valid & (mapping < staged.shape[0]) & (mapping < metadata.numel())
    errors = {
        "membership": int((valid != expected_valid).sum()),
        "row_bounds": int((valid & ~bounds).sum()),
        "invalid_sentinel": int((mapping < -1).sum()),
    }
    keys, slots = bounds.nonzero(as_tuple=True)
    ids = mapping[keys, slots].long()
    errors["duplicate_rows"] = ids.numel() - ids.unique().numel()
    packed = metadata.flatten()[ids]
    hi = packed >> 32
    decoded_key = ((hi >> 16) & 0xffff) * tokens_per_rank + (packed & 0xffffffff)
    errors["metadata"] = int(((decoded_key != keys) | ((hi & 0xffff) != slots)).sum())
    acc = torch.zeros((total, staged.shape[1]), dtype=torch.float32, device=staged.device)
    for slot in range(topk):
        selected = bounds[:, slot] & expected_valid[:, slot]
        rows = selected.nonzero(as_tuple=True)[0]
        acc[rows] = acc[rows] + staged[mapping[rows, slot].long()].float()
    return acc.to(torch.bfloat16), errors


def check_staging_delivery(expected_partials, actual_partials, output, ready, *,
                           active, epoch: int) -> dict:
    """Source-side checks; stale payload of inactive owners is intentionally ignored."""
    if actual_partials.shape != expected_partials.shape or ready.shape != active.shape:
        raise ValueError("partial/ready shape mismatch")
    mask = active[:, :, None]
    expected = expected_partials.masked_fill(~mask, 0)
    actual = actual_partials.masked_fill(~mask, 0)
    return {
        "partial_bytes": byte_difference(actual, expected),
        "output_bytes": byte_difference(output, flat_topk_bf16_reference(expected)),
        "ready": int(((ready == epoch) != active).sum()) + int(epoch <= 0),
        "nonfinite": int((~torch.isfinite(expected) | ~torch.isfinite(actual)).sum())
                     + int((~torch.isfinite(output)).sum()),
    }
