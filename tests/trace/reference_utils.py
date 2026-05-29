"""Shared helpers for trace reference correctness tests."""

from __future__ import annotations

import pytest
import torch

from flashinfer.utils import get_compute_capability


def _cc() -> tuple[int, int]:
    return get_compute_capability(torch.device("cuda"))


def _is_sm100() -> bool:
    major, _ = _cc()
    return major >= 10


def _skip_if_not_sm100():
    if not _is_sm100():
        pytest.skip("kernel requires SM100+ (Blackwell)")


def _skip_if_not_sm100_or_103():
    """Gate for kernels that run only on Blackwell proper (SM100/SM103)."""
    major, minor = _cc()
    if (major, minor) not in ((10, 0), (10, 3)):
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")


def _check(template, reference_outputs, actual_outputs, **thresholds) -> None:
    if (
        "rtol" in thresholds or "atol" in thresholds
    ) and "min_cos_sim" not in thresholds:
        thresholds["min_cos_sim"] = None
    assert template.check(reference_outputs, actual_outputs, **thresholds), (
        f"{template.name_prefix or template.op_type}.check failed"
    )


_ROPE_TOL = dict(atol=1e-2, rtol=1e-2)
_ROPE_KWARGS = dict(nnz=16, batch_size=2, num_q_heads=4, num_k_heads=2, head_dim=64)


def _init_filtered(template, **kwargs):
    """Call ``template.init(...)`` passing only kwargs the function accepts."""
    import inspect

    sig = inspect.signature(template.init)
    accepted = set(sig.parameters)
    return template.init(**{k: v for k, v in kwargs.items() if k in accepted})


def _assert_finite(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if t is None:
            continue
        assert torch.isfinite(t.float()).all(), "init/kernel produced NaN or Inf"
