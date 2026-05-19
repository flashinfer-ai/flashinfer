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


def _close(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> None:
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)


def _close_fp8(a: torch.Tensor, b: torch.Tensor, *, cos_sim_min: float = 0.99) -> None:
    """Cosine-similarity check used by APIs whose unit tests use it."""
    import torch.nn.functional as F

    cos = F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0)
    assert cos.item() > cos_sim_min, f"cos_sim={cos.item():.4f} < {cos_sim_min}"


def _close_pass_ratio(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    pass_ratio: float = 0.95,
) -> None:
    """Require at least ``pass_ratio`` of elements to pass absolute or relative tolerance."""
    a_f = a.float()
    b_f = b.float()
    diff_abs = (a_f - b_f).abs()
    diff_rel = diff_abs / (b_f.abs() + 1e-8)
    ok = (diff_abs <= atol) | (diff_rel <= rtol)
    frac = ok.float().mean().item()
    assert frac >= pass_ratio, (
        f"pass_ratio={frac:.4f} < {pass_ratio} (atol={atol}, rtol={rtol})"
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
