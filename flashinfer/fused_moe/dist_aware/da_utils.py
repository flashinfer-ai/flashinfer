"""Distribution grammar and routing-value helpers owned by DA MoE."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np
import torch


DADistributionSpec = Tuple[str, str, Any]

DEFAULT_DA_DISTRIBUTIONS: tuple[DADistributionSpec, ...] = (
    ("ddist:1.1", "ddist_factor", 1.1),
    ("ddist:1.3", "ddist_factor", 1.3),
    ("ddist:1.5", "ddist_factor", 1.5),
    ("ddist:1.7", "ddist_factor", 1.7),
    ("ddist:2", "ddist_factor", 2.0),
    ("ddist:2.5", "ddist_factor", 2.5),
    ("ddist:4", "ddist_factor", 4.0),
)


def get_da_distribution_specs(
    spec: str | None = None,
) -> tuple[DADistributionSpec, ...]:
    """Parse the DA distribution grammar without consulting process state."""

    if not spec:
        return DEFAULT_DA_DISTRIBUTIONS
    out: list[DADistributionSpec] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "uniform":
            out.append(("uniform", "uniform", 1.0))
        elif item == "single":
            out.append(("single", "single", 0.0))
        elif item.startswith(("exp:", "exp_")):
            factor = float(item.split(":" if ":" in item else "_", 1)[1])
            if factor <= 0:
                raise ValueError(f"exp factor must be positive: {item!r}")
            out.append((f"exp:{factor:g}", "exp_factor", factor))
        elif item.startswith(("ddist:", "ddist_")):
            factor = float(item.split(":" if ":" in item else "_", 1)[1])
            if factor <= 0:
                raise ValueError(f"ddist factor must be positive: {item!r}")
            out.append((f"ddist:{factor:g}", "ddist_factor", factor))
        elif item.startswith(("sparse:", "sparse_")):
            parts = item.split(":" if ":" in item else "_")
            if len(parts) != 3:
                raise ValueError(
                    f"sparse distribution must be sparse:<active_factor>:<eff_factor>: {item!r}"
                )
            active_factor, eff_factor = map(float, parts[1:])
            if active_factor <= 0 or eff_factor <= 0:
                raise ValueError(f"sparse factors must be positive: {item!r}")
            out.append(
                (
                    f"sparse:{active_factor:g}:{eff_factor:g}",
                    "sparse_factor",
                    (active_factor, eff_factor),
                )
            )
        else:
            raise ValueError(f"Unknown DA distribution spec: {item!r}")
    return tuple(out)


def pack_expert_assignments(
    expert_ids: torch.Tensor,
    weights: torch.Tensor | None = None,
    top_k: int = 1,
) -> torch.Tensor:
    """Pack expert ids and bfloat16 weights into TRT-LLM routed format."""

    if weights is None:
        weights = torch.full(
            expert_ids.shape,
            1.0 / max(top_k, 1),
            dtype=torch.bfloat16,
            device=expert_ids.device,
        )
    elif weights.dtype != torch.bfloat16:
        weights = weights.to(torch.bfloat16)
    ids_shifted = expert_ids.to(torch.int32) << 16
    weights_bits = weights.view(torch.int16).to(torch.int32) & 0xFFFF
    return ids_shifted | weights_bits


_EXP_FLOOR_FRAC = 0.1
_EXP_SEARCH_ITERS = 50
_EXP_LAMBDA_RANGE = (0.0, 20.0)
_DIRICHLET_SEARCH_ITERS = 80
_DIRICHLET_ALPHA_RANGE = (1e-6, 1e6)


def _clamp_effective_experts(target_eff: float, n_experts: int) -> float:
    return min(max(float(target_eff), 1.0), float(n_experts))


def _solve_monotonic_parameter(
    target: float,
    lo: float,
    hi: float,
    value_fn: Callable[[float], float],
    *,
    increasing: bool,
    iters: int,
) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        value = value_fn(mid)
        if (value < target and increasing) or (value > target and not increasing):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _apply_uniform_floor(probs: np.ndarray) -> np.ndarray:
    probs = (1.0 - _EXP_FLOOR_FRAC) * probs + _EXP_FLOOR_FRAC / probs.size
    probs /= probs.sum()
    return probs


def _exp_floor_probs(lam: float, n_experts: int) -> np.ndarray:
    probs = np.exp(-lam * np.arange(n_experts, dtype=np.float64))
    probs /= probs.sum()
    return _apply_uniform_floor(probs)


def _exp_floor_probs_for_target_eff(target_eff: float, n_experts: int) -> np.ndarray:
    target_eff = _clamp_effective_experts(target_eff, n_experts)
    lam = _solve_monotonic_parameter(
        target_eff,
        *_EXP_LAMBDA_RANGE,
        lambda value: float(1.0 / (_exp_floor_probs(value, n_experts) ** 2).sum()),
        increasing=False,
        iters=_EXP_SEARCH_ITERS,
    )
    return _exp_floor_probs(lam, n_experts)


def _symmetric_dirichlet_probs_for_target_eff(
    target_eff: float, n_experts: int
) -> np.ndarray:
    target_eff = _clamp_effective_experts(target_eff, n_experts)

    def expected(alpha: float) -> float:
        n = float(n_experts)
        second = (alpha + 1.0) / (n * alpha + 1.0)
        f = _EXP_FLOOR_FRAC
        return float(1.0 / ((1.0 - f) ** 2 * second + (2.0 * f - f**2) / n))

    alpha = _solve_monotonic_parameter(
        target_eff,
        *_DIRICHLET_ALPHA_RANGE,
        expected,
        increasing=True,
        iters=_DIRICHLET_SEARCH_ITERS,
    )
    rng = np.random.default_rng(42)
    probs = rng.dirichlet(np.full(n_experts, alpha, dtype=np.float64))
    probs = np.clip(probs, np.finfo(np.float64).tiny, None)
    probs /= probs.sum()
    return np.sort(_apply_uniform_floor(probs))[::-1]


def da_distribution_target_effective_experts(
    distribution: DADistributionSpec, num_local_experts: int
) -> float:
    _, kind, param = distribution
    if kind == "uniform":
        return float(num_local_experts)
    if kind == "single":
        return 1.0
    if kind in ("exp_factor", "ddist_factor"):
        return max(1.0, float(num_local_experts) / float(param))
    if kind in ("sparse_eff", "sparse_factor"):
        return _sparse_active_eff(kind, param, num_local_experts)[1]
    raise ValueError(f"Unknown DA distribution kind: {kind!r}")


def _sparse_active_eff(
    kind: str, param: Any, num_local_experts: int
) -> tuple[int, float]:
    if kind == "sparse_factor":
        active_factor, eff_factor = param
        active = min(
            max(1, int(np.floor(num_local_experts / active_factor + 0.5))),
            num_local_experts,
        )
        return active, min(max(1.0, num_local_experts / eff_factor), float(active))
    if kind == "sparse_eff":
        active_raw, eff_raw = param
        active = min(max(1, int(active_raw)), num_local_experts)
        return active, min(max(1.0, float(eff_raw)), float(active))
    raise ValueError(f"Unknown sparse distribution kind: {kind!r}")


def _sparse_probs(kind: str, param: Any, num_local_experts: int) -> np.ndarray:
    active, target_eff = _sparse_active_eff(kind, param, num_local_experts)
    probs = np.zeros(num_local_experts, dtype=np.float64)
    probs[:active] = _symmetric_dirichlet_probs_for_target_eff(target_eff, active)
    return probs


def _sample_expert_assignments_from_probs(
    probs: np.ndarray,
    original_tensor: torch.Tensor,
    top_k: int,
    local_expert_offset: int = 0,
) -> torch.Tensor:
    num_tokens = int(original_tensor.shape[0])
    probs_t = torch.from_numpy(probs).float().to(device=original_tensor.device)
    support = int(np.count_nonzero(probs > 0.0))
    if top_k <= support:
        indices = torch.multinomial(
            probs_t.expand(num_tokens, -1), top_k, replacement=False
        )
    else:
        indices = torch.multinomial(
            probs_t, num_tokens * top_k, replacement=True
        ).reshape(num_tokens, top_k)
    return indices.to(dtype=original_tensor.dtype) + int(local_expert_offset)


def _shuffle_probs(probs: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.zeros_like(probs)
    shuffled[rng.permutation(probs.size)] = probs
    return shuffled


def generate_da_distribution_assignments(
    distribution: DADistributionSpec,
    original_tensor: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    top_k: int,
    local_expert_offset: int = 0,
) -> torch.Tensor:
    """Generate synthetic local expert assignments for DA profiling."""

    del num_experts
    _, kind, param = distribution
    if kind == "single":
        return torch.full(
            (original_tensor.shape[0], top_k),
            int(local_expert_offset),
            dtype=original_tensor.dtype,
            device=original_tensor.device,
        )
    if kind == "uniform":
        probs = np.full(num_local_experts, 1.0 / num_local_experts)
    elif kind == "exp_factor":
        probs = _shuffle_probs(
            _exp_floor_probs_for_target_eff(
                da_distribution_target_effective_experts(
                    distribution, num_local_experts
                ),
                num_local_experts,
            )
        )
    elif kind == "ddist_factor":
        probs = _shuffle_probs(
            _symmetric_dirichlet_probs_for_target_eff(
                da_distribution_target_effective_experts(
                    distribution, num_local_experts
                ),
                num_local_experts,
            )
        )
    elif kind in ("sparse_eff", "sparse_factor"):
        probs = _sparse_probs(kind, param, num_local_experts)
    else:
        raise ValueError(f"Unknown DA distribution kind: {kind!r}")
    return _sample_expert_assignments_from_probs(
        probs, original_tensor, top_k, local_expert_offset
    )
