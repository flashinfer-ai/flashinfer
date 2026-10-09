"""Lazy registration of additional, per-call MoE autotune candidates.

Registrations contain no execution imports. Their support modules own shape
and semantic policy; factories prepare independent runners, not overrides of
another backend. Cold calls never construct these optional candidates.
"""

from dataclasses import dataclass
from importlib import import_module

from ..api_logging import experimental_auto_backends_allowed


@dataclass(frozen=True)
class AutoCandidateSpec:
    support_module: str
    experimental: bool = True


_AUTO_CANDIDATES = {
    "cudnn_frost_bf16": AutoCandidateSpec(
        "flashinfer.fused_moe.backends.cudnn_frost.bf16.support",
        experimental=False,
    ),
    "cudnn_frost_mxfp8": AutoCandidateSpec(
        "flashinfer.fused_moe.backends.cudnn_frost.mxfp8.support",
        experimental=False,
    ),
    "cudnn_frost_nvfp4": AutoCandidateSpec(
        "flashinfer.fused_moe.backends.cudnn_frost.nvfp4.support",
        experimental=False,
    ),
    "cudnn_frost_mxfp8_mxfp4": AutoCandidateSpec(
        "flashinfer.fused_moe.backends.cudnn_frost.mxfp8_mxfp4.support",
        experimental=False,
    ),
}


def additional_candidates(
    config, device, arch, act, weights, *, tuning, cache, exclude=()
):
    """Return eligible runners, preserving layer-owned resources across captures.

    The support-module contract is ``is_eligible(config, act, arch)`` followed by
    ``create_runner(config, device)``. The runner's ``accepts(act, weights)`` must
    validate per-call layouts/semantics before any winner-cache lookup. Disabling
    a registration or its gate excludes its cached runner without destroying
    resources still referenced by captured graphs.
    """
    candidates = []
    for key, spec in _AUTO_CANDIDATES.items():
        if key in exclude or (
            spec.experimental and not experimental_auto_backends_allowed()
        ):
            continue
        if not tuning and key not in cache:
            continue
        support = import_module(spec.support_module)
        if not support.is_eligible(config, act, arch):
            continue
        runner = cache.get(key)
        if runner is None:
            runner = support.create_runner(config, device)
            if runner is None:
                continue
            if runner.backend_key != key:
                raise ValueError(
                    f"MoE automatic candidate {key!r} has a mismatched key"
                )
            cache[key] = runner
        if runner.accepts(act, weights):
            candidates.append(runner)
    return candidates


def is_experimental_candidate(key):
    """Whether an additional candidate still requires experimental opt-in."""
    spec = _AUTO_CANDIDATES.get(key)
    return spec is not None and spec.experimental
