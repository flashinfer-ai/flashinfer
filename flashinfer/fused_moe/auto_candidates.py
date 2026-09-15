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
    # Experimental candidates require the normal gate unless a registration
    # explicitly preserves a branch-local PoC exception.
    requires_opt_in: bool = True


_AUTO_CANDIDATES = {
    "cudnn_frost_bf16": AutoCandidateSpec(
        "flashinfer.experimental.cudnn_frost_selected_kernels.support",
        # Preserve this branch's user-requested unchanged-code BF16 PoC.
        # This exception must not become the default for other registrations.
        requires_opt_in=False,
    ),
}


def additional_candidates(config, device, arch, act, weights, *, tuning, cache):
    """Return eligible runners, preserving layer-owned resources across captures.

    The support-module contract is ``is_eligible(config, act, arch)`` followed by
    ``create_runner(config, device)``. The runner's ``accepts(act, weights)`` must
    validate per-call layouts/semantics before any winner-cache lookup. Disabling
    a registration or its gate excludes its cached runner without destroying
    resources still referenced by captured graphs.
    """
    candidates = []
    for key, spec in _AUTO_CANDIDATES.items():
        if not tuning and key not in cache:
            continue
        if spec.requires_opt_in and not experimental_auto_backends_allowed():
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
