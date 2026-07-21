"""
Composable input pre-hooks for AutoTuner profiling.

An ``InputsPreHook`` runs ONCE per profile bucket -- after the per-input
:class:`TensorInitializer`\\ s have synthesized the full input list, but before
the per-tactic profile loop. Unlike a per-tensor initializer (which sees only
its own tensor), a pre-hook receives the WHOLE list and may establish
cross-tensor relationships or inject joint distributions, returning a (possibly
modified) list.

Like the other autotuner vocabularies, these are frozen (hence hashable)
dataclasses behind an ``InputsPreHook`` Protocol, so a :class:`TuningConfig`
hashes structurally. ``ChainedInputsPreHook`` composes several hooks in order.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


class InputsPreHook(Protocol):
    """A post-synthesis transform over the full profiling input list."""

    def __call__(
        self, inputs: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...


@dataclass(frozen=True)
class CapLengthToIndices:
    """Cross-tensor coupling: set ``inputs[length_idx]`` to a constant fill of
    ``inputs[indices_idx]``'s last-dim size.

    Used e.g. to make a ``topk_length`` tensor agree with the top-k width of its
    paired ``indices`` tensor -- a relationship a per-tensor initializer cannot
    express (it sees only its own tensor). Passes the list through unchanged when
    either index is out of range or ``None``.
    """

    length_idx: int
    indices_idx: int

    def __call__(self, inputs: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
        if (
            self.length_idx < len(inputs)
            and self.indices_idx < len(inputs)
            and inputs[self.length_idx] is not None
            and inputs[self.indices_idx] is not None
        ):
            out = list(inputs)
            out[self.length_idx] = torch.full_like(
                out[self.length_idx], inputs[self.indices_idx].shape[-1]
            )
            return out
        return inputs


@dataclass(frozen=True)
class ChainedInputsPreHook:
    """Apply each hook in ``hooks`` left-to-right, threading the input list."""

    hooks: tuple[InputsPreHook, ...]

    def __call__(self, inputs: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
        for hook in self.hooks:
            inputs = hook(inputs)
        return inputs
