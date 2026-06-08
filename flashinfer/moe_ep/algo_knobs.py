"""Typed AlgoKnob hierarchy.

Knobs are frozen dataclasses keyed by their own class — _index_knobs() lets
backends look up the value with ``knobs.get(KnobClass)`` rather than scanning
a list. Fleet-level knobs (set once at Fleet construction) and Handle-level
knobs (set per dispatch/combine) live in the same namespace; each backend's
constructor inspects the relevant set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, FrozenSet, Sequence

if TYPE_CHECKING:
    import torch

from .config import QuantType


class AlgoKnob:
    """Base marker for both Fleet and Handle knobs."""


# ---------------------------------------------------------------- Fleet-level


@dataclass(frozen=True)
class FleetAlgoKnobQuantization(AlgoKnob):
    """Enable per-Fleet quantization features."""

    quants: FrozenSet[QuantType] = field(default_factory=frozenset)


@dataclass(frozen=True)
class FleetAlgoKnobNumChannelsPerRank(AlgoKnob):
    n: int


@dataclass(frozen=True)
class FleetAlgoKnobNumQpsPerRank(AlgoKnob):
    n: int


@dataclass(frozen=True)
class FleetAlgoKnobRdmaBufferSize(AlgoKnob):
    bytes_: int


@dataclass(frozen=True)
class FleetAlgoKnobTopologyCapacity(AlgoKnob):
    """Reserve transport state for up to this many ranks (grow/shrink later)."""

    n: int


# --------------------------------------------------------------- Handle-level


@dataclass(frozen=True)
class HandleAlgoKnobUserStream(AlgoKnob):
    """Run the dispatch/combine on a non-default CUDA stream.

    The stream is passed as an int (cudaStream_t) since dataclasses with
    torch.cuda.Stream fields aren't hashable.
    """

    stream: int


@dataclass(frozen=True)
class HandleAlgoKnobSplitOperation(AlgoKnob):
    """Marker: enable send_only=1 + ncclEpComplete staging."""


@dataclass(frozen=True)
class HandleAlgoKnobTopKWeights(AlgoKnob):
    """Carries the topk_weights tensor for combine to reweight by."""

    weights: "torch.Tensor"


@dataclass(frozen=True)
class HandleAlgoKnobNumReceivedTokens(AlgoKnob):
    """Where to write the recv-count from ncclEpHandleGetNumRecvTokens."""

    target: "torch.Tensor"


def _index_knobs(knobs: Sequence[AlgoKnob]) -> "dict[type, AlgoKnob]":
    """Build a dict keyed by knob class so backends can ``.get(KnobClass)``.

    If the same knob class appears twice in the sequence, the later entry
    wins (mirrors dict() over key-value pairs). Knobs with class identity
    only (e.g. SplitOperation) act as flags: their presence in the dict
    signals "set"; the stored instance is the marker itself.
    """
    out: dict[type, AlgoKnob] = {}
    for k in knobs:
        if not isinstance(k, AlgoKnob):
            raise TypeError(f"expected AlgoKnob, got {type(k).__name__}")
        out[type(k)] = k
    return out
