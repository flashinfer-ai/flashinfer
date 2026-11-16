"""Read-only selector services used by the paired robustness policies.

Keep this interface independent of heuristic.py to avoid circular imports.
Policies propose changes; the selector owns feature derivation and validation.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PolicyContext:
    derive_features: Callable
    legality_errors: Callable
    semantic_errors: Callable
    correctness_qualification_errors: Callable
