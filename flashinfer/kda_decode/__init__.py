"""Compatibility imports for the former recurrent-KDA module name.

The canonical recurrent KDA API lives in :mod:`flashinfer.kda`.  This module
keeps existing direct imports working while new code uses the phase-neutral
module name.
"""

from ..kda import (
    RecurrentKDAPrefillWorkspace as RecurrentKDAPrefillWorkspace,
)
from ..kda import _RECURRENT_KDA_AVAILABLE as _RECURRENT_KDA_AVAILABLE
from ..kda import recurrent_kda as recurrent_kda

__all__ = ["RecurrentKDAPrefillWorkspace", "recurrent_kda"]
