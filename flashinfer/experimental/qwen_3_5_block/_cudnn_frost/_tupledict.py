# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``TupleDict`` -- the dict that unpacks like a tuple, the result type of the frontend-only APIs.

A module of its own, importing neither torch nor the CuTe DSL nor ``cuda-python``: the DSL-free front doors (``cudnn.QSA.selection_paged``,
``cudnn.QSA.selection``) return it, and their documented import path must resolve on every installed DSL (``python/cudnn/AGENTS.md``
Rule 7) -- ``cudnn.api_base``, the class's previous home, imports the DSL at module scope, and the public 4.7.0 wheel's
``cutlass/__init__`` imports ``cutlass.experimental``, which the 4.6.2 wheel the downstream floor admits does not have, so a front door
importing ``api_base`` would die inside the DSL on the way to its typed decline.  ``api_base`` re-exports the class, so every existing
``from cudnn.api_base import TupleDict`` still resolves to this one.
"""

from __future__ import annotations


class TupleDict(dict):
    """A dictionary that supports tuple unpacking.

    This class extends dict to allow unpacking like a tuple while still
    providing dictionary-style key access. The unpacking order is determined
    by the _keys attribute which preserves insertion order.

    Example:
        >>> result = TupleDict(a=1, b=2, c=3)
        >>> x, y, z = result  # Unpacks as (1, 2, 3)
        >>> result['a']  # Returns 1
        >>> result[0]  # Returns 1 (integer indexing)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store keys in order for tuple unpacking
        self._keys = list(self.keys())

    def __eq__(self, other):
        # Equality is the dict's own: ``_keys`` is a derived view of the insertion order, never part of the value
        # (the same behaviour as before this class carried its own ``__eq__``; stated explicitly for CodeQL py/missing-equals).
        return dict.__eq__(self, other)

    __hash__ = None  # a dict is unhashable; an explicit ``__eq__`` keeps it so

    def __iter__(self):
        """Iterate over values in insertion order for tuple unpacking."""
        return (self[k] for k in self._keys)

    def __getitem__(self, key):
        """Support both string keys and integer indices."""
        if isinstance(key, int):
            if key < 0 or key >= len(self._keys):
                raise IndexError(f"index {key} out of range for TupleDict with {len(self._keys)} items")
            return super().__getitem__(self._keys[key])
        return super().__getitem__(key)
