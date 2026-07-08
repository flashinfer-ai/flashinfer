"""NCCL-EP backend (nccl-ep-v0.1.0).

As of ``nccl-ep-v0.1.0`` the backend is driven entirely by the **nccl4py**
Python package's ``nccl.ep`` API — there is no longer an in-tree
``libnccl_ep.so`` to dlopen or a flat ``nccl_ep`` ctypes module to import.
The ``nccl`` package (the released ``nccl4py`` wheel, a base dependency of
flashinfer-python) self-loads its native library; we just import ``nccl.ep``
lazily in :mod:`.fleet` / :mod:`.handle`.

Availability is probed via :func:`flashinfer.moe_ep._probe_nccl_ep`, which checks
that ``nccl.ep`` is importable.
"""

from __future__ import annotations
