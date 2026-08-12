"""Rubin training kernel components.

VENDOR NOTE (flashinfer local diff, see ../../../../VENDOR.md): upstream
re-exports the ``.traditional`` wgrad kernels here.  Only the ``mega.fwd_glu``
(fprop) closure is vendored in this drop, so that import is removed; restore
it when the wgrad tree is migrated.
"""
