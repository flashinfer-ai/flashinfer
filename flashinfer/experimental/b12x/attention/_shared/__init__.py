"""Lowering shared by attention ops: CuTe softmax/copy/pipeline helpers
(``cute``), the FA-style building blocks and contiguous kernels
(``contiguous``), the unified MLA kernel core (``mla``), and the shared
serving arena (``workspace``). Imports only _lib and itself (plus lazy
shared-arena reach-throughs to gemm/norm workspaces documented in
``workspace.py``).
"""
