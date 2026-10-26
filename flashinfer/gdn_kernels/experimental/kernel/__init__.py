# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Kernel implementations behind the experimental fused GDN decode step.

One module per registry ``impl`` (``gdn_fused_decode_<impl>.py``), plus the
in-package JIT source they compile.  These modules may import heavy or
optional dependencies (the CuTe DSL, the JIT toolchain) at module import
time: each is imported only when a registry row names it, by
:func:`~flashinfer.gdn_kernels.experimental.gdn_fused_decode_specialized._load_impl`,
which tolerates ``ImportError``/``RuntimeError``.

This package init is therefore deliberately empty of imports -- importing
``flashinfer`` must not import a kernel.  The impl-module interface is
documented in ``../README.md``; shared host-side helpers live in
:mod:`._stream_order`.
"""
