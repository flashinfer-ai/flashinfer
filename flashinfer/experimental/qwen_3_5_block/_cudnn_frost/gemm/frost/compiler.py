# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""``cudnn.gemm.frost.compiler`` -- the compiler of the arch family serving this process.

A facade with no code of its own. Importing it resolves
:func:`cudnn.gemm.frost.arch_family.active_family` (``sm100`` / ``sm120``) and
installs that family's ``compiler`` module under THIS name in ``sys.modules``,
so ``cudnn.gemm.frost.compiler`` *is* ``cudnn.gemm.frost.<family>.compiler`` --
one module object, whichever spelling reaches it. That keeps every existing
import (``from cudnn.gemm.frost.compiler import jit_from_cudnn_graph``, the
package-relative ``from .compiler import ...``) and every test that pins an
attribute on it (``monkeypatch.setattr(C, "_current_arch", ...)``) working
against the real module rather than a copy of its namespace. Replacing oneself
in ``sys.modules`` is a case the import system provides for: the import
returns, and binds on the parent package, whatever the name maps to afterwards.

To reach one family regardless of the GPU, import it by its own name
(``cudnn.gemm.frost.sm120.compiler``) or set ``CUDNN_FRONTEND_GEMM_ARCH_FAMILY``
before the first import.
"""

import importlib
import sys
from typing import TYPE_CHECKING

from .arch_family import active_family

if TYPE_CHECKING:  # the static surface: both trees start from this one
    from .sm100.compiler import *  # noqa: F401,F403

sys.modules[__name__] = importlib.import_module(f"{__package__}.{active_family()}.compiler")
