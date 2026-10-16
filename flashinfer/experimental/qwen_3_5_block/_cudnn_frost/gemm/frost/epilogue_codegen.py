# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""``cudnn.gemm.frost.epilogue_codegen`` -- the epilogue codegen of the active arch family.

Facade, same mechanism as :mod:`cudnn.gemm.frost.compiler`: this name becomes
``cudnn.gemm.frost.<family>.epilogue_codegen`` for the family
:func:`cudnn.gemm.frost.arch_family.active_family` picks -- the very module the
active compiler imports as its sibling, so ``compiler.generate is
epilogue_codegen.generate`` holds and a test that pins one pins the other.
"""

import importlib
import sys
from typing import TYPE_CHECKING

from .arch_family import active_family

if TYPE_CHECKING:  # the static surface: both trees start from this one
    from .sm100.epilogue_codegen import *  # noqa: F401,F403

sys.modules[__name__] = importlib.import_module(f"{__package__}.{active_family()}.epilogue_codegen")
