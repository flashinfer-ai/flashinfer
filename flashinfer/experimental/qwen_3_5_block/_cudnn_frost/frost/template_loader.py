# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared kernel-template loader for FROST engines.

Why this file exists
--------------------
Kernel templates specialize at IMPORT time, but Python's import system only
lets a module exist once. Both halves of that sentence are load-bearing:

- A template body runs ``CFG, _TMA = make_cfg_*(PARAMS)`` at module scope and
  then defines its ``@cute.kernel`` / ``@cute.jit`` functions with ``CFG.*``
  as constants; the DSL tracer folds ``cutlass.const_expr(...)`` branches
  away, so a causal specialization contains no dense code and vice versa.
  That folding only works if the values are module-level constants when the
  functions are *defined* — passing them as runtime kernel arguments would
  turn compile-time specialization into runtime branching (and dtype cannot
  be a runtime branch at all: the MMA instructions differ).
- ``import sm100/prefill_d512_f16`` executes the body once and caches it in
  ``sys.modules``; one module name = one parameter set. But one process
  legitimately needs several specializations of the same file alive at once
  (a causal-fp16 graph and a dense-bf16 graph in the same session).

How it works
------------
``importlib.util.spec_from_file_location`` with a UNIQUE generated module
name executes the same file again as a fresh module, sidestepping the
``sys.modules`` dedup. The params object is injected as the module global
``FROST_TEMPLATE_PARAMS`` *before* the body runs, so the body specializes
itself. Each distinct ``(path, params)`` pair gets its own module and is
cached forever (identical pairs reuse the already-JIT-compiled one); a lock
makes it thread-safe.

Roads not taken
---------------
- Env vars + ``sys.modules.pop`` + re-import (the original approach): only
  one specialization alive at a time, re-JIT thrash on every switch, and
  mutable global state (``os.environ``/``sys.path``/``sys.modules``) that
  races across threads.
- Factory functions (``def make_kernel(params): ...``): would require
  wrapping every 1000+-line template body — module-scope constants, class
  definitions, decorated functions — in a closure, for every current and
  future kernel. This 40-line loader gets the same effect with zero changes
  to how kernel authors naturally write DSL code, which is also why it lives
  in ``frost/`` as shared framework.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import threading
from typing import Any, Hashable

_MODULES: dict = {}
_LOCK = threading.Lock()

PARAMS_GLOBAL = "FROST_TEMPLATE_PARAMS"
DIGEST_GLOBAL = "FROST_SOURCE_DIGEST"


def load_template(path: str, params: Hashable, tag: str = "template") -> Any:
    """Execute the kernel template at ``path`` specialized for ``params``.

    ``params`` must be hashable (a frozen dataclass); it is the cache key
    alongside ``path`` and is injected as the module global ``FROST_TEMPLATE_PARAMS``.
    ``tag`` only affects the generated module name (debuggability).
    """
    key = (path, params)
    with _LOCK:
        mod = _MODULES.get(key)
        if mod is not None:
            return mod
        name = f"cudnn.frost._templates.{tag}_{len(_MODULES)}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        setattr(mod, PARAMS_GLOBAL, params)
        # What the module's kernels are compiled FROM: this file, specialized by
        # these params. compiled_cache.template_key joins it with a compile()
        # call's arguments to name a persistent object.
        setattr(mod, DIGEST_GLOBAL, hashlib.sha256(pathlib.Path(path).read_bytes() + repr(params).encode("utf-8")).hexdigest()[:16])
        spec.loader.exec_module(mod)
        _MODULES[key] = mod
        return mod
