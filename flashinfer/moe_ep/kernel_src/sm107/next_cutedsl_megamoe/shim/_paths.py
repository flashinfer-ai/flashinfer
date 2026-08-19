"""Put ``kernel_src/sm107/next_cutedsl_megamoe/src`` on ``sys.path`` so ``sources`` resolves.

This is shim glue, not kernel-team code, so it lives in ``shim/`` (never inside
``src/`` — that directory is a *verbatim* drop from the kernel team).  It adds
the sibling ``src/`` directory to ``sys.path`` so the drop's single top-level
package (``sources``) imports without a separate editable install.

Unlike the SM90/SM100 fork pair (which share ``common``/``src``/
``moe_nvfp4_swapab`` top-level names and are mutually exclusive per process),
the ``next/`` greenfield tree ships only the ``sources`` package with fully
relative imports, so this tree does NOT collide with the sibling trees.  The
guard below still protects against a foreign ``sources`` module (a generic
name) already owning the process.
"""

from __future__ import annotations

import os
import sys

_BOOTSTRAPPED = False

# The drop's only top-level package. Anything else already imported under this
# name (it is a generic one) would shadow the kernel tree.
_SENTINEL_MODULES = ("sources",)


def bootstrap_paths() -> None:
    """Idempotently prepend the vendored ``src/`` directory to ``sys.path``."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # this file: .../next_cutedsl_megamoe/shim/_paths.py
    # -> sibling .../next_cutedsl_megamoe/src
    shim_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(shim_dir), "src")

    for name in _SENTINEL_MODULES:
        mod = sys.modules.get(name)
        mod_file = getattr(mod, "__file__", None) if mod is not None else None
        if mod_file is not None and not os.path.abspath(mod_file).startswith(
            src_dir + os.sep
        ):
            raise RuntimeError(
                f"kernel module {name!r} is already imported from "
                f"{mod_file!r}; it shadows the next_cutedsl_megamoe drop's "
                "top-level package. Use a separate process for whatever owns "
                "that module."
            )

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    _BOOTSTRAPPED = True
